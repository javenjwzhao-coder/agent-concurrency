# Dashboard and HTTP Feed

Components:

- `src/sidecar_http.py`
- `dashboard/index.html`
- `dashboard/dashboard.js`
- `dashboard/dashboard.css`
- `dashboard/SCHEMA.md`

The dashboard visualizes sidecar tick records. It can consume live records over
HTTP/SSE, replay a finished `sidecar.log`, or export a standalone HTML snapshot
with data and assets embedded.

## Responsibilities

### HTTP Feed

- Serve dashboard static assets.
- Keep a ring buffer of recent sidecar tick records.
- Expose the current buffer as JSON at `/state`.
- Stream live tick records over server-sent events at `/stream`.
- Replay a finished JSONL log into the same feed.

### Browser Dashboard

- Render one timeline row per agent.
- Render phase segments from agent `state` and `state_since`.
- Plot vLLM KV usage percent and offload threshold.
- Overlay admission/offload/readmit/saturation event markers.
- Show pressure, queue, preemption, and tick badges.
- Load `sidecar.log` directly from disk for browser-only replay.
- Export standalone HTML snapshots.

## HTTP Endpoints

`sidecar_http.py` uses only the Python standard library.

| Endpoint | Behavior |
| --- | --- |
| `GET /` or `/index.html` | Serves `dashboard/index.html`. |
| `GET /static/<path>` | Serves files under the dashboard directory. |
| `GET /state` | Returns `{"latest_tick": int|null, "ticks": [...]}` from the in-memory ring buffer. |
| `GET /stream?since=<tick>` | SSE stream. Replays buffered ticks newer than `since`, then streams live ticks. |
| `GET /healthz` | Returns `ok`. |

Static serving resolves paths against the dashboard directory and rejects path
traversal outside that root.

## `HTTPFeed`

`HTTPFeed` is the thread-safe bridge between the sidecar loop and HTTP clients.

Important methods:

- `publish(record)`: append a tick to the ring buffer and fan it out to current
  subscribers.
- `subscribe(since_tick)`: register an SSE queue and return buffered replay
  records.
- `unsubscribe(queue)`: remove a disconnected subscriber.
- `snapshot()`: return `(latest_tick, ticks)` for `/state`.

Subscriber queues are bounded. If a browser is too slow, the feed drops that
subscriber's oldest queued item rather than blocking the sidecar tick loop.

## Live Mode

When the runner starts the embedded sidecar with `--sidecar-http-port`, it also
creates an `HTTPFeed` and calls `start_server()`.

Browser boot flow:

1. Fetch `/state`.
2. Apply all buffered ticks.
3. Open `EventSource("/stream?since=<latest_tick>")`.
4. Apply each incoming `data:` frame as another tick.
5. Reconnects use the last applied tick as `since`.

The dashboard uses the record timestamp as timeline "now" rather than browser
time. This keeps replay and live views consistent.

## Server Replay Mode

Run:

```bash
python -m sidecar_http --replay path/to/sidecar.log --speed 1.0
```

`replay_into_feed()` reads JSONL records and publishes them into the same
`HTTPFeed`. `--speed 0` replays as fast as possible; values above `1.0` replay
faster than real time.

After replay finishes, the server remains up and `/state` still serves the
ring buffer.

## Browser Replay Mode

The `Open sidecar.log...` control lets the user choose a JSONL file locally.
`dashboard.js` parses each line, skips malformed records, resets charts, and
applies the ticks through the same render path used for live data.

While replay mode is active, the SSE connection is closed. `Back to live`
restarts the live bootstrap flow.

## Standalone Snapshot Mode

`Save standalone HTML` builds a single HTML file containing:

- current shell markup
- embedded `vis-timeline` CSS and JS
- embedded dashboard CSS and JS
- embedded tick records in a JSON script tag

Opening the exported file calls `readEmbeddedSnapshot()` and renders the saved
data without `/state`, `/stream`, or an external `sidecar.log`.
When the browser exposes `showSaveFilePicker()`, the dashboard prompts for a
destination path before writing the snapshot. Browsers without that API keep the
download-link fallback.

The snapshot payload includes a version so future format changes can be handled
explicitly.

## Rendering Model

`dashboard.js` maintains an in-memory state object with:

- tick history
- latest tick
- agent groups
- event points
- pause/live/replay state
- replay bounds
- vis-timeline datasets

Each tick is applied with `applyTick(record)`:

1. Update tick history and latest tick.
2. Apply agent phase rows.
3. Apply admission/offload/readmit/saturation event markers.
4. Apply KV chart points.
5. Update badges.
6. Auto-scroll unless paused or replaying.

## Agent Timeline

Agent rows are stable by first-seen order. The label shows elapsed runtime while
the agent is active and fixed end-to-end runtime after `state = "done"`.

Phase segments are derived from:

- `agents[id].state`
- `agents[id].state_since`
- current tick `ts`
- optional `kv_gb` for tooltips

Supported phase colors are tested in `tests/test_dashboard_events.py`.

## Events

The dashboard overlays vertical markers on both timeline and KV chart.
Use the header's `Hide event lines` / `Show event lines` button to temporarily
remove those vertical markers and their proximity tooltips while leaving event
counts, phase rows, and KV-cache lines unchanged.

| Event | Source |
| --- | --- |
| OFFLOAD | `admission.offloads[*]` where `offloaded == true` |
| ADMIT | `admission.admissions[*]` where `admitted == true` and `previously_offloaded != true` |
| READMIT | `admission.admissions[*]` where `previously_offloaded == true` |
| SAT | `admission.reasons` includes `saturation_guard` |

Event timestamps prefer event-specific fields such as `admitted_at`, falling
back to the tick timestamp.

Tooltip text combines a base controller snapshot with event-specific fields:

- tick timestamp and tick number
- free KV (`C`, `C_percent`)
- threshold values
- pressure boolean
- headroom (`w`, `w_threshold`, `s_t`, `s_prev`)
- event-specific reason/status values

## KV Chart

The KV line chart uses:

- `vllm.kv_cache_used_pct`
- `admission.threshold_percent`

The offload threshold line is displayed as used-percent threshold:

```text
100 - free_threshold_percent
```

The dashboard can derive total/free percentages from GB fields when direct
percent fields are missing.

## Schema Contract

The canonical feed schema is [../dashboard/SCHEMA.md](../dashboard/SCHEMA.md).
Any change to sidecar tick records should update that file and dashboard tests
in the same change.

## Tests

`tests/test_dashboard_events.py` checks:

- only successful offloads render OFFLOAD markers
- controller pressure badge text
- preemption badge and event line overlay
- event-specific admission timestamps
- event color consistency
- agent label panel width behavior
- elapsed vs fixed end-to-end labels
- standalone snapshot export support

## Change Guidelines

- Keep live, server replay, browser replay, and snapshot modes using the same
  tick application path.
- Avoid transforming records in `sidecar_http.py`; the schema should stay
  exactly what `sidecar.py` writes.
- Update `dashboard/SCHEMA.md` for any new or renamed fields.
- Add dashboard tests for new event types or badge logic.
