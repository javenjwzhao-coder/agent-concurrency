# Dashboard Feed JSON Schema

The `/state` and `/stream` endpoints exposed by `src/sidecar_http.py` (and the
records inside `sidecar.log`) use the same JSON shape — one *tick record* per
poll interval. The dashboard reads this shape directly with no transformation.

This file is the canonical contract. If a field changes in
[src/sidecar.py](../src/sidecar.py), update both this file and the dashboard
together.

## Endpoints

| Endpoint                  | Returns                                                      |
|---------------------------|--------------------------------------------------------------|
| `GET /state`              | `{ "latest_tick": int\|null, "ticks": [<tick record>, ...] }` — last N records from the in-memory ring buffer (default N = 2000). |
| `GET /stream?since=<n>`   | `text/event-stream`. On connect, replays buffered records with `tick > since`, then streams new ticks live as they are published. One `data:` frame per tick. |
| `GET /healthz`            | `"ok"` (200).                                                |
| `GET /` and `GET /static/*` | Static assets for the dashboard.                          |

`since` is optional. The dashboard sends it on reconnect to avoid re-applying
ticks it already rendered.

## Tick record

```jsonc
{
  "ts":   "2026-04-29T14:30:00.123456+00:00", // ISO-8601 UTC. The "now" the dashboard uses on the right edge.
  "tick": 142,                                 // Monotonic int per sidecar process.

  "vllm": {                                    // Cluster-wide KV telemetry from vLLM /metrics.
    "kv_total_gb": 12.4,                       // float | null
    "kv_used_gb":  6.81,                       // float | null
    "kv_free_gb":  5.59,                       // float | null
    "kv_cache_used_pct": 54.9,                 // float | null  → drives the KV % line chart
    "num_gpu_blocks_total": 1000,              // int   | null
    "num_gpu_blocks_used":  450,               // int   | null
    "num_gpu_blocks_free":  550                // int   | null
    // "error": "<string>" if /metrics fetch failed this tick
  },

  "agents": {                                  // Per-agent live state (key = agent_id).
    "agent_task_3_w0_2": {
      "task_id": "task_3",                     // str
      "state":   "reasoning",                  // "waiting"|"reasoning"|"tool_call"|"evicted_waiting"|"done"
      "state_since": "2026-04-29T14:29:55.000+00:00",  // ISO-8601 — phase entry timestamp
      "kv_gb":   0.34,                         // float | null
      "kv_blocks": 1024,                       // int   | null
      "admission_state": "admitted",           // "admitted"|"evicted_pending_tool"|"evicted_ready"
      "kv_policy": "pinned_short",             // optional: "pinned_short"|"pinned_long"|"readmitted"|...
      "kv_pinned": true,                       // optional bool, true while sidecar pins tool-call KV
      "tool_name": "execute_bash"              // present only during "tool_call"
      // …additional tool-call fields (tool_args_json, tool_payload_bytes, …) may also appear
    }
  },

  "admission": {                               // Controller decisions for this tick.
    "enabled": true,                           // bool
    "C":       5.59,                           // float — free KV in GB this tick
    "s_t":     0.31,                           // float — current avg KV usage of active agents
    "s_prev":  0.29,                           // float|null — previous tick's avg
    "w":       18.03,                          // float|null — headroom = C / min(s_t, s_prev)
    "threshold_gb": 0.1,                       // float — pressure threshold
    "first_saturation_seen": false,            // bool — initial launch ramp is disabled after first SAT
    "initial_admit_interval_s": 2.0,           // float — fresh-admit interval before first SAT
    "next_initial_admit_in_s": null,           // float|null — seconds until next ramped fresh admit
    "active_agent_samples": 4,                 // int

    "queue": {                                 // pending admissions
      "fresh": 2,
      "evicted_ready": 0,
      "evicted_pending_tool": 1,
      "pinned_short": 3,
      "pinned_long": 4
    },

    "heap_candidates": [                       // ranked idle-agent eviction heap (snapshot)
      { "agent_id": "...", "kv_gb": 0.5, "predicted_remaining_s": 8.0, "e_s": 4.0 }
    ],
    "skipped_candidates": [                    // tool-call agents excluded from offload
      { "agent_id": "...", "reason": "pinned_short" }
    ],
    "kv_policy_events": [                      // immediate tool-start policy decisions
      { "agent_id": "...", "policy": "pinned_long",
        "predicted_remaining_s": 8.0, "threshold_s": 2.0 }
    ],

    "evictions": [                             // events fired this tick
      { "agent_id": "...", "evicted": true,
        "kv_gb": 0.5, "freed_gb": 0.48,
        "predicted_remaining_s": 8.0, "e_s": 4.0,
        "status_code": 200, "reason": "ok" }
    ],

    "admissions": [                            // events fired this tick
      { "agent_id": "...", "admitted": true, "previously_evicted": false }
    ],

    "reasons": [                               // gating notes for this tick
      "headroom_low",                          // emitted whenever w < 1.0
      "saturation_guard",                      // emitted when w < 1.0 and runnable queue > 0
      "initial_admit_ramp_wait"                // emitted when pre-SAT fresh launch ramp delays admission
    ]
  }
}
```

### Units & conventions

- All timestamps are **ISO-8601 UTC strings**. The dashboard parses them with
  `new Date(...)`.
- All KV sizes are **GB (float)**.
- `tick` is a monotonic integer per sidecar process; restarting the sidecar
  resets it to 0.
- The dashboard's "now" marker = the most recent record's `ts`, *not* the
  browser clock — this keeps the timeline truthful even if the browser and
  the sidecar machine disagree on time.

## How the dashboard derives visuals

| Visual                               | Source                                                              |
|--------------------------------------|---------------------------------------------------------------------|
| One row per agent                    | keys of `agents` (first-seen order, never reshuffled)               |
| Phase box                            | `agents[id].state` from `state_since` to next tick's `state_since` (or to `ts` if still active) |
| Phase color                          | reasoning=blue, tool_call=green, waiting=gray, evicted_waiting=orange, done=light gray |
| KV-cache % line                      | `vllm.kv_cache_used_pct` per tick                                   |
| EVICT marker (red)                   | `admission.evictions[*]` where `evicted == true`                    |
| ADMIT marker (green)                 | `admission.admissions[*]` where `admitted && !previously_evicted`   |
| READMIT marker (purple)              | `admission.admissions[*]` where `admitted && previously_evicted`    |
| SAT marker (dashed yellow)           | `"saturation_guard"` ∈ `admission.reasons` (low headroom with runnable queued agents) |
| Event tooltip                        | `{ts, tick, C, w, s_t, s_prev}` plus event-specific fields          |
| Phase tooltip                        | phase name, start, duration, agent's `kv_gb` at that tick           |

## Replay

Two ways to view a finished `sidecar.log`:

1. **Server replay**: `python -m sidecar_http --replay path/to/sidecar.log [--speed 1.0]`
   starts the same server but the publisher reads ticks from the file at the
   chosen speed. The dashboard at `/` works exactly as it does in live mode.

2. **In-browser replay**: in the dashboard header, click *Open sidecar.log…*
   and select a JSONL file. The browser parses it client-side and replays
   every tick into the same render path. Useful when you only have the log
   file and don't want to start a Python server. SSE is disconnected while
   in replay mode; click *Back to live* to reconnect.

Both paths consume the exact same JSON contract above.
