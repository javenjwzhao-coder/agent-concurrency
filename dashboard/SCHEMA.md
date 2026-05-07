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
    "num_gpu_blocks_free":  550,               // int   | null
    "scheduler_preemptions_total": 0           // int   | null — vLLM scheduler preemptions
    // "error": "<string>" if /metrics fetch failed this tick
  },

  "agents": {                                  // Per-agent live state (key = agent_id).
    "agent_task_3_w0_2": {
      "task_id": "task_3",                     // str
      "state":   "reasoning",                  // "waiting"|"reasoning"|"tool_call"|"offloaded_waiting"|"done"
      "state_since": "2026-04-29T14:29:55.000+00:00",  // ISO-8601 — phase entry timestamp
      "started_at": "2026-04-29T14:29:54.000+00:00",   // optional ISO-8601 — run start
      "finished_at": null,                     // optional ISO-8601|null — fixed when state becomes done
      "kv_gb":   0.34,                         // float | null
      "kv_blocks": 1024,                       // int   | null
      "kv_usage_source": "llm_response",        // optional str — source of latest live KV telemetry
      "admission_state": "admitted",           // "admitted"|"offloaded_pending_tool"|"offloaded_ready"
      "kv_policy": "idle_short",               // optional: "idle_short"|"idle_long"|"readmitted"|...
      "tool_name": "execute_bash"              // present only during "tool_call"
      // …additional tool-call fields (tool_args_json, tool_payload_bytes, …) may also appear
    }
  },

  "admission": {                               // Controller decisions for this tick.
    "enabled": true,                           // bool
    "C":       5.59,                           // float|null — free KV in GB this tick
    "C_percent": 16.67,                        // float|null — free KV percent, derived from used percent when needed
    "kv_total_gb": 33.55,                      // float — total KV capacity used for percent thresholds
    "s_t":     0.31,                           // float — current avg KV usage of active agents
    "s_prev":  0.29,                           // float|null — previous tick's avg
    "w":       18.03,                          // float|null — effective headroom used for admission gating
    "w_threshold": 2.0,                         // float — minimum headroom before admitting queued agents
    "w_source": "current",                     // "current"|"after_offload"
    "w_before_offload": null,                  // float|null — pre-offload diagnostic, present only when changed
    "threshold_percent": 10.0,                 // float — configured free-KV pressure threshold
    "threshold_gb": 3.36,                      // float|null — derived pressure threshold for this tick
    "threshold_source": "percent",             // string — percent, or gb_legacy for old configs
    "fallback_long_tool_call_s": 30.0,         // float — first-run predictor fallback age
    "first_saturation_seen": false,            // bool — initial launch ramp is disabled after first SAT
    "initial_admit_interval_s": 2.0,           // float — fresh-admit interval before first SAT
    "max_fresh_admits_per_tick": 1,            // int — fresh queued agents admitted per sidecar tick
    "max_active_agents": 32,                   // int — active-agent cap, 0 means unlimited
    "active_agents": 16,                       // int — agents in reasoning|tool_call|waiting
    "active_agent_slots": 16,                  // int|null — remaining cap slots, null when uncapped
    "next_initial_admit_in_s": null,           // float|null — seconds until next ramped fresh admit
    "active_agent_samples": 4,                 // int
    "pressure": true,                          // bool — pressure gate, C_percent <= threshold_percent

    "queue": {                                 // pending admissions
      "fresh": 2,
      "offloaded_ready": 0,
      "offloaded_pending_tool": 1,
      "idle_short": 3,
      "idle_long": 4
    },

    "heap_candidates": [                       // ranked idle-agent offload heap (snapshot)
      { "agent_id": "...", "kv_gb": 0.5, "predicted_remaining_s": 8.0,
        "tool_elapsed_s": null, "policy_reason": "eligible_for_pressure_offload",
        "e_s": 4.0 }
    ],
    "skipped_candidates": [                    // tool-call agents excluded from offload
      { "agent_id": "...", "reason": "idle_short" }
    ],
    "kv_policy_events": [                      // immediate tool-start policy decisions
      { "agent_id": "...", "policy": "idle_long",
        "predicted_remaining_s": 8.0, "tool_elapsed_s": null,
        "threshold_s": 2.0, "fallback_long_tool_call_s": 30.0 }
    ],

    "offloads": [                              // offload attempts fired this tick
      { "agent_id": "...", "offloaded": true,
        "kv_gb": 0.5, "freed_gb": 0.48,
        "freed_blocks": 48, "free_blocks_before": 120,
        "free_blocks_after": 168,
        "freed_gb_source": "vllm_free_blocks_delta",
        "pending": false, "held_requests": 1,
        "known_blocks": 48, "offload_jobs": 1,
        "predicted_remaining_s": 8.0, "e_s": 4.0,
        "status_code": 200, "reason": "ok" },
      { "agent_id": "...", "offloaded": false,
        "kv_gb": 0.5, "status_code": 409,
        "reason": "no tracked KV blocks for agent" }
    ],

    "admissions": [                            // events fired this tick
      { "agent_id": "...", "admitted": true, "previously_offloaded": false,
        "w": 18.03, "w_threshold": 2.0, "w_source": "current",
        "admitted_at": "2026-04-29T14:30:00.234567+00:00" }
    ],

    "reasons": [                               // gating notes for this tick
      "headroom_low",                          // emitted whenever w <= w_threshold
      "saturation_guard",                      // emitted when effective w <= w_threshold blocks runnable queue
      "admission_blocked_by_pressure",         // emitted when queued work stays pending due to effective w pressure
      "missing_headroom",                      // emitted when pressure blocks admission before w can be computed
      "active_agent_cap",                      // emitted when max_active_agents constrains admissions
      "admission_blocked_by_active_agent_cap", // emitted when no active-agent slots are available
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
| Agent row label                      | `agent_id (elapsed: N secs)`, then `agent_id (E2E: N secs)` after `state=done`; uses `started_at`/`finished_at` when present |
| Phase box                            | `agents[id].state` from `state_since` to next tick's `state_since` (or to `ts` if still active). The dashboard renders the post-tool pending-release sub-state as `reasoning` so normal tool-completion cleanup does not appear as a separate gray wait. If sampling misses a short reasoning span between `waiting` and `tool_call`, the dashboard bridges the unobserved interval as `reasoning` rather than extending `waiting` into the next tool call. |
| Phase color                          | reasoning=blue, tool_call=green, waiting=gray, offloaded_waiting=orange, done=light gray |
| KV-cache % line                      | `vllm.kv_cache_used_pct` per tick                                   |
| Offload threshold line               | `100 - admission.threshold_percent` per tick                        |
| Free-KV pressure badge               | `admission.C`, `admission.C_percent`, `admission.threshold_percent`, and `admission.pressure` |
| vLLM preempt badge                   | `vllm.scheduler_preemptions_total` from vLLM `/metrics`             |
| OFFLOAD marker (red)                 | `admission.offloads[*]` where `offloaded == true` (KV pushed to CPU) |
| ADMIT marker (green)                 | `admission.admissions[*].admitted_at` where `admitted && !previously_offloaded`; falls back to tick `ts` if absent |
| READMIT marker (cyan)                | `admission.admissions[*].admitted_at` where `admitted && previously_offloaded`; falls back to tick `ts` if absent |
| SAT marker (dashed yellow)           | `"saturation_guard"` ∈ `admission.reasons` (low effective headroom blocks queued agents) |
| Event tooltip                        | `{ts, tick, C, C_percent, threshold_percent, threshold_gb, pressure, w, w_threshold, w_source, s_t, s_prev}` plus event-specific fields |
| Phase tooltip                        | phase name, start, duration, agent's `kv_gb` at that tick           |

## Replay

Three ways to view a finished `sidecar.log`:

1. **Server replay**: `python -m sidecar_http --replay path/to/sidecar.log [--speed 1.0]`
   starts the same server but the publisher reads ticks from the file at the
   chosen speed. The dashboard at `/` works exactly as it does in live mode.

2. **In-browser replay**: in the dashboard header, click *Open sidecar.log…*
   and select a JSONL file. The browser parses it client-side and replays
   every tick into the same render path. Useful when you only have the log
   file and don't want to start a Python server. SSE is disconnected while
   in replay mode; click *Back to live* to reconnect.

3. **Standalone snapshot**: after loading live history or replaying a log,
   click *Save standalone HTML*. Browsers with the File System Access API prompt
   for a save location; other browsers download a single `.html` file with the
   currently loaded tick records plus dashboard assets embedded. That file opens
   on another machine without `/state`, `/stream`, or a separate `sidecar.log`.

All paths consume the exact same JSON contract above.
