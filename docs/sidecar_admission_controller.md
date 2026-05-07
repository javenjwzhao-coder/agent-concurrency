# Sidecar Admission Controller

Component: `src/sidecar.py`

The sidecar is the policy and telemetry loop. It polls vLLM KV metrics, reads
live agent state, refreshes per-agent scheduler usage, decides when to admit
queued agents, decides when to offload long idle tool calls, and writes one
JSONL tick record per interval.

The sidecar can run embedded inside the ABC-Bench runner or as a standalone
process reading a live-state JSON file.

## Responsibilities

- Convert model geometry into KV bytes per block.
- Parse vLLM Prometheus metrics across standard vLLM and vllm-ascend variants.
- Compute global KV capacity, free memory, pressure thresholds, and headroom.
- Track queued fresh agents and offloaded agents waiting for readmission.
- Classify tool calls using an optional remaining-time predictor.
- Release held KV for short or completed tool calls.
- Offload selected long idle agents through patched vLLM endpoints.
- Poll exact free-block deltas after accepted offload requests.
- Emit decision details for the dashboard and post-run analysis.

## Runtime Modes

### Embedded Mode

The runner imports `sidecar.py` and starts `run_loop()` in a daemon thread. In
this mode, `get_agents()` reads the runner's in-memory `_LIVE_AGENTS` dict.
There is no live-state file serialization.

Embedded mode is the normal path for admission control because the runner and
sidecar can coordinate through `DynamicAdmissionController`.

### Standalone Mode

Run:

```bash
python src/sidecar.py \
  --vllm-url http://localhost:8027 \
  --live-state ./abc_results/.live_state.json \
  --log-file ./sidecar.log \
  --interval 5 \
  --num-layers 64 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --block-size 128 \
  --dtype bfloat16
```

Standalone mode is useful for telemetry attachment, but the current runner no
longer writes a live-state file by default. Admission-control launches are an
embedded-runner feature.

## KV Accounting

`bytes_per_block()` computes one vLLM KV block's size:

```text
block_size * num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
```

`poll_vllm()` calls `<vllm-url>/metrics` and returns:

- `kv_cache_used_pct`
- `num_gpu_blocks_total`
- `num_gpu_blocks_used`
- `num_gpu_blocks_free`
- `kv_total_gb`
- `kv_used_gb`
- `kv_free_gb`
- `scheduler_preemptions_total`

The metric parser recognizes standard vLLM and vllm-ascend metric names. When
`total_gpu_blocks` is configured, it overrides Prometheus totals because Ascend
deployments can expose a physical allocation that differs from the scheduler
limit set by `--num-gpu-blocks-override`.

## Controller State

`DynamicAdmissionController` owns admission and offload state:

| State | Purpose |
| --- | --- |
| `_fresh` | Never-started `AgentLaunchSpec` queue. |
| `_offloaded_pending` | Agents whose KV was offloaded while the tool call was still running. |
| `_offloaded_ready` | Offloaded agents whose tool result arrived and are waiting for readmission. |
| `_offloaded_events` | Per-agent threading events used to block/resume agent threads. |
| `_idle_short` | Tool-call agents predicted short; excluded from offload. |
| `_idle_long` | Tool-call agents eligible for pressure offload. |
| `_released_agents` | Agents whose held KV was released without offload. |
| `_kv_policy_events` | Recent policy decisions included in the next tick record. |
| `_prev_avg_gb` | Previous active-agent average KV size for conservative headroom. |

The controller uses an `RLock` because runner callbacks and the sidecar tick
thread can touch policy state concurrently.

## Tool-Start Classification

`on_tool_call_start(agent_id, agent)` runs immediately after the runner receives
an OpenHands `ActionEvent`.

Decision order:

1. If the controller is disabled, record a skipped event.
2. Try to predict remaining time from live tool metadata.
3. If prediction is unavailable and elapsed time is already beyond
   `fallback_long_tool_call_s`, mark the agent `idle_long`.
4. If prediction is unavailable and fallback is not reached, mark
   `predictor_unavailable`.
5. If prediction is below `short_tool_call_threshold_s`, release held KV and
   mark `idle_short`.
6. Otherwise mark `idle_long`.

Short calls are released quickly because keeping them pinned for offload would
usually cost more than it saves.

## Admission Tick

`on_tick()` is called once per sidecar loop. It returns the `admission` object
embedded in every `sidecar.log` record.

Key quantities:

| Symbol | Field | Meaning |
| --- | --- | --- |
| `C` | `C` | Free KV in GB. |
| `C_percent` | `C_percent` | Free KV as percent of total. |
| `s_t` | `s_t` | Current average KV GB among active, non-offloaded agents. |
| `s_prev` | `s_prev` | Previous tick's active-agent average. |
| `w` | `w` | `C / min(s_t, s_prev)`, using whichever samples are known. |
| threshold | `threshold_percent` / `threshold_gb` | Free-KV pressure threshold. |

Pressure offload and admission are separate:

- `threshold_percent` controls when pressure offload is considered.
- `w_threshold` controls whether queued work can be admitted.

Agents admit only when effective headroom is above `w_threshold`. Unknown
headroom can admit one agent to bootstrap measurements only while the controller
is not already under free-KV pressure.

## Pressure Offload

When free KV is at or below the pressure threshold, the sidecar builds an
offload heap from current `tool_call` agents.

Candidates are skipped when:

- the agent is already offloaded
- the tool call is marked `idle_short`
- held KV has already been released
- KV usage is missing or zero
- predictor is unavailable and fallback elapsed time has not been reached

Candidate score:

```text
score = agent_kv_gb * predicted_remaining_tool_seconds
```

When prediction is unavailable but fallback makes the call eligible, elapsed
tool time is used as the score multiplier.

The sidecar sends a POST to the offload endpoint, usually:

```text
<vllm-url>/agent_kv_cache/offload
```

An accepted async offload is marked as `pending` until vLLM reports a free-block
increase or the exact-freed timeout expires.

## Exact Freed-Memory Accounting

After an accepted offload, `_measure_exact_freed_gb()` polls vLLM metrics until
one of these happens:

- `num_gpu_blocks_free` increases above the pre-offload value
- `exact_freed_gb_timeout_s` expires

When a positive block delta appears, the sidecar records:

- `freed_blocks`
- `free_blocks_before`
- `free_blocks_after`
- `freed_gb`
- `freed_gb_source = "vllm_free_blocks_delta"`

If async transfer is still pending, the event reports
`freed_gb_source = "pending_async"`.

## Admission Order

`_admit_with_limit_locked()` admits in this order:

1. Offloaded-ready agents, which need restore/readmission.
2. Fresh queued agents.

Fresh agents are additionally constrained by:

- `max_fresh_admits_per_tick`
- initial ramp pacing before the first saturation event
- `max_active_agents`, when non-zero

Readmitted agents call the restore endpoint, usually:

```text
<vllm-url>/agent_kv_cache/restore
```

The agent thread unblocks after the controller sets its event.

## Release Paths

`release_agent_kv(agent_id, reason)` sends a release request to vLLM, usually:

```text
<vllm-url>/agent_kv_cache/release
```

Common reasons:

- `short_tool_call`
- `tool_complete`
- `final`
- `error`
- `release`

Release removes the agent from idle candidate sets and marks its live state as
`kv_policy = "released"`.

## Scheduler Usage Refresh

When `usage_endpoint` is configured, `_refresh_agent_kv_usage_locked()` queries
vLLM for each active, non-offloaded agent. Requests are issued in a bounded
parallel pool capped by `max_active_agents` when that cap is configured. If
`max_active_agents` is `0`, the pool spans the current active targets:

```text
GET /agent_kv_cache/usage?agent_id=<id>
```

The response updates live state with scheduler-owned usage:

- `kv_blocks`
- `kv_gb`
- `kv_usage_source = "scheduler_usage"`
- active/held request counts
- resident/offloadable block counts

This is preferred over callback usage because the connector can report held
finished requests that no longer look active to normal response telemetry.

## Tick Log Shape

Every sidecar loop writes one JSON object to `sidecar.log`:

```json
{
  "ts": "...",
  "tick": 12,
  "vllm": {},
  "agents": {},
  "admission": {}
}
```

The canonical schema is [../dashboard/SCHEMA.md](../dashboard/SCHEMA.md).
The dashboard reads this record directly from the HTTP feed or from a replayed
log file.

## Public Functions

| Function | Purpose |
| --- | --- |
| `bytes_per_block()` | Convert model geometry to bytes per KV block. |
| `poll_vllm()` | Return global KV cache statistics from Prometheus metrics. |
| `run_loop()` | Main sidecar polling loop. |
| `default_offload_endpoint()` | Build default offload URL from vLLM base URL. |
| `default_restore_endpoint()` | Build default restore URL. |
| `default_release_endpoint()` | Build default release URL. |
| `default_usage_endpoint()` | Build default usage URL. |

## Failure Behavior

- Missing vLLM metrics produce a warning once and fields become `null`.
- Missing free capacity blocks admission/offload decisions for that tick.
- Offload HTTP errors are recorded in `admission.offloads`.
- Offload timeouts become `reason = "offload_request_timeout"`.
- Missing predictor becomes a reason in tick reports, but fallback elapsed time
  can still classify long-running tool calls.
- Release and restore failures are recorded in live state as result payloads;
  the sidecar does not crash the runner for failed cleanup calls.

## Change Guidelines

- Keep pressure threshold and admission headroom as separate concepts.
- Keep exact freed memory based on vLLM block deltas or explicit endpoint data;
  do not report estimated freed memory as exact.
- Add new tick fields to `dashboard/SCHEMA.md` and dashboard rendering tests.
- Be careful with lock ordering. Runner callbacks and sidecar ticks can race.
- Treat release/final/error paths as safety-critical; stale held KV can reduce
  capacity for future agents.
