# ABC-Bench Runner

Component: `src/run_abc_bench_instrumented.py`

The runner launches ABC-Bench tasks through the OpenHands SDK against an
OpenAI-compatible vLLM server. It is responsible for copying task workspaces,
creating OpenHands conversations, collecting per-agent traces, publishing live
agent state for the sidecar, and writing result artifacts.

## Responsibilities

- Discover task directories under `--dataset-root`.
- Build an agent prompt from `task.yaml`.
- Copy each task into `--workspace-root` before execution.
- Construct an OpenHands `LLM`, `Agent`, and `Conversation`.
- Track every agent's phase lifecycle.
- Record rich tool-call traces through `ToolCallTraceCollector`.
- Capture patched vLLM KV usage from LiteLLM callbacks.
- Optionally run each task's `run-tests.sh`.
- Optionally embed the sidecar controller and dashboard feed.
- Write per-agent and run-level artifacts under `--results-root`.

The runner does not implement KV policy itself. It reports live state and calls
the sidecar controller hooks at phase boundaries.

## Main Flow

1. `main()` parses CLI flags and validates required model settings.
2. If `--sidecar-log-file` is set, the embedded sidecar thread starts.
3. Tasks are discovered with `find_task_dirs()` and expanded with
   `expand_tasks_round_robin()` when `--max-tasks` exceeds available tasks.
4. The launcher chooses one of three modes:
   - sequential execution
   - randomized wave execution
   - sidecar-gated admission execution
5. Each admitted task runs `_agent_thread()`, which calls `run_single_agent()`.
6. `run_single_agent()` runs the OpenHands conversation and writes artifacts in
   a `finally` block, so partial traces survive failures.
7. `run_summary.json` is written after all futures finish.
8. The embedded sidecar is stopped cleanly before process exit.

## Agent Lifecycle Tracking

`AgentPhaseTracker` owns the per-agent phase state. It listens to OpenHands
events and infers phase transitions:

| Event | Runner action |
| --- | --- |
| `run_start()` | Creates a `_LIVE_AGENTS` entry, records schedule-to-start waiting time, opens `reasoning`. |
| `MessageEvent` from user | If waiting, opens `reasoning`. |
| `ActionEvent` | Closes `reasoning`, opens `tool_call`, records pending tool metadata, notifies sidecar. |
| `ObservationEvent` or `UserRejectObservation` | Finalizes the tool call, optionally waits for readmission, opens `reasoning`. |
| `MessageEvent` from agent | Marks a possible terminal response, but keeps `reasoning` live because an SDK `ActionEvent` may still follow from the same model turn. |
| `AgentErrorEvent` | Releases held KV, closes current phase, moves to waiting. |
| `run_end()` | Finalizes unfinished tool calls or pending terminal text, marks the live agent done, releases held KV. |

Live phase names are consumed by the sidecar and dashboard. Important values are
`waiting`, `reasoning`, `tool_call`, `offloaded_waiting`, and `done`.

## Live State Contract

The module-level `_LIVE_AGENTS` dictionary is the embedded sidecar's live input.
It is protected by `_live_lock` and has one entry per agent id.

Common fields include:

| Field | Meaning |
| --- | --- |
| `task_id` | ABC-Bench task directory name. |
| `state` | Current phase or admission state shown in the dashboard. |
| `state_since` | ISO UTC timestamp for the current state. |
| `started_at` / `finished_at` | Run start and fixed finish timestamps. |
| `kv_blocks` / `kv_gb` | Latest per-agent KV estimate, refreshed either from LiteLLM usage or the sidecar usage endpoint. |
| `last_kv_updated` | Timestamp of the latest KV update. |
| `kv_offloaded` | Whether the agent is currently offloaded. |
| `admission_state` | Admission controller state such as `admitted` or `offloaded_ready`. |
| `tool_*` | Present only while the agent is inside a tool call. |

The runner also exposes a live snapshot callback to the trace collector. That
snapshot captures concurrency and KV fields at tool dispatch and completion.

## Launch Modes

### Sequential

Used when `--randomise-launch` and `--sidecar-admission-control` are both off.
Tasks run one after another. Agent ids use the suffix `seq<N>`.

### Randomized Waves

Enabled with `--randomise-launch`. `plan_launch_waves()` partitions tasks into
random batches of size `1..--max-agents-per-wave` and random delays between
`--min-wave-delay-s` and `--max-wave-delay-s`. A `ThreadPoolExecutor` runs each
agent in its own thread.

### Sidecar-Gated Admission

Enabled with `--sidecar-admission-control`. Planned agents are wrapped in
`sidecar.AgentLaunchSpec` and queued with `DynamicAdmissionController`.
The runner registers an admit callback. The sidecar invokes that callback only
when headroom, active-agent caps, and readmit priority allow work to run.

Randomized wave planning can still be requested, but in admission-control mode
the planned delay becomes metadata. The controller owns the actual launch time.

## OpenHands and LLM Setup

`make_llm()` creates the SDK `LLM` and passes `agent_id` in `extra_body`. The
vLLM patch reads this field so the connector can associate request ids with
agents. `make_agent()` enables:

- `TerminalTool`
- `FileEditorTool`
- `TaskTrackerTool`

These are the tools traced by `ToolCallTraceCollector`.

## KV Usage Capture

For each agent, `run_single_agent()` appends a temporary callback to
`litellm.success_callback`. The callback reads `kv_blocks_used` and
`kv_blocks_size_gb` from the patched vLLM usage payload and mirrors the latest
values into `_LIVE_AGENTS`.

The callback is removed in `finally`. The global LiteLLM callback list is
guarded by `_litellm_cb_lock` because many agents can start concurrently.

Scheduler-owned usage from `/agent_kv_cache/usage` can later overwrite these
values through the sidecar. That path is more authoritative for held requests.

## Sidecar Hooks

The runner calls these controller methods when admission control is enabled:

| Hook | Called from | Purpose |
| --- | --- | --- |
| `on_tool_call_start(agent_id, live_snapshot)` | `ActionEvent` | Classify tool call as short, long, or predictor-unavailable. |
| `wait_if_offloaded(agent_id)` | tool observation | Block an offloaded agent until readmitted. |
| `release_agent_kv(agent_id, reason)` | short calls, tool completion, final, error | Release held KV when CPU offload is not needed. |
| `finish_agent(agent_id)` | agent thread cleanup | Clear queued/offloaded state and release final holds. |

The runner opens a new reasoning phase at the readmission timestamp when an
agent had been offloaded. That keeps dashboard timing honest.

## Artifacts

Per agent, under `results_root/<task_id>/`:

| File | Producer | Notes |
| --- | --- | --- |
| `<agent_id>_trace.csv` | `AgentPhaseTracker.write_csv()` | Primary rich tool-call trace. |
| `<agent_id>_trace.jsonl` | `AgentPhaseTracker.write_trace_jsonl()` | JSONL companion to the CSV. |
| `<agent_id>_events.log` | `AgentPhaseTracker.write_event_log()` | Human-readable action/observation log. |
| `<agent_id>_prompt.txt` | runner | Exact prompt sent to the agent. |
| `<agent_id>_llm_metrics.json` | runner | OpenHands/LiteLLM metrics plus aggregate KV usage. |
| `<agent_id>_summary.json` | runner | Phase durations, tool-call counts, wall time, and errors. |
| `<agent_id>_test_result.json` | `maybe_run_tests()` | Present when `--run-tests` is enabled. |
| `<agent_id>_test_stdout.txt` / `<agent_id>_test_stderr.txt` | `maybe_run_tests()` | Captured test output. |
| `<agent_id>_error.json` | `_agent_thread()` | Present if the agent thread raises. |

Run-level outputs:

| File | Notes |
| --- | --- |
| `run_summary.json` | List of per-agent summaries or errors. |
| `sidecar.log` | JSONL tick log when the sidecar is enabled. |
| `prediction_metrics.json` | Written by the predictor step, not by the runner. |

## CLI Groups

Core flags:

- `--dataset-root`
- `--workspace-root`
- `--results-root`
- `--task-glob`
- `--max-tasks`
- `--max-iterations`
- `--run-tests`
- `--test-timeout-sec`
- `--model`
- `--base-url`
- `--api-key`

Launch flags:

- `--randomise-launch`
- `--max-agents-per-wave`
- `--min-wave-delay-s`
- `--max-wave-delay-s`
- `--launch-seed`

Sidecar flags:

- `--sidecar-log-file`
- `--sidecar-vllm-url`
- `--sidecar-interval`
- model geometry fields: `--sidecar-num-layers`, `--sidecar-num-kv-heads`,
  `--sidecar-head-dim`, `--sidecar-block-size`, `--sidecar-dtype`
- `--sidecar-total-gpu-blocks` (fallback when vLLM's /metrics omits block counts)
- `--sidecar-admission-control`
- admission knobs and vLLM endpoint overrides
- dashboard feed flags: `--sidecar-http-port`, `--sidecar-http-host`

## Failure Behavior

- Missing model/base URL/API key returns exit code `2`.
- Missing task directories returns exit code `2`.
- Missing sidecar model geometry while sidecar is enabled returns exit code `2`.
- Agent exceptions are recorded in per-agent files and re-raised to the future.
- The launcher collects failures as summary entries so other agents can finish.
- Trace, event, prompt, metrics, and summary files are written in `finally`
  whenever the tracker started.

## Change Guidelines

- Keep `AgentPhaseTracker` as the single owner of phase transitions.
- Keep sidecar policy decisions in `src/sidecar.py`.
- Update `dashboard/SCHEMA.md` when live-state fields exposed in sidecar ticks
  change.
- Preserve the `finally` persistence path in `run_single_agent()`; it is what
  makes failed runs debuggable.
- If new tool metadata is added to live state, consider adding it to the trace
  collector schema and predictor features together.
