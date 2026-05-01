# Proactive Concurrency Management for Agentic AI Workloads

This repository is a research and systems framework for running many coding
agents against ABC-Bench while keeping a vllm-ascend serving backend healthy on
Ascend NPUs. It is no longer just a tool-call duration predictor. The predictor
is one signal inside a larger feedback loop that monitors KV-cache growth,
admits new agents, and proactively offloads idle agents before the serving layer
runs out of cache.

The core idea is **growth-aware concurrency**: agent workloads do not consume
KV cache at a fixed rate. They alternate between reasoning phases that grow KV
state and tool-call phases where their KV state is idle but still resident. The
sidecar observes this behavior at runtime and decides how much concurrency the
system can safely support at the next poll.

## System Overview

```
ABC-Bench tasks
  -> instrumented OpenHands runner
  -> per-agent live state + tool-call traces
  -> optional tool remaining-time predictor
  -> sidecar dynamic admission controller
  -> vllm-ascend KV offload endpoint
  -> more agents admitted when KV headroom allows
```

Main components:

| Component | Role |
|-----------|------|
| `src/run_abc_bench_instrumented.py` | Runs ABC-Bench agents, records tool traces, publishes live phase/KV state, and lets the sidecar gate launches when admission control is enabled. |
| `src/sidecar.py` | Polls vLLM metrics and live agent state, computes KV headroom, scores idle agents, offloads under pressure, and admits queued agents. |
| `src/build_tool_predictor.py` | Trains or loads a model that predicts remaining tool-call time, used to score which idle agent is best to offload. |
| `src/collect_tool_trace.py` | Standalone collector for detailed ActionEvent -> ObservationEvent tool traces. |
| `src/vllm_patches/apply_patches.py` | Patches vLLM/vllm-ascend 0.13.0 to report per-agent KV usage and install the agent-aware offloading connector. |
| `run_abc-bench.sh` | YAML-driven wrapper that expands config, validates settings, starts the runner, sidecar, and optional predictor build. |

## Dynamic Admission Control

At each sidecar tick, the controller gathers:

- `C`: free KV-cache capacity in GB.
- `s_t`: current average per-active-agent KV usage.
- `s_{t-1}`: previous tick's average, memoized from the last poll.
- `w = C / min(s_t, s_{t-1})`: conservative concurrency headroom.
  `headroom_low` means `w <= w_threshold`; `saturation_guard` and
  `admission_blocked_by_pressure` are emitted only when effective low headroom
  keeps runnable queued work pending.

Policy:

1. **Saturation guard**: if effective headroom is at or below `w_threshold`,
   admit no queued agents this tick. Effective headroom is `w`, recomputed
   after any pressure offload as `w_after_offload` when offload frees KV
   capacity. `w_threshold` defaults to `2.0`.
2. **Active-agent cap**: if `max_active_agents` is positive, the sidecar
   admits no fresh or previously offloaded agents once the number of active
   admitted agents (`reasoning`, `tool_call`, or `waiting`) reaches that cap.
   Pending agents remain in the sidecar queue until a slot opens. A value of
   `0` disables this cap.
3. **Fresh launch pacing**: fresh tasks stay in the sidecar queue and are
   launched by the sidecar. Before the first real SAT, the sidecar admits at
   most one fresh queued task per
   `initial_admit_interval_s`. After SAT, `max_fresh_admits_per_tick` still
   caps fresh launches, defaulting to one per sidecar tick. READMITs are not
   delayed by the fresh-task ramp.
4. **Tool-call KV policy**: when a tool call starts, predict its remaining
   duration. Calls below `short_tool_call_threshold_s` stay resident and are
   excluded from pressure offload, and their held KV reservation is released.
   Longer calls keep the vLLM hold and are eligible for the pressure heap. If
   the predictor is unavailable on a first run, tool calls older than
   `fallback_long_tool_call_s` are treated as long idle candidates.
5. **Pressure offload**: if free KV percent is at or below
   `threshold_percent`, offload eligible idle `tool_call` agents by
   descending score:

   ```
   offload_score = agent_kv_usage_gb * predicted_remaining_tool_seconds
   ```

   When fallback mode is active, elapsed tool-call seconds replace the missing
   prediction in this score. Short predicted calls are excluded. Offloaded agents continue their current tool call. After the tool returns,
   their runner thread blocks before the next LLM call until the sidecar
   re-admits them.
   Successful offload records report `freed_gb` only from an exact vLLM
   free-block delta (`free_blocks_after - free_blocks_before`) or from an
   explicit endpoint value; the sidecar does not substitute the candidate's
   estimated KV size. Async connector offloads report `pending_async` until
   that exact delta is visible.
6. **Admission**: when effective `w > w_threshold` and the active-agent cap has
   room, launch queued agents. `threshold_percent` is not an admission gate; it
   only decides when pressure offload should run.
   Previously offloaded agents use a priority lane ahead of fresh agents. Fresh
   agents are admitted one at a time by default; the configured run path has
   no separate batch-submission knob.

Admission control is opt-in. Without it, the sidecar remains monitor-only and
the benchmark runner preserves the original launch behavior.

## vllm-ascend Patch

The vLLM patch provides two capabilities needed for the control loop:

1. Telemetry: OpenAI responses include `kv_blocks_used` and
   `kv_blocks_size_gb` when the request carries `agent_id`.
2. Control: `POST /agent_kv_cache/offload` marks held agent KV for
   connector-backed CPU offload, `POST /agent_kv_cache/release` frees held KV
   without offload, and `POST /agent_kv_cache/restore` notifies readmission.

The patch installs `AgentAwareOffloadingConnector`, a small subclass/wrapper of
vLLM's `OffloadingConnector`. It tracks `agent_id -> request_id -> block_ids`,
holds finished agent requests before their blocks reach vLLM's free queue, uses
the existing async transfer worker and the configured
`vllm_ascend.kv_offload.npu.NPUOffloadingSpec`, and changes only the save
decision logic.

Apply the patch on the NPU machine:

```bash
python src/vllm_patches/apply_patches.py \
  --vllm-dir /path/to/site-packages/vllm
```

The project targets vllm-ascend 0.13.0 with vLLM 0.13.0 on the NPU machine.
Following the official source-install flow, the starter script installs the
local Ascend runtime stack (`torch==2.8.0`, `torch-npu==2.8.0.post2`,
`triton-ascend==3.2.0`), clones vLLM v0.13.0 into `.vllm-src/`, installs it
with `VLLM_TARGET_DEVICE=empty` inside the project `.venv`, then clones
vllm-ascend v0.13.0 into `.vllm-ascend-src/`, initializes submodules, and
installs it editable for `SOC_VERSION=ascend910_9391` by default. Override
`VLLM_ASCEND_SOC_VERSION` if the target Ascend chip changes. Source clones use
GitHub SSH URLs by default; if SSH is unavailable from the NPU machine,
pre-populate those source directories or set `VLLM_GIT_URL` /
`VLLM_ASCEND_GIT_URL` to an accessible mirror or HTTPS URL. Set
`VLLM_ASCEND_INSTALL_METHOD=wheel` to use the guide's pre-built wheel path
instead. Before source builds, the script installs the CANN Python build
dependencies (`sympy`, `numpy<2.0.0`, and related utilities) into `.venv` and
exposes that site-packages path to the custom-op compiler. Custom-op
compilation defaults to `MAX_JOBS=16`; override `VLLM_ASCEND_MAX_JOBS` if the
build host needs a different cap. It also patches the cached source checkout so
CANN 8.5.1 host object files installed under `objects-*` are copied to the
top-level directory expected by `recompile_binary.py`. Runtime dependencies
advertised by the installed packages are then installed under constraints that
keep the Ascend torch/Triton/CANN stack pinned while filtering CUDA packages
and vLLM's upstream torch pins. The patcher targets the vLLM/vllm-ascend
0.13.0 layout used by the Ascend server.

## Quick Start

### 1. Run ABC-Bench with tracing

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml
```

Direct runner usage:

```bash
python src/run_abc_bench_instrumented.py \
  --dataset-root /data/ABC-Bench \
  --task-glob 'task_*' \
  --max-tasks 3 \
  --workspace-root ./abc_runs \
  --results-root ./abc_results \
  --model openai/Qwen3-30B-A3B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key dummy
```

Each agent writes:

```text
abc_results/<task_id>/agent_<...>_trace.csv
abc_results/<task_id>/agent_<...>_trace.jsonl
abc_results/<task_id>/agent_<...>_summary.json
abc_results/<task_id>/agent_<...>_llm_metrics.json
```

### 2. Train the tool remaining-time predictor

```bash
python src/build_tool_predictor.py \
  --trace-dir ./abc_results \
  --model hgb \
  --remaining \
  --save-model ./duration_model.joblib
```

The saved model can be used by the sidecar for idle-agent offload scoring.

### 3. Enable the sidecar

In `config/abc-bench_config.yaml`:

```yaml
sidecar:
  enabled: true
  interval: 1
  num_layers: 64
  num_kv_heads: 8
  head_dim: 128
  block_size: 128
  total_gpu_blocks: 1000
```

This starts the embedded sidecar and writes JSONL records containing vLLM
metrics, live agent state, and admission diagnostics when enabled. If
`sidecar.vllm_url` is omitted, the wrapper derives it from `llm.base_url` by
removing the trailing `/v1`.

### 4. Enable dynamic admission control

```yaml
sidecar:
  admission_control:
    enabled: true
    threshold_percent: 10.0
    w_threshold: 2.0
    initial_admit_interval_s: 1.0
    max_fresh_admits_per_tick: 1
    max_active_agents: 32
    fallback_long_tool_call_s: 5.0
    offload_timeout_s: 10.0
    exact_freed_gb_timeout_s: 5.0
```

The omitted predictor, endpoint, timeout, and short-tool-call settings use the
wrapper defaults. In particular, `predictor_model` defaults to
`prediction.save_model`, and offload/restore/release endpoints default under
`sidecar.vllm_url`. `threshold_percent` is a free-KV threshold, so `10.0`
means pressure offload begins when free KV capacity is at or below 10% of the
configured or reported total. `w_threshold` is an admission headroom threshold,
so `2.0` admits queued agents only after effective `w` is greater than two.
Accepted async offloads may report `freed_gb_source: pending_async` until vLLM
reports the exact free-block delta.

Then run:

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml
```

## Trace Schema

Each agent produces `<agent_id>_trace.csv` with one row per tool call:

| Column | Description |
|--------|-------------|
| `agent_id`, `task_id`, `tool_seq` | Agent/task identity and tool-call order. |
| `action_id`, `tool_call_id` | OpenHands action identifiers when available. |
| `start_ts`, `end_ts`, `duration_s` | Tool-call timing with UTC timestamps. |
| `tool_name`, `tool_command`, `tool_args_json` | Tool type and serialized inputs. |
| `tool_payload_bytes` | Size of serialized tool arguments. |
| `outcome`, `observation_*` | Tool result status, output size, preview, and hash. |
| `start_*`, `end_*` | Dispatch/end concurrency and KV-cache snapshots. |
| `phase_seq`, `cumulative_reasoning_s` | Context used by the predictor. |

The runner also writes matching JSONL traces for lossless downstream analysis.

## Predictor Features

The predictor uses dispatch-time features only, avoiding leakage from the tool
result:

- Tool type: `terminal`, `file_editor`, `task_tracker`.
- Command structure: length, tokens, pipes, semicolons, payload bytes.
- Command keywords: docker, build, test, install, git.
- Sequential context: tool sequence, cumulative reasoning time, prior tool
  count, prior average tool duration.
- Optional runtime context: active agents and KV-cache snapshots.

## Output Artifacts

Per agent:

- `agent_*_trace.csv`
- `agent_*_trace.jsonl`
- `agent_*_summary.json`
- `agent_*_llm_metrics.json`
- `agent_*_prompt.txt`
- `agent_*_test_result.json`

Aggregate:

- `run_summary.json`
- `prediction_metrics.json`
- `sidecar.log` JSONL, when the sidecar is enabled

## Design Assumptions

- vLLM runs through vllm-ascend on Ascend NPUs.
- `agent_id` is propagated through `litellm_extra_body`.
- Per-agent KV response fields are telemetry. Actual offload must happen
  inside vLLM's scheduler/block pool.
- Dynamic admission is opt-in to preserve baseline benchmark behavior.
- Tool-call prediction estimates remaining wall-clock tool time, not full task
  completion time.
