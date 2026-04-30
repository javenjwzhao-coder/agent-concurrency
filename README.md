# Proactive Concurrency Management for Agentic AI Workloads

This repository is a research and systems framework for running many coding
agents against ABC-Bench while keeping a vllm-ascend serving backend healthy on
Ascend NPUs. It is no longer just a tool-call duration predictor. The predictor
is one signal inside a larger feedback loop that monitors KV-cache growth,
admits new agents, and proactively evicts idle agents before the serving layer
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
  -> vllm-ascend KV eviction endpoint
  -> more agents admitted when KV headroom allows
```

Main components:

| Component | Role |
|-----------|------|
| `src/run_abc_bench_instrumented.py` | Runs ABC-Bench agents, records tool traces, publishes live phase/KV state, and lets the sidecar gate launches when admission control is enabled. |
| `src/sidecar.py` | Polls vLLM metrics and live agent state, computes KV headroom, scores idle agents, evicts under pressure, and admits queued agents. |
| `src/build_tool_predictor.py` | Trains or loads a model that predicts remaining tool-call time, used to score which idle agent is best to evict. |
| `src/collect_tool_trace.py` | Standalone collector for detailed ActionEvent -> ObservationEvent tool traces. |
| `src/vllm_patches/apply_patches.py` | Patches vLLM/vllm-ascend 0.11.x-0.13.x to report per-agent KV usage and expose `POST /agent_kv_cache/evict`. |
| `run_abc-bench.sh` | YAML-driven wrapper that expands config, validates settings, starts the runner, sidecar, and optional predictor build. |

## Dynamic Admission Control

At each sidecar tick, the controller gathers:

- `C`: free KV-cache capacity in GB.
- `s_t`: current average per-active-agent KV usage.
- `s_{t-1}`: previous tick's average, memoized from the last poll.
- `w = C / min(s_t, s_{t-1})`: conservative concurrency headroom.
  `headroom_low` means `w < 1`; `saturation_guard` is emitted only when
  low headroom actually blocks a runnable queued agent.

Policy:

1. **Saturation guard**: if `w < 1`, admit no fresh agents this tick.
2. **Initial launch ramp**: before the first real SAT, admit at most one fresh
   queued task per `initial_admit_interval_s`. Once SAT has happened, normal
   capacity math drains the fresh queue on each sidecar tick. READMITs are not
   delayed by the ramp.
3. **Tool-call KV policy**: when a tool call starts, predict its remaining
   duration. Calls below `short_tool_call_threshold_s` are pinned in accelerator
   KV cache and excluded from pressure offload. Longer calls are also pinned,
   but remain eligible for the pressure heap.
4. **Pressure offload**: if `C <= threshold_gb`, offload eligible idle
   `tool_call` agents by descending score:

   ```
   eviction_score = agent_kv_usage_gb * predicted_remaining_tool_seconds
   ```

   Pinned short calls are excluded. Pinned long calls are unpinned only when
   selected for offload. Offloaded agents continue their current tool call. After the tool returns,
   their runner thread blocks before the next LLM call until the sidecar
   re-admits them.
5. **Admission**: when `C > threshold_gb` and `w >= 1`, launch queued agents.
   Previously evicted agents use a priority lane ahead of fresh agents.

Admission control is opt-in. Without it, the sidecar remains monitor-only and
the benchmark runner preserves the original launch behavior.

## vllm-ascend Patch

The vLLM patch provides two capabilities needed for the control loop:

1. Telemetry: OpenAI responses include `kv_blocks_used` and
   `kv_blocks_size_gb` when the request carries `agent_id`.
2. Control: `POST /agent_kv_cache/pin` pins/unpins agent-owned cached blocks,
   `POST /agent_kv_cache/offload` offloads unpinned cached blocks, and
   `POST /agent_kv_cache/evict` remains a backward-compatible alias.

The scheduler hook tracks `agent_id -> request_id -> block_ids`, records cached
blocks as requests finish, pins short-call blocks against normal cache
pressure, and offloads only unpinned cached blocks whose `ref_cnt == 0`
when requested by the sidecar. This is required because the sidecar's GB
metrics are not enough to safely mutate vLLM's internal KV block pool.

Apply the patch on the NPU machine:

```bash
python src/vllm_patches/apply_patches.py \
  --vllm-dir /path/to/site-packages/vllm
```

The project targets vllm-ascend with vLLM 0.13.0 on the NPU machine. The
starter script installs that pair into the project `.venv` once with
`--no-deps`, then installs the runtime dependencies advertised by the wheels
while filtering out CUDA packages and the Ascend-owned torch/torch-npu stack.
The patcher keeps anchors compatible with nearby 0.11.x-0.13.x layouts where
practical.

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

The saved model can be used by the sidecar for idle-agent eviction scoring.

### 3. Enable the sidecar

In `config/abc-bench_config.yaml`:

```yaml
sidecar:
  enabled: true
  vllm_url: "http://127.0.0.1:8027"
  interval: 1
  num_layers: 64
  num_kv_heads: 8
  head_dim: 128
  block_size: 128
  dtype: "bfloat16"
  total_gpu_blocks: 1000
```

This starts the embedded sidecar and writes JSONL records containing vLLM
metrics, live agent state, and admission diagnostics when enabled.

### 4. Enable dynamic admission control

```yaml
sidecar:
  admission_control:
    enabled: true
    threshold_gb: 0.1
    initial_admit_interval_s: 2.0
    short_tool_call_threshold_s: 2.0
    predictor_model: null        # defaults to prediction.save_model
    pin_endpoint: null           # defaults to <sidecar.vllm_url>/agent_kv_cache/pin
    offload_endpoint: null       # defaults to <sidecar.vllm_url>/agent_kv_cache/offload
    eviction_endpoint: null      # backward-compatible alias for offload_endpoint
    eviction_timeout_s: 2.0
```

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
- Per-agent KV response fields are telemetry. Actual eviction must happen
  inside vLLM's scheduler/block pool.
- Dynamic admission is opt-in to preserve baseline benchmark behavior.
- Tool-call prediction estimates remaining wall-clock tool time, not full task
  completion time.
