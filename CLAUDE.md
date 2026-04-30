# CLAUDE.md

This file guides Claude Code and other coding agents working in this
repository.

## Platform

This project runs on **Ascend NPU** hardware. The inference backend is
**vllm-ascend**, not plain upstream vLLM. `start_vllm.sh` installs
vLLM/vllm-ascend 0.13.0 into the project `.venv` with `--no-deps` and reuses
that install on later starts. It installs wheel metadata runtime dependencies
while filtering CUDA packages and the Ascend-owned torch/torch-npu stack, which
comes from the shared `/opt/vllm/venv`. When investigating internals or
applying patches, target the installed vllm-ascend/vLLM package and keep
version-specific anchors in mind.

## Project Purpose

This repository is a **proactive, growth-aware concurrency management framework
for agentic AI workloads**.

The original tool-call predictor is still important, but it is one part of a
larger runtime control loop:

1. Run many ABC-Bench coding agents through OpenHands.
2. Trace their reasoning/tool phases and per-agent KV-cache usage.
3. Train or load a tool remaining-time predictor.
4. Run a sidecar that polls vLLM and live agent state.
5. Dynamically admit new agents when KV headroom allows.
6. Proactively evict idle tool-call agents when KV pressure rises.

The goal is to maximize useful agent concurrency without exhausting KV cache on
vllm-ascend.

## Mental Model

Agent workloads have bursty KV growth:

- During `reasoning`, an agent is actively using vLLM and grows KV state.
- During `tool_call`, the agent is idle from the model's perspective, but its
  KV blocks may remain resident.
- Fresh agents should be admitted only when recent average KV growth suggests
  there is safe headroom.
- Idle tool-call agents are the best eviction candidates because their current
  tool can keep running while their next model call is deferred.

The sidecar implements this policy. The predictor helps rank idle agents by:

```text
eviction_score = kv_usage_gb * predicted_remaining_tool_seconds
```

## Dependencies

There is no `requirements.txt`. Install manually as needed:

```bash
pip install openhands-sdk openhands-tools pyyaml pydantic scikit-learn numpy pandas joblib requests
```

`pytest` may not be installed in the base environment. Existing lightweight
tests can be run through direct Python harnesses.

## Common Commands

Run agents via YAML config:

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml
bash run_abc-bench.sh --config config/abc-bench_config.yaml --dry-run
bash run_abc-bench.sh --config config/abc-bench_config.yaml --override dataset.max_tasks=2
```

Run agents directly:

```bash
python src/run_abc_bench_instrumented.py \
  --dataset-root /data/ABC-Bench \
  --task-glob 'task_*' \
  --max-tasks 6 \
  --model openai/Qwen3-30B-A3B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key dummy \
  --randomise-launch \
  --max-agents-per-wave 4 \
  --min-wave-delay-s 2 \
  --max-wave-delay-s 15 \
  --launch-seed 42 \
  --run-tests \
  --results-root ./abc_results
```

Train the remaining-time predictor:

```bash
python src/build_tool_predictor.py \
  --trace-dir ./abc_results \
  --model hgb \
  --remaining \
  --save-model ./duration_model.joblib
```

Run predictor inference:

```bash
python src/build_tool_predictor.py \
  --trace-dir ./new_traces \
  --load-model ./duration_model.joblib \
  --predict-only \
  --output-predictions ./predictions.csv
```

Start vLLM backend on Ascend:

```bash
bash start_vllm.sh
```

Patch vllm-ascend/vLLM for per-agent KV telemetry and eviction:

```bash
python src/vllm_patches/apply_patches.py \
  --vllm-dir /path/to/site-packages/vllm
```

## Architecture

### Runner

`src/run_abc_bench_instrumented.py` is the orchestrator:

```text
ABC-Bench tasks
  -> YAML/CLI config
  -> launch wave planning
  -> ThreadPoolExecutor
  -> one OpenHands agent per task
  -> AgentPhaseTracker callbacks
  -> per-tool traces, summaries, LLM metrics
  -> optional embedded sidecar
  -> optional predictor build
```

When dynamic admission is disabled, planned waves submit directly to the thread
pool. When admission is enabled, planned agents become `AgentLaunchSpec`s in the
sidecar waiting queue, and the sidecar submits them when headroom allows.

### AgentPhaseTracker and Tool Tracing

The tracker keeps live phase state for the sidecar and writes detailed
tool-call traces:

| Phase | Opens on | Closes on |
|-------|----------|-----------|
| `waiting` | `run_start()` or agent message | next event or `run_end()` |
| `reasoning` | `run_start()` or `ObservationEvent` | `ActionEvent` |
| `tool_call` | `ActionEvent` | `ObservationEvent` |

On `ActionEvent`, the live state includes tool metadata such as tool name,
serialized args, payload bytes, phase sequence, and start time. On
`ObservationEvent`, an evicted agent blocks before re-entering `reasoning`
until the sidecar re-admits it.

### Sidecar

`src/sidecar.py` can run embedded in the runner or standalone. It polls:

- vLLM Prometheus metrics for total/used/free KV blocks.
- Live per-agent state from the runner.
- Per-agent KV telemetry mirrored from vLLM responses.

Dynamic admission computes:

```text
C = free KV capacity in GB
s_t = average KV usage across active agents at this tick
s_prev = previous tick's average
w = C / min(s_t, s_prev)
```

`headroom_low` means `w < 1`. `saturation_guard` is emitted only when
`w < 1` and there is runnable queued work (`fresh` or `evicted_ready`) that
cannot be admitted.

Policy order:

1. If `w < 1`, admit no new fresh agents.
2. Before the first real SAT, admit at most one fresh queued task per
   `initial_admit_interval_s`. After first SAT, fresh admissions use normal
   sidecar capacity math. READMITs bypass this launch ramp.
3. At tool-call start, predict remaining duration. Predicted-short calls
   (`< short_tool_call_threshold_s`) are pinned in accelerator KV cache.
   Predicted-long calls are pinned too, but remain eligible for pressure
   offload through the heap.
4. If `C <= threshold_gb`, offload highest-scoring eligible long idle
   `tool_call` agents. The sidecar unpins selected long agents immediately
   before offload. Pinned short calls are never offloaded by sidecar.
5. If `C > threshold_gb` and `w >= 1`, admit from the waiting queue.

The waiting queue has two lanes: previously evicted agents first, then fresh
agents FIFO.

### vllm-ascend Patch

`src/vllm_patches/apply_patches.py` patches vLLM/vllm-ascend to support both
telemetry and control.

Telemetry:

- Adds `agent_id` to chat requests.
- Adds `kv_blocks_used` and `kv_blocks_size_gb` to usage responses.

Control:

- Adds `POST /agent_kv_cache/evict`.
- Forwards `evict_agent_kv(...)` through AsyncLLM, core client, engine core,
  scheduler, KV cache manager, and block pool.
- Tracks `agent_id -> request_id -> block_ids` in the scheduler.
- Evicts only cached blocks with `ref_cnt == 0` for proactive sidecar eviction.

Do not try to evict from sidecar using only `kv_blocks_size_gb`; those fields
are telemetry. Safe mutation requires scheduler/block-pool state.

## Key Configuration

- `config/abc-bench_config.yaml`
  - `llm.*`: OpenAI-compatible vLLM endpoint.
  - `dataset.*`: ABC-Bench task selection.
  - `launch.*`: baseline launch wave behavior and thread-pool size.
  - `prediction.*`: optional predictor training.
  - `sidecar.*`: embedded sidecar and KV geometry.
  - `sidecar.admission_control.*`: dynamic admission and eviction.

- `config/vllm_config.yaml`
  - Ascend/vllm-ascend serving configuration.
  - Geometry must match the sidecar config.

Important admission-control knobs:

```yaml
sidecar:
  admission_control:
    enabled: true
    threshold_gb: 0.1
    initial_admit_interval_s: 2.0
    short_tool_call_threshold_s: 2.0
    predictor_model: null
    pin_endpoint: null
    offload_endpoint: null
    eviction_endpoint: null
    eviction_timeout_s: 2.0
```

`predictor_model: null` defaults to `prediction.save_model` in the wrapper.
`pin_endpoint: null` defaults to `<sidecar.vllm_url>/agent_kv_cache/pin`.
`offload_endpoint: null` defaults to
`<sidecar.vllm_url>/agent_kv_cache/offload`. `eviction_endpoint` is kept as
a backward-compatible alias for offload.

## Output Artifacts

Per agent under `results_root/<task_id>/`:

- `agent_*_trace.csv`
- `agent_*_trace.jsonl`
- `agent_*_summary.json`
- `agent_*_llm_metrics.json`
- `agent_*_prompt.txt`
- `agent_*_test_result.json`

Aggregate:

- `run_summary.json`
- `prediction_metrics.json`
- `sidecar.log` JSONL when sidecar is enabled

Sidecar records include an `admission` object when dynamic admission is enabled,
with KV capacity, averages, headroom, heap candidates, evictions, admissions,
and error/disabled reasons.

## Live Dashboard

When `sidecar.http_port` (or `--sidecar-http-port`) is set, the sidecar starts
an HTTP/SSE server alongside the tick loop and serves the dashboard at
`http://<host>:<port>/`. It renders, on a shared time axis:

- One Gantt row per agent showing phase boxes (reasoning, tool_call, waiting,
  evicted, done) extending in real time.
- A KV-cache-used-% line chart from `vllm.kv_cache_used_pct`.
- Vertical event markers for admission decisions: EVICT (red), ADMIT (green),
  READMIT (purple), and SAT (dashed yellow, when `w < 1`).

Two ways to view a finished run:

1. `python -m sidecar_http --replay path/to/sidecar.log [--speed 1.0]` — same
   server, but the publisher reads ticks from the log file at the chosen speed.
2. In-browser: open the dashboard, click *Open sidecar.log…* and pick a
   JSONL file. The browser parses it client-side; SSE is disconnected while in
   replay mode.

The JSON contract for `/state` and `/stream` is documented in
[dashboard/SCHEMA.md](dashboard/SCHEMA.md). Both endpoints emit the existing
sidecar tick record without transformation, so log replays and live runs use
the same render path.

## Test and Verification Commands

Syntax check:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc python3 -m py_compile \
  src/sidecar.py \
  src/run_abc_bench_instrumented.py \
  src/vllm_patches/apply_patches.py \
  tests/test_sidecar_admission.py
```

Sidecar admission tests without pytest:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc python3 - <<'PY'
import tests.test_sidecar_admission as t
for name in sorted(n for n in dir(t) if n.startswith("test_")):
    getattr(t, name)()
print("sidecar admission tests: ok")
PY
```

Wrapper syntax:

```bash
bash -n run_abc-bench.sh
```

## Design Assumptions

- OpenHands emits `ActionEvent` before the matching `ObservationEvent`.
- Per-agent traces use SDK timestamps when available and UTC wall clock as a
  fallback.
- Each agent runs in its own thread; live state and callbacks require locks.
- Dynamic admission is opt-in to preserve baseline behavior.
- Per-agent KV telemetry is approximate from the sidecar's perspective; exact
  block ownership lives inside vLLM.
- Evicted agents finish their current tool call, then block before the next LLM
  call until re-admitted.
