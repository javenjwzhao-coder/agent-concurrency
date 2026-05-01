# CLAUDE.md

This file guides Claude Code and other coding agents working in this
repository.

## Platform

This project runs on **Ascend NPU** hardware. The inference backend is
**vllm-ascend**, not plain upstream vLLM. `start_vllm.sh` installs
vLLM/vllm-ascend 0.13.0 into the project `.venv` and reuses that install on
later starts. It installs the compatible local Ascend runtime stack
(`torch==2.8.0`, `torch-npu==2.8.0.post2`, `triton-ascend==3.2.0`) before
following the official source-install flow: it clones vLLM v0.13.0 under
`.vllm-src/`, installs it with `VLLM_TARGET_DEVICE=empty`, then clones
vllm-ascend v0.13.0 under `.vllm-ascend-src/`, initializes submodules, and
installs it editable for `SOC_VERSION=ascend910_9391` by default. It installs
the CANN Python build dependencies (`sympy`, `numpy<2.0.0`, and related
utilities) into `.venv` and exposes that site-packages path to the custom-op
compiler during the source build. Custom-op compilation defaults to
`MAX_JOBS=16`; override `VLLM_ASCEND_MAX_JOBS` if the build host needs a
different cap. The script also patches the cached source checkout so CANN 8.5.1
host object files installed under `objects-*` are copied to the top-level
directory expected by `recompile_binary.py`. Runtime dependencies are installed
under constraints that keep that Ascend stack pinned while filtering CUDA
packages and vLLM's upstream torch pins. When
investigating internals or applying patches, target the installed
vllm-ascend/vLLM package and keep version-specific anchors in mind.

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
6. Proactively offload idle tool-call agents when KV pressure rises.

The goal is to maximize useful agent concurrency without exhausting KV cache on
vllm-ascend.

## Mental Model

Agent workloads have bursty KV growth:

- During `reasoning`, an agent is actively using vLLM and grows KV state.
- During `tool_call`, the agent is idle from the model's perspective, but its
  KV blocks may remain resident.
- Fresh agents should be admitted only when recent average KV growth suggests
  there is safe headroom.
- Idle tool-call agents are the best offload candidates because their current
  tool can keep running while their next model call is deferred.

The sidecar implements this policy. The predictor helps rank idle agents by:

```text
offload_score = kv_usage_gb * predicted_remaining_tool_seconds
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

Patch vllm-ascend/vLLM for per-agent KV telemetry and offload:

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
`ObservationEvent`, an offloaded agent blocks before re-entering `reasoning`
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

`headroom_low` means `w <= w_threshold`. `saturation_guard` is emitted only
when `w <= w_threshold` and there is runnable queued work (`fresh` or
`offloaded_ready`) that cannot be admitted. `w_threshold` defaults to `2.0`.

Policy order:

1. If `w <= w_threshold`, admit no new fresh agents.
2. If `max_active_agents` is positive and the number of active admitted agents
   (`reasoning`, `tool_call`, or `waiting`) has reached that cap, admit no
   fresh or previously offloaded agents.
3. Before the first real SAT, admit at most one fresh queued task per
   `initial_admit_interval_s`. After first SAT, fresh admissions use normal
   sidecar capacity math. READMITs bypass this launch ramp.
4. At tool-call start, predict remaining duration. Predicted-short calls
   (`< short_tool_call_threshold_s`) stay resident and are excluded from
   pressure offload. Predicted-long calls remain eligible for the heap. If no
   predictor is available yet, tool calls older than
   `fallback_long_tool_call_s` are treated as long idle candidates.
5. If free KV percent is at or below `threshold_percent`, offload
   highest-scoring eligible long idle `tool_call` agents through the
   agent-aware OffloadingConnector.
   Finished agent requests are held inside vLLM first; short or completed
   non-offloaded tool calls release the hold through `/agent_kv_cache/release`.
6. If `w > w_threshold` and the active-agent cap has room, admit from the waiting
   queue. The percent threshold only controls pressure offload.

Successful offload records report freed memory from the exact vLLM free-block
delta (`free_blocks_after - free_blocks_before`) or an explicit endpoint value.
The sidecar does not replace missing `freed_gb` with candidate KV estimates.
Async connector offloads report `pending_async` until that exact delta is visible.

The waiting queue has two lanes: previously offloaded agents first, then fresh
agents FIFO.

### vllm-ascend Patch

`src/vllm_patches/apply_patches.py` patches vLLM/vllm-ascend to support both
telemetry and control.

Telemetry:

- Adds `agent_id` to chat requests.
- Adds `kv_blocks_used` and `kv_blocks_size_gb` to usage responses.

Control:

- Adds `POST /agent_kv_cache/offload` and `POST /agent_kv_cache/restore`.
- Adds `POST /agent_kv_cache/release` for releasing held KV without offload.
- Installs `AgentAwareOffloadingConnector` under vLLM's KV connector package.
- Reuses vLLM's `OffloadingConnector` async worker and the configured
  vllm-ascend `NPUOffloadingSpec`.
- Tracks `agent_id -> request_id -> block_ids` and held finished requests in
  the connector scheduler.

Do not try to offload from sidecar using only `kv_blocks_size_gb`; those fields
are telemetry. Safe mutation requires connector/scheduler block ownership.

## Key Configuration

- `config/abc-bench_config.yaml`
  - `llm.*`: OpenAI-compatible vLLM endpoint.
  - `dataset.*`: ABC-Bench task selection.
  - `launch.*`: baseline launch wave behavior and thread-pool size.
  - `prediction.*`: optional predictor training.
  - `sidecar.*`: embedded sidecar and KV geometry.
  - `sidecar.admission_control.*`: dynamic admission and offload.

- `config/vllm_config.yaml`
  - Ascend/vllm-ascend serving configuration.
  - Geometry must match the sidecar config.

Important admission-control knobs:

```yaml
sidecar:
  admission_control:
    enabled: true
    threshold_percent: 10.0
    w_threshold: 2.0
    initial_admit_interval_s: 1.0
    max_active_agents: 32
    fallback_long_tool_call_s: 5.0
    offload_timeout_s: 10.0
    exact_freed_gb_timeout_s: 5.0
```

Omitted admission values use wrapper defaults. `predictor_model` defaults to
`prediction.save_model`, offload/restore/release endpoints default under
`sidecar.vllm_url`, and `sidecar.vllm_url` is derived from `llm.base_url` when
it is not set explicitly. Accepted async offloads may report
`freed_gb_source: pending_async` until vLLM reports the exact free-block delta.

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
with KV capacity, averages, headroom, heap candidates, offloads, admissions,
and error/disabled reasons.

## Live Dashboard

When `sidecar.http_port` (or `--sidecar-http-port`) is set, the sidecar starts
an HTTP/SSE server alongside the tick loop and serves the dashboard at
`http://<host>:<port>/`. It renders, on a shared time axis:

- One Gantt row per agent showing phase boxes (reasoning, tool_call, waiting,
  offloaded, done) extending in real time.
- A KV-cache-used-% line chart from `vllm.kv_cache_used_pct`.
- Vertical event markers for admission decisions: OFFLOAD (red), ADMIT (green),
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
- Offloaded agents finish their current tool call, then block before the next LLM
  call until re-admitted.
