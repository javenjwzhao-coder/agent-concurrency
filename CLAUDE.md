# CLAUDE.md

Notes for coding agents working in this repository.

## Platform

This project targets Ascend NPU hardware through vllm-ascend, not plain upstream
vLLM. `start_vllm.sh` owns the Ascend-serving environment and installs or reuses
vLLM/vllm-ascend 0.13.0 under the project `.venv`.

Keep these constraints in mind:

- The Ascend stack is pinned around `torch==2.8.0`,
  `torch-npu==2.8.0.post2`, and `triton-ascend==3.2.0`.
- Source installs cache vLLM under `.vllm-src/` and vllm-ascend under
  `.vllm-ascend-src/`.
- The default Ascend SoC is `ascend910_9391`; override
  `VLLM_ASCEND_SOC_VERSION` only when the target hardware changes.
- Do not replace the vLLM patch flow with generic CUDA or upstream-vLLM
  assumptions.

## Purpose

The repository is a growth-aware concurrency framework for ABC-Bench agents.
It tries to maximize useful agent concurrency without exhausting vLLM KV cache.

Runtime flow:

```text
ABC-Bench tasks
  -> instrumented OpenHands runner
  -> live agent phase/KV state and tool traces
  -> optional remaining-tool-time predictor
  -> sidecar admission/offload controller
  -> patched vllm-ascend agent KV endpoints
  -> dashboard and JSONL telemetry
```

## Mental Model

Agents alternate between:

- `reasoning`: the model is active and KV cache grows.
- `tool_call`: the model is idle, but its KV blocks may still be resident.
- `waiting`: the runner is idle or blocked between model calls.

The sidecar admits queued agents when KV headroom is sufficient and offloads
long idle tool-call agents under pressure. The predictor only ranks tool-call
idle time; it does not predict whole task duration.

Core formula:

```text
offload_score = kv_usage_gb * predicted_remaining_tool_seconds
```

The safety invariant is important:

```text
The sidecar chooses whether to offload; vLLM decides when blocks are safe to free.
```

## Main Files

| Path | Notes |
| --- | --- |
| `src/run_abc_bench_instrumented.py` | Runner, OpenHands callbacks, live state, embedded sidecar startup. |
| `src/sidecar.py` | Dynamic admission controller, vLLM polling, offload/readmit/release calls. |
| `src/build_tool_predictor.py` | Feature engineering, training, saved-model inference wrapper. |
| `src/collect_tool_trace.py` | Tool-call trace collector. |
| `src/sidecar_http.py` | Dashboard HTTP/SSE feed and log replay mode. |
| `src/vllm_patches/apply_patches.py` | Version-specific vLLM/vllm-ascend patcher. |
| `src/vllm_patches/agent_offloading_connector.py` | Connector that holds, stores, releases, and restores agent KV. |
| `dashboard/` | Timeline, KV chart, event markers, and schema docs. |

## Common Commands

Run through YAML:

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml
bash run_abc-bench.sh --config config/abc-bench_config.yaml --dry-run
bash run_abc-bench.sh --config config/abc-bench_config.yaml --override dataset.max_tasks=2
```

Start vLLM on Ascend:

```bash
bash start_vllm.sh config/vllm_config.yaml
```

Patch vLLM/vllm-ascend:

```bash
python src/vllm_patches/apply_patches.py \
  --vllm-dir /path/to/site-packages/vllm
```

Train a predictor:

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

## Configuration

Primary user config lives in `config/abc-bench_config.yaml`.

Important sections:

- `llm.*`: OpenAI-compatible vLLM endpoint.
- `dataset.*`: ABC-Bench root, task glob, and task count.
- `paths.*`: workspaces and results.
- `prediction.*`: optional predictor training and saved model path.
- `sidecar.*`: sidecar enablement, dashboard port, and KV geometry.
- `sidecar.admission_control.*`: dynamic admission, pressure offload, and
  endpoint settings.

If `sidecar.vllm_url` is omitted, `run_abc-bench.sh` derives it from
`llm.base_url` by removing a trailing `/v1`.

## Admission Details

At each tick the controller computes:

```text
C = free KV capacity in GB
s_t = current average KV GB across active agents
s_prev = previous tick's active-agent average
w = C / min(s_t, s_prev)
```

Policy summary:

- `threshold_percent` triggers pressure offload.
- `w_threshold` gates new admissions.
- `max_active_agents` caps fresh admits and readmits when set above zero.
- Fresh agents are paced by `initial_admit_interval_s` before the first
  saturation event and by `max_fresh_admits_per_tick` afterward.
- READMITs use a priority lane ahead of fresh agents.
- Short tool calls release held KV and are not offload candidates.
- Long/fallback-long tool calls are eligible for the pressure heap.

Exact `freed_gb` must come from vLLM free-block deltas or an explicit endpoint
value. Do not substitute estimated agent KV size for freed memory.

## Development Notes

- Prefer local patterns over new abstractions.
- Keep vLLM patch anchors version-specific and easy to audit.
- Avoid broad refactors in the runner and sidecar; they are heavily threaded and
  coupled to runtime ordering.
- The dashboard consumes the sidecar tick record directly. Keep log, live SSE,
  replay, and standalone snapshot paths on the same data shape.
- Generated benchmark outputs should stay under ignored paths such as
  `abc_results/`, `.bench-venv/`, `.vllm-src/`, and `.vllm-ascend-src/`.

## Checks

Base environments may not include `pytest` or all benchmark dependencies. Run
the checks that fit the current machine:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc python3 -m compileall -q src tests
bash -n run_abc-bench.sh
bash -n start_vllm.sh
```

Direct Python test harnesses:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc python3 tests/test_sidecar_http.py
```
