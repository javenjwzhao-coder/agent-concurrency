# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Platform

This project runs on **Ascend NPU** hardware. The inference backend is **vllm-ascend** (not upstream vLLM). All vLLM patches, config, and references to vLLM behaviour apply to the vllm-ascend fork. When investigating vLLM internals or applying patches, always target the vllm-ascend codebase.

## Project Purpose

This is an **ABC-Bench agent tracing and tool-duration prediction framework**. It runs concurrent AI agents against the [ABC-Bench](https://github.com/javenjwzhao-coder/agent-concurrency) coding benchmark, records one rich trace row per tool call, then optionally trains ML models to predict tool execution duration.

## Dependencies

There is no `requirements.txt`. Install manually:

```bash
pip install openhands-sdk openhands-tools pyyaml pydantic scikit-learn numpy pandas joblib
```

## Common Commands

**Run agents (direct CLI):**
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

**Run agents (via YAML config wrapper — preferred):**
```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml
bash run_abc-bench.sh --config config/abc-bench_config.yaml --dry-run
bash run_abc-bench.sh --config config/abc-bench_config.yaml --override dataset.max_tasks=2
```

**Train tool-duration predictor:**
```bash
python src/build_tool_predictor.py --trace-dir ./abc_results --model hgb --remaining --save-model ./duration_model.joblib
# model choices: hgb (recommended) | ridge (linear, fast) | rf (random forest)
```

**Run inference on new traces:**
```bash
python src/build_tool_predictor.py \
    --trace-dir ./new_traces \
    --load-model ./duration_model.joblib \
    --predict-only \
    --output-predictions ./predictions.csv
```

**Start vLLM backend (Ascend NPU):**
```bash
bash start_vllm.sh
```

## Architecture

### Execution Pipeline

```
ABC-Bench Tasks
  → parse CLI args / YAML config (run_abc-bench.sh expands env vars)
  → plan_launch_waves()     # randomize task grouping + staggered wave delays
  → ThreadPoolExecutor      # max_concurrent workers
      → _agent_thread()     # one thread per task
          → copy task to isolated workspace
          → run_single_agent()
              → build prompt from task.yaml
              → create LLM handle (vLLM OpenAI-compatible API)
              → create Agent with tools (Terminal, FileEditor, TaskTracker)
              → AgentPhaseTracker attached via event callbacks
              → conversation.send_message() + .run()
              → write *_trace.csv, *_summary.json, *_llm_metrics.json
          → maybe_run_tests()   # ./run-tests.sh if --run-tests
  → aggregate → run_summary.json
  → (optional) build_tool_predictor.py
```

### AgentPhaseTracker / ToolCallTraceCollector (core mechanism)

The runner still detects phase boundaries internally for live sidecar state and
summaries, but `*_trace.csv` is now a detailed tool-call trace, not a phase
trace. `ToolCallTraceCollector` correlates `ActionEvent` to `ObservationEvent`
and writes one row per tool call.

| Phase | Opens on | Closes on |
|-------|----------|-----------|
| `waiting` | `run_start()` or `MessageEvent(source=agent)` | next event or `run_end()` |
| `reasoning` | `run_start()` or `ObservationEvent` | `ActionEvent` |
| `tool_call` | `ActionEvent` | `ObservationEvent` |

Each row in the output `*_trace.csv` includes tool ids, tool name, command,
serialized arguments, start/end timestamps, duration, outcome, output
size/preview/hash, and dispatch-time concurrency/KV-cache snapshots. A matching
`*_trace.jsonl` is written for lossless JSONL consumption.

### Prediction Model (`build_tool_predictor.py`)

Loads all detailed `*_trace.csv` files, engineers dispatch-time features (tool
name, command length/tokens/pipes/keywords, payload bytes, text hashes,
concurrency/KV snapshots, cumulative reasoning, prior-tool stats), trains a
weighted regression model, and saves model + per-call predictions.

## Key Configuration Files

- **`config/abc-bench_config.yaml`** — LLM endpoint, dataset path/glob, launch strategy (wave sizing, delays, seed), and prediction settings. LLM API key read from `$LLM_API_KEY`.
- **`config/vllm_config.yaml`** — Ascend NPU-specific vLLM bare-metal config (model, memory, tensor parallelism, NUMA topology).

## Output Artifacts

Per agent run (under `results_root/<task_id>/`):
- `agent_*_trace.csv` — detailed per-tool-call trace
- `agent_*_trace.jsonl` — JSONL companion for the same tool-call records
- `agent_*_summary.json` — phase time breakdown and tool-call stats
- `agent_*_llm_metrics.json` — token usage and inference latency
- `agent_*_prompt.txt`, `agent_*_test_result.json` — task prompt and test results

Aggregate:
- `run_summary.json` — all agent summaries combined
- `prediction_metrics.json` — MAE / RMSE / R² (if prediction was enabled)

## Design Assumptions

- **OpenHands SDK event ordering**: `ActionEvent` always precedes its corresponding `ObservationEvent`.
- **Timestamps**: prefer `event.timestamp` from the SDK; fall back to `datetime.now()`. No clock-skew correction.
- **Thread safety**: each agent runs in its own thread; `AgentPhaseTracker` uses locks for callbacks.
- **Per-agent KV-cache tracking**: `agent_id` passed via `litellm_extra_body`; requires vLLM support but is optional.
- **Prediction scope**: models predict individual tool-call duration only — not end-to-end task time.
