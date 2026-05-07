# Proactive Concurrency Management for Agentic AI Workloads

This repository runs many ABC-Bench coding agents against an OpenAI-compatible
vllm-ascend backend while keeping Ascend NPU KV cache pressure under control.

The main idea is growth-aware concurrency. Agents grow KV cache while reasoning,
then often sit in tool calls while their KV blocks remain resident. The sidecar
uses live agent state, vLLM KV metrics, and an optional remaining-tool-time
predictor to decide when to admit queued agents and when to offload idle tool
calls.

For the deeper control-loop and connector design, see
[docs/agent_kv_offload_architecture.md](docs/agent_kv_offload_architecture.md).
Detailed component docs live in [docs/README.md](docs/README.md).

## Repository Map

| Path | Purpose |
| --- | --- |
| `src/run_abc_bench_instrumented.py` | Runs ABC-Bench tasks through OpenHands, records tool traces, publishes live agent state, and embeds the sidecar when enabled. |
| `src/sidecar.py` | Polls vLLM metrics, reads live per-agent KV usage, applies admission/offload policy, and writes JSONL ticks. |
| `src/build_tool_predictor.py` | Trains or loads a tool duration / remaining-time predictor from trace CSVs. |
| `src/collect_tool_trace.py` | Duck-typed ActionEvent -> ObservationEvent trace collector used by the runner. |
| `src/sidecar_http.py` | Small stdlib HTTP/SSE server for live and replay dashboard data. |
| `src/vllm_patches/` | vLLM/vllm-ascend 0.13 patcher plus the agent-aware offloading connector. |
| `dashboard/` | Browser dashboard for sidecar logs, live SSE, and standalone snapshots. |
| `config/` | Example ABC-Bench and vLLM configs. |
| `run_abc-bench.sh` | YAML-driven wrapper for runner, sidecar, dashboard, and optional predictor build. |
| `start_vllm.sh` | Ascend/vllm-ascend bootstrap and launch script. |

## Setup

There is no locked requirements file. Install the pieces you need for the path
you are running:

```bash
pip install openhands-sdk openhands-tools pyyaml pydantic requests
pip install scikit-learn numpy pandas joblib
```

`start_vllm.sh` manages the vLLM/vllm-ascend environment separately because the
Ascend runtime stack has strict version constraints.

## Quick Start

Start or verify the Ascend vLLM backend:

```bash
bash start_vllm.sh config/vllm_config.yaml
```

Run ABC-Bench with the YAML wrapper:

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml
```

Use a dry run to inspect the resolved command:

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml --dry-run
```

Override one config value without editing YAML:

```bash
bash run_abc-bench.sh \
  --config config/abc-bench_config.yaml \
  --override dataset.max_tasks=2
```

Run the Python runner directly when you do not need the wrapper:

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

Train a remaining-time predictor from collected traces:

```bash
python src/build_tool_predictor.py \
  --trace-dir ./abc_results \
  --model hgb \
  --remaining \
  --save-model ./duration_model.joblib
```

## vLLM Patch

The sidecar needs vLLM support for per-agent telemetry and safe KV mutation.
Apply the patch on the NPU machine:

```bash
python src/vllm_patches/apply_patches.py \
  --vllm-dir /path/to/site-packages/vllm
```

The patch adds:

- `agent_id` on chat requests.
- `kv_blocks_used` and `kv_blocks_size_gb` on usage telemetry.
- `POST /agent_kv_cache/offload`, `/release`, and `/restore`.
- `AgentAwareOffloadingConnector`, which holds finished agent requests before
  their KV blocks enter vLLM's free queue.

## Admission Policy

At each sidecar tick:

- `C` is free KV capacity in GB.
- `s_t` is average active-agent KV GB for this tick.
- `s_prev` is the previous active-agent average.
- `w = C / min(s_t, s_prev)` is conservative headroom, using whichever samples
  are known.

The policy is intentionally simple:

- Pressure offload runs when free KV percent is at or below
  `threshold_percent`, including when the tick only has cache-used percent
  telemetry.
- New work admits only when effective `w > w_threshold`.
- Previously offloaded agents readmit before fresh agents.
- `max_active_agents` can cap active admitted agents.
- Short predicted tool calls release held KV and are excluded from pressure
  offload unless they later exceed the fallback-long age while still resident.
- Long tool calls are scored by:

```text
offload_score = agent_kv_usage_gb * predicted_remaining_tool_seconds
```

Fallback-long calls use elapsed tool-call seconds as the multiplier.

Exact freed memory is reported only from vLLM free-block deltas or an explicit
endpoint value. Async connector offloads may report `pending_async` until that
exact delta appears.

## Dashboard

Set `sidecar.http_port` in `config/abc-bench_config.yaml` or pass
`--sidecar-http-port` to serve the dashboard:

```text
http://127.0.0.1:8765/
```

The dashboard can also open a finished `sidecar.log` in the browser and export a
standalone HTML snapshot. The live/replay JSON contract is documented in
[dashboard/SCHEMA.md](dashboard/SCHEMA.md).

## Output

Per agent under `results_root/<task_id>/`:

- `agent_*_trace.csv`
- `agent_*_trace.jsonl`
- `agent_*_events.log`
- `agent_*_summary.json`
- `agent_*_llm_metrics.json`
- `agent_*_prompt.txt`
- `agent_*_test_result.json`, when tests run

Aggregate outputs include:

- `run_summary.json`
- `run_metadata.json`
- `prediction_metrics.json`
- `sidecar.log`, when the sidecar is enabled

## Useful Checks

Syntax:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc python3 -m compileall -q src tests
```

Shell scripts:

```bash
bash -n run_abc-bench.sh
bash -n start_vllm.sh
```

Focused tests can be run as direct Python scripts when `pytest` is unavailable:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc python3 tests/test_sidecar_http.py
```
