# ABC-Bench Agent Tracing & Tool Duration Prediction

## File Structure

```
src/run_abc_bench_instrumented.py    # Step 1: multi-agent runner with tool-call tracing
src/collect_tool_trace.py            # Detailed ActionEvent -> ObservationEvent collector
src/build_tool_predictor.py          # Step 2: regression model for tool-call duration
```

## CSV Trace Schema

Each agent produces `<agent_id>_trace.csv` with one row per tool call:

| Column              | Type   | Description |
|---------------------|--------|-------------|
| `agent_id`          | str    | Unique agent identifier (e.g. `agent_task_042_w0`) |
| `task_id`           | str    | ABC-Bench task directory name |
| `tool_seq`          | int    | 0-based index of this tool call within the agent run |
| `action_id`         | str    | OpenHands action id, when available |
| `tool_call_id`      | str    | Tool-call id, when available |
| `start_ts`          | str    | ISO 8601 UTC wall-clock start (matches vLLM format) |
| `end_ts`            | str    | ISO 8601 UTC wall-clock end |
| `duration_s`        | float  | Tool wall-clock seconds |
| `tool_name`         | str    | `terminal` · `file_editor` · `task_tracker` |
| `tool_command`      | str    | Shell command string (terminal only) |
| `tool_args_json`    | str    | JSON-serialised tool arguments |
| `tool_payload_bytes`| int    | Byte length of `tool_args_json` |
| `outcome`           | str    | `ok` · `rejected` · `error` · `unfinished` |
| `observation_*`     | mixed  | Output size, line count, preview, and SHA-256 hash |
| `start_*` / `end_*` | mixed  | Live concurrency and KV-cache snapshots |
| `detail`            | str    | Human-readable action description |
| `conversation_id`   | str    | OpenHands conversation ID |

The runner also writes `<agent_id>_trace.jsonl` with the same rich records in
JSONL form. Phase transitions are tracked internally for summaries and sidecar
state, but phase rows are no longer written to `*_trace.csv`.

Timestamps use ISO 8601 with explicit `+00:00` UTC offset, matching vLLM's
internal timestamp format for cross-alignment.

## Quick Start

### 1. Run ABC-Bench and collect tool traces

```bash
# Preferred: config wrapper. The runner always writes per-tool-call traces.
bash run_abc-bench.sh --config config/abc-bench_config.yaml

# Or direct runner:
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
```

### 2. Train and save the predictor

```bash
# Default recommendation: HGB + log target + long-call weighting.
python src/build_tool_predictor.py \
    --trace-dir ./abc_results \
    --model hgb \
    --remaining \
    --save-model ./duration_model.joblib
```

### 3. Predict on new traces

```bash
python src/build_tool_predictor.py \
    --trace-dir ./new_traces \
    --load-model ./duration_model.joblib \
    --predict-only \
    --output-predictions ./predictions.csv
```

### 4. One-command run plus model build

Set `prediction.enabled: true` and `prediction.save_model` in
`config/abc-bench_config.yaml`, then run:

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml
```

## Prediction Model Features

The model uses these features extracted from each `tool_call` row:

**Categorical** (one-hot encoded): `tool_name`

**Numeric — command structure:**
- `command_length` — character count of the terminal command
- `command_token_count` — word count
- `command_pipe_count` — number of `|` operators
- `command_semicolon_count` — number of `;` operators
- `tool_payload_bytes` — byte size of serialised arguments

**Binary — command keywords:**
- `command_has_docker`, `command_has_build`, `command_has_test`,
  `command_has_install`, `command_has_git`

**Sequential context:**
- `phase_seq` — normalized tool-call sequence index (`tool_seq` in trace CSVs)
- `cumulative_reasoning_s` — total reasoning time before this call
- `prior_tool_calls` — count of preceding tool calls
- `prior_avg_tool_s` — running mean of prior tool-call durations

## Assumptions

1. **OpenHands SDK event model:** The tracker assumes `ActionEvent` always precedes its matching `ObservationEvent`, and that `event.id` or `tool_call_id` can be used to correlate them.

2. **Timestamp source:** Tool-call timestamps come from the SDK's `event.timestamp` field when available, falling back to `datetime.now(utc)`. Small clock-skew between the SDK and vLLM is expected but not corrected.

3. **Thread safety:** Each agent runs in its own thread with its own `AgentPhaseTracker`. The tracker uses a lock for event callbacks since the SDK may deliver events from a background thread.

4. **agent_id propagation:** The `agent_id` is passed to vLLM via `litellm_extra_body`. A patched vLLM can use this for per-agent KV-cache tracking, but the tracing script works without the vLLM patch.

5. **Prediction model scope:** The model predicts wall-clock duration of individual tool calls, not end-to-end agent runtime. Terminal commands have inherently high variance (docker build vs. ls), so expect higher error on terminal calls than on file_editor.

6. **Cross-agent features:** The trace captures start/end active-agent counts and KV-cache snapshots. The predictor uses only dispatch-time fields to avoid post-call leakage.
