# ABC-Bench Agent Tracing & Tool Duration Prediction

## File Structure

```
run_abc_bench_instrumented.py    # Step 1: multi-agent runner with phase tracing
predict_tool_duration.py         # Step 2: regression model for tool-call duration
generate_synthetic_traces.py     # Helper: create test data without a live backend
```

## CSV Trace Schema

Each agent produces `<agent_id>_trace.csv` with these columns:

| Column              | Type   | Description |
|---------------------|--------|-------------|
| `agent_id`          | str    | Unique agent identifier (e.g. `agent_task_042_w0`) |
| `task_id`           | str    | ABC-Bench task directory name |
| `phase_seq`         | int    | 0-based index of this phase within the agent run |
| `phase`             | str    | `reasoning` · `tool_call` · `waiting` |
| `start_ts`          | str    | ISO 8601 UTC wall-clock start (matches vLLM format) |
| `end_ts`            | str    | ISO 8601 UTC wall-clock end |
| `duration_s`        | float  | Wall-clock seconds |
| `tool_name`         | str    | `terminal` · `file_editor` · `task_tracker` (tool_call only) |
| `tool_command`      | str    | Shell command string (terminal only) |
| `tool_args_json`    | str    | JSON-serialised tool arguments |
| `tool_payload_bytes`| int    | Byte length of `tool_args_json` |
| `outcome`           | str    | `ok` · `rejected` · `error` · `tool_call` · `agent_message` · ... |
| `detail`            | str    | Human-readable description |
| `conversation_id`   | str    | OpenHands conversation ID |

### Phase boundary detection

- **reasoning** starts when the LLM begins processing (after the initial prompt or after receiving a tool result) and ends when the agent emits an `ActionEvent` (tool call) or a text message.
- **tool_call** starts at `ActionEvent` and ends at the matching `ObservationEvent`.
- **waiting** covers idle gaps: pre-launch delay, post-completion, and between conversation turns.

Timestamps use ISO 8601 with explicit `+00:00` UTC offset, matching vLLM's internal timestamp format for cross-alignment.

## Quick Start

### 1. Generate synthetic traces (no vLLM needed)

```bash
python generate_synthetic_traces.py \
    --out-dir ./synthetic_traces \
    --n-agents 20 \
    --seed 42
```

### 2. Train & evaluate the predictor

```bash
# Ridge regression (linear, fast, interpretable):
python predict_tool_duration.py \
    --trace-dir ./synthetic_traces \
    --model ridge

# Random Forest (better accuracy):
python predict_tool_duration.py \
    --trace-dir ./synthetic_traces \
    --model rf \
    --save-model ./duration_model.joblib
```

### 3. Predict on new traces

```bash
python predict_tool_duration.py \
    --trace-dir ./new_traces \
    --load-model ./duration_model.joblib \
    --predict-only \
    --output-predictions ./predictions.csv
```

### 4. Run real agents against a vLLM endpoint

```bash
# Sequential (one agent at a time):
python run_abc_bench_instrumented.py \
    --dataset-root /data/ABC-Bench \
    --task-glob 'task_*' \
    --max-tasks 3 \
    --model openai/Qwen3-30B-A3B-Instruct \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key dummy

# Randomised multi-agent launch:
python run_abc_bench_instrumented.py \
    --dataset-root /data/ABC-Bench \
    --task-glob 'task_*' \
    --max-tasks 6 \
    --randomise-launch \
    --max-agents-per-wave 4 \
    --min-wave-delay-s 2 \
    --max-wave-delay-s 15 \
    --launch-seed 42 \
    --run-tests \
    --results-root ./abc_results
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
- `phase_seq` — position in the agent's phase sequence
- `cumulative_reasoning_s` — total reasoning time before this call
- `prior_tool_calls` — count of preceding tool calls
- `prior_avg_tool_s` — running mean of prior tool-call durations

## Assumptions

1. **OpenHands SDK event model:** The tracker assumes `ActionEvent` always precedes its matching `ObservationEvent`, and that `event.id` or `tool_call_id` can be used to correlate them.

2. **Timestamp source:** Phase timestamps come from the SDK's `event.timestamp` field when available, falling back to `datetime.now(utc)`. Small clock-skew between the SDK and vLLM is expected but not corrected.

3. **Thread safety:** Each agent runs in its own thread with its own `AgentPhaseTracker`. The tracker uses a lock for event callbacks since the SDK may deliver events from a background thread.

4. **agent_id propagation:** The `agent_id` is passed to vLLM via `litellm_extra_body`. A patched vLLM can use this for per-agent KV-cache tracking, but the tracing script works without the vLLM patch.

5. **Prediction model scope:** The model predicts wall-clock duration of individual tool calls, not end-to-end agent runtime. Terminal commands have inherently high variance (docker build vs. ls), so expect higher error on terminal calls than on file_editor.

6. **No cross-agent features:** The current model treats each tool call independently. A production model could add features like concurrent agent count or current vLLM queue depth.