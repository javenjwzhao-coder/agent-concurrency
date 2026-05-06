# Tool Duration Predictor

Component: `src/build_tool_predictor.py`

The predictor trains scikit-learn regression models from tool-call trace CSVs.
It can predict total tool duration or remaining tool-call time. Remaining-time
mode is used by the sidecar to decide whether an idle tool call is worth KV
offloading.

## Responsibilities

- Load current and transitional trace artifacts recursively.
- Normalize trace rows into a common tool-call schema.
- Build dispatch-time feature matrices.
- Add per-agent sequential context without leaking future calls.
- Train and evaluate regression pipelines.
- Optionally train quantile models.
- Save fitted models with enough metadata for real-time inference.
- Provide `RealtimePredictor` for sidecar use.

## Inputs

Primary input is a directory containing `*_trace.csv` files from the runner.
The loader searches recursively:

```bash
python src/build_tool_predictor.py --trace-dir ./abc_results
```

Supported trace shapes:

- Current detailed runner traces: `*_trace.csv`
- Transitional detailed artifacts: `*_tool_calls.csv`
- Transitional detailed JSONL artifacts: `*_tool_calls.jsonl`
- Legacy phase traces, when they contain `phase == "tool_call"` rows

## Dataset Preparation

`prepare_dataset()` is the main data path:

1. `load_traces()` reads every `*_trace.csv`.
2. `load_tool_calls()` chooses detailed rows and merges transitional artifacts
   only when needed.
3. `build_features()` creates raw feature columns from dispatch-time data.
4. `enrich_sequential_features()` adds prior-call context per agent.
5. Remaining-time mode optionally expands each call into elapsed snapshots.
6. `tool_name` is one-hot encoded.
7. Targets are log-transformed unless `--no-log-target` is set.

Calls whose original duration is below `--long-threshold` get a target of
`0.0`. Predictions below the same threshold are clamped to zero.

## Feature Groups

The model intentionally uses features known when a tool call starts.

| Group | Examples |
| --- | --- |
| Tool identity | `tool_name`, `is_terminal`, `is_file_editor`, `is_task_tracker` |
| Payload size | `tool_payload_bytes`, `args_key_count`, `args_total_str_bytes` |
| Command shape | command length, token count, pipe count, semicolon count, redirect count |
| Command keywords | docker, build, test, install, git, network, python |
| File editor details | sub-command, path extension, edit size, view range size |
| Terminal details | timeout seconds |
| Task tracker details | add/list/complete sub-commands |
| Text hash | 64 char n-gram hash features over summary, detail, command, sub-command, and path |
| Concurrency | active agents and active tool calls at dispatch |
| Sequential context | prior tool count, prior average/max/std duration, cumulative reasoning time |
| Remaining mode | `elapsed_s` |

Completion-only fields, such as observation output size, are not used as
predictor features because they are not known when the sidecar must decide.

## Remaining-Time Mode

`--remaining` trains on synthetic elapsed-time snapshots. Each completed tool
call is expanded at these fractions:

```text
0.0, 0.10, 0.25, 0.50, 0.75
```

For a call with duration `D` and fraction `f`:

```text
elapsed_s = D * f
remaining_s = max(0, D - elapsed_s)
```

Short calls below `--long-threshold` are assigned `remaining_s = 0` for every
snapshot. This makes the deployed predictor naturally distinguish calls that
should not be offloaded.

## Model Types

Use `--model` to choose:

| Model | Pipeline | Notes |
| --- | --- | --- |
| `hgb` | `HistGradientBoostingRegressor` | Default. Handles mixed scales and NaN without scaling. |
| `rf` | `StandardScaler` + `RandomForestRegressor` | Useful nonlinear baseline. |
| `ridge` | `StandardScaler` + `Ridge` | Fast linear baseline. |

Long calls receive higher training weight through `build_sample_weights()`.
The effective long-call weight is:

```text
max(LONG_CALL_WEIGHT, n_short / n_long)
```

When using HGB, `min_samples_leaf` is scaled from the number of long calls so
small datasets are not forced into overly large leaves.

## Splitting and Evaluation

`group_train_test_indices()` keeps agents pure across train/test splits when
more than one agent is available. This prevents a single agent's repeated tool
pattern from leaking into both sets.

When long-call labels exist, the split is stratified so the test set includes
at least one long-call agent where possible. Single-agent datasets fall back to
a row split, optionally stratified by long/short rows.

Metrics include:

- MAE
- RMSE
- Median absolute error
- MAPE over non-zero targets
- R2
- long-call MAE/RMSE
- short-call zero-rate
- per-tool breakdown when enough samples exist

Metrics are printed and written to:

```text
<trace-dir>/prediction_metrics.json
```

## Saved Models

Use:

```bash
python src/build_tool_predictor.py \
  --trace-dir ./abc_results \
  --model hgb \
  --remaining \
  --save-model ./duration_model.joblib
```

The saved pipeline is annotated with:

- `_trained_feature_names`
- `_log_target`
- `_remaining_mode`
- `_long_call_threshold_s`
- `_effective_long_call_weight`
- `_effective_hgb_min_samples_leaf`

These attributes are required for reliable inference alignment.

## Prediction-Only Mode

Use:

```bash
python src/build_tool_predictor.py \
  --trace-dir ./new_results \
  --load-model ./duration_model.joblib \
  --predict-only \
  --output-predictions ./predictions.csv
```

If the saved model has `_trained_feature_names`, missing columns are added as
zeros and the matrix is reordered to match training.

## Quantile Models

`--quantiles` trains P10, P50, and P90 models with
`GradientBoostingRegressor(loss="quantile")`. Quantile predictions are only
included in `--output-predictions`; the saved point-estimate model remains the
primary sidecar artifact.

## Real-Time Predictor

`RealtimePredictor.load(path)` loads a saved pipeline and exposes:

- `predict_remaining(feat, elapsed_s=0.0)`
- `predict_duration(feat)`

The sidecar uses this wrapper in two ways:

1. If the predictor object has `predict_agent_remaining()`, the sidecar calls
   that directly. This supports custom predictors in tests or future code.
2. Otherwise the sidecar builds a one-row DataFrame from live tool metadata,
   calls `build_features(..., remaining_mode=True)`, and asks
   `RealtimePredictor.predict_remaining()`.

`RealtimePredictor` handles one-hot column alignment and clamps low predictions
to zero using the saved long-call threshold.

## Sidecar Integration

The wrapper script can wire predictor training and sidecar usage together:

- `prediction.save_model` controls where training writes the joblib model.
- If `sidecar.admission_control.predictor_model` is omitted, the wrapper uses
  `prediction.save_model` as the admission predictor path.
- The sidecar's `short_tool_call_threshold_s` controls runtime short-call
  classification.
- The predictor's `long_threshold` controls training labels and saved clamp
  behavior.

Keep those thresholds intentionally aligned unless testing a policy variant.

## Outputs

| Output | When |
| --- | --- |
| `prediction_metrics.json` | Always after training or prediction-only evaluation. |
| joblib model | When `--save-model` is set. |
| predictions CSV | When `--output-predictions` is set. |

## Change Guidelines

- Keep feature generation deterministic and stateless where possible.
- Do not add completion-only values as sidecar prediction features.
- If a feature depends on prior agent history, compute it in
  `enrich_sequential_features()` so it can preserve ordering rules.
- When adding new categorical features, make sure `RealtimePredictor` can align
  missing columns at inference.
- Update tests when changing split behavior; small trace datasets are easy to
  make accidentally degenerate.
