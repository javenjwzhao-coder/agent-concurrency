# Tool Trace Collector

Component: `src/collect_tool_trace.py`

The tool trace collector turns OpenHands action/observation event pairs into
feature-engineering-friendly rows. It deliberately does not import OpenHands.
Instead, it duck-types event objects so tests can use lightweight fakes and the
collector can tolerate SDK attribute changes.

## Responsibilities

- Record action-side metadata when a tool call starts.
- Match the eventual observation or rejection back to the pending action.
- Snapshot live concurrency and KV state at dispatch and completion.
- Capture terminal commands, structured arguments, outcome details, output
  previews, byte counts, line counts, return codes, timeout flags, and hashes.
- Finalize unfinished tool calls so failed runs still produce partial traces.
- Write CSV and JSONL artifacts with the same logical records.

## Data Model

### `PendingToolCall`

Created by `record_action()`. It holds metadata available at dispatch time:

- action id and tool call id
- tool name
- extracted terminal command
- raw and JSON-encoded arguments
- argument keys and payload size
- action summary and action type
- conversation id
- start timestamp
- start live-state snapshot

### `ToolCallRecord`

Created by `record_observation()` or `finalize_unfinished()`. It is the durable
row written to CSV and JSONL.

Fields are defined by `TOOL_CALL_CSV_FIELDS`; the most important groups are:

| Field group | Examples | Purpose |
| --- | --- | --- |
| Identity | `agent_id`, `task_id`, `conversation_id`, `tool_seq` | Align rows to agents and tasks. |
| Tool metadata | `tool_name`, `tool_command`, `tool_args_json`, `tool_args_keys` | Predict duration and audit behavior. |
| Timing | `start_ts`, `end_ts`, `duration_s` | Primary target for predictor training. |
| Outcome | `outcome`, `error_type`, `returncode`, `timed_out` | Identify failed or rejected tool calls. |
| Output summary | `observation_bytes`, `observation_line_count`, `observation_preview`, `observation_sha256` | Preserve useful output signals without storing unlimited text. |
| Dispatch snapshot | `start_active_agents`, `start_active_tool_calls`, `start_kv_blocks`, `start_kv_gb` | Features known when the tool call starts. |
| Completion snapshot | `end_active_agents`, `end_active_tool_calls`, `end_kv_blocks`, `end_kv_gb` | Post-call diagnostics. |

The collector writes `duration_s` as a non-negative rounded float. Observation
content is hashed with SHA-256 when present.

## Matching Rules

`record_action()` stores each pending call under the best available key:

1. `action:<action_id>`
2. `tool_call:<tool_call_id>`
3. `seq:<seq>:<pending_count>`

`pop_pending()` resolves observations in this order:

1. Exact `action_id` key.
2. Scan for a pending object with the same `action_id`.
3. Exact `tool_call_id` key.
4. Scan for a pending object with the same `tool_call_id`.
5. If exactly one pending action exists, use it as a fallback.
6. Otherwise return `None`.

This protects normal cases while still salvaging traces when SDK events omit
one identifier.

## Extraction Rules

The collector tries several common attributes before giving up.

### Tool Arguments

`extract_tool_args()` checks event attributes:

- `arguments`
- `args`
- `tool_input`
- `input`
- `kwargs`

It accepts dictionaries directly, dataclasses, Pydantic models, objects with
`.dict()`, and JSON strings. It also repeats the same checks on `event.action`.
Unparseable scalar values are stored as `{"raw": "<value>"}`.

### Terminal Commands

`extract_terminal_command()` only returns a command when the tool name looks
like a shell tool: `terminal`, `bash`, or `shell`. It checks argument keys:

- `command`
- `cmd`
- `shell_command`
- `input`

### Observation Content

`extract_observation_content()` checks:

- `content`
- `output`
- `result`
- `text`
- `observation`
- `data`
- `stdout`
- `stderr`
- `message`

Bytes are decoded as UTF-8 with replacement. Dictionaries and lists are encoded
with JSON for stable text.

### Status Details

`extract_returncode()`, `extract_timed_out()`, and `extract_error_type()` check
both direct attributes and a dictionary-valued `data` attribute.

## Live Snapshot Integration

The runner passes `AgentPhaseTracker._live_tool_snapshot()` as
`live_snapshot_fn`. The collector calls it at action start and observation end.

If the snapshot callable raises, the collector swallows the exception and uses
an empty snapshot. Tracing must never break agent execution.

Expected snapshot keys:

- `active_agents`
- `active_tool_calls`
- `agent_state`
- `state_since`
- `cumulative_reasoning_s`
- `kv_blocks`
- `kv_gb`

## Artifacts

`write_csv(out_dir, filename)` writes a CSV with the fixed
`TOOL_CALL_CSV_FIELDS` header.

`write_jsonl(out_dir, filename)` writes one JSON object per record using the
same dataclass fields.

`write_artifacts(out_dir)` writes the default names:

- `<agent_id>_tool_calls.csv`
- `<agent_id>_tool_calls.jsonl`

The current runner uses explicit filenames:

- `<agent_id>_trace.csv`
- `<agent_id>_trace.jsonl`

The predictor supports both current trace names and transitional
`*_tool_calls.*` names.

## Invariants

- The collector must be dependency-light and not import OpenHands.
- Event parsing is best-effort; missing fields become empty strings or zeros.
- Output previews are bounded by `preview_chars`.
- Failed or interrupted runs should still emit useful records through
  `finalize_unfinished()`.
- The CSV schema is stable because the predictor and downstream notebooks read
  it directly.

## Change Guidelines

- Add new fields to `TOOL_CALL_CSV_FIELDS`, `ToolCallRecord`, and
  `_make_record()` together.
- Keep dispatch-time features separate from completion-time diagnostics.
  The predictor should only use features known when the tool call starts.
- If new fields are intended for prediction, update
  `docs/tool_duration_predictor.md` and `src/build_tool_predictor.py`.
