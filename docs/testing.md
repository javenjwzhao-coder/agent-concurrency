# Testing Guide

Components:

- `tests/test_sidecar_http.py`
- `tests/test_sidecar_admission.py`
- `tests/test_collect_tool_trace.py`
- `tests/test_build_tool_predictor_unit.py`
- `tests/test_dashboard_events.py`
- `tests/test_vllm_patch_text.py`
- `tests/test_agent_resume_timestamps.py`
- `tests/test_single_agent_track.py`
- `tests/test_tool_duration_predictor.py`

The test suite mixes small unit tests, text/contract tests, and integration
tests that require vLLM and benchmark dependencies.

## Quick Syntax Checks

These do not require project dependencies beyond Python and Bash:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc \
  python3 -m compileall -q src tests

bash -n run_abc-bench.sh
bash -n start_vllm.sh
```

Use `PYTHONPYCACHEPREFIX` to keep bytecode out of the repository.

## Focused Direct-Script Tests

Several tests can run as plain Python scripts when `pytest` is unavailable:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc \
  python3 tests/test_sidecar_http.py
```

Other tests are written as pytest-style functions but can be driven manually
when they do not need fixtures. The repository has used small harnesses for:

- sidecar admission controller tests
- collect tool trace tests
- dashboard event text tests
- vLLM patch text tests
- agent resume timestamp tests

Prefer `pytest` when available; direct execution is a fallback for minimal
machines.

## Unit and Contract Coverage

### `test_sidecar_http.py`

Covers the stdlib HTTP/SSE feed:

- ring buffer retention
- subscriber replay by `since` tick
- subscriber fanout
- replay at zero speed
- `/state`
- `/healthz`
- static path traversal protection
- `/stream` replay plus live frames

### `test_sidecar_admission.py`

Covers admission and offload policy behavior:

- headroom calculations
- admission gates
- active-agent caps
- readmit priority
- short vs long tool-call classification
- release/offload callback behavior
- exact freed-memory accounting
- fallback-long behavior when predictor output is unavailable or a short
  prediction ages badly
- percent-only pressure telemetry triggering fallback offload

### `test_collect_tool_trace.py`

Covers the trace collector:

- action/observation matching
- terminal command extraction
- artifact writing
- matching by tool call id
- finalizing unfinished calls
- output hashing and previews

### `test_build_tool_predictor_unit.py`

Covers predictor data plumbing:

- detailed trace loading and merging
- leakage-sensitive feature behavior
- long-call weighting
- HGB leaf sizing
- stratified splitting
- real-time predictor alignment

### `test_dashboard_events.py`

Covers browser source contracts through static analysis:

- successful offload markers only
- pressure badge fields
- vLLM preemption badge
- event line overlay
- event-specific admission timestamps
- event color consistency
- agent label panel width
- elapsed and fixed end-to-end labels
- standalone HTML snapshot support

### `test_vllm_patch_text.py`

Covers the patch and connector source text:

- route exposure for offload/restore/release/usage
- connector reuse of OffloadingConnector machinery
- held finished agent requests
- release behavior
- resident and offloadable block usage reporting
- vLLM 0.13 forwarding anchors

### `test_agent_resume_timestamps.py`

Covers the runner-side timestamp detail that an offloaded agent's next
reasoning phase starts at readmission time rather than the earlier observation
time.

## Integration Tests

### `test_single_agent_track.py`

Runs a real or near-real single-agent path and checks KV tracking behavior. It
expects benchmark dependencies and a reachable vLLM server.

### `test_tool_duration_predictor.py`

Builds a predictor from real ABC-Bench traces and tests real-time replay. It
requires:

- Python 3.12 benchmark environment
- OpenHands SDK/tools
- pandas
- numpy
- scikit-learn
- joblib
- a vLLM backend for the real benchmark path

These tests may bootstrap `.bench-venv` with `uv`.

## Dependency Expectations

Minimal source checks:

- Python 3
- Bash

Core runner tests:

- OpenHands SDK/tools
- PyYAML
- Pydantic
- requests

Predictor tests:

- pandas
- numpy
- scikit-learn
- joblib

vLLM integration:

- vLLM/vllm-ascend environment
- Ascend runtime stack
- reachable OpenAI-compatible model endpoint

## Recommended Local Sequence

For ordinary code cleanup:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc \
  python3 -m compileall -q src tests

bash -n run_abc-bench.sh
bash -n start_vllm.sh
```

Then run focused tests for touched components. For example, after dashboard or
feed changes:

```bash
PYTHONPYCACHEPREFIX=/tmp/agent-concurrency-pyc \
  python3 tests/test_sidecar_http.py
```

If `pytest` and dependencies are installed:

```bash
pytest tests/test_sidecar_http.py tests/test_collect_tool_trace.py
```

## Change Guidelines

- Add unit tests for pure logic in `src/sidecar.py`, `src/collect_tool_trace.py`,
  and `src/build_tool_predictor.py`.
- Add dashboard source-contract tests when changing event rendering, labels, or
  snapshot behavior.
- Add patch text tests when changing vLLM route names, connector state, or
  patch anchors.
- Keep integration tests explicit about external requirements so lightweight
  checks remain usable on development machines.
