#!/usr/bin/env python3
"""
Real ABC-Bench integration test for tool-time prediction.

This test exercises the intended production workflow:

  1. Run one real ABC-Bench task with ``run_single_agent``.
     The runner automatically writes detailed per-tool-call traces:
       <results>/<task_id>/<agent_id>_trace.csv
       <results>/<task_id>/<agent_id>_trace.jsonl

  2. Build and save a predictor exactly as a user would:
       python src/build_tool_predictor.py --trace-dir <results> --remaining --save-model ...

  3. Load the saved predictor and replay prediction calls over the real tool
     calls collected from that single-agent run.

Environment
───────────
  ABC_BENCH_ROOT             Path to ABC-Bench root or tasks dir. Required.
  ABC_BENCH_TASK             Task glob/name to run (default: task_*; first match).
  ABC_BENCH_VENV             Benchmark venv path (default: .bench-venv).
  PREDICTOR_MAX_ITERATIONS   Max agent iterations (default: 15).
  PREDICTOR_MODEL            Model used by build_tool_predictor.py (default: ridge).
  PREDICTOR_MODEL_FILE       Optional model output path.
  LLM_API_KEY                vLLM API key (default: sk-local-ascend).

Run
───
  pytest tests/test_tool_duration_predictor.py -vs
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).parent.parent

# ── venv bootstrap (mirrors test_single_agent_track.py) ──────────────────────

_BENCH_VENV = Path(os.getenv("ABC_BENCH_VENV", str(REPO_ROOT / ".bench-venv")))
_BENCH_PYTHON = _BENCH_VENV / "bin" / "python"
_REEXEC_MARKER = "PREDICTOR_SINGLE_AGENT_REEXECED"
_REQUIREMENTS = (
    "openhands-sdk",
    "openhands-tools",
    "scikit-learn",
    "pandas",
    "numpy",
    "joblib",
    "pytest",
    "PyYAML",
    "requests",
)
_PROBE_IMPORTS = (
    "openhands.sdk",
    "sklearn",
    "pandas",
    "numpy",
    "joblib",
    "pytest",
    "yaml",
    "requests",
)
_BENCH_ROOT_DEFAULTS = (REPO_ROOT.parent / "tasks", REPO_ROOT / "tasks")


def _set_defaults() -> None:
    os.environ.setdefault("LLM_API_KEY", "sk-local-ascend")
    if os.getenv("ABC_BENCH_ROOT"):
        return
    for candidate in _BENCH_ROOT_DEFAULTS:
        if candidate.is_dir():
            os.environ["ABC_BENCH_ROOT"] = str(candidate)
            return


def _deps_ok() -> bool:
    if not _BENCH_PYTHON.exists():
        return False
    probe = "\n".join(f"import {name}" for name in _PROBE_IMPORTS)
    return subprocess.run(
        [str(_BENCH_PYTHON), "-c", probe],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _bootstrap() -> None:
    if not _BENCH_PYTHON.exists():
        subprocess.run(
            ["uv", "venv", str(_BENCH_VENV), "--python", "3.12"],
            cwd=str(REPO_ROOT),
            check=True,
        )
    if not _deps_ok():
        subprocess.run(
            ["uv", "pip", "install", "--python", str(_BENCH_PYTHON), *_REQUIREMENTS],
            cwd=str(REPO_ROOT),
            check=True,
        )


def _in_bench_venv() -> bool:
    venv = os.getenv("VIRTUAL_ENV")
    if venv and Path(os.path.abspath(venv)) == Path(os.path.abspath(_BENCH_VENV)):
        return True
    try:
        Path(sys.executable).relative_to(_BENCH_VENV)
        return True
    except ValueError:
        return False


def _reexec() -> None:
    if _in_bench_venv():
        if not _deps_ok():
            _bootstrap()
        return
    if os.getenv(_REEXEC_MARKER) == "1":
        raise RuntimeError(
            f"Re-exec loop detected; still running {sys.executable!r}"
        )
    _bootstrap()
    env = os.environ.copy()
    env[_REEXEC_MARKER] = "1"
    env["VIRTUAL_ENV"] = str(_BENCH_VENV)
    env["PATH"] = f"{_BENCH_VENV / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    os.execve(
        str(_BENCH_PYTHON),
        [str(_BENCH_PYTHON), "-m", "pytest", *sys.argv[1:]],
        env,
    )


_set_defaults()
_reexec()

# ── imports that require the bench venv ───────────────────────────────────────
import numpy as np
import pandas as pd
import pytest
import requests
import yaml

sys.path.insert(0, str(REPO_ROOT / "src"))

from build_tool_predictor import (
    RealtimePredictor,
    build_features,
    enrich_sequential_features,
    load_tool_calls,
    load_traces,
)
from run_abc_bench_instrumented import (
    copy_task_to_workspace,
    find_task_dirs,
    now_utc,
    run_single_agent,
)


_VLLM_CFG = yaml.safe_load((REPO_ROOT / "config" / "vllm_config.yaml").read_text())
HOST_PORT = _VLLM_CFG["native"]["port"]
LLM_MODEL = f"openai/{_VLLM_CFG['native']['served_model_name'].split('/')[-1]}"
LLM_BASE_URL = f"http://localhost:{HOST_PORT}/v1"
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-local-ascend")
MAX_ITERATIONS = int(os.getenv("PREDICTOR_MAX_ITERATIONS", "15"))

_REPLAY_ELAPSED_FRACS = [0.0, 0.25, 0.50]


class _RunBundle(NamedTuple):
    trace_dir: Path
    model_path: Path
    tool_df: pd.DataFrame
    feat: pd.DataFrame


@pytest.fixture(scope="module")
def vllm_running():
    if not os.getenv("ABC_BENCH_ROOT"):
        pytest.skip("ABC_BENCH_ROOT not set; skipping real ABC-Bench predictor test")
    try:
        requests.get(
            f"http://localhost:{HOST_PORT}/v1/models",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=10,
        ).raise_for_status()
    except Exception as exc:
        pytest.fail(f"vLLM unreachable at {LLM_BASE_URL}: {exc}")
    yield


@pytest.fixture(scope="module")
def real_single_agent_predictor(tmp_path_factory, vllm_running) -> _RunBundle:
    root = Path(os.environ["ABC_BENCH_ROOT"])
    tasks = find_task_dirs(root, os.getenv("ABC_BENCH_TASK", "task_*"))
    if not tasks:
        pytest.skip(f"No ABC-Bench tasks matched under {root}")

    task_src = tasks[0]
    workspace = tmp_path_factory.mktemp("workspace")
    results = tmp_path_factory.mktemp("abc_results")
    work_dir = copy_task_to_workspace(task_src, workspace, suffix="_predictor")
    agent_id = f"agent_{task_src.name}_predictor"
    result_dir = results / task_src.name

    print(f"\n[data collection] running real ABC-Bench task: {task_src.name}")
    run_single_agent(
        agent_id=agent_id,
        task_dir=work_dir,
        llm_model=LLM_MODEL,
        llm_base_url=LLM_BASE_URL,
        llm_api_key=LLM_API_KEY,
        max_iterations=MAX_ITERATIONS,
        results_dir=result_dir,
        scheduled_dt=now_utc(),
    )

    trace_csv = result_dir / f"{agent_id}_trace.csv"
    trace_jsonl = result_dir / f"{agent_id}_trace.jsonl"
    assert trace_csv.exists(), f"runner did not write detailed trace CSV: {trace_csv}"
    assert trace_jsonl.exists(), f"runner did not write detailed trace JSONL: {trace_jsonl}"

    env_model = os.getenv("PREDICTOR_MODEL_FILE")
    model_path = Path(env_model) if env_model else results / "single_agent_tool_predictor.joblib"
    full_df = load_traces(results)
    tool_df = (
        load_tool_calls(results, full_df)
        .dropna(subset=["duration_s"])
        .copy()
        .reset_index(drop=True)
    )
    if tool_df.empty:
        pytest.skip("The real single-agent run produced no completed tool calls")
    if len(tool_df) < 2:
        pytest.skip("At least two real tool calls are needed to train and evaluate a predictor")

    required_cols = {
        "agent_id",
        "task_id",
        "tool_seq",
        "tool_name",
        "tool_args_json",
        "start_ts",
        "end_ts",
        "duration_s",
        "outcome",
        "observation_bytes",
        "start_active_agents",
    }
    missing = sorted(required_cols.difference(tool_df.columns))
    assert not missing, f"detailed trace is missing expected columns: {missing}"

    model_name = os.getenv("PREDICTOR_MODEL", "ridge")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "build_tool_predictor.py"),
        "--trace-dir",
        str(results),
        "--model",
        model_name,
        "--remaining",
        "--save-model",
        str(model_path),
    ]
    print("\n[predictor build] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    assert model_path.exists(), f"predictor was not saved: {model_path}"

    feat = build_features(tool_df, remaining_mode=True)
    feat = enrich_sequential_features(full_df, feat, tool_df).reset_index(drop=True)
    return _RunBundle(results, model_path, tool_df, feat)


def test_single_agent_builds_predictor_from_real_abc_bench(real_single_agent_predictor):
    bundle = real_single_agent_predictor
    predictor = RealtimePredictor.load(bundle.model_path)

    actuals = bundle.tool_df["duration_s"].astype(float).values
    preds = np.array([
        predictor.predict_duration(bundle.feat.iloc[[i]])
        for i in range(len(bundle.feat))
    ])

    assert np.all(np.isfinite(preds)), "duration predictions must be finite"
    assert np.all(preds >= 0), "duration predictions must be non-negative"

    print(
        f"\n[single-agent predictor] model={bundle.model_path} "
        f"trace_dir={bundle.trace_dir} n_tool_calls={len(actuals)}"
    )
    print(f"  {'#':>3}  {'tool':<22}  {'actual_s':>9}  {'pred_s':>9}  {'err_s':>9}")
    tools = bundle.tool_df["tool_name"].fillna("unknown").values
    for i in range(min(len(actuals), 30)):
        err = preds[i] - actuals[i]
        print(
            f"  {i:>3}  {tools[i]:<22}  "
            f"{actuals[i]:>9.3f}  {preds[i]:>9.3f}  {err:>+9.3f}"
        )


def test_single_agent_realtime_tool_prediction_replay(real_single_agent_predictor):
    bundle = real_single_agent_predictor
    predictor = RealtimePredictor.load(bundle.model_path)

    actuals = bundle.tool_df["duration_s"].astype(float).values
    pred_rem = np.zeros((len(actuals), len(_REPLAY_ELAPSED_FRACS)))

    for i, actual in enumerate(actuals):
        row = bundle.feat.iloc[[i]]
        for j, frac in enumerate(_REPLAY_ELAPSED_FRACS):
            pred_rem[i, j] = predictor.predict_remaining(row, elapsed_s=actual * frac)

    assert np.all(np.isfinite(pred_rem)), "remaining-time predictions must be finite"
    assert np.all(pred_rem >= 0), "remaining-time predictions must be non-negative"

    dispatch_preds = np.array([
        predictor.predict_duration(bundle.feat.iloc[[i]])
        for i in range(len(actuals))
    ])
    np.testing.assert_array_equal(pred_rem[:, 0], dispatch_preds)

    print("\n[realtime replay] real single-agent tool calls")
    print(
        f"  {'elapsed':>8}  {'act_rem_mean':>13}  {'pred_rem_mean':>14}  "
        f"{'MAE':>8}"
    )
    for j, frac in enumerate(_REPLAY_ELAPSED_FRACS):
        actual_remaining = np.maximum(0.0, actuals * (1.0 - frac))
        mae = float(np.mean(np.abs(actual_remaining - pred_rem[:, j])))
        print(
            f"  {frac:>7.0%}  {actual_remaining.mean():>12.3f}s  "
            f"{pred_rem[:, j].mean():>13.3f}s  {mae:>7.3f}s"
        )
