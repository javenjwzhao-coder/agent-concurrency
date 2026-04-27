#!/usr/bin/env python3
"""
tests/test_tool_duration_predictor.py

End-to-end validation of the tool-call duration predictor in deployed mode.

Pipeline
────────
  Phase 1 – data collection  (trace_dir fixture)
      Run *all* tasks found under ABC_BENCH_ROOT, either sequentially or
      concurrently (PREDICTOR_CONCURRENT=1), and collect their *_trace.csv
      files.  Skip this phase by pointing PREDICTOR_TRACE_DIR at an existing
      trace directory.

  Phase 2 – training + save  (saved_model fixture)
      Split collected tool-call rows 70 / 30.  Train a remaining-time HGB
      predictor on the 70 % portion and SAVE it to disk (joblib) so it can
      be loaded and deployed independently of this test run.
      The save path is PREDICTOR_MODEL_FILE if set, otherwise
      <trace_dir>/duration_predictor.joblib.

  Phase 3 – tests
      Both test functions load the saved model from disk (mimicking real
      deployment) and evaluate it against the held-out 30 %.

      test_predictor_trains_and_scores
          Point-estimate (total-duration at elapsed=0) evaluation.
          Prints per-row actual / predicted / error table and aggregate
          MAE / MAPE / R².  Hard-asserts non-negative, finite outputs.

      test_realtime_replay
          Queries the predictor at elapsed = 0 % (dispatch), 25 %, and 50 %
          of each actual tool-call duration, exactly as it would be called in
          production.  Compares predicted remaining time to actual remaining
          time.  Hard assertions:
            1. All predictions non-negative and finite.
            2. predict_remaining(elapsed=0) == predict_duration() numerically.
            3. Predicted remaining decreases as elapsed increases for ≥ 75 %
               of calls (≤ 25 % monotonicity violation tolerated).
          Prints a per-elapsed-fraction error table and a per-call detail table.

Environment variables
─────────────────────
  ABC_BENCH_ROOT           Path to the ABC-Bench root (tasks/ inside). Required.
  ABC_BENCH_TASK           Task name glob (default: task_*).
  ABC_BENCH_VENV           Benchmark venv path (default: .bench-venv).
  PREDICTOR_TRACE_DIR      Pre-built trace directory; skips agent runs.
  PREDICTOR_MODEL_FILE     Where to save the trained model (joblib).
                           Default: <trace_dir>/duration_predictor.joblib.
  PREDICTOR_CONCURRENT     Set to "1" to run all tasks concurrently.
  PREDICTOR_MAX_CONCURRENT Max concurrent agents when PREDICTOR_CONCURRENT=1
                           (default: 4).
  PREDICTOR_MAX_ITERATIONS Max agent iterations per task (default: 15).
  LLM_API_KEY              API key for the vLLM endpoint (default: sk-local-ascend).

Run
───
  # Sequential (safe default) — assumes vLLM is already running:
  pytest tests/test_tool_duration_predictor.py -vs

  # Concurrent collection across all tasks:
  PREDICTOR_CONCURRENT=1 pytest tests/test_tool_duration_predictor.py -vs

  # Skip data collection, use existing traces, save model to a known path:
  PREDICTOR_TRACE_DIR=./abc_results \\
  PREDICTOR_MODEL_FILE=./duration_predictor.joblib \\
  pytest tests/test_tool_duration_predictor.py -vs

  # Load a pre-saved model (no training):
  PREDICTOR_TRACE_DIR=./abc_results \\
  PREDICTOR_MODEL_FILE=./duration_predictor.joblib \\
  pytest tests/test_tool_duration_predictor.py -vs
"""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent

# ── venv bootstrap (mirrors test_single_agent_track.py) ──────────────────────

_BENCH_VENV    = Path(os.getenv("ABC_BENCH_VENV", str(REPO_ROOT / ".bench-venv")))
_BENCH_PYTHON  = _BENCH_VENV / "bin" / "python"
_REEXEC_MARKER = "PREDICTOR_TEST_REEXECED"
_REQUIREMENTS  = (
    "openhands-sdk", "openhands-tools",
    "scikit-learn", "pandas", "numpy", "joblib",
    "pytest", "PyYAML",
)
_PROBE_IMPORTS = (
    "openhands.sdk", "sklearn", "pandas", "numpy", "joblib", "pytest", "yaml",
)
_BENCH_ROOT_DEFAULTS = (REPO_ROOT.parent / "tasks", REPO_ROOT / "tasks")


def _set_defaults() -> None:
    os.environ.setdefault("LLM_API_KEY", "sk-local-ascend")
    if not os.getenv("ABC_BENCH_ROOT"):
        for c in _BENCH_ROOT_DEFAULTS:
            if c.is_dir():
                os.environ["ABC_BENCH_ROOT"] = str(c)
                return


def _deps_ok() -> bool:
    if not _BENCH_PYTHON.exists():
        return False
    probe = "\n".join(f"import {n}" for n in _PROBE_IMPORTS)
    return subprocess.run(
        [str(_BENCH_PYTHON), "-c", probe],
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
import pytest
import yaml

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # type: ignore[attr-defined]

sys.path.insert(0, str(REPO_ROOT / "src"))

from predict_tool_duration import (
    _REMAINING_FRACS,
    augment_remaining,
    build_features,
    build_model,
    enrich_sequential_features,
    inv_transform,
    load_traces,
    log_transform,
)
from run_abc_bench_instrumented import (
    copy_task_to_workspace,
    find_task_dirs,
    now_utc,
    run_single_agent,
)

# ── endpoint config from vllm_config.yaml ────────────────────────────────────
_VLLM_CFG    = yaml.safe_load((REPO_ROOT / "config" / "vllm_config.yaml").read_text())
HOST_PORT    = _VLLM_CFG["native"]["port"]
LLM_MODEL    = f"openai/{_VLLM_CFG['native']['served_model_name'].split('/')[-1]}"
LLM_BASE_URL = f"http://localhost:{HOST_PORT}/v1"
LLM_API_KEY  = os.getenv("LLM_API_KEY", "sk-local-ascend")

MAX_ITERATIONS  = int(os.getenv("PREDICTOR_MAX_ITERATIONS", "15"))
MAX_CONCURRENT  = int(os.getenv("PREDICTOR_MAX_CONCURRENT", "4"))
RUN_CONCURRENT  = os.getenv("PREDICTOR_CONCURRENT", "0") == "1"

# Elapsed fractions at which the predictor is queried during real-time replay.
_REPLAY_FRACS: list[float] = [0.0, 0.25, 0.50]


# ─────────────────────────── task runner helpers ─────────────────────────────

def _run_one_task(
    task_dir: Path,
    workspace: Path,
    out_dir: Path,
    agent_idx: int,
) -> None:
    """Run a single agent on one task; write trace CSV to out_dir."""
    agent_id = f"agent_{task_dir.name}_{agent_idx}"
    work_dir = copy_task_to_workspace(task_dir, workspace, suffix=f"_{agent_idx}")
    run_single_agent(
        agent_id       = agent_id,
        task_dir       = work_dir,
        llm_model      = LLM_MODEL,
        llm_base_url   = LLM_BASE_URL,
        llm_api_key    = LLM_API_KEY,
        max_iterations = MAX_ITERATIONS,
        results_dir    = out_dir / task_dir.name,
        scheduled_dt   = now_utc(),
    )


def _collect_traces_sequential(
    tasks: list[Path],
    workspace: Path,
    out_dir: Path,
) -> None:
    for idx, task in enumerate(tasks):
        print(f"\n[data collection] [{idx + 1}/{len(tasks)}] {task.name} …")
        try:
            _run_one_task(task, workspace, out_dir, idx)
        except Exception as exc:
            print(f"  WARNING: {task.name} failed: {exc}", file=sys.stderr)


def _collect_traces_concurrent(
    tasks: list[Path],
    workspace: Path,
    out_dir: Path,
    max_workers: int,
) -> None:
    print(
        f"\n[data collection] Launching {len(tasks)} tasks "
        f"(≤{max_workers} concurrent) …"
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one_task, task, workspace, out_dir, idx): task
            for idx, task in enumerate(tasks)
        }
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                fut.result()
                print(f"  ✓ {task.name}")
            except Exception as exc:
                print(f"  WARNING: {task.name} failed: {exc}", file=sys.stderr)


# ─────────────────────────── RealtimePredictor ───────────────────────────────

class RealtimePredictor:
    """
    Wraps a saved remaining-time pipeline for deployment-style inference.

    Accepts feature rows from build_features() / enrich_sequential_features()
    where tool_name is still a raw string column.  One-hot encoding and column
    alignment are handled internally so callers do not need to know the exact
    training schema.

    This is exactly how the predictor would be used in production:
        predictor = RealtimePredictor.load("duration_predictor.joblib")
        # on ActionEvent:
        pred_total    = predictor.predict_duration(feat_row)
        # mid-execution:
        pred_remaining = predictor.predict_remaining(feat_row, elapsed_s=3.7)
    """

    def __init__(
        self,
        pipeline,
        feature_names: list[str],
        log_target: bool = True,
    ) -> None:
        self._pipeline  = pipeline
        self._feat_cols = feature_names
        self._log       = log_target

    @classmethod
    def load(cls, path: Path | str, log_target: bool = True) -> "RealtimePredictor":
        pipeline = joblib.load(path)
        feature_names = getattr(pipeline, "_trained_feature_names", None)
        if feature_names is None:
            raise ValueError(
                f"Model at {path!r} is missing _trained_feature_names — "
                "was it saved by this test suite?"
            )
        return cls(pipeline, feature_names, log_target=log_target)

    def _prepare(self, feat: pd.DataFrame, elapsed_s: float) -> pd.DataFrame:
        x = feat.copy()
        x["elapsed_s"] = elapsed_s
        if "tool_name" in x.columns:
            x = pd.get_dummies(x, columns=["tool_name"], prefix="tool", dtype=float)
        x = x.astype(float)
        for col in self._feat_cols:
            if col not in x.columns:
                x[col] = 0.0
        return x[self._feat_cols]

    def predict_remaining(self, feat: pd.DataFrame, elapsed_s: float) -> float:
        """Predicted remaining seconds at the given elapsed point in the call."""
        x   = self._prepare(feat, elapsed_s)
        raw = self._pipeline.predict(x)
        val = inv_transform(raw)[0] if self._log else float(raw[0])
        return float(max(0.0, val))

    def predict_duration(self, feat: pd.DataFrame) -> float:
        """Predicted total duration at dispatch time (elapsed_s = 0)."""
        return self.predict_remaining(feat, elapsed_s=0.0)


# ────────────────────────── shared data helpers ───────────────────────────────

class _Split(NamedTuple):
    feat_tr: pd.DataFrame   # pre-encoded features, training rows
    feat_te: pd.DataFrame   # pre-encoded features, test rows
    tool_tr: pd.DataFrame   # tool_call rows, training
    tool_te: pd.DataFrame   # tool_call rows, test
    full_df: pd.DataFrame   # entire trace (for sequential context)


def _build_split(trace_dir: Path, seed: int = 42) -> _Split:
    """
    Load all *_trace.csv files from trace_dir, build raw (pre-encoded)
    features for every tool_call row, split 70 / 30 by row index.

    'Pre-encoded' means tool_name is still a string column and elapsed_s is
    0 for every row.  Both are handled at inference time by RealtimePredictor.
    The split is deterministic via the fixed seed so both test functions see
    the same partition.
    """
    full_df = load_traces(trace_dir)
    tool_df = (
        full_df.loc[full_df["phase"] == "tool_call"]
               .dropna(subset=["duration_s"])
               .copy()
               .reset_index(drop=True)
    )

    if tool_df.empty:
        empty = pd.DataFrame()
        return _Split(empty, empty, empty, empty, full_df)

    feat = build_features(tool_df, remaining_mode=True)
    feat = enrich_sequential_features(full_df, feat).reset_index(drop=True)

    n      = len(tool_df)
    idx    = np.random.default_rng(seed).permutation(n)
    n_tr   = max(1, int(n * 0.70))
    tr_idx = idx[:n_tr]
    te_idx = idx[n_tr:]

    return _Split(
        feat_tr = feat.iloc[tr_idx].reset_index(drop=True),
        feat_te = feat.iloc[te_idx].reset_index(drop=True),
        tool_tr = tool_df.iloc[tr_idx].reset_index(drop=True),
        tool_te = tool_df.iloc[te_idx].reset_index(drop=True),
        full_df = full_df,
    )


def _fit_and_save(
    feat_tr: pd.DataFrame,
    tool_tr: pd.DataFrame,
    save_path: Path,
    model_name: str = "hgb",
) -> None:
    """
    Augment training rows with elapsed-fraction snapshots, one-hot encode,
    fit the pipeline, tag it with training column names, and save to disk.

    Augmentation order mirrors prepare_dataset():
        augment_remaining() → pd.get_dummies() → pipeline.fit() → joblib.dump()
    """
    X_aug, y_raw, _ = augment_remaining(tool_tr, feat_tr, _REMAINING_FRACS)
    X_enc = pd.get_dummies(
        X_aug, columns=["tool_name"], prefix="tool", dtype=float
    ).astype(float)

    pipeline = build_model(model_name)
    pipeline.fit(X_enc, log_transform(y_raw))
    pipeline._trained_feature_names = list(X_enc.columns)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"\n[predictor] Model saved → {save_path}")
    print(
        f"[predictor] Load later with:\n"
        f"  predictor = RealtimePredictor.load({str(save_path)!r})"
    )


# ─────────────────────────────── fixtures ────────────────────────────────────

@pytest.fixture(scope="module")
def vllm_running():
    """Assert vLLM is reachable before data collection starts."""
    import requests

    if not os.getenv("ABC_BENCH_ROOT"):
        pytest.skip(
            "ABC_BENCH_ROOT not set — skipping predictor integration test"
        )
    try:
        requests.get(
            f"http://localhost:{HOST_PORT}/v1/models",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            timeout=10,
        ).raise_for_status()
    except Exception as exc:
        pytest.fail(f"vLLM unreachable at port {HOST_PORT}: {exc}")
    yield


@pytest.fixture(scope="module")
def trace_dir(tmp_path_factory, vllm_running):
    """
    Return a Path containing *_trace.csv files.

    If PREDICTOR_TRACE_DIR is set, return that directory directly (fast path
    for re-running the tests after traces are already collected).

    Otherwise, discover all tasks under ABC_BENCH_ROOT and run each one via
    run_single_agent, either sequentially (default) or concurrently
    (PREDICTOR_CONCURRENT=1).  Each task writes its trace to a per-task
    subdirectory of out_dir so results do not collide.
    """
    env_dir = os.getenv("PREDICTOR_TRACE_DIR")
    if env_dir:
        d = Path(env_dir)
        if not d.exists():
            pytest.fail(f"PREDICTOR_TRACE_DIR={env_dir!r} does not exist")
        return d

    root  = Path(os.environ["ABC_BENCH_ROOT"])
    tasks = find_task_dirs(root, os.getenv("ABC_BENCH_TASK", "task_*"))
    if not tasks:
        pytest.skip(f"No tasks matched under {root}")

    workspace = tmp_path_factory.mktemp("workspace")
    out_dir   = tmp_path_factory.mktemp("traces")

    if RUN_CONCURRENT:
        _collect_traces_concurrent(tasks, workspace, out_dir, MAX_CONCURRENT)
    else:
        _collect_traces_sequential(tasks, workspace, out_dir)

    if not list(out_dir.rglob("*_trace.csv")):
        pytest.fail(
            "No trace CSVs were produced — all agent runs failed.  "
            "Check vLLM logs and ABC_BENCH_ROOT."
        )
    return out_dir


@pytest.fixture(scope="module")
def saved_model(trace_dir):
    """
    Build the 70 / 30 split, train a remaining-time predictor on the 70 %
    portion, save it to disk, and return (model_path, split).

    Both test functions load the model from model_path — this mirrors
    real deployment where the model is loaded once at process start.

    The save path is:
      • PREDICTOR_MODEL_FILE   if the env var is set
      • <trace_dir>/duration_predictor.joblib  otherwise
    """
    split = _build_split(trace_dir)

    if split.feat_tr.empty:
        pytest.skip("No tool-call rows found in traces")
    if len(split.tool_te) < 2:
        pytest.skip(
            f"Only {len(split.tool_te)} held-out tool-call row(s) — "
            "need ≥ 2 for evaluation.  Run more tasks or increase "
            "PREDICTOR_MAX_ITERATIONS."
        )

    env_path = os.getenv("PREDICTOR_MODEL_FILE")
    model_path = Path(env_path) if env_path else trace_dir / "duration_predictor.joblib"

    _fit_and_save(split.feat_tr, split.tool_tr, model_path)

    return model_path, split


# ──────────────────────────────── tests ──────────────────────────────────────

def test_predictor_trains_and_scores(saved_model):
    """
    Load the saved predictor from disk and evaluate point-estimate (total
    duration at elapsed=0) predictions against the held-out 30 %.

    This validates the training pipeline end-to-end and measures how well
    the model generalises to unseen tool calls.

    Hard assertions
    ───────────────
    • All predictions are non-negative.
    • All predictions are finite.

    Reported for inspection
    ───────────────────────
    • Per-row table: tool name | actual duration | predicted | signed error.
    • Aggregate MAE, MAPE, R².
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    model_path, split = saved_model
    predictor = RealtimePredictor.load(model_path)

    actuals = split.tool_te["duration_s"].astype(float).values
    preds   = np.array([
        predictor.predict_duration(split.feat_te.iloc[[i]])
        for i in range(len(split.feat_te))
    ])

    # ── hard assertions ───────────────────────────────────────────────────────
    assert np.all(preds >= 0),         "Negative duration prediction(s) detected"
    assert np.all(np.isfinite(preds)), "Non-finite duration prediction(s) detected"

    # ── per-row comparison table ──────────────────────────────────────────────
    tn = split.tool_te["tool_name"].fillna("unknown").values
    print(
        f"\n── test_predictor_trains_and_scores "
        f"(n_train={len(split.feat_tr)}, n_test={len(split.feat_te)}) ──"
    )
    print(f"  {'#':>3}  {'tool':<22}  {'actual_s':>9}  {'pred_s':>9}  {'err_s':>9}")
    for i in range(len(actuals)):
        err = preds[i] - actuals[i]
        print(
            f"  {i:>3}  {tn[i]:<22}  "
            f"{actuals[i]:>9.3f}  {preds[i]:>9.3f}  {err:>+9.3f}"
        )

    # ── aggregate metrics ─────────────────────────────────────────────────────
    mae  = mean_absolute_error(actuals, preds)
    r2   = r2_score(actuals, preds)
    nz   = actuals > 0
    mape = (
        float(np.mean(np.abs((actuals[nz] - preds[nz]) / actuals[nz])) * 100)
        if nz.any() else float("nan")
    )
    print(f"\n  MAE={mae:.4f}s   MAPE={mape:.1f}%   R²={r2:.4f}")
    print(f"\n[deploy] predictor ready at: {model_path}")


def test_realtime_replay(saved_model):
    """
    Load the saved predictor from disk and simulate real-time deployment:
    for each held-out tool call, query the predictor at elapsed = 0 %
    (dispatch), 25 %, and 50 % of the actual duration, then compare
    predicted remaining time to actual remaining time.

    This mirrors the production loop:
        # ActionEvent fires — we know features but not duration yet
        pred_total = predictor.predict_duration(feat_row)

        # poll mid-execution (e.g. every N seconds)
        pred_remaining = predictor.predict_remaining(feat_row, elapsed_s=t)

        # ObservationEvent fires — record actual, compute error
        actual = end_ts - start_ts

    Hard assertions
    ───────────────
    1. All predictions non-negative and finite.
    2. predict_remaining(feat, elapsed=0) == predict_duration(feat) exactly
       (structural: same pipeline call with elapsed_s=0).
    3. Predicted remaining is weakly decreasing as elapsed grows for ≥ 75 %
       of calls (≤ 25 % monotonicity violation tolerated for small datasets).

    Reported for inspection
    ───────────────────────
    • Per-elapsed-fraction error table: mean actual remaining | mean predicted
      remaining | MAE | MAPE.
    • Per-call detail table (up to 30 rows): actual duration and predicted
      remaining at each snapshot.
    """
    model_path, split = saved_model
    predictor = RealtimePredictor.load(model_path)

    actuals = split.tool_te["duration_s"].astype(float).values
    n_te    = len(actuals)

    # ── collect predictions: pred_rem[i, j] for call i at fraction j ─────────
    pred_rem = np.zeros((n_te, len(_REPLAY_FRACS)))
    for i, actual in enumerate(actuals):
        row = split.feat_te.iloc[[i]]
        for j, frac in enumerate(_REPLAY_FRACS):
            pred_rem[i, j] = predictor.predict_remaining(row, elapsed_s=actual * frac)

    # ── 1: non-negative + finite ──────────────────────────────────────────────
    assert np.all(pred_rem >= 0),         "Negative remaining-time prediction"
    assert np.all(np.isfinite(pred_rem)), "Non-finite remaining-time prediction"

    # ── 2: predict_remaining(elapsed=0) == predict_duration ──────────────────
    dur_preds = np.array([
        predictor.predict_duration(split.feat_te.iloc[[i]])
        for i in range(n_te)
    ])
    np.testing.assert_array_equal(
        pred_rem[:, 0],
        dur_preds,
        err_msg=(
            "predict_remaining(feat, elapsed=0) must be numerically identical "
            "to predict_duration(feat) — both call the pipeline with elapsed_s=0"
        ),
    )

    # ── 3: monotonicity (remaining should decrease as elapsed grows) ──────────
    # Compare remaining@first_frac vs remaining@last_frac.
    violations = int(np.sum(pred_rem[:, -1] > pred_rem[:, 0]))
    viol_rate  = violations / n_te
    print(
        f"\n  Monotonicity: {violations}/{n_te} calls have "
        f"remaining@{_REPLAY_FRACS[-1]:.0%} > remaining@{_REPLAY_FRACS[0]:.0%}  "
        f"[violation rate {viol_rate:.1%}]"
    )
    assert viol_rate <= 0.25, (
        f"{viol_rate:.1%} > 25 % of predictions are non-monotone "
        f"(predicted remaining grows as elapsed grows) — model quality too low"
    )

    # ── per-fraction error table ──────────────────────────────────────────────
    print(f"\n── test_realtime_replay: {n_te} held-out tool calls ──")
    print(
        f"  {'elapsed':>8}  {'act_rem_mean':>13}  {'pred_rem_mean':>14}  "
        f"{'MAE':>8}  {'MAPE':>8}"
    )
    for j, frac in enumerate(_REPLAY_FRACS):
        act_rem = np.maximum(0.0, actuals * (1.0 - frac))
        p_rem   = pred_rem[:, j]
        mae     = float(np.mean(np.abs(act_rem - p_rem)))
        nz      = act_rem > 0
        mape    = (
            float(np.mean(np.abs((act_rem[nz] - p_rem[nz]) / act_rem[nz])) * 100)
            if nz.any() else float("nan")
        )
        print(
            f"  {frac:>7.0%}  {act_rem.mean():>12.3f}s  "
            f"{p_rem.mean():>13.3f}s  {mae:>7.3f}s  {mape:>7.1f}%"
        )

    # ── per-call detail table (first 30 rows) ─────────────────────────────────
    tn = split.tool_te["tool_name"].fillna("unknown").values
    frac_header = "  ".join(f"{'rem@' + f'{f:.0%}':>11}" for f in _REPLAY_FRACS)
    print(f"\n  {'#':>3}  {'tool':<22}  {'actual_s':>9}  {frac_header}")
    for i in range(min(n_te, 30)):
        frac_vals = "  ".join(
            f"{pred_rem[i, j]:>11.3f}" for j in range(len(_REPLAY_FRACS))
        )
        print(f"  {i:>3}  {tn[i]:<22}  {actuals[i]:>9.3f}s  {frac_vals}")

    print(f"\n[deploy] predictor ready at: {model_path}")
