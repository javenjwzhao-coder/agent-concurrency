#!/usr/bin/env python3
"""
predict_tool_duration.py
────────────────────────
Train and evaluate a lightweight regression model that predicts tool-call
duration from the per-agent trace CSVs produced by
``run_abc_bench_instrumented.py``.

Design choices
──────────────
• Model:  We start with two baselines – Ridge regression (linear, fast,
  interpretable) and Random Forest (captures non-linear interactions).
  Both are scikit-learn estimators that train in seconds on thousands of
  rows.

• Features:  Extracted entirely from the CSV columns plus simple derived
  quantities.  No external data needed.  See ``FEATURE_COLUMNS`` and
  ``build_features()`` for the full list.

• Evaluation:  MAE, RMSE, R² on a held-out test split.  Also prints
  per-tool-name error breakdown so you can see which tools are easy vs.
  hard to predict.

Usage
─────
    # Train + evaluate on all CSVs in a results directory:
    python predict_tool_duration.py \
        --trace-dir ./abc_results \
        --model ridge \
        --test-fraction 0.2 \
        --seed 42

    # Save the trained model for later inference:
    python predict_tool_duration.py \
        --trace-dir ./abc_results \
        --model rf \
        --save-model ./duration_model.joblib

    # Predict on new trace data (no training):
    python predict_tool_duration.py \
        --trace-dir ./new_results \
        --load-model ./duration_model.joblib \
        --predict-only

Install:
    pip install scikit-learn pandas numpy joblib
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # type: ignore[attr-defined]


# ─────────────────────────── feature engineering ─────────────────────────────

# Features extracted from each tool_call row in the trace CSV.
FEATURE_COLUMNS = [
    # ── categorical (will be one-hot encoded) ──
    "tool_name",              # "terminal", "file_editor", "task_tracker"

    # ── numeric ──
    "tool_payload_bytes",     # size of serialised arguments
    "command_length",         # len(tool_command) in characters
    "command_token_count",    # rough word count of the terminal command
    "command_has_docker",     # 1 if command contains 'docker'
    "command_has_build",      # 1 if command contains 'build' / 'make' / 'compile'
    "command_has_test",       # 1 if command contains 'test' / 'pytest' / 'run-tests'
    "command_has_install",    # 1 if command contains 'install' / 'pip' / 'npm'
    "command_has_git",        # 1 if command contains 'git'
    "command_pipe_count",     # number of pipe (|) operators
    "command_semicolon_count",# number of chained commands (;)
    "phase_seq",              # position in the agent's phase sequence
    "cumulative_reasoning_s", # total reasoning time before this tool call
    "prior_tool_calls",       # how many tool calls preceded this one
    "prior_avg_tool_s",       # running average of prior tool-call durations
]

# Target column
TARGET = "duration_s"


def _keyword_flag(text: str, keywords: list[str]) -> int:
    """Return 1 if any keyword appears in ``text`` (case-insensitive)."""
    lower = text.lower()
    return int(any(kw in lower for kw in keywords))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw tool_call rows into a feature matrix.

    Parameters
    ----------
    df : DataFrame
        Rows where ``phase == "tool_call"``, straight from the CSV.

    Returns
    -------
    DataFrame with one column per feature in FEATURE_COLUMNS.
    """
    feat = pd.DataFrame(index=df.index)

    # ── direct copies ──
    feat["tool_name"] = df["tool_name"].fillna("unknown").astype(str)
    feat["tool_payload_bytes"] = pd.to_numeric(
        df["tool_payload_bytes"], errors="coerce").fillna(0).astype(int)

    cmd = df["tool_command"].fillna("").astype(str)
    feat["command_length"] = cmd.str.len()
    feat["command_token_count"] = cmd.str.split().str.len().fillna(0).astype(int)

    # ── keyword flags (binary) ──
    feat["command_has_docker"] = cmd.apply(
        lambda c: _keyword_flag(c, ["docker"]))
    feat["command_has_build"] = cmd.apply(
        lambda c: _keyword_flag(c, ["build", "make", "compile", "cmake",
                                     "cargo build", "go build"]))
    feat["command_has_test"] = cmd.apply(
        lambda c: _keyword_flag(c, ["test", "pytest", "run-tests", "unittest",
                                     "cargo test", "go test", "npm test"]))
    feat["command_has_install"] = cmd.apply(
        lambda c: _keyword_flag(c, ["install", "pip ", "npm ", "apt ",
                                     "yarn ", "cargo install"]))
    feat["command_has_git"] = cmd.apply(
        lambda c: _keyword_flag(c, ["git "]))

    # ── structural command features ──
    feat["command_pipe_count"] = cmd.str.count(r"\|")
    feat["command_semicolon_count"] = cmd.str.count(r";")

    # ── sequential / contextual features ──
    # These capture where in the agent's lifecycle this tool call sits.
    feat["phase_seq"] = pd.to_numeric(
        df["phase_seq"], errors="coerce").fillna(0).astype(int)

    # Cumulative reasoning time and prior tool-call stats must be computed
    # per-agent.  We do this in a grouped pass.
    feat["cumulative_reasoning_s"] = 0.0
    feat["prior_tool_calls"] = 0
    feat["prior_avg_tool_s"] = 0.0

    return feat


def enrich_sequential_features(full_df: pd.DataFrame,
                                feat_df: pd.DataFrame,
                                tool_mask: pd.Series) -> pd.DataFrame:
    """
    Compute per-agent sequential features that depend on the *full* trace
    (including reasoning rows), not just tool_call rows.

    This is called after ``build_features`` and fills in the three columns
    that require per-agent history: ``cumulative_reasoning_s``,
    ``prior_tool_calls``, ``prior_avg_tool_s``.
    """
    # Work on a copy to avoid SettingWithCopy warnings.
    feat = feat_df.copy()

    for agent_id, group in full_df.groupby("agent_id"):
        cum_reasoning = 0.0
        tool_count = 0
        tool_dur_sum = 0.0

        for idx, row in group.iterrows():
            if row["phase"] == "reasoning":
                cum_reasoning += float(row.get("duration_s", 0) or 0)
            elif row["phase"] == "tool_call" and idx in feat.index:
                feat.at[idx, "cumulative_reasoning_s"] = cum_reasoning
                feat.at[idx, "prior_tool_calls"] = tool_count
                feat.at[idx, "prior_avg_tool_s"] = (
                    (tool_dur_sum / tool_count) if tool_count > 0 else 0.0
                )
                tool_count += 1
                tool_dur_sum += float(row.get("duration_s", 0) or 0)

    return feat


# ─────────────────────────── data loading ────────────────────────────────────

def load_traces(trace_dir: Path) -> pd.DataFrame:
    """
    Recursively find all ``*_trace.csv`` files under ``trace_dir`` and
    concatenate them into a single DataFrame.
    """
    csv_files = sorted(trace_dir.rglob("*_trace.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_trace.csv files found under {trace_dir}")

    frames = []
    for p in csv_files:
        try:
            df = pd.read_csv(p, dtype=str)
            frames.append(df)
        except Exception as exc:
            print(f"WARNING: skipping {p}: {exc}", file=sys.stderr)
    if not frames:
        raise ValueError("All trace CSVs failed to load")

    combined = pd.concat(frames, ignore_index=True)
    # Coerce numeric columns.
    for col in ("phase_seq", "duration_s", "tool_payload_bytes"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    print(f"Loaded {len(combined)} phase records from {len(csv_files)} file(s)")
    return combined


def prepare_dataset(trace_dir: Path):
    """
    Load traces, filter to tool_call rows, and build feature + target
    arrays.

    Returns
    -------
    X : DataFrame   – feature matrix (numeric + one-hot encoded)
    y : Series      – target (duration_s)
    raw : DataFrame – the original tool_call rows (for diagnostics)
    full_df: DataFrame – the entire trace (for sequential feature computation)
    """
    full_df = load_traces(trace_dir)

    tool_mask = full_df["phase"] == "tool_call"
    tool_df = full_df.loc[tool_mask].copy()

    if tool_df.empty:
        raise ValueError("No tool_call rows found in traces")

    # Drop rows with missing target.
    tool_df = tool_df.dropna(subset=[TARGET])
    y = tool_df[TARGET].astype(float)

    # Build features.
    feat = build_features(tool_df)
    feat = enrich_sequential_features(full_df, feat, tool_mask)

    # One-hot encode tool_name.
    feat = pd.get_dummies(feat, columns=["tool_name"], prefix="tool",
                          dtype=float)

    # Ensure all numeric.
    X = feat.astype(float)

    print(f"Dataset: {len(X)} tool-call samples, {X.shape[1]} features")
    return X, y, tool_df, full_df


# ─────────────────────────── model building ──────────────────────────────────

def build_model(name: str) -> Pipeline:
    """
    Return a scikit-learn Pipeline for the requested model type.

    Supported: 'ridge', 'rf' (random forest).
    """
    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
    elif name == "rf":
        return Pipeline([
            # Random forest doesn't need scaling, but the pipeline keeps the
            # interface uniform.
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )),
        ])
    else:
        raise ValueError(f"Unknown model: {name!r}.  Use 'ridge' or 'rf'.")


# ─────────────────────────── evaluation ──────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             tool_names: Optional[pd.Series] = None) -> dict:
    """
    Compute regression metrics.  If ``tool_names`` is provided, also print
    a per-tool breakdown.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4),
               "r2": round(r2, 4), "n_samples": len(y_true)}

    print("\n── Overall metrics ──")
    print(f"  MAE  = {mae:.4f} s")
    print(f"  RMSE = {rmse:.4f} s")
    print(f"  R²   = {r2:.4f}")
    print(f"  n    = {len(y_true)}")

    if tool_names is not None and len(tool_names) == len(y_true):
        print("\n── Per-tool breakdown ──")
        breakdown = {}
        for name in sorted(tool_names.unique()):
            mask = tool_names.values == name
            if mask.sum() < 1:
                continue
            t_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            t_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            n = int(mask.sum())
            print(f"  {name:20s}  MAE={t_mae:.3f}s  RMSE={t_rmse:.3f}s  n={n}")
            breakdown[name] = {"mae": round(t_mae, 4),
                               "rmse": round(t_rmse, 4), "n": n}
        metrics["per_tool"] = breakdown

    return metrics


# ─────────────────────────── feature importance ──────────────────────────────

def print_feature_importance(pipeline: Pipeline, feature_names: list[str]):
    """Print feature importances if the model exposes them."""
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return

    order = np.argsort(importances)[::-1]
    print("\n── Top feature importances ──")
    for rank, idx in enumerate(order[:15]):
        print(f"  {rank+1:2d}. {feature_names[idx]:30s}  {importances[idx]:.4f}")


# ──────────────────────────────── CLI ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a tool-call duration predictor from agent traces.")
    p.add_argument("--trace-dir", required=True, type=Path,
                   help="Directory containing *_trace.csv files (searched "
                        "recursively).")
    p.add_argument("--model", default="ridge", choices=["ridge", "rf"],
                   help="Regression model to train (default: ridge).")
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="Fraction of data to hold out for testing.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-model", type=Path, default=None,
                   help="Save trained model to this path (.joblib).")
    p.add_argument("--load-model", type=Path, default=None,
                   help="Load a pre-trained model instead of training.")
    p.add_argument("--predict-only", action="store_true",
                   help="Skip training; just predict with --load-model.")
    p.add_argument("--output-predictions", type=Path, default=None,
                   help="Write predictions to this CSV path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    X, y, raw_tool_df, full_df = prepare_dataset(args.trace_dir)
    tool_names = raw_tool_df["tool_name"].fillna("unknown")

    if args.predict_only:
        # ── inference-only mode ──
        if args.load_model is None:
            print("ERROR: --predict-only requires --load-model", file=sys.stderr)
            return 2
        pipeline = joblib.load(args.load_model)
        print(f"Loaded model from {args.load_model}")

        # Align columns: the saved model was trained on a specific feature set.
        # Pad missing one-hot columns with 0, drop extras.
        train_cols = getattr(pipeline, "_trained_feature_names", None)
        if train_cols is not None:
            for col in train_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[train_cols]

        y_pred = pipeline.predict(X)
        evaluate(y.values, y_pred, tool_names)

        if args.output_predictions:
            out = raw_tool_df.copy()
            out["predicted_duration_s"] = y_pred
            out.to_csv(args.output_predictions, index=False)
            print(f"\nPredictions written → {args.output_predictions}")
        return 0

    # ── train / test split ──
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, X.index, test_size=args.test_fraction, random_state=args.seed)

    tool_names_test = tool_names.loc[idx_test]

    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")
    print(f"Model: {args.model}")

    # ── train ──
    pipeline = build_model(args.model)
    pipeline.fit(X_train, y_train)

    # Store feature names for later inference alignment.
    # Pipeline.feature_names_in_ is read-only, so use a custom attribute.
    pipeline._trained_feature_names = list(X.columns)

    # ── evaluate ──
    y_pred_test = pipeline.predict(X_test)
    metrics = evaluate(y_test.values, y_pred_test, tool_names_test)

    print_feature_importance(pipeline, list(X.columns))

    # ── save ──
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, args.save_model)
        print(f"\nModel saved → {args.save_model}")

    if args.output_predictions:
        out = raw_tool_df.loc[idx_test].copy()
        out["predicted_duration_s"] = y_pred_test
        out.to_csv(args.output_predictions, index=False)
        print(f"Test predictions written → {args.output_predictions}")

    # ── write metrics JSON ──
    metrics_path = args.trace_dir / "prediction_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written → {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())