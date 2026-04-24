#!/usr/bin/env python3
"""
predict_tool_duration.py  (v2)
──────────────────────────────
Train and evaluate a regression model that predicts tool-call
duration **or remaining time** from per-agent trace CSVs produced by
``run_abc_bench_instrumented.py``.

Improvements over v1
────────────────────
• Log-transform target    – heavy-tailed durations → much better fit
• HistGradientBoosting    – handles NaN natively, often best on tabular data
• Group-split by agent    – prevents data leakage across train / test
• GroupKFold CV           – more reliable held-out evaluation
• Remaining-time mode     – augments each call at elapsed-time snapshots;
                            target = max(0, duration - elapsed)
• Richer features         – network cmds, python, redirect count,
                            prior tool std / max
• Quantile bands          – P10 / P50 / P90 via GradientBoostingRegressor
• MAPE + MedianAE         – interpretable and outlier-robust metrics
• Clip predictions ≥ 0    – durations can't be negative

Usage
─────
    # Train + evaluate (default: hgb, log-target, agent-group split):
    python predict_tool_duration.py --trace-dir ./abc_results

    # Remaining-time prediction with 5-fold CV:
    python predict_tool_duration.py \\
        --trace-dir ./abc_results --remaining --cv 5

    # Quantile bands (P10/P50/P90) saved to CSV:
    python predict_tool_duration.py \\
        --trace-dir ./abc_results --quantiles --output-predictions preds.csv

    # Save trained model:
    python predict_tool_duration.py \\
        --trace-dir ./abc_results --model hgb --save-model model.joblib

    # Inference on new traces:
    python predict_tool_duration.py \\
        --trace-dir ./new_results --load-model model.joblib --predict-only

Install:
    pip install scikit-learn pandas numpy joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib  # type: ignore[attr-defined]


# ─────────────────────────── constants ───────────────────────────────────────

# Elapsed-time fractions at which we sample remaining-time training snapshots.
# Each completed tool call → len(_REMAINING_FRACS) training rows.
_REMAINING_FRACS: list[float] = [0.0, 0.10, 0.25, 0.50, 0.75]


# ─────────────────────────── feature engineering ─────────────────────────────

def _kw(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return int(any(kw in lower for kw in keywords))


def build_features(df: pd.DataFrame, remaining_mode: bool = False) -> pd.DataFrame:
    """Build numeric feature matrix from tool_call rows (preserves df.index)."""
    feat = pd.DataFrame(index=df.index)

    feat["tool_name"] = df["tool_name"].fillna("unknown").astype(str)
    feat["tool_payload_bytes"] = (
        pd.to_numeric(df["tool_payload_bytes"], errors="coerce").fillna(0)
    )

    cmd = df["tool_command"].fillna("").astype(str)
    feat["command_length"]      = cmd.str.len()
    feat["command_token_count"] = cmd.str.split().str.len().fillna(0)

    feat["command_has_docker"]  = cmd.apply(lambda c: _kw(c, ["docker"]))
    feat["command_has_build"]   = cmd.apply(lambda c: _kw(
        c, ["build", "make", "compile", "cmake", "cargo build", "go build"]))
    feat["command_has_test"]    = cmd.apply(lambda c: _kw(
        c, ["test", "pytest", "run-tests", "unittest",
            "cargo test", "go test", "npm test"]))
    feat["command_has_install"] = cmd.apply(lambda c: _kw(
        c, ["install", "pip ", "npm ", "apt ", "yarn ", "cargo install"]))
    feat["command_has_git"]     = cmd.apply(lambda c: _kw(c, ["git "]))
    feat["command_has_network"] = cmd.apply(lambda c: _kw(
        c, ["curl ", "wget ", "http://", "https://", "ssh ", "requests."]))
    feat["command_has_python"]  = cmd.apply(lambda c: _kw(
        c, ["python ", "python3 ", "python -", "python3 -"]))

    feat["command_pipe_count"]      = cmd.str.count(r"\|")
    feat["command_semicolon_count"] = cmd.str.count(r";")
    feat["command_redirect_count"]  = cmd.str.count(r">")

    feat["phase_seq"] = pd.to_numeric(df["phase_seq"], errors="coerce").fillna(0)

    # Filled later by enrich_sequential_features
    feat["cumulative_reasoning_s"] = 0.0
    feat["prior_tool_calls"]       = 0
    feat["prior_avg_tool_s"]       = 0.0
    feat["prior_tool_max_s"]       = 0.0
    feat["prior_tool_std_s"]       = 0.0

    if remaining_mode:
        # Set to 0 here; overwritten per-snapshot in augment_remaining()
        feat["elapsed_s"] = 0.0

    return feat


def enrich_sequential_features(full_df: pd.DataFrame,
                                feat: pd.DataFrame) -> pd.DataFrame:
    """
    Fill per-agent rolling context: cumulative reasoning time,
    count / mean / max / std of prior tool-call durations.
    """
    feat = feat.copy()

    for _agent, group in full_df.groupby("agent_id"):
        cum_reasoning = 0.0
        tool_durs: list[float] = []

        for idx, row in group.iterrows():
            if row["phase"] == "reasoning":
                cum_reasoning += float(row.get("duration_s") or 0)
            elif row["phase"] == "tool_call" and idx in feat.index:
                n = len(tool_durs)
                feat.at[idx, "cumulative_reasoning_s"] = cum_reasoning
                feat.at[idx, "prior_tool_calls"]       = n
                feat.at[idx, "prior_avg_tool_s"] = float(np.mean(tool_durs)) if n > 0 else 0.0
                feat.at[idx, "prior_tool_max_s"] = float(max(tool_durs))     if n > 0 else 0.0
                feat.at[idx, "prior_tool_std_s"] = float(np.std(tool_durs))  if n > 1 else 0.0
                tool_durs.append(float(row.get("duration_s") or 0))

    return feat


# ─────────────────────── remaining-time augmentation ─────────────────────────

def augment_remaining(
    tool_df: pd.DataFrame,
    feat: pd.DataFrame,
    fracs: list[float],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Expand each tool_call row into multiple training samples at different
    elapsed-time fractions.

    For a call with total duration D and snapshot fraction f:
        elapsed_s   = D * f
        remaining_s = max(0, D - elapsed_s)

    Returns X_aug, y_aug (remaining_s), tool_names_aug — all reset-indexed.
    """
    X_parts, y_parts, tn_parts = [], [], []
    durs = tool_df["duration_s"].astype(float).values
    tn   = tool_df["tool_name"].fillna("unknown").values

    for frac in fracs:
        snap = feat.copy()
        elapsed = durs * frac
        snap["elapsed_s"] = elapsed
        remaining = np.maximum(0.0, durs - elapsed)
        X_parts.append(snap)
        y_parts.append(remaining)
        tn_parts.append(tn)

    X_aug  = pd.concat(X_parts, ignore_index=True)
    y_aug  = pd.Series(np.concatenate(y_parts), name="remaining_s")
    tn_aug = pd.Series(np.concatenate(tn_parts), name="tool_name")
    return X_aug, y_aug, tn_aug


# ─────────────────────────── target transforms ───────────────────────────────

def log_transform(y: pd.Series) -> pd.Series:
    return np.log1p(y).reset_index(drop=True)


def inv_transform(arr: np.ndarray) -> np.ndarray:
    """Inverse of log1p, clamped to ≥ 0."""
    return np.maximum(0.0, np.expm1(arr))


# ─────────────────────────── data loading ────────────────────────────────────

def load_traces(trace_dir: Path) -> pd.DataFrame:
    csv_files = sorted(trace_dir.rglob("*_trace.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No *_trace.csv files found under {trace_dir}")

    frames = []
    for p in csv_files:
        try:
            frames.append(pd.read_csv(p, dtype=str))
        except Exception as exc:
            print(f"WARNING: skipping {p}: {exc}", file=sys.stderr)
    if not frames:
        raise ValueError("All trace CSVs failed to load")

    combined = pd.concat(frames, ignore_index=True)
    for col in ("phase_seq", "duration_s", "tool_payload_bytes"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    print(f"Loaded {len(combined)} phase records from {len(csv_files)} file(s)")
    return combined


def prepare_dataset(
    trace_dir: Path,
    remaining_mode: bool = False,
    log_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns
    -------
    X          – feature matrix (float, one-hot encoded), RangeIndex
    y          – target (log1p if log_target, else raw), RangeIndex
    tool_df    – raw tool_call rows (for output CSV alignment)
    full_df    – entire trace
    groups     – agent_id per X row (for GroupKFold / group split)
    tool_names – tool name per X row (for per-tool breakdown)
    """
    full_df = load_traces(trace_dir)
    tool_df = full_df.loc[full_df["phase"] == "tool_call"].copy()

    if tool_df.empty:
        raise ValueError("No tool_call rows found in traces")
    tool_df = tool_df.dropna(subset=["duration_s"])

    feat = build_features(tool_df, remaining_mode=remaining_mode)
    feat = enrich_sequential_features(full_df, feat)

    if remaining_mode:
        X, y_raw, tool_names = augment_remaining(tool_df, feat, _REMAINING_FRACS)
        # tile agent_ids to match the augmented order (frac0_rows, frac1_rows, ...)
        groups = pd.Series(
            np.tile(tool_df["agent_id"].fillna("unknown").values, len(_REMAINING_FRACS)),
            name="agent_id",
        )
    else:
        X          = feat.copy().reset_index(drop=True)
        y_raw      = tool_df["duration_s"].astype(float).reset_index(drop=True)
        tool_names = tool_df["tool_name"].fillna("unknown").reset_index(drop=True)
        groups     = tool_df["agent_id"].fillna("unknown").reset_index(drop=True)

    # One-hot encode tool_name
    X = pd.get_dummies(X, columns=["tool_name"], prefix="tool", dtype=float)
    X = X.astype(float)

    y = log_transform(y_raw) if log_target else y_raw.reset_index(drop=True)

    mode_lbl = "remaining_s" if remaining_mode else "duration_s"
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features  "
          f"[target={mode_lbl}, log={log_target}]")
    return X, y, tool_df, full_df, groups, tool_names


# ─────────────────────────── model building ──────────────────────────────────

def build_model(name: str) -> Pipeline:
    """Return an unfitted scikit-learn Pipeline for the requested model."""
    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ])
    if name == "rf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )),
        ])
    if name == "hgb":
        # No StandardScaler needed; HGB handles mixed scales and NaN natively.
        return Pipeline([
            ("model", HistGradientBoostingRegressor(
                max_iter=500,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=42,
            )),
        ])
    raise ValueError(f"Unknown model: {name!r}.  Use 'ridge', 'rf', or 'hgb'.")


def build_quantile_pipelines() -> dict[str, Pipeline]:
    """Three GradientBoostingRegressor pipelines for P10, P50, P90."""
    return {
        label: Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                loss="quantile", alpha=alpha,
                n_estimators=300, max_depth=5,
                learning_rate=0.05, min_samples_leaf=10,
                random_state=42,
            )),
        ])
        for alpha, label in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]
    }


# ─────────────────────────── evaluation ──────────────────────────────────────

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tool_names: Optional[pd.Series] = None,
    label: str = "test",
) -> dict:
    """Compute and print regression metrics; optionally per-tool breakdown."""
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    medae = median_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    nz    = y_true > 0
    mape  = (np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100
             if nz.any() else float("nan"))

    metrics: dict = {
        "mae":        round(float(mae),   4),
        "rmse":       round(float(rmse),  4),
        "median_ae":  round(float(medae), 4),
        "mape_pct":   round(float(mape),  2),
        "r2":         round(float(r2),    4),
        "n_samples":  int(len(y_true)),
    }

    print(f"\n── {label} metrics ──")
    print(f"  MAE      = {mae:.4f} s")
    print(f"  RMSE     = {rmse:.4f} s")
    print(f"  MedianAE = {medae:.4f} s")
    print(f"  MAPE     = {mape:.2f}%")
    print(f"  R²       = {r2:.4f}")
    print(f"  n        = {len(y_true)}")

    if tool_names is not None and len(tool_names) == len(y_true):
        print(f"\n── Per-tool breakdown ({label}) ──")
        breakdown: dict = {}
        for tname in sorted(tool_names.unique()):
            mask = tool_names.values == tname
            if mask.sum() < 2:
                continue
            t_mae  = mean_absolute_error(y_true[mask], y_pred[mask])
            t_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
            n = int(mask.sum())
            print(f"  {tname:20s}  MAE={t_mae:.3f}s  RMSE={t_rmse:.3f}s  n={n}")
            breakdown[tname] = {"mae": round(t_mae, 4), "rmse": round(t_rmse, 4), "n": n}
        metrics["per_tool"] = breakdown

    return metrics


def cross_validate(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    log_target: bool,
) -> dict:
    """GroupKFold CV; agents never straddle train/test folds."""
    n_unique = groups.nunique()
    if n_unique < n_splits:
        print(f"  WARNING: only {n_unique} agents — reducing CV folds to {n_unique}")
        n_splits = max(2, n_unique)

    gkf = GroupKFold(n_splits=n_splits)
    maes, r2s = [], []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        p = build_model(model_name)
        p.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = p.predict(X.iloc[te_idx])
        y_te = inv_transform(y.iloc[te_idx].values) if log_target else y.iloc[te_idx].values
        y_pr = inv_transform(pred) if log_target else np.maximum(0.0, pred)
        maes.append(mean_absolute_error(y_te, y_pr))
        r2s.append(r2_score(y_te, y_pr))
        print(f"  Fold {fold+1}/{n_splits}: MAE={maes[-1]:.4f}s  R²={r2s[-1]:.4f}")

    result = {
        "cv_mae_mean": round(float(np.mean(maes)), 4),
        "cv_mae_std":  round(float(np.std(maes)),  4),
        "cv_r2_mean":  round(float(np.mean(r2s)),  4),
        "cv_r2_std":   round(float(np.std(r2s)),   4),
    }
    print(f"  → CV MAE {result['cv_mae_mean']:.4f} ± {result['cv_mae_std']:.4f}")
    print(f"  → CV R²  {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}")
    return result


# ─────────────────────────── feature importance ──────────────────────────────

def print_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> None:
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


# ─────────────────────────── group-based split ───────────────────────────────

def group_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    tool_names: pd.Series,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split by unique agent_id values so no agent's rows appear in both
    train and test.  Falls back to random split if there is only one agent.
    """
    unique = groups.unique()

    if len(unique) < 2:
        print("WARNING: only one agent group — using random row split", file=sys.stderr)
        n = len(X)
        split = int(n * (1 - test_fraction))
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        tr_idx, te_idx = idx[:split], idx[split:]
    else:
        rng = np.random.default_rng(seed)
        n_test = max(1, int(len(unique) * test_fraction))
        test_agents = set(rng.choice(unique, size=n_test, replace=False))
        te_mask = groups.isin(test_agents).values
        tr_idx  = np.where(~te_mask)[0]
        te_idx  = np.where( te_mask)[0]

    return (
        X.iloc[tr_idx].reset_index(drop=True),
        X.iloc[te_idx].reset_index(drop=True),
        y.iloc[tr_idx].reset_index(drop=True),
        y.iloc[te_idx].reset_index(drop=True),
        groups.iloc[tr_idx].reset_index(drop=True),
        tool_names.iloc[te_idx].reset_index(drop=True),
    )


# ──────────────────────────────── CLI ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict tool-call duration or remaining time from agent traces.")
    p.add_argument("--trace-dir", required=True, type=Path,
                   help="Directory with *_trace.csv files (searched recursively).")
    p.add_argument("--model", default="hgb", choices=["ridge", "rf", "hgb"],
                   help="Model type (default: hgb = HistGradientBoosting).")
    p.add_argument("--test-fraction", type=float, default=0.2,
                   help="Fraction of *agents* held out for testing (default: 0.2).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv", type=int, default=0, metavar="N",
                   help="GroupKFold cross-validation folds (0 = skip).")
    p.add_argument("--remaining", action="store_true",
                   help=f"Predict remaining time instead of total duration.  "
                        f"Each tool call is expanded into {len(_REMAINING_FRACS)} "
                        f"elapsed-fraction snapshots: {_REMAINING_FRACS}.")
    p.add_argument("--no-log-target", action="store_true",
                   help="Disable log1p transform on the target (not recommended).")
    p.add_argument("--quantiles", action="store_true",
                   help="Also train P10 / P50 / P90 quantile models and "
                        "include their predictions in --output-predictions.")
    p.add_argument("--save-model", type=Path, default=None,
                   help="Save trained point-estimate model to this path (.joblib).")
    p.add_argument("--load-model", type=Path, default=None,
                   help="Load a pre-trained model; skip training.")
    p.add_argument("--predict-only", action="store_true",
                   help="Skip training; requires --load-model.")
    p.add_argument("--output-predictions", type=Path, default=None,
                   help="Write predictions CSV to this path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_target = not args.no_log_target

    X, y, tool_df, full_df, groups, tool_names = prepare_dataset(
        args.trace_dir,
        remaining_mode=args.remaining,
        log_target=log_target,
    )
    feature_names = list(X.columns)

    # ── inference-only mode ──────────────────────────────────────────────────
    if args.predict_only:
        if args.load_model is None:
            print("ERROR: --predict-only requires --load-model", file=sys.stderr)
            return 2
        pipeline = joblib.load(args.load_model)
        print(f"Loaded model from {args.load_model}")

        train_cols = getattr(pipeline, "_trained_feature_names", None)
        if train_cols is not None:
            for col in train_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[train_cols]

        pred_log = pipeline.predict(X)
        y_pred = inv_transform(pred_log) if log_target else np.maximum(0.0, pred_log)
        y_true = inv_transform(y.values) if log_target else y.values
        evaluate(y_true, y_pred, tool_names)

        if args.output_predictions:
            out = X.copy()
            pred_col = "predicted_remaining_s" if args.remaining else "predicted_duration_s"
            out[pred_col] = y_pred
            out.to_csv(args.output_predictions, index=False)
            print(f"\nPredictions written → {args.output_predictions}")
        return 0

    # ── group-based train / test split ───────────────────────────────────────
    X_tr, X_te, y_tr, y_te, g_tr, tn_te = group_train_test_split(
        X, y, groups, tool_names, args.test_fraction, args.seed)

    print(f"\nSplit (by agent): {len(X_tr)} train / {len(X_te)} test")
    print(f"Model: {args.model}  |  log-target: {log_target}  |  remaining: {args.remaining}")

    # ── optional cross-validation ─────────────────────────────────────────────
    all_metrics: dict = {}
    if args.cv > 0:
        print(f"\n── {args.cv}-fold GroupKFold CV ──")
        all_metrics["cv"] = cross_validate(
            args.model, X, y, groups, args.cv, log_target)

    # ── train point-estimate model ────────────────────────────────────────────
    pipeline = build_model(args.model)
    pipeline.fit(X_tr, y_tr)
    pipeline._trained_feature_names = feature_names  # for inference alignment

    # ── evaluate point estimates ──────────────────────────────────────────────
    pred_log  = pipeline.predict(X_te)
    y_pred_te = inv_transform(pred_log) if log_target else np.maximum(0.0, pred_log)
    y_true_te = inv_transform(y_te.values) if log_target else y_te.values

    all_metrics["test"] = evaluate(y_true_te, y_pred_te, tn_te)
    print_feature_importance(pipeline, feature_names)

    # ── optional quantile models (P10 / P50 / P90) ───────────────────────────
    q_preds_te: dict[str, np.ndarray] = {}
    if args.quantiles:
        print("\n── Training quantile models (P10 / P50 / P90) ──")
        for label, qp in build_quantile_pipelines().items():
            qp.fit(X_tr, y_tr)
            q_raw = qp.predict(X_te)
            q_preds_te[label] = inv_transform(q_raw) if log_target else np.maximum(0.0, q_raw)
            print(f"  {label}: median={np.median(q_preds_te[label]):.3f}s  "
                  f"p90={np.percentile(q_preds_te[label], 90):.3f}s")

    # ── save model ────────────────────────────────────────────────────────────
    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, args.save_model)
        print(f"\nModel saved → {args.save_model}")

    # ── write predictions CSV ─────────────────────────────────────────────────
    if args.output_predictions:
        out = X_te.copy()
        pred_col  = "predicted_remaining_s" if args.remaining else "predicted_duration_s"
        truth_col = "actual_remaining_s"    if args.remaining else "actual_duration_s"
        out[pred_col]  = y_pred_te
        out[truth_col] = y_true_te
        for label, arr in q_preds_te.items():
            out[f"{label}_s"] = arr
        out.to_csv(args.output_predictions, index=False)
        print(f"Predictions written → {args.output_predictions}")

    # ── write metrics JSON ────────────────────────────────────────────────────
    metrics_path = args.trace_dir / "prediction_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics written → {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
