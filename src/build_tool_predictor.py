#!/usr/bin/env python3
"""
build_tool_predictor.py  (v2)
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
    python src/build_tool_predictor.py --trace-dir ./abc_results

    # Remaining-time prediction with 5-fold CV:
    python src/build_tool_predictor.py \\
        --trace-dir ./abc_results --remaining --cv 5

    # Quantile bands (P10/P50/P90) saved to CSV:
    python src/build_tool_predictor.py \\
        --trace-dir ./abc_results --quantiles --output-predictions preds.csv

    # Save trained model:
    python src/build_tool_predictor.py \\
        --trace-dir ./abc_results --model hgb --save-model model.joblib

    # Inference on new traces:
    python src/build_tool_predictor.py \\
        --trace-dir ./new_results --load-model model.joblib --predict-only

Install:
    pip install scikit-learn pandas numpy joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import HashingVectorizer
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

# Calls at or above this duration get extra training weight.  The model still
# predicts all calls, but this nudges fitting toward the high-impact tail.
LONG_CALL_THRESHOLD_S: float = 2.0
LONG_CALL_WEIGHT: float = 3.0


# ─────────────────────────── feature engineering ─────────────────────────────

# Top file-extensions we one-hot for FileEditor `path` arguments.
_TOP_EXTS: tuple[str, ...] = ("py", "js", "ts", "rs", "go", "md")

# Text-hash features over the action's natural-language signal:
# summary + tool_command + path + sub-command. Stateless (no fitting), so
# safe to call at both train and inference time without leakage.
_TEXT_HASH_DIM: int = 64
_TEXT_HASHER = HashingVectorizer(
    n_features=_TEXT_HASH_DIM,
    analyzer="char_wb",
    ngram_range=(3, 5),
    alternate_sign=False,
    norm=None,
    lowercase=True,
)


def _kw(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return int(any(kw in lower for kw in keywords))


def _parse_args(s: str) -> dict:
    """Best-effort decode of the tool_args_json column. Returns {} on failure."""
    if not isinstance(s, str) or not s:
        return {}
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _path_ext(path: str) -> str:
    if not isinstance(path, str) or not path:
        return ""
    return Path(path).suffix.lstrip(".").lower()


def _view_range_size(d: dict) -> int:
    vr = d.get("view_range") if isinstance(d, dict) else None
    if isinstance(vr, (list, tuple)) and len(vr) == 2:
        try:
            return max(0, int(vr[1]) - int(vr[0]))
        except (TypeError, ValueError):
            return 0
    return 0


def _timeout(d: dict) -> float:
    if not isinstance(d, dict):
        return 0.0
    try:
        return float(d.get("timeout", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


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

    feat["phase_seq"] = _num_col(df, "phase_seq", 0)

    # Detailed traces expose these at dispatch time.  They are safe for
    # training because they are captured when the ActionEvent fires.
    feat["start_active_agents"] = _num_col(df, "start_active_agents", 0)
    feat["start_active_tool_calls"] = _num_col(df, "start_active_tool_calls", 0)

    # Filled from detailed traces when available, then refined/fallback-filled
    # by enrich_sequential_features for legacy trace directories.
    feat["cumulative_reasoning_s"] = _num_col(df, "start_cumulative_reasoning_s", 0.0)
    feat["prior_tool_calls"]       = 0
    feat["prior_avg_tool_s"]       = 0.0
    feat["prior_tool_max_s"]       = 0.0
    feat["prior_tool_std_s"]       = 0.0

    # ── Structured action-args features (mine tool_args_json) ────────────────
    if "tool_args_json" in df.columns:
        args_series = df["tool_args_json"].fillna("").astype(str).map(_parse_args)
    else:
        args_series = pd.Series([{}] * len(df), index=df.index)

    tn = feat["tool_name"]
    feat["is_terminal"]     = (tn == "terminal").astype(int)
    feat["is_file_editor"]  = (tn == "file_editor").astype(int)
    feat["is_task_tracker"] = (tn == "task_tracker").astype(int)

    feat["args_key_count"] = args_series.map(len).astype(int)
    feat["args_total_str_bytes"] = args_series.map(
        lambda d: sum(len(str(v)) for v in d.values())
    ).astype(int)

    # FileEditor sub-command (zero unless tool is file_editor)
    sub_cmd = args_series.map(lambda d: str(d.get("command", "")))
    is_fe   = feat["is_file_editor"].astype(bool)
    feat["editor_cmd_view"]        = (is_fe & (sub_cmd == "view")).astype(int)
    feat["editor_cmd_create"]      = (is_fe & (sub_cmd == "create")).astype(int)
    feat["editor_cmd_str_replace"] = (
        is_fe & sub_cmd.isin(["str_replace", "str_replace_based_edit_tool"])
    ).astype(int)
    feat["editor_cmd_insert"]      = (is_fe & (sub_cmd == "insert")).astype(int)

    # Path extension one-hot (file_editor only)
    ed_ext = args_series.map(lambda d: _path_ext(str(d.get("path", ""))))
    for ext in _TOP_EXTS:
        feat[f"path_ext_{ext}"] = (is_fe & (ed_ext == ext)).astype(int)
    feat["path_ext_other"] = (
        is_fe & ed_ext.ne("") & ~ed_ext.isin(list(_TOP_EXTS))
    ).astype(int)

    # Edit-size signals
    feat["old_str_len"]     = args_series.map(
        lambda d: len(str(d.get("old_str", "")))
    ).astype(int)
    feat["new_str_len"]     = args_series.map(
        lambda d: len(str(d.get("new_str", "")))
    ).astype(int)
    feat["edit_size_delta"] = (feat["new_str_len"] - feat["old_str_len"]).abs()
    feat["file_text_len"]   = args_series.map(
        lambda d: len(str(d.get("file_text", "")))
    ).astype(int)
    feat["view_range_size"] = args_series.map(_view_range_size).astype(int)

    # Terminal-specific
    is_term = feat["is_terminal"].astype(bool)
    feat["terminal_timeout_s"] = (args_series.map(_timeout) * is_term).astype(float)

    # TaskTracker sub-command
    is_tt = feat["is_task_tracker"].astype(bool)
    feat["tracker_cmd_add"]      = (
        is_tt & sub_cmd.isin(["add_task", "add"])
    ).astype(int)
    feat["tracker_cmd_list"]     = (
        is_tt & sub_cmd.isin(["list_tasks", "view"])
    ).astype(int)
    feat["tracker_cmd_complete"] = (
        is_tt & sub_cmd.isin(["mark_complete", "complete"])
    ).astype(int)

    # ── Text-hash features over summary + command + path + sub-cmd ──────────
    # Captures the free-form action description ("View handler.js to ...",
    # "$ ls -R", etc.) that repeats across runs and predicts duration.
    summaries = (
        df.get("action_summary", pd.Series("", index=df.index))
          .fillna("").astype(str)
        + " | "
        + df.get("detail", pd.Series("", index=df.index))
          .fillna("").astype(str)
    )
    paths = args_series.map(lambda d: str(d.get("path", "")))
    text_blob = (
        summaries + " | " + cmd + " | " + sub_cmd + " | " + paths
    ).tolist()
    if len(text_blob) > 0:
        hash_matrix = _TEXT_HASHER.transform(text_blob).toarray()
    else:
        hash_matrix = np.zeros((0, _TEXT_HASH_DIM))
    hash_df = pd.DataFrame(
        hash_matrix,
        index=df.index,
        columns=[f"text_hash_{i}" for i in range(_TEXT_HASH_DIM)],
    )
    feat = pd.concat([feat, hash_df], axis=1)

    if remaining_mode:
        # Set to 0 here; overwritten per-snapshot in augment_remaining()
        feat["elapsed_s"] = 0.0

    return feat


def enrich_sequential_features(
    full_df: pd.DataFrame,
    feat: pd.DataFrame,
    tool_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Fill per-agent rolling context: cumulative reasoning time,
    count / mean / max / std of prior tool-call durations.
    """
    feat = feat.copy()
    if tool_df is None:
        tool_df = full_df.loc[full_df["phase"] == "tool_call"].copy()

    # Detailed tool-call CSVs have their own RangeIndex, so match to rows by
    # stable dispatch-time keys and fall back to per-agent order.
    key_to_feat_idxs: dict[tuple[str, str], list[Any]] = {}
    order_by_agent: dict[str, list[Any]] = {}
    for idx, row in tool_df.iterrows():
        agent = str(row.get("agent_id", ""))
        start = str(row.get("start_ts", ""))
        key_to_feat_idxs.setdefault((agent, start), []).append(idx)
        order_by_agent.setdefault(agent, []).append(idx)
    used: set[Any] = set()

    full_sorted = full_df.copy()
    has_reasoning_rows = (
        "phase" in full_sorted.columns
        and full_sorted["phase"].astype(str).eq("reasoning").any()
    )
    if "phase_seq" in full_sorted.columns:
        full_sorted["_phase_seq_sort"] = pd.to_numeric(
            full_sorted["phase_seq"], errors="coerce"
        ).fillna(0)
        full_sorted = full_sorted.sort_values(["agent_id", "_phase_seq_sort"])

    for agent, group in full_sorted.groupby("agent_id", sort=False):
        cum_reasoning = 0.0
        tool_durs: list[float] = []

        for idx, row in group.iterrows():
            if row["phase"] == "reasoning":
                dur = float(row.get("duration_s") or 0)
                cum_reasoning += dur if np.isfinite(dur) else 0.0
            elif row["phase"] == "tool_call":
                key = (str(row.get("agent_id", "")), str(row.get("start_ts", "")))
                feat_idx = None
                candidates = key_to_feat_idxs.get(key, [])
                while candidates and candidates[0] in used:
                    candidates.pop(0)
                if candidates:
                    feat_idx = candidates.pop(0)
                else:
                    queue = order_by_agent.get(str(agent), [])
                    while queue and queue[0] in used:
                        queue.pop(0)
                    if queue:
                        feat_idx = queue.pop(0)

                if feat_idx is not None and feat_idx in feat.index:
                    used.add(feat_idx)
                    n = len(tool_durs)
                    if has_reasoning_rows:
                        feat.at[feat_idx, "cumulative_reasoning_s"] = cum_reasoning
                    feat.at[feat_idx, "prior_tool_calls"]       = n
                    feat.at[feat_idx, "prior_avg_tool_s"] = float(np.mean(tool_durs)) if n > 0 else 0.0
                    feat.at[feat_idx, "prior_tool_max_s"] = float(max(tool_durs))     if n > 0 else 0.0
                    feat.at[feat_idx, "prior_tool_std_s"] = float(np.std(tool_durs))  if n > 1 else 0.0
                dur = float(row.get("duration_s") or 0)
                tool_durs.append(dur if np.isfinite(dur) else 0.0)

    return feat


# ─────────────────────── remaining-time augmentation ─────────────────────────

def augment_remaining(
    tool_df: pd.DataFrame,
    feat: pd.DataFrame,
    fracs: list[float],
    long_threshold: float = LONG_CALL_THRESHOLD_S,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Expand each tool_call row into multiple training samples at different
    elapsed-time fractions.

    For a call with total duration D and snapshot fraction f:
        elapsed_s   = D * f
        remaining_s = max(0, D - elapsed_s) if D >= long_threshold else 0

    Short calls (D < long_threshold) get target 0 across all snapshots so
    the predictor learns to output 0 for them.

    Returns X_aug, y_aug (remaining_s), tool_names_aug — all reset-indexed.
    """
    X_parts, y_parts, tn_parts = [], [], []
    durs = tool_df["duration_s"].astype(float).values
    tn   = tool_df["tool_name"].fillna("unknown").values
    is_short = durs < long_threshold

    for frac in fracs:
        snap = feat.copy()
        elapsed = durs * frac
        snap["elapsed_s"] = elapsed
        remaining = np.where(is_short, 0.0, np.maximum(0.0, durs - elapsed))
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


def clamp_short(y: np.ndarray, threshold: float) -> np.ndarray:
    """Force predictions below ``threshold`` to exactly 0 (short-call mask)."""
    out = np.maximum(0.0, np.asarray(y, dtype=float))
    out[out < threshold] = 0.0
    return out


def build_sample_weights(
    tool_df: pd.DataFrame,
    remaining_mode: bool = False,
    long_threshold_s: float = LONG_CALL_THRESHOLD_S,
    long_weight: float = LONG_CALL_WEIGHT,
) -> pd.Series:
    """Weight long source calls more heavily while keeping all samples."""
    durs = pd.to_numeric(tool_df["duration_s"], errors="coerce").fillna(0).values
    if remaining_mode:
        durs = np.tile(durs, len(_REMAINING_FRACS))
    weights = np.where(durs >= long_threshold_s, long_weight, 1.0)
    return pd.Series(weights, name="sample_weight")


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
    if "phase" not in combined.columns:
        combined["phase"] = "tool_call"
    else:
        combined["phase"] = combined["phase"].fillna("tool_call")
    for col in (
        "phase_seq",
        "tool_seq",
        "duration_s",
        "tool_payload_bytes",
        "start_active_agents",
        "start_active_tool_calls",
        "start_cumulative_reasoning_s",
    ):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    print(f"Loaded {len(combined)} trace records from {len(csv_files)} file(s)")
    return combined


def _normalise_tool_call_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Make detailed and legacy tool-call rows share a common shape."""
    df = df.copy()
    if "phase" not in df.columns:
        df["phase"] = "tool_call"
    else:
        df["phase"] = df["phase"].fillna("tool_call")
    if "phase_seq" not in df.columns:
        df["phase_seq"] = df.get("tool_seq", 0)
    if "detail" not in df.columns:
        df["detail"] = df.get("action_summary", "")
    else:
        df["detail"] = df["detail"].fillna(df.get("action_summary", ""))
    if "tool_args_json" not in df.columns:
        df["tool_args_json"] = "{}"
    if "tool_payload_bytes" not in df.columns:
        df["tool_payload_bytes"] = df["tool_args_json"].fillna("{}").astype(str).map(
            lambda s: len(s.encode("utf-8"))
        )
    for col in (
        "phase_seq",
        "tool_seq",
        "duration_s",
        "tool_payload_bytes",
        "start_active_agents",
        "start_active_tool_calls",
        "start_cumulative_reasoning_s",
        "observation_bytes",
        "observation_line_count",
        "returncode",
        "timed_out",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_detailed_tool_calls(trace_dir: Path) -> pd.DataFrame:
    """Load rich collector artifacts, preferring CSV over JSONL."""
    csv_files = sorted(trace_dir.rglob("*_tool_calls.csv"))
    frames = []
    for p in csv_files:
        try:
            frames.append(pd.read_csv(p, dtype=str))
        except Exception as exc:
            print(f"WARNING: skipping {p}: {exc}", file=sys.stderr)

    if not frames:
        jsonl_files = sorted(trace_dir.rglob("*_tool_calls.jsonl"))
        rows: list[dict[str, Any]] = []
        for p in jsonl_files:
            try:
                with p.open("r", encoding="utf-8") as f:
                    rows.extend(json.loads(line) for line in f if line.strip())
            except Exception as exc:
                print(f"WARNING: skipping {p}: {exc}", file=sys.stderr)
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame()

    combined = _normalise_tool_call_frame(pd.concat(frames, ignore_index=True))
    print(f"Loaded {len(combined)} detailed tool-call records")
    return combined


def _tool_key_frame(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in ("agent_id", "start_ts", "end_ts"):
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
        else:
            parts.append(pd.Series("", index=df.index))
    return parts[0] + "\0" + parts[1] + "\0" + parts[2]


def load_tool_calls(trace_dir: Path, full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return training rows for tool calls.

    New ``*_trace.csv`` files already contain detailed tool-call rows.  Older
    ``*_tool_calls.csv/jsonl`` files are merged when present, preserving
    compatibility with trace directories produced during the transition.
    """
    legacy = _normalise_tool_call_frame(
        full_df.loc[full_df["phase"] == "tool_call"].copy()
    )
    detailed = load_detailed_tool_calls(trace_dir)
    if detailed.empty:
        return legacy

    detailed_keys = set(_tool_key_frame(detailed).tolist())
    legacy_missing = legacy.loc[~_tool_key_frame(legacy).isin(detailed_keys)]
    if not legacy_missing.empty:
        combined = pd.concat([detailed, legacy_missing], ignore_index=True, sort=False)
    else:
        combined = detailed
    return _normalise_tool_call_frame(combined)


def prepare_dataset(
    trace_dir: Path,
    remaining_mode: bool = False,
    log_target: bool = True,
    long_threshold: float = LONG_CALL_THRESHOLD_S,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns
    -------
    X          – feature matrix (float, one-hot encoded), RangeIndex
    y          – target (log1p if log_target, else raw), RangeIndex.
                 Calls whose original ``duration_s < long_threshold`` get
                 target 0 (long-call-only training).
    tool_df    – raw tool_call rows (for output CSV alignment)
    full_df    – entire trace
    groups     – agent_id per X row (for GroupKFold / group split)
    tool_names – tool name per X row (for per-tool breakdown)
    """
    full_df = load_traces(trace_dir)
    tool_df = load_tool_calls(trace_dir, full_df)

    if tool_df.empty:
        raise ValueError("No tool_call rows found in traces")
    tool_df = tool_df.dropna(subset=["duration_s"])

    feat = build_features(tool_df, remaining_mode=remaining_mode)
    feat = enrich_sequential_features(full_df, feat, tool_df)

    if remaining_mode:
        X, y_raw, tool_names = augment_remaining(
            tool_df, feat, _REMAINING_FRACS, long_threshold=long_threshold
        )
        # tile agent_ids to match the augmented order (frac0_rows, frac1_rows, ...)
        groups = pd.Series(
            np.tile(tool_df["agent_id"].fillna("unknown").values, len(_REMAINING_FRACS)),
            name="agent_id",
        )
    else:
        X          = feat.copy().reset_index(drop=True)
        durs       = tool_df["duration_s"].astype(float).reset_index(drop=True)
        y_raw      = durs.where(durs >= long_threshold, 0.0)
        tool_names = tool_df["tool_name"].fillna("unknown").reset_index(drop=True)
        groups     = tool_df["agent_id"].fillna("unknown").reset_index(drop=True)

    # One-hot encode tool_name
    X = pd.get_dummies(X, columns=["tool_name"], prefix="tool", dtype=float)
    X = X.astype(float)

    y = log_transform(y_raw) if log_target else y_raw.reset_index(drop=True)

    mode_lbl = "remaining_s" if remaining_mode else "duration_s"
    n_long = int((y_raw > 0).sum())
    print(
        f"Dataset: {len(X)} samples, {X.shape[1]} features  "
        f"[target={mode_lbl}, log={log_target}, long_threshold={long_threshold:.2f}s, "
        f"long_targets={n_long}/{len(y_raw)}]"
    )
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


def fit_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
) -> Pipeline:
    """Fit a pipeline, passing weights to the final estimator when provided."""
    if sample_weight is None:
        pipeline.fit(X, y)
        return pipeline
    try:
        pipeline.fit(X, y, model__sample_weight=sample_weight.values)
    except TypeError:
        pipeline.fit(X, y)
    return pipeline


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
    long_threshold: float = LONG_CALL_THRESHOLD_S,
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

    long_mask = y_true >= long_threshold
    if long_mask.any():
        long_mae = mean_absolute_error(y_true[long_mask], y_pred[long_mask])
        long_rmse = np.sqrt(mean_squared_error(y_true[long_mask], y_pred[long_mask]))
        metrics["long_calls"] = {
            "threshold_s": long_threshold,
            "mae": round(float(long_mae), 4),
            "rmse": round(float(long_rmse), 4),
            "n": int(long_mask.sum()),
        }
        print(
            f"  Long-call MAE/RMSE (>= {long_threshold:.1f}s) "
            f"= {long_mae:.4f}s / {long_rmse:.4f}s  n={int(long_mask.sum())}"
        )

    short_mask = ~long_mask
    if short_mask.any():
        short_zero = float(np.mean(y_pred[short_mask] == 0.0))
        metrics["short_calls"] = {
            "threshold_s": long_threshold,
            "n": int(short_mask.sum()),
            "frac_predicted_zero": round(short_zero, 4),
        }
        print(
            f"  Short-call zero-rate (< {long_threshold:.1f}s) "
            f"= {short_zero * 100:.1f}%  n={int(short_mask.sum())}"
        )

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
    sample_weights: Optional[pd.Series] = None,
    long_threshold: float = LONG_CALL_THRESHOLD_S,
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
        w_tr = sample_weights.iloc[tr_idx].reset_index(drop=True) if sample_weights is not None else None
        fit_pipeline(p, X.iloc[tr_idx], y.iloc[tr_idx], w_tr)
        pred = p.predict(X.iloc[te_idx])
        y_te = inv_transform(y.iloc[te_idx].values) if log_target else y.iloc[te_idx].values
        y_pr_raw = inv_transform(pred) if log_target else pred
        y_pr = clamp_short(y_pr_raw, long_threshold)
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

def group_train_test_indices(
    groups: pd.Series,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique = groups.unique()

    if len(unique) < 2:
        print("WARNING: only one agent group — using random row split", file=sys.stderr)
        n = len(groups)
        split = int(n * (1 - test_fraction))
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        return idx[:split], idx[split:]

    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(unique) * test_fraction))
    test_agents = set(rng.choice(unique, size=n_test, replace=False))
    te_mask = groups.isin(test_agents).values
    return np.where(~te_mask)[0], np.where(te_mask)[0]


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
    tr_idx, te_idx = group_train_test_indices(groups, test_fraction, seed)

    return (
        X.iloc[tr_idx].reset_index(drop=True),
        X.iloc[te_idx].reset_index(drop=True),
        y.iloc[tr_idx].reset_index(drop=True),
        y.iloc[te_idx].reset_index(drop=True),
        groups.iloc[tr_idx].reset_index(drop=True),
        tool_names.iloc[te_idx].reset_index(drop=True),
    )


# ─────────────────────────── realtime inference ──────────────────────────────

class RealtimePredictor:
    """
    Deployment wrapper around a saved remaining-time or duration pipeline.

    Callers pass rows from ``build_features`` before one-hot encoding; the
    wrapper aligns columns to the schema saved with the model.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_names: list[str],
        log_target: bool = True,
        long_threshold: float = LONG_CALL_THRESHOLD_S,
    ) -> None:
        self._pipeline = pipeline
        self._feat_cols = feature_names
        self._log = log_target
        self._long_threshold = float(long_threshold)

    @classmethod
    def load(cls, path: Path, log_target: Optional[bool] = None) -> "RealtimePredictor":
        pipeline = joblib.load(path)
        feature_names = getattr(pipeline, "_trained_feature_names", None)
        if feature_names is None:
            raise ValueError(
                f"Model at {path!r} is missing _trained_feature_names; "
                "save it with build_tool_predictor.py first."
            )
        saved_log = getattr(pipeline, "_log_target", True)
        saved_threshold = getattr(
            pipeline, "_long_call_threshold_s", LONG_CALL_THRESHOLD_S
        )
        return cls(
            pipeline,
            list(feature_names),
            log_target=saved_log if log_target is None else log_target,
            long_threshold=saved_threshold,
        )

    def _prepare(self, feat: pd.DataFrame, elapsed_s: float = 0.0) -> pd.DataFrame:
        x = feat.copy()
        if "elapsed_s" in self._feat_cols:
            x["elapsed_s"] = elapsed_s
        if "tool_name" in x.columns:
            x = pd.get_dummies(x, columns=["tool_name"], prefix="tool", dtype=float)
        x = x.astype(float)
        for col in self._feat_cols:
            if col not in x.columns:
                x[col] = 0.0
        return x[self._feat_cols]

    def predict_remaining(self, feat: pd.DataFrame, elapsed_s: float = 0.0) -> float:
        x = self._prepare(feat, elapsed_s=elapsed_s)
        raw = self._pipeline.predict(x)
        val = inv_transform(raw)[0] if self._log else float(raw[0])
        val = float(max(0.0, val))
        if val < self._long_threshold:
            return 0.0
        return val

    def predict_duration(self, feat: pd.DataFrame) -> float:
        return self.predict_remaining(feat, elapsed_s=0.0)


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
    p.add_argument("--long-threshold", type=float, default=LONG_CALL_THRESHOLD_S,
                   help=f"Tool calls with duration < this many seconds are "
                        f"treated as short: their training target is set to 0 "
                        f"and predictions below this value are clamped to 0 at "
                        f"inference. Default: {LONG_CALL_THRESHOLD_S}.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_target = not args.no_log_target
    long_threshold = float(args.long_threshold)

    X, y, tool_df, full_df, groups, tool_names = prepare_dataset(
        args.trace_dir,
        remaining_mode=args.remaining,
        log_target=log_target,
        long_threshold=long_threshold,
    )
    feature_names = list(X.columns)
    sample_weights = build_sample_weights(
        tool_df, remaining_mode=args.remaining, long_threshold_s=long_threshold
    )

    # ── inference-only mode ──────────────────────────────────────────────────
    if args.predict_only:
        if args.load_model is None:
            print("ERROR: --predict-only requires --load-model", file=sys.stderr)
            return 2
        pipeline = joblib.load(args.load_model)
        print(f"Loaded model from {args.load_model}")
        log_target = getattr(pipeline, "_log_target", log_target)
        long_threshold = float(
            getattr(pipeline, "_long_call_threshold_s", long_threshold)
        )

        train_cols = getattr(pipeline, "_trained_feature_names", None)
        if train_cols is not None:
            for col in train_cols:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[train_cols]

        pred_log = pipeline.predict(X)
        y_pred_raw = inv_transform(pred_log) if log_target else pred_log
        y_pred = clamp_short(y_pred_raw, long_threshold)
        y_true = inv_transform(y.values) if log_target else y.values
        evaluate(y_true, y_pred, tool_names, long_threshold=long_threshold)

        if args.output_predictions:
            out = X.copy()
            pred_col = "predicted_remaining_s" if args.remaining else "predicted_duration_s"
            out[pred_col] = y_pred
            out.to_csv(args.output_predictions, index=False)
            print(f"\nPredictions written → {args.output_predictions}")
        return 0

    # ── group-based train / test split ───────────────────────────────────────
    tr_idx, te_idx = group_train_test_indices(groups, args.test_fraction, args.seed)
    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[te_idx].reset_index(drop=True)
    y_tr = y.iloc[tr_idx].reset_index(drop=True)
    y_te = y.iloc[te_idx].reset_index(drop=True)
    tn_te = tool_names.iloc[te_idx].reset_index(drop=True)
    w_tr = sample_weights.iloc[tr_idx].reset_index(drop=True)

    print(f"\nSplit (by agent): {len(X_tr)} train / {len(X_te)} test")
    print(
        f"Model: {args.model}  |  log-target: {log_target}  |  remaining: {args.remaining}  "
        f"| long-call weight: >= {long_threshold:.1f}s x{LONG_CALL_WEIGHT:.1f}  "
        f"| short calls (< {long_threshold:.1f}s) → 0"
    )

    # ── optional cross-validation ─────────────────────────────────────────────
    all_metrics: dict = {}
    if args.cv > 0:
        print(f"\n── {args.cv}-fold GroupKFold CV ──")
        all_metrics["cv"] = cross_validate(
            args.model, X, y, groups, args.cv, log_target, sample_weights,
            long_threshold=long_threshold,
        )

    # ── train point-estimate model ────────────────────────────────────────────
    pipeline = build_model(args.model)
    fit_pipeline(pipeline, X_tr, y_tr, w_tr)
    pipeline._trained_feature_names = feature_names  # for inference alignment
    pipeline._log_target = log_target
    pipeline._remaining_mode = args.remaining
    pipeline._long_call_threshold_s = long_threshold

    # ── evaluate point estimates ──────────────────────────────────────────────
    pred_log  = pipeline.predict(X_te)
    y_pred_te_raw = inv_transform(pred_log) if log_target else pred_log
    y_pred_te = clamp_short(y_pred_te_raw, long_threshold)
    y_true_te = inv_transform(y_te.values) if log_target else y_te.values

    all_metrics["test"] = evaluate(
        y_true_te, y_pred_te, tn_te, long_threshold=long_threshold
    )
    print_feature_importance(pipeline, feature_names)

    # ── optional quantile models (P10 / P50 / P90) ───────────────────────────
    q_preds_te: dict[str, np.ndarray] = {}
    if args.quantiles:
        print("\n── Training quantile models (P10 / P50 / P90) ──")
        for label, qp in build_quantile_pipelines().items():
            fit_pipeline(qp, X_tr, y_tr, w_tr)
            q_raw = qp.predict(X_te)
            q_inv = inv_transform(q_raw) if log_target else q_raw
            q_preds_te[label] = clamp_short(q_inv, long_threshold)
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
