from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
import numpy as np
from pytest import approx as pytest_approx

from build_tool_predictor import (
    LONG_CALL_THRESHOLD_S,
    LONG_CALL_WEIGHT,
    RealtimePredictor,
    build_features,
    build_model,
    build_sample_weights,
    clamp_short,
    enrich_sequential_features,
    fit_pipeline,
    inv_transform,
    load_tool_calls,
    load_traces,
    log_transform,
    prepare_dataset,
)


def _write_synthetic_trace(root: Path) -> None:
    out = root / "task_a"
    out.mkdir(parents=True)

    detailed_rows = [
        {
            "agent_id": "agent_a",
            "task_id": "task_a",
            "conversation_id": "conv-a",
            "tool_seq": 0,
            "action_id": "act-1",
            "tool_call_id": "tc-1",
            "tool_name": "terminal",
            "tool_command": "pytest -q",
            "tool_args_json": json.dumps({"command": "pytest -q", "timeout": 30}),
            "tool_args_keys": "command,timeout",
            "tool_payload_bytes": 38,
            "action_summary": "run tests",
            "action_type": "ActionEvent",
            "start_ts": "2026-04-28T12:00:01+00:00",
            "end_ts": "2026-04-28T12:00:04+00:00",
            "duration_s": 3.0,
            "outcome": "ok",
            "returncode": 0,
            "timed_out": 0,
            "observation_bytes": 999999,
            "observation_line_count": 100,
            "observation_preview": "this must not be a feature",
            "start_active_agents": 3,
            "start_active_tool_calls": 2,
            "start_cumulative_reasoning_s": 1.0,
            "start_kv_blocks": 42,
            "start_kv_gb": 0.5,
            "end_active_agents": 2,
            "end_active_tool_calls": 0,
            "detail": "run tests",
        },
        {
            "agent_id": "agent_a",
            "task_id": "task_a",
            "conversation_id": "conv-a",
            "tool_seq": 1,
            "action_id": "act-2",
            "tool_call_id": "tc-2",
            "tool_name": "file_editor",
            "tool_command": "",
            "tool_args_json": json.dumps({"command": "view", "path": "app.py"}),
            "tool_args_keys": "command,path",
            "tool_payload_bytes": 37,
            "action_summary": "inspect app",
            "action_type": "ActionEvent",
            "start_ts": "2026-04-28T12:00:05+00:00",
            "end_ts": "2026-04-28T12:00:05.4+00:00",
            "duration_s": 0.4,
            "outcome": "ok",
            "returncode": "",
            "timed_out": 0,
            "observation_bytes": 500000,
            "observation_line_count": 50,
            "observation_preview": "also not a feature",
            "start_active_agents": 3,
            "start_active_tool_calls": 1,
            "start_cumulative_reasoning_s": 2.0,
            "start_kv_blocks": "",
            "start_kv_gb": "",
            "end_active_agents": 3,
            "end_active_tool_calls": 0,
            "detail": "inspect app",
        },
    ]
    pd.DataFrame(detailed_rows).to_csv(out / "agent_a_trace.csv", index=False)


def test_detailed_tool_calls_merge_and_exclude_post_call_features(tmp_path):
    _write_synthetic_trace(tmp_path)

    full_df = load_traces(tmp_path)
    tool_df = load_tool_calls(tmp_path, full_df)

    assert len(tool_df) == 2
    assert "observation_bytes" in tool_df.columns

    feat = build_features(tool_df, remaining_mode=False)
    feat = enrich_sequential_features(full_df, feat, tool_df)

    assert feat.loc[0, "start_active_agents"] == 3
    assert feat.loc[0, "start_active_tool_calls"] == 2
    assert feat.loc[0, "cumulative_reasoning_s"] == 1.0
    assert feat.loc[1, "prior_tool_calls"] == 1
    assert feat.loc[1, "prior_tool_max_s"] == 3.0

    forbidden = ("observation_", "returncode", "outcome", "end_", "start_kv")
    assert not any(col.startswith(forbidden) for col in feat.columns)

    X, y, prepared_tool_df, _, _, _ = prepare_dataset(tmp_path)
    assert len(prepared_tool_df) == 2
    assert not any(col.startswith(forbidden) for col in X.columns)
    assert y.shape[0] == 2

    # Long-call-only target reshape: short call (0.4s < 2.0s threshold) → 0,
    # long call (3.0s) → log1p(3.0).
    y_raw = inv_transform(y.values)
    assert y_raw[0] == pytest_approx(3.0)
    assert y_raw[1] == 0.0

    weights = build_sample_weights(prepared_tool_df)
    assert weights.iloc[0] == LONG_CALL_WEIGHT
    assert weights.iloc[1] == 1.0

    remaining_weights = build_sample_weights(prepared_tool_df, remaining_mode=True)
    assert len(remaining_weights) == 10
    assert int((remaining_weights == LONG_CALL_WEIGHT).sum()) == 5


def test_realtime_predictor_lives_in_source_and_aligns_columns(tmp_path):
    _write_synthetic_trace(tmp_path)
    X, y, tool_df, full_df, _, _ = prepare_dataset(tmp_path)

    pipeline = build_model("ridge")
    fit_pipeline(pipeline, X, y)
    pipeline._trained_feature_names = list(X.columns)
    pipeline._log_target = True
    pipeline._long_call_threshold_s = LONG_CALL_THRESHOLD_S

    model_path = tmp_path / "duration_predictor.joblib"
    joblib.dump(pipeline, model_path)

    predictor = RealtimePredictor.load(model_path)
    raw_feat = build_features(tool_df.iloc[[0]], remaining_mode=False)
    raw_feat = enrich_sequential_features(full_df, raw_feat, tool_df.iloc[[0]])
    pred = predictor.predict_duration(raw_feat)

    assert math.isfinite(pred)
    assert pred >= 0.0


def test_realtime_predictor_clamps_short_predictions_to_zero(tmp_path):
    """A model whose threshold is set above any plausible prediction must
    output exactly 0.0 from RealtimePredictor (the inference clamp)."""
    _write_synthetic_trace(tmp_path)
    X, y, tool_df, full_df, _, _ = prepare_dataset(tmp_path)

    pipeline = build_model("ridge")
    fit_pipeline(pipeline, X, y)
    pipeline._trained_feature_names = list(X.columns)
    pipeline._log_target = True
    pipeline._long_call_threshold_s = 10_000.0  # nothing should ever clear this

    model_path = tmp_path / "duration_predictor.joblib"
    joblib.dump(pipeline, model_path)

    predictor = RealtimePredictor.load(model_path)
    raw_feat = build_features(tool_df, remaining_mode=False)
    raw_feat = enrich_sequential_features(full_df, raw_feat, tool_df)
    for i in range(len(raw_feat)):
        pred = predictor.predict_duration(raw_feat.iloc[[i]])
        assert pred == 0.0


def test_clamp_short_helper_zeroes_below_threshold():
    arr = np.array([0.0, 0.5, 1.99, 2.0, 2.5, 10.0])
    out = clamp_short(arr, threshold=2.0)
    assert (out[:3] == 0.0).all()
    assert out[3] == pytest_approx(2.0)
    assert out[4] == pytest_approx(2.5)
    assert out[5] == pytest_approx(10.0)


def test_log_target_round_trip_zero_for_short_call():
    """Short-call target is 0 raw → 0 in log space → 0 after inv_transform."""
    y = pd.Series([0.4, 3.0, 0.0])
    long_thresh = 2.0
    reshaped = y.where(y >= long_thresh, 0.0)
    transformed = log_transform(reshaped)
    recovered = inv_transform(transformed.values)
    assert recovered[0] == 0.0
    assert recovered[1] == pytest_approx(3.0)
    assert recovered[2] == 0.0
