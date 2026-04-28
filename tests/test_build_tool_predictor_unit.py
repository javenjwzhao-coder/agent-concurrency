from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib
from build_tool_predictor import (
    LONG_CALL_WEIGHT,
    RealtimePredictor,
    build_features,
    build_model,
    build_sample_weights,
    enrich_sequential_features,
    fit_pipeline,
    load_tool_calls,
    load_traces,
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
    assert feat.loc[0, "start_kv_blocks"] == 42
    assert feat.loc[0, "cumulative_reasoning_s"] == 1.0
    assert feat.loc[1, "prior_tool_calls"] == 1
    assert feat.loc[1, "prior_tool_max_s"] == 3.0

    forbidden = ("observation_", "returncode", "outcome", "end_")
    assert not any(col.startswith(forbidden) for col in feat.columns)

    X, y, prepared_tool_df, _, _, _ = prepare_dataset(tmp_path)
    assert len(prepared_tool_df) == 2
    assert not any(col.startswith(forbidden) for col in X.columns)
    assert y.shape[0] == 2

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

    model_path = tmp_path / "duration_predictor.joblib"
    joblib.dump(pipeline, model_path)

    predictor = RealtimePredictor.load(model_path)
    raw_feat = build_features(tool_df.iloc[[0]], remaining_mode=False)
    raw_feat = enrich_sequential_features(full_df, raw_feat, tool_df.iloc[[0]])
    pred = predictor.predict_duration(raw_feat)

    assert math.isfinite(pred)
    assert pred >= 0.0
