from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from collect_tool_trace import ToolCallTraceCollector, load_tool_call_records


def _dt(second: int) -> datetime:
    return datetime(2026, 4, 28, 12, 0, second, tzinfo=timezone.utc)


def test_collector_records_terminal_call_and_writes_artifacts(tmp_path):
    snapshots = [
        {
            "active_agents": 2,
            "active_tool_calls": 1,
            "agent_state": "tool_call",
            "state_since": "2026-04-28T12:00:00+00:00",
            "cumulative_reasoning_s": 1.25,
            "kv_blocks": 12,
            "kv_gb": 0.125,
        },
        {
            "active_agents": 1,
            "active_tool_calls": 0,
            "agent_state": "reasoning",
            "kv_blocks": 8,
            "kv_gb": 0.08,
        },
    ]
    calls = {"n": 0}

    def live_snapshot():
        idx = min(calls["n"], len(snapshots) - 1)
        calls["n"] += 1
        return snapshots[idx]

    collector = ToolCallTraceCollector(
        agent_id="agent_task_a_0",
        task_id="task_a",
        live_snapshot_fn=live_snapshot,
        preview_chars=10,
    )
    action = SimpleNamespace(
        id="act-1",
        tool_call_id="tc-1",
        tool_name="terminal",
        arguments={"command": "pytest -q", "timeout": 30},
        summary="run tests",
        conversation_id="conv-1",
    )
    pending = collector.record_action(action, _dt(1), conversation_id="conv-1")

    assert pending.tool_command == "pytest -q"
    assert pending.tool_payload_bytes > 0
    assert collector.pending_count == 1

    output = "ok\nall good"
    obs = SimpleNamespace(
        action_id="act-1",
        tool_name="terminal",
        content=output,
        returncode=0,
    )
    record = collector.record_observation(obs, _dt(4), outcome="ok")

    assert record is not None
    assert record.duration_s == 3.0
    assert record.outcome == "ok"
    assert record.returncode == "0"
    assert record.observation_bytes == len(output.encode("utf-8"))
    assert record.observation_line_count == 2
    assert record.observation_preview == output[:10]
    assert record.observation_sha256 == hashlib.sha256(output.encode("utf-8")).hexdigest()
    assert record.start_active_agents == 2
    assert record.start_active_tool_calls == 1
    assert record.start_cumulative_reasoning_s == 1.25
    assert record.start_kv_blocks == "12"
    assert record.end_active_agents == 1

    paths = collector.write_artifacts(tmp_path)
    jsonl_rows = load_tool_call_records(paths["jsonl"])
    csv_rows = load_tool_call_records(paths["csv"])

    assert jsonl_rows[0]["tool_command"] == "pytest -q"
    assert csv_rows[0]["tool_name"] == "terminal"
    assert csv_rows[0]["observation_preview"] == output[:10]


def test_collector_matches_by_tool_call_id_and_finalizes_unfinished():
    collector = ToolCallTraceCollector("agent_x", "task_x")
    action = SimpleNamespace(
        id="act-2",
        tool_call_id="tc-2",
        tool_name="file_editor",
        arguments={"command": "view", "path": "app.py"},
        summary="inspect file",
    )
    collector.record_action(action, _dt(5))

    obs = SimpleNamespace(tool_call_id="tc-2", content="file contents")
    record = collector.record_observation(obs, _dt(6), outcome="ok")

    assert record is not None
    assert record.tool_name == "file_editor"
    assert collector.pending_count == 0

    collector.record_action(
        SimpleNamespace(id="act-3", tool_name="terminal", arguments={"command": "sleep 10"}),
        _dt(10),
    )
    finalized = collector.finalize_unfinished(
        _dt(12),
        outcome="error",
        detail="run crashed",
    )

    assert len(finalized) == 1
    assert finalized[0].duration_s == 2.0
    assert finalized[0].outcome == "error"
    assert finalized[0].detail == "run crashed"
    assert collector.pending_count == 0
