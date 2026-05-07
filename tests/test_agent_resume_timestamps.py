from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def test_offloaded_agent_reasoning_phase_uses_resume_time():
    src = (REPO_ROOT / "src/run_abc_bench_instrumented.py").read_text(encoding="utf-8")

    assert "return_admitted_at=True" in src
    assert "reasoning_start_dt = parse_iso_utc(admitted_at) or now_utc()" in src
    assert 'self._open_phase("reasoning", reasoning_start_dt)' in src


def test_tool_observation_leaves_tool_call_before_release_path():
    src = (REPO_ROOT / "src/run_abc_bench_instrumented.py").read_text(encoding="utf-8")

    assert 'live["state"] = "waiting"' in src
    assert 'live["admission_state"] = "tool_complete_pending_release"' in src
    assert "self.admission_controller.wait_if_offloaded(" in src


def test_agent_message_does_not_publish_waiting_before_possible_tool_action():
    src = (REPO_ROOT / "src/run_abc_bench_instrumented.py").read_text(encoding="utf-8")

    agent_branch = src.split('if source == "agent":', 1)[1].split(
        "# ── action event", 1
    )[0]
    action_prefix = src.split("if isinstance(event, ActionEvent):", 1)[1].split(
        "tool_name = ", 1
    )[0]

    assert "self._pending_agent_message_dt = event_dt" in agent_branch
    assert "release_agent_kv" not in agent_branch
    assert '_open_phase("waiting", event_dt)' not in agent_branch
    assert "self._pending_agent_message_dt = None" in action_prefix
