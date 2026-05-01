from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def test_offloaded_agent_reasoning_phase_uses_resume_time():
    src = (REPO_ROOT / "src/run_abc_bench_instrumented.py").read_text(encoding="utf-8")

    assert "return_admitted_at=True" in src
    assert "reasoning_start_dt = parse_iso_utc(admitted_at) or now_utc()" in src
    assert 'self._open_phase("reasoning", reasoning_start_dt)' in src
