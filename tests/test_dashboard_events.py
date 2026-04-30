from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_dashboard_renders_success_and_failed_offload_markers():
    js = _read("dashboard/dashboard.js")
    html = _read("dashboard/index.html")
    css = _read("dashboard/dashboard.css")

    assert "OFFLOAD: ${ev.agent_id}" in js
    assert "OFFLOAD_FAIL: ${ev.agent_id}" in js
    assert 'addEvent(ts, "offload-fail"' in js
    assert "status_code: ${fmt(ev.status_code)}" in js
    assert "reason: ${fmt(ev.reason)}" in js
    assert "threshold_gb: ${fmt(adm.threshold_gb)}" in js

    assert "OFFLOAD_FAIL" in html
    assert "event-offload-fail" in css
    assert "ct-offload-fail" in css


def test_dashboard_exposes_controller_pressure_badge():
    js = _read("dashboard/dashboard.js")
    html = _read("dashboard/index.html")
    css = _read("dashboard/dashboard.css")

    assert "pressureBadge" in html
    assert "pressureBadge" in js
    assert "controller offloads only when C <= threshold" in js
    assert "badge-pressure-active" in css
