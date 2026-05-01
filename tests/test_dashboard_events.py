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
    assert "adm.offloads || []" in js
    assert 'addEvent(ts, "offload-fail"' in js
    assert "freed_gb: ${fmt(ev.freed_gb)}" in js
    assert "freed_blocks: ${fmt(ev.freed_blocks)}" in js
    assert "free_blocks: ${fmt(ev.free_blocks_before)} -> ${fmt(ev.free_blocks_after)}" in js
    assert "freed_gb_source: ${fmt(ev.freed_gb_source)}" in js
    assert "pending: ${fmt(ev.pending)}" in js
    assert "held_requests: ${fmt(ev.held_requests)}" in js
    assert "known_blocks: ${fmt(ev.known_blocks)}" in js
    assert "offload_jobs: ${fmt(ev.offload_jobs)}" in js
    assert "status_code: ${fmt(ev.status_code)}" in js
    assert "reason: ${fmt(ev.reason)}" in js
    assert "timeout_s: ${fmt(ev.timeout_s)}" in js
    assert "threshold_percent: ${fmt(adm.threshold_percent)}" in js
    assert "threshold_gb: ${fmt(adm.threshold_gb)}" in js

    assert "OFFLOAD_FAIL" in html
    assert "event-offload-fail" in css
    assert "event-line-offload-fail" in css
    assert "eventCountOffload" in html
    assert "eventCountOffloadFail" in html
    assert "event-count" in css


def test_dashboard_exposes_controller_pressure_badge():
    js = _read("dashboard/dashboard.js")
    html = _read("dashboard/index.html")
    css = _read("dashboard/dashboard.css")

    assert "pressureBadge" in html
    assert "pressureBadge" in js
    assert "W: ${fmtW(s.effectiveW)}" in js
    assert "w_threshold: ${fmtW(s.wThreshold)}" in js
    assert "w_after_offload: ${fmtW(s.wAfterOffload)}" in js
    assert "controller offloads only when free KV percent <= threshold" in js
    assert "badge-pressure-active" in css


def test_dashboard_exposes_vllm_preempt_badge_and_event_line_overlay():
    js = _read("dashboard/dashboard.js")
    html = _read("dashboard/index.html")
    css = _read("dashboard/dashboard.css")

    assert "vllmPreemptCount" in html
    assert "vllmPreemptCount" in js
    assert "vllm preempt: ${fmtCount(s.vllmPreemptions)}" in js
    assert "scheduler_preemptions_total" in js
    assert "badge-vllm" in css

    assert "event-line-layer" in js
    assert "eventLineEntries" in js
    assert "EVENT_LINE_HIT_PX" in js
    assert "incrementEventCount(type)" in js
    assert "resetEventCounts()" in js


def test_dashboard_uses_event_specific_admission_timestamps():
    js = _read("dashboard/dashboard.js")
    schema = _read("dashboard/SCHEMA.md")

    assert 'eventTimestamp(ts, ad, ["admitted_at", "admitted_since", "ts"])' in js
    assert "previously_offloaded: ${wasOffloaded}" in js
    assert "admitted_at: ${fmt(ad.admitted_at)}" in js
    assert "function eventTimestamp(recordTs, event, keys)" in js
    assert "admissions[*].admitted_at" in schema
    assert "falls back to tick `ts` if absent" in schema


def test_dashboard_event_colors_are_distinct_and_consistent():
    css = _read("dashboard/dashboard.css")

    assert ".event-offload      { background: #ef4444" in css
    assert ".event-readmit      { background: #38bdf8" in css
    assert ".event-line-offload { background: #ef4444" in css
    assert ".event-line-readmit { background: #38bdf8" in css
    assert ".event-line-sat" in css
    assert "to bottom, #fde047 0 6px" in css


def test_dashboard_expands_agent_label_panel_for_full_ids():
    js = _read("dashboard/dashboard.js")
    css = _read("dashboard/dashboard.css")

    assert "updateAgentLabelPanelWidth(initialLabel)" in js
    assert "--agent-label-panel-width" in js
    assert "--agent-label-panel-width: 220px" in css
    assert "width: var(--agent-label-panel-width)" in css
    assert "text-overflow: clip" in css
    assert "max-width: none" in css


def test_dashboard_agent_labels_show_elapsed_then_fixed_e2e_time():
    js = _read("dashboard/dashboard.js")
    schema = _read("dashboard/SCHEMA.md")

    assert "(elapsed: ${formatDurationSeconds(recordMs - startMs)} secs)" in js
    assert "(E2E: ${formatDurationSeconds(finishedMs - startMs)} secs)" in js
    assert "agent.finished_at" in js
    assert 'agent.state === "done"' in js
    assert "started_at" in schema
    assert "finished_at" in schema
