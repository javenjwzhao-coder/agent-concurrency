from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_dashboard_renders_successful_offload_markers_only():
    js = _read("dashboard/dashboard.js")
    html = _read("dashboard/index.html")
    css = _read("dashboard/dashboard.css")

    assert "OFFLOAD: ${ev.agent_id}" in js
    assert "adm.offloads || []" in js
    assert "freed_gb: ${fmt(ev.freed_gb)}" in js
    assert "freed_blocks: ${fmt(ev.freed_blocks)}" in js
    assert "free_blocks: ${fmt(ev.free_blocks_before)} -> ${fmt(ev.free_blocks_after)}" in js
    assert "freed_gb_source: ${fmt(ev.freed_gb_source)}" in js
    assert "pending: ${fmt(ev.pending)}" in js
    assert "held_requests: ${fmt(ev.held_requests)}" in js
    assert "known_blocks: ${fmt(ev.known_blocks)}" in js
    assert "offload_jobs: ${fmt(ev.offload_jobs)}" in js
    assert "reason: ${fmt(ev.reason)}" in js

    assert "OFFLOAD_FAIL" not in html
    assert "OFFLOAD_FAIL: ${ev.agent_id}" not in js
    assert 'addEvent(ts, "offload-fail"' not in js
    assert "event-offload-fail" not in css
    assert "event-line-offload-fail" not in css
    assert "eventCountOffload" in html
    assert "eventCountOffloadFail" not in html
    assert "event-count" in css


def test_dashboard_exposes_controller_pressure_badge():
    js = _read("dashboard/dashboard.js")
    html = _read("dashboard/index.html")
    css = _read("dashboard/dashboard.css")

    assert "pressureBadge" in html
    assert "pressureBadge" in js
    assert "W: ${fmtW(s.w)}" in js
    assert "const wBeforeOffload = finiteNumber(adm.w_before_offload)" in js
    assert "w_threshold: ${fmtW(s.wThreshold)}" in js
    assert "w_source: ${fmt(s.wSource)}" in js
    assert "w_before_offload: ${fmtW(s.wBeforeOffload)}" in js
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


def test_dashboard_can_hide_event_line_overlay():
    js = _read("dashboard/dashboard.js")
    html = _read("dashboard/index.html")
    css = _read("dashboard/dashboard.css")

    assert "eventLinesBtn" in html
    assert "Hide event lines" in html
    assert "eventLinesVisible: true" in js
    assert "function setEventLinesVisible(visible)" in js
    assert "clearEventLineLayers();" in js
    assert "Show event lines" in js
    assert "lines-hidden" in css


def test_dashboard_uses_event_specific_admission_timestamps():
    js = _read("dashboard/dashboard.js")
    schema = _read("dashboard/SCHEMA.md")

    assert 'eventTimestamp(ts, ad, ["admitted_at", "admitted_since", "ts"])' in js
    assert "previously_offloaded: ${wasOffloaded}" in js
    assert "w: ${fmt(ad.w)}" in js
    assert "w_threshold: ${fmt(ad.w_threshold)}" in js
    assert "w_source: ${fmt(ad.w_source)}" in js
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


def test_dashboard_exports_standalone_html_snapshots():
    js = _read("dashboard/dashboard.js")
    html = _read("dashboard/index.html")
    schema = _read("dashboard/SCHEMA.md")
    css = _read("dashboard/dashboard.css")

    assert "saveSnapshotBtn" in html
    assert "Save standalone HTML" in html
    assert 'href="static/dashboard.css"' in html
    assert 'src="static/dashboard.js"' in html
    assert "offline asset missing" in html
    assert "dashboardSnapshotData" in js
    assert "records: []" in js
    assert "buildStandaloneHtml" in js
    assert "showSaveFilePicker" in js
    assert "saveSnapshotBlob(blob, filename, fileHandle)" in js
    assert "safeScriptJson(payload)" in js
    assert "loadSnapshotRecords(embedded.ticks, embedded.meta)" in js
    assert "assetTextForExport" in js
    assert "validateStandaloneAsset(\"vis-timeline JS\", visJs" in js
    assert "hasVisCharts()" in js
    assert "renderSnapshotFallback(" in js
    assert "dashboard assets unavailable" in js
    assert "offline-fallback" in css
    assert "without `/state`, `/stream`, or a separate" in schema


def test_dashboard_vendors_vis_timeline_for_offline_save():
    # The unpkg path standalone/umd/vis-timeline-graph2d.min.css is a 404.
    # Vendoring under dashboard/ keeps both the live <link>/<script> tags and
    # the standalone exporter (which re-fetches the same paths) working with
    # no network at run/save time.
    html = _read("dashboard/index.html")
    js = _read("dashboard/dashboard.js")

    assert (REPO_ROOT / "dashboard/vis-timeline-graph2d.min.css").is_file()
    assert (REPO_ROOT / "dashboard/vis-timeline-graph2d.min.js").is_file()
    assert 'href="static/vis-timeline-graph2d.min.css"' in html
    assert 'src="static/vis-timeline-graph2d.min.js"' in html
    # Live tags must not reference the broken upstream URL anymore.
    assert 'href="https://unpkg.com' not in html
    assert 'src="https://unpkg.com' not in html
    # Exporter fallback URLs must point at the vendored copies, not unpkg.
    assert '"/static/vis-timeline-graph2d.min.css"' in js
    assert '"/static/vis-timeline-graph2d.min.js"' in js
    assert "unpkg.com" not in js


def test_dashboard_hides_back_to_live_in_standalone_snapshots():
    # Saved standalone HTML opened over file:// has no /state to return to,
    # so the "Back to live" button must stay hidden even after a replay.
    js = _read("dashboard/dashboard.js")

    assert "function isStandaloneFile()" in js
    assert "$(\"#liveBtn\").hidden = isStandaloneFile();" in js
    assert "if (isStandaloneFile())" in js
