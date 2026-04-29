/* dashboard.js — render live sidecar tick records as a Gantt timeline plus a
 * KV-cache-usage line chart on a shared time axis.
 *
 * Data model: see dashboard/SCHEMA.md. Each tick record is the same JSON the
 * sidecar writes to sidecar.log; we derive phase rows + admission-event
 * markers + a single kv_cache_used_pct line from it, no transformation in
 * the feed.
 *
 * Modes
 * -----
 *   live    — bootstrap from /state, then subscribe to /stream (SSE).
 *   replay  — user opens a sidecar.log file from disk; we parse the JSONL
 *             client-side and replay every tick into the same render path.
 *             SSE is disconnected and auto-scroll is paused; the viewport
 *             snaps to the full extent of the loaded log.
 *
 * Rendering is incremental: per tick we extend the active phase item per
 * agent (or close it and open a new one when state changes), append vertical
 * background items + label box items for new admission events, and append
 * one point to the KV line. Switching modes resets datasets cleanly.
 */

(function () {
  "use strict";

  const VIEWPORT_PAST_MS  = 60_000;
  const VIEWPORT_AHEAD_MS = 5_000;

  const state = {
    timeline: null,
    items: null,
    groups: null,
    kvChart: null,
    kvPoints: null,
    agents: new Map(),
    agentOrder: [],
    eventCounter: 0,
    itemCounter: 0,
    customTimeIds: [],   // IDs added via addCustomTime on both charts
    latestTick: -1,
    latestTs: null,
    paused: false,
    eventSource: null,
    mode: "live",       // "live" | "replay"
    replayBounds: null, // {start, end} when in replay mode
  };

  // ── hover tooltip (used for event lines) ───────────────────────────────

  let tooltipEl = null;
  function getTooltipEl() {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.id = "tooltip";
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  }
  function showTooltip(text, x, y) {
    const el = getTooltipEl();
    el.textContent = text;
    el.style.display = "block";
    moveTooltip(x, y);
  }
  function moveTooltip(x, y) {
    const el = getTooltipEl();
    const w = el.offsetWidth || 220;
    const h = el.offsetHeight || 80;
    let left = x + 16;
    let top  = y - 12;
    if (left + w > window.innerWidth  - 8) left = x - w - 16;
    if (top  + h > window.innerHeight - 8) top  = window.innerHeight - h - 8;
    el.style.left = left + "px";
    el.style.top  = top  + "px";
  }
  function hideTooltip() {
    if (tooltipEl) tooltipEl.style.display = "none";
  }

  // ── DOM helpers ────────────────────────────────────────────────────────

  const $ = (sel) => document.querySelector(sel);

  function setStatus(text, cls) {
    const el = $("#connStatus");
    el.textContent = text;
    el.className = "status " + cls;
  }

  function setTickInfo(tick, ts) {
    $("#tickInfo").textContent = ts ? `tick ${tick} · ${ts}` : "";
  }

  // ── timeline + KV chart bootstrap ──────────────────────────────────────

  function buildCharts() {
    state.items = new vis.DataSet();
    state.groups = new vis.DataSet();
    const initialStart = new Date(Date.now() - VIEWPORT_PAST_MS);
    const initialEnd   = new Date(Date.now() + VIEWPORT_AHEAD_MS);
    const timelineOpts = {
      stack: false,
      orientation: { axis: "top", item: "top" },
      horizontalScroll: true,
      zoomKey: "ctrlKey",
      groupOrder: "order",
      margin: { item: 1, axis: 4 },
      tooltip: { followMouse: true, overflowMethod: "flip" },
      start: initialStart,
      end:   initialEnd,
      format: {
        minorLabels: { second: "HH:mm:ss" },
        majorLabels: { second: "HH:mm" },
      },
    };
    state.timeline = new vis.Timeline(
      $("#timeline"), state.items, state.groups, timelineOpts,
    );

    state.kvPoints = new vis.DataSet();
    const kvOpts = {
      style: "line",
      shaded: { enabled: true, orientation: "bottom" },
      drawPoints: { size: 3, style: "circle" },
      interpolation: false,
      start: initialStart,
      end:   initialEnd,
      dataAxis: {
        left: {
          range: { min: 0, max: 100 },
          format: (v) => v.toFixed(0) + "%",
          title: { text: "KV used %" },
        },
      },
      legend: false,
      moveable: true,
      zoomable: true,
    };
    state.kvChart = new vis.Graph2d(
      $("#kvChart"), state.kvPoints, kvOpts,
    );

    // Keep the two charts' time axes in sync when the user pans/zooms.
    state.timeline.on("rangechange", (props) => {
      state.kvChart.setWindow(props.start, props.end, { animation: false });
      // User panning implies they want to look around → pause auto-scroll.
      if (props.byUser) setPaused(true);
    });
    state.kvChart.on("rangechange", (props) => {
      state.timeline.setWindow(props.start, props.end, { animation: false });
      if (props.byUser) setPaused(true);
    });
  }

  function autoScroll(nowDate) {
    if (state.paused || state.mode === "replay") return;
    const start = new Date(nowDate.getTime() - VIEWPORT_PAST_MS);
    const end   = new Date(nowDate.getTime() + VIEWPORT_AHEAD_MS);
    state.timeline.setWindow(start, end, { animation: false });
    state.kvChart.setWindow(start, end, { animation: false });
  }

  function setPaused(p) {
    if (state.paused === p) return;
    state.paused = p;
    const btn = $("#pauseBtn");
    btn.classList.toggle("paused", p);
    btn.textContent = p ? "Resume auto-scroll" : "Pause auto-scroll";
  }

  // ── per-agent phase rendering ──────────────────────────────────────────

  function ensureAgentGroup(agentId, taskId) {
    if (state.agents.has(agentId)) return state.agents.get(agentId);
    const groupId = `agent::${agentId}`;
    state.agentOrder.push(agentId);
    state.groups.add({
      id: groupId,
      content: agentId,
      title: taskId ? `task: ${taskId}` : agentId,
      order: state.agentOrder.length,
    });
    const entry = {
      groupId,
      activeItemId: null,
      activeStart: null,
      activePhase: null,
      lastKvGb: null,
    };
    state.agents.set(agentId, entry);
    return entry;
  }

  function applyAgentTick(agentId, agent, recordTs) {
    const entry = ensureAgentGroup(agentId, agent.task_id);
    const phase = agent.state || "waiting";
    const phaseStart = agent.state_since || recordTs;
    entry.lastKvGb = (agent.kv_gb !== undefined) ? agent.kv_gb : entry.lastKvGb;

    if (entry.activeItemId !== null
        && entry.activePhase === phase
        && entry.activeStart === phaseStart) {
      state.items.update({
        id: entry.activeItemId,
        end: new Date(recordTs),
        title: phaseTooltip(agentId, agent, phaseStart, recordTs),
      });
      return;
    }

    if (entry.activeItemId !== null) {
      state.items.update({
        id: entry.activeItemId,
        end: new Date(phaseStart),
      });
    }
    const itemId = `phase::${agentId}::${++state.itemCounter}`;
    state.items.add({
      id: itemId,
      group: entry.groupId,
      start: new Date(phaseStart),
      end: new Date(recordTs),
      content: phase,
      className: "phase-" + phase,
      title: phaseTooltip(agentId, agent, phaseStart, recordTs),
    });
    entry.activeItemId = itemId;
    entry.activeStart = phaseStart;
    entry.activePhase = phase;
  }

  function phaseTooltip(agentId, agent, phaseStart, recordTs) {
    const start = new Date(phaseStart);
    const end = new Date(recordTs);
    const durSec = Math.max(0, (end - start) / 1000).toFixed(1);
    const lines = [
      `agent: ${agentId}`,
      `phase: ${agent.state}`,
      `start: ${formatHMS(start)}`,
      `dur:   ${durSec} s (so far)`,
    ];
    if (agent.kv_gb !== null && agent.kv_gb !== undefined) {
      lines.push(`kv:    ${Number(agent.kv_gb).toFixed(3)} GB`);
    }
    if (agent.tool_name) lines.push(`tool:  ${agent.tool_name}`);
    if (agent.admission_state) lines.push(`adm:   ${agent.admission_state}`);
    return escapeHtml(lines.join("\n"));
  }

  // ── events ─────────────────────────────────────────────────────────────

  function applyEventsTick(record) {
    const adm = record.admission || {};
    const ts = record.ts;
    const tooltipBase = baseTooltip(record);

    for (const ev of (adm.evictions || [])) {
      if (!ev.evicted) continue;
      addEvent(ts, "evict", `EVICT: ${ev.agent_id}`,
        tooltipBase + `\nagent: ${ev.agent_id}\n` +
        `freed_gb: ${fmt(ev.freed_gb)}\n` +
        `e_s:      ${fmt(ev.e_s)}\n` +
        `kv_gb:    ${fmt(ev.kv_gb)}\n` +
        `predicted_remaining_s: ${fmt(ev.predicted_remaining_s)}`);
    }
    for (const ad of (adm.admissions || [])) {
      if (!ad.admitted) continue;
      const type = ad.previously_evicted ? "readmit" : "admit";
      const tag = ad.previously_evicted
        ? `READMIT: ${ad.agent_id}`
        : `ADMIT: ${ad.agent_id}`;
      addEvent(ts, type, tag,
        tooltipBase + `\nagent: ${ad.agent_id}\n` +
        `previously_evicted: ${ad.previously_evicted}`);
    }
    if ((adm.reasons || []).indexOf("saturation_guard") !== -1) {
      addEvent(ts, "sat", "SAT",
        tooltipBase + `\nreasons: ${(adm.reasons || []).join(", ")}`);
    }
  }

  function baseTooltip(record) {
    const adm = record.admission || {};
    return [
      `ts:     ${record.ts}`,
      `tick:   ${record.tick}`,
      `C:      ${fmt(adm.C)}`,
      `s_t:    ${fmt(adm.s_t)}`,
      `s_prev: ${fmt(adm.s_prev)}`,
      `w:      ${fmt(adm.w)}`,
    ].join("\n");
  }

  function addCustomTimeStyled(chart, container, time, id, type, tooltipText) {
    chart.addCustomTime(time, id);
    const els = container.querySelectorAll(".vis-custom-time");
    const el = els[els.length - 1];
    if (el) {
      el.classList.add("ct-" + type);
      el.addEventListener("mouseenter", (e) => showTooltip(tooltipText, e.clientX, e.clientY));
      el.addEventListener("mousemove",  (e) => moveTooltip(e.clientX, e.clientY));
      el.addEventListener("mouseleave", hideTooltip);
    }
  }

  function addEvent(ts, type, label, tooltipText) {
    const id = `evt::${++state.eventCounter}`;
    const start = new Date(ts);
    const fullText = label + "\n\n" + tooltipText;
    addCustomTimeStyled(state.timeline, $("#timeline"), start, id, type, fullText);
    addCustomTimeStyled(state.kvChart,  $("#kvChart"),  start, id, type, fullText);
    state.customTimeIds.push(id);
  }

  // ── KV % line ──────────────────────────────────────────────────────────

  function applyKvTick(record) {
    const pct = record.vllm && record.vllm.kv_cache_used_pct;
    if (pct === null || pct === undefined) return;
    state.kvPoints.add({ x: new Date(record.ts), y: Number(pct) });
  }

  // ── tick application ───────────────────────────────────────────────────

  function applyTick(record) {
    if (typeof record.tick === "number") {
      if (record.tick <= state.latestTick) return; // dedupe / out-of-order
      state.latestTick = record.tick;
    }
    state.latestTs = record.ts;
    setTickInfo(record.tick, record.ts);

    const recordTs = record.ts;
    const agents = record.agents || {};
    for (const agentId of Object.keys(agents)) {
      applyAgentTick(agentId, agents[agentId], recordTs);
    }
    applyEventsTick(record);
    applyKvTick(record);
    if (state.mode === "live") {
      autoScroll(new Date(recordTs));
    } else if (state.mode === "replay") {
      // Track replay window so we can fit-to-data at the end.
      const t = new Date(recordTs).getTime();
      if (state.replayBounds === null) {
        state.replayBounds = { start: t, end: t };
      } else {
        if (t < state.replayBounds.start) state.replayBounds.start = t;
        if (t > state.replayBounds.end)   state.replayBounds.end = t;
      }
    }
  }

  // ── live data acquisition ──────────────────────────────────────────────

  async function bootstrapLive() {
    setStatus("loading history…", "status-pending");
    try {
      const resp = await fetch("/state");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const ticks = data.ticks || [];
      ticks.forEach(applyTick);
      setStatus(`history: ${ticks.length} tick(s)`, "status-pending");
    } catch (err) {
      console.error("bootstrap failed", err);
      setStatus("offline (no /state) · use 'Open sidecar.log'", "status-error");
    }
  }

  function connectStream() {
    if (state.eventSource) state.eventSource.close();
    const url = (state.latestTick >= 0)
      ? `/stream?since=${state.latestTick}`
      : "/stream";
    let es;
    try {
      es = new EventSource(url);
    } catch (err) {
      setStatus("SSE unsupported", "status-error");
      return;
    }
    state.eventSource = es;
    setStatus("connecting…", "status-pending");
    es.onopen = () => setStatus("live", "status-live");
    es.onerror = () => {
      if (state.mode !== "live") return;
      setStatus("disconnected · reconnecting", "status-error");
      es.close();
      setTimeout(connectStream, 1500);
    };
    es.onmessage = (msg) => {
      try { applyTick(JSON.parse(msg.data)); }
      catch (err) { console.error("bad tick frame", err, msg.data); }
    };
  }

  // ── replay (in-browser) ────────────────────────────────────────────────

  function resetCharts() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    for (const id of state.customTimeIds) {
      try { state.timeline.removeCustomTime(id); } catch (e) { /* already gone */ }
      try { state.kvChart.removeCustomTime(id); }  catch (e) { /* already gone */ }
    }
    state.customTimeIds = [];
    state.items.clear();
    state.groups.clear();
    state.kvPoints.clear();
    state.agents.clear();
    state.agentOrder = [];
    state.eventCounter = 0;
    state.itemCounter = 0;
    state.latestTick = -1;
    state.latestTs = null;
    state.replayBounds = null;
  }

  async function loadReplayFile(file) {
    setStatus(`replay: parsing ${file.name}…`, "status-pending");
    let text;
    try {
      text = await file.text();
    } catch (err) {
      setStatus("replay: read failed", "status-error");
      return;
    }
    state.mode = "replay";
    setPaused(true);
    resetCharts();

    const lines = text.split(/\r?\n/);
    let parsed = 0;
    let skipped = 0;
    for (const line of lines) {
      if (!line.trim()) continue;
      let rec;
      try { rec = JSON.parse(line); } catch (err) { skipped++; continue; }
      try { applyTick(rec); parsed++; }
      catch (err) { skipped++; }
    }
    if (state.replayBounds) {
      const pad = Math.max(1000, (state.replayBounds.end - state.replayBounds.start) * 0.02);
      const s = new Date(state.replayBounds.start - pad);
      const e = new Date(state.replayBounds.end   + pad);
      state.timeline.setWindow(s, e, { animation: false });
      state.kvChart.setWindow(s, e, { animation: false });
    }
    setStatus(
      `replay · ${parsed} tick(s)` + (skipped ? ` · ${skipped} skipped` : ""),
      "status-live",
    );
    $("#liveBtn").hidden = false;
  }

  function backToLive() {
    state.mode = "live";
    setPaused(false);
    resetCharts();
    $("#liveBtn").hidden = true;
    setStatus("connecting…", "status-pending");
    bootstrapLive().then(connectStream);
  }

  // ── controls ───────────────────────────────────────────────────────────

  function wireControls() {
    $("#pauseBtn").addEventListener("click", () => {
      const next = !state.paused;
      setPaused(next);
      if (!next && state.latestTs && state.mode === "live") {
        autoScroll(new Date(state.latestTs));
      }
    });
    $("#replayBtn").addEventListener("click", () => $("#logFile").click());
    $("#logFile").addEventListener("change", (ev) => {
      const f = ev.target.files && ev.target.files[0];
      if (f) loadReplayFile(f);
      ev.target.value = "";
    });
    $("#liveBtn").addEventListener("click", backToLive);
  }

  // ── utilities ──────────────────────────────────────────────────────────

  function fmt(v) {
    if (v === null || v === undefined) return "null";
    if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
    return String(v);
  }

  function formatHMS(d) {
    const pad = (n) => String(n).padStart(2, "0");
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll("\"", "&quot;");
  }

  // ── boot ───────────────────────────────────────────────────────────────

  document.addEventListener("DOMContentLoaded", async () => {
    buildCharts();
    wireControls();
    await bootstrapLive();
    connectStream();
  });
})();
