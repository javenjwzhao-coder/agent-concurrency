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
    kvGroups: null,
    agents: new Map(),
    agentOrder: [],
    eventCounter: 0,
    itemCounter: 0,
    customTimeIds: [],   // IDs added via addCustomTime on both charts
    eventPoints: [],     // [{ts: epochMs, type, label, text}] for proximity tooltip
    tickHistory: [],     // [{ts: epoch_ms, count: active_agent_count}]
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
    state.kvGroups = new vis.DataSet([
      {
        id: "kv",
        content: "KV used %",
        className: "kv-line-used",
        style: "stroke:#f59e0b;stroke-width:2px;fill:none;",
        options: {
          shaded: {
            enabled: true,
            orientation: "bottom",
            style: "fill:#f59e0b;fill-opacity:0.10;stroke:none;",
          },
          drawPoints: {
            enabled: true,
            size: 3,
            style: "circle",
            styles: "fill:#f59e0b;stroke:#c47a07;stroke-width:1px;",
          },
          interpolation: { enabled: false },
        },
      },
      {
        id: "threshold",
        content: "offload threshold",
        className: "kv-line-threshold",
        style: "stroke:#ff003d;stroke-width:2px;fill:none;",
        options: {
          shaded: { enabled: false },
          drawPoints: { enabled: false },
          interpolation: { enabled: false },
        },
      },
    ]);
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
      $("#kvChart"), state.kvPoints, state.kvGroups, kvOpts,
    );

    // Keep the two charts' time axes in sync when the user pans/zooms.
    state.timeline.on("rangechange", (props) => {
      state.kvChart.setWindow(props.start, props.end, { animation: false });
      if (props.byUser) setPaused(true);
      updateBadges(props.end.getTime());
    });
    state.kvChart.on("rangechange", (props) => {
      state.timeline.setWindow(props.start, props.end, { animation: false });
      if (props.byUser) setPaused(true);
      updateBadges(props.end.getTime());
    });

    setupProximityTooltips();
  }

  // ── header count badges ────────────────────────────────────────────────
  //
  // tickHistory carries a snapshot per tick so all badges stay aligned with
  // the viewport's right edge (binary search) rather than always showing the
  // latest sidecar tick. Snapshot fields:
  //   ts        — epoch ms
  //   live      — agents in reasoning|tool_call|waiting
  //   reasoning, tool_call, waiting  — phase breakdown of `live`
  //   offloaded — agents currently in evicted_waiting (KV pushed to CPU)
  //   done      — finished agents
  //   launched  — total agents the sidecar has ever seen (cumulative)
  //   heap      — admission.heap_candidates length (eligible to offload)
  //   C, threshold, pressure — free KV GB, configured offload threshold,
  //                            and whether the controller is in pressure
  //   queue     — fresh + evicted_ready waiting to be admitted

  function snapshotCounts(record) {
    const agents = record.agents || {};
    let reasoning = 0, tool_call = 0, waiting = 0, offloaded = 0, done = 0;
    let launched = 0;
    for (const id of Object.keys(agents)) {
      launched++;
      switch (agents[id].state) {
        case "reasoning":       reasoning++; break;
        case "tool_call":       tool_call++; break;
        case "waiting":         waiting++;   break;
        case "evicted_waiting": offloaded++; break;
        case "done":            done++;      break;
        default:                waiting++;   break;
      }
    }
    const adm = record.admission || {};
    const q = adm.queue || {};
    const C = finiteNumber(adm.C);
    const threshold = finiteNumber(adm.threshold_gb);
    const w = finiteNumber(adm.w);
    const wAfterOffload = finiteNumber(adm.w_after_offload);
    const effectiveW = wAfterOffload !== null ? wAfterOffload : w;
    const pressure = adm.pressure === true
      || (C !== null && threshold !== null && C <= threshold);
    return {
      live: reasoning + tool_call + waiting,
      reasoning, tool_call, waiting, offloaded, done, launched,
      queue: (q.fresh || 0) + (q.evicted_ready || 0),
      heap: (adm.heap_candidates || []).length,
      C,
      threshold,
      w,
      wAfterOffload,
      effectiveW,
      pressure,
    };
  }

  function snapshotAt(timeMs) {
    const h = state.tickHistory;
    if (!h.length) return null;
    if (h[0].ts > timeMs) return null;
    let lo = 0, hi = h.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (h[mid].ts <= timeMs) lo = mid; else hi = mid - 1;
    }
    return h[lo];
  }

  function setBadge(id, text, title) {
    const el = $("#" + id);
    if (!el) return;
    el.textContent = text;
    if (title !== undefined) el.title = title;
    el.style.display = "";
  }

  function updateBadges(timeMs) {
    const s = snapshotAt(timeMs);
    if (!s) return;
    setBadge(
      "liveCount",
      `live: ${s.live}`,
      `reasoning: ${s.reasoning}\ntool_call: ${s.tool_call}\nwaiting:   ${s.waiting}\n\nlaunched: ${s.launched}`,
    );
    setBadge("offloadedCount", `offloaded: ${s.offloaded}`,
      "agents whose KV is currently offloaded to CPU (evicted_waiting)");
    setBadge("heapCount", `heap: ${s.heap}`,
      "agents currently in the offload heap (eligible to be offloaded)");
    setBadge("pressureBadge", `C: ${fmtGb(s.C)} / W: ${fmtW(s.effectiveW)} / T: ${fmtGb(s.threshold)}`,
      "free KV GB / effective headroom W / pressure threshold GB\n" +
      `raw w: ${fmtW(s.w)}\nw_after_offload: ${fmtW(s.wAfterOffload)}\n` +
      "controller offloads only when C <= threshold");
    const pressureEl = $("#pressureBadge");
    if (pressureEl) {
      pressureEl.classList.toggle("badge-pressure-active", s.pressure);
    }
    setBadge("queueCount", `queue: ${s.queue}`,
      "fresh + evicted_ready agents waiting to be admitted");
    setBadge("doneCount", `done: ${s.done} / ${s.launched}`,
      "completed agents / total launched");
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
      const evicted = ev.evicted === true
        || (ev.evicted === undefined && ev.offloaded === true);
      if (evicted) {
        addEvent(ts, "offload", `OFFLOAD: ${ev.agent_id}`,
          `agent: ${ev.agent_id}\n` +
          `freed_gb: ${fmt(ev.freed_gb)}\n` +
          `e_s:      ${fmt(ev.e_s)}\n` +
          `kv_gb:    ${fmt(ev.kv_gb)}\n` +
          `predicted_remaining_s: ${fmt(ev.predicted_remaining_s)}`,
          tooltipBase);
      } else {
        addEvent(ts, "offload-fail", `OFFLOAD_FAIL: ${ev.agent_id}`,
          `agent: ${ev.agent_id}\n` +
          `status_code: ${fmt(ev.status_code)}\n` +
          `reason: ${fmt(ev.reason)}\n` +
          `offloaded: ${fmt(ev.offloaded)}\n` +
          `freed_gb: ${fmt(ev.freed_gb)}\n` +
          `kv_gb: ${fmt(ev.kv_gb)}\n` +
          `C: ${fmt(adm.C)}\n` +
          `threshold_gb: ${fmt(adm.threshold_gb)}`,
          tooltipBase);
      }
    }
    for (const ad of (adm.admissions || [])) {
      if (!ad.admitted) continue;
      const type = ad.previously_evicted ? "readmit" : "admit";
      const tag = ad.previously_evicted
        ? `READMIT: ${ad.agent_id}`
        : `ADMIT: ${ad.agent_id}`;
      addEvent(ts, type, tag,
        `agent: ${ad.agent_id}\n` +
        `previously_evicted: ${ad.previously_evicted}`,
        tooltipBase);
    }
    if ((adm.reasons || []).indexOf("saturation_guard") !== -1) {
      addEvent(ts, "sat", "SAT",
        `reasons: ${(adm.reasons || []).join(", ")}`,
        tooltipBase);
    }
  }

  function baseTooltip(record) {
    const adm = record.admission || {};
    return [
      `ts:     ${record.ts}`,
      `tick:   ${record.tick}`,
      `C:      ${fmt(adm.C)}`,
      `threshold_gb: ${fmt(adm.threshold_gb)}`,
      `pressure: ${pressureLabel(adm)}`,
      `s_t:    ${fmt(adm.s_t)}`,
      `s_prev: ${fmt(adm.s_prev)}`,
      `w:      ${fmt(adm.w)}`,
      `active: ${fmt(adm.active_agents)} / ${fmt(adm.max_active_agents)}`,
      `active_slots: ${fmt(adm.active_agent_slots)}`,
      `queue:  fresh=${fmt(adm.queue && adm.queue.fresh)} ` +
        `ready=${fmt(adm.queue && adm.queue.evicted_ready)} ` +
        `pending_tool=${fmt(adm.queue && adm.queue.evicted_pending_tool)}`,
      `heap_candidates: ${(adm.heap_candidates || []).length}`,
    ].join("\n");
  }

  // pointer-events:none on .vis-custom-time prevents both dragging and hover.
  // Tooltips are shown via container-level proximity detection (setupProximityTooltips).
  function addCustomTimeStyled(chart, container, chartSuffix, time, id, type) {
    const fullId = id + "::" + chartSuffix;
    try {
      chart.addCustomTime(time, fullId);
    } catch (err) {
      console.error("addCustomTime failed", fullId, err);
      return fullId;
    }
    const els = container.querySelectorAll(".vis-custom-time");
    const el = els[els.length - 1];
    if (el) el.classList.add("ct-" + type);
    return fullId;
  }

  function addEvent(ts, type, label, extraText, sharedBase) {
    const id = `evt::${++state.eventCounter}`;
    const start = new Date(ts);
    const tlId = addCustomTimeStyled(state.timeline, $("#timeline"), "tl", start, id, type);
    const kvId = addCustomTimeStyled(state.kvChart,  $("#kvChart"),  "kv", start, id, type);
    state.customTimeIds.push(tlId, kvId);
    // Store label, event-specific extra, and shared base separately so same-tick
    // events can be grouped into one tooltip without repeating the base fields.
    state.eventPoints.push({ ts: start.getTime(), type, label, extraText, sharedBase });
  }

  // ── proximity tooltip (fires from container mousemove) ─────────────────

  // Returns all eventPoints within THRESHOLD_PX of clientX, sorted by distance.
  function findEventsAtPixel(clientX, container, chart) {
    if (!state.eventPoints.length) return [];
    const win = chart.getWindow();
    const startMs = win.start.getTime();
    const endMs   = win.end.getTime();
    if (endMs <= startMs) return [];
    const cRect = container.getBoundingClientRect();
    const leftEl = container.querySelector(".vis-panel.vis-left");
    const leftW  = leftEl ? leftEl.getBoundingClientRect().width : 220;
    const chartW = cRect.width - leftW;
    if (chartW <= 0) return [];
    const THRESHOLD_PX = 7;
    const hits = [];
    for (const ep of state.eventPoints) {
      const frac = (ep.ts - startMs) / (endMs - startMs);
      const evClientX = cRect.left + leftW + frac * chartW;
      const dist = Math.abs(clientX - evClientX);
      if (dist <= THRESHOLD_PX) hits.push({ ep, dist });
    }
    hits.sort((a, b) => a.dist - b.dist);
    return hits.map(h => h.ep);
  }

  // Build tooltip text for one or more events (same tick events share base info).
  function buildTooltipText(evs) {
    if (evs.length === 1) {
      return evs[0].label + "\n\n" + evs[0].sharedBase + "\n" + evs[0].extraText;
    }
    // Group by timestamp so we print shared base only once per tick.
    const byTs = new Map();
    for (const ev of evs) {
      if (!byTs.has(ev.ts)) byTs.set(ev.ts, []);
      byTs.get(ev.ts).push(ev);
    }
    const sections = [];
    for (const group of byTs.values()) {
      const labels = group.map(ev => ev.label).join("\n");
      const extras = group.map(ev => ev.extraText).filter(Boolean).join("\n");
      sections.push(labels + "\n\n" + group[0].sharedBase + (extras ? "\n" + extras : ""));
    }
    return sections.join("\n\n━━━━━━━━━━━━━━━━━━━━\n\n");
  }

  function setupProximityTooltips() {
    for (const [container, chart] of [
      [$("#timeline"), state.timeline],
      [$("#kvChart"),  state.kvChart],
    ]) {
      container.addEventListener("mousemove", (e) => {
        const evs = findEventsAtPixel(e.clientX, container, chart);
        if (evs.length) showTooltip(buildTooltipText(evs), e.clientX, e.clientY);
        else             hideTooltip();
      });
      container.addEventListener("mouseleave", hideTooltip);
    }
  }

  // ── KV % line ──────────────────────────────────────────────────────────

  function applyKvTick(record) {
    const pct = record.vllm && record.vllm.kv_cache_used_pct;
    const ts = new Date(record.ts);
    if (pct !== null && pct !== undefined) {
      state.kvPoints.add({ x: ts, y: Number(pct), group: "kv" });
    }

    const thresholdPct = offloadThresholdPct(record);
    if (thresholdPct !== null) {
      state.kvPoints.add({ x: ts, y: thresholdPct, group: "threshold" });
    }
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function totalKvGb(record) {
    const vllm = record.vllm || {};
    const adm = record.admission || {};
    const directTotal = finiteNumber(vllm.kv_total_gb);
    if (directTotal !== null && directTotal > 0) return directTotal;

    const used = finiteNumber(vllm.kv_used_gb);
    // admission.C is live free KV, not the offload threshold. Use it only to
    // infer total KV capacity when vLLM exposes pct+free but not total memory.
    const free = finiteNumber(vllm.kv_free_gb) ?? finiteNumber(adm.C);
    if (used !== null && free !== null && used + free > 0) return used + free;

    const pct = finiteNumber(vllm.kv_cache_used_pct);
    if (pct !== null && pct > 0 && used !== null) return used / (pct / 100);
    if (pct !== null && pct < 100 && free !== null) return free / (1 - pct / 100);
    return null;
  }

  function offloadThresholdPct(record) {
    const adm = record.admission || {};
    const offloadThresholdGb = finiteNumber(adm.threshold_gb);
    const total = totalKvGb(record);
    if (offloadThresholdGb === null || total === null || total <= 0) return null;
    return Math.max(0, Math.min(100, 100 * (1 - offloadThresholdGb / total)));
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

    const tickMs = new Date(recordTs).getTime();
    state.tickHistory.push({ ts: tickMs, ...snapshotCounts(record) });
    updateBadges(tickMs);
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
    state.tickHistory = [];
    state.eventPoints = [];
    for (const id of ["liveCount", "offloadedCount", "heapCount", "pressureBadge", "queueCount", "doneCount"]) {
      const el = $("#" + id);
      if (el) el.style.display = "none";
    }
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

  function fmtGb(v) {
    return v === null || v === undefined ? "n/a" : Number(v).toFixed(2);
  }

  function fmtW(v) {
    return v === null || v === undefined ? "n/a" : Number(v).toFixed(2);
  }

  function pressureLabel(adm) {
    if (adm && adm.pressure === true) return "yes";
    const C = finiteNumber(adm && adm.C);
    const threshold = finiteNumber(adm && adm.threshold_gb);
    if (C === null || threshold === null) return "unknown";
    return C <= threshold ? "yes" : "no";
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
