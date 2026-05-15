/* dashboard.js — render live sidecar tick records as a Gantt timeline plus a
 * KV-cache-usage line chart on a shared time axis.
 *
 * The vis-timeline component is the sole time-axis source. The KV chart is a
 * custom SVG drawn into #kvChart that reads the timeline's window every frame,
 * so the two views can never drift apart (an earlier vis.Graph2d sibling was
 * dropped after saved-standalone files showed the two axes diverging).
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
 *   snapshot— a standalone saved HTML file embeds tick records in a
 *             <script type="application/json"> block and opens directly in
 *             replay-style mode without /state, /stream, or a sidecar.log.
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
  const EVENT_LINE_OFFSET_PX = 5;
  const EVENT_LINE_HIT_PX = 4;
  const AGENT_LABEL_PANEL_MIN_PX = 220;
  const AGENT_LABEL_PANEL_PADDING_PX = 28;
  const SNAPSHOT_DATA_ID = "dashboardSnapshotData";
  const SNAPSHOT_VERSION = 1;
  const EVENT_COUNT_IDS = {
    offload: "eventCountOffload",
    admit: "eventCountAdmit",
    readmit: "eventCountReadmit",
    sat: "eventCountSat",
  };

  const KV_CHART_TOP_PAD_PX = 8;
  const KV_CHART_BOTTOM_PAD_PX = 12;
  // vis-timeline's center panel has no right padding; match that here so the
  // KV chart's time scale and the event-line overlay agree pixel-for-pixel
  // with the timeline below.
  const KV_CHART_RIGHT_PAD_PX = 0;
  const KV_AXIS_LABEL_GAP_PX = 6;
  const SVG_NS = "http://www.w3.org/2000/svg";

  const state = {
    timeline: null,
    items: null,
    groups: null,
    kvSeries: { kv: [], threshold: [] },
    kvRedrawScheduled: false,
    agents: new Map(),
    agentOrder: [],
    agentLabelPanelWidth: AGENT_LABEL_PANEL_MIN_PX,
    eventCounter: 0,
    itemCounter: 0,
    eventLineRenderPending: false,
    eventLinesVisible: true,
    lastSatActive: false,
    // Per-type opt-out so users can hide specific kinds of event lines while
    // keeping the rest visible. The master toggle (#eventLinesBtn) overrides
    // this when off.
    eventTypeVisible: { offload: true, admit: true, readmit: true, sat: true },
    eventPoints: [],     // [{id, ts: epochMs, type, label, text}] for event lines + tooltips
    eventCounts: {},
    tickHistory: [],     // [{ts: epoch_ms, count: active_agent_count}]
    records: [],         // rendered tick records, used for standalone export
    latestTick: -1,
    latestTs: null,
    paused: false,
    admissionEnabled: false,
    eventSource: null,
    mode: "live",       // "live" | "replay" | "snapshot"
    replayBounds: null, // {start, end} when in replay/snapshot mode
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

  function setDashboardTitle(modeLabel) {
    const text = `Agent Concurrency · ${modeLabel}`;
    const h1 = $("#dashboardTitle");
    if (h1) h1.textContent = text;
    document.title = text;
  }

  function setTickInfo(tick, ts) {
    $("#tickInfo").textContent = ts ? `tick ${tick} · ${ts}` : "";
  }

  // ── timeline + KV chart bootstrap ──────────────────────────────────────

  function buildCharts() {
    if (!hasVisCharts()) {
      throw new Error("vis-timeline assets are unavailable");
    }
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

    setupKvChart();

    state.timeline.on("rangechange", (props) => {
      if (props.byUser) setPaused(true);
      updateBadges(props.end.getTime());
      scheduleKvRedraw();
      scheduleEventLineRender();
    });
    // vis-timeline emits "changed" after layout (new groups, panel resizes).
    // Re-render the KV chart and event-line layer so they track the timeline.
    state.timeline.on("changed", () => {
      scheduleKvRedraw();
      scheduleEventLineRender();
    });

    setupEventLineLayers();
    setupProximityTooltips();
    window.addEventListener("resize", () => {
      scheduleKvRedraw();
      scheduleEventLineRender();
    });
  }

  // ── KV chart (custom SVG, slaved to the timeline's window) ─────────────

  function setupKvChart() {
    const container = $("#kvChart");
    if (!container) return;
    container.innerHTML = "";
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "kv-chart-svg");
    container.appendChild(svg);
    state.kvSvg = svg;
    scheduleKvRedraw();
  }

  function scheduleKvRedraw() {
    if (state.kvRedrawScheduled) return;
    state.kvRedrawScheduled = true;
    requestAnimationFrame(() => {
      state.kvRedrawScheduled = false;
      redrawKvChart();
    });
  }

  function redrawKvChart() {
    const container = $("#kvChart");
    const svg = state.kvSvg;
    if (!container || !svg || !state.timeline) return;

    const win = state.timeline.getWindow();
    const tStart = win.start.getTime();
    const tEnd   = win.end.getTime();
    const tSpan  = tEnd - tStart;
    if (!(tSpan > 0)) return;

    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw <= 0 || ch <= 0) return;

    const leftPad = state.agentLabelPanelWidth || AGENT_LABEL_PANEL_MIN_PX;
    const plotW = Math.max(0, cw - leftPad - KV_CHART_RIGHT_PAD_PX);
    const plotH = Math.max(0, ch - KV_CHART_TOP_PAD_PX - KV_CHART_BOTTOM_PAD_PX);

    svg.setAttribute("width", String(cw));
    svg.setAttribute("height", String(ch));
    svg.setAttribute("viewBox", `0 0 ${cw} ${ch}`);
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    if (plotW <= 0 || plotH <= 0) return;

    const xFor = (t) => leftPad + ((t - tStart) / tSpan) * plotW;
    const yFor = (v) => KV_CHART_TOP_PAD_PX + (1 - Math.max(0, Math.min(100, v)) / 100) * plotH;
    const baseY = KV_CHART_TOP_PAD_PX + plotH;

    // Y-axis: label every 5%, with gridlines at the same cadence.
    const labelStep = 5;
    const gridStep = 5;
    for (let p = 0; p <= 100; p += gridStep) {
      if (p === 0 || p === 100) continue;
      const y = yFor(p);
      const grid = document.createElementNS(SVG_NS, "line");
      grid.setAttribute("x1", String(leftPad));
      grid.setAttribute("y1", y.toFixed(2));
      grid.setAttribute("x2", String(leftPad + plotW));
      grid.setAttribute("y2", y.toFixed(2));
      grid.setAttribute("class", "kv-grid-line");
      svg.appendChild(grid);
    }
    for (let p = labelStep; p < 100; p += labelStep) {
      const y = yFor(p);
      const text = document.createElementNS(SVG_NS, "text");
      text.setAttribute("x", String(leftPad - KV_AXIS_LABEL_GAP_PX));
      text.setAttribute("y", (y + 4).toFixed(2));
      text.setAttribute("text-anchor", "end");
      text.setAttribute("class", "kv-axis-label");
      text.textContent = p + "%";
      svg.appendChild(text);
    }

    // Build line/area paths from a sorted series, clipping to the visible
    // window and including one off-screen point on each side so the line
    // continues to the panel edges.
    const visiblePath = (series) => {
      if (!series.length) return { line: "", points: [], firstX: null, lastX: null };
      const segs = [];
      const pts = [];
      let prev = null;
      let firstX = null;
      let lastX = null;
      let started = false;
      for (let i = 0; i < series.length; i++) {
        const cur = series[i];
        const isVisible = cur.t >= tStart && cur.t <= tEnd;
        const prevVisible = prev && prev.t >= tStart && prev.t <= tEnd;
        if (isVisible || prevVisible || (prev && prev.t < tStart && cur.t > tEnd)) {
          const x = xFor(cur.t);
          const y = yFor(cur.y);
          segs.push((started ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2));
          if (firstX === null) firstX = x;
          lastX = x;
          started = true;
          if (isVisible) pts.push({ x, y });
        } else if (started) {
          break;
        }
        prev = cur;
      }
      return { line: segs.join(" "), points: pts, firstX, lastX };
    };

    // KV used % — area fill + line + circle markers.
    const kv = visiblePath(state.kvSeries.kv);
    if (kv.line) {
      const fill = document.createElementNS(SVG_NS, "path");
      fill.setAttribute(
        "d",
        `M${kv.firstX.toFixed(2)},${baseY.toFixed(2)} ` +
        kv.line.replace(/^M/, "L") +
        ` L${kv.lastX.toFixed(2)},${baseY.toFixed(2)} Z`,
      );
      fill.setAttribute("class", "kv-line-used kv-fill");
      svg.appendChild(fill);

      const line = document.createElementNS(SVG_NS, "path");
      line.setAttribute("d", kv.line);
      line.setAttribute("class", "kv-line-used");
      svg.appendChild(line);

      for (const pt of kv.points) {
        const dot = document.createElementNS(SVG_NS, "circle");
        dot.setAttribute("cx", pt.x.toFixed(2));
        dot.setAttribute("cy", pt.y.toFixed(2));
        dot.setAttribute("r", "3");
        dot.setAttribute("class", "kv-line-used kv-point");
        svg.appendChild(dot);
      }
    }

    // Offload threshold — line only.
    const th = visiblePath(state.kvSeries.threshold);
    if (th.line) {
      const line = document.createElementNS(SVG_NS, "path");
      line.setAttribute("d", th.line);
      line.setAttribute("class", "kv-line-threshold");
      svg.appendChild(line);
    }
  }

  // ── header count badges ────────────────────────────────────────────────
  //
  // tickHistory carries a snapshot per tick so all badges stay aligned with
  // the viewport's right edge (binary search) rather than always showing the
  // latest sidecar tick. Snapshot fields:
  //   ts        — epoch ms
  //   live      — agents in reasoning|tool_call|waiting
  //   reasoning, tool_call, waiting  — phase breakdown of `live`
  //   offloaded — agents currently in offloaded_waiting (KV pushed to CPU)
  //   done      — finished agents
  //   launched  — total agents the sidecar has ever seen (cumulative)
  //   heap      — admission.heap_candidates length (eligible to offload)
  //   C, threshold, pressure — free KV GB, free-KV offload threshold percent,
  //                            and whether the controller is in pressure
  //   queue     — fresh + offloaded_ready waiting to be admitted
  //   vllmPreemptions — cumulative vLLM scheduler preemptions from /metrics

  function snapshotCounts(record) {
    const agents = record.agents || {};
    const vllm = record.vllm || {};
    let reasoning = 0, tool_call = 0, waiting = 0, offloaded = 0, done = 0;
    let launched = 0;
    for (const id of Object.keys(agents)) {
      launched++;
      switch (displayPhaseForAgent(agents[id])) {
        case "reasoning":       reasoning++; break;
        case "tool_call":       tool_call++; break;
        case "waiting":         waiting++;   break;
        case "offloaded_waiting":
          offloaded++;
          break;
        case "done":            done++;      break;
        default:                waiting++;   break;
      }
    }
    const adm = record.admission || {};
    const q = adm.queue || {};
    const C = finiteNumber(adm.C);
    const threshold = thresholdFreePercent(record);
    const thresholdGb = thresholdGbForRecord(record);
    const freePercent = freeKvPercent(record);
    const w = finiteNumber(adm.w);
    const wBeforeOffload = finiteNumber(adm.w_before_offload);
    const wThreshold = finiteNumber(adm.w_threshold);
    const wSource = adm.w_source || null;
    const pressure = adm.pressure === true
      || (freePercent !== null && threshold !== null && freePercent <= threshold)
      || (C !== null && thresholdGb !== null && C <= thresholdGb);
    return {
      live: reasoning + tool_call + waiting,
      reasoning, tool_call, waiting, offloaded, done, launched,
      queue: (q.fresh || 0) + (q.offloaded_ready || 0),
      heap: (adm.heap_candidates || []).length,
      C,
      threshold,
      thresholdGb,
      freePercent,
      w,
      wBeforeOffload,
      wThreshold,
      wSource,
      pressure,
      admissionEnabled: adm.enabled === true,
      vllmPreemptions: vllmSchedulerPreemptions(vllm),
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

  function hideBadge(id) {
    const el = $("#" + id);
    if (el) el.style.display = "none";
  }

  function setAdmissionVisualsVisible(visible) {
    for (const el of [
      $("#eventLinesBtn"),
      $("#eventTypeFilter"),
      $("#eventLegendTitle"),
      $("#kvThresholdLegendChip"),
      document.querySelector(".event-offload"),
      document.querySelector(".event-admit"),
      document.querySelector(".event-readmit"),
      document.querySelector(".event-sat"),
    ]) {
      if (el) el.style.display = visible ? "" : "none";
    }
  }

  function updateBadges(timeMs) {
    const s = snapshotAt(timeMs);
    if (!s) return;
    const wBeforeOffloadText = s.wBeforeOffload === null
      ? ""
      : `w_before_offload: ${fmtW(s.wBeforeOffload)}\n`;
    setBadge(
      "liveCount",
      `live: ${s.live}`,
      `reasoning: ${s.reasoning}\ntool_call: ${s.tool_call}\nwaiting:   ${s.waiting}\n\nlaunched: ${s.launched}`,
    );
    if (s.admissionEnabled) {
      setBadge("offloadedCount", `offloaded: ${s.offloaded}`,
        "agents whose KV is currently offloaded to CPU (offloaded_waiting)");
      setBadge("heapCount", `heap: ${s.heap}`,
        "agents currently in the offload heap (eligible to be offloaded)");
      setBadge("pressureBadge", `C: ${fmtGb(s.C)} / W: ${fmtW(s.w)} / T: ${fmtPct(s.threshold)}`,
        "free KV GB / admission headroom W / free-KV threshold percent\n" +
        `free_pct: ${fmtPct(s.freePercent)}\nthreshold_gb: ${fmtGb(s.thresholdGb)}\n` +
        `w: ${fmtW(s.w)}\nw_threshold: ${fmtW(s.wThreshold)}\nw_source: ${fmt(s.wSource)}\n` +
        wBeforeOffloadText +
        "controller offloads only when free KV percent <= threshold");
      const pressureEl = $("#pressureBadge");
      if (pressureEl) {
        pressureEl.classList.toggle("badge-pressure-active", s.pressure);
      }
      setBadge("queueCount", `queue: ${s.queue}`,
        "fresh + offloaded_ready agents waiting to be admitted");
    } else {
      hideBadge("offloadedCount");
      hideBadge("heapCount");
      hideBadge("pressureBadge");
      hideBadge("queueCount");
    }
    setBadge("doneCount", `done: ${s.done} / ${s.launched}`,
      "completed agents / total launched");
    setBadge("vllmPreemptCount", `vllm preempt: ${fmtCount(s.vllmPreemptions)}`,
      "cumulative scheduler preemptions reported by vLLM /metrics\n" +
      "separate from sidecar offload/admission decisions");
  }

  function autoScroll(nowDate) {
    if (state.paused || state.mode === "replay") return;
    const start = new Date(nowDate.getTime() - VIEWPORT_PAST_MS);
    const end   = new Date(nowDate.getTime() + VIEWPORT_AHEAD_MS);
    state.timeline.setWindow(start, end, { animation: false });
  }

  function setPaused(p) {
    if (state.paused === p) return;
    state.paused = p;
    const btn = $("#pauseBtn");
    btn.classList.toggle("paused", p);
    btn.textContent = p ? "Resume auto-scroll" : "Pause auto-scroll";
  }

  function setEventLinesVisible(visible) {
    state.eventLinesVisible = visible;
    const btn = $("#eventLinesBtn");
    if (btn) {
      btn.classList.toggle("lines-hidden", !visible);
      btn.setAttribute("aria-pressed", String(!visible));
      btn.textContent = visible ? "Hide event lines" : "Show event lines";
    }
    if (visible) {
      scheduleEventLineRender();
    } else {
      clearEventLineLayers();
      hideTooltip();
    }
  }

  // ── per-agent phase rendering ──────────────────────────────────────────

  function ensureAgentGroup(agentId, agent, recordTs) {
    if (state.agents.has(agentId)) return state.agents.get(agentId);
    const groupId = `agent::${agentId}`;
    const startMs = agentStartMs(agent, recordTs);
    const recordMs = parseTimeMs(recordTs) ?? Date.now();
    const initialLabel = agentRuntimeLabel(agentId, startMs, null, recordMs);
    state.agentOrder.push(agentId);
    updateAgentLabelPanelWidth(initialLabel);
    state.groups.add({
      id: groupId,
      content: initialLabel,
      title: agentGroupTitle(agentId, agent, startMs, null),
      order: state.agentOrder.length,
    });
    const entry = {
      groupId,
      startedAtMs: startMs,
      finishedAtMs: null,
      labelText: initialLabel,
      activeItemId: null,
      activeStart: null,
      activePhase: null,
      lastRecordTs: null,
      lastKvGb: null,
    };
    state.agents.set(agentId, entry);
    return entry;
  }

  function updateAgentRuntimeLabel(entry, agentId, agent, recordTs) {
    const recordMs = parseTimeMs(recordTs) ?? Date.now();
    const explicitStartMs = parseTimeMs(agent.started_at);
    if (explicitStartMs !== null && explicitStartMs < entry.startedAtMs) {
      entry.startedAtMs = explicitStartMs;
    }
    if (entry.finishedAtMs === null) {
      const explicitFinishMs = parseTimeMs(agent.finished_at);
      if (explicitFinishMs !== null) {
        entry.finishedAtMs = explicitFinishMs;
      } else if (agent.state === "done") {
        entry.finishedAtMs = parseTimeMs(agent.state_since) ?? recordMs;
      }
    }
    const label = agentRuntimeLabel(agentId, entry.startedAtMs, entry.finishedAtMs, recordMs);
    if (label === entry.labelText) return;
    entry.labelText = label;
    updateAgentLabelPanelWidth(label);
    state.groups.update({
      id: entry.groupId,
      content: label,
      title: agentGroupTitle(agentId, agent, entry.startedAtMs, entry.finishedAtMs),
    });
  }

  function agentStartMs(agent, recordTs) {
    const recordMs = parseTimeMs(recordTs);
    return parseTimeMs(agent.started_at)
      ?? parseTimeMs(agent.state_since)
      ?? recordMs
      ?? Date.now();
  }

  function agentRuntimeLabel(agentId, startMs, finishedMs, recordMs) {
    if (finishedMs !== null) {
      return `${agentId} (E2E: ${formatDurationSeconds(finishedMs - startMs)} secs)`;
    }
    return `${agentId} (elapsed: ${formatDurationSeconds(recordMs - startMs)} secs)`;
  }

  function agentGroupTitle(agentId, agent, startMs, finishedMs) {
    const lines = [agentId];
    if (agent.task_id) lines.push(`task: ${agent.task_id}`);
    lines.push(`started: ${new Date(startMs).toISOString()}`);
    if (finishedMs !== null) {
      lines.push(`finished: ${new Date(finishedMs).toISOString()}`);
      lines.push(`E2E: ${formatDurationSeconds(finishedMs - startMs)} secs`);
    }
    return lines.join("\n");
  }

  function updateAgentLabelPanelWidth(agentId) {
    const nextWidth = Math.max(
      AGENT_LABEL_PANEL_MIN_PX,
      Math.ceil(measureAgentLabel(agentId) + AGENT_LABEL_PANEL_PADDING_PX),
    );
    if (nextWidth <= state.agentLabelPanelWidth) return;
    state.agentLabelPanelWidth = nextWidth;
    document.documentElement.style.setProperty(
      "--agent-label-panel-width",
      state.agentLabelPanelWidth + "px",
    );
    scheduleKvRedraw();
    scheduleEventLineRender();
  }

  function resetAgentLabelPanelWidth() {
    state.agentLabelPanelWidth = AGENT_LABEL_PANEL_MIN_PX;
    document.documentElement.style.setProperty(
      "--agent-label-panel-width",
      state.agentLabelPanelWidth + "px",
    );
  }

  function phaseItemContent() {
    // Labels are drawn by CSS pseudo-elements so text width cannot stretch a
    // short phase bar and make sub-second work look like multi-second work.
    return "";
  }

  function measureAgentLabel(text) {
    if (!measureAgentLabel.canvas) {
      measureAgentLabel.canvas = document.createElement("canvas");
    }
    const ctx = measureAgentLabel.canvas.getContext("2d");
    if (!ctx) return String(text).length * 8;
    ctx.font = '500 13px ui-monospace, SFMono-Regular, Menlo, monospace';
    return ctx.measureText(String(text)).width;
  }

  function applyAgentTick(agentId, agent, recordTs) {
    const entry = ensureAgentGroup(agentId, agent, recordTs);
    updateAgentRuntimeLabel(entry, agentId, agent, recordTs);
    const phase = displayPhaseForAgent(agent);
    const phaseStart = agent.state_since || recordTs;
    const previousRecordTs = entry.lastRecordTs;
    entry.lastKvGb = (agent.kv_gb !== undefined) ? agent.kv_gb : entry.lastKvGb;

    if (entry.activeItemId !== null
        && entry.activePhase === phase
        && entry.activeStart === phaseStart) {
      state.items.update({
        id: entry.activeItemId,
        end: new Date(recordTs),
        title: phaseTooltip(agentId, agent, phaseStart, recordTs),
      });
      entry.lastRecordTs = recordTs;
      return;
    }

    const bridgeMissedReasoning =
      entry.activeItemId !== null
      && entry.activePhase === "waiting"
      && phase === "tool_call"
      && previousRecordTs !== null
      && parseTimeMs(previousRecordTs) !== null
      && parseTimeMs(phaseStart) !== null
      && parseTimeMs(previousRecordTs) < parseTimeMs(phaseStart);

    if (entry.activeItemId !== null) {
      state.items.update({
        id: entry.activeItemId,
        end: new Date(bridgeMissedReasoning ? previousRecordTs : phaseStart),
      });
      if (bridgeMissedReasoning) {
        addSyntheticReasoningBridge(entry, agentId, agent, previousRecordTs, phaseStart);
      }
    }
    const itemId = `phase::${agentId}::${++state.itemCounter}`;
    state.items.add({
      id: itemId,
      group: entry.groupId,
      start: new Date(phaseStart),
      end: new Date(recordTs),
      content: phaseItemContent(),
      className: "phase-" + phase,
      title: phaseTooltip(agentId, agent, phaseStart, recordTs),
    });
    entry.activeItemId = itemId;
    entry.activeStart = phaseStart;
    entry.activePhase = phase;
    entry.lastRecordTs = recordTs;
  }

  function addSyntheticReasoningBridge(entry, agentId, agent, startTs, endTs) {
    const itemId = `phase::${agentId}::${++state.itemCounter}`;
    const bridgeAgent = Object.assign({}, agent, { state: "reasoning" });
    state.items.add({
      id: itemId,
      group: entry.groupId,
      start: new Date(startTs),
      end: new Date(endTs),
      content: phaseItemContent(),
      className: "phase-reasoning",
      title: phaseTooltip(agentId, bridgeAgent, startTs, endTs),
    });
  }

  function displayPhaseForAgent(agent) {
    const phase = agent.state || "waiting";
    if (
      phase === "waiting"
      && agent.admission_state === "tool_complete_pending_release"
    ) {
      return "reasoning";
    }
    return phase;
  }

  function phaseTooltip(agentId, agent, phaseStart, recordTs) {
    const start = new Date(phaseStart);
    const end = new Date(recordTs);
    const durSec = Math.max(0, (end - start) / 1000).toFixed(1);
    const lines = [
      `agent: ${agentId}`,
      `phase: ${displayPhaseForAgent(agent)}`,
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
    const reasons = adm.reasons || [];
    const isSaturated = reasons.indexOf("saturation_guard") !== -1;
    const wasSaturated = state.lastSatActive === true;

    for (const ev of (adm.offloads || [])) {
      if (ev.offloaded === true) {
        const offloadScore = ev.offload_score_gb_s ?? ev.e_s;
        addEvent(ts, "offload", `OFFLOAD: ${ev.agent_id}`,
          `agent: ${ev.agent_id}\n` +
          `freed_gb: ${fmt(ev.freed_gb)}\n` +
          `freed_blocks: ${fmt(ev.freed_blocks)}\n` +
          `free_blocks: ${fmt(ev.free_blocks_before)} -> ${fmt(ev.free_blocks_after)}\n` +
          `freed_gb_source: ${fmt(ev.freed_gb_source)}\n` +
          `pending: ${fmt(ev.pending)}\n` +
          `held_requests: ${fmt(ev.held_requests)}\n` +
          `known_blocks: ${fmt(ev.known_blocks)}\n` +
          `offload_jobs: ${fmt(ev.offload_jobs)}\n` +
          `reason: ${fmt(ev.reason)}\n` +
          `offload_score_gb_s: ${fmt(offloadScore)}\n` +
          `kv_gb: ${fmt(ev.kv_gb)}\n` +
          `tool_elapsed_s: ${fmt(ev.tool_elapsed_s)}\n` +
          `predicted_remaining_s: ${fmt(ev.predicted_remaining_s)}`,
          tooltipBase);
      }
    }
    for (const ad of (adm.admissions || [])) {
      if (!ad.admitted) continue;
      const wasOffloaded = ad.previously_offloaded === true;
      const type = wasOffloaded ? "readmit" : "admit";
      const tag = wasOffloaded
        ? `READMIT: ${ad.agent_id}`
        : `ADMIT: ${ad.agent_id}`;
      const adTs = eventTimestamp(ts, ad, ["admitted_at", "admitted_since", "ts"]);
      addEvent(adTs, type, tag,
        `agent: ${ad.agent_id}\n` +
        `previously_offloaded: ${wasOffloaded}\n` +
        `w: ${fmt(ad.w)}\n` +
        `w_threshold: ${fmt(ad.w_threshold)}\n` +
        `w_source: ${fmt(ad.w_source)}\n` +
        `admitted_at: ${fmt(ad.admitted_at)}`,
        tooltipBase);
    }
    // Emit a SAT event only on the rising edge of a saturation streak.
    // Previously every tick under saturation produced a vertical line on top
    // of the previous one, drowning the chart in dashed yellow.
    if (isSaturated && !wasSaturated) {
      addEvent(ts, "sat", "SAT",
        `reasons: ${reasons.join(", ")}`,
        tooltipBase);
    }
    state.lastSatActive = isSaturated;
  }

  function baseTooltip(record) {
    const adm = record.admission || {};
    const wBeforeOffload = finiteNumber(adm.w_before_offload);
    return [
      `ts:     ${record.ts}`,
      `tick:   ${record.tick}`,
      `C:      ${fmt(adm.C)}`,
      `C_percent: ${fmt(adm.C_percent)}`,
      `threshold_percent: ${fmt(adm.threshold_percent)}`,
      `threshold_gb: ${fmt(adm.threshold_gb)}`,
      `pressure: ${pressureLabel(record)}`,
      `s_t:    ${fmt(adm.s_t)}`,
      `s_prev: ${fmt(adm.s_prev)}`,
      `w:      ${fmt(adm.w)}`,
      `w_threshold: ${fmt(adm.w_threshold)}`,
      `w_source: ${fmt(adm.w_source)}`,
      ...(wBeforeOffload === null ? [] : [
        `w_before_offload: ${fmt(adm.w_before_offload)}`,
      ]),
      `active: ${fmt(adm.active_agents)} / ${fmt(adm.max_active_agents)}`,
      `active_slots: ${fmt(adm.active_agent_slots)}`,
      `queue:  fresh=${fmt(adm.queue && adm.queue.fresh)} ` +
        `ready=${fmt(adm.queue && adm.queue.offloaded_ready)} ` +
        `pending_tool=${fmt(adm.queue && adm.queue.offloaded_pending_tool)}`,
      `heap_candidates: ${(adm.heap_candidates || []).length}`,
    ].join("\n");
  }

  function setupEventLineLayers() {
    getEventLineLayer($("#kvChart"));
    getEventLineLayer($("#timeline"));
  }

  function getEventLineLayer(container) {
    let layer = container.querySelector(".event-line-layer");
    if (!layer) {
      layer = document.createElement("div");
      layer.className = "event-line-layer";
      container.appendChild(layer);
    }
    return layer;
  }

  // vis-timeline can extend its `.vis-foreground` (group rows) below the
  // container's static box. The event-line layer is positioned with
  // `inset: 0` of the container, so without this sync the vertical lines
  // (which span `top: 0; bottom: 0` of the layer) stop at the container's
  // box and miss the lowest rows. Stretch the layer to the bottom of the
  // foreground so lines reach every group.
  function syncEventLineLayerHeight(container, layer) {
    const containerRect = container.getBoundingClientRect();
    const fg = container.querySelector(".vis-panel.vis-center .vis-foreground")
            || container.querySelector(".vis-foreground");
    const reference = fg
      || container.querySelector(".vis-panel.vis-center")
      || null;
    if (!reference || !containerRect.height) {
      layer.style.removeProperty("bottom");
      layer.style.removeProperty("height");
      return;
    }
    const refRect = reference.getBoundingClientRect();
    const desired = Math.max(
      containerRect.height,
      Math.ceil(refRect.bottom - containerRect.top),
    );
    layer.style.bottom = "auto";
    layer.style.height = desired + "px";
  }

  function scheduleEventLineRender() {
    if (!state.eventLinesVisible) return;
    if (state.eventLineRenderPending) return;
    state.eventLineRenderPending = true;
    requestAnimationFrame(() => {
      state.eventLineRenderPending = false;
      renderEventLines();
    });
  }

  function renderEventLines() {
    if (!state.eventLinesVisible) {
      clearEventLineLayers();
      return;
    }
    for (const [container, chart] of [
      [$("#timeline"), state.timeline],
      [$("#kvChart"),  state.timeline],
    ]) {
      const layer = getEventLineLayer(container);
      layer.textContent = "";
      syncEventLineLayerHeight(container, layer);
      for (const hit of eventLineEntries(container, chart)) {
        const line = document.createElement("div");
        line.className = `event-line event-line-${hit.ep.type}`;
        line.style.left = hit.x + "px";
        line.dataset.eventId = hit.ep.id;
        line.dataset.eventType = hit.ep.type;
        layer.appendChild(line);
      }
    }
  }

  function clearEventLineLayers() {
    for (const container of [$("#timeline"), $("#kvChart")]) {
      const layer = container && container.querySelector(".event-line-layer");
      if (layer) layer.textContent = "";
    }
  }

  function resetEventCounts() {
    state.eventCounts = {};
    for (const type of Object.keys(EVENT_COUNT_IDS)) {
      state.eventCounts[type] = 0;
      updateEventCount(type);
    }
  }

  function incrementEventCount(type) {
    state.eventCounts[type] = (state.eventCounts[type] || 0) + 1;
    updateEventCount(type);
  }

  function updateEventCount(type) {
    const el = $("#" + EVENT_COUNT_IDS[type]);
    if (el) el.textContent = String(state.eventCounts[type] || 0);
  }

  function addEvent(ts, type, label, extraText, sharedBase) {
    const id = `evt::${++state.eventCounter}`;
    const start = new Date(ts);
    // Store label, event-specific extra, and shared base separately so same-tick
    // events can be grouped into one tooltip without repeating the base fields.
    state.eventPoints.push({ id, ts: start.getTime(), type, label, extraText, sharedBase });
    incrementEventCount(type);
    scheduleEventLineRender();
  }

  function eventTimestamp(recordTs, event, keys) {
    for (const key of keys) {
      const value = event && event[key];
      if (parseTimeMs(value) !== null) return value;
    }
    return recordTs;
  }

  // ── proximity tooltip (fires from container mousemove) ─────────────────

  function eventAxisLayout(container, chart) {
    if (!state.eventPoints.length) return null;
    const win = chart.getWindow();
    const startMs = win.start.getTime();
    const endMs   = win.end.getTime();
    if (endMs <= startMs) return null;
    const leftEl = container.querySelector(".vis-panel.vis-left");
    const leftW  = leftEl ? leftEl.getBoundingClientRect().width : state.agentLabelPanelWidth;
    const chartW = container.clientWidth - leftW;
    if (chartW <= 0) return null;
    return { startMs, endMs, leftW, chartW };
  }

  function isEventTypeVisible(type) {
    return state.eventTypeVisible[type] !== false;
  }

  function eventLineEntries(container, chart) {
    const layout = eventAxisLayout(container, chart);
    if (!layout) return [];
    const groups = new Map();
    for (const ep of state.eventPoints) {
      if (!isEventTypeVisible(ep.type)) continue;
      const frac = (ep.ts - layout.startMs) / (layout.endMs - layout.startMs);
      if (frac < -0.02 || frac > 1.02) continue;
      const key = String(ep.ts);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push({ ep, frac });
    }
    const entries = [];
    for (const group of groups.values()) {
      const count = group.length;
      for (let i = 0; i < count; i++) {
        const offset = count === 1
          ? 0
          : (i - (count - 1) / 2) * EVENT_LINE_OFFSET_PX;
        const x = layout.leftW + group[i].frac * layout.chartW + offset;
        entries.push({ ep: group[i].ep, x });
      }
    }
    return entries;
  }

  // Returns the closest rendered event line(s) to clientX.
  function findEventsAtPixel(clientX, container, chart) {
    if (!state.eventLinesVisible) return [];
    if (!state.eventPoints.length) return [];
    const cRect = container.getBoundingClientRect();
    const hits = [];
    for (const entry of eventLineEntries(container, chart)) {
      const evClientX = cRect.left + entry.x;
      const dist = Math.abs(clientX - evClientX);
      if (dist <= EVENT_LINE_HIT_PX) hits.push({ ep: entry.ep, dist });
    }
    hits.sort((a, b) => a.dist - b.dist);
    if (!hits.length) return [];
    const best = hits[0].dist;
    return hits
      .filter(h => Math.abs(h.dist - best) < 1)
      .map(h => h.ep);
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
      [$("#kvChart"),  state.timeline],
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
    const ts = parseTimeMs(record.ts);
    if (ts === null) return;
    const pct = record.vllm && record.vllm.kv_cache_used_pct;
    if (pct !== null && pct !== undefined) {
      appendKvSeriesPoint(state.kvSeries.kv, ts, Number(pct));
    }
    const thresholdPct = offloadThresholdPct(record);
    if (thresholdPct !== null) {
      appendKvSeriesPoint(state.kvSeries.threshold, ts, thresholdPct);
    }
    scheduleKvRedraw();
  }

  // Keep series sorted by time (replay loads in order; SSE may rarely emit
  // out-of-order frames, so we splice to maintain a clean monotonic series).
  function appendKvSeriesPoint(series, t, y) {
    if (!series.length || series[series.length - 1].t <= t) {
      series.push({ t, y });
      return;
    }
    let lo = 0, hi = series.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (series[mid].t <= t) lo = mid + 1; else hi = mid;
    }
    series.splice(lo, 0, { t, y });
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }

  function totalKvGb(record) {
    const vllm = record.vllm || {};
    const adm = record.admission || {};
    const directTotal = finiteNumber(vllm.kv_total_gb) ?? finiteNumber(adm.kv_total_gb);
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

  function thresholdFreePercent(record) {
    const adm = record.admission || {};
    const direct = finiteNumber(adm.threshold_percent);
    if (direct !== null) return Math.max(0, Math.min(100, direct));

    const thresholdGb = finiteNumber(adm.threshold_gb);
    const total = totalKvGb(record);
    if (thresholdGb === null || total === null || total <= 0) return null;
    return Math.max(0, Math.min(100, 100 * thresholdGb / total));
  }

  function thresholdGbForRecord(record) {
    const adm = record.admission || {};
    const direct = finiteNumber(adm.threshold_gb);
    if (direct !== null) return direct;

    const thresholdPct = thresholdFreePercent(record);
    const total = totalKvGb(record);
    if (thresholdPct === null || total === null || total <= 0) return null;
    return total * thresholdPct / 100;
  }

  function freeKvPercent(record) {
    const adm = record.admission || {};
    const direct = finiteNumber(adm.C_percent);
    if (direct !== null) return Math.max(0, Math.min(100, direct));

    const free = finiteNumber(adm.C);
    const total = totalKvGb(record);
    if (free !== null && total !== null && total > 0) {
      return Math.max(0, Math.min(100, 100 * free / total));
    }

    const usedPct = finiteNumber(record.vllm && record.vllm.kv_cache_used_pct);
    if (usedPct !== null) return Math.max(0, Math.min(100, 100 - usedPct));
    return null;
  }

  function offloadThresholdPct(record) {
    const freeThresholdPct = thresholdFreePercent(record);
    if (freeThresholdPct === null) return null;
    return Math.max(0, Math.min(100, 100 - freeThresholdPct));
  }

  function vllmSchedulerPreemptions(vllm) {
    const candidates = [
      vllm.scheduler_preemptions_total,
      vllm.num_scheduler_preemptions_total,
      vllm.num_preemptions_total,
      vllm.preemptions_total,
      vllm.preemption_count_total,
    ];
    for (const value of candidates) {
      const n = finiteNumber(value);
      if (n !== null) return Math.max(0, Math.floor(n));
    }
    return null;
  }

  // ── tick application ───────────────────────────────────────────────────

  function applyTick(record) {
    if (typeof record.tick === "number") {
      if (record.tick <= state.latestTick) return; // dedupe / out-of-order
      state.latestTick = record.tick;
    }
    state.records.push(record);
    state.latestTs = record.ts;
    setTickInfo(record.tick, record.ts);
    if (record.admission && record.admission.enabled === true) {
      state.admissionEnabled = true;
    }
    setAdmissionVisualsVisible(state.admissionEnabled);

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
    } else if (state.mode === "replay" || state.mode === "snapshot") {
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
    clearEventLineLayers();
    resetEventCounts();
    resetAgentLabelPanelWidth();
    state.items.clear();
    state.groups.clear();
    state.kvSeries.kv.length = 0;
    state.kvSeries.threshold.length = 0;
    scheduleKvRedraw();
    state.agents.clear();
    state.agentOrder = [];
    state.eventCounter = 0;
    state.itemCounter = 0;
    state.latestTick = -1;
    state.latestTs = null;
    state.replayBounds = null;
    state.admissionEnabled = false;
    state.tickHistory = [];
    state.records = [];
    state.eventPoints = [];
    state.lastSatActive = false;
    setAdmissionVisualsVisible(false);
    for (const id of [
      "liveCount",
      "offloadedCount",
      "heapCount",
      "pressureBadge",
      "queueCount",
      "doneCount",
      "vllmPreemptCount",
    ]) {
      const el = $("#" + id);
      if (el) el.style.display = "none";
    }
  }

  function fitReplayWindow() {
    if (!state.replayBounds) return;
    const span = state.replayBounds.end - state.replayBounds.start;
    const pad = Math.max(1000, span * 0.02);
    const s = new Date(state.replayBounds.start - pad);
    const e = new Date(state.replayBounds.end   + pad);
    state.timeline.setWindow(s, e, { animation: false });
    scheduleKvRedraw();
    scheduleEventLineRender();
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
    setDashboardTitle("Replay");
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
    fitReplayWindow();
    setStatus(
      `replay · ${parsed} tick(s)` + (skipped ? ` · ${skipped} skipped` : ""),
      "status-live",
    );
    // Saved standalone files have no /state to return to; keep the button
    // hidden even after a replay so users can't dead-end on file://.
    $("#liveBtn").hidden = isStandaloneFile();
  }

  function loadSnapshotRecords(ticks, meta) {
    state.mode = "snapshot";
    setDashboardTitle("Snapshot");
    setPaused(true);
    resetCharts();

    let parsed = 0;
    let skipped = 0;
    for (const rec of ticks || []) {
      try { applyTick(rec); parsed++; }
      catch (err) { skipped++; }
    }
    fitReplayWindow();
    const label = meta && (meta.source || meta.source_mode)
      ? ` · ${meta.source || meta.source_mode}`
      : "";
    setStatus(
      `snapshot · ${parsed} tick(s)` + (skipped ? ` · ${skipped} skipped` : "") + label,
      "status-live",
    );
    $("#liveBtn").hidden = true;
  }

  // True when this page was opened from a saved standalone HTML snapshot.
  // Detected by the presence of the embedded JSON payload, which is added by
  // buildStandaloneHtml() and never injected into the live-served index.html.
  // Used to gate "Back to live" — there is no /state to return to from file://.
  function isStandaloneFile() {
    return !!document.getElementById(SNAPSHOT_DATA_ID);
  }

  function readEmbeddedSnapshot() {
    const el = document.getElementById(SNAPSHOT_DATA_ID);
    if (!el) return null;
    try {
      const payload = JSON.parse(el.textContent || "null");
      if (Array.isArray(payload)) return { ticks: payload, meta: {} };
      if (payload && Array.isArray(payload.ticks)) {
        return { ticks: payload.ticks, meta: payload.meta || {} };
      }
    } catch (err) {
      console.error("bad embedded dashboard snapshot", err);
    }
    return { ticks: [], meta: { source: "invalid embedded data" } };
  }

  function hasVisCharts() {
    return Boolean(
      window.vis &&
      typeof window.vis.DataSet === "function" &&
      typeof window.vis.Timeline === "function" &&
      typeof window.vis.Graph2d === "function"
    );
  }

  function showStartupError(title, detail) {
    console.error(title, detail || "");
    setDashboardTitle("Offline");
    setStatus(title, "status-error");
    const main = document.querySelector("main");
    if (!main) return;
    main.innerHTML = "";
    const panel = document.createElement("section");
    panel.className = "offline-fallback";
    const h2 = document.createElement("h2");
    h2.textContent = title;
    const p = document.createElement("p");
    p.textContent = detail || "The dashboard could not start.";
    panel.append(h2, p);
    main.appendChild(panel);
  }

  function renderSnapshotFallback(ticks, meta, reason) {
    state.mode = "snapshot";
    setDashboardTitle("Snapshot");
    setPaused(true);
    state.records = Array.isArray(ticks) ? ticks.slice() : [];
    state.latestTick = state.records.reduce((latest, rec) => (
      typeof rec.tick === "number" ? Math.max(latest, rec.tick) : latest
    ), -1);
    const last = state.records[state.records.length - 1] || {};
    state.latestTs = last.ts || null;
    setTickInfo(last.tick, last.ts);
    setStatus(`snapshot fallback · ${state.records.length} tick(s)`, "status-error");
    const liveBtn = $("#liveBtn");
    if (liveBtn) liveBtn.hidden = true;

    const main = document.querySelector("main");
    if (!main) return;
    main.innerHTML = "";
    const panel = document.createElement("section");
    panel.className = "offline-fallback";
    const title = document.createElement("h2");
    title.textContent = "Snapshot data loaded";
    const detail = document.createElement("p");
    detail.textContent = reason ||
      "Interactive charts are unavailable because dashboard assets did not load.";
    const summary = document.createElement("p");
    summary.textContent = snapshotFallbackSummary(state.records, meta);
    panel.append(title, detail, summary, snapshotFallbackTable(state.records));
    main.appendChild(panel);
  }

  function snapshotFallbackSummary(records, meta) {
    const first = records[0] || {};
    const last = records[records.length - 1] || {};
    const source = meta && (meta.source || meta.source_mode)
      ? ` · source: ${meta.source || meta.source_mode}`
      : "";
    return `ticks: ${records.length} · first: ${first.ts || "n/a"} · last: ${last.ts || "n/a"}${source}`;
  }

  function snapshotFallbackTable(records) {
    const table = document.createElement("table");
    table.className = "offline-fallback-table";
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr><th>tick</th><th>ts</th><th>C</th><th>w</th><th>active</th><th>events</th></tr>";
    const tbody = document.createElement("tbody");
    const recent = records.slice(-200);
    for (const rec of recent) {
      const adm = rec.admission || {};
      const events = [];
      if ((adm.offloads || []).some((ev) => ev && ev.offloaded === true)) events.push("OFFLOAD");
      if ((adm.admissions || []).some((ev) => ev && ev.admitted === true && ev.previously_offloaded !== true)) events.push("ADMIT");
      if ((adm.admissions || []).some((ev) => ev && ev.admitted === true && ev.previously_offloaded === true)) events.push("READMIT");
      if ((adm.reasons || []).includes("saturation_guard")) events.push("SAT");
      const row = document.createElement("tr");
      for (const value of [
        fmt(rec.tick),
        fmt(rec.ts),
        fmt(adm.C),
        fmt(adm.w),
        fmt(adm.active_agents),
        events.join(", "),
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      }
      tbody.appendChild(row);
    }
    table.append(thead, tbody);
    return table;
  }

  // ── standalone snapshot export ─────────────────────────────────────────

  async function saveStandaloneHtml() {
    if (!state.records.length) {
      setStatus("snapshot: no ticks loaded", "status-error");
      return;
    }

    const btn = $("#saveSnapshotBtn");
    if (btn) btn.disabled = true;
    const filename = snapshotFilename();
    let fileHandle = null;
    try {
      if (supportsSaveFilePicker()) {
        setStatus("snapshot: choose save location…", "status-pending");
        try {
          fileHandle = await window.showSaveFilePicker({
            suggestedName: filename,
            types: [
              {
                description: "HTML file",
                accept: { "text/html": [".html"] },
              },
            ],
          });
        } catch (err) {
          if (isFilePickerAbort(err)) {
            setStatus("snapshot save cancelled", "status-pending");
            return;
          }
          console.warn("save picker unavailable", err);
        }
      }

      setStatus(`snapshot: bundling ${state.records.length} tick(s)…`, "status-pending");
      const html = await buildStandaloneHtml();
      const blob = new Blob([html], { type: "text/html;charset=utf-8" });
      const savedWithPicker = await saveSnapshotBlob(blob, filename, fileHandle);
      if (!savedWithPicker) {
        setStatus(`snapshot download started · ${state.records.length} tick(s)`, "status-live");
        return;
      }
      setStatus(`snapshot saved · ${state.records.length} tick(s)`, "status-live");
    } catch (err) {
      console.error("snapshot export failed", err);
      const msg = err && err.message ? err.message : String(err);
      setStatus(`snapshot save failed: ${msg}`, "status-error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function supportsSaveFilePicker() {
    return typeof window.showSaveFilePicker === "function";
  }

  function isFilePickerAbort(err) {
    return err && err.name === "AbortError";
  }

  async function saveSnapshotBlob(blob, filename, fileHandle) {
    if (fileHandle) {
      const writable = await fileHandle.createWritable();
      try {
        await writable.write(blob);
        await writable.close();
      } catch (err) {
        try { await writable.abort(); } catch (_) {}
        throw err;
      }
      return true;
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
    return false;
  }

  async function buildStandaloneHtml() {
    const [visCss, dashboardCss, visJs, dashboardJs] = await Promise.all([
      assetTextForExport("#visCssAsset", "/static/vis-timeline-graph2d.min.css"),
      assetTextForExport("#dashboardCssAsset", "/static/dashboard.css"),
      assetTextForExport("#visJsAsset", "/static/vis-timeline-graph2d.min.js"),
      assetTextForExport("#dashboardJsAsset", "/static/dashboard.js"),
    ]);
    validateStandaloneAsset("vis-timeline CSS", visCss);
    validateStandaloneAsset("dashboard CSS", dashboardCss);
    validateStandaloneAsset("vis-timeline JS", visJs, ["Graph2d", "Timeline"]);
    validateStandaloneAsset("dashboard JS", dashboardJs, ["SNAPSHOT_DATA_ID"]);

    const payload = {
      version: SNAPSHOT_VERSION,
      meta: {
        created_at: new Date().toISOString(),
        source_mode: state.mode,
        tick_count: state.records.length,
        first_ts: state.records[0] && state.records[0].ts,
        last_ts: state.records[state.records.length - 1] &&
          state.records[state.records.length - 1].ts,
      },
      ticks: state.records,
    };

    return "<!DOCTYPE html>\n" +
      "<html lang=\"en\">\n" +
      "<head>\n" +
      "  <meta charset=\"utf-8\">\n" +
      "  <title>Agent Concurrency · Snapshot</title>\n" +
      "  <style id=\"visCssAsset\">\n" + safeInlineStyle(visCss) + "\n  </style>\n" +
      "  <style id=\"dashboardCssAsset\">\n" + safeInlineStyle(dashboardCss) + "\n  </style>\n" +
      "</head>\n" +
      "<body>\n" +
      currentDashboardShellHtml() + "\n" +
      "  <script id=\"" + SNAPSHOT_DATA_ID + "\" type=\"application/json\">" +
        safeScriptJson(payload) + "</script>\n" +
      "  <script id=\"visJsAsset\">\n" + safeInlineScript(visJs) + "\n  </script>\n" +
      "  <script id=\"dashboardJsAsset\">\n" + safeInlineScript(dashboardJs) + "\n  </script>\n" +
      "</body>\n" +
      "</html>\n";
  }

  async function assetTextForExport(selector, fallbackUrl) {
    const el = document.querySelector(selector);
    const attr = el && (el.getAttribute("href") || el.getAttribute("src"));
    if (attr) {
      try {
        return await fetchAssetText(new URL(attr, window.location.href).href);
      } catch (err) {
        console.warn("asset fetch failed", attr, err);
      }
    }
    if (el && el.textContent && el.textContent.trim()) return el.textContent;
    if (fallbackUrl) return fetchAssetText(new URL(fallbackUrl, window.location.href).href);
    throw new Error(`missing asset ${selector}`);
  }

  async function fetchAssetText(url) {
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) throw new Error(`${url}: HTTP ${resp.status}`);
    return resp.text();
  }

  function validateStandaloneAsset(name, text, requiredTokens) {
    const body = String(text || "");
    if (!body.trim()) {
      throw new Error(`snapshot export missing ${name}`);
    }
    if (
      requiredTokens &&
      !requiredTokens.every((token) => body.includes(token))
    ) {
      throw new Error(`snapshot export loaded invalid ${name}`);
    }
  }

  function currentDashboardShellHtml() {
    const header = document.querySelector("header").cloneNode(true);
    const title = header.querySelector("#dashboardTitle");
    if (title) title.textContent = "Agent Concurrency · Snapshot";
    const status = header.querySelector("#connStatus");
    if (status) {
      status.textContent = "loading snapshot…";
      status.className = "status status-pending";
    }
    const liveBtn = header.querySelector("#liveBtn");
    if (liveBtn) liveBtn.hidden = true;
    const pauseBtn = header.querySelector("#pauseBtn");
    if (pauseBtn) {
      pauseBtn.classList.add("paused");
      pauseBtn.textContent = "Resume auto-scroll";
    }
    const eventLinesBtn = header.querySelector("#eventLinesBtn");
    if (eventLinesBtn) {
      eventLinesBtn.classList.toggle("lines-hidden", !state.eventLinesVisible);
      eventLinesBtn.setAttribute("aria-pressed", String(!state.eventLinesVisible));
      eventLinesBtn.textContent = state.eventLinesVisible
        ? "Hide event lines"
        : "Show event lines";
    }
    const eventTypeFilter = header.querySelector("#eventTypeFilter");
    if (eventTypeFilter) {
      // Persist the currently visible event types in the saved HTML.
      eventTypeFilter.removeAttribute("open");
      for (const cb of eventTypeFilter.querySelectorAll('input[type="checkbox"][data-event-type]')) {
        const type = cb.dataset.eventType;
        if (type && state.eventTypeVisible[type] === false) {
          cb.removeAttribute("checked");
        } else {
          cb.setAttribute("checked", "");
        }
      }
    }
    const saveBtn = header.querySelector("#saveSnapshotBtn");
    if (saveBtn) saveBtn.disabled = false;
    return "  " + header.outerHTML + "\n\n" +
      "  <main>\n" +
      "    <div id=\"kvChart\"></div>\n" +
      "    <div id=\"timeline\"></div>\n" +
      "  </main>";
  }

  function snapshotFilename() {
    const rec = state.records[state.records.length - 1];
    const source = rec && rec.ts ? rec.ts : new Date().toISOString();
    const stamp = new Date(parseTimeMs(source) ?? Date.now())
      .toISOString()
      .replace(/[:.]/g, "-");
    return `agent-concurrency-dashboard-${stamp}.html`;
  }

  function safeScriptJson(value) {
    return JSON.stringify(value)
      .replace(/[<>&\u2028\u2029]/g, (ch) => ({
        "<": "\\u003c",
        ">": "\\u003e",
        "&": "\\u0026",
        "\u2028": "\\u2028",
        "\u2029": "\\u2029",
      }[ch]));
  }

  function safeInlineScript(text) {
    return String(text).replace(/<\/script/gi, "<\\/script");
  }

  function safeInlineStyle(text) {
    return String(text).replace(/<\/style/gi, "<\\/style");
  }

  function backToLive() {
    if (isStandaloneFile()) {
      // Saved standalone files have no live server. Hide the button instead
      // of firing fetch("/state") which would just error out.
      $("#liveBtn").hidden = true;
      return;
    }
    state.mode = "live";
    setDashboardTitle("Live");
    setPaused(false);
    resetCharts();
    $("#liveBtn").hidden = true;
    setStatus("connecting…", "status-pending");
    bootstrapLive().then(connectStream);
  }

  // ── controls ───────────────────────────────────────────────────────────

  function wireControls() {
    const eventLinesBtn = $("#eventLinesBtn");
    if (eventLinesBtn) {
      state.eventLinesVisible = eventLinesBtn.getAttribute("aria-pressed") !== "true";
      setEventLinesVisible(state.eventLinesVisible);
      eventLinesBtn.addEventListener("click", () => {
        setEventLinesVisible(!state.eventLinesVisible);
      });
    }
    const filter = $("#eventTypeFilter");
    if (filter) {
      // Initialize state from any pre-set checked attributes (so a saved
      // standalone snapshot can persist the user's last filter selection).
      for (const cb of filter.querySelectorAll('input[type="checkbox"][data-event-type]')) {
        const type = cb.dataset.eventType;
        if (type) state.eventTypeVisible[type] = !!cb.checked;
      }
      filter.addEventListener("change", (ev) => {
        const cb = ev.target;
        if (!cb || cb.type !== "checkbox" || !cb.dataset.eventType) return;
        state.eventTypeVisible[cb.dataset.eventType] = !!cb.checked;
        if (state.eventLinesVisible) {
          scheduleEventLineRender();
        } else {
          clearEventLineLayers();
        }
        hideTooltip();
      });
      // Close the dropdown when clicking outside.
      document.addEventListener("click", (ev) => {
        if (!filter.open) return;
        if (filter.contains(ev.target)) return;
        filter.open = false;
      });
    }
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
    $("#saveSnapshotBtn").addEventListener("click", saveStandaloneHtml);
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

  function fmtPct(v) {
    return v === null || v === undefined ? "n/a" : `${Number(v).toFixed(1)}%`;
  }

  function fmtW(v) {
    return v === null || v === undefined ? "n/a" : Number(v).toFixed(2);
  }

  function fmtCount(v) {
    return v === null || v === undefined ? "n/a" : String(v);
  }

  function pressureLabel(record) {
    const adm = record && record.admission ? record.admission : {};
    if (adm && adm.pressure === true) return "yes";
    const C = finiteNumber(adm && adm.C);
    const thresholdGb = thresholdGbForRecord(record || {});
    if (C !== null && thresholdGb !== null) return C <= thresholdGb ? "yes" : "no";

    const freePct = freeKvPercent(record || {});
    const thresholdPct = thresholdFreePercent(record || {});
    if (freePct === null || thresholdPct === null) return "unknown";
    return freePct <= thresholdPct ? "yes" : "no";
  }

  function formatHMS(d) {
    const pad = (n) => String(n).padStart(2, "0");
    return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  }

  function parseTimeMs(value) {
    if (!value) return null;
    const ms = new Date(value).getTime();
    return Number.isFinite(ms) ? ms : null;
  }

  function formatDurationSeconds(ms) {
    const sec = Math.max(0, Math.round(ms / 1000));
    return String(sec);
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
    try {
      wireControls();
      const embedded = readEmbeddedSnapshot();
      if (!hasVisCharts()) {
        if (embedded) {
          renderSnapshotFallback(
            embedded.ticks,
            embedded.meta,
            "Interactive charts are unavailable because vis-timeline did not load.",
          );
        } else {
          showStartupError(
            "dashboard assets unavailable",
            "This file cannot run as a live dashboard without its JavaScript assets. Use the dashboard's Save standalone HTML button for an offline snapshot.",
          );
        }
        return;
      }
      buildCharts();
      if (embedded) {
        loadSnapshotRecords(embedded.ticks, embedded.meta);
        return;
      }
      setDashboardTitle("Live");
      await bootstrapLive();
      connectStream();
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      showStartupError("dashboard failed to start", msg);
    }
  });
})();
