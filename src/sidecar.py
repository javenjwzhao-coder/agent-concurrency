#!/usr/bin/env python3
"""
sidecar.py — Real-time vLLM KV cache + agent state monitor.

Can run in two modes:

  Embedded (recommended, zero IPC overhead):
      Imported by run_abc_bench_instrumented.py and started as a daemon thread.
      Reads _LIVE_AGENTS directly from memory — no serialization, no disk I/O.

  Standalone (for attaching to a running benchmark after the fact):
      python src/sidecar.py --live-state ./abc_results/.live_state.json ...
      Reads a JSON file written by the runner's --live-state-path option.

Every t seconds, appends one JSONL record to --log-file containing:
  • Global physical KV cache used (GB) from vLLM's Prometheus /metrics endpoint.
  • Per-agent: current phase, KV blocks used, KV cache GB.

Block-to-GB conversion uses the model geometry passed on the CLI, matching
the formula used by vLLM's patched serving_chat.py.

Standalone usage:
    python src/sidecar.py \\
        --vllm-url    http://localhost:8027 \\
        --live-state  ./abc_results/.live_state.json \\
        --log-file    ./sidecar.log \\
        --interval    5 \\
        --num-layers  48 --num-kv-heads 8 --head-dim 128 \\
        --block-size  16 --dtype bfloat16

Requirements:
    pip install requests
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urljoin

import requests

# Optional: prometheus_client gives robust label-aware Prometheus parsing.
# Falls back to a regex approach that extracts the device= label manually.
try:
    from prometheus_client.parser import text_string_to_metric_families as _prom_parse
    _HAS_PROM_CLIENT = True
except ImportError:
    _HAS_PROM_CLIENT = False

# ─────────────────────────────────────── constants ────────────────────────────

_DTYPE_BYTES: dict[str, int] = {
    "float32": 4, "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1,
}

# Device label values that map to the primary (NPU/GPU) accelerator.
_ACCEL_DEVICES = {"gpu", "GPU", "npu", "NPU"}

# Regex fallback: captures name, full label string, and value.
# Prometheus allows NaN / ±Inf as sample values; include them so we never
# silently drop uninitialised gauges.
_PROM_RE = re.compile(
    r'^([\w:]+)(\{[^}]*\})?\s+(NaN|[+-]?Inf|[\d.eE+\-]+)',
    re.MULTILINE,
)
# Extracts device="<val>" from a label string.
_DEVICE_RE = re.compile(r'\bdevice="([^"]*)"')

# ─────────────────────────────────────── helpers ──────────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_vllm_kv_metrics(text: str, total_blocks_hint: int = 0) -> dict:
    """Extract KV-cache block statistics from Prometheus /metrics text.

    Covers every known metric-name variant used by standard vLLM and
    vllm_ascend (Ascend NPU fork), including label-keyed block-manager gauges
    like vllm:block_manager_num_used_blocks{device="npu"}.

    Uses prometheus_client for label-aware parsing when available; otherwise
    falls back to a regex that manually extracts the device= label.

    Returned keys (any may be absent if the metric was not found):
        usage_pct     – fractional cache usage 0..1  (derived when missing)
        total_blocks  – total KV-cache blocks         (derived when missing)
        used_blocks   – actively allocated blocks
        free_blocks   – blocks in the free pool
        cached_blocks – prefix-cached blocks still in VRAM
        vllm_names    – all vllm: metric names seen (for diagnostics)
    """
    m: dict = {}
    seen: list[str] = []

    def _set_usage(val: float) -> None:
        m.setdefault("usage_pct", val)

    def _set_total(val: float) -> None:
        m.setdefault("total_blocks", int(val))

    def _set_used(val: float) -> None:
        m.setdefault("used_blocks", int(val))

    def _set_free(val: float) -> None:
        m.setdefault("free_blocks", int(val))

    def _set_cached(val: float) -> None:
        m.setdefault("cached_blocks", int(val))

    if _HAS_PROM_CLIENT:
        for family in _prom_parse(text):
            for sample in family.samples:
                name = sample.name
                val  = sample.value
                if name.startswith("vllm:") and name not in seen:
                    seen.append(name)
                if not math.isfinite(val):
                    continue
                device = (sample.labels or {}).get("device", "")

                if name in ("vllm:gpu_cache_usage_perc",
                            "vllm:npu_cache_usage_perc",
                            "vllm:kv_cache_usage_perc"):
                    _set_usage(val)
                elif name in ("vllm:num_gpu_blocks", "vllm:num_npu_blocks"):
                    _set_total(val)
                elif name in ("vllm:num_gpu_blocks_used",
                              "vllm:gpu_blocks_used",
                              "vllm:block_manager_gpu_used_blocks"):
                    _set_used(val)
                elif name == "vllm:block_manager_num_used_blocks" and device in _ACCEL_DEVICES:
                    _set_used(val)
                elif name in ("vllm:num_gpu_blocks_free",
                              "vllm:gpu_blocks_free",
                              "vllm:block_manager_gpu_free_blocks"):
                    _set_free(val)
                elif name == "vllm:block_manager_num_free_blocks" and device in _ACCEL_DEVICES:
                    _set_free(val)
                elif name in ("vllm:num_gpu_blocks_cached",
                              "vllm:gpu_blocks_cached",
                              "vllm:block_manager_gpu_cached_blocks",
                              "vllm:prefix_cache_num_blocks"):
                    _set_cached(val)
                elif name == "vllm:block_manager_num_cached_blocks" and device in _ACCEL_DEVICES:
                    _set_cached(val)

    else:
        # Regex fallback — extracts device= label manually.
        for match in _PROM_RE.finditer(text):
            name      = match.group(1)
            label_str = match.group(2) or ""
            val_str   = match.group(3)
            if name.startswith("vllm:") and name not in seen:
                seen.append(name)
            try:
                val = float(val_str)
            except ValueError:
                continue
            if not math.isfinite(val):
                continue
            dm     = _DEVICE_RE.search(label_str)
            device = dm.group(1) if dm else ""

            if name in ("vllm:gpu_cache_usage_perc",
                        "vllm:npu_cache_usage_perc",
                        "vllm:kv_cache_usage_perc"):
                _set_usage(val)
            elif name in ("vllm:num_gpu_blocks", "vllm:num_npu_blocks"):
                _set_total(val)
            elif name in ("vllm:num_gpu_blocks_used", "vllm:gpu_blocks_used",
                          "vllm:block_manager_gpu_used_blocks"):
                _set_used(val)
            elif name == "vllm:block_manager_num_used_blocks" and device in _ACCEL_DEVICES:
                _set_used(val)
            elif name in ("vllm:num_gpu_blocks_free", "vllm:gpu_blocks_free",
                          "vllm:block_manager_gpu_free_blocks"):
                _set_free(val)
            elif name == "vllm:block_manager_num_free_blocks" and device in _ACCEL_DEVICES:
                _set_free(val)
            elif name in ("vllm:num_gpu_blocks_cached", "vllm:gpu_blocks_cached",
                          "vllm:block_manager_gpu_cached_blocks",
                          "vllm:prefix_cache_num_blocks"):
                _set_cached(val)
            elif name == "vllm:block_manager_num_cached_blocks" and device in _ACCEL_DEVICES:
                _set_cached(val)

    # ── Derive missing values from what we have ────────────────────────────────
    # Caller-supplied hint (from --num-gpu-blocks-override / total_gpu_blocks in
    # config) takes priority over any Prometheus-derived value. vllm-ascend may
    # expose a block-count metric that reflects the physical NPU allocation
    # rather than the scheduler-level limit set by the override, so the hint is
    # the ground truth when provided.
    if total_blocks_hint > 0:
        m["total_blocks"] = total_blocks_hint
    elif "total_blocks" not in m:
        # Derive from component gauges when no hint is given.
        parts = [m.get(k, 0) for k in ("used_blocks", "cached_blocks", "free_blocks")]
        if any(v > 0 for v in parts):
            m["total_blocks"] = sum(parts)

    # usage_pct from used / total (when only block gauges are present).
    if "usage_pct" not in m:
        used  = m.get("used_blocks")
        total = m.get("total_blocks")
        if used is not None and total:
            m["usage_pct"] = used / total

    # used_blocks from usage_pct × total (when only the percentage is present).
    if "used_blocks" not in m:
        pct   = m.get("usage_pct")
        total = m.get("total_blocks")
        if pct is not None and total:
            m["used_blocks"] = round(total * pct)

    m["vllm_names"] = seen
    return m


def bytes_per_block(num_layers: int, num_kv_heads: int, head_dim: int,
                    block_size: int, dtype: str) -> int:
    """KV cache bytes consumed by one vLLM block.

    Matches vLLM's patched serving_chat.py formula:
        block_size × num_layers × 2(K+V) × num_kv_heads × head_dim × dtype_bytes
    """
    return block_size * num_layers * 2 * num_kv_heads * head_dim * _DTYPE_BYTES.get(dtype, 2)


def _file_agent_reader(path: Path) -> Callable[[], dict]:
    """Return a callable that reads agents from a live-state JSON file."""
    def _read() -> dict:
        try:
            return json.loads(path.read_text()).get("agents", {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    return _read


def default_eviction_endpoint(vllm_url: str) -> str:
    """Return the default local vLLM agent-KV eviction endpoint."""
    base = vllm_url.rstrip("/") + "/"
    return urljoin(base, "agent_kv_cache/evict")


def default_offload_endpoint(vllm_url: str) -> str:
    """Return the default local vLLM agent-KV CPU-offload endpoint."""
    base = vllm_url.rstrip("/") + "/"
    return urljoin(base, "agent_kv_cache/offload")


def default_restore_endpoint(vllm_url: str) -> str:
    """Return the default local vLLM agent-KV restore/readmit endpoint."""
    base = vllm_url.rstrip("/") + "/"
    return urljoin(base, "agent_kv_cache/restore")


def _parse_iso_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _agent_kv_gb(agent: dict[str, Any], bytes_per_blk: int) -> Optional[float]:
    """Return per-agent KV usage in GB, preferring block counts when present."""
    blocks = agent.get("kv_blocks")
    if isinstance(blocks, int) and blocks >= 0:
        return blocks * bytes_per_blk / 1e9
    gb = _finite_float(agent.get("kv_gb"))
    if gb is not None and gb >= 0:
        return gb
    return None


@dataclass
class AgentLaunchSpec:
    """A launch request owned by the admission controller."""

    agent_id: str
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    previously_evicted: bool = False
    queued_ts: str = field(default_factory=lambda: iso_utc(now_utc()))


@dataclass
class _IdleAgentCandidate:
    agent_id: str
    kv_gb: float
    predicted_remaining_s: float
    score: float


class DynamicAdmissionController:
    """Admission/eviction policy for multi-agent vLLM KV-cache pressure."""

    ACTIVE_STATES = {"reasoning", "tool_call", "waiting"}

    def __init__(
        self,
        *,
        enabled: bool = False,
        threshold_gb: float = 0.1,
        initial_admit_interval_s: float = 2.0,
        short_tool_call_threshold_s: float = 2.0,
        offload_endpoint: Optional[str] = None,
        restore_endpoint: Optional[str] = None,
        eviction_endpoint: Optional[str] = None,
        eviction_timeout_s: float = 2.0,
        bytes_per_blk: Optional[int] = None,
        predictor_model: Optional[Path | str] = None,
        predictor: Any = None,
        admit_callback: Optional[Callable[[AgentLaunchSpec], Any]] = None,
        state_update_callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
        evict_callback: Optional[Callable[[_IdleAgentCandidate], dict[str, Any]]] = None,
        offload_callback: Optional[Callable[[_IdleAgentCandidate], dict[str, Any]]] = None,
        restore_callback: Optional[Callable[[str], dict[str, Any]]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.threshold_gb = max(0.0, float(threshold_gb))
        self.initial_admit_interval_s = max(0.0, float(initial_admit_interval_s))
        self.short_tool_call_threshold_s = max(0.0, float(short_tool_call_threshold_s))
        self.offload_endpoint = str(offload_endpoint or eviction_endpoint or "")
        self.restore_endpoint = str(restore_endpoint) if restore_endpoint else ""
        self.eviction_endpoint = str(eviction_endpoint) if eviction_endpoint else ""
        self.eviction_timeout_s = max(0.1, float(eviction_timeout_s))
        self.bytes_per_blk = int(bytes_per_blk) if bytes_per_blk else None
        self._admit_callback = admit_callback
        self._state_update_callback = state_update_callback
        self._offload_callback = offload_callback or evict_callback
        self._restore_callback = restore_callback
        self._session = session or requests.Session()

        self._fresh: deque[AgentLaunchSpec] = deque()
        self._evicted_ready: deque[str] = deque()
        self._evicted_ready_set: set[str] = set()
        self._evicted_events: dict[str, threading.Event] = {}
        self._evicted_pending: set[str] = set()
        self._idle_short: set[str] = set()
        self._idle_long: set[str] = set()
        self._kv_policy_events: deque[dict[str, Any]] = deque(maxlen=1024)
        self._prev_avg_gb: Optional[float] = None
        self._first_saturation_seen = False
        self._last_fresh_admit_monotonic: Optional[float] = None
        self._lock = threading.RLock()

        self._predictor = predictor
        self._build_features: Any = None
        self._predictor_error: Optional[str] = None
        if predictor is None and predictor_model:
            self._load_predictor(Path(predictor_model))

    def set_admit_callback(
        self, callback: Optional[Callable[[AgentLaunchSpec], Any]]
    ) -> None:
        with self._lock:
            self._admit_callback = callback

    def enqueue_fresh(self, spec: AgentLaunchSpec) -> None:
        with self._lock:
            spec.previously_evicted = False
            self._fresh.append(spec)

    def pending_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "fresh": len(self._fresh),
                "evicted_ready": len(self._evicted_ready),
                "evicted_pending_tool": len(self._evicted_pending),
                "idle_short": len(self._idle_short),
                "idle_long": len(self._idle_long),
            }

    def on_tool_call_start(
        self, agent_id: str, agent: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Apply the immediate predictor-gated KV policy for a new tool call."""
        agent = dict(agent or {})
        agent.setdefault("agent_id", agent_id)
        with self._lock:
            event: dict[str, Any] = {
                "ts": iso_utc(now_utc()),
                "agent_id": agent_id,
                "policy": "skipped",
                "predicted_remaining_s": None,
                "threshold_s": self.short_tool_call_threshold_s,
                "reason": "",
            }
            if not self.enabled:
                event["reason"] = "disabled"
                self._record_kv_policy_event_locked(event)
                return event

            predicted = self._predict_remaining(agent)
            if predicted is None:
                event["reason"] = self._predictor_error or "predictor_unavailable"
                self._update_agent_state_locked(
                    agent_id,
                    {
                        "kv_policy": "predictor_unavailable",
                        "kv_policy_reason": event["reason"],
                        "last_kv_policy": event,
                    },
                )
                self._record_kv_policy_event_locked(event)
                return event

            event["predicted_remaining_s"] = self._round(predicted)
            if predicted < self.short_tool_call_threshold_s:
                self._idle_long.discard(agent_id)
                self._idle_short.add(agent_id)
                event.update({
                    "policy": "idle_short",
                    "reason": "predicted_short",
                })
                self._update_agent_state_locked(
                    agent_id,
                    {
                        "kv_policy": "idle_short",
                        "kv_evicted": False,
                        "admission_state": "admitted",
                        "predicted_remaining_s": self._round(predicted),
                        "last_kv_policy": event,
                    },
                )
                self._record_kv_policy_event_locked(event)
                return event

            self._idle_short.discard(agent_id)
            self._idle_long.add(agent_id)
            event.update({
                "policy": "idle_long",
                "reason": "eligible_for_pressure_offload",
            })
            self._update_agent_state_locked(
                agent_id,
                {
                    "kv_policy": event["policy"],
                    "kv_evicted": False,
                    "admission_state": "admitted",
                    "predicted_remaining_s": self._round(predicted),
                    "last_kv_policy": event,
                },
            )
            self._record_kv_policy_event_locked(event)
            return event

    def wait_if_evicted(self, agent_id: str) -> bool:
        """Block an agent after a tool call until the sidecar re-admits it."""
        with self._lock:
            event = self._evicted_events.get(agent_id)
            if event is None:
                return False
            if event.is_set():
                self._clear_evicted_locked(agent_id)
                return False
            if agent_id not in self._evicted_ready_set:
                self._evicted_ready.append(agent_id)
                self._evicted_ready_set.add(agent_id)
            self._evicted_pending.discard(agent_id)
            self._update_agent_state_locked(
                agent_id,
                {
                    "state": "evicted_waiting",
                    "state_since": iso_utc(now_utc()),
                    "admission_state": "evicted_ready",
                },
            )

        event.wait()
        with self._lock:
            self._clear_evicted_locked(agent_id)
            self._update_agent_state_locked(
                agent_id,
                {
                    "kv_evicted": False,
                    "kv_policy": "readmitted",
                    "admission_state": "admitted",
                    "admitted_since": iso_utc(now_utc()),
                },
            )
        return True

    def finish_agent(self, agent_id: str) -> None:
        """Clean up any queued/blocked admission state for a finished agent."""
        with self._lock:
            event = self._evicted_events.get(agent_id)
            if event is not None:
                event.set()
            self._clear_evicted_locked(agent_id)

    def on_tick(
        self,
        *,
        tick: int,
        vllm_info: dict[str, Any],
        agents: dict[str, dict[str, Any]],
        bytes_per_blk: int,
    ) -> dict[str, Any]:
        """Apply one admission-control decision tick and return log metadata."""
        with self._lock:
            free_gb = self._kv_free_gb(vllm_info, bytes_per_blk)
            avg_gb, avg_count = self._average_active_kv_gb(agents, bytes_per_blk)
            prev_avg_gb = self._prev_avg_gb
            w = self._headroom(free_gb, avg_gb, prev_avg_gb)
            report: dict[str, Any] = {
                "enabled": self.enabled,
                "tick": tick,
                "C": self._round(free_gb),
                "s_t": self._round(avg_gb),
                "s_prev": self._round(prev_avg_gb),
                "active_agent_samples": avg_count,
                "w": self._round(w),
                "threshold_gb": self.threshold_gb,
                "first_saturation_seen": self._first_saturation_seen,
                "initial_admit_interval_s": self.initial_admit_interval_s,
                "next_initial_admit_in_s": None,
                "queue": self.pending_counts(),
                "heap_candidates": [],
                "skipped_candidates": [],
                "evictions": [],
                "admissions": [],
                "kv_policy_events": self._drain_kv_policy_events_locked(),
                "reasons": [],
            }
            runnable_queue = (
                report["queue"].get("fresh", 0)
                + report["queue"].get("evicted_ready", 0)
            )

            if self._predictor_error:
                report["reasons"].append(f"predictor_error: {self._predictor_error}")
            if not self.enabled:
                report["reasons"].append("disabled")
                self._prev_avg_gb = avg_gb
                return report
            if free_gb is None:
                report["reasons"].append("missing_kv_free_gb")
                self._prev_avg_gb = avg_gb
                return report

            heap, skipped = self._idle_agent_heap(agents, bytes_per_blk)
            report["skipped_candidates"] = skipped
            report["heap_candidates"] = [
                {
                    "agent_id": cand.agent_id,
                    "kv_gb": self._round(cand.kv_gb),
                    "predicted_remaining_s": self._round(cand.predicted_remaining_s),
                    "e_s": self._round(cand.score),
                }
                for _, _, cand in heap
            ]

            if w is not None and w < 1.0:
                report["reasons"].append("headroom_low")
                if runnable_queue > 0:
                    report["reasons"].append("saturation_guard")
                    self._first_saturation_seen = True
                    report["first_saturation_seen"] = True

            current_free_gb = free_gb
            while current_free_gb <= self.threshold_gb and heap:
                _, _, cand = heapq.heappop(heap)
                evict_result = self._offload_candidate(cand)
                report["evictions"].append(evict_result)
                if evict_result.get("evicted"):
                    freed_gb = _finite_float(evict_result.get("freed_gb"))
                    if freed_gb is None:
                        freed_gb = cand.kv_gb
                    current_free_gb += max(0.0, freed_gb)
                    self._mark_agent_evicted_locked(cand.agent_id, evict_result)

            if current_free_gb <= self.threshold_gb:
                report["reasons"].append("pressure_threshold")

            admit_limit = self._admission_limit(current_free_gb, w)
            if admit_limit <= 0 and w is not None and w < 1.0 and runnable_queue > 0:
                report["reasons"].append("admission_blocked_by_headroom")
            elif admit_limit <= 0 and current_free_gb <= self.threshold_gb and runnable_queue > 0:
                report["reasons"].append("admission_blocked_by_pressure")

            if admit_limit > 0:
                self._admit_with_limit_locked(admit_limit, report)

            report["queue"] = self.pending_counts()
            report["first_saturation_seen"] = self._first_saturation_seen
            self._prev_avg_gb = avg_gb
            return report

    def _load_predictor(self, path: Path) -> None:
        try:
            from build_tool_predictor import RealtimePredictor, build_features

            self._predictor = RealtimePredictor.load(path)
            self._build_features = build_features
        except Exception as exc:
            self._predictor_error = f"{path}: {exc}"
            self._predictor = None
            self._build_features = None

    def _kv_free_gb(
        self, vllm_info: dict[str, Any], bytes_per_blk: int
    ) -> Optional[float]:
        free_gb = _finite_float(vllm_info.get("kv_free_gb"))
        if free_gb is not None:
            return max(0.0, free_gb)

        free_blocks = _finite_float(vllm_info.get("num_gpu_blocks_free"))
        if free_blocks is not None:
            return max(0.0, free_blocks * bytes_per_blk / 1e9)

        total = _finite_float(vllm_info.get("kv_total_gb"))
        used = _finite_float(vllm_info.get("kv_used_gb"))
        if total is not None and used is not None:
            return max(0.0, total - used)
        return None

    def _average_active_kv_gb(
        self, agents: dict[str, dict[str, Any]], bytes_per_blk: int
    ) -> tuple[Optional[float], int]:
        samples: list[float] = []
        for agent in agents.values():
            if agent.get("state") not in self.ACTIVE_STATES:
                continue
            if agent.get("kv_evicted"):
                continue
            kv_gb = _agent_kv_gb(agent, bytes_per_blk)
            if kv_gb is not None and kv_gb > 0:
                samples.append(kv_gb)
        if not samples:
            return None, 0
        return sum(samples) / len(samples), len(samples)

    def _headroom(
        self,
        free_gb: Optional[float],
        avg_gb: Optional[float],
        prev_avg_gb: Optional[float],
    ) -> Optional[float]:
        if free_gb is None:
            return None
        recent = [v for v in (avg_gb, prev_avg_gb) if v is not None and v > 0]
        if len(recent) < 2:
            return None
        denom = min(recent)
        return free_gb / denom if denom > 0 else None

    def _idle_agent_heap(
        self, agents: dict[str, dict[str, Any]], bytes_per_blk: int
    ) -> tuple[list[tuple[float, str, _IdleAgentCandidate]], list[dict[str, Any]]]:
        heap: list[tuple[float, str, _IdleAgentCandidate]] = []
        skipped: list[dict[str, Any]] = []
        for agent_id, agent in agents.items():
            if agent.get("state") != "tool_call":
                continue
            if agent_id in self._evicted_events or agent.get("kv_evicted"):
                continue
            if (
                agent_id in self._idle_short
                or agent.get("kv_policy") == "idle_short"
            ):
                skipped.append({"agent_id": agent_id, "reason": "idle_short"})
                continue
            kv_gb = _agent_kv_gb(agent, bytes_per_blk)
            if kv_gb is None or kv_gb <= 0:
                continue
            remaining_s = self._predict_remaining(agent)
            if remaining_s is None:
                skipped.append({
                    "agent_id": agent_id,
                    "reason": self._predictor_error or "predictor_unavailable",
                })
                continue
            if remaining_s < self.short_tool_call_threshold_s:
                skipped_event = {
                    "agent_id": agent_id,
                    "reason": "predicted_short",
                    "predicted_remaining_s": self._round(remaining_s),
                }
                if agent_id not in self._idle_short:
                    self._idle_long.discard(agent_id)
                    self._idle_short.add(agent_id)
                    event = {
                        "ts": iso_utc(now_utc()),
                        "agent_id": agent_id,
                        "policy": "idle_short",
                        "predicted_remaining_s": self._round(remaining_s),
                        "threshold_s": self.short_tool_call_threshold_s,
                        "reason": "predicted_short",
                    }
                    self._update_agent_state_locked(
                        agent_id,
                        {
                            "kv_policy": "idle_short",
                            "kv_evicted": False,
                            "admission_state": "admitted",
                            "predicted_remaining_s": self._round(remaining_s),
                            "last_kv_policy": event,
                        },
                    )
                    self._record_kv_policy_event_locked(event)
                skipped.append(skipped_event)
                continue
            if agent_id not in self._idle_long:
                self._idle_long.add(agent_id)
                event = {
                    "ts": iso_utc(now_utc()),
                    "agent_id": agent_id,
                    "policy": "idle_long",
                    "predicted_remaining_s": self._round(remaining_s),
                    "threshold_s": self.short_tool_call_threshold_s,
                    "reason": "eligible_for_pressure_offload",
                }
                self._update_agent_state_locked(
                    agent_id,
                    {
                        "kv_policy": "idle_long",
                        "kv_evicted": False,
                        "admission_state": "admitted",
                        "predicted_remaining_s": self._round(remaining_s),
                        "last_kv_policy": event,
                    },
                )
                self._record_kv_policy_event_locked(event)
            score = kv_gb * remaining_s
            cand = _IdleAgentCandidate(
                agent_id=agent_id,
                kv_gb=kv_gb,
                predicted_remaining_s=remaining_s,
                score=score,
            )
            heapq.heappush(heap, (-score, agent_id, cand))
        return heap, skipped

    def _predict_remaining(self, agent: dict[str, Any]) -> Optional[float]:
        start_dt = _parse_iso_ts(agent.get("tool_started_at")) or _parse_iso_ts(
            agent.get("state_since")
        )
        elapsed_s = 0.0
        if start_dt is not None:
            elapsed_s = max(0.0, (now_utc() - start_dt).total_seconds())

        if hasattr(self._predictor, "predict_agent_remaining"):
            try:
                return max(0.0, float(self._predictor.predict_agent_remaining(agent, elapsed_s)))
            except Exception as exc:
                self._predictor_error = str(exc)
                return None

        if self._predictor is None or self._build_features is None:
            if self._predictor_error is None:
                self._predictor_error = "predictor_unavailable"
            return None

        try:
            import pandas as pd

            row = {
                "tool_name": agent.get("tool_name") or "unknown",
                "tool_command": agent.get("tool_command") or "",
                "tool_args_json": agent.get("tool_args_json") or "{}",
                "tool_payload_bytes": agent.get("tool_payload_bytes") or 0,
                "phase_seq": agent.get("phase_seq") or 0,
                "start_active_agents": agent.get("start_active_agents") or 0,
                "start_active_tool_calls": agent.get("start_active_tool_calls") or 0,
                "start_cumulative_reasoning_s": (
                    agent.get("start_cumulative_reasoning_s")
                    or agent.get("cumulative_reasoning_s")
                    or 0.0
                ),
            }
            feat = self._build_features(pd.DataFrame([row]), remaining_mode=True)
            return float(self._predictor.predict_remaining(feat, elapsed_s=elapsed_s))
        except Exception as exc:
            self._predictor_error = str(exc)
            return None

    def _offload_candidate(self, cand: _IdleAgentCandidate) -> dict[str, Any]:
        result: dict[str, Any] = {
            "agent_id": cand.agent_id,
            "e_s": self._round(cand.score),
            "kv_gb": self._round(cand.kv_gb),
            "predicted_remaining_s": self._round(cand.predicted_remaining_s),
        }
        try:
            if self._offload_callback is not None:
                payload = self._offload_callback(cand) or {}
            else:
                endpoint = self.offload_endpoint or self.eviction_endpoint
                if not endpoint:
                    return {
                        **result,
                        "evicted": False,
                        "offloaded": False,
                        "reason": "missing_offload_endpoint",
                    }
                resp = self._session.post(
                    endpoint,
                    json={
                        "agent_id": cand.agent_id,
                        "only_ref_cnt_zero": True,
                        "predicted_remaining_s": cand.predicted_remaining_s,
                    },
                    timeout=self.eviction_timeout_s,
                )
                if resp.status_code >= 400:
                    return {
                        **result,
                        "evicted": False,
                        "offloaded": False,
                        "status_code": resp.status_code,
                        "reason": resp.text[:300],
                    }
                payload = resp.json()
            if "evicted" in payload:
                evicted = bool(payload.get("evicted"))
            elif "offloaded" in payload:
                evicted = bool(payload.get("offloaded"))
            else:
                evicted = True
            if evicted:
                self._idle_long.discard(cand.agent_id)
            return {**result, **payload, "evicted": evicted}
        except Exception as exc:
            return {**result, "evicted": False, "offloaded": False, "reason": str(exc)}

    def _restore_agent_locked(self, agent_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {"agent_id": agent_id}
        try:
            if self._restore_callback is not None:
                payload = self._restore_callback(agent_id) or {}
            else:
                if not self.restore_endpoint:
                    return {**result, "restored": False, "reason": "missing_restore_endpoint"}
                resp = self._session.post(
                    self.restore_endpoint,
                    json={"agent_id": agent_id},
                    timeout=self.eviction_timeout_s,
                )
                if resp.status_code >= 400:
                    return {
                        **result,
                        "restored": False,
                        "status_code": resp.status_code,
                        "reason": resp.text[:300],
                    }
                payload = resp.json()
            restored = bool(payload.get("restored", True))
            return {**result, **payload, "restored": restored}
        except Exception as exc:
            return {**result, "restored": False, "reason": str(exc)}

    def _record_kv_policy_event_locked(self, event: dict[str, Any]) -> None:
        self._kv_policy_events.append(dict(event))

    def _drain_kv_policy_events_locked(self) -> list[dict[str, Any]]:
        events = list(self._kv_policy_events)
        self._kv_policy_events.clear()
        return events

    def _mark_agent_evicted_locked(
        self, agent_id: str, evict_result: dict[str, Any]
    ) -> None:
        event = self._evicted_events.get(agent_id)
        if event is None or event.is_set():
            event = threading.Event()
            self._evicted_events[agent_id] = event
        self._evicted_pending.add(agent_id)
        self._idle_long.discard(agent_id)
        self._update_agent_state_locked(
            agent_id,
            {
                "kv_evicted": True,
                "admission_state": "evicted_pending_tool",
                "last_eviction": iso_utc(now_utc()),
                "last_eviction_result": evict_result,
            },
        )

    def _admission_limit(self, free_gb: float, w: Optional[float]) -> int:
        if free_gb <= self.threshold_gb:
            return 0
        if w is None:
            return 1
        if w < 1.0:
            return 0
        return max(0, int(math.floor(w)))

    def _admit_with_limit_locked(self, admit_limit: int, report: dict[str, Any]) -> None:
        admitted = 0

        while admitted < admit_limit:
            agent_id = self._pop_next_evicted_ready_locked()
            if agent_id is None:
                break
            report["admissions"].append(self._admit_locked(agent_id))
            admitted += 1

        remaining = admit_limit - admitted
        if remaining <= 0 or not self._fresh:
            return

        if self._admit_callback is None:
            report["reasons"].append("no_admit_callback")
            return

        fresh_limit = remaining
        if not self._first_saturation_seen and self.initial_admit_interval_s > 0:
            now_mono = time.monotonic()
            next_in = self._next_initial_admit_in_s_locked(now_mono)
            report["next_initial_admit_in_s"] = self._round(next_in)
            if next_in > 0:
                report["reasons"].append("initial_admit_ramp_wait")
                return
            report["reasons"].append("initial_admit_ramp_active")
            fresh_limit = min(fresh_limit, 1)

        for _ in range(fresh_limit):
            item = self._pop_next_fresh_locked()
            if item is None:
                break
            report["admissions"].append(self._admit_locked(item))
            self._last_fresh_admit_monotonic = time.monotonic()

        if not self._first_saturation_seen and self._fresh and self.initial_admit_interval_s > 0:
            report["next_initial_admit_in_s"] = self._round(self.initial_admit_interval_s)

    def _next_initial_admit_in_s_locked(self, now_mono: float) -> float:
        if self._last_fresh_admit_monotonic is None:
            return 0.0
        elapsed = max(0.0, now_mono - self._last_fresh_admit_monotonic)
        return max(0.0, self.initial_admit_interval_s - elapsed)

    def _pop_next_evicted_ready_locked(self) -> Optional[str]:
        while self._evicted_ready:
            agent_id = self._evicted_ready.popleft()
            self._evicted_ready_set.discard(agent_id)
            if agent_id in self._evicted_events:
                return agent_id
        return None

    def _pop_next_fresh_locked(self) -> Optional[AgentLaunchSpec]:
        if self._fresh:
            return self._fresh.popleft()
        return None

    def _admit_locked(self, item: AgentLaunchSpec | str) -> dict[str, Any]:
        if isinstance(item, str):
            agent_id = item
            restore_result = self._restore_agent_locked(agent_id)
            event = self._evicted_events.get(agent_id)
            if event is not None:
                event.set()
            self._update_agent_state_locked(
                agent_id,
                {
                    "admission_state": "admitted",
                    "admitted_since": iso_utc(now_utc()),
                    "last_restore_result": restore_result,
                },
            )
            return {
                "agent_id": agent_id,
                "previously_evicted": True,
                "admitted": True,
                "restore_result": restore_result,
            }

        assert self._admit_callback is not None
        self._admit_callback(item)
        return {
            "agent_id": item.agent_id,
            "previously_evicted": item.previously_evicted,
            "admitted": True,
        }

    def _clear_evicted_locked(self, agent_id: str) -> None:
        self._evicted_events.pop(agent_id, None)
        self._evicted_pending.discard(agent_id)
        self._evicted_ready_set.discard(agent_id)
        self._idle_short.discard(agent_id)
        self._idle_long.discard(agent_id)
        if self._evicted_ready:
            self._evicted_ready = deque(
                aid for aid in self._evicted_ready if aid != agent_id
            )

    def _update_agent_state_locked(
        self, agent_id: str, patch: dict[str, Any]
    ) -> None:
        if self._state_update_callback is not None:
            self._state_update_callback(agent_id, patch)

    @staticmethod
    def _round(value: Optional[float], digits: int = 6) -> Optional[float]:
        if value is None:
            return None
        return round(float(value), digits)


# ─────────────────────────────────────── core loop ────────────────────────────

# Tracks whether we've already emitted the "no KV metrics found" diagnostic.
_kv_metric_warn_emitted = False


def poll_vllm(session: requests.Session, vllm_url: str, bytes_per_blk: int,
              total_blocks_hint: int = 0) -> dict:
    """Fetch /metrics and return global KV cache statistics."""
    global _kv_metric_warn_emitted
    try:
        resp = session.get(f"{vllm_url}/metrics", timeout=5)
        resp.raise_for_status()
        kv = _parse_vllm_kv_metrics(resp.text, total_blocks_hint=total_blocks_hint)

        used_pct   = kv.get("usage_pct")    # 0..1
        total_blks = kv.get("total_blocks")
        used_blks  = kv.get("used_blocks")
        free_blks  = kv.get("free_blocks")
        if free_blks is None and total_blks is not None and used_blks is not None:
            free_blks = max(0, int(total_blks) - int(used_blks))

        if used_pct is None and total_blks is None and not _kv_metric_warn_emitted:
            _kv_metric_warn_emitted = True
            print(
                "[sidecar] WARNING: KV cache metrics not found in /metrics response.\n"
                f"          prometheus_client available: {_HAS_PROM_CLIENT}\n"
                f"          vllm: metrics present: {sorted(kv.get('vllm_names', []))[:30]}",
                flush=True,
            )

        total_gb: Optional[float] = (
            round(total_blks * bytes_per_blk / 1e9, 3) if total_blks is not None else None
        )
        used_gb: Optional[float] = (
            round(used_blks * bytes_per_blk / 1e9, 3) if used_blks is not None else None
        )
        free_gb: Optional[float] = (
            round(free_blks * bytes_per_blk / 1e9, 3) if free_blks is not None else None
        )
        return {
            "kv_cache_used_pct":    round(used_pct * 100, 2) if used_pct is not None else None,
            "num_gpu_blocks_total": int(total_blks) if total_blks is not None else None,
            "num_gpu_blocks_used":  used_blks,
            "num_gpu_blocks_free":  int(free_blks) if free_blks is not None else None,
            "kv_total_gb":          total_gb,
            "kv_used_gb":           used_gb,
            "kv_free_gb":           free_gb,
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_loop(args: argparse.Namespace, bytes_per_blk: int,
             get_agents: Callable[[], dict],
             stop_event: Optional[threading.Event] = None,
             admission_controller: Optional[DynamicAdmissionController] = None,
             http_feed: Optional[Any] = None) -> None:
    """Main poll loop.

    Parameters
    ----------
    args          Parsed CLI namespace (vllm_url, log_file, interval, …).
    bytes_per_blk Pre-computed bytes per KV block (from bytes_per_block()).
    get_agents    Zero-argument callable that returns the current per-agent
                  state dict.  In embedded mode this reads _LIVE_AGENTS
                  directly; in standalone mode it reads the live-state file.
    stop_event    Optional threading.Event; when set, the loop exits cleanly
                  after finishing its current tick.
    http_feed     Optional sidecar_http.HTTPFeed; when set, every record is
                  also published to the live dashboard feed.
    """
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    tick = 0

    print(f"[sidecar] polling every {args.interval}s → {log_path}", flush=True)
    print(f"[sidecar] vLLM: {args.vllm_url}", flush=True)
    print(f"[sidecar] block geometry: {bytes_per_blk} bytes/block", flush=True)

    while stop_event is None or not stop_event.is_set():
        tick_start = time.monotonic()

        agents = get_agents()           # zero-copy in embedded mode
        record = {
            "ts":     iso_utc(now_utc()),
            "tick":   tick,
            "vllm":   poll_vllm(session, args.vllm_url, bytes_per_blk,
                                 getattr(args, "total_gpu_blocks", 0)),
            "agents": agents,
        }
        if admission_controller is not None:
            record["admission"] = admission_controller.on_tick(
                tick=tick,
                vllm_info=record["vllm"],
                agents=agents,
                bytes_per_blk=bytes_per_blk,
            )
        else:
            record["admission"] = {"enabled": False, "reasons": ["not_configured"]}

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        if http_feed is not None:
            try:
                http_feed.publish(record)
            except Exception as exc:
                print(f"[sidecar] http_feed.publish failed: {exc}", flush=True)

        # Brief stdout summary.
        vllm_info = record["vllm"]
        used_pct  = vllm_info.get("kv_cache_used_pct")
        used_gb   = vllm_info.get("kv_used_gb")
        states    = [a.get("state", "?") for a in agents.values()]
        state_str = (", ".join(f"{s}: {states.count(s)}" for s in dict.fromkeys(states))
                     if states else "none")
        gb_str    = f"{used_gb:.2f} GB" if used_gb is not None else "n/a"
        pct_str   = f"{used_pct:.1f}%" if used_pct is not None else "n/a"
        print(f"[sidecar] tick={tick:4d}  kv={gb_str} ({pct_str})  "
              f"agents={len(agents)} [{state_str}]", flush=True)

        tick += 1
        elapsed = time.monotonic() - tick_start
        remaining = max(0.0, args.interval - elapsed)
        if stop_event is not None:
            stop_event.wait(timeout=remaining)
        else:
            time.sleep(remaining)


# ─────────────────────────────────────── CLI (standalone mode) ────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone sidecar: poll vLLM /metrics and a live-state "
                    "JSON file, write JSONL log.  For zero-overhead embedded "
                    "mode, use --sidecar-* flags in run_abc_bench_instrumented.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--live-state", required=True,
                   help="Live-state JSON file written by the instrumented runner "
                        "(--live-state-path / default: <results-root>/.live_state.json).")
    p.add_argument("--log-file", default="sidecar.log")
    p.add_argument("--interval", "-t", type=float, default=5.0)

    geo = p.add_argument_group("model geometry")
    geo.add_argument("--num-layers",   type=int, required=True)
    geo.add_argument("--num-kv-heads", type=int, required=True)
    geo.add_argument("--head-dim",     type=int, required=True)
    geo.add_argument("--block-size",   type=int, default=16)
    geo.add_argument("--dtype",        default="bfloat16", choices=list(_DTYPE_BYTES))
    geo.add_argument("--total-gpu-blocks", type=int, default=0,
                     help="Total GPU/NPU KV-cache blocks. Required when vLLM does not expose "
                          "a block-count metric in Prometheus (e.g. vllm_ascend with only a "
                          "usage-%% gauge). Must match --num-gpu-blocks-override in vLLM args.")

    ac = p.add_argument_group("dynamic admission control")
    ac.add_argument("--admission-control", action="store_true",
                    help="Enable admission/eviction policy decisions in the sidecar.")
    ac.add_argument("--admission-threshold-gb", type=float, default=0.1,
                    help="Free KV-cache GB threshold that triggers pressure eviction.")
    ac.add_argument("--initial-admit-interval-s", type=float, default=2.0,
                    help="Before first SAT, admit at most one fresh task per interval.")
    ac.add_argument("--short-tool-call-threshold-s", type=float, default=2.0,
                    help="Predicted tool calls below this duration stay resident.")
    ac.add_argument("--admission-predictor-model", default=None,
                    help="Saved remaining-time predictor model for idle-agent scoring.")
    ac.add_argument("--offload-endpoint", default=None,
                    help="vLLM admin endpoint for per-agent KV CPU offload. Defaults to "
                         "<vllm-url>/agent_kv_cache/offload.")
    ac.add_argument("--restore-endpoint", default=None,
                    help="vLLM admin endpoint to notify readmission. Defaults to "
                         "<vllm-url>/agent_kv_cache/restore.")
    ac.add_argument("--eviction-endpoint", default=None,
                    help="Backward-compatible alias for --offload-endpoint.")
    ac.add_argument("--eviction-timeout-s", type=float, default=2.0,
                    help="HTTP timeout for one eviction request.")

    dash = p.add_argument_group("live dashboard (optional)")
    dash.add_argument("--http-port", type=int, default=0,
                      help="Bind a live HTTP/SSE feed for the dashboard on this "
                           "port. 0 disables.")
    dash.add_argument("--http-host", default="127.0.0.1",
                      help="Bind address for the live dashboard feed.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    bpb = bytes_per_block(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        dtype=args.dtype,
    )
    get_agents = _file_agent_reader(Path(args.live_state))
    admission_controller = None
    if args.admission_control:
        admission_controller = DynamicAdmissionController(
            enabled=True,
            threshold_gb=args.admission_threshold_gb,
            initial_admit_interval_s=args.initial_admit_interval_s,
            short_tool_call_threshold_s=args.short_tool_call_threshold_s,
            offload_endpoint=(
                args.offload_endpoint
                or args.eviction_endpoint
                or default_offload_endpoint(args.vllm_url)
            ),
            restore_endpoint=args.restore_endpoint or default_restore_endpoint(args.vllm_url),
            eviction_endpoint=args.eviction_endpoint,
            eviction_timeout_s=args.eviction_timeout_s,
            bytes_per_blk=bpb,
            predictor_model=args.admission_predictor_model,
        )

    http_feed = None
    if getattr(args, "http_port", 0):
        from sidecar_http import HTTPFeed, start_server  # local import: optional dep
        http_feed = HTTPFeed()
        start_server(args.http_host, args.http_port, http_feed)
        print(f"[sidecar] dashboard feed → http://{args.http_host}:{args.http_port}/",
              flush=True)

    def _stop(sig, frame):
        print("\n[sidecar] shutting down.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    run_loop(args, bpb, get_agents, admission_controller=admission_controller,
             http_feed=http_feed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
