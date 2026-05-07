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
from concurrent.futures import ThreadPoolExecutor, as_completed
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

DEFAULT_OFFLOAD_TIMEOUT_S = 10.0
DEFAULT_EXACT_FREED_GB_TIMEOUT_S = 5.0
try:
    _REQUEST_TIMEOUT_EXCEPTIONS = (requests.exceptions.Timeout,)
except AttributeError:  # test fallback when requests is stubbed out
    _REQUEST_TIMEOUT_EXCEPTIONS = ()

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
# Extracts num_(gpu|npu)_blocks="<int>" from a cache_config_info label string.
_CACHE_BLOCKS_RE = re.compile(r'\bnum_(?:gpu|npu)_blocks="(\d+)"')

# ─────────────────────────────────────── helpers ──────────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_vllm_kv_metrics(text: str) -> dict:
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
        scheduler_preemptions_total – cumulative vLLM scheduler preemptions
        vllm_names    – all vllm: metric names seen (for diagnostics)
    """
    m: dict = {}
    seen: list[str] = []
    preempt_samples: dict[str, float] = {}

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

    def _is_preempt_metric(name: str) -> bool:
        lower = name.lower()
        if not lower.startswith("vllm:") or "preempt" not in lower:
            return False
        return not lower.endswith(("_bucket", "_sum", "_created"))

    def _add_preempt_sample(name: str, val: float) -> None:
        preempt_samples[name] = preempt_samples.get(name, 0.0) + val

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
                elif _is_preempt_metric(name):
                    _add_preempt_sample(name, val)
                elif name in ("vllm:num_gpu_blocks", "vllm:num_npu_blocks"):
                    _set_total(val)
                elif name == "vllm:cache_config_info":
                    labels = sample.labels or {}
                    blocks = labels.get("num_gpu_blocks") or labels.get("num_npu_blocks")
                    try:
                        if blocks is not None and blocks != "None":
                            _set_total(float(blocks))
                    except ValueError:
                        pass
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
            elif _is_preempt_metric(name):
                _add_preempt_sample(name, val)
            elif name in ("vllm:num_gpu_blocks", "vllm:num_npu_blocks"):
                _set_total(val)
            elif name == "vllm:cache_config_info":
                bm = _CACHE_BLOCKS_RE.search(label_str)
                if bm:
                    _set_total(float(bm.group(1)))
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
    # When vLLM does not export per-pool block counts (vLLM v1 / vllm-ascend
    # 0.13 only emit the usage-percent gauge), the total is read from the
    # vllm:cache_config_info info gauge above. As a final fallback we sum the
    # component gauges if any of them are exposed.
    if "total_blocks" not in m:
        parts = [m.get(k, 0) for k in ("used_blocks", "cached_blocks", "free_blocks")]
        if any(v > 0 for v in parts):
            m["total_blocks"] = sum(parts)

    total = m.get("total_blocks")
    free = m.get("free_blocks")

    # When both free and total come from /metrics, recompute usage_pct from
    # them so the dashboard's KV-used line shares a denominator with the
    # admission controller's free-percent. (vllm-ascend's npu_cache_usage_perc
    # gauge is derived from the worker's pre-scheduler block count and can
    # disagree with num_npu_blocks_free.)
    if free is not None and total:
        m["usage_pct"] = max(0.0, min(1.0, 1.0 - free / total))
    elif "usage_pct" not in m:
        used = m.get("used_blocks")
        if used is not None and total:
            m["usage_pct"] = used / total

    # Derive missing per-pool block counts from usage_pct × total so downstream
    # GB conversions still work when vLLM only exposes the usage gauge.
    pct = m.get("usage_pct")
    if pct is not None and total:
        if "used_blocks" not in m:
            m["used_blocks"] = max(0, int(round(total * pct)))
        if "free_blocks" not in m:
            m["free_blocks"] = max(0, int(round(total * (1.0 - pct))))

    if preempt_samples:
        # Some vLLM builds expose aliases for the same scheduler counter. Sum
        # label partitions within each metric name, then keep the largest alias.
        m["scheduler_preemptions_total"] = int(max(preempt_samples.values()))

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


def default_offload_endpoint(vllm_url: str) -> str:
    """Return the default local vLLM agent-KV CPU-offload endpoint."""
    base = vllm_url.rstrip("/") + "/"
    return urljoin(base, "agent_kv_cache/offload")


def default_restore_endpoint(vllm_url: str) -> str:
    """Return the default local vLLM agent-KV restore/readmit endpoint."""
    base = vllm_url.rstrip("/") + "/"
    return urljoin(base, "agent_kv_cache/restore")


def default_release_endpoint(vllm_url: str) -> str:
    """Return the default local vLLM agent-KV hold-release endpoint."""
    base = vllm_url.rstrip("/") + "/"
    return urljoin(base, "agent_kv_cache/release")


def default_usage_endpoint(vllm_url: str) -> str:
    """Return the default local vLLM agent-KV usage endpoint."""
    base = vllm_url.rstrip("/") + "/"
    return urljoin(base, "agent_kv_cache/usage")


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
    previously_offloaded: bool = False
    queued_ts: str = field(default_factory=lambda: iso_utc(now_utc()))


@dataclass
class _IdleAgentCandidate:
    agent_id: str
    kv_gb: float
    predicted_remaining_s: Optional[float]
    score: float
    tool_elapsed_s: Optional[float] = None
    policy_reason: str = "eligible_for_pressure_offload"


class DynamicAdmissionController:
    """Admission/offload policy for multi-agent vLLM KV-cache pressure."""

    ACTIVE_STATES = {"reasoning", "tool_call", "waiting"}

    def __init__(
        self,
        *,
        enabled: bool = False,
        threshold_percent: float = 10.0,
        threshold_gb: Optional[float] = None,
        w_threshold: float = 2.0,
        initial_admit_interval_s: float = 2.0,
        max_fresh_admits_per_tick: int = 1,
        max_active_agents: int = 0,
        short_tool_call_threshold_s: float = 2.0,
        fallback_long_tool_call_s: float = 30.0,
        offload_endpoint: Optional[str] = None,
        restore_endpoint: Optional[str] = None,
        release_endpoint: Optional[str] = None,
        usage_endpoint: Optional[str] = None,
        offload_timeout_s: float = DEFAULT_OFFLOAD_TIMEOUT_S,
        exact_freed_gb_timeout_s: float = DEFAULT_EXACT_FREED_GB_TIMEOUT_S,
        exact_freed_gb_poll_interval_s: float = 0.05,
        vllm_url: Optional[str] = None,
        bytes_per_blk: Optional[int] = None,
        predictor_model: Optional[Path | str] = None,
        predictor: Any = None,
        admit_callback: Optional[Callable[[AgentLaunchSpec], Any]] = None,
        state_update_callback: Optional[Callable[[str, dict[str, Any]], None]] = None,
        agent_state_reader: Optional[Callable[[str], Optional[dict[str, Any]]]] = None,
        offload_callback: Optional[Callable[[_IdleAgentCandidate], dict[str, Any]]] = None,
        restore_callback: Optional[Callable[[str], dict[str, Any]]] = None,
        release_callback: Optional[Callable[[str, str], dict[str, Any]]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.threshold_percent = min(100.0, max(0.0, float(threshold_percent)))
        self._legacy_threshold_gb = (
            None if threshold_gb is None else max(0.0, float(threshold_gb))
        )
        self.w_threshold = max(0.0, float(w_threshold))
        self.initial_admit_interval_s = max(0.0, float(initial_admit_interval_s))
        self.max_fresh_admits_per_tick = max(1, int(max_fresh_admits_per_tick))
        self.max_active_agents = max(0, int(max_active_agents))
        self.short_tool_call_threshold_s = max(0.0, float(short_tool_call_threshold_s))
        self.fallback_long_tool_call_s = max(0.0, float(fallback_long_tool_call_s))
        self.offload_endpoint = str(offload_endpoint or "")
        self.restore_endpoint = str(restore_endpoint) if restore_endpoint else ""
        self.release_endpoint = str(release_endpoint) if release_endpoint else ""
        self.usage_endpoint = str(usage_endpoint) if usage_endpoint else ""
        self.offload_timeout_s = max(0.1, float(offload_timeout_s))
        self.exact_freed_gb_timeout_s = max(0.0, float(exact_freed_gb_timeout_s))
        self.exact_freed_gb_poll_interval_s = max(
            0.01, float(exact_freed_gb_poll_interval_s)
        )
        self.vllm_url = str(vllm_url or "")
        self.bytes_per_blk = int(bytes_per_blk) if bytes_per_blk else None
        self._admit_callback = admit_callback
        self._state_update_callback = state_update_callback
        self._agent_state_reader = agent_state_reader
        self._offload_callback = offload_callback
        self._restore_callback = restore_callback
        self._release_callback = release_callback
        self._session = session or requests.Session()

        self._fresh: deque[AgentLaunchSpec] = deque()
        self._offloaded_ready: deque[str] = deque()
        self._offloaded_ready_set: set[str] = set()
        self._offloaded_events: dict[str, threading.Event] = {}
        self._readmitted_at: dict[str, str] = {}
        self._offloaded_pending: set[str] = set()
        self._idle_short: set[str] = set()
        self._idle_long: set[str] = set()
        self._released_agents: set[str] = set()
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
            spec.previously_offloaded = False
            self._fresh.append(spec)

    def pending_counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "fresh": len(self._fresh),
                "offloaded_ready": len(self._offloaded_ready),
                "offloaded_pending_tool": len(self._offloaded_pending),
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
                "tool_elapsed_s": None,
                "threshold_s": self.short_tool_call_threshold_s,
                "fallback_long_tool_call_s": self.fallback_long_tool_call_s,
                "reason": "",
            }
            self._released_agents.discard(agent_id)
            if not self.enabled:
                event["reason"] = "disabled"
                self._record_kv_policy_event_locked(event)
                return event

            predicted = self._predict_remaining(agent)
            if predicted is None:
                self._idle_short.discard(agent_id)
                self._idle_long.discard(agent_id)
                elapsed_s = self._tool_elapsed_s(agent)
                event["tool_elapsed_s"] = self._round(elapsed_s)
                if elapsed_s is not None and elapsed_s >= self.fallback_long_tool_call_s:
                    self._idle_long.add(agent_id)
                    event.update({
                        "policy": "idle_long",
                        "reason": "fallback_elapsed_long_tool_call",
                    })
                    self._update_agent_state_locked(
                        agent_id,
                        {
                            "kv_policy": "idle_long",
                            "kv_offloaded": False,
                            "admission_state": "admitted",
                            "predicted_remaining_s": None,
                            "tool_elapsed_s": self._round(elapsed_s),
                            "fallback_long_tool_call_s": self.fallback_long_tool_call_s,
                            "last_kv_policy": event,
                        },
                    )
                    self._record_kv_policy_event_locked(event)
                    return event

                event["reason"] = self._predictor_error or "predictor_unavailable"
                self._update_agent_state_locked(
                    agent_id,
                    {
                        "kv_policy": "predictor_unavailable",
                        "kv_policy_reason": event["reason"],
                        "tool_elapsed_s": self._round(elapsed_s),
                        "fallback_long_tool_call_s": self.fallback_long_tool_call_s,
                        "last_kv_policy": event,
                    },
                )
                self._record_kv_policy_event_locked(event)
                return event

            event["predicted_remaining_s"] = self._round(predicted)
            if predicted < self.short_tool_call_threshold_s:
                self._idle_long.discard(agent_id)
                self._idle_short.add(agent_id)
                release_result = self._release_agent_locked(
                    agent_id, "short_tool_call")
                event.update({
                    "policy": "idle_short",
                    "reason": "predicted_short",
                    "release_result": release_result,
                })
                self._update_agent_state_locked(
                    agent_id,
                    {
                        "kv_policy": "idle_short",
                        "kv_offloaded": False,
                        "admission_state": "admitted",
                        "predicted_remaining_s": self._round(predicted),
                        "last_release_result": release_result,
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
                    "kv_offloaded": False,
                    "admission_state": "admitted",
                    "predicted_remaining_s": self._round(predicted),
                    "last_kv_policy": event,
                },
            )
            self._record_kv_policy_event_locked(event)
            return event

    def wait_if_offloaded(
        self, agent_id: str, *, return_admitted_at: bool = False
    ) -> bool | tuple[bool, Optional[str]]:
        """Block an offloaded agent after a tool call until re-admission."""
        with self._lock:
            event = self._offloaded_events.get(agent_id)
            if event is None:
                return (False, None) if return_admitted_at else False
            if event.is_set():
                admitted_at = self._readmitted_at.pop(agent_id, None)
                self._clear_offloaded_locked(agent_id)
                if admitted_at is not None:
                    return (True, admitted_at) if return_admitted_at else True
                return (False, None) if return_admitted_at else False
            if agent_id not in self._offloaded_ready_set:
                self._offloaded_ready.append(agent_id)
                self._offloaded_ready_set.add(agent_id)
            self._offloaded_pending.discard(agent_id)
            self._update_agent_state_locked(
                agent_id,
                {
                    "state": "offloaded_waiting",
                    "state_since": iso_utc(now_utc()),
                    "admission_state": "offloaded_ready",
                },
            )

        event.wait()
        with self._lock:
            admitted_at = self._readmitted_at.pop(agent_id, None) or iso_utc(now_utc())
            self._clear_offloaded_locked(agent_id)
            self._update_agent_state_locked(
                agent_id,
                {
                    "kv_offloaded": False,
                    "kv_policy": "readmitted",
                    "admission_state": "admitted",
                    "admitted_since": admitted_at,
                },
            )
        return (True, admitted_at) if return_admitted_at else True

    def release_agent_kv(self, agent_id: str, reason: str = "release") -> dict[str, Any]:
        """Release any held vLLM KV blocks for an agent that was not offloaded."""
        with self._lock:
            result = self._release_agent_locked(agent_id, reason)
            self._idle_short.discard(agent_id)
            self._idle_long.discard(agent_id)
            self._released_agents.add(agent_id)
            self._update_agent_state_locked(
                agent_id,
                {
                    "kv_offloaded": False,
                    "kv_policy": "released",
                    "kv_policy_reason": reason,
                    "admission_state": "admitted",
                    "last_release_result": result,
                },
            )
            return result

    def finish_agent(self, agent_id: str) -> None:
        """Clean up any queued/blocked admission state for a finished agent."""
        with self._lock:
            release_result = self._release_agent_locked(agent_id, "final")
            event = self._offloaded_events.get(agent_id)
            if event is not None:
                event.set()
            self._update_agent_state_locked(
                agent_id,
                {"last_release_result": release_result},
            )
            self._clear_offloaded_locked(agent_id)

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
            total_gb = self._kv_total_gb(vllm_info, bytes_per_blk)
            free_percent = self._kv_free_percent(free_gb, total_gb)
            threshold_gb = self._pressure_threshold_gb(total_gb)
            threshold_percent = self._pressure_threshold_percent(total_gb)
            usage_report = (
                self._refresh_agent_kv_usage_locked(agents, bytes_per_blk)
                if self.enabled else {"enabled": False, "reason": "admission_disabled"}
            )
            active_agents = self._active_agent_count(agents)
            resident_active_agents = self._resident_active_agent_count(agents)
            avg_gb, avg_count = self._average_active_kv_gb(agents, bytes_per_blk)
            if avg_gb is None:
                avg_gb = self._global_average_active_kv_gb(
                    vllm_info,
                    bytes_per_blk,
                    resident_active_agents,
                )
                if avg_gb is not None:
                    avg_count = resident_active_agents
            prev_avg_gb = self._prev_avg_gb
            initial_w = self._headroom(free_gb, avg_gb, prev_avg_gb)
            pressure = (
                threshold_gb is not None
                and free_gb is not None
                and free_gb <= threshold_gb
            )
            report: dict[str, Any] = {
                "enabled": self.enabled,
                "tick": tick,
                "C": self._round(free_gb),
                "C_percent": self._round(free_percent),
                "kv_total_gb": self._round(total_gb),
                "s_t": self._round(avg_gb),
                "s_prev": self._round(prev_avg_gb),
                "active_agent_samples": avg_count,
                "w": self._round(initial_w),
                "w_source": "current",
                "w_threshold": self._round(self.w_threshold),
                "threshold_percent": self._round(threshold_percent),
                "threshold_gb": self._round(threshold_gb),
                "threshold_source": (
                    "gb_legacy" if self._legacy_threshold_gb is not None else "percent"
                ),
                "pressure": pressure,
                "fallback_long_tool_call_s": self.fallback_long_tool_call_s,
                "first_saturation_seen": self._first_saturation_seen,
                "initial_admit_interval_s": self.initial_admit_interval_s,
                "max_fresh_admits_per_tick": self.max_fresh_admits_per_tick,
                "max_active_agents": self.max_active_agents,
                "active_agents": active_agents,
                "active_agent_slots": self._active_agent_slots(active_agents),
                "scheduler_usage": usage_report,
                "next_initial_admit_in_s": None,
                "queue": self.pending_counts(),
                "heap_candidates": [],
                "skipped_candidates": [],
                "offloads": [],
                "admissions": [],
                "kv_policy_events": self._drain_kv_policy_events_locked(),
                "reasons": [],
            }
            runnable_queue = (
                report["queue"].get("fresh", 0)
                + report["queue"].get("offloaded_ready", 0)
            )

            if self._predictor_error:
                report["reasons"].append(f"predictor_error: {self._predictor_error}")
            if usage_report.get("errors"):
                report["reasons"].append("scheduler_usage_unavailable")
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
                    "tool_elapsed_s": self._round(cand.tool_elapsed_s),
                    "policy_reason": cand.policy_reason,
                    "e_s": self._round(cand.score),
                }
                for _, _, cand in heap
            ]

            current_free_gb = free_gb
            current_free_blocks = self._kv_free_blocks(vllm_info)
            while (
                threshold_gb is not None
                and current_free_gb <= threshold_gb
                and heap
            ):
                _, _, cand = heapq.heappop(heap)
                offload_result = self._offload_candidate(cand)
                report["offloads"].append(offload_result)
                if offload_result.get("offloaded"):
                    freed_gb = _finite_float(offload_result.get("freed_gb"))
                    if freed_gb is not None:
                        offload_result.setdefault("freed_gb_source", "offload_endpoint")
                    else:
                        observed = self._measure_exact_freed_gb(
                            before_free_blocks=current_free_blocks,
                            bytes_per_blk=bytes_per_blk,
                        )
                        if observed is not None:
                            freed_gb = observed["freed_gb"]
                            offload_result.update(observed)
                            offload_result["freed_gb"] = self._round(freed_gb)
                        else:
                            offload_result["freed_gb_source"] = (
                                "pending_async"
                                if offload_result.get("pending")
                                else "unavailable_exact"
                            )
                    if freed_gb is None:
                        freed_gb = 0.0
                    current_free_gb += max(0.0, freed_gb)
                    after_blocks = offload_result.get("free_blocks_after")
                    if after_blocks is not None:
                        current_free_blocks = int(after_blocks)
                    elif (
                        current_free_blocks is not None
                        and offload_result.get("freed_blocks") is not None
                    ):
                        current_free_blocks += int(offload_result["freed_blocks"])
                    elif _finite_float(offload_result.get("freed_gb")) is not None:
                        current_free_blocks = None
                    self._mark_agent_offloaded_locked(cand.agent_id, offload_result)
                elif "offload_attempt_failed" not in report["reasons"]:
                    report["reasons"].append("offload_attempt_failed")

            if threshold_gb is None:
                report["reasons"].append("missing_kv_total_gb")
            elif current_free_gb <= threshold_gb:
                report["reasons"].append("pressure_threshold")

            effective_w = self._headroom(current_free_gb, avg_gb, prev_avg_gb)
            w_source = "after_offload" if effective_w != initial_w else "current"
            report["w"] = self._round(effective_w)
            report["w_source"] = w_source
            if effective_w != initial_w:
                report["w_before_offload"] = self._round(initial_w)

            if effective_w is not None and effective_w <= self.w_threshold:
                report["reasons"].append("headroom_low")

            effective_pressure = (
                threshold_gb is not None
                and current_free_gb <= threshold_gb
            )
            if effective_pressure and effective_w is None and runnable_queue > 0:
                admit_limit = 0
                report["reasons"].append("missing_headroom")
                report["reasons"].append("saturation_guard")
                report["reasons"].append("admission_blocked_by_pressure")
                self._first_saturation_seen = True
                report["first_saturation_seen"] = True
            else:
                admit_limit = self._admission_limit(effective_w)
            active_slots = self._active_agent_slots(active_agents)
            if active_slots is not None:
                admit_limit = min(admit_limit, active_slots)
                report["active_agent_slots"] = active_slots
                if active_slots < runnable_queue:
                    report["reasons"].append("active_agent_cap")
                if active_slots <= 0 and runnable_queue > 0:
                    report["reasons"].append("admission_blocked_by_active_agent_cap")

            if (
                admit_limit <= 0
                and effective_w is not None
                and effective_w <= self.w_threshold
                and runnable_queue > 0
            ):
                self._mark_admission_blocked_by_headroom_locked(report)

            if admit_limit > 0:
                self._admit_with_limit_locked(
                    admit_limit,
                    report,
                    w=effective_w,
                    w_source=w_source,
                )

            report["queue"] = self.pending_counts()
            report["first_saturation_seen"] = self._first_saturation_seen
            self._prev_avg_gb = avg_gb
            return report

    def _refresh_agent_kv_usage_locked(
        self,
        agents: dict[str, dict[str, Any]],
        bytes_per_blk: int,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "enabled": bool(self.usage_endpoint),
            "updated": 0,
            "errors": 0,
            "queried_agents": 0,
            "workers": 0,
            "updated_agents": [],
            "preserved_agents": [],
            "failed_agents": [],
        }
        if not self.usage_endpoint:
            report["reason"] = "missing_usage_endpoint"
            return report

        targets: list[str] = []
        for agent_id, agent in agents.items():
            if agent.get("state") not in self.ACTIVE_STATES:
                continue
            if agent.get("kv_offloaded") or agent_id in self._offloaded_events:
                continue
            targets.append(agent_id)

        report["queried_agents"] = len(targets)
        if not targets:
            return report

        worker_cap = (
            self.max_active_agents if self.max_active_agents > 0 else len(targets)
        )
        workers = min(worker_cap, len(targets))
        report["workers"] = workers
        if workers == 1:
            results = {
                agent_id: self._fetch_agent_kv_usage(agent_id, bytes_per_blk)
                for agent_id in targets
            }
        else:
            results = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_agent = {
                    executor.submit(
                        self._fetch_agent_kv_usage, agent_id, bytes_per_blk
                    ): agent_id
                    for agent_id in targets
                }
                for future in as_completed(future_to_agent):
                    agent_id = future_to_agent[future]
                    try:
                        results[agent_id] = future.result()
                    except Exception as exc:
                        results[agent_id] = {
                            "agent_id": agent_id,
                            "error": str(exc)[:200],
                        }

        for agent_id in targets:
            result = results[agent_id]
            error = result.get("error")
            if error:
                report["errors"] += 1
                report["failed_agents"].append({
                    "agent_id": agent_id,
                    "reason": str(error)[:200],
                })
                continue

            patch = result["patch"]
            agent = agents.get(agent_id)
            preserved = self._preserve_live_tool_kv_on_zero_usage(
                agent, patch, bytes_per_blk
            )
            if agent is not None:
                agent.update(patch)
            self._update_agent_state_locked(agent_id, patch)
            report["updated"] += 1
            report["updated_agents"].append(agent_id)
            if preserved:
                report["preserved_agents"].append(agent_id)
        return report

    def _preserve_live_tool_kv_on_zero_usage(
        self,
        agent: Optional[dict[str, Any]],
        patch: dict[str, Any],
        bytes_per_blk: int,
    ) -> bool:
        """Keep a nonzero tool-call KV snapshot when scheduler usage races to 0."""
        if not agent or agent.get("state") != "tool_call":
            return False
        if patch.get("kv_blocks") != 0:
            return False
        previous_blocks = agent.get("kv_blocks")
        if not isinstance(previous_blocks, int) or previous_blocks <= 0:
            return False

        preserved_gb = previous_blocks * bytes_per_blk / 1e9
        last_usage = dict(patch.get("last_kv_usage") or {})
        last_usage.update({
            "scheduler_reported_kv_blocks": patch.get("kv_blocks"),
            "preserved_live_kv_blocks": previous_blocks,
            "preserved_live_kv_gb": preserved_gb,
            "zero_scheduler_usage_preserved": True,
        })
        patch.update({
            "kv_blocks": previous_blocks,
            "kv_gb": preserved_gb,
            "kv_usage_source": "live_snapshot_preserved_after_zero_scheduler_usage",
            "last_kv_usage": last_usage,
        })
        return True

    def _fetch_agent_kv_usage(
        self,
        agent_id: str,
        bytes_per_blk: int,
    ) -> dict[str, Any]:
        try:
            resp = self._session.get(
                self.usage_endpoint,
                params={"agent_id": agent_id},
                timeout=self.offload_timeout_s,
            )
            status_code = int(getattr(resp, "status_code", 200))
            if status_code >= 400:
                raise RuntimeError(
                    f"HTTP {status_code}: {getattr(resp, 'text', '')[:200]}"
                )
            payload = resp.json()
            if payload.get("available") is False:
                raise RuntimeError(str(payload.get("reason") or "usage unavailable"))
            block_source = "resident_kv_blocks"
            blocks = payload.get("resident_kv_blocks")
            if blocks is None:
                block_source = "kv_blocks"
                blocks = payload.get("kv_blocks")
            if blocks is None:
                block_source = "kv_blocks_used"
                blocks = payload.get("kv_blocks_used")
            if not isinstance(blocks, int):
                blocks = int(blocks)
            if blocks < 0:
                raise ValueError(f"negative kv_blocks: {blocks}")

            patch = {
                "kv_blocks": blocks,
                "kv_gb": blocks * bytes_per_blk / 1e9,
                "kv_usage_source": "scheduler_usage",
                "last_kv_updated": iso_utc(now_utc()),
                "last_kv_usage": {
                    "active_requests": int(payload.get("active_requests") or 0),
                    "held_requests": int(payload.get("held_requests") or 0),
                    "pending_offload": bool(payload.get("pending_offload")),
                    "offload_jobs": int(payload.get("offload_jobs") or 0),
                    "resident_kv_blocks": blocks,
                    "offloadable_kv_blocks": int(
                        payload.get("offloadable_kv_blocks") or 0
                    ),
                    "block_source": block_source,
                },
            }
            return {"agent_id": agent_id, "patch": patch}
        except Exception as exc:
            return {"agent_id": agent_id, "error": str(exc)[:200]}

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
        free_blocks = self._kv_free_blocks(vllm_info)
        if free_blocks is not None:
            return max(0.0, free_blocks * bytes_per_blk / 1e9)

        free_gb = _finite_float(vllm_info.get("kv_free_gb"))
        if free_gb is not None:
            return max(0.0, free_gb)

        total = _finite_float(vllm_info.get("kv_total_gb"))
        used = _finite_float(vllm_info.get("kv_used_gb"))
        if total is not None and used is not None:
            return max(0.0, total - used)
        return None

    def _kv_total_gb(
        self, vllm_info: dict[str, Any], bytes_per_blk: int
    ) -> Optional[float]:
        total_blocks = _finite_float(vllm_info.get("num_gpu_blocks_total"))
        if total_blocks is not None and total_blocks > 0:
            return total_blocks * bytes_per_blk / 1e9

        total = _finite_float(vllm_info.get("kv_total_gb"))
        if total is not None and total > 0:
            return total

        free = _finite_float(vllm_info.get("kv_free_gb"))
        used = _finite_float(vllm_info.get("kv_used_gb"))
        if free is not None and used is not None and free + used > 0:
            return free + used

        pct = _finite_float(vllm_info.get("kv_cache_used_pct"))
        if pct is not None and free is not None:
            used_ratio = pct / 100.0 if pct > 1.0 else pct
            if 0.0 <= used_ratio < 1.0:
                return free / (1.0 - used_ratio)
        return None

    def _kv_free_percent(
        self, free_gb: Optional[float], total_gb: Optional[float]
    ) -> Optional[float]:
        if free_gb is None or total_gb is None or total_gb <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * free_gb / total_gb))

    def _pressure_threshold_gb(self, total_gb: Optional[float]) -> Optional[float]:
        if self._legacy_threshold_gb is not None:
            return self._legacy_threshold_gb
        if total_gb is None or total_gb <= 0:
            return None
        return total_gb * self.threshold_percent / 100.0

    def _pressure_threshold_percent(
        self, total_gb: Optional[float]
    ) -> Optional[float]:
        if self._legacy_threshold_gb is None:
            return self.threshold_percent
        if total_gb is None or total_gb <= 0:
            return None
        return min(100.0, max(0.0, 100.0 * self._legacy_threshold_gb / total_gb))

    def _kv_free_blocks(self, vllm_info: dict[str, Any]) -> Optional[int]:
        free_blocks = _finite_float(vllm_info.get("num_gpu_blocks_free"))
        if free_blocks is None:
            return None
        return max(0, int(free_blocks))

    def _average_active_kv_gb(
        self, agents: dict[str, dict[str, Any]], bytes_per_blk: int
    ) -> tuple[Optional[float], int]:
        samples: list[float] = []
        for agent in agents.values():
            if agent.get("state") not in self.ACTIVE_STATES:
                continue
            if agent.get("kv_offloaded"):
                continue
            kv_gb = _agent_kv_gb(agent, bytes_per_blk)
            if kv_gb is not None and kv_gb > 0:
                samples.append(kv_gb)
        if not samples:
            return None, 0
        return sum(samples) / len(samples), len(samples)

    def _active_agent_count(self, agents: dict[str, dict[str, Any]]) -> int:
        return sum(
            1 for agent in agents.values()
            if agent.get("state") in self.ACTIVE_STATES
        )

    def _resident_active_agent_count(self, agents: dict[str, dict[str, Any]]) -> int:
        return sum(
            1 for agent in agents.values()
            if (
                agent.get("state") in self.ACTIVE_STATES
                and not agent.get("kv_offloaded")
            )
        )

    def _active_agent_slots(self, active_agents: int) -> Optional[int]:
        if self.max_active_agents <= 0:
            return None
        return max(0, self.max_active_agents - active_agents)

    def _global_average_active_kv_gb(
        self,
        vllm_info: dict[str, Any],
        bytes_per_blk: int,
        resident_active_agents: int,
    ) -> Optional[float]:
        if resident_active_agents <= 0:
            return None

        used_gb = _finite_float(vllm_info.get("kv_used_gb"))
        if used_gb is None:
            total_gb = _finite_float(vllm_info.get("kv_total_gb"))
            free_gb = _finite_float(vllm_info.get("kv_free_gb"))
            if total_gb is not None and free_gb is not None:
                used_gb = max(0.0, total_gb - free_gb)

        if used_gb is None:
            used_blocks = _finite_float(vllm_info.get("num_gpu_blocks_used"))
            if used_blocks is not None:
                used_gb = max(0.0, used_blocks * bytes_per_blk / 1e9)

        if used_gb is None or used_gb <= 0:
            return None
        return used_gb / resident_active_agents

    def _headroom(
        self,
        free_gb: Optional[float],
        avg_gb: Optional[float],
        prev_avg_gb: Optional[float],
    ) -> Optional[float]:
        if free_gb is None:
            return None
        recent = [v for v in (avg_gb, prev_avg_gb) if v is not None and v > 0]
        if not recent:
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
            if (
                agent_id in self._offloaded_events
                or agent.get("kv_offloaded")
            ):
                continue
            if (
                agent_id in self._idle_short
                or agent.get("kv_policy") == "idle_short"
            ):
                skipped.append({"agent_id": agent_id, "reason": "idle_short"})
                continue
            if (
                agent_id in self._released_agents
                or agent.get("kv_policy") == "released"
            ):
                skipped.append({"agent_id": agent_id, "reason": "kv_released"})
                continue
            current = self._read_current_agent_state(agent_id)
            if current is not None:
                if current.get("state") != "tool_call":
                    self._idle_long.discard(agent_id)
                    skipped.append({
                        "agent_id": agent_id,
                        "reason": "stale_snapshot_state",
                        "current_state": current.get("state"),
                    })
                    continue
                if current.get("kv_offloaded"):
                    continue
                if current.get("kv_policy") == "released":
                    self._idle_long.discard(agent_id)
                    skipped.append({"agent_id": agent_id, "reason": "kv_released"})
                    continue
                agent = {**agent, **current}
                agent.setdefault("agent_id", agent_id)
            kv_gb = _agent_kv_gb(agent, bytes_per_blk)
            if kv_gb is None or kv_gb <= 0:
                skipped.append({
                    "agent_id": agent_id,
                    "reason": "missing_or_zero_kv",
                    "kv_blocks": agent.get("kv_blocks"),
                    "kv_gb": self._round(_finite_float(agent.get("kv_gb"))),
                    "kv_usage_source": agent.get("kv_usage_source"),
                })
                continue
            remaining_s = self._predict_remaining(agent)
            if remaining_s is None:
                elapsed_s = self._tool_elapsed_s(agent)
                if elapsed_s is None or elapsed_s < self.fallback_long_tool_call_s:
                    skipped_event = {
                        "agent_id": agent_id,
                        "reason": self._predictor_error or "predictor_unavailable",
                        "tool_elapsed_s": self._round(elapsed_s),
                        "fallback_long_tool_call_s": self.fallback_long_tool_call_s,
                    }
                    skipped.append(skipped_event)
                    continue

                if agent_id not in self._idle_long:
                    self._idle_short.discard(agent_id)
                    self._idle_long.add(agent_id)
                    event = {
                        "ts": iso_utc(now_utc()),
                        "agent_id": agent_id,
                        "policy": "idle_long",
                        "predicted_remaining_s": None,
                        "tool_elapsed_s": self._round(elapsed_s),
                        "threshold_s": self.short_tool_call_threshold_s,
                        "fallback_long_tool_call_s": self.fallback_long_tool_call_s,
                        "reason": "fallback_elapsed_long_tool_call",
                    }
                    self._update_agent_state_locked(
                        agent_id,
                        {
                            "kv_policy": "idle_long",
                            "kv_offloaded": False,
                            "admission_state": "admitted",
                            "predicted_remaining_s": None,
                            "tool_elapsed_s": self._round(elapsed_s),
                            "fallback_long_tool_call_s": self.fallback_long_tool_call_s,
                            "last_kv_policy": event,
                        },
                    )
                    self._record_kv_policy_event_locked(event)
                score = kv_gb * elapsed_s
                cand = _IdleAgentCandidate(
                    agent_id=agent_id,
                    kv_gb=kv_gb,
                    predicted_remaining_s=None,
                    tool_elapsed_s=elapsed_s,
                    policy_reason="fallback_elapsed_long_tool_call",
                    score=score,
                )
                heapq.heappush(heap, (-score, agent_id, cand))
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
                    release_result = self._release_agent_locked(
                        agent_id, "short_tool_call")
                    event = {
                        "ts": iso_utc(now_utc()),
                        "agent_id": agent_id,
                        "policy": "idle_short",
                        "predicted_remaining_s": self._round(remaining_s),
                        "threshold_s": self.short_tool_call_threshold_s,
                        "reason": "predicted_short",
                        "release_result": release_result,
                    }
                    self._update_agent_state_locked(
                        agent_id,
                        {
                            "kv_policy": "idle_short",
                            "kv_offloaded": False,
                            "admission_state": "admitted",
                            "predicted_remaining_s": self._round(remaining_s),
                            "last_release_result": release_result,
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
                        "kv_offloaded": False,
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

    def _tool_elapsed_s(self, agent: dict[str, Any]) -> Optional[float]:
        start_dt = _parse_iso_ts(agent.get("tool_started_at")) or _parse_iso_ts(
            agent.get("state_since")
        )
        if start_dt is None:
            return None
        return max(0.0, (now_utc() - start_dt).total_seconds())

    def _read_current_agent_state(self, agent_id: str) -> Optional[dict[str, Any]]:
        if self._agent_state_reader is None:
            return None
        try:
            current = self._agent_state_reader(agent_id)
        except Exception:
            return None
        if not current:
            return None
        return dict(current)

    def _predict_remaining(self, agent: dict[str, Any]) -> Optional[float]:
        elapsed_s = self._tool_elapsed_s(agent) or 0.0

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
            "tool_elapsed_s": self._round(cand.tool_elapsed_s),
            "policy_reason": cand.policy_reason,
        }
        try:
            if self._offload_callback is not None:
                payload = self._offload_callback(cand) or {}
            else:
                if not self.offload_endpoint:
                    return {
                        **result,
                        "offloaded": False,
                        "reason": "missing_offload_endpoint",
                    }
                resp = self._session.post(
                    self.offload_endpoint,
                    json={
                        "agent_id": cand.agent_id,
                        "predicted_remaining_s": cand.predicted_remaining_s,
                        "tool_elapsed_s": cand.tool_elapsed_s,
                        "policy_reason": cand.policy_reason,
                    },
                    timeout=self.offload_timeout_s,
                )
                if resp.status_code >= 400:
                    return {
                        **result,
                        "offloaded": False,
                        "status_code": resp.status_code,
                        "reason": resp.text[:300],
                    }
                payload = resp.json()
            offloaded = bool(payload.get("offloaded", True))
            if offloaded:
                self._idle_long.discard(cand.agent_id)
            merged = {**result, **payload, "offloaded": offloaded}
            if not offloaded and not merged.get("reason"):
                merged["reason"] = "offload_rejected_without_reason"
            return merged
        except _REQUEST_TIMEOUT_EXCEPTIONS as exc:
            return {
                **result,
                "offloaded": False,
                "reason": "offload_request_timeout",
                "timeout_s": self._round(self.offload_timeout_s),
                "error": str(exc),
            }
        except Exception as exc:
            return {**result, "offloaded": False, "reason": str(exc)}

    def _measure_exact_freed_gb(
        self,
        *,
        before_free_blocks: Optional[int],
        bytes_per_blk: int,
    ) -> Optional[dict[str, Any]]:
        if not self.vllm_url or before_free_blocks is None:
            return None

        deadline = time.monotonic() + self.exact_freed_gb_timeout_s
        while True:
            after = poll_vllm(
                self._session,
                self.vllm_url,
                bytes_per_blk,
            )
            after_free_blocks = self._kv_free_blocks(after)
            if after_free_blocks is not None:
                delta_blocks = after_free_blocks - before_free_blocks
                if delta_blocks > 0:
                    return {
                        "freed_blocks": delta_blocks,
                        "free_blocks_before": before_free_blocks,
                        "free_blocks_after": after_free_blocks,
                        "freed_gb": delta_blocks * bytes_per_blk / 1e9,
                        "freed_gb_source": "vllm_free_blocks_delta",
                    }
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.exact_freed_gb_poll_interval_s)

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
                    timeout=self.offload_timeout_s,
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

    def _release_agent_locked(self, agent_id: str, reason: str) -> dict[str, Any]:
        result: dict[str, Any] = {"agent_id": agent_id, "release_reason": reason}
        try:
            if self._release_callback is not None:
                payload = self._release_callback(agent_id, reason) or {}
            else:
                if not self.release_endpoint:
                    return {
                        **result,
                        "released": False,
                        "reason": "missing_release_endpoint",
                    }
                resp = self._session.post(
                    self.release_endpoint,
                    json={"agent_id": agent_id, "reason": reason},
                    timeout=self.offload_timeout_s,
                )
                if resp.status_code >= 400:
                    return {
                        **result,
                        "released": False,
                        "status_code": resp.status_code,
                        "reason": resp.text[:300],
                    }
                payload = resp.json()
            released = bool(payload.get("released", False))
            return {**result, **payload, "released": released}
        except Exception as exc:
            return {**result, "released": False, "reason": str(exc)}

    def _record_kv_policy_event_locked(self, event: dict[str, Any]) -> None:
        self._kv_policy_events.append(dict(event))

    def _drain_kv_policy_events_locked(self) -> list[dict[str, Any]]:
        events = list(self._kv_policy_events)
        self._kv_policy_events.clear()
        return events

    def _mark_agent_offloaded_locked(
        self, agent_id: str, offload_result: dict[str, Any]
    ) -> None:
        event = self._offloaded_events.get(agent_id)
        if event is None or event.is_set():
            event = threading.Event()
            self._offloaded_events[agent_id] = event
        self._offloaded_pending.add(agent_id)
        self._idle_long.discard(agent_id)
        self._released_agents.discard(agent_id)
        self._update_agent_state_locked(
            agent_id,
            {
                "kv_offloaded": True,
                "admission_state": "offloaded_pending_tool",
                "last_offload": iso_utc(now_utc()),
                "last_offload_result": offload_result,
            },
        )

    def _admission_limit(self, w: Optional[float]) -> int:
        if w is None:
            return 1
        if w <= self.w_threshold:
            return 0
        if self.w_threshold <= 0:
            return max(0, int(math.floor(w)))
        return max(1, int(math.floor(w / self.w_threshold)))

    def _admit_with_limit_locked(
        self,
        admit_limit: int,
        report: dict[str, Any],
        *,
        w: Optional[float],
        w_source: str,
    ) -> None:
        if w is not None and w <= self.w_threshold:
            self._mark_admission_blocked_by_headroom_locked(report)
            return

        admitted = 0

        while admitted < admit_limit:
            agent_id = self._pop_next_offloaded_ready_locked()
            if agent_id is None:
                break
            report["admissions"].append(
                self._annotate_admission_decision(
                    self._admit_locked(agent_id),
                    w=w,
                    w_source=w_source,
                )
            )
            admitted += 1

        remaining = admit_limit - admitted
        if remaining <= 0 or not self._fresh:
            return

        if self._admit_callback is None:
            report["reasons"].append("no_admit_callback")
            return

        fresh_limit = remaining
        fresh_limit = min(fresh_limit, self.max_fresh_admits_per_tick)
        if len(self._fresh) > fresh_limit:
            report["reasons"].append("fresh_admit_tick_cap")

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
            report["admissions"].append(
                self._annotate_admission_decision(
                    self._admit_locked(item),
                    w=w,
                    w_source=w_source,
                )
            )
            self._last_fresh_admit_monotonic = time.monotonic()

        if not self._first_saturation_seen and self._fresh and self.initial_admit_interval_s > 0:
            report["next_initial_admit_in_s"] = self._round(self.initial_admit_interval_s)

    def _mark_admission_blocked_by_headroom_locked(
        self, report: dict[str, Any]
    ) -> None:
        for reason in (
            "saturation_guard",
            "admission_blocked_by_headroom",
            "admission_blocked_by_pressure",
        ):
            if reason not in report["reasons"]:
                report["reasons"].append(reason)
        self._first_saturation_seen = True
        report["first_saturation_seen"] = True

    def _annotate_admission_decision(
        self,
        admission: dict[str, Any],
        *,
        w: Optional[float],
        w_source: str,
    ) -> dict[str, Any]:
        admission["w"] = self._round(w)
        admission["w_threshold"] = self._round(self.w_threshold)
        admission["w_source"] = w_source
        return admission

    def _next_initial_admit_in_s_locked(self, now_mono: float) -> float:
        if self._last_fresh_admit_monotonic is None:
            return 0.0
        elapsed = max(0.0, now_mono - self._last_fresh_admit_monotonic)
        return max(0.0, self.initial_admit_interval_s - elapsed)

    def _pop_next_offloaded_ready_locked(self) -> Optional[str]:
        while self._offloaded_ready:
            agent_id = self._offloaded_ready.popleft()
            self._offloaded_ready_set.discard(agent_id)
            if agent_id in self._offloaded_events:
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
            event = self._offloaded_events.get(agent_id)
            admitted_at = iso_utc(now_utc())
            self._readmitted_at[agent_id] = admitted_at
            self._update_agent_state_locked(
                agent_id,
                {
                    "kv_offloaded": False,
                    "kv_policy": "readmitted",
                    "admission_state": "admitted",
                    "admitted_since": admitted_at,
                    "last_restore_result": restore_result,
                },
            )
            if event is not None:
                event.set()
            return {
                "agent_id": agent_id,
                "previously_offloaded": True,
                "admitted": True,
                "admitted_at": admitted_at,
                "restore_result": restore_result,
            }

        assert self._admit_callback is not None
        admitted_at = iso_utc(now_utc())
        self._admit_callback(item)
        return {
            "agent_id": item.agent_id,
            "previously_offloaded": item.previously_offloaded,
            "admitted": True,
            "admitted_at": admitted_at,
        }

    def _clear_offloaded_locked(self, agent_id: str) -> None:
        self._offloaded_events.pop(agent_id, None)
        self._offloaded_pending.discard(agent_id)
        self._offloaded_ready_set.discard(agent_id)
        self._readmitted_at.pop(agent_id, None)
        self._idle_short.discard(agent_id)
        self._idle_long.discard(agent_id)
        self._released_agents.discard(agent_id)
        if self._offloaded_ready:
            self._offloaded_ready = deque(
                aid for aid in self._offloaded_ready if aid != agent_id
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


def poll_vllm(session: requests.Session, vllm_url: str, bytes_per_blk: int) -> dict:
    """Fetch /metrics and return global KV cache statistics."""
    global _kv_metric_warn_emitted
    try:
        resp = session.get(f"{vllm_url}/metrics", timeout=5)
        resp.raise_for_status()
        kv = _parse_vllm_kv_metrics(resp.text)

        used_pct   = kv.get("usage_pct")    # 0..1
        total_blks = kv.get("total_blocks")
        used_blks  = kv.get("used_blocks")
        free_blks  = kv.get("free_blocks")
        if free_blks is None and total_blks is not None and used_blks is not None:
            free_blks = max(0, int(total_blks) - int(used_blks))

        if total_blks is None and not _kv_metric_warn_emitted:
            _kv_metric_warn_emitted = True
            print(
                "[sidecar] WARNING: cannot derive KV total blocks. Expected one "
                "of vllm:num_(g|n)pu_blocks, vllm:cache_config_info{num_gpu_blocks=...}, "
                "or summable per-pool counts in /metrics.\n"
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
        preemptions = kv.get("scheduler_preemptions_total")
        return {
            "kv_cache_used_pct":    round(used_pct * 100, 2) if used_pct is not None else None,
            "num_gpu_blocks_total": int(total_blks) if total_blks is not None else None,
            "num_gpu_blocks_used":  used_blks,
            "num_gpu_blocks_free":  int(free_blks) if free_blks is not None else None,
            "kv_total_gb":          total_gb,
            "kv_used_gb":           used_gb,
            "kv_free_gb":           free_gb,
            "scheduler_preemptions_total": int(preemptions) if preemptions is not None else None,
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
            "vllm":   poll_vllm(session, args.vllm_url, bytes_per_blk),
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

    ac = p.add_argument_group("dynamic admission control")
    ac.add_argument("--admission-control", action="store_true",
                    help="Enable admission/offload policy decisions in the sidecar.")
    ac.add_argument("--admission-threshold-percent", type=float, default=10.0,
                    help="Free KV-cache percent threshold that triggers pressure offload.")
    ac.add_argument("--admission-threshold-gb", type=float, default=None,
                    help=argparse.SUPPRESS)
    ac.add_argument("--admission-w-threshold", type=float, default=2.0,
                    help="Minimum headroom w required before admitting queued agents.")
    ac.add_argument("--initial-admit-interval-s", type=float, default=2.0,
                    help="Before first SAT, admit at most one fresh task per interval.")
    ac.add_argument("--max-fresh-admits-per-tick", type=int, default=1,
                    help="Maximum fresh queued agents the sidecar may launch per poll tick.")
    ac.add_argument("--max-active-agents", type=int, default=0,
                    help="Maximum active admitted agents at once (0 disables this cap).")
    ac.add_argument("--short-tool-call-threshold-s", type=float, default=2.0,
                    help="Predicted tool calls below this duration stay resident.")
    ac.add_argument("--fallback-long-tool-call-s", type=float, default=30.0,
                    help="When the predictor is unavailable, treat tool calls "
                         "older than this many seconds as offload-eligible.")
    ac.add_argument("--admission-predictor-model", default=None,
                    help="Saved remaining-time predictor model for idle-agent scoring.")
    ac.add_argument("--offload-endpoint", default=None,
                    help="vLLM admin endpoint for per-agent KV CPU offload. Defaults to "
                         "<vllm-url>/agent_kv_cache/offload.")
    ac.add_argument("--restore-endpoint", default=None,
                    help="vLLM admin endpoint to notify readmission. Defaults to "
                         "<vllm-url>/agent_kv_cache/restore.")
    ac.add_argument("--release-endpoint", default=None,
                    help="vLLM admin endpoint to release held agent KV without "
                         "offload. Defaults to <vllm-url>/agent_kv_cache/release.")
    ac.add_argument("--usage-endpoint", default=None,
                    help="vLLM admin endpoint for live per-agent KV usage. "
                         "Defaults to <vllm-url>/agent_kv_cache/usage.")
    ac.add_argument("--offload-timeout-s", type=float, default=None,
                    help="HTTP timeout for one offload request.")
    ac.add_argument("--exact-freed-gb-timeout-s", type=float, default=None,
                    help="How long to poll vLLM free blocks after an accepted "
                         "offload before reporting pending async accounting.")

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
            threshold_percent=args.admission_threshold_percent,
            threshold_gb=args.admission_threshold_gb,
            w_threshold=args.admission_w_threshold,
            initial_admit_interval_s=args.initial_admit_interval_s,
            max_fresh_admits_per_tick=args.max_fresh_admits_per_tick,
            max_active_agents=args.max_active_agents,
            short_tool_call_threshold_s=args.short_tool_call_threshold_s,
            fallback_long_tool_call_s=args.fallback_long_tool_call_s,
            offload_endpoint=args.offload_endpoint or default_offload_endpoint(args.vllm_url),
            restore_endpoint=args.restore_endpoint or default_restore_endpoint(args.vllm_url),
            release_endpoint=args.release_endpoint or default_release_endpoint(args.vllm_url),
            usage_endpoint=args.usage_endpoint or default_usage_endpoint(args.vllm_url),
            offload_timeout_s=(
                args.offload_timeout_s
                if args.offload_timeout_s is not None
                else DEFAULT_OFFLOAD_TIMEOUT_S
            ),
            exact_freed_gb_timeout_s=(
                args.exact_freed_gb_timeout_s
                if args.exact_freed_gb_timeout_s is not None
                else DEFAULT_EXACT_FREED_GB_TIMEOUT_S
            ),
            vllm_url=args.vllm_url,
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
