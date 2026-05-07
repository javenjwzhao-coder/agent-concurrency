from __future__ import annotations

import sys
import threading
import time
import types
from datetime import timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    class _FakeSession:
        def post(self, *args, **kwargs):
            raise RuntimeError("requests is not installed")

        def get(self, *args, **kwargs):
            raise RuntimeError("requests is not installed")

    sys.modules["requests"] = types.SimpleNamespace(Session=_FakeSession)

from sidecar import (
    AgentLaunchSpec,
    DynamicAdmissionController,
    iso_utc,
    now_utc,
    poll_vllm,
)


BYTES_PER_GB = 1_000_000_000


class FakePredictor:
    def __init__(self, values: dict[str, float]):
        self.values = values

    def predict_agent_remaining(self, agent: dict, elapsed_s: float) -> float:
        return self.values.get(agent.get("agent_id", ""), 0.0)


class FakeUsageResponse:
    def __init__(self, payload: dict | None = None, status_code: int = 200, text: str = ""):
        self._payload = payload or {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return dict(self._payload)


class FakeUsageSession:
    def __init__(self, payloads: dict[str, dict | FakeUsageResponse]):
        self.payloads = payloads
        self.calls: list[str] = []

    def get(self, url, params=None, timeout=None):
        agent_id = (params or {}).get("agent_id", "")
        self.calls.append(agent_id)
        payload = self.payloads[agent_id]
        if isinstance(payload, FakeUsageResponse):
            return payload
        return FakeUsageResponse(payload)


class SlowUsageSession(FakeUsageSession):
    def __init__(
        self,
        payloads: dict[str, dict | FakeUsageResponse],
        *,
        delay_s: float = 0.03,
    ):
        super().__init__(payloads)
        self.delay_s = delay_s
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def get(self, url, params=None, timeout=None):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            time.sleep(self.delay_s)
            return super().get(url, params=params, timeout=timeout)
        finally:
            with self._lock:
                self._in_flight -= 1


def test_headroom_bootstrap_and_memoized_average():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        initial_admit_interval_s=0.0,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    controller.enqueue_fresh(AgentLaunchSpec("fresh-a"))
    controller.enqueue_fresh(AgentLaunchSpec("fresh-b"))

    agents = {
        "running-a": {"state": "reasoning", "kv_blocks": 2},
        "running-b": {"state": "tool_call", "kv_blocks": 1},
    }
    first = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 5.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert first["s_t"] == 1.5
    assert first["s_prev"] is None
    assert first["w"] == 3.333333
    assert len(first["admissions"]) == 1
    assert first["admissions"][0]["agent_id"] == "fresh-a"
    assert first["admissions"][0]["previously_offloaded"] is False
    assert first["admissions"][0]["admitted"] is True
    assert first["admissions"][0]["admitted_at"]

    controller.enqueue_fresh(AgentLaunchSpec("fresh-c"))
    second = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 3.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert second["s_prev"] == 1.5
    assert second["w"] == 2.0
    assert second["w_threshold"] == 2.0
    assert second["admissions"] == []
    assert "headroom_low" in second["reasons"]
    assert "admission_blocked_by_headroom" in second["reasons"]
    assert admitted == ["fresh-a"]
    assert controller.pending_counts()["fresh"] == 2

    third = controller.on_tick(
        tick=2,
        vllm_info={"kv_free_gb": 3.1},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert third["w"] > 2.0
    assert [a["agent_id"] for a in third["admissions"]] == ["fresh-b"]
    assert "fresh_admit_tick_cap" in third["reasons"]
    assert admitted == ["fresh-a", "fresh-b"]
    assert controller.pending_counts()["fresh"] == 1


def test_poll_vllm_reports_free_capacity_when_metric_is_derived():
    class _Response:
        text = "\n".join([
            "vllm:num_gpu_blocks 10",
            "vllm:num_gpu_blocks_used 4",
        ])

        def raise_for_status(self):
            return None

    class _Session:
        def get(self, url, timeout):
            return _Response()

    info = poll_vllm(_Session(), "http://vllm.invalid", BYTES_PER_GB)

    assert info["num_gpu_blocks_total"] == 10
    assert info["num_gpu_blocks_used"] == 4
    assert info["num_gpu_blocks_free"] == 6
    assert info["kv_free_gb"] == 6.0


def test_poll_vllm_reports_scheduler_preemption_count():
    class _Response:
        text = "\n".join([
            "vllm:num_gpu_blocks 10",
            "vllm:num_gpu_blocks_used 4",
            'vllm:num_preemptions_total{worker="0"} 2',
            'vllm:num_preemptions_total{worker="1"} 3',
            "vllm:num_preemptions_created 123",
        ])

        def raise_for_status(self):
            return None

    class _Session:
        def get(self, url, timeout):
            return _Response()

    info = poll_vllm(_Session(), "http://vllm.invalid", BYTES_PER_GB)

    assert info["scheduler_preemptions_total"] == 5


def test_scheduler_usage_overrides_stale_live_kv_for_scoring():
    offloaded: list[tuple[str, float]] = []
    session = FakeUsageSession({
        "stale-large": {
            "available": True,
            "kv_blocks": 2,
            "active_requests": 1,
            "held_requests": 0,
            "pending_offload": False,
            "offload_jobs": 0,
        },
        "stale-small": {
            "available": True,
            "kv_blocks": 5,
            "active_requests": 1,
            "held_requests": 0,
            "pending_offload": False,
            "offload_jobs": 0,
        },
    })

    def offload(cand):
        offloaded.append((cand.agent_id, cand.kv_gb))
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        predictor=FakePredictor({"stale-large": 10.0, "stale-small": 10.0}),
        offload_callback=offload,
        session=session,
        usage_endpoint="http://vllm.invalid/agent_kv_cache/usage",
    )
    agents = {
        "stale-large": {"agent_id": "stale-large", "state": "tool_call", "kv_blocks": 99},
        "stale-small": {"agent_id": "stale-small", "state": "tool_call", "kv_blocks": 1},
    }

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert set(session.calls) == {"stale-large", "stale-small"}
    assert report["s_t"] == 3.5
    assert report["scheduler_usage"]["updated"] == 2
    assert agents["stale-large"]["kv_blocks"] == 2
    assert agents["stale-small"]["kv_blocks"] == 5
    assert agents["stale-small"]["kv_usage_source"] == "scheduler_usage"
    assert report["offloads"][0]["agent_id"] == "stale-small"
    assert report["offloads"][0]["kv_gb"] == 5.0
    assert offloaded == [("stale-small", 5.0)]


def test_scheduler_usage_queries_active_agents_in_parallel():
    session = SlowUsageSession({
        f"agent-{idx}": {
            "available": True,
            "kv_blocks": idx + 1,
            "active_requests": 1,
            "held_requests": 0,
            "pending_offload": False,
            "offload_jobs": 0,
        }
        for idx in range(4)
    })
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        max_active_agents=2,
        session=session,
        usage_endpoint="http://vllm.invalid/agent_kv_cache/usage",
    )
    agents = {
        f"agent-{idx}": {"state": "reasoning", "kv_blocks": 1}
        for idx in range(4)
    }

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    usage = report["scheduler_usage"]
    assert usage["queried_agents"] == 4
    assert usage["workers"] == 2
    assert usage["updated"] == 4
    assert session.max_in_flight > 1
    assert session.max_in_flight <= 2
    assert set(session.calls) == set(agents)


def test_stale_tool_call_snapshot_is_not_offloaded_after_live_state_changes():
    offloaded: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        predictor=FakePredictor({"agent": 10.0}),
        offload_callback=lambda cand: offloaded.append(cand.agent_id) or {
            "offloaded": True,
            "freed_gb": cand.kv_gb,
        },
        agent_state_reader=lambda agent_id: {
            "agent_id": agent_id,
            "state": "reasoning",
            "kv_blocks": 5,
        },
    )
    stale_snapshot = {
        "agent": {
            "agent_id": "agent",
            "state": "tool_call",
            "state_since": iso_utc(now_utc() - timedelta(seconds=10)),
            "kv_blocks": 5,
        },
    }

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents=stale_snapshot,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["offloads"] == []
    assert offloaded == []
    assert report["skipped_candidates"] == [
        {
            "agent_id": "agent",
            "reason": "stale_snapshot_state",
            "current_state": "reasoning",
        },
    ]


def test_scheduler_usage_prefers_resident_kv_blocks_for_memory_sizing():
    session = FakeUsageSession({
        "running": {
            "available": True,
            "kv_blocks": 2,
            "resident_kv_blocks": 7,
            "offloadable_kv_blocks": 2,
            "active_requests": 1,
            "held_requests": 0,
            "pending_offload": False,
            "offload_jobs": 0,
        },
    })
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        session=session,
        usage_endpoint="http://vllm.invalid/agent_kv_cache/usage",
    )
    agents = {"running": {"state": "reasoning", "kv_blocks": 1}}

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 5.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["s_t"] == 7.0
    assert agents["running"]["kv_blocks"] == 7
    assert agents["running"]["kv_gb"] == 7.0
    assert agents["running"]["last_kv_usage"]["resident_kv_blocks"] == 7
    assert agents["running"]["last_kv_usage"]["offloadable_kv_blocks"] == 2
    assert agents["running"]["last_kv_usage"]["block_source"] == "resident_kv_blocks"


def test_scheduler_usage_failure_preserves_live_kv_snapshot():
    session = FakeUsageSession({
        "running": FakeUsageResponse(status_code=500, text="boom"),
    })
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        session=session,
        usage_endpoint="http://vllm.invalid/agent_kv_cache/usage",
    )
    agents = {"running": {"state": "reasoning", "kv_blocks": 4}}

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 5.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["s_t"] == 4.0
    assert report["scheduler_usage"]["errors"] == 1
    assert "scheduler_usage_unavailable" in report["reasons"]
    assert agents["running"]["kv_blocks"] == 4
    assert "kv_usage_source" not in agents["running"]


def test_zero_scheduler_usage_preserves_tool_call_live_kv_for_fallback_offload():
    offloaded: list[tuple[str, float]] = []
    old_ts = iso_utc(now_utc() - timedelta(seconds=45))
    session = FakeUsageSession({
        "tool": {
            "available": True,
            "resident_kv_blocks": 0,
            "kv_blocks": 0,
            "offloadable_kv_blocks": 0,
            "active_requests": 0,
            "held_requests": 0,
            "pending_offload": False,
            "offload_jobs": 0,
        },
    })

    def offload(cand):
        offloaded.append((cand.agent_id, cand.kv_gb))
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        fallback_long_tool_call_s=30.0,
        offload_callback=offload,
        session=session,
        usage_endpoint="http://vllm.invalid/agent_kv_cache/usage",
    )
    agents = {
        "tool": {
            "agent_id": "tool",
            "state": "tool_call",
            "state_since": old_ts,
            "kv_blocks": 4,
            "kv_gb": 4.0,
            "kv_usage_source": "litellm_usage",
        },
    }

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["scheduler_usage"]["preserved_agents"] == ["tool"]
    assert agents["tool"]["kv_blocks"] == 4
    assert agents["tool"]["kv_gb"] == 4.0
    assert (
        agents["tool"]["kv_usage_source"]
        == "live_snapshot_preserved_after_zero_scheduler_usage"
    )
    assert agents["tool"]["last_kv_usage"]["zero_scheduler_usage_preserved"] is True
    assert report["heap_candidates"][0]["agent_id"] == "tool"
    assert report["heap_candidates"][0]["policy_reason"] == "fallback_elapsed_long_tool_call"
    assert report["offloads"][0]["agent_id"] == "tool"
    assert offloaded == [("tool", 4.0)]


def test_zero_kv_tool_call_has_explicit_skipped_reason():
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        fallback_long_tool_call_s=30.0,
    )
    agents = {
        "tool": {
            "agent_id": "tool",
            "state": "tool_call",
            "state_since": iso_utc(now_utc() - timedelta(seconds=45)),
            "kv_blocks": 0,
            "kv_gb": 0.0,
            "kv_usage_source": "scheduler_usage",
        },
    }

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["offloads"] == []
    assert report["heap_candidates"] == []
    assert report["skipped_candidates"] == [
        {
            "agent_id": "tool",
            "reason": "missing_or_zero_kv",
            "kv_blocks": 0,
            "kv_gb": 0.0,
            "kv_usage_source": "scheduler_usage",
        }
    ]


def test_saturation_guard_blocks_admission_when_headroom_below_one():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    agents = {"running": {"state": "reasoning", "kv_blocks": 2}}

    controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.5},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    controller.enqueue_fresh(AgentLaunchSpec("fresh"))
    report = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 0.5},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["w"] == 0.25
    assert "headroom_low" in report["reasons"]
    assert "saturation_guard" in report["reasons"]
    assert "admission_blocked_by_headroom" in report["reasons"]
    assert "admission_blocked_by_pressure" in report["reasons"]
    assert report["admissions"] == []
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1
    assert report["first_saturation_seen"] is True


def test_low_headroom_without_runnable_queue_does_not_emit_saturation_guard():
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
    )
    agents = {"running": {"state": "reasoning", "kv_blocks": 2}}

    controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.5},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    report = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 0.5},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["w"] == 0.25
    assert "headroom_low" in report["reasons"]
    assert "saturation_guard" not in report["reasons"]
    assert "admission_blocked_by_headroom" not in report["reasons"]
    assert report["queue"]["fresh"] == 0
    assert report["queue"]["offloaded_ready"] == 0


def test_w_threshold_controls_admission_limit():
    controller = DynamicAdmissionController(enabled=True, w_threshold=2.0)

    assert controller._admission_limit(None) == 1
    assert controller._admission_limit(2.0) == 0
    assert controller._admission_limit(2.1) == 1
    assert controller._admission_limit(4.0) == 2


def test_pressure_threshold_alone_does_not_block_fresh_admission():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=3.2,
        initial_admit_interval_s=0.0,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    agents = {"running": {"state": "reasoning", "kv_blocks": 1}}

    controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 3.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    controller.enqueue_fresh(AgentLaunchSpec("fresh"))

    report = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 3.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["w"] == 3.0
    assert [a["agent_id"] for a in report["admissions"]] == ["fresh"]
    assert "pressure_threshold" in report["reasons"]
    assert "admission_blocked_by_pressure" not in report["reasons"]
    assert admitted == ["fresh"]
    assert controller.pending_counts()["fresh"] == 0


def test_pressure_blocks_admission_with_single_headroom_sample():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_percent=15.0,
        initial_admit_interval_s=0.0,
        max_active_agents=20,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    controller.enqueue_fresh(AgentLaunchSpec("fresh"))
    agents = {
        f"running-{idx}": {"state": "reasoning"}
        for idx in range(19)
    }

    report = controller.on_tick(
        tick=0,
        vllm_info={
            "kv_free_gb": 0.3355,
            "kv_used_gb": 33.2145,
            "kv_total_gb": 33.55,
        },
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["pressure"] is True
    assert report["s_t"] == 1.748132
    assert report["s_prev"] is None
    assert report["w"] == 0.191919
    assert report["w_source"] == "current"
    assert report["active_agents"] == 19
    assert report["active_agent_slots"] == 1
    assert report["admissions"] == []
    assert "headroom_low" in report["reasons"]
    assert "saturation_guard" in report["reasons"]
    assert "admission_blocked_by_headroom" in report["reasons"]
    assert "admission_blocked_by_pressure" in report["reasons"]
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1

    next_report = controller.on_tick(
        tick=1,
        vllm_info={
            "kv_free_gb": 0.3355,
            "kv_used_gb": 33.2145,
            "kv_total_gb": 33.55,
        },
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert next_report["s_t"] == 1.748132
    assert next_report["s_prev"] == 1.748132
    assert next_report["w"] < next_report["w_threshold"]
    assert next_report["w_source"] == "current"
    assert next_report["admissions"] == []
    assert "headroom_low" in next_report["reasons"]
    assert "admission_blocked_by_headroom" in next_report["reasons"]
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1


def test_regression_pressure_does_not_admit_when_effective_w_below_threshold():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_percent=15.0,
        w_threshold=2.0,
        initial_admit_interval_s=0.0,
        max_fresh_admits_per_tick=4,
        max_active_agents=20,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    warm_agents = {
        f"running-{idx}": {"state": "reasoning", "kv_gb": 3.4939}
        for idx in range(16)
    }
    pressure_agents = {
        f"running-{idx}": {"state": "reasoning", "kv_gb": 3.2068}
        for idx in range(16)
    }

    controller.on_tick(
        tick=384,
        vllm_info={"kv_free_gb": 10.0, "kv_total_gb": 33.554667},
        agents=warm_agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    for idx in range(3):
        controller.enqueue_fresh(AgentLaunchSpec(f"fresh-{idx}"))

    report = controller.on_tick(
        tick=385,
        vllm_info={"kv_free_gb": 1.3086, "kv_total_gb": 33.554667},
        agents=pressure_agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["pressure"] is True
    assert report["active_agents"] == 16
    assert report["active_agent_slots"] == 4
    assert report["s_t"] == 3.2068
    assert report["s_prev"] == 3.4939
    assert report["w"] == 0.40807
    assert report["w"] < report["w_threshold"]
    assert report["w_source"] == "current"
    assert report["admissions"] == []
    assert "headroom_low" in report["reasons"]
    assert "saturation_guard" in report["reasons"]
    assert "admission_blocked_by_headroom" in report["reasons"]
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 3


def test_admit_with_limit_defensively_refuses_low_w():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    controller.enqueue_fresh(AgentLaunchSpec("fresh"))
    report = {"admissions": [], "reasons": [], "first_saturation_seen": False}

    controller._admit_with_limit_locked(
        1,
        report,
        w=0.5,
        w_source="current",
    )

    assert report["admissions"] == []
    assert "saturation_guard" in report["reasons"]
    assert "admission_blocked_by_headroom" in report["reasons"]
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1


def test_pressure_blocks_admission_when_headroom_has_no_samples():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=1.0,
        initial_admit_interval_s=0.0,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    controller.enqueue_fresh(AgentLaunchSpec("fresh"))

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.5},
        agents={"running": {"state": "reasoning"}},
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["pressure"] is True
    assert report["s_t"] is None
    assert report["s_prev"] is None
    assert report["w"] is None
    assert report["admissions"] == []
    assert "missing_headroom" in report["reasons"]
    assert "admission_blocked_by_pressure" in report["reasons"]
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1


def test_percent_threshold_derives_pressure_from_total_capacity():
    offloaded: list[str] = []

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_percent=10.0,
        predictor=FakePredictor({"agent": 5.0}),
        offload_callback=offload,
    )

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 3.0, "kv_total_gb": 32.0},
        agents={
            "agent": {
                "agent_id": "agent",
                "state": "tool_call",
                "kv_blocks": 1,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["threshold_percent"] == 10.0
    assert report["threshold_gb"] == 3.2
    assert report["C_percent"] == 9.375
    assert report["pressure"] is True
    assert offloaded == ["agent"]


def test_pressure_offload_pops_highest_score_idle_agent():
    offloaded: list[str] = []

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {
            "offloaded": True,
            "freed_blocks": int(cand.kv_gb),
            "freed_gb": cand.kv_gb,
            "reason": "ok",
        }

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        predictor=FakePredictor({"agent-a": 10.0, "agent-b": 1.0}),
        offload_callback=offload,
    )
    agents = {
        "agent-a": {
            "agent_id": "agent-a",
            "state": "tool_call",
            "kv_blocks": 1,
        },
        "agent-b": {
            "agent_id": "agent-b",
            "state": "tool_call",
            "kv_blocks": 3,
        },
    }

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["pressure"] is True
    assert offloaded == ["agent-a"]
    assert report["offloads"][0]["agent_id"] == "agent-a"
    assert report["offloads"][0]["e_s"] == 10.0


def test_successful_offload_without_exact_measurement_does_not_fake_freed_gb():
    def offload(cand):
        return {"offloaded": True, "reason": "queued"}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        predictor=FakePredictor({"agent": 5.0}),
        offload_callback=offload,
        bytes_per_blk=BYTES_PER_GB,
    )

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "agent": {
                "agent_id": "agent",
                "state": "tool_call",
                "kv_blocks": 2,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["offloads"][0]["offloaded"] is True
    assert "freed_gb" not in report["offloads"][0]
    assert report["offloads"][0]["freed_gb_source"] == "unavailable_exact"


def test_pending_async_offload_without_exact_measurement_reports_pending_source():
    def offload(cand):
        return {
            "offloaded": True,
            "pending": True,
            "held_requests": 1,
            "known_blocks": 2,
            "offload_jobs": 1,
            "reason": "queued for async connector offload",
        }

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        predictor=FakePredictor({"agent": 5.0}),
        offload_callback=offload,
        bytes_per_blk=BYTES_PER_GB,
    )

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "agent": {
                "agent_id": "agent",
                "state": "tool_call",
                "kv_blocks": 2,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    offload_report = report["offloads"][0]
    assert offload_report["offloaded"] is True
    assert "freed_gb" not in offload_report
    assert offload_report["freed_gb_source"] == "pending_async"
    assert offload_report["pending"] is True
    assert offload_report["held_requests"] == 1
    assert offload_report["known_blocks"] == 2


def test_successful_offload_reports_exact_freed_gb_from_vllm_free_blocks_delta():
    def offload(cand):
        return {"offloaded": True, "reason": "queued"}

    class _Response:
        text = "\n".join([
            "vllm:num_gpu_blocks 10",
            "vllm:num_gpu_blocks_used 8",
        ])

        def raise_for_status(self):
            return None

    class _Session:
        def get(self, url, timeout):
            return _Response()

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        predictor=FakePredictor({"agent": 5.0}),
        offload_callback=offload,
        bytes_per_blk=BYTES_PER_GB,
        session=_Session(),
        vllm_url="http://vllm.invalid",
        exact_freed_gb_timeout_s=0.0,
    )

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05, "num_gpu_blocks_free": 0},
        agents={
            "agent": {
                "agent_id": "agent",
                "state": "tool_call",
                "kv_blocks": 2,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["offloads"][0]["freed_blocks"] == 2
    assert report["offloads"][0]["free_blocks_before"] == 0
    assert report["offloads"][0]["free_blocks_after"] == 2
    assert report["offloads"][0]["freed_gb"] == 2.0
    assert report["offloads"][0]["freed_gb_source"] == "vllm_free_blocks_delta"


def test_initial_admit_ramp_limits_fresh_admissions_before_first_sat():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        initial_admit_interval_s=2.0,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    for idx in range(3):
        controller.enqueue_fresh(AgentLaunchSpec(f"fresh-{idx}"))
    agents = {"running": {"state": "reasoning", "kv_blocks": 1}}

    first = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    second = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    controller._last_fresh_admit_monotonic = time.monotonic() - 3.0
    third = controller.on_tick(
        tick=2,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert [a["agent_id"] for a in first["admissions"]] == ["fresh-0"]
    assert "initial_admit_ramp_active" in first["reasons"]
    assert first["next_initial_admit_in_s"] == 2.0
    assert second["admissions"] == []
    assert "initial_admit_ramp_wait" in second["reasons"]
    assert second["next_initial_admit_in_s"] > 0
    assert [a["agent_id"] for a in third["admissions"]] == ["fresh-1"]
    assert admitted == ["fresh-0", "fresh-1"]
    assert controller.pending_counts()["fresh"] == 1


def test_first_sat_disables_initial_fresh_admit_ramp():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        initial_admit_interval_s=60.0,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    agents = {"running": {"state": "reasoning", "kv_blocks": 2}}

    controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.5},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    for idx in range(3):
        controller.enqueue_fresh(AgentLaunchSpec(f"fresh-{idx}"))

    sat = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 0.5},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    after_sat = controller.on_tick(
        tick=2,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert "saturation_guard" in sat["reasons"]
    assert sat["first_saturation_seen"] is True
    assert after_sat["first_saturation_seen"] is True
    assert "initial_admit_ramp_wait" not in after_sat["reasons"]
    assert [a["agent_id"] for a in after_sat["admissions"]] == ["fresh-0"]
    assert "fresh_admit_tick_cap" in after_sat["reasons"]
    assert admitted == ["fresh-0"]
    assert controller.pending_counts()["fresh"] == 2


def test_fresh_admit_tick_cap_can_be_tuned():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        w_threshold=1.0,
        initial_admit_interval_s=0.0,
        max_fresh_admits_per_tick=2,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    for idx in range(4):
        controller.enqueue_fresh(AgentLaunchSpec(f"fresh-{idx}"))
    agents = {"running": {"state": "reasoning", "kv_blocks": 1}}

    first = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    report = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert [a["agent_id"] for a in first["admissions"]] == ["fresh-0", "fresh-1"]
    assert "fresh_admit_tick_cap" in first["reasons"]
    assert [a["agent_id"] for a in report["admissions"]] == ["fresh-2", "fresh-3"]
    assert admitted == ["fresh-0", "fresh-1", "fresh-2", "fresh-3"]
    assert controller.pending_counts()["fresh"] == 0


def test_max_active_agents_blocks_fresh_admission_when_full():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        initial_admit_interval_s=0.0,
        max_active_agents=2,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    controller.enqueue_fresh(AgentLaunchSpec("fresh"))

    agents = {
        "reasoning": {"state": "reasoning", "kv_blocks": 1},
        "tool": {"state": "tool_call", "kv_blocks": 1},
    }
    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["active_agents"] == 2
    assert report["max_active_agents"] == 2
    assert report["active_agent_slots"] == 0
    assert report["admissions"] == []
    assert "active_agent_cap" in report["reasons"]
    assert "admission_blocked_by_active_agent_cap" in report["reasons"]
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1


def test_max_active_agents_limits_admissions_to_available_slots():
    admitted: list[str] = []
    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        initial_admit_interval_s=0.0,
        max_fresh_admits_per_tick=4,
        max_active_agents=3,
        admit_callback=lambda spec: admitted.append(spec.agent_id),
    )
    agents = {
        "reasoning": {"state": "reasoning", "kv_blocks": 1},
        "waiting": {"state": "waiting", "kv_blocks": 1},
    }
    controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    for idx in range(3):
        controller.enqueue_fresh(AgentLaunchSpec(f"fresh-{idx}"))

    report = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["active_agents"] == 2
    assert report["active_agent_slots"] == 1
    assert [a["agent_id"] for a in report["admissions"]] == ["fresh-0"]
    assert admitted == ["fresh-0"]
    assert controller.pending_counts()["fresh"] == 2


def test_offloaded_ready_readmit_bypasses_initial_fresh_admit_ramp():
    state_updates: dict[str, dict] = {}
    offloaded: list[str] = []
    admitted: list[str] = []

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    def update(agent_id, patch):
        state_updates.setdefault(agent_id, {}).update(patch)

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        initial_admit_interval_s=60.0,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"offloaded-agent": 5.0}),
        admit_callback=lambda spec: admitted.append(spec.agent_id),
        offload_callback=offload,
        restore_callback=lambda agent_id: {"restored": True},
        state_update_callback=update,
        bytes_per_blk=BYTES_PER_GB,
    )
    agents = {"running": {"state": "reasoning", "kv_blocks": 1}}
    controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )
    controller.on_tool_call_start(
        "offloaded-agent",
        {"agent_id": "offloaded-agent", "state": "tool_call", "kv_blocks": 1},
    )
    controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "offloaded-agent": {
                "agent_id": "offloaded-agent",
                "state": "tool_call",
                "kv_blocks": 1,
                "kv_policy": "idle_long",
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    resumed: list[bool] = []
    waiter = threading.Thread(
        target=lambda: resumed.append(controller.wait_if_offloaded("offloaded-agent"))
    )
    waiter.start()
    deadline = time.time() + 2
    while controller.pending_counts()["offloaded_ready"] == 0 and time.time() < deadline:
        time.sleep(0.01)

    controller.enqueue_fresh(AgentLaunchSpec("fresh-agent"))
    controller._last_fresh_admit_monotonic = time.monotonic()
    report = controller.on_tick(
        tick=2,
        vllm_info={"kv_free_gb": 10.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    waiter.join(timeout=2)
    assert offloaded == ["offloaded-agent"]
    assert resumed == [True]
    assert report["admissions"][0]["agent_id"] == "offloaded-agent"
    assert report["admissions"][0]["previously_offloaded"] is True
    assert report["admissions"][0]["admitted_at"]
    assert report["admissions"][0]["restore_result"]["restored"] is True
    assert "initial_admit_ramp_wait" in report["reasons"]
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1
    assert state_updates["offloaded-agent"]["admission_state"] == "admitted"


def test_short_tool_call_is_classified_and_excluded_from_pressure_offload():
    offloaded: list[str] = []
    released: list[tuple[str, str]] = []

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    def release(agent_id, reason):
        released.append((agent_id, reason))
        return {"released": True, "held_requests": 1}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"short-agent": 1.0}),
        offload_callback=offload,
        release_callback=release,
        bytes_per_blk=BYTES_PER_GB,
    )
    policy = controller.on_tool_call_start(
        "short-agent",
        {
            "agent_id": "short-agent",
            "state": "tool_call",
            "kv_blocks": 2,
        },
    )

    assert policy["policy"] == "idle_short"
    assert policy["release_result"]["released"] is True
    assert released == [("short-agent", "short_tool_call")]

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "short-agent": {
                "agent_id": "short-agent",
                "state": "tool_call",
                "kv_blocks": 2,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert offloaded == []
    assert report["offloads"] == []
    assert report["skipped_candidates"] == [
        {"agent_id": "short-agent", "reason": "idle_short"}
    ]


def test_long_tool_call_pressure_offloads_and_uses_readmit_flow():
    state_updates: dict[str, dict] = {}
    offloaded: list[str] = []
    admitted: list[str] = []

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    def update(agent_id, patch):
        state_updates.setdefault(agent_id, {}).update(patch)

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"long-agent": 5.0}),
        admit_callback=lambda spec: admitted.append(spec.agent_id),
        offload_callback=offload,
        restore_callback=lambda agent_id: {"restored": True},
        state_update_callback=update,
        bytes_per_blk=BYTES_PER_GB,
    )

    policy = controller.on_tool_call_start(
        "long-agent",
        {
            "agent_id": "long-agent",
            "state": "tool_call",
            "kv_blocks": 3,
        },
    )

    assert policy["policy"] == "idle_long"
    assert offloaded == []
    assert controller.pending_counts()["idle_long"] == 1
    assert controller.pending_counts()["offloaded_pending_tool"] == 0
    assert state_updates["long-agent"]["kv_policy"] == "idle_long"
    assert state_updates["long-agent"]["admission_state"] == "admitted"

    no_pressure = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 5.0},
        agents={
            "long-agent": {
                "agent_id": "long-agent",
                "state": "tool_call",
                "kv_blocks": 3,
                "kv_policy": "idle_long",
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert no_pressure["offloads"] == []
    assert offloaded == []

    pressure = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "long-agent": {
                "agent_id": "long-agent",
                "state": "tool_call",
                "kv_blocks": 3,
                "kv_policy": "idle_long",
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert offloaded == ["long-agent"]
    assert pressure["offloads"][0]["agent_id"] == "long-agent"
    assert controller.pending_counts()["offloaded_pending_tool"] == 1
    assert state_updates["long-agent"]["kv_offloaded"] is True
    assert state_updates["long-agent"]["admission_state"] == "offloaded_pending_tool"

    resumed: list[bool] = []
    waiter = threading.Thread(
        target=lambda: resumed.append(controller.wait_if_offloaded("long-agent"))
    )
    waiter.start()
    deadline = time.time() + 2
    while controller.pending_counts()["offloaded_ready"] == 0 and time.time() < deadline:
        time.sleep(0.01)

    report = controller.on_tick(
        tick=2,
        vllm_info={"kv_free_gb": 7.0},
        agents={},
        bytes_per_blk=BYTES_PER_GB,
    )

    waiter.join(timeout=2)
    assert resumed == [True]
    assert report["admissions"][0]["agent_id"] == "long-agent"
    assert report["admissions"][0]["restore_result"]["restored"] is True
    assert admitted == []


def test_long_tool_call_release_removes_idle_candidate_without_offload():
    state_updates: dict[str, dict] = {}
    offloaded: list[str] = []
    released: list[tuple[str, str]] = []

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    def release(agent_id, reason):
        released.append((agent_id, reason))
        return {"released": True, "held_requests": 1}

    def update(agent_id, patch):
        state_updates.setdefault(agent_id, {}).update(patch)

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"long-agent": 5.0}),
        offload_callback=offload,
        release_callback=release,
        state_update_callback=update,
        bytes_per_blk=BYTES_PER_GB,
    )

    policy = controller.on_tool_call_start(
        "long-agent",
        {
            "agent_id": "long-agent",
            "state": "tool_call",
            "kv_blocks": 3,
        },
    )

    assert policy["policy"] == "idle_long"
    assert controller.pending_counts()["idle_long"] == 1

    result = controller.release_agent_kv("long-agent", "tool_complete")

    assert result["released"] is True
    assert released == [("long-agent", "tool_complete")]
    assert controller.pending_counts()["idle_long"] == 0
    assert state_updates["long-agent"]["kv_policy"] == "released"
    assert state_updates["long-agent"]["kv_policy_reason"] == "tool_complete"
    assert state_updates["long-agent"]["kv_offloaded"] is False
    assert state_updates["long-agent"]["admission_state"] == "admitted"

    pressure = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "long-agent": {
                "agent_id": "long-agent",
                "state": "tool_call",
                "kv_blocks": 3,
                "kv_policy": "idle_long",
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert offloaded == []
    assert pressure["offloads"] == []
    assert pressure["skipped_candidates"] == [
        {"agent_id": "long-agent", "reason": "kv_released"}
    ]


def test_pressure_offload_can_create_one_fresh_admission_slot():
    admitted: list[str] = []
    offloaded: list[str] = []

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        w_threshold=1.0,
        initial_admit_interval_s=0.0,
        predictor=FakePredictor({"idle-agent": 5.0}),
        admit_callback=lambda spec: admitted.append(spec.agent_id),
        offload_callback=offload,
        bytes_per_blk=BYTES_PER_GB,
    )

    controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 5.0},
        agents={"running": {"state": "reasoning", "kv_blocks": 2}},
        bytes_per_blk=BYTES_PER_GB,
    )
    controller.enqueue_fresh(AgentLaunchSpec("fresh-after-offload"))
    controller.on_tool_call_start(
        "idle-agent",
        {"agent_id": "idle-agent", "state": "tool_call", "kv_blocks": 3},
    )

    report = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "idle-agent": {
                "agent_id": "idle-agent",
                "state": "tool_call",
                "kv_blocks": 3,
                "kv_policy": "idle_long",
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert offloaded == ["idle-agent"]
    assert report["w_before_offload"] < 1.0
    assert report["w"] > 1.0
    assert report["w_source"] == "after_offload"
    assert "saturation_guard" not in report["reasons"]
    assert [a["agent_id"] for a in report["admissions"]] == ["fresh-after-offload"]
    assert report["admissions"][0]["w"] == report["w"]
    assert report["admissions"][0]["w_threshold"] == 1.0
    assert report["admissions"][0]["w_source"] == "after_offload"
    assert admitted == ["fresh-after-offload"]


def test_next_long_tool_call_reclassifies_short_idle_then_pressure_offloads():
    offloaded: list[str] = []
    predictor = FakePredictor({"agent": 1.0})

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=predictor,
        offload_callback=offload,
        bytes_per_blk=BYTES_PER_GB,
    )

    controller.on_tool_call_start(
        "agent",
        {"agent_id": "agent", "state": "tool_call", "kv_blocks": 1},
    )
    predictor.values["agent"] = 4.0
    policy = controller.on_tool_call_start(
        "agent",
        {
            "agent_id": "agent",
            "state": "tool_call",
            "kv_blocks": 1,
        },
    )

    assert policy["policy"] == "idle_long"
    assert offloaded == []

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "agent": {
                "agent_id": "agent",
                "state": "tool_call",
                "kv_blocks": 1,
                "kv_policy": "idle_long",
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["offloads"][0]["agent_id"] == "agent"
    assert offloaded == ["agent"]


def test_failed_long_pressure_offload_keeps_agent_admitted():
    offloaded: list[str] = []

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"offloaded": False, "reason": "busy"}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"agent": 5.0}),
        offload_callback=offload,
        bytes_per_blk=BYTES_PER_GB,
    )

    controller.on_tool_call_start(
        "agent",
        {"agent_id": "agent", "state": "tool_call", "kv_blocks": 2},
    )
    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "agent": {
                "agent_id": "agent",
                "state": "tool_call",
                "kv_blocks": 2,
                "kv_policy": "idle_long",
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert offloaded == ["agent"]
    assert report["offloads"][0]["offloaded"] is False
    assert "offload_attempt_failed" in report["reasons"]
    assert controller.pending_counts()["idle_long"] == 1
    assert controller.pending_counts()["offloaded_pending_tool"] == 0


def test_failed_pressure_offload_gets_default_reason_without_offload_state():
    def offload(cand):
        return {"offloaded": False}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"agent": 5.0}),
        offload_callback=offload,
        bytes_per_blk=BYTES_PER_GB,
    )

    controller.on_tool_call_start(
        "agent",
        {"agent_id": "agent", "state": "tool_call", "kv_blocks": 2},
    )
    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "agent": {
                "agent_id": "agent",
                "state": "tool_call",
                "kv_blocks": 2,
                "kv_policy": "idle_long",
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["pressure"] is True
    assert report["offloads"] == [
        {
            "agent_id": "agent",
            "e_s": 10.0,
            "kv_gb": 2.0,
            "predicted_remaining_s": 5.0,
            "tool_elapsed_s": None,
            "policy_reason": "eligible_for_pressure_offload",
            "offloaded": False,
            "reason": "offload_rejected_without_reason",
        }
    ]
    assert "offload_attempt_failed" in report["reasons"]
    assert controller.pending_counts()["idle_long"] == 1
    assert controller.pending_counts()["offloaded_pending_tool"] == 0


def test_predictor_unavailable_skips_idle_offload_policy():
    offloaded: list[str] = []
    released: list[tuple[str, str]] = []

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        offload_callback=lambda cand: offloaded.append(cand.agent_id) or {},
        release_callback=lambda agent_id, reason: released.append((agent_id, reason)) or {},
    )

    policy = controller.on_tool_call_start(
        "agent-no-predictor",
        {
            "agent_id": "agent-no-predictor",
            "state": "tool_call",
            "kv_blocks": 1,
        },
    )
    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "agent-no-predictor": {
                "agent_id": "agent-no-predictor",
                "state": "tool_call",
                "kv_blocks": 1,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert policy["policy"] == "skipped"
    assert policy["reason"] == "predictor_unavailable"
    assert offloaded == []
    assert released == []
    assert report["offloads"] == []
    assert report["skipped_candidates"] == [
        {
            "agent_id": "agent-no-predictor",
            "reason": "predictor_unavailable",
            "tool_elapsed_s": None,
            "fallback_long_tool_call_s": 30.0,
        }
    ]


def test_predictor_unavailable_falls_back_to_elapsed_long_tool_call():
    offloaded: list[str] = []
    state_updates: dict[str, dict] = {}
    old_ts = iso_utc(now_utc() - timedelta(seconds=45))

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    def update(agent_id, patch):
        state_updates.setdefault(agent_id, {}).update(patch)

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        fallback_long_tool_call_s=30.0,
        offload_callback=offload,
        state_update_callback=update,
        bytes_per_blk=BYTES_PER_GB,
    )
    controller._idle_short.add("fallback-agent")

    policy = controller.on_tool_call_start(
        "fallback-agent",
        {
            "agent_id": "fallback-agent",
            "state": "tool_call",
            "state_since": old_ts,
            "kv_blocks": 2,
        },
    )
    assert policy["policy"] == "idle_long"
    assert policy["reason"] == "fallback_elapsed_long_tool_call"

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "fallback-agent": {
                "agent_id": "fallback-agent",
                "state": "tool_call",
                "state_since": old_ts,
                "kv_blocks": 2,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert offloaded == ["fallback-agent"]
    assert report["heap_candidates"][0]["policy_reason"] == "fallback_elapsed_long_tool_call"
    assert report["heap_candidates"][0]["predicted_remaining_s"] is None
    assert report["heap_candidates"][0]["tool_elapsed_s"] >= 30.0
    assert report["offloads"][0]["agent_id"] == "fallback-agent"
    assert report["offloads"][0]["policy_reason"] == "fallback_elapsed_long_tool_call"
    assert report["offloads"][0]["tool_elapsed_s"] >= 30.0
    assert state_updates["fallback-agent"]["kv_policy"] == "idle_long"
    assert state_updates["fallback-agent"]["fallback_long_tool_call_s"] == 30.0


def test_release_agent_kv_uses_release_endpoint_callback():
    released: list[tuple[str, str]] = []

    def release(agent_id, reason):
        released.append((agent_id, reason))
        return {"released": True, "held_requests": 1, "released_requests": 1}

    controller = DynamicAdmissionController(
        enabled=True,
        release_callback=release,
    )

    result = controller.release_agent_kv("agent", "tool_complete")

    assert result["released"] is True
    assert result["held_requests"] == 1
    assert released == [("agent", "tool_complete")]


def test_offloaded_ready_lane_admits_before_fresh_agents():
    admitted: list[str] = []

    def offload(cand):
        return {"offloaded": True, "freed_gb": cand.kv_gb}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        predictor=FakePredictor({"offloaded-agent": 5.0}),
        admit_callback=lambda spec: admitted.append(spec.agent_id),
        offload_callback=offload,
        restore_callback=lambda agent_id: {"restored": True},
    )
    controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "offloaded-agent": {
                "agent_id": "offloaded-agent",
                "state": "tool_call",
                "kv_blocks": 1,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    resumed: list[bool] = []
    waiter = threading.Thread(
        target=lambda: resumed.append(controller.wait_if_offloaded("offloaded-agent"))
    )
    waiter.start()
    deadline = time.time() + 2
    while controller.pending_counts()["offloaded_ready"] == 0 and time.time() < deadline:
        time.sleep(0.01)

    controller.enqueue_fresh(AgentLaunchSpec("fresh-agent"))
    report = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 3.0},
        agents={},
        bytes_per_blk=BYTES_PER_GB,
    )

    waiter.join(timeout=2)
    assert resumed == [True]
    assert report["admissions"][0]["agent_id"] == "offloaded-agent"
    assert report["admissions"][0]["previously_offloaded"] is True
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1
