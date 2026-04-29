from __future__ import annotations

import sys
import threading
import time
import types
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

from sidecar import AgentLaunchSpec, DynamicAdmissionController, poll_vllm


BYTES_PER_GB = 1_000_000_000


class FakePredictor:
    def __init__(self, values: dict[str, float]):
        self.values = values

    def predict_agent_remaining(self, agent: dict, elapsed_s: float) -> float:
        return self.values.get(agent.get("agent_id", ""), 0.0)


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
    assert first["w"] is None
    assert first["admissions"] == [
        {"agent_id": "fresh-a", "previously_evicted": False, "admitted": True}
    ]

    controller.enqueue_fresh(AgentLaunchSpec("fresh-c"))
    second = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 3.0},
        agents=agents,
        bytes_per_blk=BYTES_PER_GB,
    )

    assert second["s_prev"] == 1.5
    assert second["w"] == 2.0
    assert [a["agent_id"] for a in second["admissions"]] == ["fresh-b", "fresh-c"]
    assert admitted == ["fresh-a", "fresh-b", "fresh-c"]


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
    assert report["admissions"] == []
    assert admitted == []
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
    assert report["queue"]["evicted_ready"] == 0


def test_pressure_eviction_pops_highest_score_idle_agent():
    evicted: list[str] = []

    def evict(cand):
        evicted.append(cand.agent_id)
        return {
            "evicted": True,
            "freed_blocks": int(cand.kv_gb),
            "freed_gb": cand.kv_gb,
            "reason": "ok",
        }

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        predictor=FakePredictor({"agent-a": 10.0, "agent-b": 1.0}),
        evict_callback=evict,
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

    assert evicted == ["agent-a"]
    assert report["evictions"][0]["agent_id"] == "agent-a"
    assert report["evictions"][0]["e_s"] == 10.0


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
    assert [a["agent_id"] for a in after_sat["admissions"]] == [
        "fresh-0",
        "fresh-1",
        "fresh-2",
    ]
    assert admitted == ["fresh-0", "fresh-1", "fresh-2"]


def test_evicted_ready_readmit_bypasses_initial_fresh_admit_ramp():
    state_updates: dict[str, dict] = {}
    offloaded: list[str] = []
    admitted: list[str] = []

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"evicted": True, "offloaded": True, "freed_gb": cand.kv_gb}

    def update(agent_id, patch):
        state_updates.setdefault(agent_id, {}).update(patch)

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        initial_admit_interval_s=60.0,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"evicted-agent": 5.0}),
        admit_callback=lambda spec: admitted.append(spec.agent_id),
        offload_callback=offload,
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
        "evicted-agent",
        {"agent_id": "evicted-agent", "state": "tool_call", "kv_blocks": 1},
    )
    controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "evicted-agent": {
                "agent_id": "evicted-agent",
                "state": "tool_call",
                "kv_blocks": 1,
                "kv_policy": "pinned_long",
                "kv_pinned": True,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    resumed: list[bool] = []
    waiter = threading.Thread(
        target=lambda: resumed.append(controller.wait_if_evicted("evicted-agent"))
    )
    waiter.start()
    deadline = time.time() + 2
    while controller.pending_counts()["evicted_ready"] == 0 and time.time() < deadline:
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
    assert offloaded == ["evicted-agent"]
    assert resumed == [True]
    assert report["admissions"] == [
        {"agent_id": "evicted-agent", "previously_evicted": True, "admitted": True}
    ]
    assert "initial_admit_ramp_wait" in report["reasons"]
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1
    assert state_updates["evicted-agent"]["admission_state"] == "admitted"


def test_short_tool_call_is_pinned_and_excluded_from_pressure_offload():
    pin_calls: list[tuple[str, bool]] = []
    offloaded: list[str] = []

    def pin(agent_id, pin):
        pin_calls.append((agent_id, pin))
        return {"pinned": pin, "changed_blocks": 1, "reason": "ok"}

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"evicted": True, "freed_gb": cand.kv_gb}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"short-agent": 1.0}),
        pin_callback=pin,
        offload_callback=offload,
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

    assert policy["policy"] == "pinned_short"
    assert pin_calls == [("short-agent", True)]

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
    assert report["evictions"] == []
    assert report["skipped_candidates"] == [
        {"agent_id": "short-agent", "reason": "pinned_short"}
    ]


def test_long_tool_call_pins_then_pressure_offloads_and_uses_readmit_flow():
    state_updates: dict[str, dict] = {}
    pin_calls: list[tuple[str, bool]] = []
    offloaded: list[str] = []
    admitted: list[str] = []

    def pin(agent_id, pin):
        pin_calls.append((agent_id, pin))
        return {"pinned": pin, "changed_blocks": 1, "reason": "ok"}

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"evicted": True, "offloaded": True, "freed_gb": cand.kv_gb}

    def update(agent_id, patch):
        state_updates.setdefault(agent_id, {}).update(patch)

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"long-agent": 5.0}),
        admit_callback=lambda spec: admitted.append(spec.agent_id),
        pin_callback=pin,
        offload_callback=offload,
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

    assert policy["policy"] == "pinned_long"
    assert pin_calls == [("long-agent", True)]
    assert offloaded == []
    assert controller.pending_counts()["pinned_long"] == 1
    assert controller.pending_counts()["evicted_pending_tool"] == 0
    assert state_updates["long-agent"]["kv_policy"] == "pinned_long"
    assert state_updates["long-agent"]["admission_state"] == "admitted"

    no_pressure = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 5.0},
        agents={
            "long-agent": {
                "agent_id": "long-agent",
                "state": "tool_call",
                "kv_blocks": 3,
                "kv_policy": "pinned_long",
                "kv_pinned": True,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert no_pressure["evictions"] == []
    assert offloaded == []

    pressure = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "long-agent": {
                "agent_id": "long-agent",
                "state": "tool_call",
                "kv_blocks": 3,
                "kv_policy": "pinned_long",
                "kv_pinned": True,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert offloaded == ["long-agent"]
    assert pin_calls == [("long-agent", True), ("long-agent", False)]
    assert pressure["evictions"][0]["agent_id"] == "long-agent"
    assert pressure["evictions"][0]["unpin_result"]["pinned"] is False
    assert controller.pending_counts()["evicted_pending_tool"] == 1
    assert state_updates["long-agent"]["kv_evicted"] is True
    assert state_updates["long-agent"]["admission_state"] == "evicted_pending_tool"

    resumed: list[bool] = []
    waiter = threading.Thread(
        target=lambda: resumed.append(controller.wait_if_evicted("long-agent"))
    )
    waiter.start()
    deadline = time.time() + 2
    while controller.pending_counts()["evicted_ready"] == 0 and time.time() < deadline:
        time.sleep(0.01)

    report = controller.on_tick(
        tick=2,
        vllm_info={"kv_free_gb": 5.0},
        agents={},
        bytes_per_blk=BYTES_PER_GB,
    )

    waiter.join(timeout=2)
    assert resumed == [True]
    assert report["admissions"] == [
        {"agent_id": "long-agent", "previously_evicted": True, "admitted": True}
    ]
    assert admitted == []


def test_next_long_tool_call_reclassifies_short_pin_then_pressure_offloads():
    pin_calls: list[tuple[str, bool]] = []
    offloaded: list[str] = []
    predictor = FakePredictor({"agent": 1.0})

    def pin(agent_id, pin):
        pin_calls.append((agent_id, pin))
        return {"pinned": pin, "changed_blocks": 1, "reason": "ok"}

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"evicted": True, "offloaded": True, "freed_gb": cand.kv_gb}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=predictor,
        pin_callback=pin,
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
            "kv_pinned": True,
        },
    )

    assert policy["policy"] == "pinned_long"
    assert pin_calls == [("agent", True), ("agent", True)]
    assert offloaded == []

    report = controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "agent": {
                "agent_id": "agent",
                "state": "tool_call",
                "kv_blocks": 1,
                "kv_policy": "pinned_long",
                "kv_pinned": True,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert report["evictions"][0]["agent_id"] == "agent"
    assert pin_calls == [("agent", True), ("agent", True), ("agent", False)]
    assert offloaded == ["agent"]


def test_failed_long_pressure_offload_restores_pin():
    pin_calls: list[tuple[str, bool]] = []
    offloaded: list[str] = []

    def pin(agent_id, pin):
        pin_calls.append((agent_id, pin))
        return {"pinned": pin, "changed_blocks": 1, "reason": "ok"}

    def offload(cand):
        offloaded.append(cand.agent_id)
        return {"evicted": False, "offloaded": False, "reason": "busy"}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        short_tool_call_threshold_s=2.0,
        predictor=FakePredictor({"agent": 5.0}),
        pin_callback=pin,
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
                "kv_policy": "pinned_long",
                "kv_pinned": True,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    assert offloaded == ["agent"]
    assert pin_calls == [("agent", True), ("agent", False), ("agent", True)]
    assert report["evictions"][0]["evicted"] is False
    assert report["evictions"][0]["repin_result"]["pinned"] is True
    assert controller.pending_counts()["pinned_long"] == 1
    assert controller.pending_counts()["evicted_pending_tool"] == 0


def test_predictor_unavailable_skips_pin_and_offload_policy():
    pin_calls: list[tuple[str, bool]] = []
    offloaded: list[str] = []

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        pin_callback=lambda agent_id, pin: pin_calls.append((agent_id, pin)) or {},
        offload_callback=lambda cand: offloaded.append(cand.agent_id) or {},
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
    assert pin_calls == []
    assert offloaded == []
    assert report["evictions"] == []
    assert report["skipped_candidates"] == [
        {"agent_id": "agent-no-predictor", "reason": "predictor_unavailable"}
    ]


def test_evicted_ready_lane_admits_before_fresh_agents():
    admitted: list[str] = []

    def evict(cand):
        return {"evicted": True, "freed_gb": cand.kv_gb}

    controller = DynamicAdmissionController(
        enabled=True,
        threshold_gb=0.1,
        predictor=FakePredictor({"evicted-agent": 5.0}),
        admit_callback=lambda spec: admitted.append(spec.agent_id),
        evict_callback=evict,
    )
    controller.on_tick(
        tick=0,
        vllm_info={"kv_free_gb": 0.05},
        agents={
            "evicted-agent": {
                "agent_id": "evicted-agent",
                "state": "tool_call",
                "kv_blocks": 1,
            }
        },
        bytes_per_blk=BYTES_PER_GB,
    )

    resumed: list[bool] = []
    waiter = threading.Thread(
        target=lambda: resumed.append(controller.wait_if_evicted("evicted-agent"))
    )
    waiter.start()
    deadline = time.time() + 2
    while controller.pending_counts()["evicted_ready"] == 0 and time.time() < deadline:
        time.sleep(0.01)

    controller.enqueue_fresh(AgentLaunchSpec("fresh-agent"))
    report = controller.on_tick(
        tick=1,
        vllm_info={"kv_free_gb": 5.0},
        agents={},
        bytes_per_blk=BYTES_PER_GB,
    )

    waiter.join(timeout=2)
    assert resumed == [True]
    assert report["admissions"][0] == {
        "agent_id": "evicted-agent",
        "previously_evicted": True,
        "admitted": True,
    }
    assert admitted == []
    assert controller.pending_counts()["fresh"] == 1
