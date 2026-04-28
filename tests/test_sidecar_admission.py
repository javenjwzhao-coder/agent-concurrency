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
    assert "saturation_guard" in report["reasons"]
    assert report["admissions"] == []
    assert admitted == []


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
