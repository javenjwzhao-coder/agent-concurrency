import importlib.util
import sys
import types
from pathlib import Path


PATCH_TEXT = Path("src/vllm_patches/apply_patches.py").read_text()
CONNECTOR_TEXT = Path("src/vllm_patches/agent_offloading_connector.py").read_text()


def _install_fake_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load_connector_with_fake_vllm():
    for name in (
        "vllm",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
    ):
        _install_fake_module(name)

    offloading = _install_fake_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector"
    )
    base = _install_fake_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.base"
    )

    class OffloadingConnector:
        pass

    class OffloadingConnectorScheduler:
        pass

    class OffloadingConnectorWorker:
        def __init__(self, spec):
            self.spec = spec

    class OffloadingConnectorMetadata:
        def __init__(self, reqs_to_load=None, reqs_to_store=None):
            self.reqs_to_load = reqs_to_load or {}
            self.reqs_to_store = reqs_to_store or {}

    class GPULoadStoreSpec:
        def __init__(self, block_ids):
            self.block_ids = block_ids

    offloading.OffloadingConnector = OffloadingConnector
    offloading.OffloadingConnectorScheduler = OffloadingConnectorScheduler
    offloading.OffloadingConnectorWorker = OffloadingConnectorWorker
    offloading.OffloadingConnectorMetadata = OffloadingConnectorMetadata
    offloading.GPULoadStoreSpec = GPULoadStoreSpec
    offloading.logger = types.SimpleNamespace(debug=lambda *a, **k: None,
                                              warning=lambda *a, **k: None)
    offloading.yield_req_data = lambda scheduler_output: ()

    base.KVConnectorRole = types.SimpleNamespace(SCHEDULER="scheduler",
                                                WORKER="worker")

    spec = importlib.util.spec_from_file_location(
        "agent_offloading_connector_under_test",
        "src/vllm_patches/agent_offloading_connector.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vllm_patch_exposes_offload_restore_routes():
    assert '"/agent_kv_cache/offload"' in PATCH_TEXT
    assert '"/agent_kv_cache/usage"' in PATCH_TEXT
    assert '"/agent_kv_cache/restore"' in PATCH_TEXT
    assert '"/agent_kv_cache/release"' in PATCH_TEXT
    legacy_method = "evi" + "ct_agent_kv"
    assert legacy_method not in PATCH_TEXT
    assert legacy_method not in CONNECTOR_TEXT
    assert "offload_agent_kv" in PATCH_TEXT
    assert "restore_agent_kv" in PATCH_TEXT
    assert "release_agent_kv" in PATCH_TEXT
    assert "get_agent_kv_usage" in PATCH_TEXT
    assert "get_agent_kv_usage" in CONNECTOR_TEXT
    assert '"router = APIRouter()\\n"' in PATCH_TEXT
    assert "agent KV route is not module-level" in PATCH_TEXT
    assert "api_server.py syntax error" in PATCH_TEXT


def test_custom_connector_reuses_offloading_infrastructure():
    assert "class AgentAwareOffloadingConnector(OffloadingConnector)" in CONNECTOR_TEXT
    assert "class AgentAwareOffloadingWorker(OffloadingConnectorWorker)" in CONNECTOR_TEXT
    assert "GPULoadStoreSpec" in CONNECTOR_TEXT
    assert "_AGENT_STORE_PREFIX" in CONNECTOR_TEXT


def test_custom_connector_holds_real_request_for_synthetic_store_jobs():
    assert "_make_agent_store_job_id(agent_id, req_id" in CONNECTOR_TEXT
    assert "_parse_agent_store_job_id(req_id)" in CONNECTOR_TEXT
    assert "_agent_real_req_pending_jobs[req_id].add(job_id)" in CONNECTOR_TEXT
    assert "return request_being_stored or agent_store_pending or held, params" in CONNECTOR_TEXT
    assert "finished_sending.add(real_req_id)" in CONNECTOR_TEXT


def test_custom_connector_holds_and_releases_finished_agent_requests():
    assert "_held_agent_requests" in CONNECTOR_TEXT
    assert "_held_request_snapshots" in CONNECTOR_TEXT
    assert "held = self._hold_request(agent_id, req_id)" in CONNECTOR_TEXT
    assert "return request_being_stored or agent_store_pending or held, params" in CONNECTOR_TEXT
    assert "def release_agent_kv(" in CONNECTOR_TEXT
    assert "released_request_ids" in CONNECTOR_TEXT
    assert "_release_stale_holds" in CONNECTOR_TEXT
    assert 'if not snapshot.get("block_ids"):' in CONNECTOR_TEXT
    assert 'not snapshot.get("block_ids") or not snapshot.get("block_hashes")' not in CONNECTOR_TEXT


def test_custom_connector_reports_resident_and_offloadable_blocks_separately():
    assert "resident_kv_blocks" in CONNECTOR_TEXT
    assert "offloadable_kv_blocks" in CONNECTOR_TEXT
    assert "resident_blocks = len(block_ids)" in CONNECTOR_TEXT
    assert "offloadable_blocks = len(block_hashes)" in CONNECTOR_TEXT
    assert '"kv_blocks": total_resident_blocks' in CONNECTOR_TEXT
    assert '"resident_kv_blocks": resident_blocks' in CONNECTOR_TEXT
    assert '"reason": "no offloadable held KV blocks for agent"' in CONNECTOR_TEXT


def test_custom_connector_holds_resident_blocks_without_hashes():
    connector = _load_connector_with_fake_vllm()
    base = types.SimpleNamespace(_requests={})
    scheduler = connector.AgentAwareOffloadingScheduler(
        base, {"agent_hold_ttl_s": 0}
    )
    scheduler._agent_snapshots["agent"]["req"] = {
        "block_ids": [11, 12, 13],
        "block_hashes": [],
    }

    assert scheduler._hold_request("agent", "req") is True

    usage = scheduler.get_agent_kv_usage("agent")
    assert usage["resident_kv_blocks"] == 3
    assert usage["offloadable_kv_blocks"] == 0
    assert usage["held_requests"] == 1

    offload = scheduler.offload_agent_kv("agent")
    assert offload["offloaded"] is False
    assert offload["resident_kv_blocks"] == 3
    assert offload["known_blocks"] == 0
    assert offload["reason"] == "no offloadable held KV blocks for agent"


def test_vllm_patch_has_v013_engine_forwarding_anchors():
    assert "reset_running_requests: bool = False, reset_connector: bool = False" in PATCH_TEXT
    assert "return self.engine_core.reset_prefix_cache(" in PATCH_TEXT
    assert '\\"reset_prefix_cache\\", reset_running_requests, reset_connector' in PATCH_TEXT
    assert "return await self.call_utility_async(" in PATCH_TEXT
    assert "return self.scheduler.reset_prefix_cache(" in PATCH_TEXT
