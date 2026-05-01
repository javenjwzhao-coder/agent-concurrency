from pathlib import Path


PATCH_TEXT = Path("src/vllm_patches/apply_patches.py").read_text()
CONNECTOR_TEXT = Path("src/vllm_patches/agent_offloading_connector.py").read_text()


def test_vllm_patch_exposes_offload_restore_routes():
    assert '"/agent_kv_cache/offload"' in PATCH_TEXT
    assert '"/agent_kv_cache/restore"' in PATCH_TEXT
    legacy_method = "evi" + "ct_agent_kv"
    assert legacy_method not in PATCH_TEXT
    assert legacy_method not in CONNECTOR_TEXT
    assert "offload_agent_kv" in PATCH_TEXT
    assert "restore_agent_kv" in PATCH_TEXT
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
    assert "return request_being_stored or agent_store_pending, params" in CONNECTOR_TEXT
    assert "finished_sending.add(real_req_id)" in CONNECTOR_TEXT


def test_vllm_patch_has_v013_engine_forwarding_anchors():
    assert "reset_running_requests: bool = False, reset_connector: bool = False" in PATCH_TEXT
    assert "return self.engine_core.reset_prefix_cache(" in PATCH_TEXT
    assert '\\"reset_prefix_cache\\", reset_running_requests, reset_connector' in PATCH_TEXT
    assert "return await self.call_utility_async(" in PATCH_TEXT
    assert "return self.scheduler.reset_prefix_cache(" in PATCH_TEXT
