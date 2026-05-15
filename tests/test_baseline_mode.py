from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_runner_baseline_omits_patch_only_request_and_callback():
    src = _read("src/run_abc_bench_instrumented.py")

    assert "p.add_argument(\"--baseline-mode\"" in src
    assert "p.add_argument(\"--open-loop-launch\"" in src
    assert "if not baseline_mode:\n        extra_body[\"agent_id\"] = agent_id" in src
    assert "if not baseline_mode:\n        import litellm as _litellm" in src
    assert "ERROR: --baseline-mode cannot be combined with " in src
    assert "elif args.open_loop_launch:" in src


def test_wrapper_baseline_forces_open_loop_and_disables_optimizations():
    src = _read("run_abc-bench.sh")

    assert "--baseline" in src
    assert "PREDICTION_ENABLED=false" in src
    assert "SIDECAR_ENABLED=true" in src
    assert "SIDECAR_ADMISSION_ENABLED=false" in src
    assert "--baseline-mode" in src
    assert "--open-loop-launch" in src
    assert "cmd+=(--baseline)" in src
    assert "baseline_vllm_metrics.json" in src
    assert '"vllm_mode": "$(if [[ "$BASELINE" == "true" ]]' in src
    assert "__baseline_probe__" in src


def test_start_vllm_baseline_uses_clean_paths_and_stock_offloading_connector():
    src = _read("start_vllm.sh")

    assert "--baseline)" in src
    assert ".venv-baseline" in src
    assert ".vllm-baseline-src" in src
    assert ".vllm-ascend-baseline-src" in src
    assert "Baseline mode: skipping agent-aware vLLM patches." in src
    assert "Baseline vLLM unexpectedly exposes patched agent_id request field" in src
    assert "Baseline mode: skipping agent KV offload route probe." in src
    assert "KV_TRANSFER_CONFIG='{\"kv_connector\": \"OffloadingConnector\"" in src
    assert '"kv_role": "kv_both"' in src
    assert '"spec_name": "NPUOffloadingSpec"' in src
    assert '"spec_module_path": "vllm_ascend.kv_offload.npu"' in src
    baseline_assignment = (
        "KV_TRANSFER_CONFIG='{\"kv_connector\": \"OffloadingConnector\", "
        "\"kv_role\": \"kv_both\""
    )
    baseline_idx = src.index(baseline_assignment)
    optimized_idx = src.index(
        "KV_TRANSFER_CONFIG='{\"kv_connector\": \"AgentAwareOffloadingConnector\""
    )
    baseline_block = src[baseline_idx:optimized_idx]
    assert "kv_connector_module_path" not in baseline_block
    assert "agent_hold_finished_requests" not in baseline_block


def test_baseline_sidecar_and_dashboard_keep_global_telemetry_only():
    sidecar = _read("src/sidecar.py")
    dashboard = _read("dashboard/dashboard.js")
    html = _read("dashboard/index.html")

    assert '"mode": "baseline" if baseline_mode else "telemetry_only"' in sidecar
    assert '"reasons": ["baseline"] if baseline_mode else ["not_configured"]' in sidecar
    assert "scheduler_preemptions_total" in dashboard
    assert "admissionEnabled: adm.enabled === true" in dashboard
    assert "setAdmissionVisualsVisible(state.admissionEnabled)" in dashboard
    assert "hideBadge(\"pressureBadge\")" in dashboard
    assert "eventLegendTitle" in html
    assert "kvThresholdLegendChip" in html
