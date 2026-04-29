from pathlib import Path


PATCH_TEXT = Path("src/vllm_patches/apply_patches.py").read_text()


def test_vllm_patch_exposes_pin_offload_and_compat_routes():
    assert '"/agent_kv_cache/pin"' in PATCH_TEXT
    assert '"/agent_kv_cache/offload"' in PATCH_TEXT
    assert '"/agent_kv_cache/evict"' in PATCH_TEXT
    assert "pin_agent_kv" in PATCH_TEXT
    assert "offload_agent_kv" in PATCH_TEXT


def test_vllm_patch_skips_pinned_blocks_until_unpinned():
    assert "_agent_kv_pinned_blocks" in PATCH_TEXT
    assert "def pin_blocks" in PATCH_TEXT
    assert "block_ids.difference_update(pinned)" in PATCH_TEXT
    assert "if block_id_int in pinned:" in PATCH_TEXT
    assert "return await offload_agent_kv_cache(raw_request)" in PATCH_TEXT
