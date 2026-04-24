#!/usr/bin/env python3
"""
apply_patches.py — KV-block tracking patches for vLLM v0.13.x

Run during bare-metal venv setup to add per-agent KV
cache block reporting to:
  • vllm/entrypoints/openai/protocol.py  — UsageInfo + ChatCompletionRequest
  • vllm/entrypoints/openai/serving_chat.py — computation + population

Pass --vllm-dir to target a specific vllm package directory (e.g. a
project-local venv's site-packages/vllm). Defaults to the shared venv path.
"""

import argparse
import pathlib
import re
import sys
import textwrap

_parser = argparse.ArgumentParser(description="Apply KV-block tracking patches to vLLM")
_parser.add_argument(
    "--vllm-dir",
    default=f"/opt/vllm/venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/vllm",
    help="Path to the vllm package directory to patch",
)
_args = _parser.parse_args()

VLLM_DIR = pathlib.Path(_args.vllm_dir)
PROTO   = VLLM_DIR / "entrypoints/openai/protocol.py"
SERVING = VLLM_DIR / "entrypoints/openai/serving_chat.py"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        print(f"[ERROR] Pattern not found in {label}:", file=sys.stderr)
        print(f"        {old!r}", file=sys.stderr)
        sys.exit(1)
    count = text.count(old)
    if count > 1:
        print(f"[WARN]  Pattern found {count} times in {label} — replacing first occurrence only")
    return text.replace(old, new, 1)


def _try_replace_candidates(
    text: str, candidates: list[tuple[str, str]], label: str
) -> tuple[str, bool]:
    """Try each (old, new) pair in order. Return (modified_text, True) on first match."""
    for old, new in candidates:
        if old in text:
            count = text.count(old)
            if count > 1:
                print(f"[WARN]  Pattern found {count} times in {label} — replacing first occurrence only")
            return text.replace(old, new, 1), True
    return text, False


def _patch_via_regex_usage_block(
    text: str,
    usage_var: str,
    kv_injection: str,
    label: str,
) -> tuple[str, bool]:
    """
    Regex fallback: find 'usage_var.prompt_tokens_details = PromptTokenUsageInfo(...)'
    (with optional preceding if-guard) and append kv_injection after it.
    Handles any variable name for num_cached_tokens and any indentation depth.
    """
    pattern = re.compile(
        r'([ \t]+(?:if\s+[^\n]+\n[ \t]+)?'
        + re.escape(usage_var)
        + r'\.prompt_tokens_details\s*=\s*PromptTokenUsageInfo\(\s*\n'
        r'[ \t]+cached_tokens=[^\n]+'
        r'(?:\n[ \t]+\))?)',          # closing ) may be inline or on its own line
        re.MULTILINE,
    )
    m = pattern.search(text)
    if m:
        old = m.group(1)
        new = old + "\n" + kv_injection
        print(f"[OK]  serving_chat.py: {label} — kv fields populated (regex anchor)")
        return text.replace(old, new, 1), True

    # Last resort: report what PromptTokenUsageInfo occurrences exist so the user
    # can add an explicit candidate anchor.
    occurrences = [
        (i, text[max(0, i - 120): i + 120])
        for i in [m2.start() for m2 in re.finditer(r'PromptTokenUsageInfo', text)]
    ]
    print(f"[WARN] Could not auto-detect '{usage_var}.prompt_tokens_details' block for {label}.",
          file=sys.stderr)
    if occurrences:
        print(f"       Found {len(occurrences)} PromptTokenUsageInfo occurrence(s):", file=sys.stderr)
        for _, ctx in occurrences[:3]:
            print(f"       ---\n{ctx}\n       ---", file=sys.stderr)
    else:
        print("       No PromptTokenUsageInfo occurrences found — file may lack cached-token tracking.",
              file=sys.stderr)
    print(f"[WARN] Skipping {label} patch. KV blocks will not be reported for this path.",
          file=sys.stderr)
    return text, False


# ─────────────────────────── protocol.py ─────────────────────────────────────

def patch_protocol() -> None:
    txt = PROTO.read_text()
    if "kv_blocks_used" in txt and "agent_id" in txt:
        print("[SKIP] protocol.py already patched")
        return

    # ── 1. Add kv_blocks fields to UsageInfo ─────────────────────────────────
    # The class always ends with prompt_tokens_details (both | None and Optional forms).
    # We try the v0.13 form first, then fall back to the v0.9 form.
    usage_tail_v13  = "    prompt_tokens_details: PromptTokenUsageInfo | None = None"
    usage_tail_v09  = "    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None"

    kv_fields = textwrap.dedent("""\
        kv_blocks_used: int | None = Field(
            default=None,
            description=(
                "GPU KV-cache blocks consumed by this request: "
                "ceil(total_tokens / block_size). "
                "Only populated when agent_id is present in the request."
            ),
        )
        kv_blocks_size_gb: float | None = Field(
            default=None,
            description="GiB of GPU KV-cache used by kv_blocks_used blocks.",
        )
    """)
    # Indent to match class body (4 spaces)
    kv_fields_indented = textwrap.indent(kv_fields, "    ")

    if usage_tail_v13 in txt:
        txt = _replace_once(
            txt,
            usage_tail_v13,
            usage_tail_v13 + "\n" + kv_fields_indented,
            "protocol.py (UsageInfo v0.13)",
        )
    elif usage_tail_v09 in txt:
        txt = _replace_once(
            txt,
            usage_tail_v09,
            usage_tail_v09 + "\n" + kv_fields_indented,
            "protocol.py (UsageInfo v0.9)",
        )
    else:
        print("[ERROR] Could not find UsageInfo.prompt_tokens_details in protocol.py",
              file=sys.stderr)
        sys.exit(1)
    print("[OK]  protocol.py: UsageInfo — kv_blocks_used + kv_blocks_size_gb added")

    # ── 2. Add agent_id to ChatCompletionRequest ──────────────────────────────
    # Strategy: find the ChatCompletionRequest class body, locate the transition
    # from fields to methods/validators (first "@" or "    def " after the class
    # header), and insert the agent_id field just before it.
    agent_id_field = textwrap.dedent("""\
        agent_id: str | None = Field(
            default=None,
            description=(
                "Caller-supplied agent identifier. When provided, the server "
                "returns kv_blocks_used and kv_blocks_size_gb in usage so the "
                "caller can track per-agent GPU KV-cache consumption."
            ),
        )
    """)
    agent_id_field_indented = textwrap.indent(agent_id_field, "    ")

    # Find the ChatCompletionRequest class
    class_match = re.search(r'\nclass ChatCompletionRequest\b', txt)
    if not class_match:
        print("[ERROR] class ChatCompletionRequest not found in protocol.py", file=sys.stderr)
        sys.exit(1)

    class_start = class_match.end()
    rest = txt[class_start:]

    # Find first method/validator within the class (4-space indent + @/def)
    method_match = re.search(r'\n    (?:@|def )', rest)
    if not method_match:
        print("[ERROR] No method/validator found in ChatCompletionRequest", file=sys.stderr)
        sys.exit(1)

    insert_pos = class_start + method_match.start()
    txt = txt[:insert_pos] + "\n" + agent_id_field_indented + txt[insert_pos:]
    print("[OK]  protocol.py: ChatCompletionRequest — agent_id field added")

    # Guard: don't double-patch
    if txt.count("kv_blocks_used") > 2:
        print("[WARN]  kv_blocks_used appears >2 times — possible double-patch")

    PROTO.write_text(txt)
    print(f"[OK]  protocol.py written ({PROTO})")


# ─────────────────────────── serving_chat.py ─────────────────────────────────

def patch_serving_chat() -> None:
    txt = SERVING.read_text()
    if "kv_blocks_used" in txt:
        print("[SKIP] serving_chat.py already patched")
        return

    # ── 1. Precompute KV block geometry in __init__ ───────────────────────────
    # Insert after `self.enable_force_include_usage = enable_force_include_usage`
    init_anchor = "        self.enable_force_include_usage = enable_force_include_usage"
    kv_init_code = """
        # KV-block tracking: precompute block geometry once at startup.
        # These are used in _compute_kv_blocks() to answer per-request queries.
        import torch as _torch
        _cache_cfg = self.engine_client.vllm_config.cache_config
        _model_cfg = self.model_config
        self._kv_block_size: int = (_cache_cfg.block_size or 128)
        _dtype_bytes: int = {
            _torch.float32: 4, _torch.float16: 2,
            _torch.bfloat16: 2, _torch.float8_e4m3fn: 1, _torch.float8_e5m2: 1,
        }.get(_model_cfg.dtype, 2)
        self._kv_bytes_per_block_gb: float = (
            self._kv_block_size
            * getattr(_model_cfg.hf_text_config, "num_hidden_layers", 1)
            * 2  # K + V
            * _model_cfg.get_total_num_kv_heads()
            * _model_cfg.get_head_size()
            * _dtype_bytes
        ) / (1024 ** 3)"""

    txt = _replace_once(
        txt,
        init_anchor,
        init_anchor + kv_init_code,
        "serving_chat.py (__init__ anchor)",
    )
    print("[OK]  serving_chat.py: __init__ — KV geometry precomputed")

    # ── 2. Add _compute_kv_blocks helper method ───────────────────────────────
    # Insert before `async def create_chat_completion(` (main public method).
    public_method = "    async def create_chat_completion("
    helper_method = '''\
    def _compute_kv_blocks(self, total_tokens: int) -> tuple[int, float]:
        """Return (kv_blocks_used, kv_blocks_size_gb) for a request.

        Uses ceil(total_tokens / block_size) — exact because each sequence
        occupies complete, non-shared blocks.
        """
        import math
        blocks = math.ceil(total_tokens / self._kv_block_size)
        return blocks, round(blocks * self._kv_bytes_per_block_gb, 6)

'''
    txt = _replace_once(
        txt,
        public_method,
        helper_method + public_method,
        "serving_chat.py (create_chat_completion anchor)",
    )
    print("[OK]  serving_chat.py: _compute_kv_blocks method added")

    # ── 3. Non-streaming path: populate kv fields after UsageInfo() ───────────
    # Try candidate anchors across vLLM versions / vllm_ascend variants, then
    # fall back to regex auto-detection on the usage.prompt_tokens_details block.
    nonstream_kv = """\
        if getattr(request, "agent_id", None):
            _kv_blocks, _kv_size_gb = self._compute_kv_blocks(
                num_prompt_tokens + num_generated_tokens
            )
            usage.kv_blocks_used = _kv_blocks
            usage.kv_blocks_size_gb = _kv_size_gb"""

    def _ns_candidate(cached_expr: str, guard: str = "if self.enable_prompt_tokens_details and ",
                      inline_paren: bool = False) -> str:
        close = f")" if inline_paren else f"\n            )"
        return (
            f"        {guard}{cached_expr}:\n"
            f"            usage.prompt_tokens_details = PromptTokenUsageInfo(\n"
            f"                cached_tokens={cached_expr}{close}"
        )

    nonstream_candidates = [
        # vllm_ascend 0.11 — inline closing paren (confirmed from diagnostics)
        (_ns_candidate("final_res.num_cached_tokens", inline_paren=True),
         _ns_candidate("final_res.num_cached_tokens", inline_paren=True) + "\n" + nonstream_kv),
        (_ns_candidate("num_cached_tokens", inline_paren=True),
         _ns_candidate("num_cached_tokens", inline_paren=True) + "\n" + nonstream_kv),
        (_ns_candidate("final_output.num_cached_tokens", inline_paren=True),
         _ns_candidate("final_output.num_cached_tokens", inline_paren=True) + "\n" + nonstream_kv),
        # Without guard, inline paren
        (_ns_candidate("final_res.num_cached_tokens", "if ", inline_paren=True),
         _ns_candidate("final_res.num_cached_tokens", "if ", inline_paren=True) + "\n" + nonstream_kv),
        # v0.13.x / upstream — closing paren on its own line
        (_ns_candidate("final_res.num_cached_tokens"),
         _ns_candidate("final_res.num_cached_tokens") + "\n" + nonstream_kv),
        (_ns_candidate("final_output.num_cached_tokens"),
         _ns_candidate("final_output.num_cached_tokens") + "\n" + nonstream_kv),
        (_ns_candidate("num_cached_tokens"),
         _ns_candidate("num_cached_tokens") + "\n" + nonstream_kv),
        (_ns_candidate("final_res.num_cached_tokens", "if "),
         _ns_candidate("final_res.num_cached_tokens", "if ") + "\n" + nonstream_kv),
        (_ns_candidate("num_cached_tokens", "if "),
         _ns_candidate("num_cached_tokens", "if ") + "\n" + nonstream_kv),
    ]

    txt, ok = _try_replace_candidates(txt, nonstream_candidates, "serving_chat.py (non-streaming)")
    if ok:
        print("[OK]  serving_chat.py: non-streaming path — kv fields populated")
    else:
        txt, ok = _patch_via_regex_usage_block(txt, "usage", nonstream_kv, "non-streaming path")

    # ── 4. Streaming path: populate kv fields after final_usage construction ──
    stream_kv = """\
                if getattr(request, "agent_id", None):
                    _kv_blocks, _kv_size_gb = self._compute_kv_blocks(
                        num_prompt_tokens + completion_tokens
                    )
                    final_usage.kv_blocks_used = _kv_blocks
                    final_usage.kv_blocks_size_gb = _kv_size_gb"""

    def _st_candidate(cached_expr: str, guard: str = "if self.enable_prompt_tokens_details and ",
                      inline_paren: bool = False) -> str:
        close = ")" if inline_paren else "\n                    )"
        return (
            f"                {guard}{cached_expr}:\n"
            f"                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(\n"
            f"                        cached_tokens={cached_expr}{close}"
        )

    stream_candidates = [
        # vllm_ascend 0.11 — inline closing paren (confirmed from diagnostics)
        (_st_candidate("num_cached_tokens", inline_paren=True),
         _st_candidate("num_cached_tokens", inline_paren=True) + "\n" + stream_kv),
        (_st_candidate("final_res.num_cached_tokens", inline_paren=True),
         _st_candidate("final_res.num_cached_tokens", inline_paren=True) + "\n" + stream_kv),
        (_st_candidate("final_output.num_cached_tokens", inline_paren=True),
         _st_candidate("final_output.num_cached_tokens", inline_paren=True) + "\n" + stream_kv),
        # Without guard, inline paren
        (_st_candidate("num_cached_tokens", "if ", inline_paren=True),
         _st_candidate("num_cached_tokens", "if ", inline_paren=True) + "\n" + stream_kv),
        # v0.13.x / upstream — closing paren on its own line
        (_st_candidate("num_cached_tokens"),
         _st_candidate("num_cached_tokens") + "\n" + stream_kv),
        (_st_candidate("final_res.num_cached_tokens"),
         _st_candidate("final_res.num_cached_tokens") + "\n" + stream_kv),
        (_st_candidate("final_output.num_cached_tokens"),
         _st_candidate("final_output.num_cached_tokens") + "\n" + stream_kv),
        (_st_candidate("num_cached_tokens", "if "),
         _st_candidate("num_cached_tokens", "if ") + "\n" + stream_kv),
    ]

    txt, ok = _try_replace_candidates(txt, stream_candidates, "serving_chat.py (streaming)")
    if ok:
        print("[OK]  serving_chat.py: streaming path — kv fields populated")
    else:
        txt, ok = _patch_via_regex_usage_block(txt, "final_usage", stream_kv, "streaming path")

    SERVING.write_text(txt)
    print(f"[OK]  serving_chat.py written ({SERVING})")


# ─────────────────────────── validation ──────────────────────────────────────

def validate() -> None:
    import importlib.util, types

    # Only exec-validate protocol.py; serving_chat.py imports third-party packages
    # (e.g. `regex`, `openai_harmony`) that may not be available at patch time.
    try:
        exec(compile(PROTO.read_text(), str(PROTO), "exec"), types.ModuleType("proto").__dict__)
    except SyntaxError as e:
        print(f"[ERROR] Syntax error in protocol.py: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        pass  # ImportError is expected — we only care about syntax here

    # Text-level checks: confirm KV fields landed in both patched files
    proto_txt = PROTO.read_text()
    serving_txt = SERVING.read_text()
    errors = []
    for field in ("kv_blocks_used", "kv_blocks_size_gb"):
        if field not in proto_txt:
            errors.append(f"protocol.py missing field: {field}")
    if "agent_id" not in proto_txt:
        errors.append("protocol.py missing field: agent_id")
    if "_compute_kv_blocks" not in serving_txt:
        errors.append("serving_chat.py missing _compute_kv_blocks method")
    if "_kv_block_size" not in serving_txt:
        errors.append("serving_chat.py missing KV geometry init block")

    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    kv_injected = serving_txt.count("kv_blocks_used")
    if kv_injected == 0:
        print("[WARN] serving_chat.py: kv_blocks_used not injected into any response path "
              "— non-streaming and streaming patches both skipped.", file=sys.stderr)
    else:
        print(f"[OK]  serving_chat.py: kv_blocks_used injected into {kv_injected} response path(s)")
    print("[OK]  Validation passed: all expected fields present")


if __name__ == "__main__":
    print("=== Applying vLLM KV-block tracking patches ===")
    patch_protocol()
    patch_serving_chat()
    validate()
    print("=== All patches applied successfully ===")
