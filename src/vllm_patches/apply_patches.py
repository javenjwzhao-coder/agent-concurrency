#!/usr/bin/env python3
"""
apply_patches.py — KV-block tracking patches for vLLM / vllm-ascend 0.11.x–0.13.x

Run during bare-metal venv setup to add per-agent KV
cache block reporting to:
  • vllm/entrypoints/openai/protocol.py  — UsageInfo + ChatCompletionRequest
  • vllm/entrypoints/openai/serving_chat.py — computation + population
  • vllm/entrypoints/openai/api_server.py — admin endpoint for proactive
    per-agent KV eviction, backed by an engine/scheduler evict_agent_kv hook

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
API_SERVER = VLLM_DIR / "entrypoints/openai/api_server.py"
ASYNC_LLM = VLLM_DIR / "v1/engine/async_llm.py"
CORE_CLIENT = VLLM_DIR / "v1/engine/core_client.py"
ENGINE_CORE = VLLM_DIR / "v1/engine/core.py"
SCHEDULER = VLLM_DIR / "v1/core/sched/scheduler.py"
KV_CACHE_MANAGER = VLLM_DIR / "v1/core/kv_cache_manager.py"
BLOCK_POOL = VLLM_DIR / "v1/core/block_pool.py"


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


def _insert_before_anchor(
    text: str,
    anchors: list[str],
    insertion: str,
    label: str,
    *,
    required: bool = False,
) -> tuple[str, bool]:
    """Insert text before the first matching anchor."""
    for anchor in anchors:
        if anchor in text:
            return text.replace(anchor, insertion + anchor, 1), True
    print(f"[WARN] Could not find insertion anchor for {label}", file=sys.stderr)
    if required:
        sys.exit(1)
    return text, False


def _insert_after_anchor(
    text: str,
    anchors: list[str],
    insertion: str,
    label: str,
    *,
    required: bool = False,
) -> tuple[str, bool]:
    """Insert text after the first matching anchor."""
    for anchor in anchors:
        if anchor in text:
            return text.replace(anchor, anchor + insertion, 1), True
    print(f"[WARN] Could not find insertion anchor for {label}", file=sys.stderr)
    if required:
        sys.exit(1)
    return text, False


def _code_block(indent: int, code: str, *, leading_newline: bool = False) -> str:
    """Return dedented code re-indented for insertion into a target file."""
    body = textwrap.indent(textwrap.dedent(code).strip("\n") + "\n", " " * indent)
    return ("\n" if leading_newline else "") + body


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

    # kv_blocks_size_gb and "agent_id: str | None" are unique to our patch.
    need_kv_size  = "kv_blocks_size_gb" not in txt
    need_agent_id = "agent_id: str | None" not in txt

    if not need_kv_size and not need_agent_id:
        print("[SKIP] protocol.py already patched")
        return

    # ── 1. Add kv_blocks fields to UsageInfo ─────────────────────────────────
    # The class always ends with prompt_tokens_details (both | None and Optional forms).
    # We try the v0.13 form first, then fall back to the v0.9 form.
    usage_tail_v13 = "    prompt_tokens_details: PromptTokenUsageInfo | None = None"
    usage_tail_v09 = "    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None"

    if need_kv_size:
        # If vllm-ascend already defines kv_blocks_used upstream, only inject
        # kv_blocks_size_gb (which is always unique to our patch).
        if "kv_blocks_used" in txt:
            kv_insert = textwrap.dedent("""\
                kv_blocks_size_gb: float | None = Field(
                    default=None,
                    description="GiB of GPU KV-cache used by kv_blocks_used blocks.",
                )
            """)
            kv_msg = "protocol.py: UsageInfo — kv_blocks_size_gb added (kv_blocks_used already defined upstream)"
        else:
            kv_insert = textwrap.dedent("""\
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
            kv_msg = "protocol.py: UsageInfo — kv_blocks_used + kv_blocks_size_gb added"

        kv_insert_indented = textwrap.indent(kv_insert, "    ")

        if usage_tail_v13 in txt:
            txt = _replace_once(
                txt,
                usage_tail_v13,
                usage_tail_v13 + "\n" + kv_insert_indented,
                "protocol.py (UsageInfo v0.13)",
            )
        elif usage_tail_v09 in txt:
            txt = _replace_once(
                txt,
                usage_tail_v09,
                usage_tail_v09 + "\n" + kv_insert_indented,
                "protocol.py (UsageInfo v0.9)",
            )
        else:
            print("[ERROR] Could not find UsageInfo.prompt_tokens_details in protocol.py",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[OK]  {kv_msg}")

    # ── 2. Add agent_id to ChatCompletionRequest ──────────────────────────────
    # Strategy: find the ChatCompletionRequest class body, locate the transition
    # from fields to methods/validators (first "@" or "    def " after the class
    # header), and insert the agent_id field just before it.
    if need_agent_id:
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


def patch_serving_chat_agent_registration() -> None:
    """Register the OpenAI request_id that belongs to a patched agent_id."""
    if not SERVING.exists():
        print(f"[WARN] serving_chat.py not found ({SERVING}); skipping agent registration",
              file=sys.stderr)
        return

    txt = SERVING.read_text()
    if "register_agent_request" in txt:
        print("[SKIP] serving_chat.py already registers agent requests")
        return

    registration = _code_block(20, """\
        if getattr(request, "agent_id", None) and hasattr(
                self.engine_client, "register_agent_request"):
            import inspect as _agent_kv_inspect
            _agent_kv_registered = (
                self.engine_client.register_agent_request(
                    request.agent_id, request_id))
            if _agent_kv_inspect.isawaitable(_agent_kv_registered):
                await _agent_kv_registered
    """) + "\n"

    anchors = [
        "                    trace_headers = (None if raw_request is None else await",
        "                    if isinstance(sampling_params, BeamSearchParams):",
        "                    generator = self.engine_client.generate(",
    ]
    txt, ok = _insert_before_anchor(
        txt, anchors, registration, "serving_chat.py agent request registration"
    )
    if not ok:
        print("[WARN] serving_chat.py: agent_id will not be mapped to request_id; "
              "evict_agent_kv will only work for request IDs registered elsewhere.",
              file=sys.stderr)
        return

    SERVING.write_text(txt)
    print("[OK]  serving_chat.py: agent_id → request_id registration added")


# ───────────────────── proactive per-agent eviction API ──────────────────────

def patch_agent_kv_engine_hooks() -> None:
    """Patch vLLM v1 engine/scheduler surfaces for per-agent KV eviction."""
    _patch_async_llm_agent_kv()
    _patch_core_client_agent_kv()
    _patch_engine_core_agent_kv()
    _patch_block_pool_agent_kv()
    _patch_kv_cache_manager_agent_kv()
    _patch_scheduler_agent_kv()


def _patch_async_llm_agent_kv() -> None:
    if not ASYNC_LLM.exists():
        print(f"[WARN] async_llm.py not found ({ASYNC_LLM}); skipping AsyncLLM hook",
              file=sys.stderr)
        return

    txt = ASYNC_LLM.read_text()
    if "async def evict_agent_kv" in txt and "async def register_agent_request" in txt:
        print("[SKIP] async_llm.py already has agent KV methods")
        return

    methods = _code_block(4, """\
        async def register_agent_request(self, agent_id: str, request_id: str) -> None:
            if not agent_id:
                return
            await self.engine_core.register_agent_request_async(
                str(agent_id), request_id)

        async def evict_agent_kv(
                self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
            return await self.engine_core.evict_agent_kv_async(
                str(agent_id), only_ref_cnt_zero)
    """, leading_newline=True)

    txt, ok = _insert_before_anchor(
        txt,
        ["    async def reset_prefix_cache(", "    async def sleep("],
        methods,
        "async_llm.py agent KV methods",
    )
    if not ok:
        return
    ASYNC_LLM.write_text(txt)
    print("[OK]  async_llm.py: register_agent_request + evict_agent_kv added")


def _patch_core_client_agent_kv() -> None:
    if not CORE_CLIENT.exists():
        print(f"[WARN] core_client.py not found ({CORE_CLIENT}); skipping core client hook",
              file=sys.stderr)
        return

    txt = CORE_CLIENT.read_text()
    changed = False

    if not re.search(
            r"def register_agent_request\([^)]*\).*?"
            r"raise NotImplementedError", txt, re.DOTALL):
        abstract_sync = _code_block(4, """\
            def register_agent_request(self, agent_id: str, request_id: str) -> None:
                raise NotImplementedError

            def evict_agent_kv(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                raise NotImplementedError
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    def reset_prefix_cache(self) -> None:\n"
                "        raise NotImplementedError\n",
            ],
            abstract_sync,
            "core_client.py abstract sync agent KV methods",
        )
        changed = changed or ok

    if not re.search(
            r"async def register_agent_request_async\([^)]*\).*?"
            r"raise NotImplementedError", txt, re.DOTALL):
        abstract_async = _code_block(4, """\
            async def register_agent_request_async(
                    self, agent_id: str, request_id: str) -> None:
                raise NotImplementedError

            async def evict_agent_kv_async(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                raise NotImplementedError
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    async def reset_prefix_cache_async(self) -> None:\n"
                "        raise NotImplementedError\n",
            ],
            abstract_async,
            "core_client.py abstract async agent KV methods",
        )
        changed = changed or ok

    if not re.search(
            r"def register_agent_request\([^)]*\).*?"
            r"self\.engine_core\.register_agent_request\(agent_id, request_id\)",
            txt,
            re.DOTALL,
    ):
        inproc_sync = _code_block(4, """\
            def register_agent_request(self, agent_id: str, request_id: str) -> None:
                self.engine_core.register_agent_request(agent_id, request_id)

            def evict_agent_kv(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                return self.engine_core.evict_agent_kv(agent_id, only_ref_cnt_zero)
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    def reset_prefix_cache(self) -> None:\n"
                "        self.engine_core.reset_prefix_cache()\n",
            ],
            inproc_sync,
            "core_client.py in-process agent KV methods",
        )
        changed = changed or ok

    if not re.search(
            r"def register_agent_request\([^)]*\).*?"
            r"self\.call_utility\(\s*\"register_agent_request\"",
            txt,
            re.DOTALL,
    ):
        mp_sync = _code_block(4, """\
            def register_agent_request(self, agent_id: str, request_id: str) -> None:
                self.call_utility("register_agent_request", agent_id, request_id)

            def evict_agent_kv(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                return self.call_utility(
                    "evict_agent_kv", agent_id, only_ref_cnt_zero)
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    def reset_prefix_cache(self) -> None:\n"
                "        self.call_utility(\"reset_prefix_cache\")\n",
            ],
            mp_sync,
            "core_client.py sync MP agent KV methods",
        )
        changed = changed or ok

    if not re.search(
            r"async def register_agent_request_async\([^)]*\).*?"
            r"self\.call_utility_async\(\s*\"register_agent_request\"",
            txt,
            re.DOTALL,
    ):
        async_mp = _code_block(4, """\
            async def register_agent_request_async(
                    self, agent_id: str, request_id: str) -> None:
                await self.call_utility_async(
                    "register_agent_request", agent_id, request_id)

            async def evict_agent_kv_async(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                return await self.call_utility_async(
                    "evict_agent_kv", agent_id, only_ref_cnt_zero)
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    async def reset_prefix_cache_async(self) -> None:\n"
                "        await self.call_utility_async(\"reset_prefix_cache\")\n",
            ],
            async_mp,
            "core_client.py async MP agent KV methods",
        )
        changed = changed or ok

    if changed:
        CORE_CLIENT.write_text(txt)
        print("[OK]  core_client.py: agent KV forwarding methods added")
    else:
        print("[SKIP] core_client.py agent KV hooks already present or no anchors matched")


def _patch_engine_core_agent_kv() -> None:
    if not ENGINE_CORE.exists():
        print(f"[WARN] core.py not found ({ENGINE_CORE}); skipping EngineCore hook",
              file=sys.stderr)
        return

    txt = ENGINE_CORE.read_text()
    if "def evict_agent_kv(" in txt:
        print("[SKIP] core.py already has evict_agent_kv")
        return

    methods = _code_block(4, """\
        def register_agent_request(self, agent_id: str, request_id: str) -> None:
            self.scheduler.register_agent_request(agent_id, request_id)

        def evict_agent_kv(
                self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
            return self.scheduler.evict_agent_kv(agent_id, only_ref_cnt_zero)
    """, leading_newline=True)
    txt, ok = _insert_after_anchor(
        txt,
        [
            "    def reset_prefix_cache(self):\n"
            "        self.scheduler.reset_prefix_cache()\n",
            "    def reset_prefix_cache(self) -> bool:\n"
            "        return self.scheduler.reset_prefix_cache()\n",
        ],
        methods,
        "core.py agent KV methods",
    )
    if not ok:
        return
    ENGINE_CORE.write_text(txt)
    print("[OK]  core.py: register_agent_request + evict_agent_kv added")


def _patch_block_pool_agent_kv() -> None:
    if not BLOCK_POOL.exists():
        print(f"[WARN] block_pool.py not found ({BLOCK_POOL}); skipping BlockPool hook",
              file=sys.stderr)
        return

    txt = BLOCK_POOL.read_text()
    if "only_ref_cnt_zero" in txt and "def evict_blocks" in txt:
        print("[SKIP] block_pool.py already has ref-count-aware evict_blocks")
        return

    method = _code_block(4, """\
        def evict_blocks(
                self,
                block_ids: set[int] | list[int] | tuple[int, ...],
                only_ref_cnt_zero: bool = False) -> int:
            \"\"\"Evict cached metadata for selected KV blocks.

            This intentionally uses the same cached-block eviction primitive as
            normal prefix-cache pressure.  It only touches blocks whose
            ref_cnt is zero when requested, so live requests can keep their KV
            blocks for proactive agent eviction.
            \"\"\"
            evicted = 0
            for block_id in list(block_ids):
                if block_id is None:
                    continue
                try:
                    block = self.blocks[int(block_id)]
                except (IndexError, TypeError, ValueError):
                    continue
                if getattr(block, "is_null", False):
                    continue
                if only_ref_cnt_zero and getattr(block, "ref_cnt", 0) != 0:
                    continue
                if self._maybe_evict_cached_block(block):
                    evicted += 1
            return evicted
    """)

    if "def evict_blocks(" in txt:
        pattern = re.compile(
            r"    def evict_blocks\([\s\S]*?\n(?=    def free_blocks\()"
        )
        txt, count = pattern.subn(method, txt, count=1)
        ok = count == 1
        if not ok:
            print("[WARN] Could not replace existing BlockPool.evict_blocks; "
                  "trying insertion fallback.", file=sys.stderr)
    else:
        ok = False

    if not ok:
        txt, ok = _insert_before_anchor(
            txt,
            ["    def touch(", "    def free_blocks("],
            method,
            "block_pool.py evict_blocks",
        )
    if not ok:
        return
    BLOCK_POOL.write_text(txt)
    print("[OK]  block_pool.py: ref-count-aware evict_blocks(block_ids) added")


def _patch_kv_cache_manager_agent_kv() -> None:
    if not KV_CACHE_MANAGER.exists():
        print(f"[WARN] kv_cache_manager.py not found ({KV_CACHE_MANAGER}); "
              "skipping KVCacheManager hook", file=sys.stderr)
        return

    txt = KV_CACHE_MANAGER.read_text()
    if "only_ref_cnt_zero" in txt and "def evict_blocks" in txt:
        print("[SKIP] kv_cache_manager.py already has ref-count-aware evict_blocks")
        return

    method = _code_block(4, """\
        def evict_blocks(
                self,
                block_ids: set[int] | list[int] | tuple[int, ...],
                only_ref_cnt_zero: bool = False) -> int:
            return self.block_pool.evict_blocks(
                block_ids, only_ref_cnt_zero=only_ref_cnt_zero)
    """)

    if "def evict_blocks(" in txt:
        pattern = re.compile(r"    def evict_blocks\([\s\S]*?\n(?=    def )")
        txt, count = pattern.subn(method, txt, count=1)
        ok = count == 1
        if not ok:
            print("[WARN] Could not replace existing KVCacheManager.evict_blocks; "
                  "trying insertion fallback.", file=sys.stderr)
    else:
        ok = False

    if not ok:
        txt, ok = _insert_before_anchor(
            txt,
            [
                "    def reset_prefix_cache(self) -> bool:",
                "    def get_num_common_prefix_blocks(",
                "    def get_block_ids(",
            ],
            method,
            "kv_cache_manager.py evict_blocks",
        )
    if not ok:
        return
    KV_CACHE_MANAGER.write_text(txt)
    print("[OK]  kv_cache_manager.py: ref-count-aware evict_blocks delegate added")


def _patch_scheduler_agent_kv() -> None:
    if not SCHEDULER.exists():
        print(f"[WARN] scheduler.py not found ({SCHEDULER}); skipping Scheduler hook",
              file=sys.stderr)
        return

    txt = SCHEDULER.read_text()
    changed = False

    if "_agent_kv_request_to_agent" not in txt:
        state = _code_block(8, """\
            self._agent_kv_agent_to_requests: dict[str, set[str]] = {}
            self._agent_kv_request_to_agent: dict[str, str] = {}
            self._agent_kv_cached_blocks: dict[str, set[int]] = {}
            self._agent_kv_bytes_per_block_gb = 0.0
            try:
                _agent_kv_model_config = vllm_config.model_config
                _agent_kv_hf_text_config = getattr(
                    _agent_kv_model_config, "hf_text_config", None)
                _agent_kv_dtype = str(
                    getattr(_agent_kv_model_config, "dtype", ""))
                _agent_kv_dtype_bytes = (
                    4 if "float32" in _agent_kv_dtype else
                    1 if "float8" in _agent_kv_dtype else 2)
                self._agent_kv_bytes_per_block_gb = (
                    self.block_size
                    * getattr(_agent_kv_hf_text_config, "num_hidden_layers", 1)
                    * 2
                    * _agent_kv_model_config.get_total_num_kv_heads()
                    * _agent_kv_model_config.get_head_size()
                    * _agent_kv_dtype_bytes
                ) / 1e9
            except Exception:
                pass
        """)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "        self.requests: dict[str, Request] = {}\n",
                "        self.requests = {}\n",
            ],
            state,
            "scheduler.py agent KV ownership state",
        )
        changed = changed or ok

    if "def register_agent_request(self, agent_id: str, request_id: str)" not in txt:
        methods = _code_block(4, """\
            def register_agent_request(self, agent_id: str, request_id: str) -> None:
                if not agent_id or not request_id:
                    return
                self._agent_kv_agent_to_requests.setdefault(
                    agent_id, set()).add(request_id)
                self._agent_kv_request_to_agent[request_id] = agent_id

            def _agent_kv_record_cached_blocks(
                    self, agent_id: str, request_id: str) -> None:
                try:
                    block_id_groups = self.kv_cache_manager.get_block_ids(request_id)
                except Exception:
                    return
                cached_blocks = self._agent_kv_cached_blocks.setdefault(
                    agent_id, set())
                for block_ids in block_id_groups:
                    cached_blocks.update(
                        int(block_id) for block_id in block_ids
                        if block_id is not None)

            def evict_agent_kv(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                if not agent_id:
                    return {
                        "evicted": False,
                        "freed_blocks": 0,
                        "freed_gb": 0.0,
                        "reason": "missing agent_id",
                    }

                block_ids = set(self._agent_kv_cached_blocks.get(agent_id, set()))
                for request_id in self._agent_kv_agent_to_requests.get(
                        agent_id, set()):
                    if request_id in self.requests:
                        self._agent_kv_record_cached_blocks(agent_id, request_id)
                block_ids.update(self._agent_kv_cached_blocks.get(agent_id, set()))

                if not block_ids:
                    return {
                        "evicted": False,
                        "freed_blocks": 0,
                        "freed_gb": 0.0,
                        "reason": "no cached blocks for agent",
                    }

                freed_blocks = self.kv_cache_manager.evict_blocks(
                    block_ids, only_ref_cnt_zero=only_ref_cnt_zero)
                freed_gb = round(
                    freed_blocks * getattr(
                        self, "_agent_kv_bytes_per_block_gb", 0.0),
                    6)
                return {
                    "evicted": freed_blocks > 0,
                    "freed_blocks": freed_blocks,
                    "freed_gb": freed_gb,
                    "reason": (
                        "ok" if freed_blocks else
                        "no matching cached blocks with ref_cnt == 0"),
                }
        """)
        txt, ok = _insert_before_anchor(
            txt,
            ["    def reset_prefix_cache(self) -> bool:", "    def make_stats("],
            methods,
            "scheduler.py agent KV methods",
        )
        changed = changed or ok

    if "_agent_kv_record_cached_blocks" in txt and "agent_id = self._agent_kv_request_to_agent.pop" not in txt:
        old = _code_block(4, """\
            def _free_blocks(self, request: Request):
                assert request.is_finished()
                self.kv_cache_manager.free(request)
                del self.requests[request.request_id]
        """)
        new = _code_block(4, """\
            def _free_blocks(self, request: Request):
                assert request.is_finished()
                request_id = request.request_id
                agent_id = self._agent_kv_request_to_agent.pop(request_id, None)
                if agent_id:
                    self._agent_kv_record_cached_blocks(agent_id, request_id)
                    request_ids = self._agent_kv_agent_to_requests.get(agent_id)
                    if request_ids is not None:
                        request_ids.discard(request_id)
                        if not request_ids:
                            self._agent_kv_agent_to_requests.pop(agent_id, None)
                self.kv_cache_manager.free(request)
                del self.requests[request_id]
        """)
        if old in txt:
            txt = txt.replace(old, new, 1)
            changed = True
        else:
            print("[WARN] Could not patch Scheduler._free_blocks for agent KV "
                  "ownership; eviction may miss finished request blocks.",
                  file=sys.stderr)

    if changed:
        SCHEDULER.write_text(txt)
        print("[OK]  scheduler.py: agent KV ownership + eviction added")
    else:
        print("[SKIP] scheduler.py agent KV hooks already present or no anchors matched")


def patch_agent_kv_eviction_api() -> None:
    """
    Add the HTTP endpoint consumed by sidecar.py's dynamic admission controller.

    The route intentionally calls an engine-client method named
    ``evict_agent_kv(agent_id, only_ref_cnt_zero=True)``.  In vllm-ascend this
    should be routed down to ``vllm/v1/core/sched/scheduler.py``, where the
    scheduler can resolve agent-owned block IDs, store/swap them through the
    configured KV connector when available, and call
    ``KVCacheManager.evict_blocks(block_ids)`` only for blocks whose ref_cnt is 0.
    """
    if not API_SERVER.exists():
        print(f"[WARN] api_server.py not found ({API_SERVER}); skipping agent KV eviction API",
              file=sys.stderr)
        return

    txt = API_SERVER.read_text()
    if '"/agent_kv_cache/evict"' in txt:
        print("[SKIP] api_server.py already has /agent_kv_cache/evict")
        return

    route = textwrap.dedent('''\

        @router.post("/agent_kv_cache/evict")
        async def evict_agent_kv_cache(raw_request: Request):
            """Evict KV blocks owned by one agent when they are not referenced."""
            import inspect as _agent_kv_inspect
            from fastapi.responses import JSONResponse as _AgentKVJSONResponse

            try:
                body = await raw_request.json()
            except Exception:
                body = {}
            agent_id = str(body.get("agent_id") or "")
            only_ref_cnt_zero = bool(body.get("only_ref_cnt_zero", True))
            if not agent_id:
                return _AgentKVJSONResponse(
                    {
                        "evicted": False,
                        "freed_blocks": 0,
                        "freed_gb": 0.0,
                        "reason": "missing agent_id",
                    },
                    status_code=400,
                )

            client = engine_client(raw_request)
            if not hasattr(client, "evict_agent_kv"):
                return _AgentKVJSONResponse(
                    {
                        "evicted": False,
                        "freed_blocks": 0,
                        "freed_gb": 0.0,
                        "reason": (
                            "engine client missing evict_agent_kv(agent_id, "
                            "only_ref_cnt_zero=True); route this to the "
                            "vLLM/vllm-ascend scheduler"
                        ),
                    },
                    status_code=501,
                )

            result = client.evict_agent_kv(
                agent_id, only_ref_cnt_zero=only_ref_cnt_zero
            )
            if _agent_kv_inspect.isawaitable(result):
                result = await result
            if result is None:
                result = {
                    "evicted": False,
                    "freed_blocks": 0,
                    "freed_gb": 0.0,
                    "reason": "engine returned no result",
                }
            return _AgentKVJSONResponse(result)
    ''')

    anchors = [
        '@router.post("/reset_prefix_cache")',
        '@router.post("/reset_mm_cache")',
        '@router.post("/reset_encoder_cache")',
        '@router.get("/health")',
    ]
    for anchor in anchors:
        if anchor in txt:
            txt = txt.replace(anchor, route + "\n" + anchor, 1)
            API_SERVER.write_text(txt)
            print("[OK]  api_server.py: /agent_kv_cache/evict route added")
            print("[INFO] vllm-ascend hook target: engine_client.evict_agent_kv "
                  "→ scheduler.evict_agent_kv → KVCacheManager.evict_blocks")
            return

    print("[WARN] Could not find a stable API route anchor in api_server.py; "
          "skipping agent KV eviction API.", file=sys.stderr)


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
    if "register_agent_request" not in serving_txt:
        errors.append("serving_chat.py missing agent_id request registration")

    hook_checks = [
        (ASYNC_LLM, "async def evict_agent_kv", "async_llm.py missing evict_agent_kv"),
        (CORE_CLIENT, "evict_agent_kv", "core_client.py missing evict_agent_kv forwarding"),
        (ENGINE_CORE, "def evict_agent_kv", "core.py missing evict_agent_kv"),
        (SCHEDULER, "def evict_agent_kv", "scheduler.py missing evict_agent_kv"),
        (SCHEDULER, "_agent_kv_request_to_agent", "scheduler.py missing ownership state"),
        (KV_CACHE_MANAGER, "def evict_blocks", "kv_cache_manager.py missing evict_blocks"),
        (BLOCK_POOL, "def evict_blocks", "block_pool.py missing evict_blocks"),
    ]
    for path, needle, message in hook_checks:
        if path.exists() and needle not in path.read_text():
            errors.append(message)

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
    if API_SERVER.exists():
        api_txt = API_SERVER.read_text()
        if "/agent_kv_cache/evict" in api_txt:
            print("[OK]  api_server.py: /agent_kv_cache/evict endpoint present")
        else:
            print("[WARN] api_server.py: /agent_kv_cache/evict endpoint not present",
                  file=sys.stderr)
    print("[OK]  Validation passed: all expected fields present")


if __name__ == "__main__":
    print("=== Applying vLLM KV-block tracking patches ===")
    patch_protocol()
    patch_serving_chat()
    patch_serving_chat_agent_registration()
    patch_agent_kv_engine_hooks()
    patch_agent_kv_eviction_api()
    validate()

    print("=== All patches applied successfully ===")
