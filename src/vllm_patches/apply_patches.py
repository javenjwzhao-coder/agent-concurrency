#!/usr/bin/env python3
"""
apply_patches.py — agent-aware KV offloading patches for vLLM/vllm-ascend 0.13.

The runtime now uses a custom KV connector instead of direct block-pool
pin/offload hooks.  The connector reuses vLLM's OffloadingConnector transfer
machinery and changes only policy: the sidecar can mark a long idle agent for
CPU offload, and the connector stores/restores that agent's KV blocks through
the configured OffloadingSpec.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys
import textwrap


_parser = argparse.ArgumentParser(description="Apply agent-aware KV patches to vLLM")
_parser.add_argument(
    "--vllm-dir",
    default=f"/opt/vllm/venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/vllm",
    help="Path to the vllm package directory to patch",
)
_args = _parser.parse_args()

VLLM_DIR = pathlib.Path(_args.vllm_dir)
PROTO = VLLM_DIR / "entrypoints/openai/protocol.py"
SERVING = VLLM_DIR / "entrypoints/openai/serving_chat.py"
API_SERVER = VLLM_DIR / "entrypoints/openai/api_server.py"
ASYNC_LLM = VLLM_DIR / "v1/engine/async_llm.py"
CORE_CLIENT = VLLM_DIR / "v1/engine/core_client.py"
ENGINE_CORE = VLLM_DIR / "v1/engine/core.py"
SCHEDULER = VLLM_DIR / "v1/core/sched/scheduler.py"
CONNECTOR_DST = (
    VLLM_DIR
    / "distributed/kv_transfer/kv_connector/v1/agent_offloading_connector.py"
)
CONNECTOR_SRC = pathlib.Path(__file__).with_name("agent_offloading_connector.py")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        print(f"[ERROR] Pattern not found in {label}: {old!r}", file=sys.stderr)
        sys.exit(1)
    if text.count(old) > 1:
        print(f"[WARN] Pattern found more than once in {label}; replacing first")
    return text.replace(old, new, 1)


def _try_replace_candidates(
    text: str, candidates: list[tuple[str, str]], label: str
) -> tuple[str, bool]:
    for old, new in candidates:
        if old in text:
            if text.count(old) > 1:
                print(f"[WARN] Pattern found more than once in {label}; replacing first")
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
    for anchor in anchors:
        if anchor in text:
            return text.replace(anchor, insertion + anchor, 1), True
    msg = f"[WARN] Could not find insertion anchor for {label}"
    print(msg, file=sys.stderr)
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
    for anchor in anchors:
        if anchor in text:
            return text.replace(anchor, anchor + insertion, 1), True
    msg = f"[WARN] Could not find insertion anchor for {label}"
    print(msg, file=sys.stderr)
    if required:
        sys.exit(1)
    return text, False


def _code_block(indent: int, code: str, *, leading_newline: bool = False) -> str:
    body = textwrap.indent(textwrap.dedent(code).strip("\n") + "\n", " " * indent)
    return ("\n" if leading_newline else "") + body


def _strip_function_block(text: str, name: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith(f"def {name}") or stripped.startswith(f"async def {name}"):
            base_indent = len(line) - len(stripped)
            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.lstrip()
                if next_stripped and len(next_line) - len(next_stripped) <= base_indent:
                    break
                i += 1
            changed = True
            continue
        out.append(line)
        i += 1
    return "".join(out), changed


def _strip_old_offload_aliases(text: str) -> tuple[str, bool]:
    changed = False
    for name in ("evi" + "ct_agent_kv", "evi" + "ct_agent_kv_async"):
        text, removed = _strip_function_block(text, name)
        changed = changed or removed
    return text, changed


def _patch_via_regex_usage_block(
    text: str,
    usage_var: str,
    kv_injection: str,
    label: str,
) -> tuple[str, bool]:
    pattern = re.compile(
        r"([ \t]+(?:if\s+[^\n]+\n[ \t]+)?"
        + re.escape(usage_var)
        + r"\.prompt_tokens_details\s*=\s*PromptTokenUsageInfo\(\s*\n"
        r"[ \t]+cached_tokens=[^\n]+"
        r"(?:\n[ \t]+\))?)",
        re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        print(f"[WARN] Could not auto-detect {label}; skipping", file=sys.stderr)
        return text, False
    old = match.group(1)
    return text.replace(old, old + "\n" + kv_injection, 1), True


def patch_protocol() -> None:
    txt = PROTO.read_text()
    need_kv_size = "kv_blocks_size_gb" not in txt
    need_agent_id = "agent_id: str | None" not in txt
    if not need_kv_size and not need_agent_id:
        print("[SKIP] protocol.py already patched")
        return

    if need_kv_size:
        if "kv_blocks_used" in txt:
            kv_insert = """\
kv_blocks_size_gb: float | None = Field(
    default=None,
    description="GiB of accelerator KV-cache used by kv_blocks_used blocks.",
)
"""
        else:
            kv_insert = """\
kv_blocks_used: int | None = Field(
    default=None,
    description=(
        "Accelerator KV-cache blocks consumed by this request: "
        "ceil(total_tokens / block_size)."
    ),
)
kv_blocks_size_gb: float | None = Field(
    default=None,
    description="GiB of accelerator KV-cache used by kv_blocks_used blocks.",
)
"""
        kv_insert = textwrap.indent(textwrap.dedent(kv_insert), "    ")
        anchors = [
            "    prompt_tokens_details: PromptTokenUsageInfo | None = None",
            "    prompt_tokens_details: Optional[PromptTokenUsageInfo] = None",
        ]
        for anchor in anchors:
            if anchor in txt:
                txt = txt.replace(anchor, anchor + "\n" + kv_insert, 1)
                print("[OK] protocol.py: UsageInfo KV fields added")
                break
        else:
            print("[ERROR] Could not find UsageInfo.prompt_tokens_details", file=sys.stderr)
            sys.exit(1)

    if need_agent_id:
        agent_id_field = textwrap.indent(textwrap.dedent("""\
            agent_id: str | None = Field(
                default=None,
                description=(
                    "Caller-supplied agent identifier used by the custom KV "
                    "offloading connector to group request-owned blocks."
                ),
            )
        """), "    ")
        class_match = re.search(r"\nclass ChatCompletionRequest\b", txt)
        if not class_match:
            print("[ERROR] class ChatCompletionRequest not found", file=sys.stderr)
            sys.exit(1)
        class_start = class_match.end()
        rest = txt[class_start:]
        method_match = re.search(r"\n    (?:@|def )", rest)
        if not method_match:
            print("[ERROR] No method/validator found in ChatCompletionRequest", file=sys.stderr)
            sys.exit(1)
        insert_pos = class_start + method_match.start()
        txt = txt[:insert_pos] + "\n" + agent_id_field + txt[insert_pos:]
        print("[OK] protocol.py: ChatCompletionRequest.agent_id added")

    PROTO.write_text(txt)


def patch_serving_chat() -> None:
    txt = SERVING.read_text()
    if "kv_blocks_size_gb" in txt and "_compute_kv_blocks" in txt:
        print("[SKIP] serving_chat.py KV telemetry already patched")
    else:
        init_anchor = "        self.enable_force_include_usage = enable_force_include_usage"
        kv_init_code = """
        # Agent KV telemetry: precompute block geometry once at startup.
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
            * 2
            * _model_cfg.get_total_num_kv_heads()
            * _model_cfg.get_head_size()
            * _dtype_bytes
        ) / (1024 ** 3)"""
        txt = _replace_once(
            txt,
            init_anchor,
            init_anchor + kv_init_code,
            "serving_chat.py __init__",
        )

        public_method = "    async def create_chat_completion("
        helper_method = """\
    def _compute_kv_blocks(self, total_tokens: int) -> tuple[int, float]:
        import math
        blocks = math.ceil(total_tokens / self._kv_block_size)
        return blocks, round(blocks * self._kv_bytes_per_block_gb, 6)

"""
        txt = _replace_once(
            txt,
            public_method,
            helper_method + public_method,
            "serving_chat.py create_chat_completion",
        )

        nonstream_kv = """\
        if getattr(request, "agent_id", None):
            _kv_blocks, _kv_size_gb = self._compute_kv_blocks(
                num_prompt_tokens + num_generated_tokens
            )
            usage.kv_blocks_used = _kv_blocks
            usage.kv_blocks_size_gb = _kv_size_gb"""

        def _ns_candidate(cached_expr: str, guard: str = "if self.enable_prompt_tokens_details and ",
                          inline_paren: bool = False) -> str:
            close = ")" if inline_paren else "\n            )"
            return (
                f"        {guard}{cached_expr}:\n"
                f"            usage.prompt_tokens_details = PromptTokenUsageInfo(\n"
                f"                cached_tokens={cached_expr}{close}"
            )

        nonstream_candidates = []
        for expr in ("final_res.num_cached_tokens", "final_output.num_cached_tokens", "num_cached_tokens"):
            for inline in (True, False):
                old = _ns_candidate(expr, inline_paren=inline)
                nonstream_candidates.append((old, old + "\n" + nonstream_kv))
                old = _ns_candidate(expr, "if ", inline_paren=inline)
                nonstream_candidates.append((old, old + "\n" + nonstream_kv))
        txt, ok = _try_replace_candidates(
            txt, nonstream_candidates, "serving_chat.py non-streaming usage")
        if not ok:
            txt, _ = _patch_via_regex_usage_block(
                txt, "usage", nonstream_kv, "non-streaming usage")

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

        stream_candidates = []
        for expr in ("num_cached_tokens", "final_res.num_cached_tokens", "final_output.num_cached_tokens"):
            for inline in (True, False):
                old = _st_candidate(expr, inline_paren=inline)
                stream_candidates.append((old, old + "\n" + stream_kv))
                old = _st_candidate(expr, "if ", inline_paren=inline)
                stream_candidates.append((old, old + "\n" + stream_kv))
        txt, ok = _try_replace_candidates(
            txt, stream_candidates, "serving_chat.py streaming usage")
        if not ok:
            txt, _ = _patch_via_regex_usage_block(
                txt, "final_usage", stream_kv, "streaming usage")
        print("[OK] serving_chat.py: KV telemetry patched")

    if "register_agent_request" not in txt:
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
        txt, ok = _insert_before_anchor(
            txt,
            [
                "                    trace_headers = (None if raw_request is None else await",
                "                    if isinstance(sampling_params, BeamSearchParams):",
                "                    generator = self.engine_client.generate(",
            ],
            registration,
            "serving_chat.py agent registration",
        )
        if ok:
            print("[OK] serving_chat.py: agent_id -> request_id registration added")
    SERVING.write_text(txt)


def install_agent_connector() -> None:
    if not CONNECTOR_SRC.exists():
        print(f"[ERROR] Missing connector template: {CONNECTOR_SRC}", file=sys.stderr)
        sys.exit(1)
    CONNECTOR_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONNECTOR_SRC, CONNECTOR_DST)
    print(f"[OK] agent-aware connector installed: {CONNECTOR_DST}")


def patch_engine_hooks() -> None:
    _patch_async_llm()
    _patch_core_client()
    _patch_engine_core()
    _patch_scheduler()


def _patch_async_llm() -> None:
    if not ASYNC_LLM.exists():
        print(f"[WARN] async_llm.py not found ({ASYNC_LLM})")
        return
    txt = ASYNC_LLM.read_text()
    txt, removed_old = _strip_old_offload_aliases(txt)
    if "async def restore_agent_kv" in txt and "async def offload_agent_kv" in txt:
        if removed_old:
            ASYNC_LLM.write_text(txt)
            print("[OK] async_llm.py: old agent connector aliases removed")
            return
        print("[SKIP] async_llm.py already has agent connector methods")
        return
    methods = _code_block(4, """\
        async def register_agent_request(self, agent_id: str, request_id: str) -> None:
            if not agent_id:
                return
            await self.engine_core.register_agent_request_async(
                str(agent_id), request_id)

        async def offload_agent_kv(
                self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
            return await self.engine_core.offload_agent_kv_async(
                str(agent_id), only_ref_cnt_zero)

        async def restore_agent_kv(self, agent_id: str) -> dict:
            return await self.engine_core.restore_agent_kv_async(str(agent_id))
    """, leading_newline=True)
    txt, ok = _insert_before_anchor(
        txt,
        ["    async def reset_prefix_cache(", "    async def sleep("],
        methods,
        "async_llm.py agent connector methods",
    )
    if ok:
        ASYNC_LLM.write_text(txt)
        print("[OK] async_llm.py: agent connector methods added")
    elif removed_old:
        ASYNC_LLM.write_text(txt)
        print("[OK] async_llm.py: old agent connector aliases removed")


def _patch_core_client() -> None:
    if not CORE_CLIENT.exists():
        print(f"[WARN] core_client.py not found ({CORE_CLIENT})")
        return
    txt = CORE_CLIENT.read_text()
    txt, changed = _strip_old_offload_aliases(txt)

    if "def restore_agent_kv(self, agent_id: str) -> dict:" not in txt:
        abstract_sync = _code_block(4, """\
            def register_agent_request(self, agent_id: str, request_id: str) -> None:
                raise NotImplementedError

            def offload_agent_kv(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                raise NotImplementedError

            def restore_agent_kv(self, agent_id: str) -> dict:
                raise NotImplementedError
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    def reset_prefix_cache(self) -> None:\n"
                "        raise NotImplementedError\n",
                "    def reset_prefix_cache(\n"
                "        self, reset_running_requests: bool = False, reset_connector: bool = False\n"
                "    ) -> bool:\n"
                "        raise NotImplementedError\n",
            ],
            abstract_sync,
            "core_client.py abstract sync methods",
        )
        changed = changed or ok

    if "async def restore_agent_kv_async" not in txt:
        abstract_async = _code_block(4, """\
            async def register_agent_request_async(
                    self, agent_id: str, request_id: str) -> None:
                raise NotImplementedError

            async def offload_agent_kv_async(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                raise NotImplementedError

            async def restore_agent_kv_async(self, agent_id: str) -> dict:
                raise NotImplementedError
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    async def reset_prefix_cache_async(self) -> None:\n"
                "        raise NotImplementedError\n",
                "    async def reset_prefix_cache_async(\n"
                "        self, reset_running_requests: bool = False, reset_connector: bool = False\n"
                "    ) -> bool:\n"
                "        raise NotImplementedError\n",
            ],
            abstract_async,
            "core_client.py abstract async methods",
        )
        changed = changed or ok

    if "self.engine_core.restore_agent_kv(agent_id)" not in txt:
        inproc_sync = _code_block(4, """\
            def register_agent_request(self, agent_id: str, request_id: str) -> None:
                self.engine_core.register_agent_request(agent_id, request_id)

            def offload_agent_kv(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                return self.engine_core.offload_agent_kv(
                    agent_id, only_ref_cnt_zero)

            def restore_agent_kv(self, agent_id: str) -> dict:
                return self.engine_core.restore_agent_kv(agent_id)
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    def reset_prefix_cache(self) -> None:\n"
                "        self.engine_core.reset_prefix_cache()\n",
                "    def reset_prefix_cache(\n"
                "        self, reset_running_requests: bool = False, reset_connector: bool = False\n"
                "    ) -> bool:\n"
                "        return self.engine_core.reset_prefix_cache(\n"
                "            reset_running_requests, reset_connector\n"
                "        )\n",
            ],
            inproc_sync,
            "core_client.py in-process methods",
        )
        changed = changed or ok

    if 'self.call_utility("restore_agent_kv"' not in txt:
        mp_sync = _code_block(4, """\
            def register_agent_request(self, agent_id: str, request_id: str) -> None:
                self.call_utility("register_agent_request", agent_id, request_id)

            def offload_agent_kv(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                return self.call_utility(
                    "offload_agent_kv", agent_id, only_ref_cnt_zero)

            def restore_agent_kv(self, agent_id: str) -> dict:
                return self.call_utility("restore_agent_kv", agent_id)
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    def reset_prefix_cache(self) -> None:\n"
                "        self.call_utility(\"reset_prefix_cache\")\n",
                "    def reset_prefix_cache(\n"
                "        self, reset_running_requests: bool = False, reset_connector: bool = False\n"
                "    ) -> bool:\n"
                "        return self.call_utility(\n"
                "            \"reset_prefix_cache\", reset_running_requests, reset_connector\n"
                "        )\n",
            ],
            mp_sync,
            "core_client.py MP sync methods",
        )
        changed = changed or ok

    if 'self.call_utility_async("restore_agent_kv"' not in txt:
        async_mp = _code_block(4, """\
            async def register_agent_request_async(
                    self, agent_id: str, request_id: str) -> None:
                await self.call_utility_async(
                    "register_agent_request", agent_id, request_id)

            async def offload_agent_kv_async(
                    self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
                return await self.call_utility_async(
                    "offload_agent_kv", agent_id, only_ref_cnt_zero)

            async def restore_agent_kv_async(self, agent_id: str) -> dict:
                return await self.call_utility_async("restore_agent_kv", agent_id)
        """, leading_newline=True)
        txt, ok = _insert_after_anchor(
            txt,
            [
                "    async def reset_prefix_cache_async(self) -> None:\n"
                "        await self.call_utility_async(\"reset_prefix_cache\")\n",
                "    async def reset_prefix_cache_async(\n"
                "        self, reset_running_requests: bool = False, reset_connector: bool = False\n"
                "    ) -> bool:\n"
                "        return await self.call_utility_async(\n"
                "            \"reset_prefix_cache\", reset_running_requests, reset_connector\n"
                "        )\n",
            ],
            async_mp,
            "core_client.py MP async methods",
        )
        changed = changed or ok

    if changed:
        CORE_CLIENT.write_text(txt)
        print("[OK] core_client.py: agent connector forwarding added")
    else:
        print("[SKIP] core_client.py already patched or no anchors matched")


def _patch_engine_core() -> None:
    if not ENGINE_CORE.exists():
        print(f"[WARN] core.py not found ({ENGINE_CORE})")
        return
    txt = ENGINE_CORE.read_text()
    txt, removed_old = _strip_old_offload_aliases(txt)
    if "def restore_agent_kv(" in txt and "def offload_agent_kv(" in txt:
        if removed_old:
            ENGINE_CORE.write_text(txt)
            print("[OK] core.py: old agent connector aliases removed")
            return
        print("[SKIP] core.py already has agent connector methods")
        return
    methods = _code_block(4, """\
        def register_agent_request(self, agent_id: str, request_id: str) -> None:
            self.scheduler.register_agent_request(agent_id, request_id)

        def offload_agent_kv(
                self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
            return self.scheduler.offload_agent_kv(agent_id, only_ref_cnt_zero)

        def restore_agent_kv(self, agent_id: str) -> dict:
            return self.scheduler.restore_agent_kv(agent_id)
    """, leading_newline=True)
    txt, ok = _insert_after_anchor(
        txt,
        [
            "    def reset_prefix_cache(self):\n"
            "        self.scheduler.reset_prefix_cache()\n",
            "    def reset_prefix_cache(self) -> bool:\n"
            "        return self.scheduler.reset_prefix_cache()\n",
            "    def reset_prefix_cache(\n"
            "        self, reset_running_requests: bool = False, reset_connector: bool = False\n"
            "    ) -> bool:\n"
            "        return self.scheduler.reset_prefix_cache(\n"
            "            reset_running_requests, reset_connector\n"
            "        )\n",
        ],
        methods,
        "core.py agent connector methods",
    )
    if ok:
        ENGINE_CORE.write_text(txt)
        print("[OK] core.py: agent connector methods added")
    elif removed_old:
        ENGINE_CORE.write_text(txt)
        print("[OK] core.py: old agent connector aliases removed")


def _patch_scheduler() -> None:
    if not SCHEDULER.exists():
        print(f"[WARN] scheduler.py not found ({SCHEDULER})")
        return
    txt = SCHEDULER.read_text()
    txt, removed_old = _strip_old_offload_aliases(txt)
    if "def restore_agent_kv(self, agent_id: str) -> dict:" in txt:
        if removed_old:
            SCHEDULER.write_text(txt)
            print("[OK] scheduler.py: old agent connector aliases removed")
            return
        print("[SKIP] scheduler.py already has agent connector methods")
        return
    methods = _code_block(4, """\
        def _agent_kv_connector_scheduler(self):
            connector = self.get_kv_connector() if hasattr(
                self, "get_kv_connector") else getattr(self, "connector", None)
            return getattr(connector, "connector_scheduler", connector)

        def register_agent_request(self, agent_id: str, request_id: str) -> None:
            scheduler = self._agent_kv_connector_scheduler()
            if scheduler is not None and hasattr(scheduler, "register_agent_request"):
                scheduler.register_agent_request(agent_id, request_id)

        def offload_agent_kv(
                self, agent_id: str, only_ref_cnt_zero: bool = True) -> dict:
            scheduler = self._agent_kv_connector_scheduler()
            if scheduler is None or not hasattr(scheduler, "offload_agent_kv"):
                return {
                    "offloaded": False,
                    "reason": "agent-aware KV connector is not configured",
                }
            return scheduler.offload_agent_kv(agent_id, only_ref_cnt_zero)

        def restore_agent_kv(self, agent_id: str) -> dict:
            scheduler = self._agent_kv_connector_scheduler()
            if scheduler is None or not hasattr(scheduler, "restore_agent_kv"):
                return {
                    "restored": False,
                    "reason": "agent-aware KV connector is not configured",
                }
            return scheduler.restore_agent_kv(agent_id)
    """)
    txt, ok = _insert_before_anchor(
        txt,
        ["    def reset_prefix_cache(", "    def make_stats("],
        methods,
        "scheduler.py agent connector methods",
    )
    if ok:
        SCHEDULER.write_text(txt)
        print("[OK] scheduler.py: agent connector methods added")
    elif removed_old:
        SCHEDULER.write_text(txt)
        print("[OK] scheduler.py: old agent connector aliases removed")


def patch_agent_kv_api() -> None:
    if not API_SERVER.exists():
        print(f"[WARN] api_server.py not found ({API_SERVER})")
        return
    txt = API_SERVER.read_text()
    route = textwrap.dedent("""\

# BEGIN agent-concurrency agent KV API routes
async def _agent_kv_json(raw_request: Request):
    try:
        return await raw_request.json()
    except Exception:
        return {}


def _agent_kv_response(payload, status_code: int = 200):
    from fastapi.responses import JSONResponse as _AgentKVJSONResponse
    return _AgentKVJSONResponse(payload, status_code=status_code)


async def _agent_kv_maybe_await(result):
    import inspect as _agent_kv_inspect
    if _agent_kv_inspect.isawaitable(result):
        return await result
    return result


@router.post("/agent_kv_cache/offload")
async def offload_agent_kv_cache(raw_request: Request):
    body = await _agent_kv_json(raw_request)
    agent_id = str(body.get("agent_id") or "")
    only_ref_cnt_zero = bool(body.get("only_ref_cnt_zero", True))
    if not agent_id:
        return _agent_kv_response(
            {"offloaded": False, "reason": "missing agent_id"},
            status_code=400,
        )
    client = engine_client(raw_request)
    if not hasattr(client, "offload_agent_kv"):
        return _agent_kv_response(
            {
                "offloaded": False,
                "reason": "engine client missing offload_agent_kv",
            },
            status_code=501,
        )
    result = client.offload_agent_kv(
        agent_id, only_ref_cnt_zero=only_ref_cnt_zero)
    result = await _agent_kv_maybe_await(result)
    return _agent_kv_response(result or {
        "offloaded": False,
        "reason": "engine returned no result",
    })


@router.post("/agent_kv_cache/restore")
async def restore_agent_kv_cache(raw_request: Request):
    body = await _agent_kv_json(raw_request)
    agent_id = str(body.get("agent_id") or "")
    if not agent_id:
        return _agent_kv_response(
            {"restored": False, "reason": "missing agent_id"},
            status_code=400,
        )
    client = engine_client(raw_request)
    if not hasattr(client, "restore_agent_kv"):
        return _agent_kv_response(
            {
                "restored": False,
                "reason": "engine client missing restore_agent_kv",
            },
            status_code=501,
        )
    result = client.restore_agent_kv(agent_id)
    result = await _agent_kv_maybe_await(result)
    return _agent_kv_response(result or {
        "restored": False,
        "reason": "engine returned no result",
    })
# END agent-concurrency agent KV API routes
    """)

    old_block = re.compile(
        r"\n[ \t]*# BEGIN agent-concurrency agent KV API routes\n"
        r".*?"
        r"\n[ \t]*# END agent-concurrency agent KV API routes\n",
        re.DOTALL,
    )
    txt, removed = old_block.subn("\n", txt)

    anchor = "router = APIRouter()\n"
    if anchor in txt:
        txt = txt.replace(anchor, anchor + route + "\n", 1)
        API_SERVER.write_text(txt)
        action = "repositioned" if removed else "added"
        print(f"[OK] api_server.py: agent offload/restore routes {action}")
        return

    print("[WARN] Could not find module-level router anchor; trying route anchors", file=sys.stderr)
    for fallback_anchor in (
        '@router.post("/reset_prefix_cache")',
        '@router.post("/reset_mm_cache")',
        '@router.post("/reset_encoder_cache")',
        '@router.get("/health")',
    ):
        if fallback_anchor in txt:
            idx = txt.index(fallback_anchor)
            line_start = txt.rfind("\n", 0, idx) + 1
            indent = txt[line_start:idx]
            block = textwrap.indent(route, indent) if indent else route
            txt = txt.replace(indent + fallback_anchor,
                              block + "\n" + indent + fallback_anchor,
                              1)
            API_SERVER.write_text(txt)
            action = "repositioned" if removed else "added"
            print(f"[OK] api_server.py: agent offload/restore routes {action}")
            return
    print("[WARN] Could not find API route anchor; skipping routes", file=sys.stderr)


def _validate_agent_kv_api_routes(errors: list[str]) -> None:
    if not API_SERVER.exists():
        return
    txt = API_SERVER.read_text()
    try:
        compile(txt, str(API_SERVER), "exec")
    except SyntaxError as exc:
        errors.append(f"api_server.py syntax error: {exc}")
        return

    for route_path in (
        '"/agent_kv_cache/offload"',
        '"/agent_kv_cache/restore"',
    ):
        if route_path not in txt:
            errors.append(f"api_server.py missing {route_path} route")

    router_idx = txt.find("router = APIRouter()")
    route_idx = txt.find('@router.post("/agent_kv_cache/offload")')
    if router_idx < 0 or route_idx < 0:
        return
    line_start = txt.rfind("\n", 0, route_idx) + 1
    if txt[line_start:route_idx]:
        errors.append("api_server.py agent KV route is not module-level")
    if route_idx < router_idx:
        errors.append("api_server.py agent KV route appears before router is defined")


def validate() -> None:
    import types

    errors: list[str] = []
    try:
        exec(compile(PROTO.read_text(), str(PROTO), "exec"), types.ModuleType("proto").__dict__)
    except SyntaxError as exc:
        errors.append(f"protocol.py syntax error: {exc}")
    except Exception:
        pass

    try:
        compile(CONNECTOR_DST.read_text(), str(CONNECTOR_DST), "exec")
    except SyntaxError as exc:
        errors.append(f"agent_offloading_connector.py syntax error: {exc}")

    checks = [
        (PROTO, "kv_blocks_used", "protocol.py missing kv_blocks_used"),
        (PROTO, "kv_blocks_size_gb", "protocol.py missing kv_blocks_size_gb"),
        (PROTO, "agent_id", "protocol.py missing agent_id"),
        (SERVING, "_compute_kv_blocks", "serving_chat.py missing KV telemetry helper"),
        (SERVING, "register_agent_request", "serving_chat.py missing agent request registration"),
        (CONNECTOR_DST, "class AgentAwareOffloadingConnector", "custom connector not installed"),
        (ASYNC_LLM, "async def offload_agent_kv", "async_llm.py missing offload_agent_kv"),
        (ASYNC_LLM, "async def restore_agent_kv", "async_llm.py missing restore_agent_kv"),
        (CORE_CLIENT, "restore_agent_kv", "core_client.py missing restore forwarding"),
        (ENGINE_CORE, "def restore_agent_kv", "core.py missing restore_agent_kv"),
        (SCHEDULER, "def restore_agent_kv", "scheduler.py missing restore_agent_kv"),
    ]
    for path, needle, message in checks:
        if path.exists() and needle not in path.read_text():
            errors.append(message)
    _validate_agent_kv_api_routes(errors)

    if errors:
        for err in errors:
            print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)
    print("[OK] validation passed")


def _backup_or_restore(paths: list[pathlib.Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        bak = path.with_suffix(path.suffix + ".bak")
        if bak.exists():
            shutil.copy2(bak, path)
            print(f"[RESTORE] {path.name} <- {bak.name}")
        else:
            shutil.copy2(path, bak)
            print(f"[BACKUP] {path.name} -> {bak.name}")


if __name__ == "__main__":
    _backup_or_restore([
        PROTO,
        SERVING,
        API_SERVER,
        ASYNC_LLM,
        CORE_CLIENT,
        ENGINE_CORE,
        SCHEDULER,
    ])
    print("=== Applying agent-aware KV offloading patches ===")
    patch_protocol()
    patch_serving_chat()
    install_agent_connector()
    patch_engine_hooks()
    patch_agent_kv_api()
    validate()
    print("=== All patches applied successfully ===")
