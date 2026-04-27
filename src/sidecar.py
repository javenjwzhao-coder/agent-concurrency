#!/usr/bin/env python3
"""
sidecar.py — Real-time vLLM KV cache + agent state monitor.

Can run in two modes:

  Embedded (recommended, zero IPC overhead):
      Imported by run_abc_bench_instrumented.py and started as a daemon thread.
      Reads _LIVE_AGENTS directly from memory — no serialization, no disk I/O.

  Standalone (for attaching to a running benchmark after the fact):
      python src/sidecar.py --live-state ./abc_results/.live_state.json ...
      Reads a JSON file written by the runner's --live-state-path option.

Every t seconds, appends one JSONL record to --log-file containing:
  • Global physical KV cache used (GB) from vLLM's Prometheus /metrics endpoint.
  • Per-agent: current phase, KV blocks used, KV cache GB.

Block-to-GB conversion uses the model geometry passed on the CLI, matching
the formula used by vLLM's patched serving_chat.py.

Standalone usage:
    python src/sidecar.py \\
        --vllm-url    http://localhost:8027 \\
        --live-state  ./abc_results/.live_state.json \\
        --log-file    ./sidecar.log \\
        --interval    5 \\
        --num-layers  48 --num-kv-heads 8 --head-dim 128 \\
        --block-size  16 --dtype bfloat16

Requirements:
    pip install requests
"""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

# ─────────────────────────────────────── constants ────────────────────────────

_DTYPE_BYTES: dict[str, int] = {
    "float32": 4, "float16": 2, "bfloat16": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "int8": 1,
}

# Metric name candidates — tried in order; first match wins.
# vllm_ascend (Ascend NPU fork) may emit "npu" variants instead of "gpu".
_METRIC_USAGE_PCT_CANDIDATES  = [
    "vllm:gpu_cache_usage_perc",   # standard vLLM (GPU)
    "vllm:npu_cache_usage_perc",   # vllm-ascend (Ascend NPU)
]
_METRIC_TOTAL_BLKS_CANDIDATES = [
    "vllm:num_gpu_blocks",         # standard vLLM (GPU)
    "vllm:num_npu_blocks",         # vllm-ascend (Ascend NPU)
]

# Matches one Prometheus text-format sample line, e.g.:
#   vllm:gpu_cache_usage_perc{model_name="..."} 0.452
#   vllm:gpu_cache_usage_perc{model_name="..."} NaN
# Prometheus allows NaN / +Inf / -Inf as sample values; include them so we
# don't silently skip metrics that haven't been initialised yet.
_PROM_RE = re.compile(
    r'^([\w:]+)(?:\{[^}]*\})?\s+(NaN|[+-]?Inf|[\d.eE+\-]+)',
    re.MULTILINE,
)

# ─────────────────────────────────────── helpers ──────────────────────────────

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus text format → {metric_name: first_finite_sample_value}.

    NaN / ±Inf samples are kept as-is (math.isfinite can filter them later).
    """
    result: dict[str, float] = {}
    for m in _PROM_RE.finditer(text):
        name = m.group(1)
        if name not in result:
            result[name] = float(m.group(2))
    return result


def _first_metric(metrics: dict[str, float], candidates: list[str]) -> Optional[float]:
    """Return the first finite value found among candidate metric names, or None."""
    for name in candidates:
        val = metrics.get(name)
        if val is not None and math.isfinite(val):
            return val
    return None


def bytes_per_block(num_layers: int, num_kv_heads: int, head_dim: int,
                    block_size: int, dtype: str) -> int:
    """KV cache bytes consumed by one vLLM block.

    Matches vLLM's patched serving_chat.py formula:
        block_size × num_layers × 2(K+V) × num_kv_heads × head_dim × dtype_bytes
    """
    return block_size * num_layers * 2 * num_kv_heads * head_dim * _DTYPE_BYTES.get(dtype, 2)


def _file_agent_reader(path: Path) -> Callable[[], dict]:
    """Return a callable that reads agents from a live-state JSON file."""
    def _read() -> dict:
        try:
            return json.loads(path.read_text()).get("agents", {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    return _read


# ─────────────────────────────────────── core loop ────────────────────────────

# Tracks whether we've already emitted the "no KV metrics found" diagnostic.
_kv_metric_warn_emitted = False


def poll_vllm(session: requests.Session, vllm_url: str, bytes_per_blk: int) -> dict:
    """Fetch /metrics and return global KV cache statistics."""
    global _kv_metric_warn_emitted
    try:
        resp = session.get(f"{vllm_url}/metrics", timeout=5)
        resp.raise_for_status()
        metrics = parse_prometheus(resp.text)

        used_pct   = _first_metric(metrics, _METRIC_USAGE_PCT_CANDIDATES)
        total_blks = _first_metric(metrics, _METRIC_TOTAL_BLKS_CANDIDATES)

        if used_pct is None and total_blks is None and not _kv_metric_warn_emitted:
            _kv_metric_warn_emitted = True
            known = sorted(k for k in metrics if k.startswith("vllm:"))
            print(
                "[sidecar] WARNING: KV cache metrics not found in /metrics response.\n"
                f"          Tried usage: {_METRIC_USAGE_PCT_CANDIDATES}\n"
                f"          Tried blocks: {_METRIC_TOTAL_BLKS_CANDIDATES}\n"
                f"          vllm: metrics present: {known[:20]}",
                flush=True,
            )

        used_blks: Optional[int] = (
            round(total_blks * used_pct)
            if (total_blks is not None and used_pct is not None) else None
        )
        total_gb: Optional[float] = (
            round(total_blks * bytes_per_blk / 1e9, 3) if total_blks is not None else None
        )
        used_gb: Optional[float] = (
            round(used_blks * bytes_per_blk / 1e9, 3) if used_blks is not None else None
        )
        return {
            "kv_cache_used_pct":    round(used_pct * 100, 2) if used_pct is not None else None,
            "num_gpu_blocks_total": int(total_blks) if total_blks is not None else None,
            "num_gpu_blocks_used":  used_blks,
            "kv_total_gb":          total_gb,
            "kv_used_gb":           used_gb,
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_loop(args: argparse.Namespace, bytes_per_blk: int,
             get_agents: Callable[[], dict]) -> None:
    """Main poll loop.

    Parameters
    ----------
    args          Parsed CLI namespace (vllm_url, log_file, interval, …).
    bytes_per_blk Pre-computed bytes per KV block (from bytes_per_block()).
    get_agents    Zero-argument callable that returns the current per-agent
                  state dict.  In embedded mode this reads _LIVE_AGENTS
                  directly; in standalone mode it reads the live-state file.
    """
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    tick = 0

    print(f"[sidecar] polling every {args.interval}s → {log_path}", flush=True)
    print(f"[sidecar] vLLM: {args.vllm_url}", flush=True)
    print(f"[sidecar] block geometry: {bytes_per_blk} bytes/block", flush=True)

    while True:
        tick_start = time.monotonic()

        agents = get_agents()           # zero-copy in embedded mode
        record = {
            "ts":     iso_utc(now_utc()),
            "tick":   tick,
            "vllm":   poll_vllm(session, args.vllm_url, bytes_per_blk),
            "agents": agents,
        }

        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

        # Brief stdout summary.
        vllm_info = record["vllm"]
        used_pct  = vllm_info.get("kv_cache_used_pct")
        used_gb   = vllm_info.get("kv_used_gb")
        states    = [a.get("state", "?") for a in agents.values()]
        state_str = (", ".join(f"{s}: {states.count(s)}" for s in dict.fromkeys(states))
                     if states else "none")
        gb_str    = f"{used_gb:.2f} GB" if used_gb is not None else "n/a"
        pct_str   = f"{used_pct:.1f}%" if used_pct is not None else "n/a"
        print(f"[sidecar] tick={tick:4d}  kv={gb_str} ({pct_str})  "
              f"agents={len(agents)} [{state_str}]", flush=True)

        tick += 1
        elapsed = time.monotonic() - tick_start
        time.sleep(max(0.0, args.interval - elapsed))


# ─────────────────────────────────────── CLI (standalone mode) ────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone sidecar: poll vLLM /metrics and a live-state "
                    "JSON file, write JSONL log.  For zero-overhead embedded "
                    "mode, use --sidecar-* flags in run_abc_bench_instrumented.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--vllm-url", default="http://localhost:8000")
    p.add_argument("--live-state", required=True,
                   help="Live-state JSON file written by the instrumented runner "
                        "(--live-state-path / default: <results-root>/.live_state.json).")
    p.add_argument("--log-file", default="sidecar.log")
    p.add_argument("--interval", "-t", type=float, default=5.0)

    geo = p.add_argument_group("model geometry")
    geo.add_argument("--num-layers",   type=int, required=True)
    geo.add_argument("--num-kv-heads", type=int, required=True)
    geo.add_argument("--head-dim",     type=int, required=True)
    geo.add_argument("--block-size",   type=int, default=16)
    geo.add_argument("--dtype",        default="bfloat16", choices=list(_DTYPE_BYTES))

    return p.parse_args()


def main() -> int:
    args = parse_args()

    bpb = bytes_per_block(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        dtype=args.dtype,
    )
    get_agents = _file_agent_reader(Path(args.live_state))

    def _stop(sig, frame):
        print("\n[sidecar] shutting down.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    run_loop(args, bpb, get_agents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
