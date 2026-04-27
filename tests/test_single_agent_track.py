"""
tests/test_single_agent_track.py

Integration test: start vLLM, run one agent on one ABC-Bench task, then
validate that the per-agent KV cache metrics returned by the patched vLLM
(kv_blocks_used, kv_blocks_size_gb in UsageInfo) are consistent with what
vLLM's own Prometheus /metrics endpoint reports.

Three properties are checked:

  1. Patch active      — kv_blocks_used > 0 in at least one LLM response.
  2. Formula match     — kv_blocks_size_gb ≈ kv_blocks_used × bytes_per_block
                         converted to GiB, matching the patched vLLM field
                         (verifies the patch's geometry calculation end-to-end).
  3. Prometheus bound  — peak Prometheus num_gpu_blocks_used ≥ kv_blocks_used
                         for the largest single call (global metric must have
                         seen at least as many blocks as the agent reported).

Environment variables
─────────────────────
  ABC_BENCH_ROOT   (optional) Path to the ABC-Bench root containing tasks/.
                   Defaults to ../tasks, then ./tasks, when either exists.
  ABC_BENCH_TASK   (optional) Task-name glob; defaults to the first task_* match.
  ABC_BENCH_VENV   (optional) Benchmark venv path; defaults to ./.bench-venv.
  LLM_API_KEY      (optional) API key; defaults to "sk-local-ascend".
  SKIP_VLLM_START  Defaults to "1" to assume vLLM is already running.

Run
───
  # Assumes vLLM is already running and automatically re-runs inside ./.bench-venv.
  pytest tests/test_single_agent_track.py -vs

  # Override defaults when needed:
  ABC_BENCH_ROOT=/data/ABC-Bench SKIP_VLLM_START=0 pytest tests/test_single_agent_track.py -vs
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── make src/ importable without installing the package ──────────────────────
REPO_ROOT = Path(__file__).parent.parent


# ── benchmark venv handoff ───────────────────────────────────────────────────

_BENCH_VENV = Path(os.getenv("ABC_BENCH_VENV", REPO_ROOT / ".bench-venv"))
_BENCH_PYTHON = _BENCH_VENV / "bin" / "python"
_BENCH_REEXEC_MARKER = "ABC_BENCH_TEST_REEXECED"
_BENCH_TEST_REQUIREMENTS = (
    "openhands-sdk",
    "openhands-tools",
    "pytest",
    "requests",
    "PyYAML",
)
_BENCH_TEST_IMPORTS = (
    "openhands.sdk",
    "openhands.tools.file_editor",
    "openhands.tools.task_tracker",
    "openhands.tools.terminal",
    "pytest",
    "requests",
    "yaml",
)
_ABC_BENCH_ROOT_DEFAULTS = (
    REPO_ROOT.parent / "tasks",
    REPO_ROOT / "tasks",
)


def _set_default_test_env() -> None:
    os.environ.setdefault("SKIP_VLLM_START", "1")
    os.environ.setdefault("LLM_API_KEY", "sk-local-ascend")
    if os.getenv("ABC_BENCH_ROOT"):
        return
    for candidate in _ABC_BENCH_ROOT_DEFAULTS:
        if candidate.is_dir():
            os.environ["ABC_BENCH_ROOT"] = str(candidate)
            return


def _path_inside(path: Path, parent: Path) -> bool:
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    parent = Path(os.path.abspath(os.path.expanduser(str(parent))))
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _running_inside_bench_venv() -> bool:
    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env and Path(os.path.abspath(virtual_env)) == Path(os.path.abspath(_BENCH_VENV)):
        return True
    return _path_inside(Path(sys.executable), _BENCH_VENV) or _path_inside(Path(sys.prefix), _BENCH_VENV)


def _bench_dependencies_ready() -> bool:
    if not _BENCH_PYTHON.exists():
        return False
    check = "\n".join(f"import {name}" for name in _BENCH_TEST_IMPORTS)
    return subprocess.run(
        [str(_BENCH_PYTHON), "-c", check],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
    ).returncode == 0


def _bootstrap_bench_venv() -> None:
    if not _BENCH_PYTHON.exists():
        subprocess.run(
            ["uv", "venv", str(_BENCH_VENV), "--python", "3.12"],
            cwd=str(REPO_ROOT),
            check=True,
        )

    if _bench_dependencies_ready():
        return

    env = os.environ.copy()
    env.setdefault("UV_HTTP_TIMEOUT", env.get("BENCH_UV_HTTP_TIMEOUT", "300"))
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(_BENCH_PYTHON),
            *_BENCH_TEST_REQUIREMENTS,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )


def _prepare_bench_venv() -> None:
    try:
        _bootstrap_bench_venv()
    except FileNotFoundError as exc:
        missing = exc.filename or "uv"
        raise RuntimeError(
            f"Cannot prepare benchmark venv at {_BENCH_VENV}: {missing!r} was "
            "not found. Install uv or run this test from an environment that "
            "already has the benchmark dependencies."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Cannot prepare benchmark venv at {_BENCH_VENV}; command exited "
            f"with status {exc.returncode}: {' '.join(map(str, exc.cmd))}"
        ) from exc


def _reexec_in_bench_venv() -> None:
    if _running_inside_bench_venv():
        if not _bench_dependencies_ready():
            _prepare_bench_venv()
        return
    if os.getenv(_BENCH_REEXEC_MARKER) == "1":
        raise RuntimeError(
            f"pytest was re-executed for the benchmark venv, but is still "
            f"running from {sys.executable!r} instead of {_BENCH_PYTHON!s}."
        )

    _prepare_bench_venv()

    env = os.environ.copy()
    env[_BENCH_REEXEC_MARKER] = "1"
    env["VIRTUAL_ENV"] = str(_BENCH_VENV)
    env["PATH"] = f"{_BENCH_VENV / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    os.execve(
        str(_BENCH_PYTHON),
        [str(_BENCH_PYTHON), "-m", "pytest", *sys.argv[1:]],
        env,
    )


_set_default_test_env()
_reexec_in_bench_venv()

import pytest
import requests
import yaml

sys.path.insert(0, str(REPO_ROOT / "src"))

import sidecar as _sidecar
from run_abc_bench_instrumented import (
    _litellm_cb_lock,
    copy_task_to_workspace,
    find_task_dirs,
    now_utc,
    run_single_agent,
)

# ── derive constants from vllm_config.yaml ────────────────────────────────────
_VLLM_CFG = yaml.safe_load(
    (REPO_ROOT / "config" / "vllm_config.yaml").read_text()
)
_SC       = _VLLM_CFG["sidecar"]
HOST_PORT = _VLLM_CFG["native"]["port"]
VLLM_URL  = f"http://localhost:{HOST_PORT}"

BYTES_PER_BLK = _sidecar.bytes_per_block(
    num_layers   = _SC["num_layers"],
    num_kv_heads = _SC["num_kv_heads"],
    head_dim     = _SC["head_dim"],
    block_size   = _SC["block_size"],
    dtype        = _SC["dtype"],
)

# LiteLLM expects "openai/<model-name>" for OpenAI-compatible endpoints.
_RAW_MODEL  = _VLLM_CFG["native"]["served_model_name"]  # e.g. Qwen/Qwen3-30B-A3B-Instruct-2507
LLM_MODEL   = f"openai/{_RAW_MODEL.split('/')[-1]}"  # e.g. openai/Qwen3-30B-A3B-Instruct-2507
LLM_BASE_URL = f"{VLLM_URL}/v1"
LLM_API_KEY  = os.getenv("LLM_API_KEY", "sk-local-ascend")


# ── Prometheus helpers ────────────────────────────────────────────────────────

def _fetch_metrics() -> dict[str, float]:
    try:
        r = requests.get(f"{VLLM_URL}/metrics", timeout=5)
        r.raise_for_status()
        return _sidecar.parse_prometheus(r.text)
    except Exception:
        return {}


class _PrometheusPoller:
    """Polls /metrics in a background thread and tracks per-metric peak values."""

    def __init__(self, interval_s: float = 0.5):
        self._interval = interval_s
        self._stop     = threading.Event()
        self._lock     = threading.Lock()
        self._peaks: dict[str, float] = {}
        self._thread   = threading.Thread(target=self._run, daemon=True, name="prom-poller")

    def start(self) -> "_PrometheusPoller":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            sample = _fetch_metrics()
            with self._lock:
                for k, v in sample.items():
                    if v > self._peaks.get(k, 0.0):
                        self._peaks[k] = v

    def peak(self, metric: str) -> float:
        with self._lock:
            return self._peaks.get(metric, 0.0)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vllm_running():
    """Start (or verify) vLLM before any test in this module runs."""
    if not os.getenv("ABC_BENCH_ROOT"):
        pytest.skip("ABC_BENCH_ROOT not set — skipping hardware integration test")

    if os.getenv("SKIP_VLLM_START") == "1":
        try:
            requests.get(
                f"{VLLM_URL}/v1/models",
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                timeout=10,
            ).raise_for_status()
        except Exception as exc:
            pytest.fail(
                f"SKIP_VLLM_START=1 but vLLM is unreachable at {VLLM_URL}: {exc}"
            )
    else:
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "start_vllm.sh"),
             str(REPO_ROOT / "config" / "vllm_config.yaml")],
            cwd=str(REPO_ROOT),
            timeout=600,   # up to 10 min for model load on first run
        )
        if result.returncode != 0:
            pytest.fail(f"start_vllm.sh exited with code {result.returncode}")

    yield VLLM_URL


@pytest.fixture(scope="module")
def task_dir(tmp_path_factory):
    """Return a writable sandbox copy of one ABC-Bench task."""
    root      = Path(os.environ["ABC_BENCH_ROOT"])
    task_name = os.getenv("ABC_BENCH_TASK")

    if task_name:
        candidates = sorted((root / "tasks").glob(task_name)) or sorted(root.glob(task_name))
    else:
        candidates = find_task_dirs(root, "task_*")

    if not candidates:
        pytest.skip(f"No tasks matched under {root}")

    src       = candidates[0]
    workspace = tmp_path_factory.mktemp("workspace")
    return copy_task_to_workspace(src, workspace, suffix="_test")


# ── test ──────────────────────────────────────────────────────────────────────

def test_single_agent_kv_tracking(vllm_running, task_dir, tmp_path):
    """
    End-to-end KV tracking validation for a single agent run.

    Assertions
    ──────────
    1. Patch active:   at least one LLM call returned kv_blocks_used > 0.
    2. Formula match:  kv_blocks_size_gb == kv_blocks_used × bytes_per_block / 1024^3
                       within 1 % relative error.
    3. Prometheus ≥:   peak Prometheus num_gpu_blocks_used (sampled every 0.5 s
                       throughout the run) is ≥ the agent's max kv_blocks_used.
    4. Fraction sane:  gpu_cache_usage_perc remains in [0, 1].
    """
    import litellm as _litellm

    kv_calls: list[dict] = []   # one entry per LLM call that returned KV data

    def _capture_cb(kwargs, completion_response, start_time, end_time) -> None:
        usage = getattr(completion_response, "usage", None)
        if usage is None:
            return
        kb = getattr(usage, "kv_blocks_used",   None)
        ks = getattr(usage, "kv_blocks_size_gb", None)
        if isinstance(kb, int):
            kv_calls.append({
                "kv_blocks_used":   kb,
                "kv_blocks_size_gb": float(ks) if isinstance(ks, (int, float)) else None,
            })

    # Register our capture callback alongside the runner's own _kv_cb.
    with _litellm_cb_lock:
        if not isinstance(getattr(_litellm, "success_callback", None), list):
            _litellm.success_callback = []
        _litellm.success_callback.append(_capture_cb)

    # Poll Prometheus throughout the entire agent run to catch peak allocation.
    poller = _PrometheusPoller(interval_s=0.5).start()

    try:
        run_single_agent(
            agent_id    = "agent_test_single",
            task_dir    = task_dir,
            llm_model   = LLM_MODEL,
            llm_base_url= LLM_BASE_URL,
            llm_api_key = LLM_API_KEY,
            max_iterations = 10,          # keep the test short
            results_dir    = tmp_path / "results",
            scheduled_dt   = now_utc(),
        )
    finally:
        poller.stop()
        with _litellm_cb_lock:
            try:
                _litellm.success_callback.remove(_capture_cb)
            except (ValueError, AttributeError):
                pass

    # ── 1. patch active ───────────────────────────────────────────────────────
    assert kv_calls, (
        "No kv_blocks_used captured from any LLM response.\n"
        "Likely causes: the vLLM patch is not active (image not rebuilt), "
        "or agent_id was not forwarded in the request extra_body."
    )

    # Pick the call with the highest token count for comparison.
    peak_call  = max(kv_calls, key=lambda c: c["kv_blocks_used"])
    kv_blocks  = peak_call["kv_blocks_used"]
    kv_gb      = peak_call["kv_blocks_size_gb"]

    assert kv_blocks > 0, "kv_blocks_used must be > 0"

    # ── 2. formula match ──────────────────────────────────────────────────────
    if kv_gb is not None:
        expected_gb = kv_blocks * BYTES_PER_BLK / (1024 ** 3)
        rel_err = abs(kv_gb - expected_gb) / max(expected_gb, 1e-12)
        assert rel_err < 0.01, (
            f"kv_blocks_size_gb formula mismatch:\n"
            f"  response  : {kv_gb:.6f} GiB\n"
            f"  expected  : {expected_gb:.6f} GiB "
            f"({kv_blocks} blocks × {BYTES_PER_BLK} B/block)\n"
            f"  rel_error : {rel_err:.2%}  (tolerance: 1 %)\n"
            f"  Geometry  : layers={_SC['num_layers']} kv_heads={_SC['num_kv_heads']} "
            f"head_dim={_SC['head_dim']} block_size={_SC['block_size']} "
            f"dtype={_SC['dtype']}"
        )

    # ── 3. Prometheus peak ≥ agent peak ───────────────────────────────────────
    total_blks = poller.peak("vllm:num_gpu_blocks")
    used_pct   = poller.peak("vllm:gpu_cache_usage_perc")   # fraction 0–1
    peak_prom_blocks = round(total_blks * used_pct) if total_blks and used_pct else 0

    assert peak_prom_blocks >= kv_blocks, (
        f"Prometheus peak blocks ({peak_prom_blocks}) < "
        f"agent kv_blocks_used ({kv_blocks}).\n"
        f"  vllm:num_gpu_blocks peak          : {total_blks}\n"
        f"  vllm:gpu_cache_usage_perc peak     : {used_pct:.4f}\n"
        f"  Prometheus derived used blocks     : {peak_prom_blocks}\n"
        f"  Agent kv_blocks_used (peak call)   : {kv_blocks}\n"
        "The Prometheus poller (0.5 s interval) may have missed the allocation "
        "window, or the metrics endpoint was not updated in time."
    )

    # ── 4. fraction sanity ────────────────────────────────────────────────────
    if used_pct:
        assert 0.0 <= used_pct <= 1.0, (
            f"gpu_cache_usage_perc={used_pct} is outside [0, 1]"
        )

    # Summary printed on pass (visible with pytest -v or -s).
    print(
        f"\n[PASS] kv_blocks_used={kv_blocks}  "
        f"kv_gb={kv_gb:.4f}  "
        f"prometheus_peak_blocks={peak_prom_blocks}  "
        f"({len(kv_calls)} LLM calls, "
        f"bytes_per_block={BYTES_PER_BLK})"
    )
