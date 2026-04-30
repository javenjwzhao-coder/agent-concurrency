#!/bin/bash
# start_vllm.sh — Start vLLM on Ascend NPU (native bare-metal)
# Usage: ./start_vllm.sh [config.yaml]
set -euo pipefail

CONFIG="${1:-config/vllm_config.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config file not found: $CONFIG"
    exit 1
fi

if ! python3 -c "import yaml" 2>/dev/null; then
    echo "[ERROR] PyYAML is required. Install: pip install pyyaml"
    exit 1
fi

_CFG_JSON=$(python3 -c "import yaml,json; print(json.dumps(yaml.safe_load(open('$CONFIG'))))")

cfg() {
    echo "$_CFG_JSON" | python3 -c "
import json,sys
c=json.load(sys.stdin)
for k in '$1'.split('.'): c=c[k]
print(' '.join(str(i) for i in c) if isinstance(c,list) else ('' if c is None else c))
"
}

export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

MAX_RETRIES=30
RETRY_INTERVAL=10

# =============================================================================
# HEALTH CHECK
# =============================================================================
wait_for_ready() {
    echo "[INFO] Waiting for vLLM on port ${HOST_PORT}..."
    for i in $(seq 1 "$MAX_RETRIES"); do
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
            -H "Authorization: Bearer ${VLLM_API_KEY}" \
            "http://localhost:${HOST_PORT}/v1/models" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "[INFO] vLLM is READY."
            return 0
        fi
        [ $((i % 3)) -eq 0 ] && echo "[INFO]   attempt ${i}/${MAX_RETRIES}, status=${code}"
        sleep "$RETRY_INTERVAL"
    done
    echo "[ERROR] Timed out. Last status: ${code}"
    return 1
}

# =============================================================================
# NATIVE LAUNCH (no Docker)
# Creates a project-local uv venv that inherits non-vLLM dependencies from the
# shared venv, then installs and patches vllm-ascend locally. The shared
# /opt/vllm/venv is never modified.
# =============================================================================
start_native() {
    local TOOLKIT=$(cfg native.ascend_toolkit_path)
    local ATB=$(cfg native.atb_path)
    local SHARED_PY=$(cfg native.shared_venv_python)
    local WORKDIR=$(cfg native.workdir)
    local DEVICES=$(cfg native.devices)
    local MODEL=$(cfg native.model_path)
    local SERVED_NAME=$(cfg native.served_model_name)
    local HOST=$(cfg native.host)
    local PORT=$(cfg native.port)
    local API_KEY=$(cfg native.api_key)
    local TP=$(cfg native.tensor_parallel_size)
    local DTYPE=$(cfg native.dtype)
    local EXTRA=$(cfg native.extra_args)
    local PID_FILE=$(cfg native.pid_file)

    REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
    VENV="$REPO_DIR/.venv"
    VLLM_ASCEND_VERSION="${VLLM_ASCEND_VERSION:-0.13.0}"

    # ── 1. Create project venv via uv (skipped if already exists) ────────────
    if [ ! -f "$VENV/bin/activate" ]; then
        echo "[INFO] Creating project venv with uv..."
        uv venv "$VENV" --python "$SHARED_PY" --system-site-packages
    fi

    # ── 2. Link shared non-vLLM deps, then install local vllm-ascend once ────
    PYTHON_VER=$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    VENV_SITE="$VENV/lib/python${PYTHON_VER}/site-packages"
    VENV_VLLM="$VENV_SITE/vllm"
    # Make all packages importable from the shared venv (torch, transformers,
    # torch_npu, etc.) by snapshotting the shared Python's full sys.path into a
    # .pth file. Python's site module appends .pth entries AFTER the project
    # venv's own site-packages, so locally installed vllm/vllm-ascend wins.
    # Using sys.path directly (rather than guessing site-packages location)
    # handles editable installs and Ascend-specific path layouts correctly.
    mkdir -p "$VENV_SITE"
    "$SHARED_PY" -c "
import sys
for p in sys.path:
    if p and p != '.':
        print(p)
" > "$VENV_SITE/shared_venv.pth"
    echo "[INFO] Linked shared venv packages ($(wc -l < "$VENV_SITE/shared_venv.pth") paths) for non-vllm deps"

    check_local_vllm_ascend() {
        VENV="$VENV" DESIRED_VLLM_ASCEND="$VLLM_ASCEND_VERSION" "$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import importlib.util
import os
import pathlib
import sys

desired = os.environ["DESIRED_VLLM_ASCEND"]
venv = pathlib.Path(os.environ["VENV"]).resolve()

def dist_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

def package_path(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    if spec.origin:
        return pathlib.Path(spec.origin).resolve()
    locations = spec.submodule_search_locations
    if locations:
        return pathlib.Path(next(iter(locations))).resolve()
    return None

def is_local(path):
    if path is None:
        return False
    try:
        path.relative_to(venv)
        return True
    except ValueError:
        return False

vllm_version = dist_version("vllm")
ascend_version = dist_version("vllm-ascend")
vllm_path = package_path("vllm")
ascend_path = package_path("vllm_ascend")
ready = (
    vllm_version == desired
    and ascend_version == desired
    and is_local(vllm_path)
    and is_local(ascend_path)
)

if ready:
    print(f"[INFO] Reusing local vllm=={vllm_version}, vllm-ascend=={ascend_version}")
else:
    print(
        "[INFO] Local vllm-ascend install needed "
        f"(vllm={vllm_version!r} at {vllm_path}, "
        f"vllm-ascend={ascend_version!r} at {ascend_path})"
    )
sys.exit(0 if ready else 1)
PY
    }

    if check_local_vllm_ascend; then
        :
    else
        echo "[INFO] Installing vllm==${VLLM_ASCEND_VERSION} and vllm-ascend==${VLLM_ASCEND_VERSION} into $VENV ..."
        rm -rf \
          "$VENV_SITE"/vllm \
          "$VENV_SITE"/vllm_ascend \
          "$VENV_SITE"/vllm-*.dist-info \
          "$VENV_SITE"/vllm_ascend-*.dist-info \
          "$VENV_SITE"/vllm_ascend-*.egg-info
        # Keep torch/torch_npu and other Ascend runtime deps from the shared
        # environment. vLLM's PyPI metadata pins upstream CUDA torch versions
        # that can conflict with vllm-ascend's Ascend torch stack.
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$VENV/bin/python" --upgrade --no-deps \
              "vllm==${VLLM_ASCEND_VERSION}" \
              "vllm-ascend==${VLLM_ASCEND_VERSION}"
        else
            "$VENV/bin/python" -m pip install --upgrade --no-deps \
              "vllm==${VLLM_ASCEND_VERSION}" \
              "vllm-ascend==${VLLM_ASCEND_VERSION}"
        fi
        if ! check_local_vllm_ascend; then
            echo "[ERROR] vllm-ascend install verification failed after install" >&2
            exit 1
        fi
    fi

    VLLM_RUNTIME_REQS="$VENV/vllm-runtime-deps.txt"
    VENV_RUNTIME_REQS="$VLLM_RUNTIME_REQS" "$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import os
import pathlib
import re
import sys

try:
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
except Exception:
    default_environment = None
    Requirement = None

skip_names = {
    "flash-attn",
    "flashinfer-python",
    "torch",
    "torch-npu",
    "torchaudio",
    "torchvision",
    "triton",
    "vllm",
    "vllm-ascend",
    "vllm-flash-attn",
    "xformers",
}
skip_prefixes = ("nvidia-",)

def canonicalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

def should_skip(name: str) -> bool:
    normalized = canonicalize(name)
    return normalized in skip_names or any(
        normalized.startswith(prefix) for prefix in skip_prefixes)

def fallback_name(req_line: str) -> str:
    return re.split(r"[ ;<>=!~\[]", req_line, maxsplit=1)[0]

env = default_environment() if default_environment is not None else {}
if env:
    env["extra"] = ""

requirements = []
seen = set()
for dist_name in ("vllm", "vllm-ascend"):
    try:
        req_lines = metadata.requires(dist_name) or []
    except metadata.PackageNotFoundError:
        continue
    for req_line in req_lines:
        if Requirement is None:
            name = fallback_name(req_line)
            if not name or should_skip(name) or "extra ==" in req_line:
                continue
            requirement = req_line
        else:
            req = Requirement(req_line)
            if should_skip(req.name):
                continue
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            requirement = str(req)
        if requirement not in seen:
            seen.add(requirement)
            requirements.append(requirement)

path = pathlib.Path(os.environ["VENV_RUNTIME_REQS"])
path.write_text("\n".join(requirements) + ("\n" if requirements else ""))
print(f"[INFO] Prepared {len(requirements)} vLLM runtime dependency requirement(s)")
PY
    if [ -s "$VLLM_RUNTIME_REQS" ]; then
        echo "[INFO] Installing vLLM runtime deps from wheel metadata (excluding Ascend torch stack) ..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$VENV/bin/python" --upgrade \
              -r "$VLLM_RUNTIME_REQS"
        else
            "$VENV/bin/python" -m pip install --upgrade \
              -r "$VLLM_RUNTIME_REQS"
        fi
    fi

    unset -f check_local_vllm_ascend

    # ── 3. Apply KV-tracking patches to project venv (idempotent) ────────────
    echo "[INFO] Applying patches to project venv..."
    "$VENV/bin/python" "$REPO_DIR/src/vllm_patches/apply_patches.py" \
        --vllm-dir "$VENV_VLLM"

    # ── 4. Verify the project venv imports the patched copy ──────────────────
    echo "[INFO] Verifying patched vLLM import path..."
    "$VENV/bin/python" - <<'PY'
import vllm.entrypoints.openai.protocol as protocol
import vllm.entrypoints.openai.serving_chat as serving_chat

assert "kv_blocks_used" in protocol.UsageInfo.model_fields
assert "kv_blocks_size_gb" in protocol.UsageInfo.model_fields
assert "agent_id" in protocol.ChatCompletionRequest.model_fields

print(f"[INFO] protocol.py -> {protocol.__file__}")
print(f"[INFO] serving_chat.py -> {serving_chat.__file__}")
PY

    # ── 5. Skip if already running ────────────────────────────────────────────
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "[INFO] vLLM already running (PID $(cat $PID_FILE)). Skipping launch."
        return
    fi

    # ── 6. Launch vllm serve natively ────────────────────────────────────────
    echo "[INFO] Starting native vLLM (model=$(basename $MODEL), tp=$TP, port=$PORT)..."
    (
        set +u
        source "$TOOLKIT"
        source "$ATB"
        set -u
        source "$VENV/bin/activate"
        VLLM_CLI="$VENV/bin/vllm"
        if [ ! -f "$VLLM_CLI" ]; then
            echo "[ERROR] Cannot find vLLM CLI script at $VLLM_CLI"
            exit 1
        fi
        cd "$WORKDIR"
        unset VLLM_USE_MODELSCOPE MODELSCOPE_ENVIRONMENT
        export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
        export OMP_NUM_THREADS=1

        "$VENV/bin/python" "$VLLM_CLI" serve "$MODEL" \
          --served-model-name "$SERVED_NAME" \
          --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
          --tensor-parallel-size "$TP" \
          --dtype "$DTYPE" \
          --kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"num_cpu_blocks": 8192, "caching_hash_algo": "sha256_cbor", "spec_name": "NPUOffloadingSpec", "spec_module_path": "vllm_ascend.kv_offload.npu"}}' \
          $EXTRA
    ) > /dev/null 2>&1 &
    echo $! > "$PID_FILE"
    echo "[INFO] vLLM PID: $(cat $PID_FILE)"
}

# =============================================================================
# MAIN
# =============================================================================
HOST_PORT=$(cfg native.port)
VLLM_API_KEY=$(cfg native.api_key)
start_native

wait_for_ready || exit 1

# Export sidecar geometry for callers of run_abc_bench_instrumented.py
SC_NUM_LAYERS=$(cfg sidecar.num_layers)
SC_NUM_KV_HEADS=$(cfg sidecar.num_kv_heads)
SC_HEAD_DIM=$(cfg sidecar.head_dim)
SC_BLOCK_SIZE=$(cfg sidecar.block_size)
SC_DTYPE=$(cfg sidecar.dtype)

export SIDECAR_NUM_LAYERS="$SC_NUM_LAYERS"
export SIDECAR_NUM_KV_HEADS="$SC_NUM_KV_HEADS"
export SIDECAR_HEAD_DIM="$SC_HEAD_DIM"
export SIDECAR_BLOCK_SIZE="$SC_BLOCK_SIZE"
export SIDECAR_DTYPE="$SC_DTYPE"
export SIDECAR_VLLM_URL="http://localhost:${HOST_PORT}"
echo "[INFO] Sidecar env: layers=${SC_NUM_LAYERS} kv_heads=${SC_NUM_KV_HEADS} head_dim=${SC_HEAD_DIM} block_size=${SC_BLOCK_SIZE} dtype=${SC_DTYPE}"
