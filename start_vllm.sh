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

MAX_RETRIES=60
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
# Creates a project-local uv venv that inherits from the shared venv but
# keeps a patched copy of the 2 affected vLLM files in its own site-packages.
# The shared /opt/vllm/venv is never modified.
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
    local PROC_NAME=$(cfg native.process_name)
    local PID_FILE=$(cfg native.pid_file)

    REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
    VENV="$REPO_DIR/.venv"

    # ── 1. Create project venv via uv (skipped if already exists) ────────────
    if [ ! -f "$VENV/bin/activate" ]; then
        echo "[INFO] Creating project venv with uv..."
        uv venv "$VENV" --python "$SHARED_PY" --system-site-packages
    fi

    # ── 2. Copy the 2 affected vLLM files into project venv site-packages ────
    PYTHON_VER=$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    # Resolve vllm's actual location via Python import — handles editable/empty installs
    # where files are NOT under the venv's own site-packages/vllm/ tree.
    SHARED_OAI=$("$SHARED_PY" -c \
        "import os, vllm.entrypoints.openai as m; print(os.path.dirname(m.__file__))" \
        2>/dev/null) || {
        echo "[ERROR] Cannot locate vllm.entrypoints.openai via $SHARED_PY"
        echo "        Is vllm installed in the shared venv? (checked: $SHARED_PY)"
        exit 1
    }
    VENV_VLLM="$VENV/lib/python${PYTHON_VER}/site-packages/vllm"
    VENV_OAI="$VENV_VLLM/entrypoints/openai"
    mkdir -p "$VENV_OAI"

    for f in protocol.py serving_chat.py; do
        SRC="$SHARED_OAI/$f"
        DST="$VENV_OAI/$f"
        # Recopy if shared vLLM is newer (e.g. after an upgrade)
        if [ ! -f "$DST" ] || [ "$SRC" -nt "$DST" ]; then
            cp "$SRC" "$DST"
            echo "[INFO] Copied $f from shared venv into project venv"
        fi
    done
    # Stub __init__.py files so Python treats the dirs as packages
    touch "$VENV_VLLM/__init__.py" \
          "$VENV_VLLM/entrypoints/__init__.py" \
          "$VENV_OAI/__init__.py"

    # ── 3. Apply KV-tracking + process-name patches to project venv (idempotent) ─
    echo "[INFO] Applying patches to project venv..."
    SHARED_SITE="$(dirname "$(dirname "$(dirname "$SHARED_OAI")")")"
    "$VENV/bin/python" "$REPO_DIR/src/vllm_patches/apply_patches.py" \
        --vllm-dir "$VENV_VLLM" \
        --shared-site "$SHARED_SITE" \
        --process-name "$PROC_NAME"

    # ── 4. Skip if already running ────────────────────────────────────────────
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "[INFO] vLLM already running (PID $(cat $PID_FILE)). Skipping launch."
        return
    fi

    # ── 5. Launch vllm serve natively ────────────────────────────────────────
    echo "[INFO] Starting native vLLM (model=$(basename $MODEL), tp=$TP, port=$PORT)..."
    (
        set +u
        source "$TOOLKIT"
        source "$ATB"
        set -u
        source "$VENV/bin/activate"
        # vllm executable lives in the shared venv's bin, not in the project venv.
        # Prepend it so `vllm` is found while Python still imports from .venv first.
        export PATH="$(dirname "$SHARED_PY"):$PATH"
        cd "$WORKDIR"
        unset VLLM_USE_MODELSCOPE MODELSCOPE_ENVIRONMENT
        export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
        export OMP_NUM_THREADS=1
        export VLLM_PROCESS_NAME="$PROC_NAME"

        vllm serve "$MODEL" \
          --served-model-name "$SERVED_NAME" \
          --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
          --tensor-parallel-size "$TP" \
          --dtype "$DTYPE" \
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
