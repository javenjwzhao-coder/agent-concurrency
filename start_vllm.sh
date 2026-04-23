#!/bin/bash
# start_vllm.sh — Start vLLM on Ascend NPU (native bare-metal or Docker)
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

cfg() {
    python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
keys = '$1'.split('.')
v = c
for k in keys:
    v = v[k]
if isinstance(v, list):
    print(' '.join(str(i) for i in v))
elif v is None:
    print('')
else:
    print(v)
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
    SHARED_OAI="$(dirname "$SHARED_PY")/../lib/python${PYTHON_VER}/site-packages/vllm/entrypoints/openai"
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

    # ── 3. Apply KV-tracking patches to project venv copies (idempotent) ─────
    echo "[INFO] Applying KV-tracking patch to project venv..."
    "$VENV/bin/python" "$REPO_DIR/src/vllm_patches/apply_patches.py" \
        --vllm-dir "$VENV_VLLM"

    # ── 4. Skip if already running ────────────────────────────────────────────
    if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
        echo "[INFO] vLLM already running (PID $(cat $PID_FILE)). Skipping launch."
        return
    fi

    # ── 5. Launch vllm serve natively ────────────────────────────────────────
    echo "[INFO] Starting native vLLM (model=$(basename $MODEL), tp=$TP, port=$PORT)..."
    (
        source "$TOOLKIT"
        source "$ATB"
        source "$VENV/bin/activate"
        cd "$WORKDIR"
        unset VLLM_USE_MODELSCOPE MODELSCOPE_ENVIRONMENT
        export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
        export OMP_NUM_THREADS=1

        vllm serve "$MODEL" \
          --served-model-name "$SERVED_NAME" \
          --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
          --tensor-parallel-size "$TP" \
          --dtype "$DTYPE" \
          $EXTRA
    ) &
    echo $! > "$PID_FILE"
    echo "[INFO] vLLM PID: $(cat $PID_FILE)"
}

# =============================================================================
# DOCKER LAUNCH (original path — unchanged)
# =============================================================================
build_npu_flags() {
    local DEVICES=$(cfg reasoning.devices)
    local flags="" idx=0
    for dev in $DEVICES; do
        flags+=" --device ${dev}:/dev/davinci${idx}"
        idx=$((idx + 1))
    done
    flags+=" --device /dev/davinci_manager:/dev/davinci_manager"
    flags+=" --device /dev/devmm_svm:/dev/devmm_svm"
    flags+=" --device /dev/hisi_hdc:/dev/hisi_hdc"
    flags+=" -v /usr/local/dcmi:/usr/local/dcmi"
    flags+=" -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi"
    flags+=" -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/"
    flags+=" -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info"
    flags+=" -v /etc/ascend_install.info:/etc/ascend_install.info"
    echo "$flags"
}

start_docker() {
    local CONTAINER_NAME=$(cfg reasoning.container_name)
    local IMAGE=$(cfg reasoning.image)
    local MODEL=$(cfg reasoning.model)
    local MODEL_PATH=$(cfg reasoning.model_path)
    local TENSOR_PARALLEL=$(cfg reasoning.tensor_parallel)
    local DTYPE=$(cfg reasoning.dtype)
    local MEM=$(cfg reasoning.mem)
    local SHM_SIZE=$(cfg reasoning.shm_size)
    local EXTRA_ARGS=$(cfg reasoning.extra_args)

    REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        echo "[INFO] Patched image '$IMAGE' not found locally — building..."
        docker build \
            -f "${REPO_DIR}/Dockerfile.kv_tracking" \
            -t "$IMAGE" \
            "${REPO_DIR}/src"
        echo "[INFO] Build complete: $IMAGE"
    else
        echo "[INFO] Patched image already present: $IMAGE"
    fi

    if docker ps --format '{{.Names}}' | grep -qw "$CONTAINER_NAME"; then
        echo "[INFO] Container '$CONTAINER_NAME' already running. Skipping launch."
        return
    fi

    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    METRICS_DIR="/tmp/vllm_metrics_${CONTAINER_NAME}"
    mkdir -p "$METRICS_DIR" && rm -rf "${METRICS_DIR:?}"/*
    NPU_FLAGS=$(build_npu_flags)

    echo "[INFO] Starting $CONTAINER_NAME (model=$MODEL, tp=$TENSOR_PARALLEL, port=$HOST_PORT)..."
    docker run -d --name "$CONTAINER_NAME" \
        $NPU_FLAGS \
        --shm-size "$SHM_SIZE" \
        --memory="$MEM" \
        --memory-swap="$MEM" \
        -v "${MODEL_PATH}:${MODEL_PATH}" \
        -v "${METRICS_DIR}:/tmp/vllm_metrics" \
        -p "${HOST_PORT}:8000" \
        -e HF_HUB_OFFLINE=1 \
        -e TASK_QUEUE_ENABLE=1 \
        -e LCCL_DETERMINISTIC=1 \
        -e HCCL_DETERMINISTIC=true \
        -e ATB_MATMUL_SHUFFLE_K_ENABLE=0 \
        -e ATB_LLM_LCOC_ENABLE=0 \
        -e PROMETHEUS_MULTIPROC_DIR=/tmp/vllm_metrics \
        -e no_proxy="localhost,127.0.0.1" \
        "$IMAGE" \
        vllm serve "$MODEL" \
        --dtype "$DTYPE" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"num_cpu_blocks": 8192, "caching_hash_algo": "sha256_cbor", "spec_name": "NPUOffloadingSpec", "spec_module_path": "vllm_ascend.kv_offload.npu"}}' \
        $EXTRA_ARGS
}

# =============================================================================
# MAIN — mode dispatch
# =============================================================================
NATIVE_ENABLED=$(cfg native.enabled 2>/dev/null || echo "false")

if [ "$NATIVE_ENABLED" = "True" ] || [ "$NATIVE_ENABLED" = "true" ]; then
    HOST_PORT=$(cfg native.port)
    start_native
else
    HOST_PORT=$(cfg reasoning.host_port)
    start_docker
fi

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
