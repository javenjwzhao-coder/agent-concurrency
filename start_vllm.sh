#!/bin/bash
# start_reasoning.sh — Start vLLM reasoning on Ascend NPU, then run mini-swe-agent
# Usage: ./start_reasoning.sh [config.yaml]
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

CONTAINER_NAME=$(cfg reasoning.container_name)
IMAGE=$(cfg reasoning.image)
HOST_PORT=$(cfg reasoning.host_port)
MODEL=$(cfg reasoning.model)
MODEL_PATH=$(cfg reasoning.model_path)
TENSOR_PARALLEL=$(cfg reasoning.tensor_parallel)
DTYPE=$(cfg reasoning.dtype)
MEM=$(cfg reasoning.mem)
SHM_SIZE=$(cfg reasoning.shm_size)
EXTRA_ARGS=$(cfg reasoning.extra_args)
DEVICES=$(cfg reasoning.devices)

CPUSET_CPUS=$(cfg numa.cpuset_cpus)
CPUSET_MEMS=$(cfg numa.cpuset_mems)
CPU_AFFINITY=$(cfg numa.cpu_affinity)
OMP_NUM_THREADS=$(cfg numa.omp_num_threads)

MAX_RETRIES=60
RETRY_INTERVAL=10

# =============================================================================
# 1.4. BUILD BASE IMAGE IF NOT ALREADY PRESENT
# =============================================================================
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_IMAGE=$(grep -m1 '^FROM ' "${REPO_DIR}/Dockerfile.kv_tracking" | awk '{print $2}')
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    echo "[INFO] Base image '$BASE_IMAGE' not found locally — building from source..."
    docker build \
        -f "${REPO_DIR}/Dockerfile.vllm_ascend_base" \
        -t "$BASE_IMAGE" \
        "${REPO_DIR}/src"
    echo "[INFO] Base image build complete: $BASE_IMAGE"
else
    echo "[INFO] Base image already present: $BASE_IMAGE"
fi

# =============================================================================
# 1.5. BUILD PATCHED IMAGE IF NOT ALREADY PRESENT
# =============================================================================
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

# =============================================================================
# 2. BUILD NPU DEVICE FLAGS
# =============================================================================
build_npu_flags() {
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

# =============================================================================
# 3. HEALTH CHECK
# =============================================================================
wait_for_ready() {
    echo "[INFO] Waiting for vLLM on port ${HOST_PORT}..."
    for i in $(seq 1 "$MAX_RETRIES"); do
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
            "http://localhost:${HOST_PORT}/v1/models" 2>/dev/null || echo "000")
        if [ "$code" = "200" ]; then
            echo "[INFO] vLLM reasoning is READY."
            return 0
        fi
        [ $((i % 3)) -eq 0 ] && echo "[INFO]   attempt ${i}/${MAX_RETRIES}, status=${code}"
        sleep "$RETRY_INTERVAL"
    done
    echo "[ERROR] Timed out. Last status: ${code}"
    echo "[ERROR] Container logs (last 30 lines):"
    docker logs --tail 30 "$CONTAINER_NAME" 2>&1 || true
    return 1
}

# =============================================================================
# 4. START VLLM CONTAINER
# =============================================================================
if docker ps --format '{{.Names}}' | grep -qw "$CONTAINER_NAME"; then
    echo "[INFO] Container '$CONTAINER_NAME' already running. Skipping launch."
else
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    METRICS_DIR="/tmp/vllm_metrics_${CONTAINER_NAME}"
    mkdir -p "$METRICS_DIR" && rm -rf "${METRICS_DIR:?}"/*

    NPU_FLAGS=$(build_npu_flags)

    echo "[INFO] Starting $CONTAINER_NAME (model=$MODEL, tp=$TENSOR_PARALLEL, port=$HOST_PORT)..."

    docker run -d --name "$CONTAINER_NAME" \
        $NPU_FLAGS \
        --shm-size "$SHM_SIZE" \
        --cpuset-cpus="$CPUSET_CPUS" \
        --cpuset-mems="$CPUSET_MEMS" \
        --memory="$MEM" \
        --memory-swap="$MEM" \
        -v "${MODEL_PATH}:${MODEL_PATH}" \
        -v "${METRICS_DIR}:/tmp/vllm_metrics" \
        -p "${HOST_PORT}:8000" \
        -e HF_HUB_OFFLINE=1 \
        -e CPU_AFFINITY_CONF="$CPU_AFFINITY" \
        -e TASK_QUEUE_ENABLE=1 \
        -e OMP_NUM_THREADS="$OMP_NUM_THREADS" \
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
fi

wait_for_ready || exit 1