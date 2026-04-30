#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_with_config.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launch the instrumented ABC-Bench runner from a single YAML config file.
#
# Features:
#   • Reads all parameters from abc_bench_config.yaml (or any --config path).
#   • ${ENV_VAR} references in YAML values are expanded at runtime.
#   • --override key.path=value for quick tweaks without editing YAML.
#   • Validates the config before launching (fails fast on missing fields).
#   • --dry-run prints the resolved commands without executing.
#   • Optionally runs build_tool_predictor.py after agents finish.
#
# Usage:
#   bash run_with_config.sh --config abc_bench_config.yaml
#   bash run_with_config.sh --config abc_bench_config.yaml --dry-run
#   bash run_with_config.sh --config abc_bench_config.yaml \
#       --override dataset.max_tasks=2 \
#       --override launch.randomise=false
#
# Requirements:
#   • Python 3 with PyYAML installed (used only for YAML parsing)
#   • run_abc_bench_instrumented.py in the same directory as this script
#   • src/build_tool_predictor.py (if prediction enabled)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─────────────────────────── defaults ────────────────────────────────────────

CONFIG_FILE="${SCRIPT_DIR}/config/abc-bench_config.yaml"
DRY_RUN=false
SKIP_PREDICTION=false
BENCH_UV_HTTP_TIMEOUT="${BENCH_UV_HTTP_TIMEOUT:-300}"
BENCH_UV_INSTALL_RETRIES="${BENCH_UV_INSTALL_RETRIES:-3}"
declare -a OVERRIDES=()

# ─────────────────────────── colours ─────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ─────────────────────────── argument parsing ────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") --config <config.yaml> [OPTIONS]

Options:
  --config, -c FILE       Path to YAML config file (required)
  --override, -o K=V      Override a config value (repeatable)
                           Examples: dataset.max_tasks=2  launch.randomise=false
  --dry-run               Print resolved config and commands without executing
  --skip-prediction       Skip the prediction step even if enabled in config
  --help, -h              Show this help

Examples:
  $(basename "$0") --config abc_bench_config.yaml
  $(basename "$0") -c abc_bench_config.yaml --dry-run
  $(basename "$0") -c abc_bench_config.yaml -o dataset.max_tasks=2 -o launch.randomise=false
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-c)
            CONFIG_FILE="$2"; shift 2 ;;
        --override|-o)
            OVERRIDES+=("$2"); shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --skip-prediction)
            SKIP_PREDICTION=true; shift ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            echo -e "${RED}ERROR: unknown argument: $1${RESET}" >&2
            usage >&2; exit 2 ;;
    esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}ERROR: config file not found: ${CONFIG_FILE}${RESET}" >&2
    exit 2
fi

# ─────────────────────────── YAML parser ─────────────────────────────────────
# We use a small inline Python script to:
#   1. Parse the YAML
#   2. Apply --override patches
#   3. Expand ${ENV_VAR} references
#   4. Emit the resolved values as KEY=VALUE lines for bash to eval
#
# This avoids requiring yq or other external tools beyond python3 + PyYAML.

parse_config() {
    local config_file="$1"
    shift
    # Remaining args are override strings: "key.path=value" ...

    python3 - "$config_file" "$@" <<'PYEOF'
import os, re, sys, yaml

def expand_env(val):
    """Replace ${VAR} with environment variable value."""
    if not isinstance(val, str):
        return val
    def _sub(m):
        v = os.environ.get(m.group(1))
        if v is None:
            print(f"ERROR: Environment variable ${{{m.group(1)}}} referenced "
                  f"in config but not set.", file=sys.stderr)
            sys.exit(2)
        return v
    return re.sub(r"\$\{(\w+)\}", _sub, val)

def walk_expand(obj):
    if isinstance(obj, str):
        return expand_env(obj)
    if isinstance(obj, dict):
        return {k: walk_expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk_expand(v) for v in obj]
    return obj

def auto_cast(s):
    if s.lower() == "true": return True
    if s.lower() == "false": return False
    if s.lower() in ("null", "none"): return None
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: pass
    return s

def set_nested(d, dotted_key, value):
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

def get_nested(d, dotted_key, default=""):
    keys = dotted_key.split(".")
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    if d is None:
        return ""
    return d

config_file = sys.argv[1]
overrides = sys.argv[2:]

with open(config_file) as f:
    cfg = yaml.safe_load(f) or {}

# Apply overrides before env expansion so overrides can contain ${VAR} too.
for ov in overrides:
    if "=" not in ov:
        print(f"ERROR: override must be key=value, got: {ov!r}", file=sys.stderr)
        sys.exit(2)
    k, v = ov.split("=", 1)
    set_nested(cfg, k, auto_cast(v))

cfg = walk_expand(cfg)

# Flatten to KEY=VALUE lines that bash can eval.
# Keys use underscore-separated uppercase: llm.model -> LLM_MODEL
flat_keys = [
    ("LLM_MODEL",              "llm.model"),
    ("LLM_BASE_URL",           "llm.base_url"),
    ("LLM_API_KEY",            "llm.api_key"),
    ("DATASET_ROOT",           "dataset.root"),
    ("TASK_GLOB",              "dataset.task_glob"),
    ("MAX_TASKS",              "dataset.max_tasks"),
    ("WORKSPACE_ROOT",         "paths.workspace_root"),
    ("RESULTS_ROOT",           "paths.results_root"),
    ("MAX_ITERATIONS",         "agent.max_iterations"),
    ("LAUNCH_RANDOMISE",       "launch.randomise"),
    ("MAX_AGENTS_PER_WAVE",    "launch.max_agents_per_wave"),
    ("MIN_WAVE_DELAY_S",       "launch.min_wave_delay_s"),
    ("MAX_WAVE_DELAY_S",       "launch.max_wave_delay_s"),
    ("LAUNCH_SEED",            "launch.seed"),
    ("TESTING_ENABLED",        "testing.enabled"),
    ("TEST_TIMEOUT_SEC",       "testing.timeout_sec"),
    ("PREDICTION_ENABLED",     "prediction.enabled"),
    ("PREDICTION_MODEL",       "prediction.model"),
    ("PREDICTION_TEST_FRAC",   "prediction.test_fraction"),
    ("PREDICTION_SAVE_MODEL",  "prediction.save_model"),
    ("PREDICTION_SEED",        "prediction.seed"),
    ("PREDICTION_LONG_THRESHOLD_S", "prediction.long_call_threshold_s"),
    ("LOG_LEVEL",              "logging.level"),
    ("LOG_FILE",               "logging.file"),
    ("SIDECAR_ENABLED",        "sidecar.enabled"),
    ("SIDECAR_LOG_FILE",       "sidecar.log_file"),
    ("SIDECAR_VLLM_URL",       "sidecar.vllm_url"),
    ("SIDECAR_INTERVAL",       "sidecar.interval"),
    ("SIDECAR_NUM_LAYERS",     "sidecar.num_layers"),
    ("SIDECAR_NUM_KV_HEADS",   "sidecar.num_kv_heads"),
    ("SIDECAR_HEAD_DIM",       "sidecar.head_dim"),
    ("SIDECAR_BLOCK_SIZE",       "sidecar.block_size"),
    ("SIDECAR_DTYPE",            "sidecar.dtype"),
    ("SIDECAR_TOTAL_GPU_BLOCKS", "sidecar.total_gpu_blocks"),
    ("SIDECAR_HTTP_PORT",        "sidecar.http_port"),
    ("SIDECAR_HTTP_HOST",        "sidecar.http_host"),
    ("SIDECAR_ADMISSION_ENABLED", "sidecar.admission_control.enabled"),
    ("SIDECAR_ADMISSION_THRESHOLD_GB", "sidecar.admission_control.threshold_gb"),
    ("SIDECAR_INITIAL_ADMIT_INTERVAL_S", "sidecar.admission_control.initial_admit_interval_s"),
    ("SIDECAR_MAX_FRESH_ADMITS_PER_TICK", "sidecar.admission_control.max_fresh_admits_per_tick"),
    ("SIDECAR_SHORT_TOOL_CALL_THRESHOLD_S", "sidecar.admission_control.short_tool_call_threshold_s"),
    ("SIDECAR_FALLBACK_LONG_TOOL_CALL_S", "sidecar.admission_control.fallback_long_tool_call_s"),
    ("SIDECAR_ADMISSION_PREDICTOR_MODEL", "sidecar.admission_control.predictor_model"),
    ("SIDECAR_OFFLOAD_ENDPOINT", "sidecar.admission_control.offload_endpoint"),
    ("SIDECAR_RESTORE_ENDPOINT", "sidecar.admission_control.restore_endpoint"),
    ("SIDECAR_EVICTION_ENDPOINT", "sidecar.admission_control.eviction_endpoint"),
    ("SIDECAR_EVICTION_TIMEOUT_S", "sidecar.admission_control.eviction_timeout_s"),
]

for bash_name, dotted in flat_keys:
    val = get_nested(cfg, dotted, "")
    # Booleans → lowercase for bash [[ ]] comparisons.
    if isinstance(val, bool):
        val = "true" if val else "false"
    print(f"{bash_name}={val}")

PYEOF
}

# ─────────────────────────── load config ─────────────────────────────────────

echo -e "${CYAN}Loading config: ${CONFIG_FILE}${RESET}"

# Parse YAML → bash variables.
CONFIG_LINES=$(parse_config "$CONFIG_FILE" "${OVERRIDES[@]+"${OVERRIDES[@]}"}")
parse_exit=$?
if [[ $parse_exit -ne 0 ]]; then
    exit $parse_exit
fi

# Source the KEY=VALUE lines into this shell.
while IFS='=' read -r key value; do
    # Use declare to avoid issues with special characters in values.
    declare "$key=$value"
done <<< "$CONFIG_LINES"

derive_sidecar_url() {
    local url="${1%/}"
    if [[ "$url" == */v1 ]]; then
        url="${url%/v1}"
    fi
    printf '%s\n' "$url"
}

if [[ -z "${SIDECAR_VLLM_URL:-}" && -n "${LLM_BASE_URL:-}" ]]; then
    SIDECAR_VLLM_URL="$(derive_sidecar_url "$LLM_BASE_URL")"
fi

# ─────────────────────────── validation ──────────────────────────────────────

ERRORS=()

[[ -z "${LLM_MODEL:-}" ]]    && ERRORS+=("llm.model is required")
[[ -z "${LLM_BASE_URL:-}" ]] && ERRORS+=("llm.base_url is required")
[[ -z "${LLM_API_KEY:-}" ]]  && ERRORS+=("llm.api_key is required")
[[ -z "${DATASET_ROOT:-}" ]] && ERRORS+=("dataset.root is required")

if [[ -n "${DATASET_ROOT:-}" && ! -d "$DATASET_ROOT" ]]; then
    ERRORS+=("dataset.root does not exist: ${DATASET_ROOT}")
fi

if [[ -n "${MIN_WAVE_DELAY_S:-}" && -n "${MAX_WAVE_DELAY_S:-}" ]]; then
    if python3 -c "exit(0 if float('${MIN_WAVE_DELAY_S}') <= float('${MAX_WAVE_DELAY_S}') else 1)" 2>/dev/null; then
        : # ok
    else
        ERRORS+=("launch.min_wave_delay_s (${MIN_WAVE_DELAY_S}) must be <= launch.max_wave_delay_s (${MAX_WAVE_DELAY_S})")
    fi
fi

if [[ -z "${SIDECAR_ADMISSION_PREDICTOR_MODEL:-}" ]]; then
    SIDECAR_ADMISSION_PREDICTOR_MODEL="${PREDICTION_SAVE_MODEL:-}"
fi
if [[ "${SIDECAR_ADMISSION_ENABLED:-false}" == "true" && "${SIDECAR_ENABLED:-false}" != "true" ]]; then
    ERRORS+=("sidecar.admission_control.enabled requires sidecar.enabled=true")
fi

RUNNER_SCRIPT="${SCRIPT_DIR}/src/run_abc_bench_instrumented.py"
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
    ERRORS+=("Runner script not found: ${RUNNER_SCRIPT}")
fi

PREDICTOR_SCRIPT="${SCRIPT_DIR}/src/build_tool_predictor.py"

if [[ ${#ERRORS[@]} -gt 0 ]]; then
    echo -e "\n${RED}Config validation failed:${RESET}" >&2
    for err in "${ERRORS[@]}"; do
        echo -e "  ${RED}•${RESET} $err" >&2
    done
    exit 2
fi

# ─────────────────────────── build runner command ────────────────────────────

build_runner_cmd() {
    local -a cmd=(
        python3 "$RUNNER_SCRIPT"
        --dataset-root  "$DATASET_ROOT"
        --task-glob     "${TASK_GLOB:-task_*}"
        --max-tasks     "${MAX_TASKS:-1}"
        --workspace-root "${WORKSPACE_ROOT:-./abc_runs}"
        --results-root  "${RESULTS_ROOT:-./abc_results}"
        --model         "$LLM_MODEL"
        --base-url      "$LLM_BASE_URL"
        --api-key       "$LLM_API_KEY"
        --max-iterations "${MAX_ITERATIONS:-80}"
    )

    if [[ "${LAUNCH_RANDOMISE:-false}" == "true" ]]; then
        cmd+=(
            --randomise-launch
            --max-agents-per-wave "${MAX_AGENTS_PER_WAVE:-4}"
            --min-wave-delay-s    "${MIN_WAVE_DELAY_S:-2.0}"
            --max-wave-delay-s    "${MAX_WAVE_DELAY_S:-15.0}"
        )
        if [[ -n "${LAUNCH_SEED:-}" && "${LAUNCH_SEED}" != "None" ]]; then
            cmd+=(--launch-seed "$LAUNCH_SEED")
        fi
    fi

    if [[ "${TESTING_ENABLED:-false}" == "true" ]]; then
        cmd+=(
            --run-tests
            --test-timeout-sec "${TEST_TIMEOUT_SEC:-1800}"
        )
    fi

    if [[ "${SIDECAR_ENABLED:-false}" == "true" ]]; then
        _sc_log="${SIDECAR_LOG_FILE:-${RESULTS_ROOT:-./abc_results}/sidecar.log}"
        cmd+=(
            --sidecar-log-file     "$_sc_log"
            --sidecar-vllm-url     "${SIDECAR_VLLM_URL:-http://localhost:8000}"
            --sidecar-interval     "${SIDECAR_INTERVAL:-5.0}"
            --sidecar-num-layers   "${SIDECAR_NUM_LAYERS}"
            --sidecar-num-kv-heads "${SIDECAR_NUM_KV_HEADS}"
            --sidecar-head-dim     "${SIDECAR_HEAD_DIM}"
            --sidecar-block-size        "${SIDECAR_BLOCK_SIZE:-16}"
            --sidecar-dtype             "${SIDECAR_DTYPE:-bfloat16}"
            --sidecar-total-gpu-blocks  "${SIDECAR_TOTAL_GPU_BLOCKS:-0}"
        )
        if [[ -n "${SIDECAR_HTTP_PORT:-}" && "${SIDECAR_HTTP_PORT}" != "0" && "${SIDECAR_HTTP_PORT}" != "" ]]; then
            cmd+=(
                --sidecar-http-port "$SIDECAR_HTTP_PORT"
                --sidecar-http-host "${SIDECAR_HTTP_HOST:-127.0.0.1}"
            )
        fi
        if [[ "${SIDECAR_ADMISSION_ENABLED:-false}" == "true" ]]; then
            cmd+=(
                --sidecar-admission-control
                --sidecar-admission-threshold-gb "${SIDECAR_ADMISSION_THRESHOLD_GB:-0.1}"
                --sidecar-initial-admit-interval-s "${SIDECAR_INITIAL_ADMIT_INTERVAL_S:-2.0}"
                --sidecar-max-fresh-admits-per-tick "${SIDECAR_MAX_FRESH_ADMITS_PER_TICK:-1}"
                --sidecar-short-tool-call-threshold-s "${SIDECAR_SHORT_TOOL_CALL_THRESHOLD_S:-2.0}"
                --sidecar-fallback-long-tool-call-s "${SIDECAR_FALLBACK_LONG_TOOL_CALL_S:-30.0}"
                --sidecar-eviction-timeout-s "${SIDECAR_EVICTION_TIMEOUT_S:-2.0}"
            )
            if [[ -n "${SIDECAR_ADMISSION_PREDICTOR_MODEL:-}" ]]; then
                cmd+=(--sidecar-admission-predictor-model "$SIDECAR_ADMISSION_PREDICTOR_MODEL")
            fi
            if [[ -n "${SIDECAR_OFFLOAD_ENDPOINT:-}" ]]; then
                cmd+=(--sidecar-offload-endpoint "$SIDECAR_OFFLOAD_ENDPOINT")
            fi
            if [[ -n "${SIDECAR_RESTORE_ENDPOINT:-}" ]]; then
                cmd+=(--sidecar-restore-endpoint "$SIDECAR_RESTORE_ENDPOINT")
            fi
            if [[ -n "${SIDECAR_EVICTION_ENDPOINT:-}" ]]; then
                cmd+=(--sidecar-eviction-endpoint "$SIDECAR_EVICTION_ENDPOINT")
            fi
        fi
    fi

    echo "${cmd[@]}"
}

build_predictor_cmd() {
    local -a cmd=(
        python3 "$PREDICTOR_SCRIPT"
        --trace-dir      "${RESULTS_ROOT:-./abc_results}"
        --model          "${PREDICTION_MODEL:-hgb}"
        --test-fraction  "${PREDICTION_TEST_FRAC:-0.2}"
        --seed           "${PREDICTION_SEED:-42}"
    )
    if [[ -n "${PREDICTION_SAVE_MODEL:-}" ]]; then
        cmd+=(--save-model "$PREDICTION_SAVE_MODEL")
    fi
    if [[ -n "${PREDICTION_LONG_THRESHOLD_S:-}" ]]; then
        cmd+=(--long-threshold "$PREDICTION_LONG_THRESHOLD_S")
    fi
    echo "${cmd[@]}"
}

# ─────────────────────────── display config ──────────────────────────────────

# Redact API key for display.
DISPLAY_KEY="${LLM_API_KEY}"
if [[ ${#DISPLAY_KEY} -gt 8 ]]; then
    DISPLAY_KEY="${DISPLAY_KEY:0:4}****${DISPLAY_KEY: -4}"
fi

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}  Resolved Configuration${RESET}"
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  ${CYAN}LLM${RESET}"
echo "    model:              ${LLM_MODEL}"
echo "    base_url:           ${LLM_BASE_URL}"
echo "    api_key:            ${DISPLAY_KEY}"
echo ""
echo -e "  ${CYAN}Dataset${RESET}"
echo "    root:               ${DATASET_ROOT}"
echo "    task_glob:          ${TASK_GLOB:-task_*}"
echo "    max_tasks:          ${MAX_TASKS:-1}"
echo ""
echo -e "  ${CYAN}Paths${RESET}"
echo "    workspace_root:     ${WORKSPACE_ROOT:-./abc_runs}"
echo "    results_root:       ${RESULTS_ROOT:-./abc_results}"
echo ""
echo -e "  ${CYAN}Agent${RESET}"
echo "    max_iterations:     ${MAX_ITERATIONS:-80}"
echo ""
echo -e "  ${CYAN}Launch${RESET}"
echo "    randomise:          ${LAUNCH_RANDOMISE:-false}"
if [[ "${LAUNCH_RANDOMISE:-false}" == "true" ]]; then
    echo "    max_agents_per_wave: ${MAX_AGENTS_PER_WAVE:-4}"
    echo "    min_wave_delay_s:   ${MIN_WAVE_DELAY_S:-2.0}"
    echo "    max_wave_delay_s:   ${MAX_WAVE_DELAY_S:-15.0}"
    echo "    seed:               ${LAUNCH_SEED:-null}"
fi
echo ""
echo -e "  ${CYAN}Testing${RESET}"
echo "    enabled:            ${TESTING_ENABLED:-false}"
echo "    timeout_sec:        ${TEST_TIMEOUT_SEC:-1800}"
echo ""
echo -e "  ${CYAN}Prediction${RESET}"
echo "    enabled:            ${PREDICTION_ENABLED:-false}"
echo "    model:              ${PREDICTION_MODEL:-hgb}"
echo "    test_fraction:      ${PREDICTION_TEST_FRAC:-0.2}"
echo "    save_model:         ${PREDICTION_SAVE_MODEL:-}"
echo "    long_threshold_s:   ${PREDICTION_LONG_THRESHOLD_S:-2.0}"
echo ""
echo -e "  ${CYAN}Sidecar (KV cache monitor)${RESET}"
echo "    enabled:            ${SIDECAR_ENABLED:-false}"
if [[ "${SIDECAR_ENABLED:-false}" == "true" ]]; then
    _sc_log_display="${SIDECAR_LOG_FILE:-${RESULTS_ROOT:-./abc_results}/sidecar.log}"
    echo "    log_file:           ${_sc_log_display}"
    echo "    vllm_url:           ${SIDECAR_VLLM_URL:-http://localhost:8000}"
    echo "    interval:           ${SIDECAR_INTERVAL:-5.0}s"
    echo "    geometry:           layers=${SIDECAR_NUM_LAYERS}  kv_heads=${SIDECAR_NUM_KV_HEADS}  head_dim=${SIDECAR_HEAD_DIM}  block=${SIDECAR_BLOCK_SIZE}  dtype=${SIDECAR_DTYPE:-bfloat16}"
    if [[ -n "${SIDECAR_HTTP_PORT:-}" && "${SIDECAR_HTTP_PORT}" != "0" && "${SIDECAR_HTTP_PORT}" != "" ]]; then
        echo "    dashboard:          http://${SIDECAR_HTTP_HOST:-127.0.0.1}:${SIDECAR_HTTP_PORT}/"
    fi
    echo "    admission_control:  ${SIDECAR_ADMISSION_ENABLED:-false}"
    if [[ "${SIDECAR_ADMISSION_ENABLED:-false}" == "true" ]]; then
        echo "    threshold_gb:       ${SIDECAR_ADMISSION_THRESHOLD_GB:-0.1}"
        echo "    initial_admit_s:    ${SIDECAR_INITIAL_ADMIT_INTERVAL_S:-2.0}"
        echo "    fresh_admits/tick:  ${SIDECAR_MAX_FRESH_ADMITS_PER_TICK:-1}"
        echo "    short_tool_call_s:  ${SIDECAR_SHORT_TOOL_CALL_THRESHOLD_S:-2.0}"
        echo "    fallback_long_s:    ${SIDECAR_FALLBACK_LONG_TOOL_CALL_S:-30.0}"
        echo "    predictor_model:    ${SIDECAR_ADMISSION_PREDICTOR_MODEL:-}"
        echo "    offload_endpoint:   ${SIDECAR_OFFLOAD_ENDPOINT:-${SIDECAR_EVICTION_ENDPOINT:-${SIDECAR_VLLM_URL:-http://localhost:8000}/agent_kv_cache/offload}}"
        echo "    restore_endpoint:   ${SIDECAR_RESTORE_ENDPOINT:-${SIDECAR_VLLM_URL:-http://localhost:8000}/agent_kv_cache/restore}"
        echo "    eviction_timeout_s: ${SIDECAR_EVICTION_TIMEOUT_S:-2.0}"
    fi
fi
echo ""
echo -e "${BOLD}════════════════════════════════════════════════════════════════════════${RESET}"

# ─────────────────────────── display commands ────────────────────────────────

RUNNER_CMD=$(build_runner_cmd)
echo ""
echo -e "${BOLD}── Agent runner command ──${RESET}"
echo -e "${GREEN}${RUNNER_CMD}${RESET}" | sed 's/ --/\n    --/g; s/^/  /'
echo ""

RUN_PREDICTION=false
if [[ "${PREDICTION_ENABLED:-false}" == "true" && "$SKIP_PREDICTION" == "false" ]]; then
    RUN_PREDICTION=true
    if [[ ! -f "$PREDICTOR_SCRIPT" ]]; then
        echo -e "${YELLOW}WARNING: prediction enabled but ${PREDICTOR_SCRIPT} not found; skipping.${RESET}"
        RUN_PREDICTION=false
    fi
    # Skip training if a saved predictor model already exists.
    if [[ "$RUN_PREDICTION" == "true" && -n "${PREDICTION_SAVE_MODEL:-}" ]]; then
        _model_path="${PREDICTION_SAVE_MODEL}"
        if [[ "${_model_path}" != /* ]]; then
            _model_path="${SCRIPT_DIR}/${_model_path#./}"
        fi
        if [[ -f "$_model_path" ]]; then
            echo -e "${CYAN}  Predictor model already exists at ${_model_path}; skipping training.${RESET}"
            RUN_PREDICTION=false
        fi
    fi
fi

if [[ "$RUN_PREDICTION" == "true" ]]; then
    PREDICTOR_CMD=$(build_predictor_cmd)
    echo -e "${BOLD}── Prediction model command ──${RESET}"
    echo -e "${GREEN}${PREDICTOR_CMD}${RESET}" | sed 's/ --/\n    --/g; s/^/  /'
    echo ""
fi

# ─────────────────────────── dry run exit ────────────────────────────────────

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "  ${YELLOW}[dry-run] No processes were started.${RESET}"
    exit 0
fi

# ─────────────────────────── ensure vLLM is running ──────────────────────────

VLLM_HEALTH_URL="${LLM_BASE_URL%/}/models"
echo -e "${CYAN}Checking vLLM at ${LLM_BASE_URL} ...${RESET}"
VLLM_HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    -H "Authorization: Bearer ${LLM_API_KEY}" \
    "$VLLM_HEALTH_URL" 2>/dev/null || echo "000")

if [[ "$VLLM_HTTP_CODE" == "200" ]]; then
    echo -e "${GREEN}  vLLM is already running.${RESET}"
else
    echo -e "${YELLOW}  vLLM not responding (HTTP ${VLLM_HTTP_CODE}). Starting via start_vllm.sh ...${RESET}"
    VLLM_START_SCRIPT="${SCRIPT_DIR}/start_vllm.sh"
    if [[ ! -f "$VLLM_START_SCRIPT" ]]; then
        echo -e "${RED}ERROR: start_vllm.sh not found: ${VLLM_START_SCRIPT}${RESET}" >&2
        exit 2
    fi
    bash "$VLLM_START_SCRIPT"
fi

# ── Benchmark venv (Python 3.12, separate from the vLLM .venv) ──────────────
# openhands-sdk requires Python >=3.12; the vLLM .venv must stay on the
# shared Python (3.11) so that torch C extensions load correctly.
bench_dependencies_ready() {
    python3 - <<'PYEOF' >/dev/null 2>&1
import openhands.sdk
import joblib
import numpy
import pandas
import sklearn
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool
PYEOF
}

install_bench_dependencies() {
    local attempt=1
    while (( attempt <= BENCH_UV_INSTALL_RETRIES )); do
        echo -e "${CYAN}  Installing benchmark dependencies (attempt ${attempt}/${BENCH_UV_INSTALL_RETRIES}, UV_HTTP_TIMEOUT=${BENCH_UV_HTTP_TIMEOUT}s)...${RESET}"
        if UV_HTTP_TIMEOUT="${BENCH_UV_HTTP_TIMEOUT}" \
            uv pip install openhands-sdk openhands-tools pandas numpy scikit-learn joblib; then
            return 0
        fi

        if (( attempt == BENCH_UV_INSTALL_RETRIES )); then
            echo -e "${RED}ERROR: failed to install benchmark dependencies after ${BENCH_UV_INSTALL_RETRIES} attempts.${RESET}" >&2
            echo -e "${YELLOW}Hint: rerun with a larger timeout, for example:${RESET} BENCH_UV_HTTP_TIMEOUT=600 bash run_abc-bench.sh --config ...${RESET}" >&2
            return 1
        fi

        echo -e "${YELLOW}  Install attempt ${attempt} failed; retrying in 5s...${RESET}"
        sleep 5
        attempt=$((attempt + 1))
    done
}

BENCH_VENV="${SCRIPT_DIR}/.bench-venv"
if [[ ! -f "${BENCH_VENV}/bin/activate" ]]; then
    echo -e "${CYAN}  Creating benchmark venv at ${BENCH_VENV} (Python 3.12)...${RESET}"
    uv venv "${BENCH_VENV}" --python 3.12
fi
source "${BENCH_VENV}/bin/activate"
if bench_dependencies_ready; then
    echo -e "${GREEN}  Benchmark dependencies already available in ${BENCH_VENV}.${RESET}"
else
    install_bench_dependencies
fi
echo -e "${CYAN}  Using $(python3 --version) from ${BENCH_VENV}${RESET}"

# ─────────────────────────── execute ─────────────────────────────────────────

# Create output directories.
mkdir -p "${WORKSPACE_ROOT:-./abc_runs}"
mkdir -p "${RESULTS_ROOT:-./abc_results}"

# Set up logging to file if configured.
if [[ -n "${LOG_FILE:-}" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Logging to: ${LOG_FILE}"
fi

START_TS=$(date -u +"%Y-%m-%dT%H:%M:%S+00:00")
START_EPOCH=$(date +%s)

echo ""
echo -e "${GREEN}▶ Starting agent runner at ${START_TS}${RESET}"
echo ""

# Run the agent runner.
eval "$RUNNER_CMD"
RUNNER_EXIT=$?

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))

if [[ $RUNNER_EXIT -ne 0 ]]; then
    echo -e "${RED}✗ Agent runner exited with code ${RUNNER_EXIT} after ${ELAPSED}s${RESET}" >&2
    exit $RUNNER_EXIT
fi

echo ""
echo -e "${GREEN}✓ Agent runner completed in ${ELAPSED}s${RESET}"

# Run prediction model if enabled.
if [[ "$RUN_PREDICTION" == "true" ]]; then
    echo ""
    echo -e "${GREEN}▶ Starting prediction model${RESET}"
    echo ""
    eval "$PREDICTOR_CMD"
    PRED_EXIT=$?
    if [[ $PRED_EXIT -ne 0 ]]; then
        echo -e "${RED}✗ Prediction model exited with code ${PRED_EXIT}${RESET}" >&2
        exit $PRED_EXIT
    fi
    echo -e "${GREEN}✓ Prediction model completed${RESET}"
fi

# Write run metadata.
RESULTS_DIR="${RESULTS_ROOT:-./abc_results}"
cat > "${RESULTS_DIR}/run_metadata.json" <<METAEOF
{
  "config_file": "$(realpath "$CONFIG_FILE")",
  "overrides": $(python3 -c "import json; print(json.dumps($(printf '%s\n' "${OVERRIDES[@]+"${OVERRIDES[@]}"}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo '[]')))" 2>/dev/null || echo '[]'),
  "started_at": "${START_TS}",
  "elapsed_s": ${ELAPSED},
  "runner_exit_code": ${RUNNER_EXIT},
  "prediction_enabled": ${RUN_PREDICTION}
}
METAEOF

echo ""
echo -e "${GREEN}✓ All done. Results → ${RESULTS_DIR}${RESET}"
echo -e "  Run metadata → ${RESULTS_DIR}/run_metadata.json"
