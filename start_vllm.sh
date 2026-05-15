#!/bin/bash
# start_vllm.sh — Start vLLM on Ascend NPU (native bare-metal)
# Usage: ./start_vllm.sh [config.yaml]
set -euo pipefail

ACTION="start"
CONFIG="config/vllm_config.yaml"
BASELINE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --stop|stop)
            ACTION="stop"; shift ;;
        --start|start)
            ACTION="start"; shift ;;
        --baseline)
            BASELINE=true; shift ;;
        --config|-c)
            CONFIG="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--start|--stop] [--baseline] [--config <path>] [<config>]"
            exit 0 ;;
        --*)
            echo "[ERROR] Unknown option: $1" >&2
            exit 2 ;;
        *)
            CONFIG="$1"; shift ;;
    esac
done

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

pid_file_for_mode() {
    local configured
    configured=$(cfg native.pid_file)
    if [ "$BASELINE" = "true" ]; then
        case "$configured" in
            *.pid) printf '%s\n' "${configured%.pid}_baseline.pid" ;;
            *)     printf '%s\n' "${configured}.baseline" ;;
        esac
    else
        printf '%s\n' "$configured"
    fi
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

is_vllm_pid() {
    local pid="$1"
    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
    esac
    kill -0 "$pid" 2>/dev/null || return 1
    local cmdline=""
    if [ -r "/proc/$pid/cmdline" ]; then
        cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    elif command -v ps >/dev/null 2>&1; then
        cmdline=$(ps -p "$pid" -o command= 2>/dev/null || true)
    fi
    case "$cmdline" in
        *vllm*) return 0 ;;
    esac
    return 1
}

list_vllm_worker_pids() {
    {
        pgrep -x VLLMWorker_TP 2>/dev/null || true
        pgrep -f VLLMWorker_TP 2>/dev/null || true
    } | awk '!seen[$0]++'
}

snapshot_vllm_worker_pids() {
    local pid_file="$1"
    list_vllm_worker_pids > "${pid_file}.workers.before" 2>/dev/null || true
}

clear_vllm_worker_snapshot() {
    local pid_file
    pid_file=$(pid_file_for_mode)
    rm -f "${pid_file}.workers.before"
}

stop_vllm_from_pid_file() {
    local pid_file
    pid_file=$(pid_file_for_mode)

    new_vllm_worker_pids_since_snapshot() {
        local snapshot="${pid_file}.workers.before"
        local worker_pids pid
        [ -f "$snapshot" ] || return 0
        worker_pids=$(list_vllm_worker_pids)
        for pid in $worker_pids; do
            if ! grep -qx "$pid" "$snapshot" 2>/dev/null; then
                echo "$pid"
            fi
        done
    }

    collect_descendant_pids() {
        local root="$1"
        local queue="$root"
        local descendants=""
        local parent children child
        while [ -n "$queue" ]; do
            parent="${queue%% *}"
            if [ "$queue" = "$parent" ]; then
                queue=""
            else
                queue="${queue#* }"
            fi
            children=$(pgrep -P "$parent" 2>/dev/null || true)
            for child in $children; do
                descendants="${descendants} ${child}"
                queue="${queue:+$queue }${child}"
            done
        done
        echo "$descendants"
    }

    process_group_id() {
        ps -o pgid= -p "$1" 2>/dev/null | tr -d '[:space:]'
    }

    dedupe_pids() {
        local seen="" out="" pid
        for pid in "$@"; do
            case "$pid" in
                ''|*[!0-9]*) continue ;;
            esac
            case " $seen " in
                *" $pid "*) continue ;;
            esac
            seen="${seen} ${pid}"
            out="${out} ${pid}"
        done
        echo "$out"
    }

    send_signal_to_pids() {
        local signal="$1"
        shift
        local pid
        for pid in "$@"; do
            case "$pid" in
                ''|*[!0-9]*) continue ;;
            esac
            [ "$pid" = "$$" ] && continue
            kill "-${signal}" "$pid" 2>/dev/null || true
        done
    }

    pids_still_running() {
        local running="" pid
        for pid in "$@"; do
            case "$pid" in
                ''|*[!0-9]*) continue ;;
            esac
            if kill -0 "$pid" 2>/dev/null; then
                running="${running} ${pid}"
            fi
        done
        echo "$running"
    }

    wait_for_vllm_shutdown() {
        local pgid="$1"
        shift
        local pids="$*"
        local running group_running
        for _ in $(seq 1 15); do
            running=$(pids_still_running $pids)
            if [ -n "$pgid" ] && command -v pgrep >/dev/null 2>&1; then
                group_running=$(pgrep -g "$pgid" 2>/dev/null || true)
                running=$(dedupe_pids $running $group_running)
            fi
            if [ -z "$running" ]; then
                return 0
            fi
            sleep 1
        done
        return 1
    }

    if [ ! -f "$pid_file" ]; then
        echo "[WARN] No vLLM PID file found at $pid_file; nothing to kill." >&2
        return 0
    fi

    local pid
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [ -z "$pid" ]; then
        echo "[WARN] vLLM PID file is empty: $pid_file" >&2
        rm -f "$pid_file"
        return 0
    fi

    local descendants worker_pids target_pids pgid self_pgid kill_pgid
    descendants=""
    pgid=""
    kill_pgid=""
    if kill -0 "$pid" 2>/dev/null; then
        descendants=$(collect_descendant_pids "$pid")
        pgid=$(process_group_id "$pid")
    fi
    worker_pids=$(new_vllm_worker_pids_since_snapshot)
    target_pids=$(dedupe_pids "$pid" $descendants $worker_pids)

    if [ -z "$target_pids" ]; then
        echo "[WARN] vLLM PID $pid is not running; removing stale $pid_file" >&2
        rm -f "$pid_file" "${pid_file}.workers.before"
        return 0
    fi

    self_pgid=$(process_group_id "$$")
    if [ -n "$pgid" ] && [ "$pgid" != "$self_pgid" ]; then
        kill_pgid="$pgid"
        echo "[ERROR] Killing vLLM process group $pgid rooted at PID $pid because startup validation failed." >&2
        kill -TERM "-$pgid" 2>/dev/null || true
    else
        echo "[ERROR] Killing vLLM process tree rooted at PID $pid because startup validation failed." >&2
    fi
    if [ -n "$worker_pids" ]; then
        echo "[ERROR] Also killing vLLM worker PID(s):$(dedupe_pids $worker_pids)" >&2
    fi
    send_signal_to_pids TERM $target_pids
    if wait_for_vllm_shutdown "$kill_pgid" $target_pids; then
        rm -f "$pid_file" "${pid_file}.workers.before"
        return 0
    fi

    echo "[ERROR] vLLM processes did not exit after SIGTERM; sending SIGKILL." >&2
    if [ -n "$kill_pgid" ]; then
        kill -KILL "-$kill_pgid" 2>/dev/null || true
    fi
    send_signal_to_pids KILL $target_pids
    rm -f "$pid_file" "${pid_file}.workers.before"
}

verify_agent_kv_offload_route() {
    local url="http://localhost:${HOST_PORT}/agent_kv_cache/offload"
    local response body code

    echo "[INFO] Verifying agent KV offload route: POST $url"
    if ! response=$(curl -sS --max-time 10 -X POST "$url" \
            -H "Content-Type: application/json" \
            -d '{"agent_id":"__startup_probe__"}' \
            -w $'\n%{http_code}' 2>&1); then
        echo "[ERROR] Agent KV offload probe failed to reach vLLM:" >&2
        echo "$response" >&2
        stop_vllm_from_pid_file
        return 1
    fi

    code="${response##*$'\n'}"
    body="${response%$'\n'*}"
    case "$code" in
        ''|*[!0-9]*)
            echo "[ERROR] Agent KV offload probe returned invalid HTTP code: $code" >&2
            echo "[ERROR] Response body: $body" >&2
            stop_vllm_from_pid_file
            return 1
            ;;
    esac

    if [ "$code" -lt 200 ] || [ "$code" -ge 300 ]; then
        echo "[ERROR] Agent KV offload probe returned HTTP $code" >&2
        echo "[ERROR] Response body: $body" >&2
        stop_vllm_from_pid_file
        return 1
    fi

    if ! AGENT_KV_PROBE_BODY="$body" python3 - <<'PY'
import json
import os
import sys

body = os.environ.get("AGENT_KV_PROBE_BODY", "")
try:
    payload = json.loads(body)
except Exception as exc:
    print(f"[ERROR] Agent KV offload probe returned non-JSON body: {exc}", file=sys.stderr)
    print(body, file=sys.stderr)
    sys.exit(1)

reason = str(payload.get("reason", ""))
ok = (
    payload.get("offloaded") is True
    or "no tracked KV blocks for agent" in reason
    or "no held KV blocks for agent" in reason
)
if not ok:
    print("[ERROR] Agent KV offload route responded, but payload is not usable:", file=sys.stderr)
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    sys.exit(1)

print("[INFO] Agent KV offload probe OK: " + json.dumps(payload, sort_keys=True))
PY
    then
        stop_vllm_from_pid_file
        return 1
    fi
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
    local PID_FILE=$(pid_file_for_mode)

    REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
    if [ "$BASELINE" = "true" ]; then
        VENV="${VLLM_BASELINE_VENV:-$REPO_DIR/.venv-baseline}"
    else
        VENV="$REPO_DIR/.venv"
    fi
    VLLM_ASCEND_VERSION="${VLLM_ASCEND_VERSION:-0.13.0}"
    VLLM_VERSION="${VLLM_VERSION:-$VLLM_ASCEND_VERSION}"
    TORCH_VERSION="${VLLM_ASCEND_TORCH_VERSION:-2.8.0}"
    TORCH_NPU_VERSION="${VLLM_ASCEND_TORCH_NPU_VERSION:-2.8.0.post2}"
    TORCHVISION_VERSION="${VLLM_ASCEND_TORCHVISION_VERSION:-0.23.0}"
    TORCHAUDIO_VERSION="${VLLM_ASCEND_TORCHAUDIO_VERSION:-2.8.0}"
    TRITON_ASCEND_VERSION="${VLLM_ASCEND_TRITON_ASCEND_VERSION:-3.2.0}"
    VLLM_ASCEND_SOC_VERSION="${VLLM_ASCEND_SOC_VERSION:-ascend910_9391}"
    VLLM_INSTALL_METHOD="${VLLM_ASCEND_INSTALL_METHOD:-source}"
    VLLM_GIT_REF="${VLLM_GIT_REF:-v${VLLM_VERSION}}"
    VLLM_GIT_URL="${VLLM_GIT_URL:-git@github.com:vllm-project/vllm.git}"
    if [ "$BASELINE" = "true" ]; then
        VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-$REPO_DIR/.vllm-baseline-src/${VLLM_GIT_REF}}"
    else
        VLLM_SOURCE_DIR="${VLLM_SOURCE_DIR:-$REPO_DIR/.vllm-src/${VLLM_GIT_REF}}"
    fi
    VLLM_ASCEND_GIT_REF="${VLLM_ASCEND_GIT_REF:-v${VLLM_ASCEND_VERSION}}"
    VLLM_ASCEND_GIT_URL="${VLLM_ASCEND_GIT_URL:-git@github.com:vllm-project/vllm-ascend.git}"
    VLLM_ASCEND_MAX_JOBS="${VLLM_ASCEND_MAX_JOBS:-16}"
    if [ "$BASELINE" = "true" ]; then
        VLLM_ASCEND_SOURCE_DIR="${VLLM_ASCEND_SOURCE_DIR:-$REPO_DIR/.vllm-ascend-baseline-src/${VLLM_ASCEND_GIT_REF}-${VLLM_ASCEND_SOC_VERSION}}"
    else
        VLLM_ASCEND_SOURCE_DIR="${VLLM_ASCEND_SOURCE_DIR:-$REPO_DIR/.vllm-ascend-src/${VLLM_ASCEND_GIT_REF}-${VLLM_ASCEND_SOC_VERSION}}"
    fi
    VLLM_ASCEND_SOC_MARKER="$VENV/.vllm-ascend-soc-version"
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"

    # ── 1. Create project venv via uv (skipped if already exists) ────────────
    if [ ! -f "$VENV/bin/activate" ]; then
        echo "[INFO] Creating project venv with uv..."
        uv venv "$VENV" --python "$SHARED_PY" --system-site-packages
    fi

    # ── 2. Link shared deps, then install local Ascend/vLLM packages once ────
    PYTHON_VER=$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    VENV_SITE="$VENV/lib/python${PYTHON_VER}/site-packages"
    VENV_VLLM="$VENV_SITE/vllm"
    if [ "$VLLM_INSTALL_METHOD" = "source" ]; then
        VLLM_PATCH_DIR="$VLLM_SOURCE_DIR/vllm"
    else
        VLLM_PATCH_DIR="$VENV_VLLM"
    fi
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

    if [ "$VLLM_INSTALL_METHOD" = "source" ]; then
        VLLM_SOURCE_DIR="$VLLM_SOURCE_DIR" \
        VLLM_ASCEND_SOURCE_DIR="$VLLM_ASCEND_SOURCE_DIR" \
        VENV_SITE="$VENV_SITE" \
        "$VENV/bin/python" - <<'PY'
import os
import pathlib

paths = [
    str(pathlib.Path(os.environ["VLLM_SOURCE_DIR"]).resolve()),
    str(pathlib.Path(os.environ["VLLM_ASCEND_SOURCE_DIR"]).resolve()),
]
pth = pathlib.Path(os.environ["VENV_SITE"]) / "00_vllm_source_paths.pth"
pth.write_text(
    "import sys; "
    f"paths = {paths!r}; "
    "[sys.path.insert(0, p) for p in reversed(paths) if p and p not in sys.path]\n"
)
PY
    fi

    check_local_vllm_ascend() {
        VENV="$VENV" \
        DESIRED_VLLM="$VLLM_VERSION" \
        DESIRED_VLLM_ASCEND="$VLLM_ASCEND_VERSION" \
        DESIRED_VLLM_ASCEND_SOC="$VLLM_ASCEND_SOC_VERSION" \
        VLLM_INSTALL_METHOD="$VLLM_INSTALL_METHOD" \
        VLLM_SOURCE_DIR="$VLLM_SOURCE_DIR" \
        VLLM_ASCEND_SOURCE_DIR="$VLLM_ASCEND_SOURCE_DIR" \
        VLLM_ASCEND_SOC_MARKER="$VLLM_ASCEND_SOC_MARKER" \
        "$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import importlib.util
import os
import pathlib
import sys

desired_vllm = os.environ["DESIRED_VLLM"]
desired_ascend = os.environ["DESIRED_VLLM_ASCEND"]
desired_soc = os.environ["DESIRED_VLLM_ASCEND_SOC"]
install_method = os.environ["VLLM_INSTALL_METHOD"]
soc_marker = pathlib.Path(os.environ["VLLM_ASCEND_SOC_MARKER"])
venv = pathlib.Path(os.environ["VENV"]).resolve()
vllm_source = pathlib.Path(os.environ["VLLM_SOURCE_DIR"]).resolve()
ascend_source = pathlib.Path(os.environ["VLLM_ASCEND_SOURCE_DIR"]).resolve()

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

def is_under(path, root):
    if path is None:
        return False
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def is_release_or_local_build(version, desired):
    return version == desired or (
        version is not None and version.startswith(desired + "+")
    )

vllm_version = dist_version("vllm")
ascend_version = dist_version("vllm-ascend")
vllm_path = package_path("vllm")
ascend_path = package_path("vllm_ascend")
soc = soc_marker.read_text().strip() if soc_marker.exists() else None
if install_method == "source":
    vllm_path_ready = is_under(vllm_path, vllm_source)
    ascend_path_ready = is_under(ascend_path, ascend_source)
    vllm_version_ready = is_release_or_local_build(vllm_version, desired_vllm)
    ascend_version_ready = is_release_or_local_build(ascend_version, desired_ascend)
else:
    vllm_path_ready = is_local(vllm_path)
    ascend_path_ready = is_local(ascend_path)
    vllm_version_ready = vllm_version == desired_vllm
    ascend_version_ready = ascend_version == desired_ascend
ready = (
    vllm_version_ready
    and ascend_version_ready
    and vllm_path_ready
    and ascend_path_ready
    and soc == desired_soc
)

if ready:
    print(
        f"[INFO] Reusing local vllm=={vllm_version}, "
        f"vllm-ascend=={ascend_version} ({soc}, {install_method})"
    )
else:
    print(
        "[INFO] Local vllm-ascend install needed "
        f"(vllm={vllm_version!r} at {vllm_path}, "
        f"vllm-ascend={ascend_version!r} at {ascend_path}, "
        f"soc={soc!r}, method={install_method!r})"
    )
sys.exit(0 if ready else 1)
PY
    }

    check_local_ascend_stack() {
        VENV="$VENV" \
        DESIRED_TORCH="$TORCH_VERSION" \
        DESIRED_TORCH_NPU="$TORCH_NPU_VERSION" \
        DESIRED_TORCHVISION="$TORCHVISION_VERSION" \
        DESIRED_TORCHAUDIO="$TORCHAUDIO_VERSION" \
        DESIRED_TRITON_ASCEND="$TRITON_ASCEND_VERSION" \
        "$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import importlib.util
import os
import pathlib
import sys

venv = pathlib.Path(os.environ["VENV"]).resolve()
desired = {
    "torch": os.environ["DESIRED_TORCH"],
    "torch-npu": os.environ["DESIRED_TORCH_NPU"],
    "torchvision": os.environ["DESIRED_TORCHVISION"],
    "torchaudio": os.environ["DESIRED_TORCHAUDIO"],
    "triton-ascend": os.environ["DESIRED_TRITON_ASCEND"],
}
modules = {
    "torch": "torch",
    "torch-npu": "torch_npu",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "triton-ascend": "triton",
}

def dist_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

def dist_path(name):
    try:
        return pathlib.Path(metadata.distribution(name).locate_file("")).resolve()
    except metadata.PackageNotFoundError:
        return None

def module_path(module_name):
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

state = []
ready = True
for dist_name, want in desired.items():
    version = dist_version(dist_name)
    path = module_path(modules[dist_name])
    state.append(f"{dist_name}={version!r} at {path}")
    ready = ready and version == want and is_local(path)

triton_dist_path = dist_path("triton")
if is_local(triton_dist_path):
    state.append(f"local standard triton={dist_version('triton')!r} at {triton_dist_path}")
    ready = False

if ready:
    print("[INFO] Reusing local Ascend runtime stack: " + ", ".join(state))
else:
    print("[INFO] Local Ascend runtime stack install needed (" + ", ".join(state) + ")")
sys.exit(0 if ready else 1)
PY
    }

    verify_git_source_ref() {
        local name="$1"
        local ref="$2"
        local dest="$3"
        local head ref_commit

        if ! head=$(git -C "$dest" rev-parse HEAD 2>/dev/null); then
            echo "[ERROR] Could not inspect $name source checkout at $dest" >&2
            exit 1
        fi
        if ! ref_commit=$(git -C "$dest" rev-parse "${ref}^{commit}" 2>/dev/null); then
            echo "[ERROR] $name source checkout at $dest does not contain requested release ref $ref" >&2
            echo "[ERROR] Fetch tags into that checkout, remove it so the script can reclone, or set ${name}_GIT_REF explicitly." >&2
            exit 1
        fi
        if [ "$head" != "$ref_commit" ]; then
            echo "[ERROR] $name source checkout at $dest is not at requested release ref $ref" >&2
            echo "[ERROR] Current HEAD: ${head:0:12}; $ref: ${ref_commit:0:12}" >&2
            echo "[ERROR] Remove $dest so the script can reclone the release, or set ${name}_SOURCE_DIR to a clean $ref checkout." >&2
            exit 1
        fi
        echo "[INFO] Verified $name source checkout at $dest ($ref -> ${head:0:12})"
    }

    clone_git_source() {
        local name="$1"
        local url="$2"
        local ref="$3"
        local dest="$4"

        if [ -d "$dest/.git" ]; then
            verify_git_source_ref "$name" "$ref" "$dest"
            return 0
        fi

        if [ -d "$dest" ] && [ -z "$(find "$dest" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
            rmdir "$dest"
        fi

        if [ -e "$dest" ]; then
            echo "[ERROR] $name source path exists but is not a git checkout: $dest" >&2
            echo "[ERROR] Remove it or set ${name}_SOURCE_DIR to a clean checkout, then rerun." >&2
            exit 1
        fi

        echo "[INFO] Cloning $name $ref source from $url ..."
        mkdir -p "$(dirname "$dest")"
        if ! git clone --depth 1 --branch "$ref" "$url" "$dest"; then
            echo "[ERROR] Failed to clone $name from $url" >&2
            echo "[ERROR] If GitHub is blocked, pre-populate $dest or set ${name}_GIT_URL to an accessible mirror." >&2
            exit 1
        fi
        verify_git_source_ref "$name" "$ref" "$dest"
    }

    if check_local_ascend_stack; then
        :
    else
        echo "[INFO] Installing Ascend runtime stack into $VENV ..."
        rm -rf \
          "$VENV_SITE"/torch \
          "$VENV_SITE"/torch-*.dist-info \
          "$VENV_SITE"/torch_npu \
          "$VENV_SITE"/torch_npu-*.dist-info \
          "$VENV_SITE"/torchvision \
          "$VENV_SITE"/torchvision-*.dist-info \
          "$VENV_SITE"/torchaudio \
          "$VENV_SITE"/torchaudio-*.dist-info \
          "$VENV_SITE"/triton \
          "$VENV_SITE"/triton-*.dist-info \
          "$VENV_SITE"/triton_ascend-*.dist-info
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$VENV/bin/python" --upgrade \
              "torch==${TORCH_VERSION}" \
              "torch-npu==${TORCH_NPU_VERSION}" \
              "torchvision==${TORCHVISION_VERSION}" \
              "torchaudio==${TORCHAUDIO_VERSION}" \
              "triton-ascend==${TRITON_ASCEND_VERSION}"
        else
            "$VENV/bin/python" -m pip install --upgrade \
              "torch==${TORCH_VERSION}" \
              "torch-npu==${TORCH_NPU_VERSION}" \
              "torchvision==${TORCHVISION_VERSION}" \
              "torchaudio==${TORCHAUDIO_VERSION}" \
              "triton-ascend==${TRITON_ASCEND_VERSION}"
        fi
        if ! check_local_ascend_stack; then
            echo "[ERROR] Ascend runtime stack verification failed after install" >&2
            exit 1
        fi
    fi

    if check_local_vllm_ascend; then
        :
    else
        echo "[INFO] Installing vllm==${VLLM_VERSION} and vllm-ascend==${VLLM_ASCEND_VERSION} (${VLLM_ASCEND_SOC_VERSION}, ${VLLM_INSTALL_METHOD}) into $VENV ..."
        rm -rf \
          "$VENV_SITE"/vllm \
          "$VENV_SITE"/vllm_ascend \
          "$VENV_SITE"/vllm-*.dist-info \
          "$VENV_SITE"/vllm-*.egg-info \
          "$VENV_SITE"/vllm_ascend-*.dist-info \
          "$VENV_SITE"/vllm_ascend-*.egg-info \
          "$VENV_SITE"/__editable__.*vllm-* \
          "$VENV_SITE"/__editable__.*vllm.* \
          "$VENV_SITE"/__editable__.*vllm_ascend* \
          "$VLLM_ASCEND_SOC_MARKER"

        if [ "$VLLM_INSTALL_METHOD" = "wheel" ]; then
            echo "[INFO] Installing official pre-built wheels from pip index: $PIP_INDEX_URL"
            if command -v uv >/dev/null 2>&1; then
                uv pip install --python "$VENV/bin/python" --upgrade \
                  "vllm==${VLLM_VERSION}"
                uv pip install --python "$VENV/bin/python" --upgrade \
                  "vllm-ascend==${VLLM_ASCEND_VERSION}"
            else
                "$VENV/bin/python" -m pip install --upgrade \
                  "vllm==${VLLM_VERSION}"
                "$VENV/bin/python" -m pip install --upgrade \
                  "vllm-ascend==${VLLM_ASCEND_VERSION}"
            fi
        elif [ "$VLLM_INSTALL_METHOD" = "source" ]; then
            "$VENV/bin/python" -m pip install --upgrade \
              attrs "numpy<2.0.0" decorator sympy cffi pyyaml pathlib2 psutil \
              protobuf scipy requests absl-py typing_extensions \
              cmake ninja packaging pybind11 setuptools setuptools-scm wheel

            clone_git_source "VLLM" "$VLLM_GIT_URL" "$VLLM_GIT_REF" "$VLLM_SOURCE_DIR"
            clone_git_source "VLLM_ASCEND" "$VLLM_ASCEND_GIT_URL" "$VLLM_ASCEND_GIT_REF" "$VLLM_ASCEND_SOURCE_DIR"

            echo "[INFO] Updating vllm-ascend submodules for Atlas A3 custom operators ..."
            git -C "$VLLM_ASCEND_SOURCE_DIR" submodule update --init --recursive

            VLLM_BUILD_PYTHONPATH="$VENV_SITE${PYTHONPATH:+:$PYTHONPATH}"
            echo "[INFO] Building/installing vLLM from source with VLLM_TARGET_DEVICE=empty ..."
            PATH="$VENV/bin:$PATH" \
            PYTHONPATH="$VLLM_BUILD_PYTHONPATH" \
            SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_VERSION" \
            SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="$VLLM_VERSION" \
            VLLM_TARGET_DEVICE=empty \
            "$VENV/bin/python" -m pip install -v --no-build-isolation --no-deps -e \
              "$VLLM_SOURCE_DIR"
        else
            echo "[ERROR] Unsupported VLLM_ASCEND_INSTALL_METHOD=$VLLM_INSTALL_METHOD (expected source or wheel)" >&2
            exit 1
        fi

        if [ "$VLLM_INSTALL_METHOD" = "source" ]; then
            VLLM_ASCEND_SOURCE_DIR="$VLLM_ASCEND_SOURCE_DIR" "$VENV/bin/python" - <<'PY'
import os
import pathlib

cmake_lists = pathlib.Path(os.environ["VLLM_ASCEND_SOURCE_DIR"]) / "CMakeLists.txt"
text = cmake_lists.read_text()
marker = "vllm_ascend_flatten_host_objs"
if marker not in text:
    anchor = """ascendc_library(vllm_ascend_kernels SHARED
    ${VLLM_ASCEND_CUSTOM_OP}
)
"""
    patch = anchor + """
# CANN 8.5.1 installs host object-library outputs under objects-Release/.
# recompile_binary.py scans only this top-level host dir, so copy the generated
# .o files back before the final ascendc library link step.
set(VLLM_ASCEND_HOST_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/vllm_ascend_kernels_host_dir")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/flatten_vllm_ascend_host_objs.cmake" "
set(host_dir \\"${VLLM_ASCEND_HOST_OBJ_DIR}\\")
file(GLOB direct_host_objs \\"\\${host_dir}/*.o\\")
if(NOT direct_host_objs)
  file(GLOB_RECURSE host_objs \\"\\${host_dir}/objects-*/*.o\\")
  if(NOT host_objs)
    message(WARNING \\"No host object files found to flatten under \\${host_dir}\\")
  endif()
  foreach(host_obj IN LISTS host_objs)
    file(COPY \\"\\${host_obj}\\" DESTINATION \\"\\${host_dir}\\")
  endforeach()
endif()
")
add_custom_target(vllm_ascend_flatten_host_objs
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/flatten_vllm_ascend_host_objs.cmake"
    DEPENDS vllm_ascend_kernels_host
)
add_dependencies(vllm_ascend_kernels vllm_ascend_flatten_host_objs)
"""
    if anchor not in text:
        raise SystemExit(f"Could not patch {cmake_lists}: ascendc_library anchor not found")
    cmake_lists.write_text(text.replace(anchor, patch, 1))
    print(f"[INFO] Patched {cmake_lists} for CANN host object layout")
else:
    print(f"[INFO] Reusing existing CANN host object layout patch in {cmake_lists}")
PY
            VLLM_ASCEND_BUILD_PYTHONPATH="$VENV_SITE${PYTHONPATH:+:$PYTHONPATH}"
            rm -rf \
              "$VLLM_ASCEND_SOURCE_DIR"/build \
              "$VLLM_ASCEND_SOURCE_DIR"/csrc/build \
              "$VLLM_ASCEND_SOURCE_DIR"/vllm_ascend/_cann_ops_custom
            echo "[INFO] Building/installing vllm-ascend from source for SOC_VERSION=${VLLM_ASCEND_SOC_VERSION} (MAX_JOBS=${VLLM_ASCEND_MAX_JOBS}) ..."
            export CMAKE_BUILD_PARALLEL_LEVEL="$VLLM_ASCEND_MAX_JOBS"
            export MAX_JOBS="$VLLM_ASCEND_MAX_JOBS"
            CMAKE_BUILD_PARALLEL_LEVEL="$VLLM_ASCEND_MAX_JOBS" \
            MAX_JOBS="$VLLM_ASCEND_MAX_JOBS" \
            PATH="$VENV/bin:$PATH" \
            PYTHONPATH="$VLLM_ASCEND_BUILD_PYTHONPATH" \
            SETUPTOOLS_SCM_PRETEND_VERSION="$VLLM_ASCEND_VERSION" \
            SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_ASCEND="$VLLM_ASCEND_VERSION" \
            SOC_VERSION="$VLLM_ASCEND_SOC_VERSION" \
            "$VENV/bin/python" -m pip install -v --no-build-isolation --no-deps -e \
              "$VLLM_ASCEND_SOURCE_DIR"
        fi
        printf '%s\n' "$VLLM_ASCEND_SOC_VERSION" > "$VLLM_ASCEND_SOC_MARKER"
        if ! check_local_vllm_ascend; then
            echo "[ERROR] vllm-ascend install verification failed after install" >&2
            exit 1
        fi
    fi

    VLLM_RUNTIME_REQS="$VENV/vllm-runtime-deps.txt"
    VLLM_CONSTRAINTS="$VENV/vllm-ascend-constraints.txt"
    cat > "$VLLM_CONSTRAINTS" <<EOF
torch==${TORCH_VERSION}
torch-npu==${TORCH_NPU_VERSION}
torchvision==${TORCHVISION_VERSION}
torchaudio==${TORCHAUDIO_VERSION}
triton-ascend==${TRITON_ASCEND_VERSION}
numpy<2.0.0
EOF
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
        echo "[INFO] Installing vLLM runtime deps from wheel metadata with Ascend torch constraints ..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$VENV/bin/python" --upgrade \
              -c "$VLLM_CONSTRAINTS" \
              -r "$VLLM_RUNTIME_REQS"
        else
            "$VENV/bin/python" -m pip install --upgrade \
              -c "$VLLM_CONSTRAINTS" \
              -r "$VLLM_RUNTIME_REQS"
        fi
    fi

    unset -f check_local_ascend_stack
    unset -f check_local_vllm_ascend

    if [ "$BASELINE" = "true" ]; then
        # ── 3. Baseline: verify clean upstream vLLM/vllm-ascend ──────────────
        echo "[INFO] Baseline mode: skipping agent-aware vLLM patches."
        echo "[INFO] Verifying clean baseline vLLM import path..."
        VLLM_PATCH_DIR="$VLLM_PATCH_DIR" \
        DESIRED_VLLM="$VLLM_VERSION" \
        DESIRED_VLLM_ASCEND="$VLLM_ASCEND_VERSION" \
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import importlib.util
import pathlib

import vllm
import vllm.entrypoints.openai.api_server as api_server
import vllm.entrypoints.openai.protocol as protocol
import vllm.entrypoints.openai.serving_chat as serving_chat

target = pathlib.Path(__import__("os").environ["VLLM_PATCH_DIR"]).resolve()
vllm_file = pathlib.Path(vllm.__file__).resolve()
api_server_file = pathlib.Path(api_server.__file__).resolve()
protocol_file = pathlib.Path(protocol.__file__).resolve()
serving_chat_file = pathlib.Path(serving_chat.__file__).resolve()
desired = __import__("os").environ["DESIRED_VLLM"]
desired_ascend = __import__("os").environ["DESIRED_VLLM_ASCEND"]
version = metadata.version("vllm")
ascend_version = metadata.version("vllm-ascend")

def is_release_or_local_build(actual, expected):
    return actual == expected or actual.startswith(expected + "+")

for path in (vllm_file, api_server_file, protocol_file, serving_chat_file):
    if not path.is_relative_to(target):
        raise AssertionError(f"Imported vLLM path {path} is not under baseline target {target}")

if not is_release_or_local_build(version, desired):
    raise AssertionError(f"Imported vLLM version {version!r} does not match target {desired!r}")
if not is_release_or_local_build(ascend_version, desired_ascend):
    raise AssertionError(
        f"Imported vllm-ascend version {ascend_version!r} does not match "
        f"target {desired_ascend!r}"
    )

usage_fields = getattr(protocol.UsageInfo, "model_fields", {})
chat_fields = getattr(protocol.ChatCompletionRequest, "model_fields", {})
if "kv_blocks_used" in usage_fields or "kv_blocks_size_gb" in usage_fields:
    raise AssertionError("Baseline vLLM unexpectedly exposes patched KV usage fields")
if "agent_id" in chat_fields:
    raise AssertionError("Baseline vLLM unexpectedly exposes patched agent_id request field")
route_paths = {getattr(route, "path", "") for route in api_server.router.routes}
patched_routes = {
    "/agent_kv_cache/offload",
    "/agent_kv_cache/usage",
    "/agent_kv_cache/restore",
    "/agent_kv_cache/release",
} & route_paths
if patched_routes:
    raise AssertionError(f"Baseline api_server has patched agent KV route(s): {sorted(patched_routes)}")
spec = importlib.util.find_spec(
    "vllm.distributed.kv_transfer.kv_connector.v1.agent_offloading_connector"
)
if spec is not None:
    raise AssertionError(f"Baseline unexpectedly imports AgentAware connector from {spec.origin}")

print(f"[INFO] baseline vllm version -> {version}")
print(f"[INFO] baseline vllm-ascend version -> {ascend_version}")
print(f"[INFO] baseline vllm package -> {vllm_file}")
print(f"[INFO] baseline api_server.py -> {api_server_file}")
print("[INFO] baseline agent KV routes -> none")
PY
    else
        # ── 3. Apply KV-tracking patches to project venv (idempotent) ────────
        echo "[INFO] Applying patches to vLLM package at $VLLM_PATCH_DIR ..."
        "$VENV/bin/python" "$REPO_DIR/src/vllm_patches/apply_patches.py" \
            --vllm-dir "$VLLM_PATCH_DIR"

        # ── 4. Verify the project venv imports the patched copy ──────────────
        echo "[INFO] Verifying patched vLLM import path..."
        VLLM_PATCH_DIR="$VLLM_PATCH_DIR" \
        DESIRED_VLLM="$VLLM_VERSION" \
        DESIRED_VLLM_ASCEND="$VLLM_ASCEND_VERSION" \
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import importlib
import pathlib

import vllm
import vllm.entrypoints.openai.api_server as api_server
import vllm.entrypoints.openai.protocol as protocol
import vllm.entrypoints.openai.serving_chat as serving_chat

target = pathlib.Path(__import__("os").environ["VLLM_PATCH_DIR"]).resolve()
vllm_file = pathlib.Path(vllm.__file__).resolve()
api_server_file = pathlib.Path(api_server.__file__).resolve()
protocol_file = pathlib.Path(protocol.__file__).resolve()
serving_chat_file = pathlib.Path(serving_chat.__file__).resolve()
desired = __import__("os").environ["DESIRED_VLLM"]
desired_ascend = __import__("os").environ["DESIRED_VLLM_ASCEND"]
version = metadata.version("vllm")
ascend_version = metadata.version("vllm-ascend")

def is_release_or_local_build(actual, expected):
    return actual == expected or actual.startswith(expected + "+")

for path in (vllm_file, api_server_file, protocol_file, serving_chat_file):
    if not path.is_relative_to(target):
        raise AssertionError(f"Imported vLLM path {path} is not under patched target {target}")

if not is_release_or_local_build(version, desired):
    raise AssertionError(f"Imported vLLM version {version!r} does not match target {desired!r}")
if not is_release_or_local_build(ascend_version, desired_ascend):
    raise AssertionError(
        f"Imported vllm-ascend version {ascend_version!r} does not match "
        f"target {desired_ascend!r}"
    )

assert "kv_blocks_used" in protocol.UsageInfo.model_fields
assert "kv_blocks_size_gb" in protocol.UsageInfo.model_fields
assert "agent_id" in protocol.ChatCompletionRequest.model_fields
connector = importlib.import_module(
    "vllm.distributed.kv_transfer.kv_connector.v1.agent_offloading_connector"
)
assert hasattr(connector, "AgentAwareOffloadingConnector")
route_paths = {getattr(route, "path", "") for route in api_server.router.routes}
missing_routes = {
    "/agent_kv_cache/offload",
    "/agent_kv_cache/usage",
    "/agent_kv_cache/restore",
    "/agent_kv_cache/release",
} - route_paths
if missing_routes:
    raise AssertionError(
        f"api_server router missing agent KV route(s): {sorted(missing_routes)}; "
        f"api_server.py={api_server_file}"
    )

print(f"[INFO] vllm version -> {version}")
print(f"[INFO] vllm-ascend version -> {ascend_version}")
print(f"[INFO] vllm package -> {vllm_file}")
print(f"[INFO] api_server.py -> {api_server_file}")
print(f"[INFO] protocol.py -> {protocol_file}")
print(f"[INFO] serving_chat.py -> {serving_chat_file}")
print(f"[INFO] connector.py -> {pathlib.Path(connector.__file__).resolve()}")
print(f"[INFO] agent KV routes -> {sorted(p for p in route_paths if 'agent_kv' in p)}")
PY
    fi

    # ── 5. Skip if already running ────────────────────────────────────────────
    # Verify the recorded PID is actually a vLLM process before assuming the
    # service is up. A bare `kill -0` only checks PID existence, which can
    # match a recycled PID belonging to an unrelated process and leave us
    # waiting forever on a port that will never open.
    if [ -f "$PID_FILE" ]; then
        EXISTING_PID=$(cat "$PID_FILE" 2>/dev/null || true)
        if is_vllm_pid "$EXISTING_PID"; then
            echo "[INFO] vLLM already running (PID $EXISTING_PID). Skipping launch."
            return
        fi
        echo "[WARN] Stale vLLM PID file at $PID_FILE (PID '${EXISTING_PID:-<empty>}' is not a running vllm process). Removing." >&2
        rm -f "$PID_FILE" "${PID_FILE}.workers.before"
    fi

    if [ "$BASELINE" = "true" ]; then
        BASELINE_HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
            -H "Authorization: Bearer ${API_KEY}" \
            "http://localhost:${PORT}/v1/models" 2>/dev/null || echo "000")
        if [ "$BASELINE_HEALTH_CODE" = "200" ]; then
            AGENT_KV_ROUTE_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
                -X POST \
                -H "Content-Type: application/json" \
                -d '{"agent_id":"__baseline_probe__"}' \
                "http://localhost:${PORT}/agent_kv_cache/offload" \
                2>/dev/null || echo "000")
            case "$AGENT_KV_ROUTE_CODE" in
                2*)
                    echo "[ERROR] Baseline mode found patched agent KV routes on the running vLLM server." >&2
                    echo "[ERROR] Stop that server before launching the clean baseline." >&2
                    exit 2
                    ;;
            esac
            echo "[INFO] Baseline-compatible vLLM already serving on port ${PORT}. Skipping launch."
            return
        fi
    fi

    # ── 6. Launch vllm serve natively ────────────────────────────────────────
    if [ "$BASELINE" = "true" ]; then
        echo "[INFO] Starting baseline native vLLM (model=$(basename $MODEL), tp=$TP, port=$PORT)..."
    else
        echo "[INFO] Starting native vLLM (model=$(basename $MODEL), tp=$TP, port=$PORT)..."
    fi
    snapshot_vllm_worker_pids "$PID_FILE"
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

        VLLM_CMD=("$VENV/bin/python" "$VLLM_CLI" serve "$MODEL" \
          --served-model-name "$SERVED_NAME" \
          --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
          --tensor-parallel-size "$TP" \
          --dtype "$DTYPE")
        if [ "$BASELINE" = "true" ]; then
            echo "[INFO] Baseline mode: disabling --kv-transfer-config."
        else
            KV_TRANSFER_CONFIG='{"kv_connector": "AgentAwareOffloadingConnector", "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.agent_offloading_connector", "kv_role": "kv_both", "kv_connector_extra_config": {"num_cpu_blocks": 8192, "caching_hash_algo": "sha256_cbor", "spec_name": "NPUOffloadingSpec", "spec_module_path": "vllm_ascend.kv_offload.npu", "agent_hold_finished_requests": true, "agent_hold_ttl_s": 300.0}}'
            VLLM_CMD+=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
        fi
        if [ -n "$EXTRA" ]; then
            # shellcheck disable=SC2206
            EXTRA_ARGS=($EXTRA)
            VLLM_CMD+=("${EXTRA_ARGS[@]}")
        fi
        if command -v setsid >/dev/null 2>&1; then
            exec setsid "${VLLM_CMD[@]}"
        fi
        exec "${VLLM_CMD[@]}"
    ) > /dev/null 2>&1 &
    echo $! > "$PID_FILE"
    echo "[INFO] vLLM PID: $(cat $PID_FILE)"
}

# =============================================================================
# MAIN
# =============================================================================
HOST_PORT=$(cfg native.port)
VLLM_API_KEY=$(cfg native.api_key)

if [ "$ACTION" = "stop" ]; then
    stop_vllm_from_pid_file
    clear_vllm_worker_snapshot
    exit 0
fi

start_native

if ! wait_for_ready; then
    stop_vllm_from_pid_file
    exit 1
fi
if [ "$BASELINE" = "true" ]; then
    echo "[INFO] Baseline mode: skipping agent KV offload route probe."
else
    verify_agent_kv_offload_route || exit 1
fi
clear_vllm_worker_snapshot

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
