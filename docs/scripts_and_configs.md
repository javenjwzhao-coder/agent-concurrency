# Scripts and Configs

Components:

- `run_abc-bench.sh`
- `start_vllm.sh`
- `config/abc-bench_config.yaml`
- `config/vllm_config.yaml`

The scripts provide the operational path around the Python components. The
benchmark wrapper resolves YAML into runner/predictor commands. The vLLM script
builds or reuses an Ascend-compatible environment, applies vLLM patches, and
launches the server with the custom connector.

## `run_abc-bench.sh`

This is the main benchmark entrypoint:

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml
```

Use dry run to inspect the resolved settings:

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml --dry-run
```

Use overrides for one-off changes:

```bash
bash run_abc-bench.sh \
  --config config/abc-bench_config.yaml \
  --override dataset.max_tasks=2 \
  --override launch.randomise=false
```

### Wrapper Responsibilities

- Parse YAML with Python and PyYAML.
- Expand `${ENV_VAR}` references in config values.
- Apply repeated `--override key.path=value` patches.
- Validate required fields and dataset existence.
- Derive `sidecar.vllm_url` from `llm.base_url` when omitted.
- Print a redacted resolved configuration.
- Build shell-escaped runner and predictor commands.
- Check whether vLLM is already serving `/v1/models`.
- Start vLLM through `start_vllm.sh` when needed.
- Create `.bench-venv` with Python 3.12 for OpenHands dependencies.
- Install benchmark dependencies with retry support.
- Run the instrumented runner.
- Optionally train the predictor after the run.

### Important Environment Variables

| Variable | Purpose |
| --- | --- |
| `BENCH_UV_HTTP_TIMEOUT` | Timeout for `uv pip install`; default `300`. |
| `BENCH_UV_INSTALL_RETRIES` | Install retry count; default `3`. |
| Values referenced as `${VAR}` in YAML | Expanded before command construction. |

### Execution Sequence

1. Load and flatten config.
2. Validate inputs.
3. Display resolved config and command lines.
4. Exit early for `--dry-run`.
5. Check vLLM health at `<llm.base_url>/models`.
6. Start vLLM if health check fails.
7. Create and activate `.bench-venv`.
8. Install benchmark dependencies if imports fail.
9. Create workspace/results directories.
10. Run `src/run_abc_bench_instrumented.py`.
11. Run `src/build_tool_predictor.py` if prediction is enabled and not skipped.

### Prediction Skip Logic

If prediction is enabled and `prediction.save_model` already exists, the
wrapper skips training. This prevents accidental retraining on every benchmark
run when a model is already available for sidecar admission control.

## `config/abc-bench_config.yaml`

This YAML config is intentionally minimal. Omitted values fall back to
`run_abc-bench.sh` or runner defaults.

Major sections:

| Section | Purpose |
| --- | --- |
| `llm` | Model name, OpenAI-compatible base URL, and API key. |
| `dataset` | ABC-Bench task root, task glob, and task count. |
| `paths` | Workspace and result roots. |
| `testing` | Whether to run each task's validation script. |
| `prediction` | Whether to train/use the duration predictor and where to save it. |
| `sidecar` | Telemetry, model geometry, dashboard port, and admission-control policy. |

Admission-control highlights:

- `threshold_percent` triggers pressure offload when free KV is below the
  configured percent of total capacity.
- `w_threshold` gates new admissions based on headroom.
- `initial_admit_interval_s` paces startup before the controller has stable
  active-agent averages.
- `max_fresh_admits_per_tick` limits fresh launches per tick.
- `max_active_agents` provides an optional hard cap.
- `fallback_long_tool_call_s` makes long-running tools offloadable even when
  no predictor is available.
- `offload_timeout_s` and `exact_freed_gb_timeout_s` control vLLM endpoint and
  exact accounting waits.

Model geometry fields must match the vLLM server:

- `num_layers`
- `num_kv_heads`
- `head_dim`
- `block_size`
- `dtype`
- `total_gpu_blocks`

If these drift from server settings, sidecar GB values and thresholds will be
wrong.

## `start_vllm.sh`

This script starts vLLM on Ascend NPU using the native bare-metal path:

```bash
bash start_vllm.sh config/vllm_config.yaml
```

### Responsibilities

- Load `config/vllm_config.yaml`.
- Source Ascend toolkit and ATB environment scripts.
- Create a project-local vLLM environment from the configured shared Python.
- Install or update vLLM/vllm-ascend dependencies.
- Clone or prepare local source trees when needed.
- Apply agent-aware vLLM patches.
- Launch `vllm serve` with `AgentAwareOffloadingConnector`.
- Write a PID file.
- Wait for `/v1/models` to become healthy.
- Stop the process tree if startup validation fails.
- Verify the agent KV offload route is reachable.

The script uses careful process cleanup because vLLM can spawn worker processes
outside the original shell's immediate child list.

### Health and Cleanup

`wait_for_ready()` polls:

```text
http://localhost:<port>/v1/models
```

with the configured API key.

If startup validation fails, `stop_vllm_from_pid_file()` kills the root process,
descendants, and newly observed `VLLMWorker_TP` processes. It removes the PID
file and worker snapshot after cleanup.

### Patch Application

Before launch, the script runs:

```bash
python src/vllm_patches/apply_patches.py --vllm-dir <managed-vllm-dir>
```

The patcher installs the connector and validates route/method wiring.

## `config/vllm_config.yaml`

Major fields:

| Field | Purpose |
| --- | --- |
| `native.ascend_toolkit_path` | Ascend toolkit env script. |
| `native.atb_path` | ATB env script. |
| `native.shared_venv_python` | Python interpreter used to create the local vLLM env. |
| `native.workdir` | Working directory for vLLM setup. |
| `native.devices` | `ASCEND_RT_VISIBLE_DEVICES` value. |
| `native.model_path` | Local model path passed to vLLM. |
| `native.served_model_name` | Model alias exposed by the server. |
| `native.host` / `native.port` | Bind address and port. |
| `native.api_key` | API key expected by the OpenAI-compatible server. |
| `native.tensor_parallel_size` | vLLM tensor parallel size. |
| `native.dtype` | Model dtype. |
| `native.extra_args` | Extra `vllm serve` flags. |
| `native.pid_file` | PID file for health/cleanup. |
| `sidecar.*` | Matching model geometry for KV GB conversion. |

`native.extra_args` must stay consistent with sidecar geometry. In particular,
`--block-size` and `--num-gpu-blocks-override` must match the sidecar's
`block_size` and `total_gpu_blocks`.

## Operational Checks

Syntax:

```bash
bash -n run_abc-bench.sh
bash -n start_vllm.sh
```

Wrapper dry run:

```bash
bash run_abc-bench.sh --config config/abc-bench_config.yaml --dry-run
```

vLLM health:

```bash
curl -H "Authorization: Bearer <api-key>" \
  http://127.0.0.1:<port>/v1/models
```

Dashboard health:

```bash
curl http://127.0.0.1:8765/healthz
```

## Change Guidelines

- Keep YAML keys stable where possible; scripts flatten specific dotted paths.
- Add new config keys to the parser, resolved-config display, and command
  construction in the same change.
- Keep benchmark dependencies separate from the vLLM runtime environment.
- Treat `start_vllm.sh` cleanup logic carefully; failed startup should not
  leave orphan worker processes.
- Keep vLLM launch flags and sidecar geometry synchronized.
