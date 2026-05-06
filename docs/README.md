# Component Documentation

This directory holds the detailed docs for the major moving parts in the
agent-concurrency codebase. The top-level `README.md` stays short; this folder
is the place to look when changing behavior or debugging a run.

## Components

| Doc | Covers |
| --- | --- |
| [ABC-Bench runner](abc_bench_runner.md) | OpenHands task execution, agent phase tracking, live state, launch modes, and run artifacts. |
| [Tool trace collector](tool_trace_collector.md) | Action/observation matching, trace schema, CSV/JSONL artifacts, and extraction rules. |
| [Tool duration predictor](tool_duration_predictor.md) | Feature engineering, model training, remaining-time inference, metrics, and sidecar integration. |
| [Sidecar admission controller](sidecar_admission_controller.md) | vLLM polling, KV accounting, admission gates, offload decisions, and tick records. |
| [Dashboard and HTTP feed](dashboard_and_feed.md) | `/state`, `/stream`, live/replay modes, standalone snapshots, and browser rendering. |
| [vLLM patch and connector](vllm_patch_and_connector.md) | Patch targets, API routes, request forwarding, held KV blocks, async CPU offload, and restore behavior. |
| [Scripts and configs](scripts_and_configs.md) | YAML wrappers, vLLM bootstrap, config fields, environment assumptions, and operational flow. |
| [Testing guide](testing.md) | Test files, coverage areas, direct-script fallbacks, and dependency expectations. |

## Architecture Reference

For the end-to-end control-loop design and diagrams, see
[agent_kv_offload_architecture.md](agent_kv_offload_architecture.md).

## Reading Order

Start with [ABC-Bench runner](abc_bench_runner.md), then
[Sidecar admission controller](sidecar_admission_controller.md), then
[vLLM patch and connector](vllm_patch_and_connector.md). Those three docs
explain the runtime path. The trace and predictor docs explain the data path
that feeds the controller.
