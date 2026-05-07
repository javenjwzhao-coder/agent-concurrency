# vLLM Patch and Agent-Aware Connector

Components:

- `src/vllm_patches/apply_patches.py`
- `src/vllm_patches/agent_offloading_connector.py`
- `start_vllm.sh`

The sidecar needs vLLM to understand `agent_id`, expose per-agent KV usage, and
hold finished request KV blocks until policy decides whether to release or CPU
offload them. The patcher injects those hooks into a vLLM/vllm-ascend 0.13
install and installs a custom connector that reuses vLLM's existing
OffloadingConnector transfer machinery.

## High-Level Contract

The runtime boundary is:

```text
runner -> vLLM chat request with agent_id
vLLM serving layer -> request_id registered under agent_id
sidecar -> /agent_kv_cache/* control endpoints
vLLM scheduler -> AgentAwareOffloadingConnector
connector -> hold, release, async store, restore by prefix lookup
```

The sidecar chooses whether an agent should offload. vLLM and the connector
choose when blocks are safe to free.

## Patcher Targets

`apply_patches.py` patches these files inside the target `vllm` package:

| Target | Patch purpose |
| --- | --- |
| `entrypoints/openai/protocol.py` | Add `agent_id` to chat requests and KV fields to usage info. |
| `entrypoints/openai/serving_chat.py` | Compute KV blocks/GB in usage and register `agent_id -> request_id`. |
| `entrypoints/openai/api_server.py` | Add `/agent_kv_cache/offload`, `/usage`, `/restore`, and `/release` routes. |
| `v1/engine/async_llm.py` | Forward async agent-KV calls to engine core. |
| `v1/engine/core_client.py` | Add sync/async forwarding for in-process and multiprocess clients. |
| `v1/engine/core.py` | Forward agent-KV calls to the scheduler. |
| `v1/core/sched/scheduler.py` | Forward agent-KV calls to the configured connector scheduler. |
| `distributed/kv_transfer/kv_connector/v1/agent_offloading_connector.py` | Installed connector implementation. |

The script creates `.bak` backups when no backup exists, or restores from
existing backups before applying patches. This keeps repeated patch runs from
compounding old edits.

## Applying the Patch

Run on the machine where vLLM is installed:

```bash
python src/vllm_patches/apply_patches.py \
  --vllm-dir /path/to/site-packages/vllm
```

`start_vllm.sh` applies the patch automatically to its managed vLLM environment
before launch.

After patching, `validate()` checks for:

- syntax errors in patched files
- protocol usage fields
- `agent_id`
- telemetry helper in `serving_chat.py`
- connector installation
- forwarding methods through async LLM, core client, engine core, and scheduler
- all four API routes

## Public vLLM API Routes

The patch adds:

| Endpoint | Method | Payload | Purpose |
| --- | --- | --- | --- |
| `/agent_kv_cache/usage` | GET | query `agent_id` | Diagnostic scheduler-owned agent KV usage. The sidecar admission loop uses `_LIVE_AGENTS` response telemetry instead. |
| `/agent_kv_cache/offload` | POST | `{"agent_id": "...", "only_ref_cnt_zero": true}` | Queue held agent KV for async CPU offload. |
| `/agent_kv_cache/release` | POST | `{"agent_id": "...", "reason": "..."}` | Release held KV without CPU offload. |
| `/agent_kv_cache/restore` | POST | `{"agent_id": "..."}` | Clear pending offload/readmit state. Next request restores by prefix lookup. |

The route functions tolerate sync or async engine client methods.

## Connector Configuration

`start_vllm.sh` launches vLLM with a KV transfer config similar to:

```json
{
  "kv_connector": "AgentAwareOffloadingConnector",
  "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.agent_offloading_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "num_cpu_blocks": 8192,
    "caching_hash_algo": "sha256_cbor",
    "spec_name": "NPUOffloadingSpec",
    "spec_module_path": "vllm_ascend.kv_offload.npu",
    "agent_hold_finished_requests": true,
    "agent_hold_ttl_s": 300.0
  }
}
```

No Ascend-specific module is imported by the connector code itself. The
configured OffloadingSpec selects the Ascend transfer implementation.

## Connector Classes

### `AgentAwareOffloadingConnector`

Wraps vLLM's `OffloadingConnector`. Depending on role:

- scheduler role wraps the scheduler side with
  `AgentAwareOffloadingScheduler`
- worker role wraps the worker side with `AgentAwareOffloadingWorker`

It exposes the methods used by the patched scheduler:

- `register_agent_request()`
- `offload_agent_kv()`
- `restore_agent_kv()`
- `release_agent_kv()`
- `get_agent_kv_usage()`

### `AgentAwareOffloadingScheduler`

Owns policy-facing state:

| State | Purpose |
| --- | --- |
| `_agent_to_requests` | Active request ids per agent. |
| `_request_to_agent` | Reverse request id lookup. |
| `_agent_snapshots` | Latest block id/hash snapshots by agent/request. |
| `_held_agent_requests` | Finished requests whose blocks are deliberately held. |
| `_held_request_snapshots` | Held block ids, hashes, agent id, and hold timestamp. |
| `_pending_agent_offloads` | Agents selected by the sidecar for async offload. |
| `_agent_store_jobs` | Synthetic store job id -> agent id. |
| `_agent_store_real_reqs` | Synthetic store job id -> real request id. |
| `_agent_real_req_pending_jobs` | Real request id -> synthetic jobs still running. |

### `AgentAwareOffloadingWorker`

Extends the worker side so synthetic agent store jobs can complete without
confusing vLLM's normal request-finish flow.

## Request Registration

`serving_chat.py` registers the mapping before generation:

```text
agent_id -> request_id
```

The runner supplies `agent_id` through `extra_body` on each OpenAI-compatible
chat request. Without this field, requests are treated as normal vLLM requests
and are not held for sidecar policy.

## Hold-Before-Free Safety

When vLLM tells the connector a request is finished:

1. `request_finished()` snapshots block ids and block hashes.
2. The base OffloadingConnector still runs its normal finish logic.
3. If the request belongs to an agent and holding is enabled, the connector
   records the request in `_held_agent_requests` whenever resident block ids
   exist. Hash-backed block metadata is not required for holding.
4. The connector returns `True` to delay freeing.
5. Blocks remain resident until release or async store completion.

This is the central safety invariant: finished agent request blocks are held
before they can enter vLLM's free queue.

Resident and offloadable accounting are intentionally separate. Resident
`block_ids` let connector control responses report nonzero native KV while
`block_hashes` are required only for CPU offload. If a held request has
resident blocks but no hash-backed blocks, the offload endpoint rejects the
attempt as `no offloadable held KV blocks for agent` while still reporting the
resident block count.

## Offload Flow

When the sidecar calls `/agent_kv_cache/offload`:

1. `offload_agent_kv()` checks held snapshots for that agent.
2. If no held blocks exist, the request is rejected with a reason.
3. Otherwise the agent id is added to `_pending_agent_offloads`.
4. On the next scheduler metadata build, `build_connector_meta()` creates
   synthetic store jobs for held snapshots.
5. `_prepare_store()` asks the base offload manager which block hashes need to
   be stored and maps them to source GPU/NPU block ids.
6. The worker starts async transfer jobs with those synthetic ids.
7. `update_connector_output()` observes synthetic completion, drops the hold,
   and lets the real request finish.

The offload endpoint can return `pending = true` because actual block freeing
depends on the async transfer and the next scheduler cycle.

## Release Flow

When the sidecar or runner calls `/agent_kv_cache/release`:

1. `release_agent_kv()` scans held requests for the agent.
2. Requests with pending store jobs stay held.
3. Requests without pending stores are dropped from the held sets.
4. The patched scheduler frees released request blocks through `_free_blocks()`.

Release is used for short tool calls, completed non-offloaded calls, final
messages, errors, and TTL cleanup.

## Restore Flow

When an offloaded agent is readmitted:

1. The sidecar calls `/agent_kv_cache/restore`.
2. `restore_agent_kv()` clears pending-offload state.
3. It reports known block and held-request counts.
4. The next request from that agent restores CPU KV through the existing
   OffloadingConnector prefix lookup path.

Restore does not directly copy blocks during the HTTP request.

## Usage Reporting

`get_agent_kv_usage(agent_id)` reports scheduler-owned state:

- resident KV blocks
- offloadable KV blocks
- active request count
- held request count
- pending offload boolean
- offload job count
- per-request active/held details

The sidecar admission loop no longer refreshes live state through this route;
it scores agents from `_LIVE_AGENTS`, which is updated from response usage
telemetry after each LLM call.

## Synthetic Job IDs

Synthetic store jobs use the prefix:

```text
__agent_offload_store__:
```

Agent id and real request id are URL-safe base64 encoded into the job id. The
worker and scheduler parse that id to map synthetic completion back to the real
request hold.

## TTL Cleanup

`agent_hold_ttl_s` controls stale hold cleanup. If a held request has no pending
store and its hold age exceeds the TTL, the scheduler marks it ready to free.
Set the TTL carefully: too low can reduce offload opportunity; too high can
hold KV longer than intended after abnormal runner behavior.

## Failure Behavior

- Missing connector support returns structured `available/restored/offloaded`
  false payloads rather than crashing routes.
- Offload without held KV returns `offloaded = false` with a reason.
- Store preparation can return no-op when all hashes are already present in CPU
  cache; the held request can then become ready to free.
- Synthetic job ids are removed from `finished_sending` before vLLM treats them
  as real request ids.

## Change Guidelines

- Preserve the hold-before-free invariant.
- Keep the connector using vLLM's OffloadingConnector transfer path rather than
  adding direct device-specific copy logic here.
- Update `apply_patches.py` and `tests/test_vllm_patch_text.py` together when
  changing route names or forwarding method names.
- Keep route payloads stable because `src/sidecar.py` calls them directly.
- Validate on the target vLLM/vllm-ascend version after changing anchors.
