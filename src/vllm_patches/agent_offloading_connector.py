"""Agent-aware KV offloading connector for vLLM/vllm-ascend 0.13.

This connector intentionally reuses vLLM's OffloadingConnector worker and
transfer specs.  The only changed part is scheduler-side policy:

* the sidecar can mark an idle long-tool-call agent for offload;
* the connector saves that agent's known KV blocks to the configured CPU
  offloading manager using the same async transfer path as OffloadingConnector;
* when the agent is readmitted, normal OffloadingConnector prefix lookup loads
  the saved blocks for the next request.

No Ascend-specific module is imported here.  The NPU implementation is still
selected by kv_connector_extra_config.spec_module_path/spec_name, e.g.
vllm_ascend.kv_offload.npu.NPUOffloadingSpec.
"""

from __future__ import annotations

import base64
import time
from collections import defaultdict
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    GPULoadStoreSpec,
    OffloadingConnector,
    OffloadingConnectorMetadata,
    OffloadingConnectorScheduler,
    OffloadingConnectorWorker,
    logger,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


_AGENT_STORE_PREFIX = "__agent_offload_store__:"


def _encode_id(value: str) -> str:
    raw = str(value).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_id(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii")).decode("utf-8")


def _make_agent_store_job_id(agent_id: str, req_id: str, seq: int) -> str:
    return f"{_AGENT_STORE_PREFIX}{_encode_id(agent_id)}:{_encode_id(req_id)}:{seq}"


def _parse_agent_store_job_id(job_id: str) -> tuple[str, str] | None:
    text = str(job_id)
    if not text.startswith(_AGENT_STORE_PREFIX):
        return None
    try:
        agent_b64, req_b64, _seq = text[len(_AGENT_STORE_PREFIX):].rsplit(":", 2)
        return _decode_id(agent_b64), _decode_id(req_b64)
    except Exception:
        logger.warning("Could not parse synthetic agent offload job id %r", job_id)
        return None


class AgentAwareOffloadingConnector(OffloadingConnector):
    """OffloadingConnector with agent-aware save/readmit policy."""

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: Any | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        extra = getattr(vllm_config.kv_transfer_config,
                        "kv_connector_extra_config", {}) or {}
        if role == KVConnectorRole.SCHEDULER:
            assert self.connector_scheduler is not None
            self.connector_scheduler = AgentAwareOffloadingScheduler(
                self.connector_scheduler, extra)
        elif role == KVConnectorRole.WORKER:
            assert self.connector_worker is not None
            self.connector_worker = AgentAwareOffloadingWorker(
                self.connector_worker.spec)

    def register_agent_request(self, agent_id: str, request_id: str) -> None:
        assert self.connector_scheduler is not None
        scheduler = self.connector_scheduler
        if hasattr(scheduler, "register_agent_request"):
            scheduler.register_agent_request(agent_id, request_id)

    def offload_agent_kv(
        self,
        agent_id: str,
        only_ref_cnt_zero: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert self.connector_scheduler is not None
        scheduler = self.connector_scheduler
        if not hasattr(scheduler, "offload_agent_kv"):
            return {
                "offloaded": False,
                "reason": "agent-aware scheduler is unavailable",
            }
        return scheduler.offload_agent_kv(
            agent_id, only_ref_cnt_zero=only_ref_cnt_zero, **kwargs)

    def restore_agent_kv(self, agent_id: str) -> dict[str, Any]:
        assert self.connector_scheduler is not None
        scheduler = self.connector_scheduler
        if not hasattr(scheduler, "restore_agent_kv"):
            return {"restored": False, "reason": "agent-aware scheduler is unavailable"}
        return scheduler.restore_agent_kv(agent_id)

    def release_agent_kv(
        self,
        agent_id: str,
        reason: str = "release",
    ) -> dict[str, Any]:
        assert self.connector_scheduler is not None
        scheduler = self.connector_scheduler
        if not hasattr(scheduler, "release_agent_kv"):
            return {"released": False, "reason": "agent-aware scheduler is unavailable"}
        return scheduler.release_agent_kv(agent_id, reason=reason)

    def get_agent_kv_usage(self, agent_id: str) -> dict[str, Any]:
        assert self.connector_scheduler is not None
        scheduler = self.connector_scheduler
        if not hasattr(scheduler, "get_agent_kv_usage"):
            return {
                "agent_id": str(agent_id or ""),
                "available": False,
                "reason": "agent-aware scheduler is unavailable",
            }
        return scheduler.get_agent_kv_usage(agent_id)


class AgentAwareOffloadingScheduler:
    """Scheduler-side policy wrapper around OffloadingConnectorScheduler."""

    def __init__(
        self,
        base: OffloadingConnectorScheduler,
        extra_config: dict[str, Any],
    ):
        self._base = base
        self._agent_to_requests: dict[str, set[str]] = defaultdict(set)
        self._request_to_agent: dict[str, str] = {}
        self._agent_snapshots: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._held_agent_requests: dict[str, set[str]] = defaultdict(set)
        self._held_request_snapshots: dict[str, dict[str, Any]] = {}
        self._release_ready_req_ids: set[str] = set()
        self._pending_agent_offloads: set[str] = set()
        self._agent_store_jobs: dict[str, str] = {}
        self._agent_store_real_reqs: dict[str, str] = {}
        self._agent_real_req_pending_jobs: dict[str, set[str]] = defaultdict(set)
        self._job_seq = 0
        self._hold_finished_requests = bool(
            extra_config.get("agent_hold_finished_requests", True))
        try:
            self._hold_ttl_s = max(
                0.0, float(extra_config.get("agent_hold_ttl_s", 300.0)))
        except (TypeError, ValueError):
            self._hold_ttl_s = 300.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def register_agent_request(self, agent_id: str, request_id: str) -> None:
        if not agent_id or not request_id:
            return
        agent_id = str(agent_id)
        request_id = str(request_id)
        self._request_to_agent[request_id] = agent_id
        self._agent_to_requests[agent_id].add(request_id)

    def offload_agent_kv(
        self,
        agent_id: str,
        only_ref_cnt_zero: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not agent_id:
            return {
                "offloaded": False,
                "pending": False,
                "reason": "missing agent_id",
            }

        agent_id = str(agent_id)
        self._release_stale_holds()
        held_req_ids = list(self._held_agent_requests.get(agent_id, ()))
        snapshots = {
            req_id: self._held_request_snapshots[req_id]
            for req_id in held_req_ids
            if req_id in self._held_request_snapshots
        }
        known_blocks = sum(
            len(snapshot.get("block_hashes", ())) for snapshot in snapshots.values())
        active_reqs = len(self._agent_to_requests.get(agent_id, ()))
        if known_blocks <= 0:
            return {
                "offloaded": False,
                "pending": False,
                "known_blocks": known_blocks,
                "held_requests": len(snapshots),
                "active_requests": active_reqs,
                "reason": "no held KV blocks for agent",
            }

        self._pending_agent_offloads.add(agent_id)
        offload_jobs = sum(
            len(self._agent_real_req_pending_jobs.get(req_id, ()))
            for req_id in snapshots)
        return {
            "offloaded": True,
            "pending": True,
            "known_blocks": known_blocks,
            "held_requests": len(snapshots),
            "active_requests": active_reqs,
            "offload_jobs": offload_jobs,
            "reason": "queued for async connector offload",
        }

    def release_agent_kv(
        self,
        agent_id: str,
        reason: str = "release",
    ) -> dict[str, Any]:
        if not agent_id:
            return {"released": False, "reason": "missing agent_id"}

        agent_id = str(agent_id)
        req_ids = list(self._held_agent_requests.get(agent_id, ()))
        released_req_ids: list[str] = []
        pending_req_ids: list[str] = []
        for req_id in req_ids:
            if self._request_has_pending_store(req_id):
                pending_req_ids.append(req_id)
                continue
            self._drop_hold(agent_id, req_id)
            released_req_ids.append(req_id)

        if not self._held_agent_requests.get(agent_id):
            self._pending_agent_offloads.discard(agent_id)

        return {
            "released": bool(released_req_ids),
            "held_requests": len(req_ids),
            "released_requests": len(released_req_ids),
            "pending_requests": len(pending_req_ids),
            "released_request_ids": released_req_ids,
            "pending_request_ids": pending_req_ids,
            "reason": reason,
        }

    def restore_agent_kv(self, agent_id: str) -> dict[str, Any]:
        if not agent_id:
            return {"restored": False, "reason": "missing agent_id"}
        agent_id = str(agent_id)
        self._release_stale_holds()
        pending = agent_id in self._pending_agent_offloads
        self._pending_agent_offloads.discard(agent_id)
        known_blocks = sum(
            len(snapshot.get("block_hashes", ()))
            for snapshot in self._agent_snapshots.get(agent_id, {}).values())
        held_requests = len(self._held_agent_requests.get(agent_id, ()))
        return {
            "restored": True,
            "pending_offload_cleared": pending,
            "known_blocks": known_blocks,
            "held_requests": held_requests,
            "reason": (
                "blocks will be loaded by OffloadingConnector prefix lookup "
                "when the readmitted agent submits its next request"
            ),
        }

    def get_agent_kv_usage(self, agent_id: str) -> dict[str, Any]:
        if not agent_id:
            return {
                "agent_id": "",
                "available": False,
                "reason": "missing agent_id",
            }

        agent_id = str(agent_id)
        self._release_stale_holds()

        active_req_ids = set(self._agent_to_requests.get(agent_id, ()))
        held_req_ids = set(self._held_agent_requests.get(agent_id, ()))

        for req_id in list(active_req_ids):
            req = self._base._requests.get(req_id)
            if req is None:
                active_req_ids.discard(req_id)
                self._agent_to_requests[agent_id].discard(req_id)
                if not self._agent_to_requests.get(agent_id):
                    self._agent_to_requests.pop(agent_id, None)
                self._request_to_agent.pop(req_id, None)
                self._drop_snapshot(agent_id, req_id)
                continue
            self._snapshot_request(req_id, req)

        snapshots = self._agent_snapshots.get(agent_id, {})
        requests: list[dict[str, Any]] = []
        total_resident_blocks = 0
        total_offloadable_blocks = 0
        scoped_req_ids = active_req_ids | held_req_ids
        for req_id in sorted(scoped_req_ids):
            active = req_id in active_req_ids
            held = req_id in held_req_ids
            snapshot = (
                self._held_request_snapshots.get(req_id)
                if held else snapshots.get(req_id)
            )
            block_ids = list((snapshot or {}).get("block_ids") or ())
            block_hashes = list((snapshot or {}).get("block_hashes") or ())
            resident_blocks = len(block_ids)
            offloadable_blocks = len(block_hashes)
            total_resident_blocks += resident_blocks
            total_offloadable_blocks += offloadable_blocks
            requests.append({
                "request_id": req_id,
                "kv_blocks": resident_blocks,
                "resident_kv_blocks": resident_blocks,
                "offloadable_kv_blocks": offloadable_blocks,
                "active": active,
                "held": held,
            })

        offload_jobs = sum(
            len(self._agent_real_req_pending_jobs.get(req_id, ()))
            for req_id in scoped_req_ids
        )
        return {
            "agent_id": agent_id,
            "available": True,
            "kv_blocks": total_resident_blocks,
            "kv_blocks_used": total_resident_blocks,
            "resident_kv_blocks": total_resident_blocks,
            "offloadable_kv_blocks": total_offloadable_blocks,
            "active_requests": len(active_req_ids),
            "held_requests": len(held_req_ids),
            "pending_offload": agent_id in self._pending_agent_offloads,
            "offload_jobs": offload_jobs,
            "requests": requests,
        }

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ):
        result = self._base.update_state_after_alloc(
            request, blocks, num_external_tokens)
        self._snapshot_request(request.request_id, request)
        return result

    def build_connector_meta(
        self,
        scheduler_output: Any,
    ) -> OffloadingConnectorMetadata:
        self._release_stale_holds()
        reqs_to_store: dict[str, Any] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(
                scheduler_output):
            if preempted:
                self._base._request_block_ids[req_id] = []

            if req_id not in self._base._request_block_ids:
                self._base._request_block_ids[req_id] = []

            if new_block_id_groups:
                new_block_ids = new_block_id_groups[0]
                self._base._request_block_ids[req_id] += new_block_ids

            req = self._base._requests.get(req_id)
            if req is None:
                continue

            self._snapshot_request(req_id, req)

        for agent_id in list(self._pending_agent_offloads):
            held_req_ids = list(self._held_agent_requests.get(agent_id, ()))
            if not held_req_ids:
                self._pending_agent_offloads.discard(agent_id)
                continue
            for req_id in held_req_ids:
                snapshot = self._held_request_snapshots.get(req_id)
                if snapshot is None:
                    continue
                job_id = self._next_agent_store_job_id(agent_id, req_id)
                store_spec = self._prepare_store(job_id, agent_id, req_id, snapshot)
                if store_spec is not None:
                    reqs_to_store[job_id] = store_spec

        meta = OffloadingConnectorMetadata(
            reqs_to_load=self._base._reqs_to_load,
            reqs_to_store=reqs_to_store,
        )
        self._base._reqs_to_load = {}
        return meta

    def update_connector_output(self, connector_output: Any):
        finished_sending = set(connector_output.finished_sending or ())
        agent_finished = {
            req_id for req_id in finished_sending
            if str(req_id).startswith(_AGENT_STORE_PREFIX)
        }
        release_ready = set(self._release_ready_req_ids)
        self._release_ready_req_ids.clear()

        self._base.update_connector_output(connector_output)

        for req_id in agent_finished:
            agent_id = self._agent_store_jobs.pop(req_id, None)
            real_req_id = self._agent_store_real_reqs.pop(req_id, None)
            if real_req_id is not None:
                pending = self._agent_real_req_pending_jobs.get(real_req_id)
                if pending is not None:
                    pending.discard(req_id)
                    if not pending:
                        self._agent_real_req_pending_jobs.pop(real_req_id, None)
                        if agent_id is not None:
                            self._drop_hold(agent_id, real_req_id)
                            if not self._held_agent_requests.get(agent_id):
                                self._pending_agent_offloads.discard(agent_id)
            if agent_id is not None:
                logger.debug("Agent %s KV offload store completed via %s",
                             agent_id, req_id)

        for req_id in finished_sending - agent_finished:
            snapshot = self._held_request_snapshots.get(req_id)
            if snapshot is not None:
                self._drop_hold(snapshot.get("agent_id"), req_id)

        remaining = (finished_sending - agent_finished) | release_ready
        try:
            connector_output.finished_sending = remaining
        except Exception:
            logger.warning(
                "Could not remove agent offload jobs from finished_sending; "
                "vLLM scheduler may try to free synthetic connector jobs")

    def request_finished(
        self,
        request: Any,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        self._snapshot_request(req_id, request, block_ids=block_ids)
        agent_id = self._request_to_agent.get(req_id)
        request_being_stored, params = self._base.request_finished(
            request, block_ids)
        held = False
        if agent_id and self._hold_finished_requests:
            held = self._hold_request(agent_id, req_id)
        agent_store_pending = bool(self._agent_real_req_pending_jobs.get(req_id))
        if agent_id:
            self._agent_to_requests[agent_id].discard(req_id)
        return request_being_stored or agent_store_pending or held, params

    def _snapshot_request(
        self,
        req_id: str,
        request: Any,
        block_ids: list[int] | None = None,
    ) -> None:
        agent_id = self._request_to_agent.get(req_id)
        if not agent_id:
            return

        if block_ids is None:
            block_ids = list(self._base._request_block_ids.get(req_id, ()))
        else:
            block_ids = list(block_ids)

        if not block_ids:
            self._drop_snapshot(agent_id, req_id)
            return

        num_offloadable_blocks = min(
            len(block_ids) // self._base.block_size_factor,
            len(getattr(request, "block_hashes", ()))
            // self._base.block_size_factor,
            getattr(request, "num_tokens", 0) // self._base.offloaded_block_size,
        )
        block_hashes: list[Any] = []
        if num_offloadable_blocks > 0:
            block_hashes = list(self._base._get_block_hashes(
                request, end_idx=num_offloadable_blocks))

        self._agent_snapshots[agent_id][req_id] = {
            "block_ids": block_ids,
            "block_hashes": block_hashes,
        }

    def _drop_snapshot(self, agent_id: str, req_id: str) -> None:
        snapshots = self._agent_snapshots.get(agent_id)
        if snapshots is None:
            return
        snapshots.pop(req_id, None)
        if not snapshots:
            self._agent_snapshots.pop(agent_id, None)

    def _next_agent_store_job_id(self, agent_id: str, req_id: str) -> str:
        self._job_seq += 1
        return _make_agent_store_job_id(agent_id, req_id, self._job_seq)

    def _prepare_store(
        self,
        job_id: str,
        agent_id: str,
        req_id: str,
        snapshot: dict[str, Any],
    ) -> Any | None:
        block_hashes = list(snapshot.get("block_hashes") or ())
        block_ids = list(snapshot.get("block_ids") or ())
        if not block_hashes or not block_ids:
            return None
        if self._agent_real_req_pending_jobs.get(req_id):
            return None

        store_output = self._base.manager.prepare_store(block_hashes)
        if store_output is None:
            logger.warning("Agent %s: cannot prepare CPU KV store", agent_id)
            return None
        if not store_output.block_hashes_to_store:
            self._base.manager.touch(block_hashes)
            if req_id in self._held_request_snapshots:
                self._mark_held_request_ready_to_free(agent_id, req_id)
            return None

        block_hashes_to_store = set(store_output.block_hashes_to_store)
        src_block_ids: list[int] = []
        for offloaded_block_idx, block_hash in enumerate(block_hashes):
            if block_hash not in block_hashes_to_store:
                continue
            gpu_block_idx = offloaded_block_idx * self._base.block_size_factor
            for offset in range(self._base.block_size_factor):
                idx = gpu_block_idx + offset
                if idx < len(block_ids):
                    src_block_ids.append(block_ids[idx])

        if not src_block_ids:
            return None

        self._base.manager.touch(block_hashes)
        self._base._reqs_being_stored[job_id].update(block_hashes_to_store)
        self._agent_store_jobs[job_id] = agent_id
        self._agent_store_real_reqs[job_id] = req_id
        self._agent_real_req_pending_jobs[req_id].add(job_id)
        logger.debug("Agent %s offloading %d KV blocks via %s",
                     agent_id, len(block_hashes_to_store), job_id)
        return GPULoadStoreSpec(src_block_ids), store_output.store_spec

    def _hold_request(self, agent_id: str, req_id: str) -> bool:
        snapshot = self._agent_snapshots.get(agent_id, {}).get(req_id)
        if snapshot is None:
            return False
        if not snapshot.get("block_ids") or not snapshot.get("block_hashes"):
            return False
        held = dict(snapshot)
        held["agent_id"] = agent_id
        held["held_since"] = time.monotonic()
        self._held_request_snapshots[req_id] = held
        self._held_agent_requests[agent_id].add(req_id)
        return True

    def _drop_hold(self, agent_id: str | None, req_id: str) -> None:
        snapshot = self._held_request_snapshots.pop(req_id, None)
        if agent_id is None and snapshot is not None:
            agent_id = snapshot.get("agent_id")
        if agent_id:
            held = self._held_agent_requests.get(agent_id)
            if held is not None:
                held.discard(req_id)
                if not held:
                    self._held_agent_requests.pop(agent_id, None)

    def _mark_held_request_ready_to_free(self, agent_id: str, req_id: str) -> None:
        if req_id not in self._held_request_snapshots:
            return
        self._drop_hold(agent_id, req_id)
        self._release_ready_req_ids.add(req_id)
        if not self._held_agent_requests.get(agent_id):
            self._pending_agent_offloads.discard(agent_id)

    def _request_has_pending_store(self, req_id: str) -> bool:
        if self._agent_real_req_pending_jobs.get(req_id):
            return True
        return bool(self._base._reqs_being_stored.get(req_id))

    def _release_stale_holds(self) -> None:
        if self._hold_ttl_s <= 0:
            return
        cutoff = time.monotonic() - self._hold_ttl_s
        for req_id, snapshot in list(self._held_request_snapshots.items()):
            held_since = float(snapshot.get("held_since") or 0.0)
            if held_since <= 0.0 or held_since > cutoff:
                continue
            if self._request_has_pending_store(req_id):
                continue
            agent_id = str(snapshot.get("agent_id") or "")
            self._mark_held_request_ready_to_free(agent_id, req_id)


class AgentAwareOffloadingWorker(OffloadingConnectorWorker):
    """Worker that completes synthetic agent store jobs without request finish."""

    def __init__(self, spec: Any):
        super().__init__(spec)
        self._agent_store_real_req_jobs: dict[str, set[str]] = defaultdict(set)

    def start_store_kv(self, metadata: OffloadingConnectorMetadata):
        for req_id, transfer_spec in metadata.reqs_to_store.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, True)
            self._store_jobs[req_id].add(job_id)
            parsed = _parse_agent_store_job_id(req_id)
            if parsed is not None:
                _agent_id, real_req_id = parsed
                self._agent_store_real_req_jobs[real_req_id].add(req_id)
            assert self.worker.transfer_async(job_id, transfer_spec)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_sending = set()
        finished_recving = set()

        for job_id, success in self.worker.get_finished():
            assert success
            req_id, store = self._jobs.pop(job_id)
            if store:
                req_jobs = self._store_jobs[req_id]
                req_jobs.remove(job_id)
                if req_jobs:
                    continue

                if str(req_id).startswith(_AGENT_STORE_PREFIX):
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
                    parsed = _parse_agent_store_job_id(req_id)
                    if parsed is not None:
                        _agent_id, real_req_id = parsed
                        real_req_jobs = self._agent_store_real_req_jobs.get(
                            real_req_id)
                        if real_req_jobs is not None:
                            real_req_jobs.discard(req_id)
                            if not real_req_jobs:
                                self._agent_store_real_req_jobs.pop(
                                    real_req_id, None)
                                self._finished_reqs_waiting_for_store.discard(
                                    real_req_id)
                                finished_sending.add(real_req_id)
                elif req_id in self._finished_reqs_waiting_for_store:
                    if not self._agent_store_real_req_jobs.get(req_id):
                        self._finished_reqs_waiting_for_store.remove(req_id)
                        finished_sending.add(req_id)
                        del self._store_jobs[req_id]
            else:
                req_job = self._load_job[req_id]
                assert job_id == req_job
                del self._load_job[req_id]
                finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending_req_jobs = self._store_jobs.get(req_id)
            pending_agent_jobs = self._agent_store_real_req_jobs.get(req_id)
            if pending_req_jobs or pending_agent_jobs:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending_req_jobs is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving
