#!/usr/bin/env python3
"""
Detailed tool-call tracing for ABC-Bench runs.

This module intentionally avoids importing OpenHands classes.  The collector
duck-types action and observation events so it can be tested with lightweight
fake objects and can tolerate small SDK attribute-name changes.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional


LiveSnapshotFn = Callable[[], dict[str, Any]]


TOOL_CALL_CSV_FIELDS = [
    "agent_id",
    "task_id",
    "conversation_id",
    "tool_seq",
    "action_id",
    "tool_call_id",
    "tool_name",
    "tool_command",
    "tool_args_json",
    "tool_args_keys",
    "tool_payload_bytes",
    "action_summary",
    "action_type",
    "start_ts",
    "end_ts",
    "duration_s",
    "outcome",
    "error_type",
    "returncode",
    "timed_out",
    "observation_bytes",
    "observation_line_count",
    "observation_preview",
    "observation_sha256",
    "start_active_agents",
    "start_active_tool_calls",
    "start_cumulative_reasoning_s",
    "start_kv_blocks",
    "start_kv_gb",
    "start_live_state",
    "start_live_state_since",
    "end_active_agents",
    "end_active_tool_calls",
    "end_kv_blocks",
    "end_kv_gb",
    "detail",
]


def json_dumps_safe(obj: Any, **kw: Any) -> str:
    return json.dumps(obj, default=str, **kw)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json_dumps_safe(value, sort_keys=True)
    return str(value)


def _as_optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_optional_int(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def _as_bool(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if value is None or value == "":
        return 0
    if isinstance(value, (int, float)):
        return int(bool(value))
    return int(str(value).strip().lower() in {"1", "true", "yes", "timeout", "timed_out"})


def _snapshot_get(snapshot: dict[str, Any], key: str, default: Any = "") -> Any:
    value = snapshot.get(key, default)
    return default if value is None else value


@dataclass
class PendingToolCall:
    """Action-side metadata for an in-flight tool call."""

    action_id: str
    tool_call_id: str
    tool_name: str
    tool_command: str
    tool_args: dict[str, Any]
    tool_args_json: str
    tool_args_keys: str
    tool_payload_bytes: int
    start_dt: datetime
    action_summary: str = ""
    action_type: str = ""
    conversation_id: str = ""
    start_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def detail(self) -> str:
        return self.action_summary or self.tool_name


@dataclass
class ToolCallRecord:
    """One completed or finalized tool-call trace row."""

    agent_id: str
    task_id: str
    conversation_id: str
    tool_seq: int
    action_id: str
    tool_call_id: str
    tool_name: str
    tool_command: str
    tool_args_json: str
    tool_args_keys: str
    tool_payload_bytes: int
    action_summary: str
    action_type: str
    start_ts: str
    end_ts: str
    duration_s: float
    outcome: str
    error_type: str = ""
    returncode: str = ""
    timed_out: int = 0
    observation_bytes: int = 0
    observation_line_count: int = 0
    observation_preview: str = ""
    observation_sha256: str = ""
    start_active_agents: int = 0
    start_active_tool_calls: int = 0
    start_cumulative_reasoning_s: float = 0.0
    start_kv_blocks: str = ""
    start_kv_gb: str = ""
    start_live_state: str = ""
    start_live_state_since: str = ""
    end_active_agents: int = 0
    end_active_tool_calls: int = 0
    end_kv_blocks: str = ""
    end_kv_gb: str = ""
    detail: str = ""


class ToolCallTraceCollector:
    """Collects detailed, feature-engineering-friendly tool-call records."""

    def __init__(
        self,
        agent_id: str,
        task_id: str,
        live_snapshot_fn: Optional[LiveSnapshotFn] = None,
        preview_chars: int = 2000,
    ) -> None:
        self.agent_id = agent_id
        self.task_id = task_id
        self.preview_chars = preview_chars
        self._live_snapshot_fn = live_snapshot_fn
        self._pending: dict[str, PendingToolCall] = {}
        self.records: list[ToolCallRecord] = []
        self._seq = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def record_action(
        self,
        event: Any,
        start_dt: datetime,
        conversation_id: str = "",
    ) -> PendingToolCall:
        """Record an ActionEvent and return the pending call metadata."""
        tool_name = getattr(event, "tool_name", "unknown") or "unknown"
        action_id = _as_optional_str(getattr(event, "id", "")) or f"seq-{self._seq}"
        tool_call_id = _as_optional_str(getattr(event, "tool_call_id", ""))
        raw_args = self.extract_tool_args(event)
        args_json = json_dumps_safe(raw_args, sort_keys=True)
        keys = sorted(str(k) for k in raw_args.keys())
        action = getattr(event, "action", None)
        action_type = type(action).__name__ if action is not None else type(event).__name__
        summary = (getattr(event, "summary", "") or "").strip()

        pending = PendingToolCall(
            action_id=action_id,
            tool_call_id=tool_call_id,
            tool_name=str(tool_name),
            tool_command=self.extract_terminal_command(event, raw_args),
            tool_args=raw_args,
            tool_args_json=args_json,
            tool_args_keys=",".join(keys),
            tool_payload_bytes=len(args_json.encode("utf-8")),
            start_dt=start_dt,
            action_summary=summary,
            action_type=action_type,
            conversation_id=conversation_id,
            start_snapshot=self._snapshot(),
        )
        self._pending[self._pending_key(pending)] = pending
        return pending

    def record_observation(
        self,
        event: Any,
        end_dt: datetime,
        outcome: str = "ok",
        conversation_id: str = "",
    ) -> Optional[ToolCallRecord]:
        """Match an ObservationEvent to its action and append a detailed row."""
        pending = self.pop_pending(event)
        if pending is None:
            return None
        record = self._make_record(
            pending=pending,
            end_dt=end_dt,
            outcome=outcome,
            conversation_id=conversation_id or pending.conversation_id,
            observation_event=event,
        )
        self.records.append(record)
        self._seq += 1
        return record

    def finalize_unfinished(
        self,
        end_dt: datetime,
        outcome: str = "unfinished",
        detail: str = "tool call ended without observation",
        conversation_id: str = "",
    ) -> list[ToolCallRecord]:
        """Finalize pending calls so partial traces survive failed runs."""
        finalized: list[ToolCallRecord] = []
        for key, pending in list(self._pending.items()):
            self._pending.pop(key, None)
            record = self._make_record(
                pending=pending,
                end_dt=end_dt,
                outcome=outcome,
                conversation_id=conversation_id or pending.conversation_id,
                observation_event=None,
                detail=detail,
            )
            self.records.append(record)
            self._seq += 1
            finalized.append(record)
        return finalized

    def pop_pending(self, event: Any) -> Optional[PendingToolCall]:
        action_id = _as_optional_str(getattr(event, "action_id", ""))
        if action_id:
            found = self._pending.pop(f"action:{action_id}", None)
            if found is not None:
                return found
            for key, cand in list(self._pending.items()):
                if cand.action_id == action_id:
                    return self._pending.pop(key)

        tool_call_id = _as_optional_str(getattr(event, "tool_call_id", ""))
        if tool_call_id:
            found = self._pending.pop(f"tool_call:{tool_call_id}", None)
            if found is not None:
                return found
            for key, cand in list(self._pending.items()):
                if cand.tool_call_id == tool_call_id:
                    return self._pending.pop(key)

        if len(self._pending) == 1:
            _, found = self._pending.popitem()
            return found
        return None

    def write_jsonl(self, out_dir: Path, filename: Optional[str] = None) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (filename or f"{self.agent_id}_tool_calls.jsonl")
        with path.open("w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json_dumps_safe(dataclasses.asdict(rec), sort_keys=True) + "\n")
        return path

    def write_csv(self, out_dir: Path, filename: Optional[str] = None) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (filename or f"{self.agent_id}_tool_calls.csv")
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TOOL_CALL_CSV_FIELDS)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(dataclasses.asdict(rec))
        return path

    def write_artifacts(self, out_dir: Path) -> dict[str, Path]:
        return {
            "jsonl": self.write_jsonl(out_dir),
            "csv": self.write_csv(out_dir),
        }

    def _make_record(
        self,
        pending: PendingToolCall,
        end_dt: datetime,
        outcome: str,
        conversation_id: str,
        observation_event: Any = None,
        detail: str = "",
    ) -> ToolCallRecord:
        duration = max(0.0, (end_dt - pending.start_dt).total_seconds())
        content = self.extract_observation_content(observation_event) if observation_event is not None else ""
        content_bytes = content.encode("utf-8", errors="replace")
        preview = content[: self.preview_chars]
        sha = hashlib.sha256(content_bytes).hexdigest() if content else ""
        end_snapshot = self._snapshot()

        return ToolCallRecord(
            agent_id=self.agent_id,
            task_id=self.task_id,
            conversation_id=conversation_id,
            tool_seq=self._seq,
            action_id=pending.action_id,
            tool_call_id=pending.tool_call_id,
            tool_name=pending.tool_name,
            tool_command=pending.tool_command,
            tool_args_json=pending.tool_args_json,
            tool_args_keys=pending.tool_args_keys,
            tool_payload_bytes=pending.tool_payload_bytes,
            action_summary=pending.action_summary,
            action_type=pending.action_type,
            start_ts=iso_utc(pending.start_dt),
            end_ts=iso_utc(end_dt),
            duration_s=round(duration, 6),
            outcome=outcome,
            error_type=self.extract_error_type(observation_event),
            returncode=self.extract_returncode(observation_event),
            timed_out=self.extract_timed_out(observation_event),
            observation_bytes=len(content_bytes),
            observation_line_count=len(content.splitlines()) if content else 0,
            observation_preview=preview,
            observation_sha256=sha,
            start_active_agents=int(_snapshot_get(pending.start_snapshot, "active_agents", 0) or 0),
            start_active_tool_calls=int(_snapshot_get(pending.start_snapshot, "active_tool_calls", 0) or 0),
            start_cumulative_reasoning_s=float(
                _snapshot_get(pending.start_snapshot, "cumulative_reasoning_s", 0.0) or 0.0
            ),
            start_kv_blocks=_as_optional_int(_snapshot_get(pending.start_snapshot, "kv_blocks", "")),
            start_kv_gb=_as_optional_str(_snapshot_get(pending.start_snapshot, "kv_gb", "")),
            start_live_state=_as_optional_str(_snapshot_get(pending.start_snapshot, "agent_state", "")),
            start_live_state_since=_as_optional_str(_snapshot_get(pending.start_snapshot, "state_since", "")),
            end_active_agents=int(_snapshot_get(end_snapshot, "active_agents", 0) or 0),
            end_active_tool_calls=int(_snapshot_get(end_snapshot, "active_tool_calls", 0) or 0),
            end_kv_blocks=_as_optional_int(_snapshot_get(end_snapshot, "kv_blocks", "")),
            end_kv_gb=_as_optional_str(_snapshot_get(end_snapshot, "kv_gb", "")),
            detail=detail or pending.detail,
        )

    def _pending_key(self, pending: PendingToolCall) -> str:
        if pending.action_id:
            return f"action:{pending.action_id}"
        if pending.tool_call_id:
            return f"tool_call:{pending.tool_call_id}"
        return f"seq:{self._seq}:{len(self._pending)}"

    def _snapshot(self) -> dict[str, Any]:
        if self._live_snapshot_fn is None:
            return {}
        try:
            return dict(self._live_snapshot_fn() or {})
        except Exception:
            return {}

    @staticmethod
    def extract_tool_args(event: Any) -> dict[str, Any]:
        """Best-effort action argument extraction across SDK versions."""
        for attr in ("arguments", "args", "tool_input", "input", "kwargs"):
            value = getattr(event, attr, None)
            if value is None:
                continue
            if isinstance(value, dict):
                return value
            if dataclasses.is_dataclass(value):
                return dataclasses.asdict(value)
            if hasattr(value, "model_dump"):
                data = value.model_dump()
                return data if isinstance(data, dict) else {"raw": data}
            if hasattr(value, "dict"):
                data = value.dict()
                return data if isinstance(data, dict) else {"raw": data}
            try:
                parsed = json.loads(str(value))
            except (json.JSONDecodeError, TypeError, ValueError):
                return {"raw": str(value)}
            return parsed if isinstance(parsed, dict) else {"raw": parsed}

        action = getattr(event, "action", None)
        if action is not None:
            for attr in ("arguments", "args", "tool_input", "input", "kwargs"):
                value = getattr(action, attr, None)
                if isinstance(value, dict):
                    return value
            if dataclasses.is_dataclass(action):
                data = dataclasses.asdict(action)
                return data if isinstance(data, dict) else {"raw": data}
            if hasattr(action, "model_dump"):
                data = action.model_dump()
                return data if isinstance(data, dict) else {"raw": data}
            if hasattr(action, "dict"):
                data = action.dict()
                return data if isinstance(data, dict) else {"raw": data}
        return {}

    @staticmethod
    def extract_terminal_command(event: Any, args: dict[str, Any]) -> str:
        tool_name = str(getattr(event, "tool_name", "") or "").lower()
        if tool_name in {"terminal", "bash", "shell"}:
            for key in ("command", "cmd", "shell_command", "input"):
                value = args.get(key)
                if isinstance(value, str):
                    return value
        return ""

    @staticmethod
    def extract_observation_content(event: Any) -> str:
        if event is None:
            return ""
        for attr in (
            "content",
            "output",
            "result",
            "text",
            "observation",
            "data",
            "stdout",
            "stderr",
            "message",
        ):
            value = getattr(event, attr, None)
            if value is not None and value != "":
                return _as_text(value)
        return ""

    @staticmethod
    def extract_returncode(event: Any) -> str:
        if event is None:
            return ""
        for attr in ("returncode", "return_code", "exit_code", "status_code"):
            value = getattr(event, attr, None)
            code = _as_optional_int(value)
            if code != "":
                return code
        data = getattr(event, "data", None)
        if isinstance(data, dict):
            for key in ("returncode", "return_code", "exit_code", "status_code"):
                code = _as_optional_int(data.get(key))
                if code != "":
                    return code
        return ""

    @staticmethod
    def extract_timed_out(event: Any) -> int:
        if event is None:
            return 0
        for attr in ("timed_out", "timeout", "is_timeout"):
            value = getattr(event, attr, None)
            if value is not None:
                return _as_bool(value)
        data = getattr(event, "data", None)
        if isinstance(data, dict):
            for key in ("timed_out", "timeout", "is_timeout"):
                if key in data:
                    return _as_bool(data[key])
        return 0

    @staticmethod
    def extract_error_type(event: Any) -> str:
        if event is None:
            return ""
        for attr in ("error_type", "error", "exception"):
            value = getattr(event, attr, None)
            if value:
                return type(value).__name__ if not isinstance(value, str) else value
        data = getattr(event, "data", None)
        if isinstance(data, dict):
            for key in ("error_type", "error", "exception"):
                value = data.get(key)
                if value:
                    return str(value)
        return ""


def load_tool_call_records(path: Path) -> list[dict[str, Any]]:
    """Load tool-call records from a collector CSV or JSONL artifact."""
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
