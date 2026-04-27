#!/usr/bin/env python3
"""
run_abc_bench_instrumented.py
─────────────────────────────
Run ABC-Bench tasks via the OpenHands SDK against a local vLLM endpoint.
Records a per-agent execution trace of three phases:

    reasoning   – LLM is generating tokens (between receiving context and
                  emitting an action or final message)
    tool_call   – a tool is executing (ActionEvent → ObservationEvent)
    waiting     – the agent is idle (before launch, between conversation
                  rounds, or after completion)

Each agent's trace is saved to a separate CSV file whose columns are
documented in ``TRACE_CSV_FIELDS`` below.

Key differences from the original script:
  • Multi-agent: a randomised launcher starts groups of agents at staggered
    wall-clock offsets so that multiple agents overlap on the same vLLM
    instance (to study KV-cache contention).
  • Phase-level tracing: every phase transition is recorded with wall-clock
    timestamps in ISO 8601 UTC – the same format vLLM uses internally – so
    traces can be aligned with vLLM scheduler / block-manager logs.
  • Tool-call metadata: the trace captures tool name, arguments, terminal
    commands, and payload sizes to feed the downstream duration-prediction
    model (see ``predict_tool_duration.py``).

Install:
    pip install openhands-sdk openhands-tools pyyaml pydantic scikit-learn

Environment variables (or CLI flags):
    LLM_MODEL      e.g. openai/Qwen3-30B-A3B-Instruct
    LLM_BASE_URL   e.g. http://127.0.0.1:8000/v1
    LLM_API_KEY    API key for the vLLM server

Example – launch 1-4 agents at random intervals over 6 tasks:
    python run_abc_bench_instrumented.py \
        --dataset-root /data/ABC-Bench \
        --task-glob 'task_*' \
        --max-tasks 6 \
        --randomise-launch \
        --max-agents-per-wave 4 \
        --run-tests \
        --results-root ./abc_results
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.event import (
    Event,
    ActionEvent,
    ObservationEvent,
    MessageEvent,
    UserRejectObservation,
    AgentErrorEvent,
)
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("abc_instrumented")

# Guards concurrent append/remove on litellm.success_callback (a global list).
_litellm_cb_lock = threading.Lock()

# ─────────────────────────── live state (shared with embedded sidecar) ────────
# Agent threads write phase transitions and latest KV stats here.
# The embedded sidecar thread reads this dict directly — no serialization.

_LIVE_AGENTS: dict[str, dict] = {}
_live_lock   = threading.Lock()

# ─────────────────────────────────────── helpers ──────────────────────────────

def json_dumps_safe(obj: Any, **kw) -> str:
    return json.dumps(obj, default=str, **kw)


def json_dump_safe(obj: Any, fp, **kw):
    return json.dump(obj, fp, default=str, **kw)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    """ISO 8601 with explicit +00:00 – matches vLLM's timestamp format."""
    return dt.astimezone(timezone.utc).isoformat()


def parse_ts(ts: Optional[str]) -> datetime:
    """Parse an ISO 8601 timestamp, handling the trailing-Z convention."""
    if not ts:
        return now_utc()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ────────────────────────── trace schema & data classes ───────────────────────

# CSV columns written per agent.  Downstream consumers (predict_tool_duration.py)
# rely on these names, so treat them as a stable interface.
TRACE_CSV_FIELDS = [
    "agent_id",          # unique agent identifier (e.g. "agent_task_042_wave0")
    "task_id",           # ABC-Bench task directory name
    "phase_seq",         # 0-based sequential index of this phase within the agent run
    "phase",             # "reasoning" | "tool_call" | "waiting"
    "start_ts",          # ISO 8601 UTC wall-clock start
    "end_ts",            # ISO 8601 UTC wall-clock end
    "duration_s",        # wall-clock seconds (float)
    "tool_name",         # e.g. "terminal", "file_editor" (tool_call only)
    "tool_command",      # terminal command string if available
    "tool_args_json",    # JSON-serialised tool arguments / payload
    "tool_payload_bytes",# byte-length of the serialised arguments
    "outcome",           # "ok" | "rejected" | "error" | "agent_message" | etc.
    "detail",            # human-readable short description
    "conversation_id",   # OpenHands conversation id
]


@dataclass
class PhaseRecord:
    """One row in the per-agent trace CSV."""
    agent_id: str
    task_id: str
    phase_seq: int
    phase: str             # reasoning | tool_call | waiting
    start_ts: str
    end_ts: str
    duration_s: float
    tool_name: str = ""
    tool_command: str = ""
    tool_args_json: str = ""
    tool_payload_bytes: int = 0
    outcome: str = ""
    detail: str = ""
    conversation_id: str = ""


# ──────────────────────── per-agent phase tracker ─────────────────────────────

@dataclass
class _PendingToolCall:
    """Bookkeeping for an in-flight tool call."""
    action_id: str
    tool_call_id: Optional[str]
    tool_name: str
    tool_command: str
    tool_args_json: str
    tool_payload_bytes: int
    start_dt: datetime
    detail: str = ""


class AgentPhaseTracker:
    """
    Tracks the phase lifecycle of a single agent.

    Phase boundary detection
    ────────────────────────
    The OpenHands SDK emits events on a single callback.  We infer phase
    boundaries as follows:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  run_start()                                                       │
    │      → emit WAITING phase (time between schedule and actual start) │
    │      → set phase = "reasoning" (LLM starts working on the prompt)  │
    │                                                                    │
    │  on_event(MessageEvent source=user)                                │
    │      → context delivered to LLM; reasoning phase continues         │
    │                                                                    │
    │  on_event(ActionEvent)                                             │
    │      → close current REASONING phase                               │
    │      → open TOOL_CALL phase                                        │
    │                                                                    │
    │  on_event(ObservationEvent)                                        │
    │      → close TOOL_CALL phase                                       │
    │      → open REASONING phase (LLM processes tool output)            │
    │                                                                    │
    │  on_event(MessageEvent source=agent)                               │
    │      → close current REASONING phase                               │
    │      → open WAITING phase (agent done for this turn)               │
    │                                                                    │
    │  run_end()                                                         │
    │      → close whatever phase is open as WAITING                     │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, agent_id: str, task_id: str, scheduled_dt: datetime):
        self.agent_id = agent_id
        self.task_id = task_id
        self.scheduled_dt = scheduled_dt   # when the launch was *scheduled*

        self.records: list[PhaseRecord] = []
        self._seq = 0
        self._current_phase: Optional[str] = None
        self._phase_start_dt: Optional[datetime] = None
        self._pending_tools: dict[str, _PendingToolCall] = {}
        self._conversation_id: str = ""
        self._lock = threading.Lock()

    # ── internal helpers ──

    def _close_phase(self, end_dt: datetime, outcome: str = "", detail: str = ""):
        """Close the current phase and append a PhaseRecord."""
        if self._current_phase is None or self._phase_start_dt is None:
            return
        dur = (end_dt - self._phase_start_dt).total_seconds()
        if dur < 0:
            dur = 0.0
        self.records.append(PhaseRecord(
            agent_id=self.agent_id,
            task_id=self.task_id,
            phase_seq=self._seq,
            phase=self._current_phase,
            start_ts=iso_utc(self._phase_start_dt),
            end_ts=iso_utc(end_dt),
            duration_s=round(dur, 6),
            outcome=outcome,
            detail=detail,
            conversation_id=self._conversation_id,
        ))
        self._seq += 1
        self._current_phase = None
        self._phase_start_dt = None

    def _open_phase(self, phase: str, start_dt: datetime):
        self._current_phase = phase
        self._phase_start_dt = start_dt
        with _live_lock:
            if self.agent_id in _LIVE_AGENTS:
                _LIVE_AGENTS[self.agent_id]["state"] = phase
                _LIVE_AGENTS[self.agent_id]["state_since"] = iso_utc(start_dt)

    # ── public API ──

    def run_start(self):
        """Called just before conversation.send_message()."""
        with _live_lock:
            _LIVE_AGENTS[self.agent_id] = {
                "task_id": self.task_id,
                "state": "waiting",
                "state_since": iso_utc(now_utc()),
                "kv_blocks": None,
                "kv_gb": None,
                "last_kv_updated": None,
            }
        with self._lock:
            start_dt = now_utc()
            # Record the waiting gap between scheduled time and actual start.
            wait_dur = (start_dt - self.scheduled_dt).total_seconds()
            if wait_dur > 0.01:
                self.records.append(PhaseRecord(
                    agent_id=self.agent_id,
                    task_id=self.task_id,
                    phase_seq=self._seq,
                    phase="waiting",
                    start_ts=iso_utc(self.scheduled_dt),
                    end_ts=iso_utc(start_dt),
                    duration_s=round(wait_dur, 6),
                    outcome="launch_delay",
                    detail="time between scheduled launch and actual start",
                ))
                self._seq += 1
            # The agent is now reasoning (processing the initial prompt).
            self._open_phase("reasoning", start_dt)

    def run_end(self):
        """Called after conversation.run() returns."""
        with _live_lock:
            if self.agent_id in _LIVE_AGENTS:
                _LIVE_AGENTS[self.agent_id]["state"] = "done"
                _LIVE_AGENTS[self.agent_id]["state_since"] = iso_utc(now_utc())
        with self._lock:
            end_dt = now_utc()
            self._close_phase(end_dt, outcome="run_complete", detail="agent finished")

    def on_event(self, event: Event):
        """OpenHands SDK event callback – drives phase transitions."""
        with self._lock:
            self._conversation_id = getattr(event, "conversation_id", self._conversation_id) or ""
            event_dt = parse_ts(getattr(event, "timestamp", None))

            # ── user message: context delivered, reasoning continues ──
            if isinstance(event, MessageEvent):
                source = getattr(event, "source", "")
                if source == "user":
                    # If we were waiting (e.g. between conversation turns),
                    # transition to reasoning.
                    if self._current_phase == "waiting":
                        self._close_phase(event_dt, outcome="new_input")
                        self._open_phase("reasoning", event_dt)
                    return

                if source == "agent":
                    # Agent emitted a text message (not a tool call).
                    # Close the reasoning phase; the agent is now idle.
                    self._close_phase(event_dt, outcome="agent_message",
                                      detail="assistant text message")
                    self._open_phase("waiting", event_dt)
                    return

            # ── action event: agent issues a tool call ──
            if isinstance(event, ActionEvent):
                tool_name = getattr(event, "tool_name", "unknown") or "unknown"
                tool_call_id = getattr(event, "tool_call_id", None)
                # Capture as much metadata as possible for the predictor.
                raw_args = self._extract_tool_args(event)
                args_json = json_dumps_safe(raw_args)
                tool_command = self._extract_terminal_command(event, raw_args)

                # Close the reasoning phase that preceded this tool call.
                self._close_phase(event_dt, outcome="tool_call", detail=tool_name)

                # Open a tool_call phase.
                self._open_phase("tool_call", event_dt)

                # Stash metadata so we can enrich the record when the
                # ObservationEvent arrives.
                self._pending_tools[event.id] = _PendingToolCall(
                    action_id=event.id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_command=tool_command,
                    tool_args_json=args_json,
                    tool_payload_bytes=len(args_json.encode("utf-8")),
                    start_dt=event_dt,
                    detail=getattr(event, "summary", "") or tool_name,
                )
                return

            # ── observation event: tool call completed ──
            if isinstance(event, (ObservationEvent, UserRejectObservation)):
                pending = self._match_pending(event)
                outcome = "rejected" if isinstance(event, UserRejectObservation) else "ok"

                if pending is not None:
                    dur = (event_dt - pending.start_dt).total_seconds()
                    # Close the tool_call phase we opened above, but override
                    # with the enriched record that carries tool metadata.
                    # First discard the generic _close_phase record:
                    self._current_phase = None
                    self._phase_start_dt = None

                    self.records.append(PhaseRecord(
                        agent_id=self.agent_id,
                        task_id=self.task_id,
                        phase_seq=self._seq,
                        phase="tool_call",
                        start_ts=iso_utc(pending.start_dt),
                        end_ts=iso_utc(event_dt),
                        duration_s=round(max(dur, 0.0), 6),
                        tool_name=pending.tool_name,
                        tool_command=pending.tool_command,
                        tool_args_json=pending.tool_args_json,
                        tool_payload_bytes=pending.tool_payload_bytes,
                        outcome=outcome,
                        detail=pending.detail,
                        conversation_id=self._conversation_id,
                    ))
                    self._seq += 1
                else:
                    # No matching pending action – close generically.
                    self._close_phase(event_dt, outcome=outcome,
                                      detail="tool result (no matching action)")

                # After a tool result the LLM starts reasoning again.
                self._open_phase("reasoning", event_dt)
                return

            # ── error event ──
            if isinstance(event, AgentErrorEvent):
                self._close_phase(event_dt, outcome="error",
                                  detail=str(getattr(event, "message", "agent error")))
                self._open_phase("waiting", event_dt)
                return

    # ── tool metadata extraction ──

    @staticmethod
    def _extract_tool_args(event: ActionEvent) -> dict[str, Any]:
        """
        Pull whatever argument dict the SDK exposes.  Attribute names vary
        across SDK versions, so we try several.
        """
        for attr in ("arguments", "args", "tool_input", "input", "kwargs"):
            val = getattr(event, attr, None)
            if val is not None:
                if isinstance(val, dict):
                    return val
                try:
                    return json.loads(str(val))
                except (json.JSONDecodeError, TypeError):
                    return {"raw": str(val)}
        return {}

    @staticmethod
    def _extract_terminal_command(event: ActionEvent, args: dict) -> str:
        """
        If the tool is ``terminal``, try to extract the shell command string.
        This is the single most predictive feature for duration estimation.
        """
        tool_name = getattr(event, "tool_name", "") or ""
        if tool_name.lower() in ("terminal", "bash", "shell"):
            for key in ("command", "cmd", "shell_command", "input"):
                if key in args and isinstance(args[key], str):
                    return args[key]
        return ""

    def _match_pending(self, event) -> Optional[_PendingToolCall]:
        """Match an observation back to its pending action."""
        action_id = getattr(event, "action_id", None)
        if action_id and action_id in self._pending_tools:
            return self._pending_tools.pop(action_id)
        # Fallback: match by tool_call_id.
        tcid = getattr(event, "tool_call_id", None)
        if tcid:
            for k, cand in list(self._pending_tools.items()):
                if cand.tool_call_id == tcid:
                    return self._pending_tools.pop(k)
        return None

    # ── persistence ──

    def write_csv(self, out_dir: Path):
        """Write the trace to ``<out_dir>/<agent_id>_trace.csv``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.agent_id}_trace.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRACE_CSV_FIELDS)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(dataclasses.asdict(rec))
        log.info("Wrote %d phase records → %s", len(self.records), path)
        return path

    def summary(self) -> dict[str, Any]:
        by_phase: dict[str, float] = defaultdict(float)
        tool_durs: list[float] = []
        for r in self.records:
            by_phase[r.phase] += r.duration_s
            if r.phase == "tool_call":
                tool_durs.append(r.duration_s)
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "n_phases": len(self.records),
            "reasoning_s": round(by_phase.get("reasoning", 0), 3),
            "tool_call_s": round(by_phase.get("tool_call", 0), 3),
            "waiting_s": round(by_phase.get("waiting", 0), 3),
            "n_tool_calls": len(tool_durs),
            "avg_tool_call_s": round(sum(tool_durs) / len(tool_durs), 3) if tool_durs else 0,
            "max_tool_call_s": round(max(tool_durs), 3) if tool_durs else 0,
        }


# ────────────────────────── task loading (unchanged) ──────────────────────────

PROMPT_KEYS = (
    "instruction", "instructions", "prompt", "task",
    "goal", "description", "problem_statement",
)


def find_task_dirs(dataset_root: Path, task_glob: str) -> list[Path]:
    root = dataset_root
    if (root / "tasks").is_dir():
        root = root / "tasks"
    return sorted(p for p in root.glob(task_glob) if p.is_dir())


def expand_tasks_round_robin(task_dirs: list[Path], max_tasks: int) -> list[Path]:
    """Return up to max_tasks entries, repeating task_dirs in round-robin if needed."""
    if not task_dirs or max_tasks <= 0:
        return []
    return [task_dirs[i % len(task_dirs)] for i in range(max_tasks)]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in YAML file: {path}")
    return data


def _find_first_promptlike_value(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for key in PROMPT_KEYS:
            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                return obj[key].strip()
        for _, value in obj.items():
            found = _find_first_promptlike_value(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_promptlike_value(item)
            if found:
                return found
    return None


def build_agent_prompt(task_yaml: dict[str, Any], task_dir: Path) -> str:
    base = _find_first_promptlike_value(task_yaml)
    if base is None:
        base = (
            "The task instruction field could not be identified automatically. "
            "Use the task metadata below to infer the objective.\n\n"
            f"task.yaml contents:\n{yaml.safe_dump(task_yaml, sort_keys=False)}"
        )
    suffix = (
        "\n\nAdditional execution rules for this local ABC-Bench run:\n"
        "- You are already inside the root of the benchmark task repository.\n"
        "- Make the minimum code/config changes needed to solve the task.\n"
        "- Do not edit the benchmark tests unless the task explicitly requires it.\n"
        "- You may inspect files, edit source code, run shell commands, build containers, "
        "start services, and execute ./run-tests.sh for validation.\n"
        "- Stop when the repository is in a state that should pass the provided validator.\n"
        f"- Benchmark task directory: {task_dir}\n"
    )
    return base.rstrip() + suffix


def copy_task_to_workspace(src: Path, workspace_root: Path, suffix: str = "") -> Path:
    dst = workspace_root / f"{src.name}{suffix}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


# ─────────────────────────── single-agent runner ─────────────────────────────

def make_llm(model: str, base_url: str, api_key: str, agent_id: str) -> LLM:
    """
    Build an LLM handle.  ``agent_id`` is passed through ``extra_body`` so
    that a patched vLLM can track per-agent KV-cache ownership.
    """
    return LLM(
        usage_id=agent_id,
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        litellm_extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "agent_id": agent_id,
        },
    )


def make_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ],
    )


def run_single_agent(
    agent_id: str,
    task_dir: Path,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    max_iterations: int,
    results_dir: Path,
    scheduled_dt: datetime,
) -> dict[str, Any]:
    """
    Execute one agent on one task and return a summary dict.
    The per-agent CSV trace is written to ``results_dir``.
    """
    task_yaml_path = task_dir / "task.yaml"
    if not task_yaml_path.exists():
        raise FileNotFoundError(f"Missing task.yaml in {task_dir}")

    task_yaml = load_yaml(task_yaml_path)
    prompt = build_agent_prompt(task_yaml, task_dir)

    llm = make_llm(llm_model, llm_base_url, llm_api_key, agent_id)
    agent = make_agent(llm)
    tracker = AgentPhaseTracker(agent_id=agent_id, task_id=task_dir.name,
                                scheduled_dt=scheduled_dt)

    conversation = Conversation(
        agent=agent,
        workspace=str(task_dir),
        callbacks=[tracker.on_event],
        max_iteration_per_run=max_iterations,
        persistence_dir=str(task_dir / ".openhands_conversations"),
    )

    log.info("[%s] Starting on %s", agent_id, task_dir.name)

    # Per-agent accumulator for KV-block stats returned by the patched vLLM.
    # litellm.success_callback fires after every LLM call with the assembled
    # response; we read kv_blocks_used / kv_blocks_size_gb from usage there.
    _kv_acc: dict[str, Any] = {
        "total_kv_blocks": 0,
        "total_kv_size_gb": 0.0,
        "call_count": 0,
    }

    def _kv_cb(kwargs: Any, completion_response: Any,
                start_time: Any, end_time: Any) -> None:
        try:
            usage = getattr(completion_response, "usage", None)
            if usage is None:
                return
            kb = getattr(usage, "kv_blocks_used", None)
            ks = getattr(usage, "kv_blocks_size_gb", None)
            if isinstance(kb, int):
                _kv_acc["total_kv_blocks"] += kb
                _kv_acc["call_count"] += 1
            if isinstance(ks, (int, float)):
                _kv_acc["total_kv_size_gb"] += float(ks)
            # Mirror latest per-call KV stats into the live state for sidecar.py.
            if isinstance(kb, int):
                with _live_lock:
                    if agent_id in _LIVE_AGENTS:
                        _LIVE_AGENTS[agent_id]["kv_blocks"] = kb
                        _LIVE_AGENTS[agent_id]["kv_gb"] = (
                            float(ks) if isinstance(ks, (int, float)) else None
                        )
                        _LIVE_AGENTS[agent_id]["last_kv_updated"] = iso_utc(now_utc())
        except Exception:
            pass  # callbacks must never raise

    import litellm as _litellm
    with _litellm_cb_lock:
        if not isinstance(getattr(_litellm, "success_callback", None), list):
            _litellm.success_callback = []
        _litellm.success_callback.append(_kv_cb)

    tracker.run_start()
    wall_start = time.monotonic()
    conversation.send_message(prompt)
    conversation.run()
    wall_end = time.monotonic()
    tracker.run_end()

    with _litellm_cb_lock:
        try:
            _litellm.success_callback.remove(_kv_cb)
        except (ValueError, AttributeError):
            pass

    # Persist trace CSV.
    results_dir.mkdir(parents=True, exist_ok=True)
    tracker.write_csv(results_dir)

    # Persist prompt for reproducibility.
    (results_dir / f"{agent_id}_prompt.txt").write_text(prompt, encoding="utf-8")

    # Persist LLM-level metrics if available.
    llm_metrics: dict[str, Any] = {}
    if getattr(llm, "metrics", None) is not None:
        try:
            llm_metrics = llm.metrics.model_dump()
        except Exception:
            llm_metrics = {"repr": repr(llm.metrics)}

    # Append per-agent KV-block stats captured via the litellm callback.
    # These are populated by the patched vLLM when agent_id is in extra_body.
    llm_metrics["kv_blocks_used"]       = _kv_acc["total_kv_blocks"]
    llm_metrics["kv_blocks_size_gb"]    = round(_kv_acc["total_kv_size_gb"], 6)
    llm_metrics["kv_blocks_call_count"] = _kv_acc["call_count"]

    with (results_dir / f"{agent_id}_llm_metrics.json").open("w") as f:
        json_dump_safe(llm_metrics, f, indent=2)

    summary = tracker.summary()
    summary["elapsed_wall_s"] = round(wall_end - wall_start, 3)
    summary["results_dir"] = str(results_dir)

    with (results_dir / f"{agent_id}_summary.json").open("w") as f:
        json_dump_safe(summary, f, indent=2)

    log.info("[%s] Finished in %.1fs  (%d tool calls)",
             agent_id, summary["elapsed_wall_s"], summary["n_tool_calls"])
    return summary


# ──────────────────── optional: run tests after agent ─────────────────────────

def maybe_run_tests(task_dir: Path, timeout_sec: int, out_dir: Path,
                    agent_id: str) -> dict[str, Any]:
    script = task_dir / "run-tests.sh"
    if not script.exists():
        return {"ran_tests": False, "reason": "run-tests.sh not found"}
    started = time.time()
    try:
        proc = subprocess.run(
            ["bash", str(script)], cwd=str(task_dir),
            capture_output=True, text=True, timeout=timeout_sec,
            env=os.environ.copy(),
        )
        result = {
            "ran_tests": True,
            "returncode": proc.returncode,
            "passed": proc.returncode == 0,
            "elapsed_s": round(time.time() - started, 3),
        }
        (out_dir / f"{agent_id}_test_stdout.txt").write_text(proc.stdout)
        (out_dir / f"{agent_id}_test_stderr.txt").write_text(proc.stderr)
    except subprocess.TimeoutExpired:
        result = {
            "ran_tests": True, "returncode": None,
            "passed": False, "timed_out": True,
            "elapsed_s": round(time.time() - started, 3),
        }
    with (out_dir / f"{agent_id}_test_result.json").open("w") as f:
        json_dump_safe(result, f, indent=2)
    return result


# ─────────────────── randomised multi-agent launcher ─────────────────────────

def plan_launch_waves(
    task_dirs: list[Path],
    max_agents_per_wave: int,
    min_delay_s: float,
    max_delay_s: float,
    seed: Optional[int] = None,
) -> list[tuple[float, list[Path]]]:
    """
    Partition ``task_dirs`` into waves of random size, separated by random
    delays.  Returns a list of ``(delay_from_start_s, [task_dirs])`` tuples.

    Example output with 6 tasks, max_agents_per_wave=4:
        [(0.0,   [task_001, task_002]),         # wave 0: 2 agents at t=0
         (4.2,   [task_003]),                   # wave 1: 1 agent  at t≈4s
         (11.7,  [task_004, task_005, task_006])]# wave 2: 3 agents at t≈12s
    """
    rng = random.Random(seed)
    remaining = list(task_dirs)
    rng.shuffle(remaining)

    waves: list[tuple[float, list[Path]]] = []
    cumulative_delay = 0.0
    while remaining:
        n = rng.randint(1, min(max_agents_per_wave, len(remaining)))
        batch, remaining = remaining[:n], remaining[n:]
        waves.append((cumulative_delay, batch))
        if remaining:
            cumulative_delay += rng.uniform(min_delay_s, max_delay_s)
    return waves


def launch_agents(
    waves: list[tuple[float, list[Path]]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """
    Execute the planned launch waves using a thread pool.
    Each agent runs in its own thread; waves are staggered by sleeping
    on the main thread.
    """
    all_summaries: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=args.max_concurrent or 8)
    futures = []
    wave_start = time.monotonic()
    global_idx = 0  # unique across all agents in all waves

    for wave_idx, (delay_s, task_batch) in enumerate(waves):
        # Sleep until this wave's scheduled time.
        elapsed = time.monotonic() - wave_start
        sleep_needed = delay_s - elapsed
        if sleep_needed > 0:
            log.info("Sleeping %.1fs before wave %d …", sleep_needed, wave_idx)
            time.sleep(sleep_needed)

        scheduled_dt = now_utc()
        for task_dir in task_batch:
            agent_id = f"agent_{task_dir.name}_w{wave_idx}_{global_idx}"
            work_dir = copy_task_to_workspace(
                task_dir, args.workspace_root, suffix=f"_w{wave_idx}_{global_idx}")
            global_idx += 1
            result_dir = args.results_root / task_dir.name

            fut = executor.submit(
                _agent_thread,
                agent_id=agent_id,
                task_dir=work_dir,
                original_task_dir=task_dir,
                args=args,
                result_dir=result_dir,
                scheduled_dt=scheduled_dt,
            )
            futures.append(fut)
        log.info("Wave %d: launched %d agent(s)  [%s]",
                 wave_idx, len(task_batch),
                 ", ".join(t.name for t in task_batch))

    # Collect results.
    for fut in as_completed(futures):
        try:
            all_summaries.append(fut.result())
        except Exception as exc:
            log.error("Agent failed: %s", exc, exc_info=True)
            all_summaries.append({"error": repr(exc)})

    executor.shutdown(wait=True)
    return all_summaries


def _agent_thread(
    agent_id: str,
    task_dir: Path,
    original_task_dir: Path,
    args: argparse.Namespace,
    result_dir: Path,
    scheduled_dt: datetime,
) -> dict[str, Any]:
    """Target function for each agent thread."""
    try:
        summary = run_single_agent(
            agent_id=agent_id,
            task_dir=task_dir,
            llm_model=args.model,
            llm_base_url=args.base_url,
            llm_api_key=args.api_key,
            max_iterations=args.max_iterations,
            results_dir=result_dir,
            scheduled_dt=scheduled_dt,
        )
        if args.run_tests:
            test_result = maybe_run_tests(
                task_dir, args.test_timeout_sec, result_dir, agent_id)
            summary["test_result"] = test_result
        return summary
    except Exception as exc:
        err = {"agent_id": agent_id, "task_id": original_task_dir.name,
               "error": repr(exc)}
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"{agent_id}_error.json").write_text(
            json_dumps_safe(err, indent=2))
        raise


# ──────────────────────────────── CLI ─────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ABC-Bench with per-agent phase tracing and optional "
                    "randomised multi-agent launch.")
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--workspace-root", type=Path, default=Path("./abc_runs"))
    p.add_argument("--results-root", type=Path, default=Path("./abc_results"))
    p.add_argument("--task-glob", default="task_*")
    p.add_argument("--max-tasks", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=80)
    p.add_argument("--run-tests", action="store_true")
    p.add_argument("--test-timeout-sec", type=int, default=1800)

    # LLM endpoint.
    p.add_argument("--model", default=os.getenv("LLM_MODEL"))
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL"))
    p.add_argument("--api-key", default=os.getenv("LLM_API_KEY"))

    # Multi-agent launch options.
    p.add_argument("--randomise-launch", action="store_true",
                   help="Launch agents in randomised waves instead of "
                        "sequentially.")
    p.add_argument("--max-agents-per-wave", type=int, default=4)
    p.add_argument("--min-wave-delay-s", type=float, default=2.0)
    p.add_argument("--max-wave-delay-s", type=float, default=15.0)
    p.add_argument("--launch-seed", type=int, default=None,
                   help="RNG seed for reproducible launch schedules.")
    p.add_argument("--max-concurrent", type=int, default=8,
                   help="Thread pool size for concurrent agents.")

    # ── embedded sidecar (optional) ──────────────────────────────────────────
    # Set --sidecar-log-file to enable.  The sidecar runs as a daemon thread
    # and reads _LIVE_AGENTS directly — no file I/O or serialization.
    sc = p.add_argument_group("embedded sidecar (omit to disable)")
    sc.add_argument("--sidecar-log-file", default=None,
                    help="Enable the embedded sidecar and write its JSONL log here.")
    sc.add_argument("--sidecar-vllm-url", default="http://localhost:8000")
    sc.add_argument("--sidecar-interval", type=float, default=5.0,
                    help="Sidecar poll interval in seconds.")
    sc.add_argument("--sidecar-num-layers",   type=int, default=None)
    sc.add_argument("--sidecar-num-kv-heads", type=int, default=None)
    sc.add_argument("--sidecar-head-dim",     type=int, default=None)
    sc.add_argument("--sidecar-block-size",   type=int, default=16)
    sc.add_argument("--sidecar-dtype",        default="bfloat16")
    sc.add_argument("--sidecar-total-gpu-blocks", type=int, default=0,
                    help="Total GPU/NPU KV-cache blocks. Required when vLLM does not expose "
                         "a block-count metric in Prometheus (e.g. vllm_ascend). "
                         "Must match --num-gpu-blocks-override in vLLM args.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    for name, val in [("model", args.model), ("base_url", args.base_url),
                      ("api_key", args.api_key)]:
        if not val:
            print(f"ERROR: missing --{name.replace('_','-')} or "
                  f"{name.upper()}", file=sys.stderr)
            return 2

    # ── optional embedded sidecar ────────────────────────────────────────────
    if args.sidecar_log_file:
        missing = [f"--sidecar-{f}" for f in ("num-layers", "num-kv-heads", "head-dim")
                   if getattr(args, f"sidecar_{f.replace('-', '_')}") is None]
        if missing:
            print(f"ERROR: sidecar enabled but missing: {', '.join(missing)}",
                  file=sys.stderr)
            return 2
        import sidecar as _sidecar
        _sc_args = argparse.Namespace(
            vllm_url=args.sidecar_vllm_url,
            log_file=args.sidecar_log_file,
            interval=args.sidecar_interval,
            num_layers=args.sidecar_num_layers,
            num_kv_heads=args.sidecar_num_kv_heads,
            head_dim=args.sidecar_head_dim,
            block_size=args.sidecar_block_size,
            dtype=args.sidecar_dtype,
            total_gpu_blocks=args.sidecar_total_gpu_blocks,
        )
        _sc_bpb = _sidecar.bytes_per_block(
            _sc_args.num_layers, _sc_args.num_kv_heads,
            _sc_args.head_dim, _sc_args.block_size, _sc_args.dtype,
        )
        def _get_agents() -> dict:
            with _live_lock:
                return dict(_LIVE_AGENTS)
        _sc_stop = threading.Event()
        _sc_thread = threading.Thread(
            target=_sidecar.run_loop,
            args=(_sc_args, _sc_bpb, _get_agents, _sc_stop),
            daemon=True, name="sidecar",
        )
        _sc_thread.start()
        log.info("[sidecar] embedded sidecar started → %s", args.sidecar_log_file)

    task_dirs = find_task_dirs(args.dataset_root, args.task_glob)
    if not task_dirs:
        print(f"ERROR: no tasks found under {args.dataset_root} "
              f"with glob {args.task_glob}", file=sys.stderr)
        return 2
    if args.max_tasks > len(task_dirs):
        log.info(
            "max_tasks=%d > available tasks=%d — repeating in round-robin",
            args.max_tasks, len(task_dirs),
        )
    task_dirs = expand_tasks_round_robin(task_dirs, args.max_tasks)

    args.workspace_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)

    # ── choose launch strategy ──
    if args.randomise_launch:
        waves = plan_launch_waves(
            task_dirs,
            max_agents_per_wave=args.max_agents_per_wave,
            min_delay_s=args.min_wave_delay_s,
            max_delay_s=args.max_wave_delay_s,
            seed=args.launch_seed,
        )
        log.info("Planned %d wave(s) for %d task(s):", len(waves), len(task_dirs))
        for i, (delay, batch) in enumerate(waves):
            log.info("  wave %d @ +%.1fs: %s", i, delay,
                     [t.name for t in batch])
        summaries = launch_agents(waves, args)
    else:
        # Sequential fallback (one agent at a time).
        summaries = []
        for seq_idx, task_dir in enumerate(task_dirs):
            agent_id = f"agent_{task_dir.name}_seq{seq_idx}"
            work_dir = copy_task_to_workspace(task_dir, args.workspace_root, suffix=f"_seq{seq_idx}")
            result_dir = args.results_root / task_dir.name
            result_dir.mkdir(parents=True, exist_ok=True)
            try:
                summary = run_single_agent(
                    agent_id=agent_id,
                    task_dir=work_dir,
                    llm_model=args.model,
                    llm_base_url=args.base_url,
                    llm_api_key=args.api_key,
                    max_iterations=args.max_iterations,
                    results_dir=result_dir,
                    scheduled_dt=now_utc(),
                )
                if args.run_tests:
                    summary["test_result"] = maybe_run_tests(
                        work_dir, args.test_timeout_sec, result_dir, agent_id)
                summaries.append(summary)
            except Exception as exc:
                log.error("Task %s failed: %s", task_dir.name, exc,
                          exc_info=True)
                summaries.append({"agent_id": agent_id, "error": repr(exc)})

    # ── aggregate results ──
    with (args.results_root / "run_summary.json").open("w") as f:
        json_dump_safe(summaries, f, indent=2)

    log.info("Done. %d agent(s) ran. Results → %s",
             len(summaries), args.results_root)

    if args.sidecar_log_file:
        log.info("[sidecar] shutting down…")
        _sc_stop.set()
        _sc_thread.join(timeout=args.sidecar_interval + 5)
        log.info("[sidecar] stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())