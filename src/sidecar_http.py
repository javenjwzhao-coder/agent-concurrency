#!/usr/bin/env python3
"""
sidecar_http.py — Live HTTP feed for the real-time agent state dashboard.

Exposes the per-tick records produced by the sidecar tick loop over a small
HTTP/SSE interface so the browser dashboard at dashboard/index.html can render
them in real time. No new dependencies — stdlib http.server only.

Endpoints
---------
    GET /              → dashboard/index.html
    GET /static/<path> → files under dashboard/ (dashboard.js/.css, …)
    GET /state         → JSON {"latest_tick": int|null, "ticks": [<tick records>]}
    GET /stream        → text/event-stream; one ``data:`` frame per tick.
                         Optional query param ?since=<tick> replays buffered
                         records after that tick before going live.
    GET /healthz       → "ok"

The tick record shape on /state and /stream is the same JSON the sidecar
writes to sidecar.log (no transformation) — see dashboard/SCHEMA.md.

Usage
-----
Embedded — call ``HTTPFeed`` from run_loop in sidecar.py::

    feed = HTTPFeed()
    server, thread = start_server(host, port, feed)
    # in tick loop, after writing JSONL:
    feed.publish(record)

Replay — replay a finished sidecar.log against the same dashboard::

    python -m sidecar_http --replay path/to/sidecar.log [--speed 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

# ─────────────────────────────────────── feed ─────────────────────────────────


_DEFAULT_HISTORY = 2000
_SUBSCRIBER_QUEUE_MAX = 1024
_SSE_HEARTBEAT_SECONDS = 15.0


class HTTPFeed:
    """Thread-safe ring buffer of tick records + SSE subscriber registry."""

    def __init__(self, history_size: int = _DEFAULT_HISTORY) -> None:
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._lock = threading.Lock()

    def publish(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._history.append(record)
            dead: list[queue.Queue[dict[str, Any]]] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(record)
                except queue.Full:
                    # Drop the oldest entry to keep up; the dashboard would
                    # rather lose the slowest-watcher's history than block the
                    # sidecar tick loop.
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(record)
                    except queue.Full:
                        dead.append(q)
            for q in dead:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    def subscribe(self, since_tick: Optional[int] = None) -> tuple[
        queue.Queue[dict[str, Any]], list[dict[str, Any]]
    ]:
        """Register a subscriber and return (queue, replay_records).

        ``replay_records`` are buffered records strictly newer than
        ``since_tick`` (or all of them if ``since_tick`` is None). The caller
        should send these before consuming live frames from the queue.
        """
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        with self._lock:
            self._subscribers.append(q)
            if since_tick is None:
                replay = list(self._history)
            else:
                replay = [r for r in self._history if _record_tick(r) > since_tick]
        return q, replay

    def unsubscribe(self, q: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def snapshot(self) -> tuple[Optional[int], list[dict[str, Any]]]:
        with self._lock:
            ticks = list(self._history)
        latest = _record_tick(ticks[-1]) if ticks else None
        return latest, ticks


def _record_tick(record: dict[str, Any]) -> int:
    t = record.get("tick")
    return int(t) if isinstance(t, int) else -1


# ─────────────────────────────────────── http handler ─────────────────────────


def _make_handler(feed: HTTPFeed, dashboard_dir: Path):
    """Return a BaseHTTPRequestHandler subclass bound to ``feed``."""

    static_root = dashboard_dir.resolve()

    class DashboardHandler(BaseHTTPRequestHandler):
        # Override default noisy stderr access logging.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

        def do_GET(self) -> None:  # noqa: N802 — http.server API
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path == "/" or path == "/index.html":
                    self._serve_file(static_root / "index.html", "text/html; charset=utf-8")
                elif path.startswith("/static/"):
                    self._serve_static(path[len("/static/"):])
                elif path == "/state":
                    self._serve_state()
                elif path == "/stream":
                    self._serve_stream(parsed.query)
                elif path == "/healthz":
                    self._send_text(HTTPStatus.OK, "ok")
                else:
                    self._send_text(HTTPStatus.NOT_FOUND, "not found")
            except (BrokenPipeError, ConnectionResetError):
                # Client went away mid-response; nothing to do.
                return

        # ── handlers ────────────────────────────────────────────────────────

        def _serve_state(self) -> None:
            latest, ticks = feed.snapshot()
            body = json.dumps(
                {"latest_tick": latest, "ticks": ticks}, default=str
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_stream(self, query: str) -> None:
            since = _parse_since(query)
            q, replay = feed.subscribe(since_tick=since)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")  # nginx pass-through
            self.end_headers()
            try:
                for rec in replay:
                    self._sse_send(rec)
                last_beat = time.monotonic()
                while True:
                    try:
                        rec = q.get(timeout=_SSE_HEARTBEAT_SECONDS)
                    except queue.Empty:
                        rec = None
                    now = time.monotonic()
                    if rec is not None:
                        self._sse_send(rec)
                        last_beat = now
                    elif now - last_beat >= _SSE_HEARTBEAT_SECONDS:
                        # Comment frame keeps idle proxies from reaping us.
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        last_beat = now
            finally:
                feed.unsubscribe(q)

        def _sse_send(self, record: dict[str, Any]) -> None:
            payload = json.dumps(record, default=str)
            self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
            self.wfile.flush()

        # ── static serving ──────────────────────────────────────────────────

        _MIME = {
            ".html": "text/html; charset=utf-8",
            ".js":   "application/javascript; charset=utf-8",
            ".css":  "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg":  "image/svg+xml",
            ".png":  "image/png",
            ".map":  "application/json; charset=utf-8",
        }

        def _serve_static(self, rel: str) -> None:
            target = (static_root / rel).resolve()
            try:
                target.relative_to(static_root)
            except ValueError:
                self._send_text(HTTPStatus.FORBIDDEN, "forbidden")
                return
            mime = self._MIME.get(target.suffix.lower(), "application/octet-stream")
            self._serve_file(target, mime)

        def _serve_file(self, path: Path, mime: str) -> None:
            if not path.is_file():
                self._send_text(HTTPStatus.NOT_FOUND, f"not found: {path.name}")
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, status: HTTPStatus, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return DashboardHandler


def _parse_since(query: str) -> Optional[int]:
    if not query:
        return None
    qs = parse_qs(query)
    raw = qs.get("since", [None])[0]
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


# ─────────────────────────────────────── server bootstrap ─────────────────────


_DEFAULT_DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


def start_server(
    host: str,
    port: int,
    feed: HTTPFeed,
    dashboard_dir: Optional[Path] = None,
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Boot a daemon HTTP server thread; return (server, thread)."""
    dashboard_dir = (dashboard_dir or _DEFAULT_DASHBOARD_DIR).resolve()
    handler_cls = _make_handler(feed, dashboard_dir)
    server = ThreadingHTTPServer((host, port), handler_cls)
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever,
        name="sidecar-http",
        daemon=True,
    )
    thread.start()
    return server, thread


# ─────────────────────────────────────── replay mode ──────────────────────────


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _ts_to_epoch(ts: Any) -> Optional[float]:
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


def replay_into_feed(
    feed: HTTPFeed,
    log_path: Path,
    speed: float = 1.0,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Stream records from ``log_path`` into ``feed`` at the given speed.

    speed=1.0 → real-time (use ts deltas from the log).
    speed=0   → as-fast-as-possible (no sleeping).
    speed>1   → faster than real-time.
    """
    prev_epoch: Optional[float] = None
    for record in _iter_jsonl(log_path):
        if stop_event is not None and stop_event.is_set():
            return
        if speed > 0:
            cur_epoch = _ts_to_epoch(record.get("ts"))
            if cur_epoch is not None and prev_epoch is not None:
                delay = max(0.0, (cur_epoch - prev_epoch) / speed)
                if stop_event is not None:
                    if stop_event.wait(timeout=delay):
                        return
                else:
                    time.sleep(delay)
            prev_epoch = cur_epoch if cur_epoch is not None else prev_epoch
        feed.publish(record)


# ─────────────────────────────────────── CLI ──────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Live HTTP/SSE feed for the agent-state dashboard. With --replay, "
            "streams a finished sidecar.log into the same endpoint at the "
            "configured speed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--dashboard-dir", default=str(_DEFAULT_DASHBOARD_DIR),
                   help="Directory containing index.html and static assets.")
    p.add_argument("--history", type=int, default=_DEFAULT_HISTORY,
                   help="Ring-buffer size for /state.")
    p.add_argument("--replay", default=None,
                   help="Path to a sidecar.log JSONL to replay through the feed.")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Replay speed multiplier (0 = as fast as possible).")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    feed = HTTPFeed(history_size=args.history)
    server, _ = start_server(args.host, args.port, feed, Path(args.dashboard_dir))
    print(f"[sidecar-http] listening on http://{args.host}:{args.port}/", flush=True)

    if args.replay:
        log_path = Path(args.replay)
        if not log_path.is_file():
            print(f"[sidecar-http] replay log not found: {log_path}",
                  file=sys.stderr, flush=True)
            return 2
        print(f"[sidecar-http] replaying {log_path} @ speed={args.speed}",
              flush=True)
        try:
            replay_into_feed(feed, log_path, speed=args.speed)
        except KeyboardInterrupt:
            pass
        print("[sidecar-http] replay finished; server still serving "
              "ring buffer. Ctrl-C to exit.", flush=True)

    try:
        # Keep the main thread alive while the server thread serves.
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[sidecar-http] shutting down.", flush=True)
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
