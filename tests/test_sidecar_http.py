from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sidecar_http import (  # noqa: E402  (path setup above)
    HTTPFeed,
    replay_into_feed,
    start_server,
)


def _tick(n: int, ts: str = None) -> dict:
    return {
        "ts": ts or f"2026-04-29T00:00:{n:02d}+00:00",
        "tick": n,
        "vllm": {"kv_cache_used_pct": 10.0 + n},
        "agents": {},
        "admission": {"enabled": False, "reasons": ["not_configured"]},
    }


def test_ring_buffer_keeps_last_n_records():
    feed = HTTPFeed(history_size=3)
    for i in range(5):
        feed.publish(_tick(i))
    latest, ticks = feed.snapshot()
    assert latest == 4
    assert [r["tick"] for r in ticks] == [2, 3, 4]


def test_subscribe_replay_filters_by_since_tick():
    feed = HTTPFeed(history_size=10)
    for i in range(5):
        feed.publish(_tick(i))
    q, replay = feed.subscribe(since_tick=2)
    assert [r["tick"] for r in replay] == [3, 4]
    feed.publish(_tick(5))
    item = q.get(timeout=1.0)
    assert item["tick"] == 5
    feed.unsubscribe(q)


def test_subscribe_without_since_replays_all_buffer():
    feed = HTTPFeed(history_size=10)
    for i in range(3):
        feed.publish(_tick(i))
    _, replay = feed.subscribe()
    assert [r["tick"] for r in replay] == [0, 1, 2]


def test_publish_dispatches_to_all_subscribers():
    feed = HTTPFeed(history_size=10)
    q1, _ = feed.subscribe()
    q2, _ = feed.subscribe()
    feed.publish(_tick(7))
    assert q1.get(timeout=1.0)["tick"] == 7
    assert q2.get(timeout=1.0)["tick"] == 7


def test_replay_into_feed_at_zero_speed_is_instant():
    feed = HTTPFeed(history_size=100)
    log_path = REPO_ROOT / "tests" / "_replay_tmp.jsonl"
    log_path.write_text("\n".join(json.dumps(_tick(i)) for i in range(5)) + "\n")
    try:
        t0 = time.monotonic()
        replay_into_feed(feed, log_path, speed=0)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0
        _, ticks = feed.snapshot()
        assert [r["tick"] for r in ticks] == [0, 1, 2, 3, 4]
    finally:
        log_path.unlink(missing_ok=True)


def test_http_state_endpoint_returns_buffered_records():
    feed = HTTPFeed(history_size=10)
    server, _ = start_server("127.0.0.1", 0, feed,
                             dashboard_dir=REPO_ROOT / "dashboard")
    port = server.server_address[1]
    try:
        for i in range(3):
            feed.publish(_tick(i))
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/state", timeout=2.0
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["latest_tick"] == 2
        assert [r["tick"] for r in body["ticks"]] == [0, 1, 2]
    finally:
        server.shutdown()


def test_http_healthz_returns_ok():
    feed = HTTPFeed()
    server, _ = start_server("127.0.0.1", 0, feed,
                             dashboard_dir=REPO_ROOT / "dashboard")
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=2.0
        ) as resp:
            assert resp.read().decode("utf-8") == "ok"
    finally:
        server.shutdown()


def test_http_static_serving_blocks_path_traversal():
    feed = HTTPFeed()
    server, _ = start_server("127.0.0.1", 0, feed,
                             dashboard_dir=REPO_ROOT / "dashboard")
    port = server.server_address[1]
    try:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/static/../src/sidecar.py", timeout=2.0
            )
            raised = False
        except urllib.error.HTTPError as e:
            raised = True
            assert e.code in (403, 404)
        assert raised, "expected path traversal to be rejected"
    finally:
        server.shutdown()


def test_http_stream_replays_buffered_then_streams_live():
    feed = HTTPFeed(history_size=10)
    server, _ = start_server("127.0.0.1", 0, feed,
                             dashboard_dir=REPO_ROOT / "dashboard")
    port = server.server_address[1]
    try:
        for i in range(2):
            feed.publish(_tick(i))
        ticks_seen: list[int] = []
        done = threading.Event()

        def reader():
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/stream?since=-1",
                headers={"Accept": "text/event-stream"},
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                buf = b""
                while not done.is_set():
                    chunk = resp.read1(2048)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n\n" in buf:
                        frame, buf = buf.split(b"\n\n", 1)
                        for line in frame.splitlines():
                            if line.startswith(b"data: "):
                                rec = json.loads(line[len(b"data: "):])
                                ticks_seen.append(rec["tick"])
                                if len(ticks_seen) >= 3:
                                    done.set()
                                    return

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        # Give the subscriber a moment to attach, then publish a live record.
        time.sleep(0.2)
        feed.publish(_tick(2))
        t.join(timeout=5.0)
        assert ticks_seen == [0, 1, 2]
    finally:
        done.set()
        server.shutdown()


if __name__ == "__main__":
    for name in sorted(n for n in globals() if n.startswith("test_")):
        print(f"running {name} …", flush=True)
        globals()[name]()
        print(f"  ✓ {name}", flush=True)
    print("sidecar_http tests: ok")
