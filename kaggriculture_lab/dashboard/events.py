"""The server-sent event stream that keeps an open page current.

The browser holds one `EventSource` per tab and this end pushes the page's
`<main>` fragment whenever it changes. Three things keep that from costing a full
page render per tab per second:

* a watermark query, one row out of shared memory, that says whether anything was
  written at all -- if not, nothing is rendered;
* the shared render cache, so N tabs on the same page at the same database state
  pay for one render between them;
* a backoff, so an idle stream drifts from one second up to ten rather than asking
  forever at full rate.

The number of concurrent streams is capped, because each one holds a thread and a
dedicated connection and nothing else bounds them.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlsplit

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard.cache import page_key
from kaggriculture_lab.dashboard.errors import DashboardBadRequest
from kaggriculture_lab.dashboard.html import _main_fragment
from kaggriculture_lab.dashboard.routes import normalise, resolve, stream_allows

POLL_SECONDS = 1.0
MAX_POLL_SECONDS = 10.0
BACKOFF_FACTOR = 1.5
KEEPALIVE_SECONDS = 15.0
MAX_CONCURRENT_STREAMS = 32


class StreamLimit:
    """A counter of live streams, so a page left open in fifty tabs cannot exhaust
    the server's threads and connections."""

    def __init__(self, limit: int = MAX_CONCURRENT_STREAMS):
        self.limit = limit
        self._live = 0
        self._lock = threading.Lock()

    def claim(self) -> bool:
        with self._lock:
            if self._live >= self.limit:
                return False
            self._live += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._live = max(0, self._live - 1)

    @property
    def live(self) -> int:
        with self._lock:
            return self._live


_streams = StreamLimit()


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def stream_dashboard(
    handler: Any,
    query: dict[str, list[str]],
    *,
    poll_seconds: float = POLL_SECONDS,
    max_poll_seconds: float = MAX_POLL_SECONDS,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
    max_ticks: int | None = None,
    limit: StreamLimit | None = None,
) -> None:
    """Stream `<main>` fragments for the page named by `query["path"]`.

    The keyword arguments exist so a test can drive this without waiting; nothing
    in the server passes them.
    """
    target = query.get("path", ["/"])[0]
    if not stream_allows(target):
        raise DashboardBadRequest("invalid dashboard event path")
    parsed = urlsplit(target)
    path = normalise(parsed.path)
    target_query = parse_qs(parsed.query)
    key = page_key(path, target_query)

    limit = limit or _streams
    if not limit.claim():
        handler._send_text(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Too many open dashboard event streams.",
            head=False,
            headers={"Retry-After": "5"},
        )
        return

    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()

    previous_digest = ""
    previous_watermark: int | None = None
    interval = poll_seconds
    last_write = time.monotonic()
    ticks = 0
    try:
        # Dedicated rather than pooled: this connection is held for as long as the
        # tab stays open, and a handful of forgotten tabs holding pool slots would
        # starve ordinary page loads.
        with handler.repository.connect_dedicated() as connection:
            while max_ticks is None or ticks < max_ticks:
                watermark = lq.watermark(connection)
                now = time.monotonic()
                if watermark != previous_watermark:
                    previous_watermark = watermark
                    document = handler.render_cache.get(key, watermark)
                    if document is None:
                        document = resolve(path, target_query)(connection)
                        handler.render_cache.put(key, watermark, document)
                    fragment = _main_fragment(document)
                    digest = hashlib.sha256(fragment.encode()).hexdigest()
                    if digest != previous_digest:
                        handler.wfile.write(
                            _sse(
                                "dashboard.fragment",
                                {
                                    "selector": "main",
                                    "mergeMode": "outer",
                                    "fragments": fragment,
                                },
                            )
                        )
                        previous_digest = digest
                        last_write = now
                        interval = poll_seconds
                        handler.wfile.flush()
                        ticks += 1
                        if max_ticks is None or ticks < max_ticks:
                            time.sleep(interval)
                        continue
                # Nothing to send. Back off, but keep the connection warm on a
                # wall-clock schedule -- a keepalive counted in ticks silently
                # stretches to minutes as the interval grows, and proxies drop
                # streams that go quiet.
                if now - last_write >= keepalive_seconds:
                    handler.wfile.write(b": keepalive\n\n")
                    last_write = now
                interval = min(max_poll_seconds, interval * BACKOFF_FACTOR)
                handler.wfile.flush()
                ticks += 1
                if max_ticks is None or ticks < max_ticks:
                    time.sleep(interval)
    except (BrokenPipeError, ConnectionResetError):
        return
    except Exception as error:
        print(f"dashboard event stream failed: {error!r}", file=sys.stderr)
        return
    finally:
        limit.release()
