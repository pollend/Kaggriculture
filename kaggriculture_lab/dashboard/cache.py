"""A byte-capped cache of rendered pages, keyed by page and database watermark.

Ten browser tabs on one generation page used to cost ten renders a second. The
watermark makes the whole set collapse: the key includes it, so every tab on the
same page at the same database state shares one entry, and the moment anything is
written every key changes at once and the stale entries fall out of the cache.

Entries are dropped least-recently-used once the total exceeds `max_bytes`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any
from urllib.parse import urlencode

# Roughly a hundred generation pages. Large enough that a lab with a few sessions
# and a few open tabs never evicts; small enough to bound a runaway.
DEFAULT_MAX_BYTES = 8 * 1024 * 1024


def page_key(path: str, query: dict[str, list[str]]) -> str:
    """A stable key for one page and its query.

    Sorted and re-encoded so `?a=1&b=2` and `?b=2&a=1` are one entry rather than
    two, and so a repeated parameter cannot make two callers disagree about which
    page they are looking at.
    """
    pairs = sorted((key, value) for key, values in query.items() for value in values)
    encoded = urlencode(pairs)
    return f"{path}?{encoded}" if encoded else path


class RenderCache:
    def __init__(self, *, max_bytes: int = DEFAULT_MAX_BYTES):
        self._entries: OrderedDict[str, tuple[int, str]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.max_bytes = max_bytes
        self.hits = 0
        self.misses = 0

    def get(self, key: str, watermark: int) -> str | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0] != watermark:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[1]

    def put(self, key: str, watermark: int, document: str) -> None:
        size = len(document)
        if size > self.max_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= len(previous[1])
            self._entries[key] = (watermark, document)
            self._bytes += size
            while self._bytes > self.max_bytes and len(self._entries) > 1:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= len(evicted[1])

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "hits": self.hits,
                "misses": self.misses,
            }
