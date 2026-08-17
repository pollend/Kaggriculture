"""The dashboard's CSS and JavaScript, read from `static/` and cached in memory.

They live as real files rather than string constants so an editor can lint and
highlight them. Each is read once and kept alongside a content hash, which is what
the handler serves as an `ETag` -- a browser that already has the file then
revalidates with `If-None-Match` and gets a 304 instead of the bytes.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

STATIC = Path(__file__).with_name("static")


class Asset:
    __slots__ = ("body", "content_type", "etag")

    def __init__(self, body: bytes, content_type: str):
        self.body = body
        self.content_type = content_type
        self.etag = f'"{hashlib.sha256(body).hexdigest()[:32]}"'


@lru_cache(maxsize=None)
def asset(name: str) -> Asset:
    """Load one static asset by file name. Unknown names raise `KeyError`."""
    content_types = {
        "dashboard.css": "text/css; charset=utf-8",
        "dashboard.js": "text/javascript; charset=utf-8",
    }
    if name not in content_types:
        raise KeyError(name)
    return Asset((STATIC / name).read_bytes(), content_types[name])


def stylesheet() -> str:
    return asset("dashboard.css").body.decode()
