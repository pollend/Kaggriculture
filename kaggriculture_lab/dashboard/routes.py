"""Turning a request path into the page renderer that answers it.

Resolution is deliberately separate from the HTTP handler: `resolve` needs only a
path and a query dict, so every routing rule -- and every way a request can be
rejected -- is testable without binding a socket.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import unquote

from kaggriculture_lab.dashboard.errors import DashboardBadRequest, DashboardNotFound
from kaggriculture_lab.dashboard.pages.episode import _episode_page
from kaggriculture_lab.dashboard.pages.failures import _failures_page
from kaggriculture_lab.dashboard.pages.generation import _generation_page
from kaggriculture_lab.dashboard.pages.model import _model_page
from kaggriculture_lab.dashboard.pages.session import _session_page
from kaggriculture_lab.dashboard.pages.sessions import _sessions_page

# Paths the SSE stream must never be pointed at: its own endpoint (which would
# recurse), the static asset, the health probe, and the replay routes, which are
# neither HTML fragments nor cheap to re-render.
STREAM_EXCLUDED_PREFIXES = ("/events/", "/dashboard.js", "/dashboard.css", "/healthz")
STREAM_EXCLUDED_SUFFIXES = ("/replay", "/replay.json")

MAX_FILTER_LENGTH = 80


def normalise(path: str) -> str:
    return path.rstrip("/") or "/"


def identifier(path: str, prefix: str, suffix: str = "") -> str:
    """Read a percent-encoded id out of a path segment.

    Rejects an embedded `/` before decoding: a decoded slash would silently change
    which route the id appears to belong to.
    """
    encoded = path[len(prefix):]
    if suffix:
        encoded = encoded[: -len(suffix)]
    if not encoded or "/" in encoded:
        raise DashboardBadRequest("invalid identifier")
    value = unquote(encoded)
    if not value or "\x00" in value:
        raise DashboardBadRequest("invalid identifier")
    return value


def parse_page(query: dict[str, list[str]]) -> int:
    raw = query.get("page", ["1"])[0]
    try:
        page = int(raw)
    except ValueError as error:
        raise DashboardBadRequest("page must be a positive integer") from error
    if page < 1:
        raise DashboardBadRequest("page must be a positive integer")
    return page


def resolve(path: str, query: dict[str, list[str]]) -> Callable[[Any], str]:
    """Return a renderer that takes a connection and produces the page document.

    Raises `DashboardBadRequest` for a malformed path or query and
    `DashboardNotFound` for one that matches no route, so the caller maps both to
    a status code without knowing the routing table.
    """
    path = normalise(path)
    if path == "/":
        page = parse_page(query)
        return lambda connection: _sessions_page(connection, page)
    if path.startswith("/sessions/") and path.endswith("/failures"):
        session_id = identifier(path, "/sessions/", "/failures")
        return lambda connection: _failures_page(connection, session_id)
    if path.startswith("/sessions/"):
        session_id = identifier(path, "/sessions/")
        return lambda connection: _session_page(connection, session_id)
    if path.startswith("/generations/"):
        generation_id = identifier(path, "/generations/")
        page = parse_page(query)
        status_filter = query.get("status", ["all"])[0]
        kind_filter = query.get("kind", ["all"])[0]
        sort = query.get("sort", ["round"])[0]
        direction = query.get("direction", ["asc"])[0]
        if any(
            len(value) > MAX_FILTER_LENGTH
            for value in (status_filter, kind_filter, sort, direction)
        ):
            raise DashboardBadRequest("filter value is too long")
        return lambda connection: _generation_page(
            connection,
            generation_id,
            page,
            status_filter,
            kind_filter,
            sort,
            direction,
        )
    if path.startswith("/episodes/"):
        match_id = identifier(path, "/episodes/")
        return lambda connection: _episode_page(connection, match_id)
    if path.startswith("/models/"):
        # Model ids are `<declared_model_id>@<agent_name>` -- no slash, so the
        # usual identifier rules apply unchanged.
        model_id = identifier(path, "/models/")
        return lambda connection: _model_page(connection, model_id)
    raise DashboardNotFound("page not found")


def stream_allows(target: str) -> bool:
    """Whether the event stream may re-render this path.

    A protocol-relative target (`//host/...`) is rejected along with the excluded
    routes: it is not a path on this server at all.
    """
    if not target.startswith("/") or target.startswith("//"):
        return False
    from urllib.parse import urlsplit

    path = normalise(urlsplit(target).path)
    if path.startswith(STREAM_EXCLUDED_PREFIXES):
        return False
    return not (path.startswith("/episodes/") and path.endswith(STREAM_EXCLUDED_SUFFIXES))
