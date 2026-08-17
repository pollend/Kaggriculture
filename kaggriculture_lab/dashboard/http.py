"""The HTTP request handler: routing, static assets, and error translation."""

from __future__ import annotations

import hashlib
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlsplit

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard import mcp_transport
from kaggriculture_lab.dashboard.assets import asset
from kaggriculture_lab.dashboard.cache import RenderCache, page_key
from kaggriculture_lab.dashboard.errors import (
    DashboardBadRequest,
    DashboardNotFound,
    DashboardUnavailable,
)
from kaggriculture_lab.dashboard.events import stream_dashboard
from kaggriculture_lab.dashboard.html import _escape, _layout
from kaggriculture_lab.dashboard.pages.replay import _load_replay, _replay_page
from kaggriculture_lab.dashboard.repository import DashboardRepository
from kaggriculture_lab.dashboard.routes import normalise, resolve
from kaggriculture_lab.progress_mcp import ProgressMCP

DEFAULT_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self'; connect-src 'self'; frame-src 'self'; base-uri 'none'; "
    "frame-ancestors 'none'"
)

# The replay document is rendered by kaggle-environments as one self-contained
# blob of inline script, so it cannot run under the CSP the rest of the site uses.
REPLAY_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data:; font-src data:; media-src data:; connect-src 'none'; "
    "base-uri 'none'; frame-ancestors 'self'"
)

STATIC_ROUTES = {"/dashboard.js": "dashboard.js", "/dashboard.css": "dashboard.css"}


class DashboardHandler(BaseHTTPRequestHandler):
    repository: DashboardRepository
    mcp_backend: ProgressMCP
    http_log_stream: Any
    render_cache: RenderCache
    server_version = "KaggricultureDashboard/0.1"

    def render_page(self, path: str, query: dict[str, list[str]]) -> tuple[str, str]:
        """Render one page, or return the cached copy. Also gives back its ETag.

        The watermark query runs first and costs one row from shared memory, so a
        cache hit answers without touching a single league table. A conditional
        GET that matches never even renders.
        """
        render = resolve(path, query)
        key = page_key(path, query)
        with self.repository.connect() as connection:
            watermark = lq.watermark(connection)
            # hashlib rather than `hash()`: the built-in is salted per process,
            # which would invalidate every browser's cached page on restart.
            digest = hashlib.sha256(key.encode()).hexdigest()[:12]
            etag = f'W/"{digest}-{watermark:x}"'
            if self.headers.get("If-None-Match") == etag:
                return "", etag
            cached = self.render_cache.get(key, watermark)
            if cached is not None:
                return cached, etag
            document = render(connection)
        self.render_cache.put(key, watermark, document)
        return document, etag

    def log_message(self, format: str, *args: Any) -> None:
        """Keep HTTP access logs out of the captured training console."""

        self.http_log_stream.write(
            f'{self.address_string()} - - [{self.log_date_time_string()}] {format % args}\n'
        )
        self.http_log_stream.flush()

    def do_GET(self) -> None:  # noqa: N802
        self._handle(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle(head=True)

    def do_POST(self) -> None:  # noqa: N802
        if normalise(urlsplit(self.path).path) != "/mcp":
            self._send_text(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "Method not allowed",
                head=False,
                headers={"Allow": "GET, HEAD"},
            )
            return
        mcp_transport.handle_mcp(self)

    def _handle(self, *, head: bool) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        path = normalise(parsed.path)
        try:
            if path == "/mcp":
                self._send_text(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    "This stateless MCP endpoint accepts POST requests.",
                    head=head,
                    headers={"Allow": "POST"},
                )
                return
            if path in STATIC_ROUTES:
                self._send_asset(STATIC_ROUTES[path], head=head)
                return
            if path == "/events/dashboard":
                if head:
                    self._send_text(
                        HTTPStatus.METHOD_NOT_ALLOWED,
                        "Method not allowed",
                        head=True,
                        headers={"Allow": "GET"},
                    )
                    return
                stream_dashboard(self, query)
                return
            if path == "/healthz":
                with self.repository.connect() as connection:
                    sessions = int(
                        connection.execute("SELECT count(*) FROM training_sessions").fetchone()[0]
                    )
                self._send_json(HTTPStatus.OK, {"status": "ok", "sessions": sessions}, head=head)
                return
            if path.startswith("/episodes/") and path.endswith("/replay.json"):
                from kaggriculture_lab.dashboard.routes import identifier

                match_id = identifier(path, "/episodes/", "/replay.json")
                with self.repository.connect() as connection:
                    payload, _ = _load_replay(connection, match_id)
                self._send(
                    HTTPStatus.OK,
                    payload,
                    "application/json; charset=utf-8",
                    head=head,
                    headers={"Content-Disposition": 'attachment; filename="replay.json"'},
                )
                return
            if path.startswith("/episodes/") and path.endswith("/replay"):
                from kaggriculture_lab.dashboard.routes import identifier

                match_id = identifier(path, "/episodes/", "/replay")
                with self.repository.connect() as connection:
                    document = _replay_page(connection, match_id)
                self._send(
                    HTTPStatus.OK,
                    document.encode(),
                    "text/html; charset=utf-8",
                    head=head,
                    content_security_policy=REPLAY_CSP,
                )
                return
            document, etag = self.render_page(path, query)
            headers = {"Cache-Control": "no-cache", "ETag": etag}
            if not document:
                self._send(
                    HTTPStatus.NOT_MODIFIED,
                    b"",
                    "text/html; charset=utf-8",
                    head=True,
                    headers=headers,
                )
                return
            self._send(
                HTTPStatus.OK,
                document.encode(),
                "text/html; charset=utf-8",
                head=head,
                headers=headers,
            )
        except DashboardBadRequest as error:
            self._error(HTTPStatus.BAD_REQUEST, "Bad request", str(error), head=head)
        except DashboardNotFound as error:
            self._error(HTTPStatus.NOT_FOUND, "Not found", str(error), head=head)
        except DashboardUnavailable as error:
            cause = error.__cause__ or error
            print(f"dashboard database unavailable: {cause}", file=sys.stderr)
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Database unavailable",
                "PostgreSQL is unavailable. Check the local lab stack and refresh this page.",
                head=head,
                headers={"Retry-After": "5"},
            )
        except Exception as error:
            print(f"dashboard request failed: {error!r}", file=sys.stderr)
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Dashboard error",
                "The request could not be completed. See the server log for details.",
                head=head,
            )

    def _send_asset(self, name: str, *, head: bool) -> None:
        """Serve a static asset, answering 304 when the browser already has it.

        `no-cache` rather than `no-store`: the browser must revalidate, but a
        matching ETag then costs a header exchange instead of the whole file.
        """
        static = asset(name)
        headers = {"Cache-Control": "no-cache", "ETag": static.etag}
        if self.headers.get("If-None-Match") == static.etag:
            self._send(HTTPStatus.NOT_MODIFIED, b"", static.content_type, head=True, headers=headers)
            return
        self._send(HTTPStatus.OK, static.body, static.content_type, head=head, headers=headers)

    def _error(
        self,
        status: HTTPStatus,
        title: str,
        message: str,
        *,
        head: bool,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = _layout(
            title,
            f'<div class="error"><h1>{_escape(title)}</h1><p>{_escape(message)}</p>'
            '<p><a href="/">Return to sessions</a></p></div>',
        )
        self._send(status, body.encode(), "text/html; charset=utf-8", head=head, headers=headers)

    def _send_json(self, status: HTTPStatus, value: Any, *, head: bool) -> None:
        self._send(
            status,
            json.dumps(value, sort_keys=True).encode(),
            "application/json; charset=utf-8",
            head=head,
        )

    def _send_text(
        self,
        status: HTTPStatus,
        value: str,
        *,
        head: bool,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._send(status, value.encode(), "text/plain; charset=utf-8", head=head, headers=headers)

    def _send(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
        *,
        head: bool,
        headers: dict[str, str] | None = None,
        content_security_policy: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Security-Policy", content_security_policy or DEFAULT_CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head:
            self.wfile.write(payload)
