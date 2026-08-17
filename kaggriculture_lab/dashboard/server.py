"""Building and running the dashboard server."""

from __future__ import annotations

import sys
from http.server import ThreadingHTTPServer

from kaggriculture_lab.codex_config import dashboard_mcp_url, sync_codex_mcp_config
from kaggriculture_lab.dashboard.cache import RenderCache
from kaggriculture_lab.dashboard.http import DashboardHandler
from kaggriculture_lab.dashboard.repository import DashboardRepository
from kaggriculture_lab.database import DatabaseTarget
from kaggriculture_lab.progress_mcp import ProgressMCP


def make_server(
    database: str | DatabaseTarget,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    repository = DashboardRepository(database)
    available = repository.validate()

    class BoundDashboardHandler(DashboardHandler):
        pass

    BoundDashboardHandler.repository = repository
    BoundDashboardHandler.mcp_backend = ProgressMCP(repository)
    BoundDashboardHandler.http_log_stream = sys.stderr
    # One cache per server, shared by every request thread and every event
    # stream: that sharing is the point, since it makes N tabs on one page cost
    # one render rather than N.
    BoundDashboardHandler.render_cache = RenderCache()
    server = ThreadingHTTPServer((host, port), BoundDashboardHandler)
    server.daemon_threads = True
    server.database_available_at_startup = available  # type: ignore[attr-defined]
    # `server_close` is what tests and `serve` call on the way out; draining the
    # pool there means a test that builds a server does not leave idle PostgreSQL
    # backends behind.
    close_socket = server.server_close

    def server_close() -> None:
        try:
            close_socket()
        finally:
            repository.close()

    server.server_close = server_close  # type: ignore[method-assign]
    return server


def serve(database: str | DatabaseTarget, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = make_server(database, host, port)
    actual_host, actual_port = server.server_address[:2]
    if host not in {"127.0.0.1", "::1", "localhost"}:
        print("warning: dashboard has no authentication and is bound beyond loopback", file=sys.stderr)
    mcp_url = dashboard_mcp_url(str(actual_host), int(actual_port))
    try:
        config_path = sync_codex_mcp_config(
            mcp_url,
        )
    except (OSError, ValueError) as error:
        print(f"warning: could not update project Codex MCP config: {error}", file=sys.stderr)
    else:
        print(f"Codex MCP config: {config_path}")
    print(f"Kaggriculture dashboard: http://{actual_host}:{actual_port}/")
    print(f"Kaggriculture MCP: {mcp_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
