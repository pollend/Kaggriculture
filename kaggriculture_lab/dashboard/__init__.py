"""Small web dashboard for persistent Kaggriculture leagues.

The package is layered: `repository` opens connections, `league_queries` (outside
this package) reads rows, `html` turns rows into markup, `pages/*` compose one page
each, `routes` maps a request to a page, and `http`/`events`/`mcp_transport` speak
the protocols. Only `mcp_transport` knows the MCP backend exists.

Everything the rest of the lab and the tests import is re-exported here, so this
module name stays the stable entry point it was when it was a single file.
"""

from __future__ import annotations

from kaggriculture_lab.dashboard.assets import asset, stylesheet
from kaggriculture_lab.dashboard.cache import RenderCache, page_key
from kaggriculture_lab.dashboard.errors import (
    DashboardBadRequest,
    DashboardNotFound,
    DashboardUnavailable,
)
from kaggriculture_lab.dashboard.events import stream_dashboard
from kaggriculture_lab.dashboard.html import (
    PAGE_SIZE,
    _agent_identity_html,
    _agent_replay_name,
    _cards,
    _chart,
    _duration,
    _escape,
    _href,
    _id_path,
    _layout,
    _main_fragment,
    _number,
    _pager,
    _percent,
    _policy_role_label,
    _progress,
    _status,
)
from kaggriculture_lab.dashboard.http import DashboardHandler
from kaggriculture_lab.dashboard.pages.episode import _episode_page
from kaggriculture_lab.dashboard.pages.failures import _failures_page
from kaggriculture_lab.dashboard.pages.generation import _generation_page
from kaggriculture_lab.dashboard.pages.model import _model_page
from kaggriculture_lab.dashboard.pages.replay import _load_replay, _replay_page
from kaggriculture_lab.dashboard.pages.session import _session_page
from kaggriculture_lab.dashboard.pages.sessions import _sessions_page
from kaggriculture_lab.dashboard.repository import DashboardRepository
from kaggriculture_lab.dashboard.routes import identifier, parse_page, resolve
from kaggriculture_lab.dashboard.server import make_server, serve

# The underscore-prefixed names are module-private to their own file but public
# here: they are the surface the tests and the rest of the lab import, and were
# importable from `kaggriculture_lab.dashboard` when it was one module.
__all__ = [
    "PAGE_SIZE",
    "DashboardBadRequest",
    "DashboardHandler",
    "DashboardNotFound",
    "DashboardRepository",
    "DashboardUnavailable",
    "_agent_identity_html",
    "_agent_replay_name",
    "_cards",
    "_chart",
    "_duration",
    "_episode_page",
    "_escape",
    "_failures_page",
    "_generation_page",
    "_href",
    "_id_path",
    "_layout",
    "_load_replay",
    "_model_page",
    "_main_fragment",
    "_number",
    "_pager",
    "_percent",
    "_policy_role_label",
    "_progress",
    "_replay_page",
    "_session_page",
    "_sessions_page",
    "_status",
    "RenderCache",
    "asset",
    "identifier",
    "make_server",
    "page_key",
    "parse_page",
    "resolve",
    "serve",
    "stream_dashboard",
    "stylesheet",
]
