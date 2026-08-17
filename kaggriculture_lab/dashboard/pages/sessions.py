"""The session list at `/`.

Split into `fetch` (everything that touches the database) and `render` (everything
that produces markup), with `_sessions_page` the composition of the two. The split
is what lets the markup be tested against hand-built rows, with no PostgreSQL.
"""

from __future__ import annotations

import json
from typing import Any

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard.html import (
    PAGE_SIZE,
    _escape,
    _id_path,
    _layout,
    _number,
    _pager,
    _progress,
    _status,
)


def fetch(connection: Any, page: int) -> dict[str, Any]:
    offset = (page - 1) * PAGE_SIZE
    sessions = lq.sessions(connection, limit=PAGE_SIZE + 1, offset=offset)
    has_next = len(sessions) > PAGE_SIZE
    sessions = sessions[:PAGE_SIZE]
    session_ids = [row["session_id"] for row in sessions]
    return {
        "page": page,
        "has_next": has_next,
        "sessions": sessions,
        "rollups": {row["session_id"]: row for row in lq.session_rollups(connection, session_ids)},
        "leaders": {row["session_id"]: row for row in lq.session_leaders(connection, session_ids)},
    }


def render(data: dict[str, Any]) -> str:
    sessions = data["sessions"]
    rollups, leaders = data["rollups"], data["leaders"]
    rows_html = []
    for session in sessions:
        session_id = session["session_id"]
        stats = rollups.get(session_id, {})
        leader = leaders.get(session_id)
        pairings = int(stats.get("pairings") or 0)
        completed = int(stats.get("completed_pairings") or 0)
        session_config = json.loads(str(session["config_json"]))
        trainer_only = session_config.get("pairing_mode") == "trainer_only"
        if trainer_only:
            planned_rounds = int(session_config.get("rounds_per_generation", 0)) * int(
                stats.get("generations") or 0
            )
            progress = (
                f'{int(stats.get("completed_rounds") or 0):,} / {planned_rounds:,} rounds'
                f'<br><small>{completed:,} pairings recorded</small>'
            )
        else:
            progress = _progress(completed, pairings)
        leader_html = "—" if not leader else f'{_escape(leader["display_name"])}<br><small>{_number(leader["conservative"], 1)}</small>'
        rows_html.append(
            f'<tr><td><a href="{_escape(_id_path("sessions", session_id))}"><b>{_escape(session["name"])}</b></a><br><small>{_escape(session_id)}</small></td>'
            f'<td>{_escape(session["league"])}</td><td>{_status(session["status"])}</td><td>{int(stats.get("generations") or 0):,}</td>'
            f'<td>{progress}</td><td>{int(stats.get("episodes") or 0):,}</td>'
            f'<td>{int(stats.get("failed_episodes") or 0):,}</td><td>{leader_html}</td><td>{_escape(session["created_at"])}</td></tr>'
        )
    table = (
        '<div class="panel"><table><thead><tr><th>Session</th><th>League</th><th>Status</th><th>Generations</th><th>Progress</th><th>Episodes</th><th>Failures</th><th>Latest leader</th><th>Created</th></tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table>{_pager("/", data["page"], data["has_next"])}</div>'
        if sessions
        else '<div class="panel empty">No league sessions have been recorded yet.</div>'
    )
    return _layout("Sessions", f'<h1>League sessions</h1><p class="muted">Every persistent training league in the shared database.</p>{table}')


def _sessions_page(connection: Any, page: int) -> str:
    return render(fetch(connection, page))
