"""Small, read-only web dashboard for persistent Kaggriculture leagues."""

from __future__ import annotations

import gzip
import html
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlsplit

from kaggriculture_lab.storage import SCHEMA_VERSION

PAGE_SIZE = 50


class DashboardUnavailable(RuntimeError):
    """The catalog cannot currently be opened, normally because a writer owns it."""


class DashboardNotFound(LookupError):
    """A requested dashboard resource does not exist."""


class DashboardBadRequest(ValueError):
    """A dashboard route or query parameter is invalid."""


def _rows(connection: Any, sql: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, parameters or [])
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _row(connection: Any, sql: str, parameters: list[Any] | None = None) -> dict[str, Any] | None:
    rows = _rows(connection, sql, parameters)
    return rows[0] if rows else None


class DashboardRepository:
    """Narrow read-only access to the league portion of the experiment catalog."""

    def __init__(self, database: str | Path):
        self.path = Path(database)

    @contextmanager
    def connect(self) -> Iterator[Any]:
        import duckdb

        try:
            connection = duckdb.connect(str(self.path), read_only=True)
        except duckdb.Error as error:
            raise DashboardUnavailable("The database is temporarily unavailable.") from error
        try:
            yield connection
        finally:
            connection.close()

    def validate(self) -> bool:
        """Validate when possible; return False when a running writer holds the file."""
        if not self.path.is_file():
            raise FileNotFoundError(
                f"database does not exist: {self.path} (create a league session first)"
            )
        try:
            with self.connect() as connection:
                row = connection.execute("SELECT max(version) FROM schema_migrations").fetchone()
        except DashboardUnavailable:
            return False
        except Exception as error:
            raise ValueError(f"not a Kaggriculture experiment database: {self.path}") from error
        version = row[0] if row else None
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported experiment schema version {version!r}; expected {SCHEMA_VERSION}"
            )
        return True


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _number(value: Any, digits: int = 0) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:,.{digits}f}"


def _percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def _status(value: Any) -> str:
    label = str(value or "UNKNOWN")
    kind = "good" if label in {"COMPLETE", "DONE", "SUCCESS"} else "bad" if label in {"ERROR", "FAILED"} else "warn"
    return f'<span class="status {kind}">{_escape(label)}</span>'


def _href(path: str, **query: Any) -> str:
    values = {key: value for key, value in query.items() if value not in (None, "", "all")}
    return path + (f"?{urlencode(values)}" if values else "")


def _id_path(prefix: str, identifier: Any) -> str:
    return f"/{prefix}/{quote(str(identifier), safe='')}"


def _agent_identity_html(
    display_name: Any,
    strategy_id: Any,
    revision_id: Any,
    checkpoint_id: Any,
    *,
    label: str | None = None,
) -> str:
    heading = _escape(display_name or revision_id or "Unknown agent")
    if label:
        heading = f'<span class="muted">{_escape(label)}</span> · {heading}'
    details = []
    if strategy_id:
        details.append(f"strategy {_escape(strategy_id)}")
    if revision_id:
        details.append(f"revision {_escape(revision_id)}")
    if checkpoint_id:
        details.append(f"checkpoint {_escape(checkpoint_id)}")
    suffix = "<br>".join(f"<small>{item}</small>" for item in details)
    return f"<b>{heading}</b>" + (f"<br>{suffix}" if suffix else "")


def _agent_replay_name(display_name: Any, strategy_id: Any, revision_id: Any, checkpoint_id: Any) -> str:
    name = str(display_name or "Agent")
    variant = strategy_id or checkpoint_id or revision_id
    return f"{name} · {variant}" if variant else name


CSS = """
:root{color-scheme:dark;--bg:#0b1220;--panel:#121c2f;--line:#263650;--ink:#e8eef8;
--muted:#9dafc8;--accent:#62d4c7;--gold:#f0c66b;--bad:#ff8181;--warn:#ffcf70}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}header{border-bottom:1px solid var(--line);
background:#0e1828;position:sticky;top:0;z-index:2}.bar{max-width:1200px;margin:auto;padding:14px 22px;display:flex;
align-items:center;justify-content:space-between;gap:20px}.brand{font-weight:750;color:var(--ink);letter-spacing:.02em}
main{max-width:1200px;margin:auto;padding:28px 22px 60px}h1{font-size:29px;margin:0 0 8px}h2{font-size:20px;margin:32px 0 12px}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:20px 0}
.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px}.card{padding:14px}.card b{font-size:22px;display:block}
.panel{padding:18px;margin:14px 0;overflow:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
table{border-collapse:collapse;width:100%;min-width:680px}th,td{text-align:left;padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}tr:last-child td{border-bottom:0}
.status{font-size:11px;font-weight:700;padding:3px 7px;border-radius:999px;background:#263650}.status.good{color:#7de3ae;background:#153b31}
.status.bad{color:#ffaaa5;background:#482329}.status.warn{color:var(--warn);background:#473b1e}.progress{height:8px;background:#263650;border-radius:9px;overflow:hidden}
.progress i{display:block;height:100%;background:var(--accent)}.actions{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.button{border:1px solid var(--line);
padding:7px 11px;border-radius:8px;background:var(--panel)}form.filters{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin:0 0 16px}
label{font-size:12px;color:var(--muted)}select{display:block;margin-top:4px;background:#0e1828;color:var(--ink);border:1px solid var(--line);padding:7px;border-radius:7px}
pre{white-space:pre-wrap;word-break:break-word;background:#0a111d;border:1px solid var(--line);padding:12px;border-radius:8px;max-height:420px;overflow:auto}
details{margin:8px 0}.pager{display:flex;justify-content:space-between;margin-top:16px}.chart{width:100%;height:auto;min-width:500px}.empty{text-align:center;padding:42px;color:var(--muted)}
.error{max-width:700px;margin:70px auto;padding:28px;background:var(--panel);border:1px solid var(--line);border-radius:14px}
.versus{color:var(--muted);font-size:12px;font-weight:700;margin:8px 0;text-transform:uppercase}
@media(max-width:650px){main{padding:20px 12px}.bar{padding:12px}.grid{grid-template-columns:1fr}h1{font-size:24px}}
"""


def _layout(title: str, body: str, *, refresh: bool = True) -> str:
    refresh_link = '<a class="button" href="">Refresh</a>' if refresh else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(title)} · Kaggriculture</title>
<style>{CSS}</style></head><body><header><div class="bar"><a class="brand" href="/">Kaggriculture League</a>
<div class="actions">{refresh_link}</div></div></header><main>{body}</main></body></html>"""


def _cards(items: list[tuple[str, str]]) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><span class="muted">{_escape(label)}</span><b>{value}</b></div>'
        for label, value in items
    ) + "</div>"


def _progress(done: int, total: int) -> str:
    fraction = min(1.0, done / total) if total else 0.0
    return f'<div class="progress" title="{done} of {total}"><i style="width:{100 * fraction:.1f}%"></i></div><small>{done:,} / {total:,}</small>'


def _pager(path: str, page: int, has_next: bool, **query: Any) -> str:
    previous = f'<a class="button" href="{_escape(_href(path, page=page - 1, **query))}">← Previous</a>' if page > 1 else "<span></span>"
    following = f'<a class="button" href="{_escape(_href(path, page=page + 1, **query))}">Next →</a>' if has_next else "<span></span>"
    return f'<div class="pager">{previous}<span class="muted">Page {page}</span>{following}</div>'


def _chart(trend: list[dict[str, Any]]) -> str:
    values = [float(row[key]) for row in trend for key in ("champion", "median_rating") if row[key] is not None]
    if not values:
        return '<div class="empty">Ratings appear after a generation closes.</div>'
    width, height, left, top, right, bottom = 760, 230, 55, 20, 20, 38
    low, high = min(values), max(values)
    padding = max(10.0, (high - low) * 0.1)
    low, high = low - padding, high + padding
    span = max(1.0, high - low)
    count = max(1, len(trend) - 1)

    def points(key: str) -> str:
        result = []
        for index, row in enumerate(trend):
            if row[key] is None:
                continue
            x = left + index * (width - left - right) / count
            y = top + (high - float(row[key])) * (height - top - bottom) / span
            result.append(f"{x:.1f},{y:.1f}")
        return " ".join(result)

    ticks = "".join(
        f'<text x="{left + index * (width - left - right) / count:.1f}" y="{height - 12}" fill="#9dafc8" text-anchor="middle">g{int(row["generation_index"])}</text>'
        for index, row in enumerate(trend)
    )
    axes = "".join(
        f'<g><line x1="{left}" x2="{width-right}" y1="{top + step * (height-top-bottom)/4:.1f}" y2="{top + step * (height-top-bottom)/4:.1f}" stroke="#263650"/>'
        f'<text x="{left-8}" y="{top + step * (height-top-bottom)/4 + 4:.1f}" fill="#9dafc8" text-anchor="end">{high-step*span/4:.0f}</text></g>'
        for step in range(5)
    )
    return f'''<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Rating trend">
{axes}{ticks}<polyline fill="none" stroke="#62d4c7" stroke-width="3" points="{points('champion')}"/>
<polyline fill="none" stroke="#f0c66b" stroke-width="3" points="{points('median_rating')}"/>
<text x="{left}" y="14" fill="#62d4c7">Champion conservative rating</text>
<text x="270" y="14" fill="#f0c66b">Median conservative rating</text></svg>'''


def _parse_page(query: dict[str, list[str]]) -> int:
    raw = query.get("page", ["1"])[0]
    try:
        page = int(raw)
    except ValueError as error:
        raise DashboardBadRequest("page must be a positive integer") from error
    if page < 1:
        raise DashboardBadRequest("page must be a positive integer")
    return page


def _sessions_page(connection: Any, page: int) -> str:
    offset = (page - 1) * PAGE_SIZE
    sessions = _rows(
        connection,
        """
        SELECT session_id, name, status, seed, created_at
        FROM training_sessions ORDER BY created_at DESC, session_id LIMIT ? OFFSET ?
        """,
        [PAGE_SIZE + 1, offset],
    )
    has_next = len(sessions) > PAGE_SIZE
    sessions = sessions[:PAGE_SIZE]
    session_ids = [row["session_id"] for row in sessions]
    rollups: dict[str, dict[str, Any]] = {}
    leaders: dict[str, dict[str, Any]] = {}
    if session_ids:
        placeholders = ",".join("?" for _ in session_ids)
        rollups = {
            row["session_id"]: row
            for row in _rows(
                connection,
                f"""
                SELECT g.session_id, count(DISTINCT g.generation_id) AS generations,
                       count(DISTINCT lp.pairing_id) AS pairings,
                       count(DISTINCT CASE WHEN lp.status = 'COMPLETE' THEN lp.pairing_id END) AS completed_pairings,
                       count(DISTINCT pe.match_id) AS episodes,
                       count(DISTINCT CASE WHEN e.failed THEN e.match_id END) AS failed_episodes
                FROM generations g
                LEFT JOIN league_pairings lp USING (generation_id)
                LEFT JOIN pairing_episodes pe USING (pairing_id)
                LEFT JOIN episodes e USING (match_id)
                WHERE g.session_id IN ({placeholders}) GROUP BY g.session_id
                """,
                session_ids,
            )
        }
        leaders = {
            row["session_id"]: row
            for row in _rows(
                connection,
                f"""
                SELECT session_id, revision_id, display_name, conservative FROM (
                    SELECT g.session_id, rs.revision_id,
                           coalesce(af.name, ar.family_id, rs.revision_id) AS display_name,
                           rs.rating - 2 * rs.deviation AS conservative,
                           row_number() OVER (
                               PARTITION BY g.session_id
                               ORDER BY g.generation_index DESC, conservative DESC, rs.revision_id
                           ) AS position
                    FROM generations g JOIN rating_snapshots rs USING (generation_id)
                    LEFT JOIN agent_revisions ar USING (revision_id)
                    LEFT JOIN agent_families af USING (family_id)
                    WHERE g.session_id IN ({placeholders}) AND NOT rs.is_frozen
                ) WHERE position = 1
                """,
                session_ids,
            )
        }
    rows_html = []
    for session in sessions:
        session_id = session["session_id"]
        stats = rollups.get(session_id, {})
        leader = leaders.get(session_id)
        pairings = int(stats.get("pairings") or 0)
        completed = int(stats.get("completed_pairings") or 0)
        leader_html = "—" if not leader else f'{_escape(leader["display_name"])}<br><small>{_number(leader["conservative"], 1)}</small>'
        rows_html.append(
            f'<tr><td><a href="{_escape(_id_path("sessions", session_id))}"><b>{_escape(session["name"])}</b></a><br><small>{_escape(session_id)}</small></td>'
            f'<td>{_status(session["status"])}</td><td>{int(stats.get("generations") or 0):,}</td>'
            f'<td>{_progress(completed, pairings)}</td><td>{int(stats.get("episodes") or 0):,}</td>'
            f'<td>{int(stats.get("failed_episodes") or 0):,}</td><td>{leader_html}</td><td>{_escape(session["created_at"])}</td></tr>'
        )
    table = (
        '<div class="panel"><table><thead><tr><th>Session</th><th>Status</th><th>Generations</th><th>Pairings</th><th>Episodes</th><th>Failures</th><th>Latest leader</th><th>Created</th></tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table>{_pager("/", page, has_next)}</div>'
        if sessions
        else '<div class="panel empty">No league sessions have been recorded yet.</div>'
    )
    return _layout("Sessions", f'<h1>League sessions</h1><p class="muted">Every persistent training league in this catalog.</p>{table}')


def _session_page(connection: Any, session_id: str) -> str:
    session = _row(
        connection,
        "SELECT session_id, name, status, seed, config_json, created_at FROM training_sessions WHERE session_id = ?",
        [session_id],
    )
    if session is None:
        raise DashboardNotFound(f"session not found: {session_id}")
    generations = _rows(
        connection,
        """
        WITH member_counts AS (
            SELECT generation_id, count(*) AS members
            FROM generation_members GROUP BY generation_id
        ), pairing_counts AS (
            SELECT lp.generation_id, count(DISTINCT lp.pairing_id) AS pairings,
                   count(DISTINCT CASE WHEN lp.status = 'COMPLETE' THEN lp.pairing_id END) AS completed_pairings,
                   count(DISTINCT pe.match_id) AS episodes,
                   count(DISTINCT CASE WHEN e.failed THEN e.match_id END) AS failed_episodes
            FROM league_pairings lp
            LEFT JOIN pairing_episodes pe USING (pairing_id)
            LEFT JOIN episodes e USING (match_id)
            GROUP BY lp.generation_id
        )
        SELECT g.generation_id, g.generation_index, g.status, g.created_at, g.completed_at,
               coalesce(mc.members, 0) AS members, coalesce(pc.pairings, 0) AS pairings,
               coalesce(pc.completed_pairings, 0) AS completed_pairings,
               coalesce(pc.episodes, 0) AS episodes,
               coalesce(pc.failed_episodes, 0) AS failed_episodes
        FROM generations g
        LEFT JOIN member_counts mc USING (generation_id)
        LEFT JOIN pairing_counts pc USING (generation_id)
        WHERE g.session_id = ?
        ORDER BY g.generation_index DESC
        """,
        [session_id],
    )
    trend = _rows(
        connection,
        """
        SELECT g.generation_index, max(rs.rating - 2 * rs.deviation) AS champion,
               median(rs.rating - 2 * rs.deviation) AS median_rating
        FROM generations g JOIN rating_snapshots rs USING (generation_id)
        WHERE g.session_id = ? AND NOT rs.is_frozen
        GROUP BY g.generation_index ORDER BY g.generation_index
        """,
        [session_id],
    )
    evaluation = _row(
        connection,
        """
        SELECT count(*) AS evaluations,
               count(*) FILTER (WHERE ee.status = 'ERROR') AS errors,
               count(*) FILTER (
                   WHERE ee.status = 'COMPLETE'
                     AND coalesce(
                           CAST(json_extract(ee.result_json, '$.should_reevaluate') AS BOOLEAN),
                           false
                         )
               ) AS reevaluation_requests
        FROM episode_evaluations ee
        JOIN pairing_episodes pe USING (match_id)
        JOIN league_pairings lp USING (pairing_id)
        JOIN generations g USING (generation_id) WHERE g.session_id = ?
        """,
        [session_id],
    ) or {}
    total_pairings = sum(int(row["pairings"] or 0) for row in generations)
    completed_pairings = sum(int(row["completed_pairings"] or 0) for row in generations)
    episodes = sum(int(row["episodes"] or 0) for row in generations)
    failures = sum(int(row["failed_episodes"] or 0) for row in generations)
    generation_rows = "".join(
        f'<tr><td><a href="{_escape(_id_path("generations", row["generation_id"]))}"><b>Generation {int(row["generation_index"])}</b></a><br><small>{_escape(row["generation_id"])}</small></td>'
        f'<td>{_status(row["status"])}</td><td>{int(row["members"] or 0):,}</td>'
        f'<td>{_progress(int(row["completed_pairings"] or 0), int(row["pairings"] or 0))}</td>'
        f'<td>{int(row["episodes"] or 0):,}</td><td>{int(row["failed_episodes"] or 0):,}</td>'
        f'<td>{_escape(row["completed_at"] or row["created_at"])}</td></tr>'
        for row in generations
    )
    config = json.dumps(json.loads(str(session["config_json"])), indent=2, sort_keys=True)
    body = f'''<p><a href="/">← Sessions</a></p><h1>{_escape(session["name"])}</h1>
<p class="muted">{_escape(session_id)} · seed {_escape(session["seed"])} · created {_escape(session["created_at"])} · {_status(session["status"])}</p>
{_cards([("Generations", f"{len(generations):,}"), ("Pairings", f"{completed_pairings:,} / {total_pairings:,}"),
          ("Episodes", f"{episodes:,}"), ("Episode failures", f"{failures:,}"),
          ("Hook errors", f"{int(evaluation.get('errors') or 0):,} / {int(evaluation.get('evaluations') or 0):,}"),
          ("Re-evaluation requests", f"{int(evaluation.get('reevaluation_requests') or 0):,}")])}
<h2>Rating trend</h2><div class="panel">{_chart(trend)}</div><h2>Generations</h2>
<div class="panel"><table><thead><tr><th>Generation</th><th>Status</th><th>Members</th><th>Pairings</th><th>Episodes</th><th>Failures</th><th>Updated</th></tr></thead><tbody>{generation_rows}</tbody></table></div>
<details class="panel"><summary>Session configuration</summary><pre>{_escape(config)}</pre></details>'''
    return _layout(str(session["name"]), body)


def _generation_page(connection: Any, generation_id: str, page: int, status_filter: str, kind_filter: str) -> str:
    generation = _row(
        connection,
        """
        SELECT g.generation_id, g.generation_index, g.status, g.created_at, g.completed_at,
               ts.session_id, ts.name AS session_name
        FROM generations g JOIN training_sessions ts USING (session_id) WHERE g.generation_id = ?
        """,
        [generation_id],
    )
    if generation is None:
        raise DashboardNotFound(f"generation not found: {generation_id}")
    leaderboard = _rows(
        connection,
        """
        SELECT rs.revision_id, coalesce(af.name, ar.family_id, rs.revision_id) AS display_name,
               ar.strategy_id, gm.role, rs.rating, rs.deviation,
               rs.rating - 2 * rs.deviation AS conservative, rs.games AS rating_games,
               gs.games, gs.wins, gs.draws, gs.losses, gs.mean_margin, gs.median_margin,
               gs.margin_ci_low, gs.margin_ci_high, gs.opponent_diversity
        FROM rating_snapshots rs
        LEFT JOIN agent_revisions ar USING (revision_id)
        LEFT JOIN agent_families af USING (family_id)
        LEFT JOIN generation_members gm USING (generation_id, revision_id)
        LEFT JOIN generation_stats gs USING (generation_id, revision_id)
        WHERE rs.generation_id = ? AND NOT rs.is_frozen
        ORDER BY conservative DESC, gs.mean_margin DESC NULLS LAST, rs.revision_id
        """,
        [generation_id],
    )
    summary = _row(
        connection,
        """
        SELECT count(*) AS pairings,
               count(*) FILTER (WHERE status = 'COMPLETE') AS completed,
               count(*) FILTER (WHERE status = 'FAILED') AS failed
        FROM league_pairings WHERE generation_id = ?
        """,
        [generation_id],
    ) or {}
    episode_summary = _row(
        connection,
        """
        SELECT count(DISTINCT pe.match_id) AS episodes,
               count(DISTINCT CASE WHEN e.failed THEN e.match_id END) AS failed_episodes
        FROM league_pairings lp LEFT JOIN pairing_episodes pe USING (pairing_id)
        LEFT JOIN episodes e USING (match_id) WHERE lp.generation_id = ?
        """,
        [generation_id],
    ) or {}
    conditions = ["lp.generation_id = ?"]
    parameters: list[Any] = [generation_id]
    if status_filter != "all":
        conditions.append("lp.status = ?")
        parameters.append(status_filter)
    if kind_filter != "all":
        conditions.append("lp.pairing_kind = ?")
        parameters.append(kind_filter)
    offset = (page - 1) * PAGE_SIZE
    pairings = _rows(
        connection,
        f"""
        SELECT lp.pairing_id, lp.round_index, lp.pairing_kind, lp.seed, lp.revision_a,
               lp.revision_b, lp.status, pr.paired_margin, pr.outcome_a, pr.failed,
               pr.episodes_completed,
               coalesce(afa.name, ara.family_id, lp.revision_a) AS agent_a_name,
               ara.strategy_id AS agent_a_strategy_id, ara.checkpoint_id AS agent_a_checkpoint_id,
               coalesce(afb.name, arb.family_id, lp.revision_b) AS agent_b_name,
               arb.strategy_id AS agent_b_strategy_id, arb.checkpoint_id AS agent_b_checkpoint_id
        FROM league_pairings lp LEFT JOIN pairing_results pr USING (pairing_id)
        LEFT JOIN agent_revisions ara ON ara.revision_id = lp.revision_a
        LEFT JOIN agent_families afa ON afa.family_id = ara.family_id
        LEFT JOIN agent_revisions arb ON arb.revision_id = lp.revision_b
        LEFT JOIN agent_families afb ON afb.family_id = arb.family_id
        WHERE {' AND '.join(conditions)} ORDER BY lp.round_index, lp.pairing_id LIMIT ? OFFSET ?
        """,
        [*parameters, PAGE_SIZE + 1, offset],
    )
    has_next = len(pairings) > PAGE_SIZE
    pairings = pairings[:PAGE_SIZE]
    episodes_by_pairing: dict[str, list[dict[str, Any]]] = {}
    if pairings:
        ids = [row["pairing_id"] for row in pairings]
        placeholders = ",".join("?" for _ in ids)
        for episode in _rows(
            connection,
            f"""
            SELECT pe.pairing_id, pe.orientation, pe.match_id, e.failed,
                   ep0.final_money AS seat0_money, ep1.final_money AS seat1_money
            FROM pairing_episodes pe JOIN episodes e USING (match_id)
            LEFT JOIN episode_players ep0 ON ep0.match_id = pe.match_id AND ep0.seat = 0
            LEFT JOIN episode_players ep1 ON ep1.match_id = pe.match_id AND ep1.seat = 1
            WHERE pe.pairing_id IN ({placeholders}) ORDER BY pe.pairing_id, pe.orientation
            """,
            ids,
        ):
            episodes_by_pairing.setdefault(str(episode["pairing_id"]), []).append(episode)
    status_values = [row[0] for row in connection.execute("SELECT DISTINCT status FROM league_pairings WHERE generation_id = ? ORDER BY status", [generation_id]).fetchall()]
    kind_values = [row[0] for row in connection.execute("SELECT DISTINCT pairing_kind FROM league_pairings WHERE generation_id = ? ORDER BY pairing_kind", [generation_id]).fetchall()]
    leaderboard_rows = "".join(
        f'<tr><td>{index}</td><td><b>{_escape(row["display_name"])}</b><br><small>{_escape(row["revision_id"])}</small></td>'
        f'<td>{_escape(row["role"] or "—")}</td><td>{_number(row["conservative"], 1)}</td><td>{_number(row["rating"], 1)} ± {_number(row["deviation"], 1)}</td>'
        f'<td>{int(row["wins"] or 0)}–{int(row["draws"] or 0)}–{int(row["losses"] or 0)}</td>'
        f'<td>{_percent((row["wins"] + 0.5 * row["draws"]) / row["games"] if row["games"] else None)}</td>'
        f'<td>{_number(row["mean_margin"], 1)}<br><small>CI {_number(row["margin_ci_low"], 1)} to {_number(row["margin_ci_high"], 1)}</small></td>'
        f'<td>{int(row["opponent_diversity"] or 0)}</td></tr>'
        for index, row in enumerate(leaderboard, 1)
    )
    pairing_rows = []
    for row in pairings:
        links = []
        for episode in episodes_by_pairing.get(str(row["pairing_id"]), []):
            links.append(
                f'<a href="{_escape(_id_path("episodes", episode["match_id"]))}">seat order {int(episode["orientation"])}: '
                f'{_number(episode["seat0_money"])}–{_number(episode["seat1_money"])}</a>'
                + (" ⚠" if episode["failed"] else "")
            )
        pairing_rows.append(
            f'<tr><td>{int(row["round_index"])}</td><td><small>{_escape(row["pairing_id"])}</small></td><td>{_escape(row["pairing_kind"])}</td>'
            f'<td>{_agent_identity_html(row["agent_a_name"], row["agent_a_strategy_id"], row["revision_a"], row["agent_a_checkpoint_id"], label="Agent A")}'
            f'<div class="versus">vs</div>{_agent_identity_html(row["agent_b_name"], row["agent_b_strategy_id"], row["revision_b"], row["agent_b_checkpoint_id"], label="Agent B")}</td><td>{_status(row["status"])}</td>'
            f'<td>{_number(row["paired_margin"], 1)}</td><td>{"<br>".join(links) or "—"}</td></tr>'
        )
    path = _id_path("generations", generation_id)
    status_options = '<option value="all">All</option>' + "".join(f'<option value="{_escape(value)}" {"selected" if value == status_filter else ""}>{_escape(value)}</option>' for value in status_values)
    kind_options = '<option value="all">All</option>' + "".join(f'<option value="{_escape(value)}" {"selected" if value == kind_filter else ""}>{_escape(value)}</option>' for value in kind_values)
    total, completed = int(summary.get("pairings") or 0), int(summary.get("completed") or 0)
    body = f'''<p><a href="{_escape(_id_path("sessions", generation["session_id"]))}">← {_escape(generation["session_name"])}</a></p>
<h1>Generation {int(generation["generation_index"])}</h1><p class="muted">{_escape(generation_id)} · {_status(generation["status"])}</p>
{_cards([("Pairings", f"{completed:,} / {total:,}"), ("Pairing failures", f"{int(summary.get('failed') or 0):,}"),
          ("Episodes", f"{int(episode_summary.get('episodes') or 0):,}"), ("Episode failures", f"{int(episode_summary.get('failed_episodes') or 0):,}"),
          ("Ranked agents", f"{len(leaderboard):,}")])}
<h2>Leaderboard</h2><div class="panel">{('<table><thead><tr><th>#</th><th>Agent</th><th>Role</th><th>Conservative</th><th>Rating</th><th>W–D–L</th><th>Score</th><th>Mean margin</th><th>Opponents</th></tr></thead><tbody>'+leaderboard_rows+'</tbody></table>') if leaderboard else '<div class="empty">Ratings appear after this generation closes.</div>'}</div>
<h2>Pairings and episodes</h2><div class="panel"><form class="filters" method="get"><label>Status<select name="status">{status_options}</select></label>
<label>Kind<select name="kind">{kind_options}</select></label><button class="button" type="submit">Apply filters</button></form>
{('<table><thead><tr><th>Round</th><th>Pairing</th><th>Kind</th><th>Agents</th><th>Status</th><th>Margin A</th><th>Episodes</th></tr></thead><tbody>'+''.join(pairing_rows)+'</tbody></table>') if pairings else '<div class="empty">No pairings match these filters.</div>'}
{_pager(path, page, has_next, status=status_filter, kind=kind_filter)}</div>'''
    return _layout(f"Generation {generation['generation_index']}", body)


def _episode_page(connection: Any, match_id: str) -> str:
    episode = _row(
        connection,
        """
        SELECT e.match_id, e.seed, e.steps, e.wallclock, e.failed, e.stage,
               pe.orientation, lp.pairing_id, lp.generation_id, lp.revision_a, lp.revision_b,
               coalesce(afa.name, ara.family_id, lp.revision_a) AS agent_a_name,
               ara.strategy_id AS agent_a_strategy_id, ara.checkpoint_id AS agent_a_checkpoint_id,
               coalesce(afb.name, arb.family_id, lp.revision_b) AS agent_b_name,
               arb.strategy_id AS agent_b_strategy_id, arb.checkpoint_id AS agent_b_checkpoint_id
        FROM episodes e JOIN pairing_episodes pe USING (match_id)
        JOIN league_pairings lp USING (pairing_id)
        LEFT JOIN agent_revisions ara ON ara.revision_id = lp.revision_a
        LEFT JOIN agent_families afa ON afa.family_id = ara.family_id
        LEFT JOIN agent_revisions arb ON arb.revision_id = lp.revision_b
        LEFT JOIN agent_families afb ON afb.family_id = arb.family_id
        WHERE e.match_id = ?
        """,
        [match_id],
    )
    if episode is None:
        raise DashboardNotFound(f"league episode not found: {match_id}")
    players = _rows(
        connection,
        """
        SELECT seat, strategy_id, opponent_id, status, failed, reward, final_money,
               opponent_money, money_difference, win, total_unit_actions, productive_actions,
               movement_actions, idle_actions, productive_action_percentage, movement_percentage,
               idle_percentage, plants_lost_inferred, final_shed_items, final_carried_items, metrics_json
        FROM episode_players WHERE match_id = ? ORDER BY seat
        """,
        [match_id],
    )
    evaluations = _rows(
        connection,
        """
        SELECT revision_id, seat, evaluator_path, status, result_json, error_text, created_at
        FROM episode_evaluations WHERE match_id = ? ORDER BY seat, created_at DESC
        """,
        [match_id],
    )
    has_replay = connection.execute("SELECT 1 FROM episode_payloads WHERE match_id = ?", [match_id]).fetchone() is not None
    orientation = int(episode["orientation"])
    agent_a = (episode["agent_a_name"], episode["agent_a_strategy_id"], episode["revision_a"], episode["agent_a_checkpoint_id"])
    agent_b = (episode["agent_b_name"], episode["agent_b_strategy_id"], episode["revision_b"], episode["agent_b_checkpoint_id"])
    identities = [agent_a, agent_b] if orientation == 0 else [agent_b, agent_a]
    player_rows = "".join(
        f'<tr><td>{int(row["seat"])}</td><td>{_agent_identity_html(*identities[int(row["seat"])])}</td>'
        f'<td>{_status(row["status"])}</td><td>{_number(row["final_money"])}</td><td>{_number(row["money_difference"])}</td><td>{_number(row["reward"], 1)}</td>'
        f'<td>{_percent(row["productive_action_percentage"])}</td><td>{_percent(row["movement_percentage"])}</td><td>{_percent(row["idle_percentage"])}</td>'
        f'<td>{int(row["plants_lost_inferred"] or 0)}</td><td>{int(row["final_shed_items"] or 0)} / {int(row["final_carried_items"] or 0)}</td></tr>'
        for row in players
    )
    evaluation_html = "".join(
        f'<details><summary>Seat {int(row["seat"])} · {_escape(row["evaluator_path"])} · {_status(row["status"])}</summary>'
        + (f'<pre>{_escape(json.dumps(json.loads(str(row["result_json"])), indent=2, sort_keys=True))}</pre>' if row["result_json"] is not None else "")
        + (f'<pre>{_escape(row["error_text"])}</pre>' if row["error_text"] else "") + "</details>"
        for row in evaluations
    ) or '<div class="empty">No post-episode evaluations were recorded.</div>'
    replay_link = (
        f'<a class="button" href="{_escape(_id_path("episodes", match_id) + "/replay")}" target="_blank" rel="noopener">Watch replay</a>'
        f'<a class="button" href="{_escape(_id_path("episodes", match_id) + "/replay.json")}">Download replay JSON</a>'
        if has_replay else '<span class="muted">Replay unavailable</span>'
    )
    metrics_html = "".join(
        f'<details><summary>Seat {int(row["seat"])} full metrics</summary><pre>{_escape(json.dumps(json.loads(str(row["metrics_json"])), indent=2, sort_keys=True))}</pre></details>'
        for row in players
    )
    body = f'''<p><a href="{_escape(_id_path("generations", episode["generation_id"]))}">← Generation</a></p><h1>Episode</h1>
<p class="muted">{_escape(match_id)}</p>{_cards([("Seed", _escape(episode["seed"])), ("Steps", _number(episode["steps"])),
    ("Wall clock", f"{_number(episode['wallclock'], 2)}s"), ("Status", _status("FAILED" if episode["failed"] else "COMPLETE"))])}
<div class="actions">{replay_link}</div><h2>Players</h2><div class="panel"><table><thead><tr><th>Seat</th><th>Agent</th><th>Status</th><th>Money</th><th>Margin</th><th>Reward</th><th>Productive</th><th>Movement</th><th>Idle</th><th>Plants lost</th><th>Shed / carried</th></tr></thead><tbody>{player_rows}</tbody></table>{metrics_html}</div>
<h2>Evaluations</h2><div class="panel">{evaluation_html}</div>'''
    return _layout(f"Episode {match_id}", body)


def _load_replay(connection: Any, match_id: str) -> tuple[bytes, dict[str, Any]]:
    row = connection.execute("SELECT replay_gzip FROM episode_payloads WHERE match_id = ?", [match_id]).fetchone()
    if row is None:
        raise DashboardNotFound(f"replay not found: {match_id}")
    try:
        payload = gzip.decompress(bytes(row[0]))
        replay = json.loads(payload)
    except (OSError, ValueError, TypeError) as error:
        raise DashboardUnavailable("The stored replay is invalid.") from error
    if not isinstance(replay, dict) or not isinstance(replay.get("steps"), list):
        raise DashboardUnavailable("The stored replay is invalid.")
    return payload, replay


def _replay_page(connection: Any, match_id: str) -> str:
    episode = _row(
        connection,
        """
        SELECT pe.orientation, lp.revision_a, lp.revision_b,
               coalesce(afa.name, ara.family_id, lp.revision_a) AS agent_a_name,
               ara.strategy_id AS agent_a_strategy_id, ara.checkpoint_id AS agent_a_checkpoint_id,
               coalesce(afb.name, arb.family_id, lp.revision_b) AS agent_b_name,
               arb.strategy_id AS agent_b_strategy_id, arb.checkpoint_id AS agent_b_checkpoint_id
        FROM pairing_episodes pe JOIN league_pairings lp USING (pairing_id)
        LEFT JOIN agent_revisions ara ON ara.revision_id = lp.revision_a
        LEFT JOIN agent_families afa ON afa.family_id = ara.family_id
        LEFT JOIN agent_revisions arb ON arb.revision_id = lp.revision_b
        LEFT JOIN agent_families afb ON afb.family_id = arb.family_id
        WHERE pe.match_id = ?
        """,
        [match_id],
    )
    if episode is None:
        raise DashboardNotFound(f"league episode not found: {match_id}")
    _, replay = _load_replay(connection, match_id)
    agent_a = (episode["agent_a_name"], episode["agent_a_strategy_id"], episode["revision_a"], episode["agent_a_checkpoint_id"])
    agent_b = (episode["agent_b_name"], episode["agent_b_strategy_id"], episode["revision_b"], episode["agent_b_checkpoint_id"])
    identities = [agent_a, agent_b] if int(episode["orientation"]) == 0 else [agent_b, agent_a]
    info = dict(replay.get("info") or {})
    info["Agents"] = [
        {"index": seat, "name": _agent_replay_name(*identity)}
        for seat, identity in enumerate(identities)
    ]
    try:
        from kaggle_environments import make

        environment = make(
            str(replay.get("name") or "kaggriculture"),
            configuration=dict(replay.get("configuration") or {}),
            info=info,
            steps=replay["steps"],
        )
        document = environment.render(mode="html", playing=True, controls=True)
    except Exception as error:
        raise DashboardUnavailable("The stored replay could not be rendered.") from error
    if not isinstance(document, str) or not document:
        raise DashboardUnavailable("The stored replay could not be rendered.")
    return document


class DashboardHandler(BaseHTTPRequestHandler):
    repository: DashboardRepository
    server_version = "KaggricultureDashboard/0.1"

    def do_GET(self) -> None:  # noqa: N802
        self._handle(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle(head=True)

    def do_POST(self) -> None:  # noqa: N802
        self._send_text(HTTPStatus.METHOD_NOT_ALLOWED, "Method not allowed", head=False, headers={"Allow": "GET, HEAD"})

    def _handle(self, *, head: bool) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/healthz":
                with self.repository.connect() as connection:
                    sessions = connection.execute("SELECT count(*) FROM training_sessions").fetchone()[0]
                self._send_json(HTTPStatus.OK, {"status": "ok", "sessions": int(sessions)}, head=head)
                return
            if path.startswith("/episodes/") and path.endswith("/replay.json"):
                match_id = self._identifier(path, "/episodes/", "/replay.json")
                with self.repository.connect() as connection:
                    payload, _ = _load_replay(connection, match_id)
                self._send(HTTPStatus.OK, payload, "application/json; charset=utf-8", head=head, headers={"Content-Disposition": 'attachment; filename="replay.json"'})
                return
            if path.startswith("/episodes/") and path.endswith("/replay"):
                match_id = self._identifier(path, "/episodes/", "/replay")
                with self.repository.connect() as connection:
                    document = _replay_page(connection, match_id)
                replay_csp = (
                    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                    "img-src data:; font-src data:; media-src data:; connect-src 'none'; "
                    "base-uri 'none'; frame-ancestors 'self'"
                )
                self._send(
                    HTTPStatus.OK,
                    document.encode(),
                    "text/html; charset=utf-8",
                    head=head,
                    content_security_policy=replay_csp,
                )
                return
            with self.repository.connect() as connection:
                if path == "/":
                    document = _sessions_page(connection, _parse_page(query))
                elif path.startswith("/sessions/"):
                    document = _session_page(connection, self._identifier(path, "/sessions/"))
                elif path.startswith("/generations/"):
                    status_filter = query.get("status", ["all"])[0]
                    kind_filter = query.get("kind", ["all"])[0]
                    if len(status_filter) > 80 or len(kind_filter) > 80:
                        raise DashboardBadRequest("filter value is too long")
                    document = _generation_page(connection, self._identifier(path, "/generations/"), _parse_page(query), status_filter, kind_filter)
                elif path.startswith("/episodes/"):
                    document = _episode_page(connection, self._identifier(path, "/episodes/"))
                else:
                    raise DashboardNotFound("page not found")
            self._send(HTTPStatus.OK, document.encode(), "text/html; charset=utf-8", head=head)
        except DashboardBadRequest as error:
            self._error(HTTPStatus.BAD_REQUEST, "Bad request", str(error), head=head)
        except DashboardNotFound as error:
            self._error(HTTPStatus.NOT_FOUND, "Not found", str(error), head=head)
        except DashboardUnavailable as error:
            cause = error.__cause__ or error
            print(f"dashboard database unavailable: {cause}", file=sys.stderr)
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "Database unavailable", "A league process may be writing to DuckDB. Pause it and refresh this page.", head=head, headers={"Retry-After": "5"})
        except Exception as error:
            print(f"dashboard request failed: {error!r}", file=sys.stderr)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard error", "The request could not be completed. See the server log for details.", head=head)

    @staticmethod
    def _identifier(path: str, prefix: str, suffix: str = "") -> str:
        encoded = path[len(prefix):]
        if suffix:
            encoded = encoded[:-len(suffix)]
        from urllib.parse import unquote

        if not encoded or "/" in encoded:
            raise DashboardBadRequest("invalid identifier")
        value = unquote(encoded)
        if not value or "\x00" in value:
            raise DashboardBadRequest("invalid identifier")
        return value

    def _error(self, status: HTTPStatus, title: str, message: str, *, head: bool, headers: dict[str, str] | None = None) -> None:
        body = _layout(title, f'<div class="error"><h1>{_escape(title)}</h1><p>{_escape(message)}</p><p><a href="/">Return to sessions</a></p></div>', refresh=False)
        self._send(status, body.encode(), "text/html; charset=utf-8", head=head, headers=headers)

    def _send_json(self, status: HTTPStatus, value: Any, *, head: bool) -> None:
        self._send(status, json.dumps(value, sort_keys=True).encode(), "application/json; charset=utf-8", head=head)

    def _send_text(self, status: HTTPStatus, value: str, *, head: bool, headers: dict[str, str] | None = None) -> None:
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
        self.send_header(
            "Content-Security-Policy",
            content_security_policy
            or "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head:
            self.wfile.write(payload)


def make_server(database: str | Path, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    repository = DashboardRepository(database)
    available = repository.validate()

    class BoundDashboardHandler(DashboardHandler):
        pass

    BoundDashboardHandler.repository = repository
    server = ThreadingHTTPServer((host, port), BoundDashboardHandler)
    server.daemon_threads = True
    server.database_available_at_startup = available  # type: ignore[attr-defined]
    return server


def serve(database: str | Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = make_server(database, host, port)
    actual_host, actual_port = server.server_address[:2]
    if host not in {"127.0.0.1", "::1", "localhost"}:
        print("warning: dashboard has no authentication and is bound beyond loopback", file=sys.stderr)
    if not getattr(server, "database_available_at_startup", True):
        print("database is currently busy; the dashboard will retry on each request", file=sys.stderr)
    print(f"Kaggriculture dashboard: http://{actual_host}:{actual_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
