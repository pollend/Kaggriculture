"""Failure triage at `/sessions/<session_id>/failures`.

Every other page filters `NOT failed` and shows a failure *count*. Nothing showed
a failure *reason*, so a session reporting forty-seven failures gave no way to
learn whether that was forty-seven problems or one problem forty-seven times.

`model_updates.error_text` holds a whole formatted traceback, so the grouping key
is its first line -- the exception and message -- with the full text kept one click
away. That is the difference between a wall of stack frames and a short list of
distinct causes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard.errors import DashboardNotFound
from kaggriculture_lab.dashboard.html import (
    _cards,
    _escape,
    _id_path,
    _layout,
)

MAX_CAUSE_LENGTH = 200


def fetch(connection: Any, session_id: str) -> dict[str, Any]:
    session = lq.session(connection, session_id)
    if session is None:
        raise DashboardNotFound(f"session not found: {session_id}")
    return {"session": session, **lq.failure_summary(connection, session_id)}


def cause_of(error_text: Any) -> str:
    """The one line worth grouping on.

    A `traceback.format_exception` string ends with the exception and its message
    and begins with "Traceback (most recent call last):", so the last non-empty
    line is the cause and the first is boilerplate every failure shares.
    """
    if not error_text:
        return "no error text recorded"
    lines = [line.strip() for line in str(error_text).splitlines() if line.strip()]
    if not lines:
        return "no error text recorded"
    cause = lines[-1] if len(lines) > 1 else lines[0]
    if len(cause) > MAX_CAUSE_LENGTH:
        cause = cause[: MAX_CAUSE_LENGTH - 1] + "…"
    return cause


def _grouped(rows: list[dict[str, Any]], label: str) -> str:
    if not rows:
        return f'<div class="panel empty">No {label} failed.</div>'
    counts = Counter(cause_of(row.get("error_text")) for row in rows)
    samples: dict[str, dict[str, Any]] = {}
    for row in rows:
        samples.setdefault(cause_of(row.get("error_text")), row)
    body = "".join(
        f"<tr><td>{count:,}</td><td>{_escape(cause)}</td>"
        f'<td>{_escape(samples[cause].get("model_id") or "—")}</td>'
        f'<td><details><summary>Full text</summary>'
        f'<pre>{_escape(samples[cause].get("error_text") or "—")}</pre></details></td></tr>'
        for cause, count in counts.most_common()
    )
    return (
        f'<div class="panel"><table><thead><tr><th>Count</th><th>Cause</th><th>Model</th>'
        f"<th>First occurrence</th></tr></thead><tbody>{body}</tbody></table></div>"
    )


def render(data: dict[str, Any]) -> str:
    session = data["session"]
    session_id = str(session["session_id"])
    hooks, updates = data["hooks"], data["updates"]
    episodes, pairings = data["episodes"], data["pairings"]

    episode_rows = "".join(
        f'<tr><td><a href="{_escape(_id_path("generations", row["generation_id"]))}">'
        f'{int(row["generation_index"])}</a></td>'
        f'<td>{int(row["round_index"])}</td><td>{_escape(row["pairing_kind"])}</td>'
        f'<td><a href="{_escape(_id_path("episodes", row["match_id"]))}">'
        f'<small>{_escape(row["match_id"])}</small></a></td>'
        f'<td>{int(row["steps"] or 0):,}</td></tr>'
        for row in episodes
    )
    episode_panel = (
        '<div class="panel"><table><thead><tr><th>Generation</th><th>Round</th><th>Kind</th>'
        f'<th>Episode</th><th>Steps</th></tr></thead><tbody>{episode_rows}</tbody></table>'
        + (
            '<p class="muted">Showing the 200 most recent failed episodes.</p>'
            if len(episodes) >= 200
            else ""
        )
        + "</div>"
        if episodes
        else '<div class="panel empty">No episode crashed.</div>'
    )

    pairing_rows = "".join(
        f'<tr><td><a href="{_escape(_id_path("generations", row["generation_id"]))}">'
        f'{int(row["generation_index"])}</a></td>'
        f'<td>{int(row["round_index"])}</td><td>{_escape(row["pairing_kind"])}</td>'
        f'<td><small>{_escape(row["pairing_id"])}</small></td></tr>'
        for row in pairings
    )
    pairing_panel = (
        '<div class="panel"><table><thead><tr><th>Generation</th><th>Round</th><th>Kind</th>'
        f'<th>Pairing</th></tr></thead><tbody>{pairing_rows}</tbody></table></div>'
        if pairings
        else '<div class="panel empty">No pairing failed.</div>'
    )

    distinct = len(
        {cause_of(row.get("error_text")) for row in [*hooks, *updates]}
    )
    cards = _cards([
        ("Failed episodes", f"{len(episodes):,}"),
        ("Failed pairings", f"{len(pairings):,}"),
        ("Hook errors", f"{len(hooks):,}"),
        ("Update errors", f"{len(updates):,}"),
        ("Distinct causes", f"{distinct:,}"),
    ])

    body = f'''<p><a href="{_escape(_id_path("sessions", session_id))}">← {_escape(session["name"])}</a></p>
<h1>Failures</h1><p class="muted">{_escape(session_id)}</p>
{cards}
<h2>Round hook errors</h2>{_grouped(hooks, "round hooks")}
<h2>Model update errors</h2>{_grouped(updates, "model updates")}
<h2>Failed episodes</h2>{episode_panel}
<h2>Failed pairings</h2>{pairing_panel}'''
    return _layout(f"Failures · {session['name']}", body)


def _failures_page(connection: Any, session_id: str) -> str:
    return render(fetch(connection, session_id))
