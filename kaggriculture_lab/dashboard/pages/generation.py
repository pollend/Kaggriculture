"""One generation at `/generations/<generation_id>`, with its performance panels.

Split `fetch` / `render` for the reason given in `pages.sessions`. The panels take
their rows rather than a connection for the same reason: each is one empty-state
branch away from being wrong, and those branches are cheap to test directly.
"""

from __future__ import annotations

import json
from typing import Any

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard import charts
from kaggriculture_lab.dashboard.errors import DashboardBadRequest, DashboardNotFound
from kaggriculture_lab.dashboard.html import (
    PAGE_SIZE,
    _agent_identity_html,
    _cards,
    _duration,
    _escape,
    _href,
    _id_path,
    _layout,
    _number,
    _pager,
    _percent,
    _status,
)


def _margin_histogram(buckets: list[dict[str, Any]]) -> str:
    """Where the margins actually fall, rather than only their mean.

    Six summary numbers cannot tell a steady edge from two blowouts around a pile
    of draws, and those are different findings. The bars diverge around zero
    because `paired_margin` is seat-cancelled, so zero is a real "no edge" line.
    """
    if not buckets:
        return ""
    rows = [
        {
            "label": f'{float(bucket["centre"]):,.0f}',
            "episodes": int(bucket["episodes"] or 0),
        }
        for bucket in buckets
    ]
    chart = charts.bar_chart(
        rows,
        "label",
        "episodes",
        label="Paired margin distribution",
        empty_message="No settled pairings to bin.",
    )
    return (
        f'<div class="panel"><h3>Margin distribution</h3>{chart}'
        '<p class="muted">Settled pairings by seat-cancelled margin. Bars left of the '
        'centre are pairings the first revision lost.</p></div>'
    )


def _outcome_panel(
    margins: dict[str, Any],
    seats: list[dict[str, Any]],
    buckets: list[dict[str, Any]] | None = None,
) -> str:
    """Margin distribution over settled pairings, plus the seat-order bias check."""
    if not int(margins.get("settled_pairings") or 0):
        return '<div class="panel empty">No pairings have been settled yet.</div>'
    # Every pairing is played in both orientations, so a standing seat-to-seat gap is the
    # environment talking, not the agents.
    seat_rows = "".join(
        f'<tr><td>Seat {int(row["seat"])}</td><td>{int(row["episodes"] or 0):,}</td>'
        f'<td>{_number(row["mean_final_money"])}</td><td>{_number(row["mean_margin"], 1)}</td>'
        f'<td>{_percent(row["win_rate"])}</td></tr>'
        for row in seats
    )
    cards = _cards([
        ("Settled pairings", f"{int(margins['settled_pairings']):,}"),
        ("Mean margin", _number(margins["mean_paired_margin"], 1)),
        ("Median margin", _number(margins["median_paired_margin"], 1)),
        ("Margin spread", f"± {_number(margins['paired_margin_stddev'], 1)}"),
        ("Margin range", f"{_number(margins['min_paired_margin'])} to {_number(margins['max_paired_margin'])}"),
        ("Drawn", f"{int(margins.get('drawn') or 0):,}"),
    ])
    seat_table = (
        '<table><thead><tr><th>Seat</th><th>Episodes</th><th>Mean money</th>'
        '<th>Mean margin</th><th>Win rate</th></tr></thead>'
        f"<tbody>{seat_rows}</tbody></table>"
        if seats
        else '<div class="empty">No completed episodes yet.</div>'
    )
    return (
        f'{cards}{_margin_histogram(buckets or [])}<div class="panel"><h3>Seat-order bias</h3>'
        f'<p class="muted">Each pairing is played in both seat orders. A persistent gap here is '
        f'environment asymmetry rather than agent skill.</p>{seat_table}</div>'
    )


def _agent_panel(agents: list[dict[str, Any]]) -> str:
    """Per-agent money and behaviour mix.

    `episode_players.strategy_id` already carries the seat-correct revision id for league
    episodes -- `league.py` swaps the pair by orientation before handing it to `add_match`
    -- so the seat needs no remapping here.
    """
    if not agents:
        return '<div class="panel empty">Per-agent statistics appear once episodes complete.</div>'
    body = "".join(
        f'<tr><td><b>{_escape(row["display_name"])}</b><br><small>{_escape(row["revision_id"])}</small></td>'
        f'<td>{int(row["episodes"] or 0):,}</td><td>{_number(row["mean_money"])}</td>'
        f'<td>{_number(row["median_money"])}</td><td>{_number(row["mean_margin"], 1)}</td>'
        f'<td>{_percent(row["win_rate"])}</td><td>{_number(row["mean_reward"], 1)}</td>'
        f'<td>{_percent(row["productive"])}</td><td>{_percent(row["movement"])}</td>'
        f'<td>{_percent(row["idle"])}</td><td>{_number(row["plants_lost"], 1)}</td>'
        f'<td>{_number(row["unsold"], 1)}</td></tr>'
        for row in agents
    )
    return (
        '<div class="panel"><table><thead><tr><th>Agent</th><th>Episodes</th><th>Mean money</th>'
        '<th>Median money</th><th>Mean margin</th><th>Win rate</th><th>Mean reward</th>'
        '<th>Productive</th><th>Movement</th><th>Idle</th><th>Plants lost</th><th>Unsold</th>'
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _runtime_panel(timing: dict[str, Any], slowest: list[dict[str, Any]]) -> str:
    """Episode wall clock and step counts.

    Per-agent-call latency (`calls`, `time_p99`) is measured by `sim.runner` but never
    persisted by `storage.add_match`, so episode wall clock is all that is available.
    """
    episodes = int(timing.get("episodes") or 0)
    if not episodes:
        return '<div class="panel empty">Timings appear once episodes run.</div>'
    failed = int(timing.get("failed") or 0)
    cards = _cards([
        ("Episode wall clock", _duration(timing["total_wallclock"])),
        ("Mean episode", _duration(timing["mean_wallclock"])),
        ("Median episode", _duration(timing["median_wallclock"])),
        ("95th percentile", _duration(timing["p95_wallclock"])),
        ("Slowest episode", _duration(timing["max_wallclock"])),
        ("Failure rate", _percent(failed / episodes)),
        ("Mean steps", _number(timing["mean_steps"])),
    ])
    slow_rows = "".join(
        f'<tr><td>{int(row["round_index"])}</td>'
        f'<td><a href="{_escape(_id_path("episodes", row["match_id"]))}"><small>{_escape(row["match_id"])}</small></a></td>'
        f'<td>{_duration(row["wallclock"])}</td><td>{_number(row["steps"])}</td>'
        f'<td>{_status("FAILED" if row["failed"] else "COMPLETE")}</td></tr>'
        for row in slowest
    )
    return (
        f'{cards}<div class="panel"><h3>Slowest episodes</h3><table><thead><tr><th>Round</th>'
        '<th>Episode</th><th>Wall clock</th><th>Steps</th><th>Status</th></tr></thead>'
        f"<tbody>{slow_rows}</tbody></table></div>"
    )


def _hook_panel(evaluation: dict[str, Any]) -> str:
    evaluations = int(evaluation.get("evaluations") or 0)
    if not evaluations:
        return '<div class="panel empty">No post-episode evaluations were recorded.</div>'
    errors = int(evaluation.get("errors") or 0)
    return _cards([
        ("Evaluations", f"{evaluations:,}"),
        ("Errors", f"{errors:,}"),
        ("Error rate", _percent(errors / evaluations)),
        ("Re-evaluation requests", f"{int(evaluation.get('reevaluation_requests') or 0):,}"),
    ])


def _performance_section(data: dict[str, Any]) -> str:
    """Everything the league already records about this generation, summarised in one place."""
    return (
        "<h2>Performance</h2>"
        f"<h3>Match outcomes</h3>{_outcome_panel(data['outcomes'], data['seats'], data.get('margin_buckets'))}"
        f"<h3>Agent money and behaviour</h3>{_agent_panel(data['agents'])}"
        f"<h3>Runtime cost</h3>{_runtime_panel(data['runtime'], data['slowest'])}"
        f"<h3>Evaluation hooks</h3>{_hook_panel(data['evaluation'])}"
    )


def fetch(
    connection: Any,
    generation_id: str,
    page: int,
    status_filter: str,
    kind_filter: str,
    sort: str = "round",
    direction: str = "asc",
) -> dict[str, Any]:
    generation = lq.generation(connection, generation_id)
    if generation is None:
        raise DashboardNotFound(f"generation not found: {generation_id}")
    # Validated here rather than inside the query builder so a bad request keeps
    # producing a 400 rather than a 500.
    if sort not in lq.PAIRING_SORT_COLUMNS:
        raise DashboardBadRequest("invalid pairing sort")
    if direction not in lq.SORT_DIRECTIONS:
        raise DashboardBadRequest("invalid pairing sort direction")
    offset = (page - 1) * PAGE_SIZE
    pairings = lq.generation_pairings(
        connection,
        generation_id,
        limit=PAGE_SIZE + 1,
        offset=offset,
        status=None if status_filter == "all" else status_filter,
        kind=None if kind_filter == "all" else kind_filter,
        sort=sort,
        direction=direction,
    )
    has_next = len(pairings) > PAGE_SIZE
    pairings = pairings[:PAGE_SIZE]
    episodes_by_pairing: dict[str, list[dict[str, Any]]] = {}
    for episode in lq.pairing_episode_summaries(
        connection, [row["pairing_id"] for row in pairings]
    ):
        episodes_by_pairing.setdefault(str(episode["pairing_id"]), []).append(episode)
    return {
        "generation": generation,
        "generation_id": generation_id,
        "page": page,
        "has_next": has_next,
        "status_filter": status_filter,
        "kind_filter": kind_filter,
        "sort": sort,
        "direction": direction,
        "leaderboard": lq.generation_leaderboard(connection, generation_id),
        "work": lq.generation_work(connection, generation_id),
        "pairings": pairings,
        "episodes_by_pairing": episodes_by_pairing,
        "filter_values": lq.pairing_filter_values(connection, generation_id),
        "hook_rows": lq.generation_round_hooks(connection, generation_id),
        "outcomes": lq.generation_outcomes(connection, generation_id),
        "seats": lq.seat_balance(connection, generation_id),
        "margin_buckets": lq.margin_histogram(connection, generation_id),
        "agents": lq.agent_episode_stats(connection, generation_id),
        "runtime": lq.episode_runtime(connection, generation_id),
        "slowest": lq.slowest_episodes(connection, generation_id),
        "evaluation": lq.evaluation_rollup(connection, generation_id=generation_id),
    }


def render(data: dict[str, Any]) -> str:
    generation = data["generation"]
    generation_id = str(data["generation_id"])
    page, has_next = data["page"], data["has_next"]
    status_filter, kind_filter = data["status_filter"], data["kind_filter"]
    sort, direction = data["sort"], data["direction"]
    leaderboard, work = data["leaderboard"], data["work"]
    pairings = data["pairings"]
    episodes_by_pairing = data["episodes_by_pairing"]
    status_values = data["filter_values"]["status"]
    kind_values = data["filter_values"]["kind"]
    leaderboard_rows = "".join(
        f'<tr><td>{index}</td><td><b>{_escape(row["display_name"])}</b><br><small>{_escape(row["revision_id"])}</small></td>'
        f'<td>{_escape(row["role"] or "—")}</td><td>{_number(row["conservative"], 1)}</td><td>{_number(row["rating"], 1)} ± {_number(row["deviation"], 1)}</td>'
        f'<td>{int(row["wins"] or 0)}–{int(row["draws"] or 0)}–{int(row["losses"] or 0)}</td>'
        f'<td>{_percent((row["wins"] + 0.5 * row["draws"]) / row["paired_games"] if row["paired_games"] else None)}</td>'
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
            f'<td>{_agent_identity_html(row["agent_a_name"], row["agent_a_strategy_id"], row["revision_a"], row["agent_a_checkpoint_id"], label="Agent A", params_json=row["agent_a_params_json"])}'
            f'<div class="versus">vs</div>{_agent_identity_html(row["agent_b_name"], row["agent_b_strategy_id"], row["revision_b"], row["agent_b_checkpoint_id"], label="Agent B", params_json=row["agent_b_params_json"])}</td><td>{_status(row["status"])}</td>'
            f'<td>{_number(row["paired_margin"], 1)}</td><td>{"<br>".join(links) or "—"}</td></tr>'
        )
    path = _id_path("generations", generation_id)
    def sort_header(label: str, key: str) -> str:
        active = sort == key
        next_direction = "desc" if active and direction == "asc" else "asc"
        arrow = " ↑" if active and direction == "asc" else " ↓" if active else ""
        href = _href(
            path,
            status=status_filter,
            kind=kind_filter,
            sort=key,
            direction=next_direction,
        )
        return (
            f'<th><a data-dashboard-nav href="{_escape(href)}" '
            f'aria-current="{"true" if active else "false"}">{_escape(label)}{arrow}</a></th>'
        )

    status_options = '<option value="all">All</option>' + "".join(f'<option value="{_escape(value)}" {"selected" if value == status_filter else ""}>{_escape(value)}</option>' for value in status_values)
    kind_options = '<option value="all">All</option>' + "".join(f'<option value="{_escape(value)}" {"selected" if value == kind_filter else ""}>{_escape(value)}</option>' for value in kind_values)
    total, completed = int(work.get("pairings") or 0), int(work.get("completed_pairings") or 0)
    session_config = json.loads(str(generation["config_json"]))
    trainer_only = session_config.get("pairing_mode") == "trainer_only"
    if trainer_only:
        completed_rounds = int(work.get("completed_rounds") or 0)
        work_cards = [
            ("Rounds", f'{completed_rounds:,} / {int(session_config.get("rounds_per_generation", 0)):,}'),
            ("Pairings recorded", f"{completed:,}"),
        ]
    else:
        work_cards = [("Pairings", f"{completed:,} / {total:,}")]
    hook_rows = data["hook_rows"]
    hook_rows_html = "".join(
        f'<tr><td>{int(row["round_index"])}</td>'
        f'<td><a href="{_escape(_id_path("models", row["model_id"]))}">'
        f'<code>{_escape(row["model_id"])}</code></a></td>'
        f'<td>{_status(row["status"])}</td><td>{int(row["batches_recorded"] or 0):,}</td>'
        f'<td>{_escape(row["error_text"] or "—")}</td>'
        f'<td>{_escape(row["completed_at"] or row["started_at"])}</td></tr>'
        for row in hook_rows
    )
    rounds_section = (
        '<h2>Training rounds</h2><div class="panel"><table><thead><tr>'
        '<th>Round</th><th>Model</th><th>Status</th><th>Batches</th><th>Error</th><th>Updated</th>'
        f'</tr></thead><tbody>{hook_rows_html}</tbody></table></div>'
        if hook_rows
        else (
            '<h2>Training rounds</h2><div class="panel empty">No trainer hooks have been recorded yet.</div>'
            if trainer_only
            else ""
        )
    )
    body = f'''<p><a href="{_escape(_id_path("sessions", generation["session_id"]))}">← {_escape(generation["session_name"])}</a></p>
<h1>Generation {int(generation["generation_index"])}</h1><p class="muted">{_escape(generation_id)} · {_status(generation["status"])}</p>
{_cards([*work_cards, ("Pairing failures", f"{int(work.get('failed_pairings') or 0):,}"),
          ("Episodes", f"{int(work.get('episodes') or 0):,}"), ("Episode failures", f"{int(work.get('failed_episodes') or 0):,}"),
          ("Ranked agents", f"{len(leaderboard):,}")])}
{rounds_section}
<h2>Leaderboard</h2><div class="panel">{('<table><thead><tr><th>#</th><th>Agent</th><th>Role</th><th>Conservative</th><th>Rating</th><th>W–D–L</th><th>Score</th><th>Mean margin</th><th>Opponents</th></tr></thead><tbody>'+leaderboard_rows+'</tbody></table>') if leaderboard else '<div class="empty">Ratings appear after this generation closes.</div>'}</div>
{_performance_section(data)}
<h2>Pairings and episodes</h2><div class="panel"><form class="filters" method="get"><label>Status<select name="status">{status_options}</select></label>
<label>Kind<select name="kind">{kind_options}</select></label><input type="hidden" name="sort" value="{_escape(sort)}"><input type="hidden" name="direction" value="{_escape(direction)}"><button class="button" type="submit">Apply filters</button></form>
{('<table><thead><tr>'+sort_header('Round', 'round')+sort_header('Pairing', 'pairing')+sort_header('Kind', 'kind')+sort_header('Agents', 'agents')+sort_header('Status', 'status')+sort_header('Margin A', 'margin')+sort_header('Episodes', 'episodes')+'</tr></thead><tbody>'+''.join(pairing_rows)+'</tbody></table>') if pairings else '<div class="empty">No pairings match these filters.</div>'}
{_pager(path, page, has_next, status=status_filter, kind=kind_filter, sort=sort, direction=direction)}</div>'''
    return _layout(f"Generation {generation['generation_index']}", body)


def _generation_page(
    connection: Any,
    generation_id: str,
    page: int,
    status_filter: str,
    kind_filter: str,
    sort: str = "round",
    direction: str = "asc",
) -> str:
    return render(
        fetch(connection, generation_id, page, status_filter, kind_filter, sort, direction)
    )
