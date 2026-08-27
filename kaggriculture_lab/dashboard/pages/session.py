"""One session at `/sessions/<session_id>`.

Split `fetch` / `render` for the reason given in `pages.sessions`.
"""

from __future__ import annotations

import json
from typing import Any

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard import charts
from kaggriculture_lab.dashboard.errors import DashboardNotFound
from kaggriculture_lab.dashboard.html import (
    _cards,
    _chart,
    _escape,
    _id_path,
    _layout,
    _number,
    _percent,
    _progress,
    _status,
)


def fetch(connection: Any, session_id: str) -> dict[str, Any]:
    session = lq.session(connection, session_id)
    if session is None:
        raise DashboardNotFound(f"session not found: {session_id}")
    session_config = json.loads(str(session["config_json"]))
    generations = lq.session_generations(connection, session_id)
    trainer_only = session_config.get("pairing_mode") == "trainer_only"
    return {
        "session": session,
        "session_config": session_config,
        "generations": generations,
        "trend": lq.rating_trend(connection, session_id),
        "evaluation": lq.evaluation_rollup(connection, session_id=session_id),
        # One query, not two: the roster carries the batch watermark as well, so
        # the page does not pay a second round trip for the same models.
        "roster": lq.session_learner_roster(connection, session_id),
        "head_to_head": lq.head_to_head_by_family(connection, session_id),
        # Only the trainer-only layout shows a round count, and it is the one
        # number the per-generation rollup cannot be summed to produce: a round
        # spans models, not generations.
        "completed_rounds": (
            lq.completed_round_count(connection, session_id=session_id) if trainer_only else 0
        ),
    }


def _roster_row(row: dict[str, Any]) -> str:
    """One learner slot, as of its newest update.

    `reseed_index` is the slot's generation marker and leads the row: the slot id
    survives every restart, so it alone cannot say which policy is in there now.
    """

    latest = int(row.get("latest_batch_seq") or 0)
    attempted = int(row.get("last_attempted_batch_seq") or 0)
    minimum = int(row.get("min_new_batches") or 0)
    rate = row["trading_rate"]
    money = row["last_money"]
    # The two collapses this family fails into, named rather than left to be
    # inferred from two numbers: idle keeps its stake by never trading, broke
    # traded itself out. Both read as "no activity" and they are not the same.
    if rate is None:
        mode = ""
    elif float(rate) < 0.01:
        mode = " idle" if (money or 0) > 1500 else " broke"
    else:
        mode = ""
    # Two schedules render here. The learning-rate ladder prints its stage and
    # counters; a member pool has no ladder, so it prints the lineage the position
    # is currently carrying and how many members the slot holds.
    if row.get("members"):
        schedule = (
            f'<code>{_escape(row["stage"] or "—")}</code>'
            f'<br><small>{row["members"]} members</small>'
        )
    else:
        schedule = (
            f'{_escape(row["stage"] or "—")}'
            f'<br><small>decay {row["decay_steps"] if row["decay_steps"] is not None else "—"}'
            f' · stall {row["stall_updates"] if row["stall_updates"] is not None else "—"}'
            f' · +{row["updates_since_reseed"] if row["updates_since_reseed"] is not None else "—"}</small>'
        )
    release = (
        f'<small>{_escape(row["active_checkpoint_id"])}</small>'
        f'<br><small class="muted">{int(row["promotions"] or 0)} promoted</small>'
        if row["active_checkpoint_id"]
        else '<span class="muted">none</span>'
    )
    pool = (
        f'{_status("IN POOL")}<br><small>{_number(row["pool_score"], 1)}</small>'
        if row["in_pool"]
        else '<span class="muted">—</span>'
    )
    # The money is a median over however many measurements the slot has taken
    # since its reseed, and two of them is not eleven of them. Printed with the
    # number rather than beside it, because a reader comparing two slots is
    # comparing two different amounts of evidence and nothing else on the row
    # says so.
    #
    # Zero is not "no samples" but the state before the window can run at all --
    # `released_measurement`, the incumbent's single stored draw -- and saying "0
    # samples" for the one reading standing on the least evidence is the wrong way
    # round.
    samples = row.get("reference_samples")
    money_cell = _number(money, 1)
    if samples is not None:
        count = int(samples)
        evidence = "single draw" if count < 2 else f"{count} samples"
        money_cell += f'<br><small class="muted">{evidence}</small>'
    return (
        f'<tr><td><a href="{_escape(_id_path("models", row["model_id"]))}">'
        f'<code>{_escape(str(row["model_id"]).split("@")[-1])}</code></a></td>'
        f'<td><b>#{row["reseed_index"] if row["reseed_index"] is not None else "—"}</b></td>'
        f'<td>{schedule}</td>'
        f'<td>{latest - attempted:,} / {minimum:,}</td>'
        f'<td>{money_cell}</td>'
        f'<td>{_percent(rate)}{mode}</td>'
        f'<td>{release}</td><td>{pool}</td></tr>'
    )


def _head_to_head_section(rows: list[dict[str, Any]], trainer_only: bool = False) -> str:
    """Which family each family actually loses to.

    Rendered per focal family as a diverging bar rather than as a full
    family-by-family heatmap: the matrix is antisymmetric, so the heatmap would
    say everything twice, and a bar has a zero line a reader can trust.

    Under `trainer_only` the same bars mean something else entirely and are
    relabelled rather than hidden. There is no league opponent in that mode: every
    pairing is one learner against mirrors of its own checkpoints, drawn from a pool
    `_self_play_pool` deliberately does not filter on `promoted`, because release is
    gated and the behaviour distribution is not. So the margin is self-referential,
    and it is largest when the slot is doing worst -- a slot whose gate rejects
    everything mirrors nothing but its own rejects. `cashherd_02` posted +18,148 at a
    0.96 win rate against three checkpoints its own gate had refused. Left as a
    quality ranking that reads as the best slot in the session; it was the most
    stuck. The number to rank on is the benchmark money in Models below.
    """
    if not rows:
        return ""
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["me"]), []).append(row)
    panels = []
    for family, opponents in sorted(by_family.items()):
        # A family only ever paired against itself has no comparison to draw.
        if len(opponents) < 2 and all(row["opponent"] == family for row in opponents):
            continue
        chart = charts.bar_chart(
            [
                {
                    "opponent": f'{row["opponent_name"]} ({int(row["pairings"] or 0)})',
                    "margin": float(row["mean_margin"] or 0.0),
                }
                for row in opponents
            ],
            "opponent",
            "margin",
            diverging=True,
            horizontal=True,
            label=f"{family} mean margin by opponent family",
        )
        panels.append(f'<div class="panel"><h3>{_escape(family)}</h3>{chart}</div>')
    if not panels:
        return ""
    if trainer_only:
        return (
            "<h2>Rollout matchups</h2>"
            '<p class="muted">Mean seat-cancelled margin against each mirror family, over '
            "complete generations; the count in brackets is pairings played. "
            "<b>Not a quality ranking.</b> This session pairs each learner against "
            "mirrors of its own checkpoints, promoted or not, so the margin is measured "
            "against the slot itself: a slot whose gate rejects everything mirrors its "
            "own rejects and posts a large positive margin for being stuck. Rank on the "
            "benchmark money under Models instead.</p>"
            f'{"".join(panels)}'
        )
    return (
        "<h2>Head to head</h2>"
        '<p class="muted">Mean seat-cancelled margin against each opponent family, over '
        "complete generations. Bars left of centre are families this one loses to; the "
        "count in brackets is pairings played.</p>"
        f'{"".join(panels)}'
    )


def render(data: dict[str, Any]) -> str:
    session = data["session"]
    session_id = str(session["session_id"])
    generations = data["generations"]
    trend, evaluation = data["trend"], data["evaluation"]
    total_pairings = sum(int(row["pairings"] or 0) for row in generations)
    completed_pairings = sum(int(row["completed_pairings"] or 0) for row in generations)
    episodes = sum(int(row["episodes"] or 0) for row in generations)
    failures = sum(int(row["failed_episodes"] or 0) for row in generations)
    session_config = data["session_config"]
    trainer_only = session_config.get("pairing_mode") == "trainer_only"
    completed_rounds = int(data["completed_rounds"]) if trainer_only else 0
    planned_rounds = (
        len(generations) * int(session_config.get("rounds_per_generation", 0))
        if trainer_only
        else 0
    )
    generation_rows = "".join(
        f'<tr><td><a href="{_escape(_id_path("generations", row["generation_id"]))}"><b>Generation {int(row["generation_index"])}</b></a><br><small>{_escape(row["generation_id"])}</small></td>'
        f'<td>{_status(row["status"])}</td><td>{int(row["members"] or 0):,}</td>'
        f'<td>{f"{int(row["completed_rounds"] or 0):,} / {int(session_config.get("rounds_per_generation", 0)):,} rounds<br><small>{int(row["completed_pairings"] or 0):,} pairings recorded</small>" if trainer_only else _progress(int(row["completed_pairings"] or 0), int(row["pairings"] or 0))}</td>'
        f'<td>{int(row["episodes"] or 0):,}</td><td>{int(row["failed_episodes"] or 0):,}</td>'
        f'<td>{_escape(row["completed_at"] or row["created_at"])}</td></tr>'
        for row in generations
    )
    head_to_head = _head_to_head_section(data["head_to_head"], trainer_only)
    model_rows = "".join(_roster_row(row) for row in data["roster"])
    # Without this the session page carries no training information at all: the
    # models that produced every checkpoint in it were reachable only over MCP.
    #
    # Ordered so the questions people actually arrive with are answerable left to
    # right: which policy is in this slot, how far through its schedule, what did
    # it last measure, and has anything of it reached the shared pool.
    models_section = (
        '<h2>Models</h2><div class="panel"><table><thead><tr>'
        '<th>Slot</th><th>Gen</th><th>Schedule</th><th>New / threshold</th>'
        '<th>Last money</th><th>Trading</th><th>Release</th><th>In pool</th>'
        f'</tr></thead><tbody>{model_rows}</tbody></table>'
        '<p class="muted">Gen is the slot\'s reseed index: #0 is the policy it '
        'started with, and it steps every time the slot is restarted from the '
        'mature pool. Schedule is stage, then decay steps and stalls against the '
        'thresholds that end a slot.</p></div>'
        if data["roster"]
        else ""
    )
    config = json.dumps(session_config, indent=2, sort_keys=True)
    work_cards = (
        [("Rounds", f"{completed_rounds:,} / {planned_rounds:,}"),
         ("Pairings recorded", f"{completed_pairings:,}")]
        if trainer_only else [("Pairings", f"{completed_pairings:,} / {total_pairings:,}")]
    )
    body = f'''<p><a href="/">← Sessions</a></p><h1>{_escape(session["name"])}</h1>
<p class="muted">League {_escape(session["league"])} · {_escape(session_id)} · seed {_escape(session["seed"])} · created {_escape(session["created_at"])} · {_status(session["status"])}</p>
{_cards([("Generations", f"{len(generations):,}"), *work_cards,
          ("Episodes", f"{episodes:,}"), ("Episode failures", f"{failures:,}"),
          ("Hook errors", f"{int(evaluation.get('errors') or 0):,} / {int(evaluation.get('evaluations') or 0):,}"),
          ("Re-evaluation requests", f"{int(evaluation.get('reevaluation_requests') or 0):,}")])}
<p class="actions"><a class="button" href="{_escape(_id_path("sessions", session_id) + "/failures")}">Failure triage</a></p>
<h2>Rating trend</h2><div class="panel">{_chart(trend)}</div>{head_to_head}{models_section}<h2>Generations</h2>
<div class="panel"><table><thead><tr><th>Generation</th><th>Status</th><th>Members</th><th>Progress</th><th>Episodes</th><th>Failures</th><th>Updated</th></tr></thead><tbody>{generation_rows}</tbody></table></div>
<details class="panel"><summary>Session configuration</summary><pre>{_escape(config)}</pre></details>'''
    return _layout(str(session["name"]), body)


def _session_page(connection: Any, session_id: str) -> str:
    return render(fetch(connection, session_id))
