"""One episode at `/episodes/<match_id>`.

Split `fetch` / `render` for the reason given in `pages.sessions`.
"""

from __future__ import annotations

import json
from typing import Any

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard.errors import DashboardNotFound
from kaggriculture_lab.dashboard.html import (
    _agent_identity_html,
    _cards,
    _escape,
    _id_path,
    _layout,
    _number,
    _percent,
    _status,
)


def fetch(connection: Any, match_id: str) -> dict[str, Any]:
    episode = lq.episode(connection, match_id)
    if episode is None:
        raise DashboardNotFound(f"league episode not found: {match_id}")
    return {
        "match_id": match_id,
        "episode": episode,
        "players": lq.episode_seats(connection, match_id),
        "evaluations": lq.episode_evaluations(connection, match_id),
        "has_replay": lq.has_replay(connection, match_id),
    }


def render(data: dict[str, Any]) -> str:
    match_id = str(data["match_id"])
    episode = data["episode"]
    players, evaluations = data["players"], data["evaluations"]
    has_replay = data["has_replay"]
    orientation = int(episode["orientation"])
    agent_a = (
        episode["agent_a_name"], episode["agent_a_strategy_id"], episode["revision_a"],
        episode["agent_a_checkpoint_id"], episode["agent_a_params_json"],
    )
    agent_b = (
        episode["agent_b_name"], episode["agent_b_strategy_id"], episode["revision_b"],
        episode["agent_b_checkpoint_id"], episode["agent_b_params_json"],
    )
    identities = [agent_a, agent_b] if orientation == 0 else [agent_b, agent_a]
    player_panels = "".join(
        f'<section class="panel player"><h3>Player {int(row["seat"]) + 1} · {_status(row["status"])}</h3>'
        f'{_agent_identity_html(*identities[int(row["seat"])][:4], params_json=identities[int(row["seat"])][4])}'
        f'<dl class="player-stats"><div><dt>Money</dt><dd>{_number(row["final_money"])}</dd></div>'
        f'<div><dt>Margin</dt><dd>{_number(row["money_difference"])}</dd></div>'
        f'<div><dt>Reward</dt><dd>{_number(row["reward"], 1)}</dd></div>'
        f'<div><dt>Productive</dt><dd>{_percent(row["productive_action_percentage"])}</dd></div>'
        f'<div><dt>Movement</dt><dd>{_percent(row["movement_percentage"])}</dd></div>'
        f'<div><dt>Idle</dt><dd>{_percent(row["idle_percentage"])}</dd></div>'
        f'<div><dt>Plants lost</dt><dd>{int(row["plants_lost_inferred"] or 0)}</dd></div>'
        f'<div><dt>Shed / carried</dt><dd>{int(row["final_shed_items"] or 0)} / {int(row["final_carried_items"] or 0)}</dd></div></dl>'
        f'<details><summary>Full metrics</summary><pre>{_escape(json.dumps(json.loads(str(row["metrics_json"])), indent=2, sort_keys=True))}</pre></details></section>'
        for row in players
    )
    evaluation_html = "".join(
        f'<details><summary>Seat {int(row["seat"])} · {_escape(row["evaluator_path"])} · {_status(row["status"])}</summary>'
        + (f'<pre>{_escape(json.dumps(json.loads(str(row["result_json"])), indent=2, sort_keys=True))}</pre>' if row["result_json"] is not None else "")
        + (f'<pre>{_escape(row["error_text"])}</pre>' if row["error_text"] else "") + "</details>"
        for row in evaluations
    ) or '<div class="empty">No post-episode evaluations were recorded.</div>'
    replay_link = (
        f'<a class="button" href="{_escape(_id_path("episodes", match_id) + "/replay.json")}">Download replay JSON</a>'
        if has_replay else '<span class="muted">Replay unavailable</span>'
    )
    replay_html = (
        f'<h2>Replay</h2><iframe class="replay-frame" src="{_escape(_id_path("episodes", match_id) + "/replay")}" '
        'title="Episode replay" loading="eager" scrolling="no"></iframe>'
        if has_replay else ""
    )
    body = f'''<p><a href="{_escape(_id_path("generations", episode["generation_id"]))}">← Generation</a></p><h1>Episode</h1>
<p class="muted">{_escape(match_id)}</p>{_cards([("Seed", _escape(episode["seed"])), ("Steps", _number(episode["steps"])),
    ("Wall clock", f"{_number(episode['wallclock'], 2)}s"), ("Status", _status("FAILED" if episode["failed"] else "COMPLETE"))])}
<div class="actions">{replay_link}</div><h2>Players</h2><div class="players">{player_panels}</div>{replay_html}
<h2>Evaluations</h2><div class="panel">{evaluation_html}</div>'''
    return _layout(f"Episode {match_id}", body)


def _episode_page(connection: Any, match_id: str) -> str:
    return render(fetch(connection, match_id))
