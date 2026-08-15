"""The embedded kaggle-environments replay at `/episodes/<match_id>/replay`."""

from __future__ import annotations

import gzip
import json
from typing import Any

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard.errors import DashboardNotFound, DashboardUnavailable
from kaggriculture_lab.dashboard.html import _agent_replay_name


def _load_replay(connection: Any, match_id: str) -> tuple[bytes, dict[str, Any]]:
    stored = lq.replay_gzip(connection, match_id)
    if stored is None:
        raise DashboardNotFound(f"replay not found: {match_id}")
    try:
        payload = gzip.decompress(stored)
        replay = json.loads(payload)
    except (OSError, ValueError, TypeError) as error:
        raise DashboardUnavailable("The stored replay is invalid.") from error
    if not isinstance(replay, dict) or not isinstance(replay.get("steps"), list):
        raise DashboardUnavailable("The stored replay is invalid.")
    return payload, replay


def _replay_page(connection: Any, match_id: str) -> str:
    episode = lq.replay_identities(connection, match_id)
    if episode is None:
        raise DashboardNotFound(f"league episode not found: {match_id}")
    _, replay = _load_replay(connection, match_id)
    agent_a = (
        episode["agent_a_name"], episode["agent_a_strategy_id"], episode["revision_a"],
        episode["agent_a_checkpoint_id"], episode["agent_a_params_json"],
    )
    agent_b = (
        episode["agent_b_name"], episode["agent_b_strategy_id"], episode["revision_b"],
        episode["agent_b_checkpoint_id"], episode["agent_b_params_json"],
    )
    identities = [agent_a, agent_b] if int(episode["orientation"]) == 0 else [agent_b, agent_a]
    info = dict(replay.get("info") or {})
    info["Agents"] = [
        {"index": seat, "name": _agent_replay_name(*identity, seat=seat)}
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
