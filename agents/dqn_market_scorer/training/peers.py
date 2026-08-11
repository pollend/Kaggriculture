"""Read-only checkpoint discovery for market-scorer self-play experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _checkpoint_spec(entry_point: object, artifact_path: object) -> str | None:
    path = Path(str(artifact_path))
    if not path.is_file():
        return None
    return f"{entry_point}?weights={path}"


def active_checkpoint_opponents(
    context: Any,
    reader: Any,
    agent_types: Sequence[str],
) -> list[str]:
    """Active checkpoints belonging to other configured training instances."""

    types = tuple(dict.fromkeys(str(item) for item in agent_types))
    if not types:
        return []
    placeholders = ",".join("%s" for _ in types)
    rows = reader.fetchall(
        f"""
        SELECT af.entry_point, cp.inference_artifact_path
        FROM training_models tm
        JOIN agent_families af USING (family_id)
        JOIN model_training_checkpoints cp
          ON cp.checkpoint_id = tm.active_checkpoint_id
        WHERE tm.agent_type IN ({placeholders}) AND tm.model_id != %s
        ORDER BY tm.agent_type, tm.agent_name, tm.model_id
        """,
        [*types, context.model_id],
    )
    return list(
        dict.fromkeys(
            spec
            for entry_point, path in rows
            if (spec := _checkpoint_spec(entry_point, path)) is not None
        )
    )


def historical_checkpoint_opponents(
    context: Any,
    reader: Any,
    agent_types: Sequence[str],
    limit: int,
) -> list[str]:
    """Recent promoted checkpoints, excluding every instance's active pointer."""

    types = tuple(dict.fromkeys(str(item) for item in agent_types))
    if not types or limit < 1:
        return []
    placeholders = ",".join("%s" for _ in types)
    rows = reader.fetchall(
        f"""
        SELECT af.entry_point, cp.inference_artifact_path
        FROM model_training_checkpoints cp
        JOIN training_models tm USING (model_id)
        JOIN agent_families af USING (family_id)
        WHERE tm.agent_type IN ({placeholders}) AND cp.promoted
          AND cp.checkpoint_id NOT IN (
              SELECT active_checkpoint_id FROM training_models
              WHERE active_checkpoint_id IS NOT NULL
          )
        ORDER BY cp.created_at DESC, cp.checkpoint_id DESC
        """,
        list(types),
    )
    opponents: list[str] = []
    for entry_point, path in rows:
        spec = _checkpoint_spec(entry_point, path)
        if spec is not None and spec not in opponents:
            opponents.append(spec)
            if len(opponents) == limit:
                break
    return opponents
