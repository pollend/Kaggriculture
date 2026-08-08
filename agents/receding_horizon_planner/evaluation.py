"""Post-episode diagnostics for the receding-horizon planner family."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from kaggriculture_lab.hooks import EpisodeHookContext, EpisodeReader


def evaluate_episode(context: EpisodeHookContext, reader: EpisodeReader) -> dict[str, Any]:
    summary = reader.summary()
    replay = reader.replay()
    metrics = summary["metrics"]
    return {
        "schema_version": 1,
        "revision_id": context.revision_id,
        "turns_in_replay": len(replay.get("steps", [])),
        "final_money": summary["final_money"],
        "money_difference": summary["money_difference"],
        "productive_action_percentage": metrics.get("productive_action_percentage", 0.0),
        "movement_percentage": metrics.get("movement_percentage", 0.0),
        "plants_lost_inferred": metrics.get("plants_lost_inferred", 0),
        "final_unsold_items": metrics.get("final_shed_items", 0) + metrics.get("final_carried_items", 0),
    }
