"""Post-episode diagnostics for the adaptive route family."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from kaggriculture_lab.hooks import EpisodeHookContext, EpisodeReader


def evaluate_episode(context: EpisodeHookContext, reader: EpisodeReader) -> dict[str, Any]:
    summary = reader.summary()
    replay = reader.replay()
    metrics = summary["metrics"]
    actions: Counter[str] = Counter()
    phase_actions: Counter[str] = Counter()
    steps = replay.get("steps", []) or []
    for step, states in enumerate(steps):
        if context.seat >= len(states):
            continue
        state = states[context.seat] or {}
        action = state.get("action") if isinstance(state, dict) else getattr(state, "action", None)
        action = action or {}
        phase = "bootstrap" if step < 168 else "expand" if step < 264 else "compound" if step < 660 else "closeout"
        for order in [action.get("farmer"), *(action.get("hands", []) or [])]:
            if order:
                actions[str(order[0])] += 1
                if order[0] != "PASS":
                    phase_actions[phase] += 1
        for order in action.get("market", []) or []:
            if order:
                actions[f"MARKET_{order[0]}"] += 1
                phase_actions[phase] += 1
    return {
        "schema_version": 1,
        "revision_id": context.revision_id,
        "turns_in_replay": len(steps),
        "final_money": summary["final_money"],
        "money_difference": summary["money_difference"],
        "productive_action_percentage": metrics.get("productive_action_percentage", 0.0),
        "movement_percentage": metrics.get("movement_percentage", 0.0),
        "plants_lost_inferred": metrics.get("plants_lost_inferred", 0),
        "final_unsold_items": metrics.get("final_shed_items", 0) + metrics.get("final_carried_items", 0),
        "farm_action_counts": dict(actions),
        "phase_action_counts": dict(phase_actions),
    }

