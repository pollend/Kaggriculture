"""Post-episode mixed-economy diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents.mixed_farm_rf_market.rules import ANIMALS, PRODUCTS

if TYPE_CHECKING:  # pragma: no cover
    from kaggriculture_lab.hooks import EpisodeHookContext, EpisodeReader


def _state(step: Any, seat: int) -> dict[str, Any]:
    if not isinstance(step, list) or seat >= len(step) or not isinstance(step[seat], dict):
        return {}
    return step[seat]


def evaluate_episode(context: EpisodeHookContext, reader: EpisodeReader) -> dict[str, Any]:
    summary = reader.summary()
    replay = reader.replay()
    steps = replay.get("steps", []) or []
    purchased_animals = {animal: 0 for animal in ANIMALS}
    sold_products = {product: 0 for product in PRODUCTS}
    land_purchases = care_actions = fertilizer_collections = feed_actions = 0
    max_consecutive_unfed = 0

    for raw_step in steps:
        state = _state(raw_step, context.seat)
        action = state.get("action") or {}
        for order in action.get("market", []) or []:
            if not order:
                continue
            if order[0] == "BUY_ANIMAL" and len(order) >= 3 and order[1] in purchased_animals:
                purchased_animals[str(order[1])] += int(order[2])
            elif order[0] == "BUY_LAND":
                land_purchases += 1
            elif order[0] == "SELL" and len(order) >= 3 and order[1] in sold_products:
                sold_products[str(order[1])] += int(order[2])
        unit_actions = [action.get("farmer", []), *(action.get("hands", []) or [])]
        for unit_action in unit_actions:
            operation = unit_action[0] if unit_action else "PASS"
            care_actions += int(operation == "CARE")
            feed_actions += int(operation == "FEED")
            fertilizer_collections += int(operation == "COLLECT_FERTILIZER")

        observation = state.get("observation") or {}
        farms = observation.get("farms", []) or []
        if context.seat < len(farms):
            for row in farms[context.seat].get("tiles", []) or []:
                for tile in row:
                    if isinstance(tile, dict) and tile.get("animal") in ANIMALS:
                        max_consecutive_unfed = max(
                            max_consecutive_unfed, int(tile.get("consecutive_unfed", 0))
                        )

    final_observation = _state(steps[-1], context.seat).get("observation", {}) if steps else {}
    final_farms = final_observation.get("farms", []) or []
    final_farm = final_farms[context.seat] if context.seat < len(final_farms) else {}
    final_animals = {animal: 0 for animal in ANIMALS}
    for row in final_farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in final_animals:
                final_animals[str(tile["animal"])] += 1

    metrics = summary["metrics"]
    return {
        "schema_version": 1,
        "revision_id": context.revision_id,
        "turns_in_replay": len(steps),
        "final_money": summary["final_money"],
        "money_difference": summary["money_difference"],
        "quadrants_unlocked": len(final_farm.get("unlocked_quadrants", []) or []),
        "land_purchases": land_purchases,
        "purchased_animals": purchased_animals,
        "final_animals": final_animals,
        "feed_actions": feed_actions,
        "care_actions": care_actions,
        "fertilizer_collections": fertilizer_collections,
        "max_consecutive_unfed": max_consecutive_unfed,
        "sold_products": sold_products,
        "productive_action_percentage": metrics.get("productive_action_percentage", 0.0),
        "movement_percentage": metrics.get("movement_percentage", 0.0),
        "plants_lost_inferred": metrics.get("plants_lost_inferred", 0),
        "final_unsold_items": metrics.get("final_shed_items", 0) + metrics.get("final_carried_items", 0),
    }

