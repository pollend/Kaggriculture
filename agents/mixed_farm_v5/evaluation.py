"""Post-episode mixed-economy diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agents.mixed_farm_v5.rules import ANIMALS, PRODUCTS, hire_cost

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
    daily_hires: dict[int, int] = {}
    daily_hire_spend: dict[int, int] = {}
    hand_productive = hand_movement = hand_idle = hand_actions = 0
    hand_counts: list[int] = []
    reserve_violations = 0
    reserve_row = reader.fetchone(
        "SELECT params_json FROM agent_revisions WHERE revision_id = %s",
        [context.revision_id],
    )
    reserve = 300.0
    if reserve_row and reserve_row[0]:
        import json
        reserve = float(json.loads(str(reserve_row[0])).get("cash_reserve", reserve))
    max_consecutive_unfed = 0
    previous_farm: dict[str, Any] = {}
    previous_day: int | None = None

    for raw_step in steps:
        state = _state(raw_step, context.seat)
        action = state.get("action") or {}
        observation = state.get("observation") or {}
        day = int(observation.get("day", 0))
        farms = observation.get("farms", []) or []
        farm = farms[context.seat] if context.seat < len(farms) else {}
        hand_counts.append(len(farm.get("hands", []) or []))
        # A replay step records the state *after* its own action was applied, so the
        # pre-action farm — the one the hire price and the cash reserve are judged
        # against — is the previous step's. Reading `farm` here would charge each
        # hire from an index that already includes this turn's own hires.
        same_day = previous_day == day
        hire_day = previous_day if same_day else day
        hires_before = int(previous_farm.get("hires_today", 0)) if same_day else 0
        money_before = float(previous_farm.get("money", farm.get("money", 0)))
        hires_in_action = 0
        for order in action.get("market", []) or []:
            if not order:
                continue
            if order[0] == "BUY_ANIMAL" and len(order) >= 3 and order[1] in purchased_animals:
                purchased_animals[str(order[1])] += int(order[2])
            elif order[0] == "BUY_LAND":
                land_purchases += 1
            elif order[0] == "HIRE":
                cost = hire_cost(hires_before + hires_in_action)
                daily_hires[hire_day] = daily_hires.get(hire_day, 0) + 1
                daily_hire_spend[hire_day] = daily_hire_spend.get(hire_day, 0) + cost
                hires_in_action += 1
            elif order[0] == "SELL" and len(order) >= 3 and order[1] in sold_products:
                sold_products[str(order[1])] += int(order[2])
        hand_units = action.get("hands", []) or []
        for unit_action in hand_units:
            operation = unit_action[0] if unit_action else "PASS"
            hand_actions += 1
            hand_idle += int(operation == "PASS")
            hand_movement += int(operation in {"NORTH", "SOUTH", "EAST", "WEST"})
            hand_productive += int(operation not in {"PASS", "NORTH", "SOUTH", "EAST", "WEST"})
        unit_actions = [action.get("farmer", []), *hand_units]
        for unit_action in unit_actions:
            operation = unit_action[0] if unit_action else "PASS"
            care_actions += int(operation == "CARE")
            feed_actions += int(operation == "FEED")
            fertilizer_collections += int(operation == "COLLECT_FERTILIZER")

        if hires_in_action and money_before - sum(
            hire_cost(hires_before + offset) for offset in range(hires_in_action)
        ) < reserve:
            reserve_violations += 1
        if context.seat < len(farms):
            for row in farms[context.seat].get("tiles", []) or []:
                for tile in row:
                    if isinstance(tile, dict) and tile.get("animal") in ANIMALS:
                        max_consecutive_unfed = max(
                            max_consecutive_unfed, int(tile.get("consecutive_unfed", 0))
                        )
        previous_farm = farm
        previous_day = day

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
        # 3: hire price and reserve check read the pre-action step, not the post-action one.
        "schema_version": 3,
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
        "daily_hires": daily_hires,
        "daily_hire_spend": daily_hire_spend,
        "total_hires": sum(daily_hires.values()),
        "total_hire_spend": sum(daily_hire_spend.values()),
        "max_hands": max(hand_counts, default=0),
        "mean_hands": sum(hand_counts) / max(1, len(hand_counts)),
        "hand_productive_percentage": hand_productive / max(1, hand_actions),
        "hand_movement_percentage": hand_movement / max(1, hand_actions),
        "hand_idle_percentage": hand_idle / max(1, hand_actions),
        "reserve_violations": reserve_violations,
    }
