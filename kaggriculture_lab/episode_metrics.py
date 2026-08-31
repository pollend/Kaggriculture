"""Kaggriculture-specific episode analysis, lifted out of `sim`.

`sim` is a game-agnostic match runner: it knows about seats, statuses, rewards
and timings, and nothing about crops. Everything that reads a farm -- action
mix, crops planted, plants lost, market orders, money -- lives here and is
injected into `run_match` by the callers that want it.

Used by the league to persist per-episode metrics, which agents' own evaluation
hooks then read back out of PostgreSQL.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def _observation(state, seat: int) -> Any:
    return state[seat].observation


def _obs_get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def final_money(steps: list, seat: int) -> float | None:
    obs = _observation(steps[-1], seat)
    farms = _obs_get(obs, "farms", []) or []
    if seat >= len(farms):
        return None
    return float(farms[seat].get("money", 0.0))


def episode_metrics(steps: list, seat: int) -> dict[str, Any]:
    """Extract exact action counts and conservative state-derived metrics."""

    field_ops: Counter[str] = Counter()
    market_ops: Counter[str] = Counter()
    crops_planted: Counter[str] = Counter()
    crop_peak: Counter[str] = Counter()
    requested_sales: Counter[str] = Counter()
    requested_seeds: Counter[str] = Counter()
    inferred_losses = 0
    total_unit_actions = 0
    sale_value_at_quote = 0.0
    sale_units_at_quote = 0

    previous_plants: dict[tuple[int, int, int, str], dict] = {}
    previous_day: int | None = None
    previous_hires_today = 0
    hires_observed = 0
    hand_actions = 0
    hand_idle = 0

    for state in steps:
        seat_state = state[seat]
        action = seat_state.action if isinstance(seat_state.action, dict) else {}
        hands_actions = list(action.get("hands", []) or [])
        unit_actions = [action.get("farmer", ["PASS"]), *hands_actions]
        for unit_action in unit_actions:
            op = unit_action[0] if isinstance(unit_action, list) and unit_action else "PASS"
            field_ops[str(op)] += 1
            total_unit_actions += 1
        # Hands alone, so an idle crew is not diluted by a busy farmer.
        for unit_action in hands_actions:
            hand_actions += 1
            if not (isinstance(unit_action, list) and unit_action) or unit_action[0] == "PASS":
                hand_idle += 1
        for order in action.get("market", []) or []:
            if not isinstance(order, list) or not order:
                continue
            op = str(order[0])
            market_ops[op] += 1
            if op == "SELL" and len(order) >= 3:
                item, count = str(order[1]), max(0, int(order[2]))
                requested_sales[item] += count
                obs = seat_state.observation
                price = ((_obs_get(obs, "market", {}) or {}).get("prices", {}) or {}).get(item, 0)
                sale_value_at_quote += count * float(price)
                sale_units_at_quote += count
            elif op == "BUY_SEED" and len(order) >= 3:
                requested_seeds[str(order[1])] += max(0, int(order[2]))

        obs = seat_state.observation
        day = int(_obs_get(obs, "day", 0))
        farms = _obs_get(obs, "farms", []) or []
        if seat >= len(farms):
            continue
        farm = farms[seat]
        if previous_day is not None and day != previous_day:
            # `hires_today` resets at the day roll, so the outgoing day's total is the
            # last value seen while that day was still current, not the first value of
            # the new one.
            hires_observed += previous_hires_today
        previous_day = day
        previous_hires_today = int(farm.get("hires_today", 0))

        current_plants: dict[tuple[int, int, int, str], dict] = {}
        for y, row in enumerate(farm.get("tiles", [])):
            for x, tile in enumerate(row):
                if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                    continue
                crop = str(tile.get("crop"))
                key = (x, y, int(tile.get("planted_day", -1)), crop)
                current_plants[key] = tile
                crop_peak[crop] = max(crop_peak[crop], int(tile.get("yield_units", 0)))
                if key not in previous_plants:
                    crops_planted[crop] += 1
        for key in previous_plants.keys() - current_plants.keys():
            x, y, _, _ = key
            tile = farm.get("tiles", [])[y][x]
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                inferred_losses += 1
        previous_plants = current_plants

    hires_observed += previous_hires_today

    final_obs = _observation(steps[-1], seat)
    private = _obs_get(final_obs, "private", {}) or {}
    shed = private.get("shed", {}) or {}
    carried = sum(sum(int(n) for n in inv.values()) for inv in private.get("inventories", []) or [])
    movement = sum(field_ops[op] for op in ("NORTH", "SOUTH", "EAST", "WEST"))
    idle = field_ops["PASS"]
    productive = max(0, total_unit_actions - movement - idle)

    return {
        "final_money": final_money(steps, seat),
        "total_unit_actions": total_unit_actions,
        "productive_actions": productive,
        "movement_actions": movement,
        "idle_actions": idle,
        "productive_action_percentage": productive / total_unit_actions if total_unit_actions else 0.0,
        "movement_percentage": movement / total_unit_actions if total_unit_actions else 0.0,
        "idle_percentage": idle / total_unit_actions if total_unit_actions else 0.0,
        "field_actions": dict(field_ops),
        "market_orders": dict(market_ops),
        "crops_planted_inferred": dict(crops_planted),
        "crop_peak_yield_observed": dict(crop_peak),
        "plants_lost_inferred": inferred_losses,
        "hires_observed_inferred": hires_observed,
        "hand_actions": hand_actions,
        "hand_idle_share": hand_idle / hand_actions if hand_actions else 0.0,
        "requested_sales": dict(requested_sales),
        "requested_seed_purchases": dict(requested_seeds),
        "average_requested_sale_price": sale_value_at_quote / sale_units_at_quote if sale_units_at_quote else 0.0,
        "final_shed_items": sum(int(n) for n in shed.values()),
        "final_carried_items": carried,
    }
