"""Budgeted, price-impact-aware market controller for the planner."""

from __future__ import annotations

from typing import Any

from agents.receding_horizon_planner.planner import Plan
from agents.receding_horizon_planner.rules import (
    ANIMAL_DATA,
    ANIMAL_ORDER,
    CROP_DATA,
    CROPS,
    PRODUCTS,
    get,
    hire_cost,
    market_price,
    next_land,
)
from agents.receding_horizon_planner.strategy import Strategy

MAX_ORDERS = 10


def _counts(farm: dict, private: dict) -> tuple[dict[str, int], dict[str, int]]:
    crops = {crop: 0 for crop in CROPS}
    animals = {animal: 0 for animal in ANIMAL_ORDER}
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT" and tile.get("crop") in crops:
                crops[str(tile["crop"])] += 1
            if tile.get("animal") in animals:
                animals[str(tile["animal"])] += 1
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    for animal in ANIMAL_ORDER:
        animals[animal] += int(shed.get(animal, 0))
        animals[animal] += sum(int((slot or {}).get(animal, 0)) for slot in inventories)
    return crops, animals


def _execution_revenue(obs: Any, item: str, quantity: int) -> float:
    market = get(obs, "market", {}) or {}
    inventory = float((market.get("inventory", {}) or {}).get(item, 10_000))
    return sum(float(market_price(item, inventory + offset, obs)) for offset in range(quantity))


def market_orders(
    obs: Any,
    farm: dict,
    private: dict,
    plan: Plan,
    strategy: Strategy,
    episode_steps: int = 720,
    turns_per_day: int = 24,
    farm_hand_cost_mult: int = 1,
) -> list[list[Any]]:
    step = int(get(obs, "step", 0))
    hour = int(get(obs, "hour", step % max(1, turns_per_day)))
    terminal = step >= episode_steps - strategy.liquidation_turns
    shed = private.get("shed", {}) or {}
    market = get(obs, "market", {}) or {}
    prices = market.get("prices", {}) or {}
    market_inventory = market.get("inventory", {}) or {}
    crops, animals = _counts(farm, private)

    sale_rows: list[tuple[float, str, int]] = []
    shed_total = sum(int(value) for value in shed.values())
    for item in PRODUCTS:
        quantity = int(shed.get(item, 0))
        reserve = 0
        if item == "WHEAT" and not terminal:
            reserve = plan.wheat_reserve
        elif item == "FERTILIZER" and not terminal:
            reserve = min(3, sum(plan.target_crops.values()))
        quantity = max(0, quantity - reserve)
        if quantity <= 0:
            continue
        params_base = market_price(item, 10_000, obs)
        ratio = float(prices.get(item, params_base)) / max(1.0, float(params_base))
        should_sell = terminal or shed_total >= 80 or ratio >= strategy.sell_price_ratio
        if should_sell:
            revenue = _execution_revenue(obs, item, quantity)
            current_quote = float(prices.get(item, params_base)) * quantity
            impact_loss = current_quote - revenue
            sale_rows.append((revenue + impact_loss * 0.25, item, quantity))

    orders: list[list[Any]] = [
        ["SELL", item, quantity] for _, item, quantity in sorted(sale_rows, key=lambda row: (-row[0], row[1]))
    ][:MAX_ORDERS]
    if terminal or len(orders) >= MAX_ORDERS:
        return orders

    budget = max(0.0, float(farm.get("money", 0.0)) - strategy.cash_reserve)

    # Land is a portfolio decision and gets first claim on investment cash.
    land = next_land(farm)
    if plan.buy_land and land is not None and budget >= land[1] and len(orders) < MAX_ORDERS:
        orders.append(["BUY_LAND"])
        budget -= land[1]

    # Hire early enough for the new hand to perform useful work this day.
    hands = len(farm.get("hands") or [])
    hires_today = int(farm.get("hires_today", hands))
    while hands < plan.target_hands and hour <= 3 and len(orders) < MAX_ORDERS:
        cost = hire_cost(hires_today, farm_hand_cost_mult)
        if budget < cost:
            break
        orders.append(["HIRE"])
        budget -= cost
        hands += 1
        hires_today += 1

    # Reserve feed before purchasing new productive assets.
    carried_wheat = sum(int((slot or {}).get("WHEAT", 0)) for slot in private.get("inventories", []) or [])
    wheat_shortfall = max(0, plan.wheat_reserve - int(shed.get("WHEAT", 0)) - carried_wheat)
    if wheat_shortfall and len(orders) < MAX_ORDERS:
        inventory = float(market_inventory.get("WHEAT", 10_000))
        unit_costs = [market_price("WHEAT", inventory - offset, obs) for offset in range(wheat_shortfall)]
        affordable = 0
        spent = 0.0
        for cost in unit_costs:
            if spent + cost > budget:
                break
            spent += cost
            affordable += 1
        if affordable:
            orders.append(["BUY_PRODUCT", "WHEAT", affordable])
            budget -= spent

    for animal in sorted(
        ANIMAL_ORDER,
        key=lambda name: (
            -float(prices.get(str(ANIMAL_DATA[name]["product"]), ANIMAL_DATA[name]["base"]))
            / float(ANIMAL_DATA[name]["cost"]),
            name,
        ),
    ):
        deficit = max(0, plan.target_animals.get(animal, 0) - animals[animal])
        cost = int(ANIMAL_DATA[animal]["cost"])
        affordable = min(deficit, int(budget // cost))
        if affordable and len(orders) < MAX_ORDERS:
            orders.append(["BUY_ANIMAL", animal, affordable])
            budget -= affordable * cost

    seeds = private.get("seeds", {}) or {}
    for crop in sorted(
        CROPS,
        key=lambda name: (
            -float(prices.get(name, CROP_DATA[name]["base"]))
            * float(CROP_DATA[name]["yield"])
            / float(CROP_DATA[name]["seed"]),
            name,
        ),
    ):
        remaining_days = (episode_steps - step - 1) / max(1, turns_per_day)
        if remaining_days < int(CROP_DATA[crop]["first"]):
            continue
        deficit = max(0, plan.target_crops.get(crop, 0) - crops[crop] - int(seeds.get(crop, 0)))
        cost = int(CROP_DATA[crop]["seed"])
        affordable = min(deficit, int(budget // cost))
        if affordable and len(orders) < MAX_ORDERS:
            orders.append(["BUY_SEED", crop, affordable])
            budget -= affordable * cost

    return orders[:MAX_ORDERS]
