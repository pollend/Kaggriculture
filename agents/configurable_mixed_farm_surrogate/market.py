"""Budgeted market decisions for the mixed-farm portfolio."""

from __future__ import annotations

import math
from typing import Any

from agents.configurable_mixed_farm_surrogate.portfolio import Plan
from agents.configurable_mixed_farm_surrogate.rules import (
    ANIMAL_DATA,
    ANIMALS,
    CROP_DATA,
    CROPS,
    PRODUCTS,
    count_assets,
    get,
    hire_cost,
    unlocked_empty_tiles,
)
from agents.configurable_mixed_farm_surrogate.strategy import Strategy

MAX_ORDERS = 10


def _scaled_crop_targets(plan: Plan, total: int) -> dict[str, int]:
    """Scale the season crop mix into a small deterministic setup tranche."""

    total = max(0, min(total, sum(plan.target_crops.values())))
    denominator = sum(max(0, int(plan.target_crops[crop])) for crop in CROPS)
    if total == 0 or denominator == 0:
        return {crop: 0 for crop in CROPS}
    exact = {
        crop: total * max(0, int(plan.target_crops[crop])) / denominator
        for crop in CROPS
    }
    result = {crop: int(math.floor(exact[crop])) for crop in CROPS}
    remaining = total - sum(result.values())
    order = sorted(
        CROPS,
        key=lambda crop: (-(exact[crop] - result[crop]), -plan.crop_scores[crop], crop),
    )
    for crop in order[:remaining]:
        result[crop] += 1
    return result


def market_orders(
    obs: Any,
    farm: dict,
    private: dict,
    plan: Plan,
    strategy: Strategy,
    *,
    episode_steps: int = 720,
    turns_per_day: int = 24,
    farm_hand_cost_mult: int = 1,
) -> list[list[Any]]:
    step = int(get(obs, "step", 0))
    hour = int(get(obs, "hour", step % max(1, turns_per_day)))
    liquidation = step >= episode_steps - strategy.liquidation_turns
    final_day = step >= episode_steps - turns_per_day
    shed = private.get("shed", {}) or {}
    raw_prices = (get(obs, "market", {}) or {}).get("prices", {}) or {}
    prices = {name: float(raw_prices.get(name, CROP_DATA[name]["base"])) for name in CROPS}
    for data in ANIMAL_DATA.values():
        prices[str(data["product"])] = float(raw_prices.get(str(data["product"]), data["base"]))
    prices["FERTILIZER"] = float(raw_prices.get("FERTILIZER", 100))
    shed_total = sum(int(quantity) for quantity in shed.values())
    crops, animals, _ = count_assets(farm)
    inventories = private.get("inventories", []) or []
    current_animal_total = sum(animals.values())
    owned_animal_total = current_animal_total + sum(
        int(shed.get(animal, 0))
        + sum(int((slot or {}).get(animal, 0)) for slot in inventories)
        for animal in ANIMALS
    )
    remaining_days = max(0, (episode_steps - step - 1 + turns_per_day - 1) // turns_per_day)
    operating_wheat_reserve = owned_animal_total * min(
        strategy.feed_reserve_days, remaining_days
    )

    orders: list[list[Any]] = []
    for item in PRODUCTS:
        quantity = int(shed.get(item, 0))
        reserve = 0
        if not final_day and item == "WHEAT":
            reserve = operating_wheat_reserve
        elif not liquidation and item == "FERTILIZER":
            reserve = plan.fertilizer_reserve
        quantity = max(0, quantity - reserve)
        if quantity <= 0:
            continue
        base = (
            float(CROP_DATA[item]["base"])
            if item in CROP_DATA
            else next(
                (float(data["base"]) for data in ANIMAL_DATA.values() if data["product"] == item),
                100.0,
            )
        )
        ratio = prices[item] / max(1.0, base)
        cash_starved = float(farm.get("money", 0)) < strategy.cash_reserve * 0.8
        if (
            liquidation
            or cash_starved
            or ratio >= strategy.sell_price_ratio
            or shed_total >= 100 * strategy.shed_pressure_fraction
        ):
            orders.append(["SELL", item, quantity])
            if len(orders) >= MAX_ORDERS:
                return orders

    if final_day:
        return orders

    available_cash = max(0.0, float(farm.get("money", 0)))
    budget = max(0.0, available_cash - strategy.cash_reserve)
    carried_wheat = sum(int((slot or {}).get("WHEAT", 0)) for slot in inventories)
    wheat_shortfall = max(
        0, operating_wheat_reserve - int(shed.get("WHEAT", 0)) - carried_wheat
    )
    if wheat_shortfall and len(orders) < MAX_ORDERS:
        unit_cost = prices["WHEAT"]
        affordable = min(wheat_shortfall, int(available_cash // max(1.0, unit_cost)))
        if affordable:
            orders.append(["BUY_PRODUCT", "WHEAT", affordable])
            spent = affordable * unit_cost
            available_cash -= spent
            budget = max(0.0, budget - spent)

    if liquidation:
        return orders[:MAX_ORDERS]

    if plan.buy_land and plan.next_land_price is not None and budget >= plan.next_land_price:
        orders.append(["BUY_LAND"])
        budget -= plan.next_land_price
        available_cash -= plan.next_land_price

    seeds = {crop: int((private.get("seeds", {}) or {}).get(crop, 0)) for crop in CROPS}
    committed_crops = sum(crops.values()) + sum(seeds.values())
    crop_capacity = sum(crops.values()) + len(unlocked_empty_tiles(farm))

    owned_animals = dict(animals)
    for animal in ANIMALS:
        owned_animals[animal] += int(shed.get(animal, 0))
        owned_animals[animal] += sum(int((slot or {}).get(animal, 0)) for slot in inventories)
    owned_animal_total = sum(owned_animals.values())
    hands = len(farm.get("hands", []) or [])

    setup_total = min(
        sum(plan.target_crops.values()),
        crop_capacity,
        max(
            strategy.opening_crop_tranche,
            2 * owned_animal_total + 4,
            sum(crops.values()) + 2 * (1 + hands),
        ),
    )
    setup_targets = _scaled_crop_targets(plan, setup_total)
    minimum_crop_commitment = min(
        strategy.animal_unlock_crop_tiles,
        sum(plan.target_crops.values()),
    )
    prospective_animal_cap = (
        min(
            sum(plan.target_animals.values()),
            owned_animal_total + strategy.opening_animal_batch,
        )
        if setup_total >= minimum_crop_commitment
        else 0
    )
    wheat_floor = min(
        int(plan.target_crops["WHEAT"]),
        math.ceil(max(owned_animal_total, prospective_animal_cap) / 2),
        setup_total,
    )
    if setup_targets["WHEAT"] < wheat_floor:
        needed = wheat_floor - setup_targets["WHEAT"]
        for crop in sorted(
            (crop for crop in CROPS if crop != "WHEAT"),
            key=lambda crop: (plan.crop_scores[crop], crop),
        ):
            moved = min(needed, setup_targets[crop])
            setup_targets[crop] -= moved
            setup_targets["WHEAT"] += moved
            needed -= moved
            if needed == 0:
                break

    def buy_seed_targets(targets: dict[str, int]) -> None:
        nonlocal available_cash, budget, committed_crops
        crops_by_priority = sorted(
            CROPS,
            key=lambda crop: (crop != "WHEAT", -plan.crop_scores[crop], crop),
        )
        for crop in crops_by_priority:
            if len(orders) >= MAX_ORDERS:
                return
            deficit = max(0, int(targets[crop]) - crops[crop] - seeds[crop])
            cost = int(CROP_DATA[crop]["seed"])
            affordable = min(deficit, int(budget // cost))
            if affordable <= 0:
                continue
            orders.append(["BUY_SEED", crop, affordable])
            spent = affordable * cost
            budget -= spent
            available_cash -= spent
            seeds[crop] += affordable
            committed_crops += affordable

    # Establish enough plantable work before adding more livestock.
    buy_seed_targets(setup_targets)

    plantable_seeds = min(sum(seeds.values()), len(unlocked_empty_tiles(farm)))
    unplaced_animals = max(0, owned_animal_total - sum(animals.values()))
    staged_workload = (
        sum(crops.values())
        + 2 * plantable_seeds
        + 3 * sum(animals.values())
        + 4 * unplaced_animals
    )
    prospective_new_animals = (
        min(
            strategy.opening_animal_batch,
            min(sum(plan.target_animals.values()), committed_crops),
        )
        if owned_animal_total == 0
        else 0
    )
    staged_workload += 4 * prospective_new_animals
    staged_target_hands = min(
        strategy.max_hands,
        max(0, math.ceil(staged_workload / strategy.workload_per_hand) - 1),
    )
    target_hands = max(plan.target_hands, staged_target_hands)
    hires_today = int(farm.get("hires_today", hands))
    while (
        hands < target_hands
        and hands < strategy.max_hands
        and hour <= strategy.hire_until_hour
        and len(orders) < MAX_ORDERS
    ):
        cost = hire_cost(hires_today, farm_hand_cost_mult)
        # Daily labor is operational capacity, not a long-lived investment;
        # allow its small Fibonacci cost to use the cash reserve just as feed
        # can, otherwise a stocked farm can become permanently understaffed.
        if available_cash < cost:
            break
        orders.append(["HIRE"])
        budget = max(0.0, budget - cost)
        available_cash -= cost
        hands += 1
        hires_today += 1

    # Tie livestock staging to the selected crop tranche. The previous fixed
    # threshold made legacy strategies targeting fewer than eight crop tiles
    # accumulate cash indefinitely without ever unlocking their animals.
    animal_cap = (
        committed_crops
        if committed_crops >= minimum_crop_commitment
        else 0
    )
    remaining_animal_slots = max(
        0,
        min(sum(plan.target_animals.values()), animal_cap) - owned_animal_total,
    )
    for animal in sorted(ANIMALS, key=lambda name: (-plan.animal_scores[name], name)):
        if remaining_animal_slots <= 0:
            break
        deficit = max(0, plan.target_animals[animal] - owned_animals[animal])
        cost = int(ANIMAL_DATA[animal]["cost"])
        affordable = min(deficit, remaining_animal_slots, int(budget // cost))
        if affordable and len(orders) < MAX_ORDERS:
            orders.append(["BUY_ANIMAL", animal, affordable])
            spent = affordable * cost
            budget -= spent
            available_cash -= spent
            owned_animals[animal] += affordable
            owned_animal_total += affordable
            remaining_animal_slots -= affordable

    return orders[:MAX_ORDERS]
