"""Turn learned market targets into safe, legal Kaggriculture orders."""

from __future__ import annotations

from typing import Any

from agents.mixed_farm_rf_market.policy import PurchasePlan
from agents.mixed_farm_rf_market.portfolio import Plan
from agents.mixed_farm_rf_market.rules import (
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
from agents.mixed_farm_rf_market.strategy import Strategy

MAX_ORDERS = 10


def _carried(private: dict, item: str) -> int:
    return sum(int((slot or {}).get(item, 0)) for slot in private.get("inventories", []) or [])


def learned_market_orders(
    obs: Any,
    farm: dict,
    private: dict,
    base_plan: Plan,
    purchase: PurchasePlan,
    reserves: dict[str, int],
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
    market = get(obs, "market", {}) or {}
    raw_prices = market.get("prices", {}) or {}
    prices = {name: float(raw_prices.get(name, CROP_DATA[name]["base"])) for name in CROPS}
    for data in ANIMAL_DATA.values():
        prices[str(data["product"])] = float(raw_prices.get(str(data["product"]), data["base"]))
    prices["FERTILIZER"] = float(raw_prices.get("FERTILIZER", 100))
    crops, animals, _ = count_assets(farm)
    owned_animals = dict(animals)
    for animal in ANIMALS:
        owned_animals[animal] += int(shed.get(animal, 0)) + _carried(private, animal)
    survival_wheat = sum(owned_animals.values())

    orders: list[list[Any]] = []
    money = max(0.0, float(farm.get("money", 0)))
    spendable = max(0.0, money - strategy.cash_reserve)

    # Survival feed owns the first slot and may use reserve cash.
    wheat_target = max(survival_wheat, purchase.wheat_reserve)
    shortfall = max(0, wheat_target - int(shed.get("WHEAT", 0)) - _carried(private, "WHEAT"))
    if shortfall and not liquidation:
        quantity = min(shortfall, int(money // max(1.0, prices["WHEAT"])))
        if quantity:
            orders.append(["BUY_PRODUCT", "WHEAT", quantity])
            cost = quantity * prices["WHEAT"]
            money -= cost
            spendable = max(0.0, spendable - cost)

    # Product reserves are learned; survival and terminal rules are not.
    for item in PRODUCTS:
        available = int(shed.get(item, 0))
        reserve = 0 if final_day or liquidation else max(0, int(reserves.get(item, available)))
        if item == "WHEAT" and not final_day and not liquidation:
            reserve = max(reserve, survival_wheat)
        quantity = min(max(0, available - reserve), strategy.sell_batch_size)
        if quantity:
            orders.append(["SELL", item, quantity])
            if len(orders) == MAX_ORDERS:
                return orders

    if final_day or liquidation:
        return orders[:MAX_ORDERS]

    current_land = max(1, len(farm.get("unlocked_quadrants", []) or ["NW"]))
    if (
        purchase.target_land > current_land
        and current_land <= 3
        and current_land < strategy.max_quadrants
        and base_plan.next_land_price is not None
        and spendable >= base_plan.next_land_price
        and len(orders) < MAX_ORDERS
    ):
        orders.append(["BUY_LAND"])
        spendable -= base_plan.next_land_price
        money -= base_plan.next_land_price

    target_crops = purchase.crop_targets()
    target_animals = purchase.animal_targets()
    seeds = {name: int((private.get("seeds", {}) or {}).get(name, 0)) for name in CROPS}
    capacity = sum(crops.values()) + len(unlocked_empty_tiles(farm))
    committed = sum(crops.values()) + sum(seeds.values())
    remaining_days = max(0.0, (episode_steps - step - 1) / max(1, turns_per_day))
    for crop in sorted(CROPS, key=lambda name: (-base_plan.crop_scores[name], name)):
        if len(orders) >= MAX_ORDERS or committed >= capacity:
            break
        if remaining_days < float(CROP_DATA[crop]["first"]):
            continue
        deficit = max(0, target_crops[crop] - crops[crop] - seeds[crop])
        cost = int(CROP_DATA[crop]["seed"])
        quantity = min(deficit, capacity - committed, int(spendable // cost))
        if quantity:
            orders.append(["BUY_SEED", crop, quantity])
            spendable -= quantity * cost
            money -= quantity * cost
            seeds[crop] += quantity
            committed += quantity

    # Hiring remains the original deterministic operational decision.
    hands = len(farm.get("hands", []) or [])
    hires_today = int(farm.get("hires_today", hands))
    while (
        hands < base_plan.target_hands
        and hands < strategy.max_hands
        and hour <= strategy.hire_until_hour
        and len(orders) < MAX_ORDERS
    ):
        cost = hire_cost(hires_today, farm_hand_cost_mult)
        if money < cost:
            break
        orders.append(["HIRE"])
        money -= cost
        spendable = max(0.0, spendable - cost)
        hands += 1
        hires_today += 1

    for animal in sorted(ANIMALS, key=lambda name: (-base_plan.animal_scores[name], name)):
        if len(orders) >= MAX_ORDERS:
            break
        if remaining_days < float(ANIMAL_DATA[animal]["first"]) + 1.0:
            continue
        deficit = max(0, target_animals[animal] - owned_animals[animal])
        cost = int(ANIMAL_DATA[animal]["cost"])
        quantity = min(deficit, int(spendable // cost))
        if quantity:
            orders.append(["BUY_ANIMAL", animal, quantity])
            spendable -= quantity * cost
            owned_animals[animal] += quantity

    return orders[:MAX_ORDERS]
