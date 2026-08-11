"""Reserve-safe market execution for accepted v5 intents."""

from __future__ import annotations

from typing import Any

from agents.mixed_farm_v5.labor_v5 import LaborPlan
from agents.mixed_farm_v5.portfolio_v5 import Plan
from agents.mixed_farm_v5.rules import ANIMAL_DATA, CROP_DATA, PRODUCTS, get
from agents.mixed_farm_v5.strategy_v5 import Strategy

MAX_ORDERS = 10


def market_orders(obs: Any, farm: dict, private: dict, plan: Plan, labor: LaborPlan, strategy: Strategy, *, episode_steps: int = 720, turns_per_day: int = 24) -> list[list[Any]]:
    step = int(get(obs, "step", 0))
    final_day = step >= episode_steps - turns_per_day
    liquidation = step >= episode_steps - strategy.liquidation_turns
    shed = private.get("shed", {}) or {}
    prices = (get(obs, "market", {}) or {}).get("prices", {}) or {}
    orders: list[list[Any]] = []
    for item in PRODUCTS:
        amount = int(shed.get(item, 0))
        reserve = plan.wheat_reserve if item == "WHEAT" and not final_day else plan.fertilizer_reserve if item == "FERTILIZER" and not liquidation else 0
        amount = min(strategy.sell_batch_size, max(0, amount - reserve))
        base = float(CROP_DATA[item]["base"]) if item in CROP_DATA else next(
            (float(data["base"]) for data in ANIMAL_DATA.values() if data["product"] == item),
            100.0,
        )
        favorable_price = float(prices.get(item, base)) / max(1.0, base) >= strategy.sell_price_ratio
        # `<=`, not `<`: money settles at exactly the reserve once the budget
        # (money - cash_reserve) hits zero and buying stops, so a strict compare
        # never fires and a high sell_price_ratio can sit on a full shed for the
        # rest of the season.
        if amount and (liquidation or favorable_price or float(farm.get("money", 0)) <= strategy.cash_reserve or sum(map(int, shed.values())) >= 100 * strategy.shed_pressure_fraction):
            orders.append(["SELL", item, amount])
    if final_day:
        return orders[:MAX_ORDERS]
    money = float(farm.get("money", 0))
    # Feeding is the sole emergency purchase allowed to cross the reserve.
    wheat = int(shed.get("WHEAT", 0)) + sum(
        int((slot or {}).get("WHEAT", 0)) for slot in private.get("inventories", []) or []
    )
    if wheat < plan.wheat_reserve:
        unit = float(prices.get("WHEAT", CROP_DATA["WHEAT"]["base"]))
        amount = min(plan.wheat_reserve - wheat, int(money // max(1.0, unit)))
        if amount:
            orders.append(["BUY_PRODUCT", "WHEAT", amount])
            money -= amount * unit
    if liquidation:
        return orders[:MAX_ORDERS]
    effective_hires = labor.hire_count if money - labor.hire_cost >= strategy.cash_reserve else 0
    effective_hire_cost = labor.hire_cost if effective_hires else 0
    budget = max(0.0, money - strategy.cash_reserve - effective_hire_cost)
    for intent in plan.intents:
        if len(orders) >= MAX_ORDERS - effective_hires or intent.cost > budget:
            continue
        if intent.kind == "seed":
            orders.append(["BUY_SEED", intent.item, intent.quantity])
        elif intent.kind == "animal":
            orders.append(["BUY_ANIMAL", intent.item, intent.quantity])
        elif intent.kind == "land":
            orders.append(["BUY_LAND"])
        else:
            continue
        budget -= intent.cost
    orders.extend([["HIRE"] for _ in range(effective_hires)])
    return orders[:MAX_ORDERS]
