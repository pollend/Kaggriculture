"""Economic intents for the v5 mixed-farm controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.mixed_farm_v5.portfolio import build_plan as legacy_plan
from agents.mixed_farm_v5.rules import ANIMAL_DATA, ANIMALS, CROP_DATA, CROPS, count_assets, get
from agents.mixed_farm_v5.strategy_v5 import Strategy


@dataclass(frozen=True, slots=True)
class InvestmentIntent:
    kind: str
    item: str
    quantity: int
    cost: float
    value: float
    setup_work: float
    recurring_work: float
    capacity: int
    deadline: int

    @property
    def net_value(self) -> float:
        return self.value - self.cost


@dataclass(frozen=True, slots=True)
class Plan:
    target_crops: dict[str, int]
    target_animals: dict[str, int]
    crop_scores: dict[str, float]
    animal_scores: dict[str, float]
    wheat_reserve: int
    fertilizer_reserve: int
    next_land_price: int | None
    intents: tuple[InvestmentIntent, ...]


def build_plan(obs: Any, farm: dict, strategy: Strategy, *, episode_steps: int = 720, turns_per_day: int = 24) -> Plan:
    old = legacy_plan(
        obs, farm, strategy.to_legacy(), episode_steps=episode_steps,
        turns_per_day=turns_per_day,
    )
    step = int(get(obs, "step", 0))
    remaining_turns = max(0, episode_steps - step - 1)
    crops, animals, structures = count_assets(farm)
    private = get(obs, "private", {}) or {}
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    intents: list[InvestmentIntent] = []
    for crop in CROPS:
        owned = crops[crop] + int(seeds.get(crop, 0))
        quantity = min(max(0, old.target_crops[crop] - owned), strategy.opening_crop_tranche)
        if quantity:
            cost = float(CROP_DATA[crop]["seed"]) * quantity
            value = max(0.0, old.crop_scores[crop]) * quantity
            if value > cost and remaining_turns > int(CROP_DATA[crop]["first"]) * turns_per_day:
                intents.append(InvestmentIntent("seed", crop, quantity, cost, value, 2.0 * quantity, quantity, quantity, remaining_turns))
    for animal in ANIMALS:
        owned = animals[animal] + int(shed.get(animal, 0)) + sum(int((slot or {}).get(animal, 0)) for slot in inventories)
        quantity = min(max(0, old.target_animals[animal] - owned), strategy.opening_animal_batch)
        if quantity:
            cost = float(ANIMAL_DATA[animal]["cost"]) * quantity
            value = max(0.0, old.animal_scores[animal]) * quantity
            structure = str(ANIMAL_DATA[animal]["structure"])
            free = max(0, structures[structure] - sum(animals.values()))
            setup = quantity * (2.0 if free else 4.0)
            recurring = quantity * 3.0
            if value > cost * strategy.animal_risk_premium and remaining_turns > (int(ANIMAL_DATA[animal]["first"]) + 1) * turns_per_day:
                intents.append(InvestmentIntent("animal", animal, quantity, cost, value, setup, recurring, quantity, remaining_turns))
    if old.buy_land and old.next_land_price is not None:
        marginal = max([0.0, *old.crop_scores.values(), *old.animal_scores.values()])
        value = marginal * 25
        if value >= old.next_land_price * strategy.land_hurdle:
            intents.append(InvestmentIntent("land", "LAND", 1, float(old.next_land_price), value, 1.0, 0.0, 25, remaining_turns))
    intents.sort(key=lambda item: (-(item.net_value / max(1.0, item.setup_work + item.recurring_work)), item.kind, item.item))
    discretionary_budget = max(0.0, float(farm.get("money", 0)) - strategy.cash_reserve)
    funded: list[InvestmentIntent] = []
    for intent in intents:
        if intent.cost <= discretionary_budget:
            funded.append(intent)
            discretionary_budget -= intent.cost
    return Plan(old.target_crops, old.target_animals, old.crop_scores, old.animal_scores,
                old.wheat_reserve, old.fertilizer_reserve, old.next_land_price, tuple(funded))
