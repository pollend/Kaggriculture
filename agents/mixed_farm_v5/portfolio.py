"""Greedy remaining-season portfolio construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from agents.mixed_farm_v5.rules import (
    ANIMAL_DATA,
    ANIMALS,
    CROP_DATA,
    CROPS,
    LAND_PRICES,
    PRODUCTS,
    count_assets,
    get,
    opponent_supply,
    town_counts,
    unlocked_empty_tiles,
)
from agents.mixed_farm_v5.strategy import Strategy


@dataclass(frozen=True, slots=True)
class Plan:
    target_crops: dict[str, int]
    target_animals: dict[str, int]
    crop_scores: dict[str, float]
    animal_scores: dict[str, float]
    target_hands: int
    wheat_reserve: int
    fertilizer_reserve: int
    buy_land: bool
    next_land_price: int | None


def _prices(obs: Any) -> dict[str, float]:
    raw = (get(obs, "market", {}) or {}).get("prices", {}) or {}
    defaults = {name: float(CROP_DATA[name]["base"]) for name in CROPS}
    defaults.update(
        {str(data["product"]): float(data["base"]) for data in ANIMAL_DATA.values()}
    )
    defaults["FERTILIZER"] = 100.0
    return {name: float(raw.get(name, defaults[name])) for name in PRODUCTS}


def _allocate(total: int, scores: dict[str, float]) -> dict[str, int]:
    positive = {name: max(0.0, value) for name, value in scores.items()}
    denominator = sum(positive.values())
    if denominator <= 0:
        return {name: 0 for name in scores}
    exact = {name: total * value / denominator for name, value in positive.items()}
    result = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(scores, key=lambda name: (-(exact[name] - result[name]), -scores[name], name))
    for name in order[:remaining]:
        result[name] += 1
    return result


def build_plan(
    obs: Any,
    farm: dict,
    strategy: Strategy,
    *,
    episode_steps: int = 720,
    turns_per_day: int = 24,
) -> Plan:
    """Select a profitable target portfolio without simulating field actions."""

    player = int(get(obs, "player", 0))
    step = int(get(obs, "step", 0))
    remaining_days = max(0.0, (episode_steps - step - 1) / max(1, turns_per_day))
    prices = _prices(obs)
    demand = town_counts(obs)
    supply = opponent_supply(obs, player)

    crop_scores: dict[str, float] = {}
    for crop in CROPS:
        data = CROP_DATA[crop]
        first = int(data["first"])
        if remaining_days < first:
            crop_scores[crop] = -float(data["seed"])
            continue
        if bool(data["ongoing"]):
            cycles = 1
            harvests = int(data["yield"])
        else:
            cycle_days = int(data["peak"]) + 1
            cycles = max(1, int(remaining_days // cycle_days))
            harvests = int(data["yield"]) * cycles
        revenue = prices[crop] * harvests
        costs = float(data["seed"]) * cycles
        demand_boost = 1.0 + strategy.town_demand_weight * demand[crop] / 4.0
        supply_discount = 1.0 + strategy.opponent_supply_weight * supply[crop] / 25.0
        labor = max(1.0, first + cycles * 2.0)
        crop_scores[crop] = (
            (revenue * demand_boost / supply_discount - costs)
            * strategy.crop_weights[crop]
            / labor
        )

    animal_scores: dict[str, float] = {}
    target_animals: dict[str, int] = {}
    for animal in ANIMALS:
        data = ANIMAL_DATA[animal]
        first = int(data["first"])
        product = str(data["product"])
        if remaining_days < first + 1:
            animal_scores[animal] = -float(data["cost"])
            target_animals[animal] = 0
            continue
        productions = 1 + int(max(0.0, remaining_days - first) // int(data["interval"]))
        care_multiplier = 1.0 + int(data["interval"]) * min(1.0, strategy.care_value_weight)
        product_revenue = productions * prices[product] * care_multiplier
        fertilizer_revenue = (
            remaining_days * prices["FERTILIZER"] * 0.45 * strategy.fertilizer_value_weight
        )
        feed_cost = remaining_days * prices["WHEAT"]
        demand_boost = 1.0 + strategy.town_demand_weight * demand[product] / 4.0
        supply_discount = 1.0 + strategy.opponent_supply_weight * supply[product] / 12.0
        net = (
            product_revenue * demand_boost / supply_discount
            + fertilizer_revenue
            - feed_cost
            - float(data["cost"])
        ) / strategy.animal_risk_premium
        animal_scores[animal] = net * strategy.animal_weights[animal]
        target_animals[animal] = strategy.animal_targets[animal] if net > 0 else 0

    requested_animal_total = sum(target_animals.values())
    # Crop scale and herd support are learned policy controls. Physical board
    # capacity is a hard bound: impossible portfolios waste most of a small
    # surrogate data set teaching the forest that two independent maxima cannot
    # coexist.
    tiles = farm.get("tiles", []) or []
    board_height = len(tiles)
    board_width = max((len(row) for row in tiles), default=board_height)
    quadrant_capacity = max(1, (board_width // 2) * (board_height // 2))
    farm_capacity = min(
        board_width * board_height,
        strategy.max_quadrants * quadrant_capacity,
    )
    animal_total = requested_animal_total
    while animal_total > 0:
        supported = max(
            strategy.target_crop_tiles,
            strategy.crop_target_floor + math.ceil(
                animal_total * strategy.crop_tiles_per_animal
            ),
        )
        if animal_total + supported <= farm_capacity:
            break
        animal_total -= 1
    if animal_total < requested_animal_total:
        remaining = animal_total
        clipped = {animal: 0 for animal in ANIMALS}
        for animal in sorted(ANIMALS, key=lambda name: (-animal_scores[name], name)):
            amount = min(target_animals[animal], remaining)
            clipped[animal] = amount
            remaining -= amount
        target_animals = clipped
    supported_crop_target = strategy.crop_target_floor + math.ceil(
        animal_total * strategy.crop_tiles_per_animal
    )
    productive_crop_target = min(
        max(0, farm_capacity - animal_total),
        max(strategy.target_crop_tiles, supported_crop_target),
    )
    target_crops = _allocate(productive_crop_target, crop_scores)
    # Market wheat is a useful bridge, not a sustainable primary feed source.
    # One well-kept wheat tile produces about 0.8 units/day, so reserve enough
    # of the crop portfolio to feed the selected herd in steady state.
    wheat_floor = min(productive_crop_target, math.ceil(animal_total / 0.8))
    wheat_missing = max(0, wheat_floor - target_crops["WHEAT"])
    target_crops["WHEAT"] += wheat_missing
    for crop in sorted(
        (name for name in CROPS if name != "WHEAT"),
        key=lambda name: (crop_scores[name], name),
    ):
        moved = min(wheat_missing, target_crops[crop])
        target_crops[crop] -= moved
        wheat_missing -= moved
        if wheat_missing == 0:
            break
    desired_tiles = sum(target_crops.values()) + animal_total
    unlocked = max(1, len(farm.get("unlocked_quadrants", []) or ["NW"]))
    next_land_price = LAND_PRICES[unlocked - 1] if unlocked <= len(LAND_PRICES) else None
    money = float(farm.get("money", 0))
    buy_land = False
    private = get(obs, "private", {}) or {}
    crops_now, animals_now, _ = count_assets(farm)
    pending = sum(crops_now.values()) + sum(animals_now.values())
    pending += sum(int(value) for value in (private.get("seeds", {}) or {}).values())
    pending += sum(int((private.get("shed", {}) or {}).get(name, 0)) for name in ANIMALS)
    pending += sum(
        int((slot or {}).get(name, 0))
        for slot in (private.get("inventories", []) or [])
        for name in ANIMALS
    )
    capacity_pressure = desired_tiles > unlocked * quadrant_capacity
    forced_expansion = (
        strategy.land_expansion_day < 30
        and int(get(obs, "day", step // max(1, turns_per_day)))
        >= strategy.land_expansion_day
        and capacity_pressure
    )
    normally_ready = pending >= math.ceil(
        unlocked * quadrant_capacity * strategy.land_commitment_fraction
    )
    if (
        next_land_price is not None
        and unlocked < strategy.max_quadrants
        and capacity_pressure
        and (normally_ready or forced_expansion)
        and money - strategy.cash_reserve >= next_land_price
    ):
        extra_tiles = min(25, desired_tiles - unlocked * 25)
        marginal = max(0.0, *(crop_scores.values()), *(animal_scores.values()))
        buy_land = forced_expansion or (
            marginal * extra_tiles >= next_land_price * strategy.land_hurdle
        )

    # Labor follows work that can be acted on now, not the eventual portfolio.
    # The former target-based estimate hired a full crew before the farm owned
    # enough seeds, structures, or animals to keep it busy.  Seeds and animals
    # already in transit count extra because setup consumes several actions.
    seeds = private.get("seeds", {}) or {}
    plantable_seeds = min(
        sum(int(seeds.get(crop, 0)) for crop in CROPS),
        len(unlocked_empty_tiles(farm)),
    )
    inventories = private.get("inventories", []) or []
    shed = private.get("shed", {}) or {}
    unplaced_animals = sum(
        int(shed.get(animal, 0))
        + sum(int((slot or {}).get(animal, 0)) for slot in inventories)
        for animal in ANIMALS
    )
    near_term_actions = (
        sum(crops_now.values())
        + 2 * plantable_seeds
        + 3 * sum(animals_now.values())
        + 4 * unplaced_animals
    )
    target_hands = min(
        strategy.max_hands,
        max(
            strategy.minimum_hands,
            max(0, math.ceil(near_term_actions / strategy.workload_per_hand) - 1),
        ),
    )
    wheat_reserve = (sum(animals_now.values()) + unplaced_animals) * strategy.feed_reserve_days
    fertilizer_reserve = min(6, math.ceil(sum(target_crops.values()) / 8))
    return Plan(
        target_crops,
        target_animals,
        crop_scores,
        animal_scores,
        target_hands,
        wheat_reserve,
        fertilizer_reserve,
        buy_land,
        next_land_price,
    )
