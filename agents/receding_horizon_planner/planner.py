"""Three-day aggregate beam planner for farm portfolio targets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any

from agents.receding_horizon_planner.rules import (
    ANIMAL_DATA,
    ANIMAL_ORDER,
    CROP_DATA,
    CROPS,
    PRODUCTS,
    SHOP_DEMAND,
    get,
    market_price,
    next_land,
)
from agents.receding_horizon_planner.strategy import Strategy

HORIZON_DAYS = 3
BEAM_WIDTH = 64


@dataclass(frozen=True, slots=True)
class Plan:
    day: int
    target_crops: dict[str, int]
    target_animals: dict[str, int]
    target_hands: int
    buy_land: bool
    wheat_reserve: int
    score: float
    rationale: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Investment:
    kind: str
    item: str
    quantity: int
    cost: float
    value: float
    maintenance: float
    tiles: int

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.item}:{self.quantity}"


@dataclass(frozen=True, slots=True)
class _BeamState:
    cash: float
    score: float
    crops: tuple[int, ...]
    animals: tuple[int, ...]
    hands: int
    land_count: int
    capacity: int
    maintenance: float
    rationale: tuple[str, ...]
    day0_crops: tuple[int, ...] | None = None
    day0_animals: tuple[int, ...] | None = None
    day0_hands: int | None = None
    day0_land_count: int | None = None


def _asset_counts(farm: dict, private: dict) -> tuple[dict[str, int], dict[str, int], int]:
    crops = {crop: 0 for crop in CROPS}
    animals = {animal: 0 for animal in ANIMAL_ORDER}
    occupied = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            occupied += 1
            if tile.get("kind") == "PLANT" and tile.get("crop") in crops:
                crops[str(tile["crop"])] += 1
            animal = tile.get("animal")
            if animal in animals:
                animals[str(animal)] += 1

    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    seeds = private.get("seeds", {}) or {}
    for crop in CROPS:
        # Purchased seeds are committed portfolio capacity. Counting them keeps
        # the next daily replan from abandoning yesterday's purchases and buying
        # a different crop before the first batch is planted.
        crops[crop] += int(seeds.get(crop, 0))
    for animal in ANIMAL_ORDER:
        animals[animal] += int(shed.get(animal, 0))
        animals[animal] += sum(int((slot or {}).get(animal, 0)) for slot in inventories)
    return crops, animals, occupied


def _town_drain(obs: Any, item: str) -> float:
    day = int(get(obs, "day", 0))
    center = 0.0 if item == "FERTILIZER" else (2.0 if day < 10 else 4.0 if day < 20 else 8.0)
    shops = 0.0
    for name in (get(obs, "town", {}) or {}).get("unlocked_shops", []) or []:
        demand = SHOP_DEMAND.get(str(name), ())
        if item in demand:
            shops += 12.0 if len(demand) == 1 else 6.0
    return center + shops


def _opponent_supply(obs: Any, player: int, item: str) -> float:
    supply = 0.0
    for index, farm in enumerate(get(obs, "farms", []) or []):
        if index == player:
            continue
        for row in farm.get("tiles") or []:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                if tile.get("kind") == "PLANT" and tile.get("crop") == item:
                    data = CROP_DATA[item]
                    supply += float(data["yield"]) / max(1.0, float(data["last"]) + 1.0)
                animal = tile.get("animal")
                if animal in ANIMAL_DATA and ANIMAL_DATA[str(animal)]["product"] == item:
                    supply += 1.0 / float(ANIMAL_DATA[str(animal)]["interval"])
    return supply


def _forecast_prices(obs: Any, player: int, strategy: Strategy) -> dict[str, float]:
    market = get(obs, "market", {}) or {}
    inventory = market.get("inventory", {}) or {}
    forecast: dict[str, float] = {}
    for item in PRODUCTS:
        projected = float(inventory.get(item, 10_000))
        projected += HORIZON_DAYS * strategy.opponent_supply_weight * _opponent_supply(obs, player, item)
        projected -= HORIZON_DAYS * strategy.town_demand_weight * _town_drain(obs, item)
        forecast[item] = float(market_price(item, projected, obs))
    return forecast


def _crop_value(crop: str, remaining_days: float, prices: dict[str, float], strategy: Strategy) -> tuple[float, float]:
    data = CROP_DATA[crop]
    if remaining_days < float(data["first"]):
        return -1e9, 0.0
    revenue = float(data["yield"]) * prices[crop]
    actions = 2.0 + min(remaining_days, float(data["last"]) + 1.0)
    labor = actions * strategy.movement_cost
    value = strategy.risk_discount * revenue - float(data["seed"]) - labor
    # One care action plus an allowance for travel and occasional harvest.
    maintenance = 3.0
    return value, maintenance


def _animal_value(
    animal: str, remaining_days: float, prices: dict[str, float], strategy: Strategy
) -> tuple[float, float]:
    data = ANIMAL_DATA[animal]
    if remaining_days < float(data["first"]):
        return -1e9, 0.0
    cycles = 1 + int((remaining_days - float(data["first"])) // float(data["interval"]))
    care_units = strategy.care_value_weight * min(float(data["interval"]), float(data["max"]) - 1.0)
    product_revenue = cycles * (1.0 + care_units) * prices[str(data["product"])]
    fertilizer_revenue = remaining_days * prices["FERTILIZER"] * 0.45 * strategy.fertilizer_value_weight
    feed_cost = remaining_days * prices["WHEAT"]
    labor = remaining_days * (2.0 + 1.0 / float(data["interval"])) * strategy.movement_cost
    value = (
        strategy.risk_discount * (product_revenue + fertilizer_revenue)
        - float(data["cost"]) * strategy.animal_risk_premium
        - feed_cost
        - labor
    )
    # Feed + care every day, plus travel, harvest, and fertilizer collection.
    maintenance = 5.0 + 2.0 / float(data["interval"])
    return value, maintenance


def _options(
    state: _BeamState, remaining_days: float, prices: dict[str, float], strategy: Strategy, land_price: int | None
) -> list[Investment]:
    options: list[Investment] = []
    for crop, quantity in ((crop, quantity) for crop in CROPS for quantity in (1, 3, 6)):
        value, maintenance = _crop_value(crop, remaining_days, prices, strategy)
        if value > 0:
            options.append(
                Investment(
                    "crop",
                    crop,
                    quantity,
                    CROP_DATA[crop]["seed"] * quantity,
                    value * quantity,
                    maintenance * quantity,
                    quantity,
                )
            )
    for animal, quantity in ((animal, quantity) for animal in ANIMAL_ORDER for quantity in (1, 2)):
        value, maintenance = _animal_value(animal, remaining_days, prices, strategy)
        if value > 0:
            options.append(
                Investment(
                    "animal",
                    animal,
                    quantity,
                    ANIMAL_DATA[animal]["cost"] * quantity,
                    value * quantity,
                    maintenance * quantity,
                    quantity,
                )
            )
    if state.hands < strategy.max_hands:
        options.append(Investment("hand", "HAND", 1, 8.0, 260.0, 0.0, 0))

    best_tile_value = max((item.value / max(1, item.tiles) for item in options if item.tiles), default=0.0)
    usage = sum(state.crops) + sum(state.animals)
    if land_price is not None and usage >= int(state.capacity * 0.72):
        value = best_tile_value * min(12, state.capacity) - land_price * strategy.land_hurdle
        options.append(Investment("land", "LAND", 1, float(land_price), value, 0.0, 0))
    return sorted(options, key=lambda item: (item.value, -item.cost, item.label), reverse=True)[:12]


def _apply(
    state: _BeamState, investments: tuple[Investment, ...], strategy: Strategy, day_offset: int
) -> _BeamState | None:
    candidate = state
    for item in investments:
        if candidate.cash - item.cost < strategy.cash_reserve:
            return None
        crops = list(candidate.crops)
        animals = list(candidate.animals)
        hands = candidate.hands
        land_count = candidate.land_count
        capacity = candidate.capacity
        if item.kind == "crop":
            crops[CROPS.index(item.item)] += item.quantity
        elif item.kind == "animal":
            animals[ANIMAL_ORDER.index(item.item)] += item.quantity
        elif item.kind == "hand":
            hands += item.quantity
        elif item.kind == "land":
            land_count += 1
            capacity += 25

        if sum(crops) + sum(animals) > capacity:
            return None
        maintenance = candidate.maintenance + item.maintenance
        available_actions = (1 + hands) * 24.0 * (1.0 - strategy.capacity_reserve)
        if maintenance > available_actions:
            return None
        candidate = replace(
            candidate,
            cash=candidate.cash - item.cost,
            score=candidate.score + item.value * (strategy.risk_discount**day_offset),
            crops=tuple(crops),
            animals=tuple(animals),
            hands=hands,
            land_count=land_count,
            capacity=capacity,
            maintenance=maintenance,
            rationale=(*candidate.rationale, f"d+{day_offset}:{item.label}"),
        )

    if day_offset == 0:
        candidate = replace(
            candidate,
            day0_crops=candidate.crops,
            day0_animals=candidate.animals,
            day0_hands=candidate.hands,
            day0_land_count=candidate.land_count,
        )
    return candidate


def _deduplicate(states: list[_BeamState]) -> list[_BeamState]:
    best: dict[tuple[Any, ...], _BeamState] = {}
    for state in states:
        key = (state.crops, state.animals, state.hands, state.land_count, state.day0_crops, state.day0_animals)
        incumbent = best.get(key)
        if incumbent is None or state.score > incumbent.score:
            best[key] = state
    return sorted(best.values(), key=lambda item: (item.score, item.cash, item.rationale), reverse=True)[:BEAM_WIDTH]


def build_plan(
    obs: Any,
    farm: dict,
    private: dict,
    player: int,
    strategy: Strategy,
    episode_steps: int = 720,
    turns_per_day: int = 24,
) -> Plan:
    day = int(get(obs, "day", 0))
    step = int(get(obs, "step", 0))
    remaining_days = max(0.0, (episode_steps - step) / max(1, turns_per_day))
    crop_counts, animal_counts, occupied = _asset_counts(farm, private)
    land_count = len(farm.get("unlocked_quadrants") or ("NW",))
    capacity = max(occupied, land_count * 25)
    hands = len(farm.get("hands") or [])
    maintenance = 3.0 * sum(crop_counts.values()) + sum(
        animal_counts[animal] * (5.0 + 2.0 / float(ANIMAL_DATA[animal]["interval"])) for animal in ANIMAL_ORDER
    )
    initial = _BeamState(
        cash=float(farm.get("money", 0.0)),
        score=0.0,
        crops=tuple(crop_counts[crop] for crop in CROPS),
        animals=tuple(animal_counts[animal] for animal in ANIMAL_ORDER),
        hands=hands,
        land_count=land_count,
        capacity=capacity,
        maintenance=maintenance,
        rationale=(),
    )
    prices = _forecast_prices(obs, player, strategy)
    land = next_land(farm)
    beam = [initial]
    for day_offset in range(HORIZON_DAYS):
        expanded: list[_BeamState] = []
        for state in beam:
            choices = _options(
                state,
                max(0.0, remaining_days - day_offset),
                prices,
                strategy,
                land[1] if land and state.land_count == land_count else None,
            )
            groups: list[tuple[Investment, ...]] = [()]
            groups.extend((item,) for item in choices)
            pairs = sorted(
                combinations(choices, 2),
                key=lambda pair: (pair[0].value + pair[1].value, pair[0].label, pair[1].label),
                reverse=True,
            )[:12]
            groups.extend(pairs)
            for group in groups:
                candidate = _apply(state, group, strategy, day_offset)
                if candidate is not None:
                    expanded.append(candidate)
        beam = _deduplicate(expanded)

    chosen = beam[0] if beam else _apply(initial, (), strategy, 0) or initial
    day0_crops = chosen.day0_crops or initial.crops
    day0_animals = chosen.day0_animals or initial.animals
    day0_hands = chosen.day0_hands if chosen.day0_hands is not None else initial.hands
    day0_land = chosen.day0_land_count if chosen.day0_land_count is not None else initial.land_count
    animals_total = sum(day0_animals)
    wheat_reserve = max(0, animals_total * HORIZON_DAYS)
    return Plan(
        day=day,
        target_crops=dict(zip(CROPS, day0_crops)),
        target_animals=dict(zip(ANIMAL_ORDER, day0_animals)),
        target_hands=min(strategy.max_hands, day0_hands),
        buy_land=day0_land > initial.land_count,
        wheat_reserve=wheat_reserve,
        score=chosen.score * strategy.terminal_asset_discount,
        rationale=chosen.rationale,
    )
