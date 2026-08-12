"""Feature encoding and Random Forest market decisions.

The forest never emits raw Kaggriculture orders.  It chooses a bounded purchase
target once per day and a product reserve whenever inventory can be sold.  The
executor in :mod:`market` owns legality, affordability, survival, and terminal
overrides.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from agents.mixed_farm_rf_market.portfolio import Plan
from agents.mixed_farm_rf_market.rules import (
    ANIMAL_DATA,
    ANIMALS,
    CROP_DATA,
    CROPS,
    PRODUCTS,
    count_assets,
    get,
    opponent_supply,
    town_counts,
    unlocked_empty_tiles,
)
from agents.mixed_farm_rf_market.strategy import Strategy

BUNDLE_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
PURCHASE_ACTION_WIDTH = len(CROPS) + len(ANIMALS) + 3
SALE_ACTION_WIDTH = len(PRODUCTS) + 4
RESERVE_FRACTIONS = (0.0, 0.25, 0.5, 1.0)


@dataclass(frozen=True, slots=True)
class PurchasePlan:
    target_crops: tuple[int, ...]
    target_animals: tuple[int, ...]
    target_land: int
    wheat_reserve: int
    scale: float

    def crop_targets(self) -> dict[str, int]:
        return dict(zip(CROPS, self.target_crops))

    def animal_targets(self) -> dict[str, int]:
        return dict(zip(ANIMALS, self.target_animals))


@dataclass(slots=True)
class MarketPolicyBundle:
    strategy: Strategy
    purchase_model: Any | None
    sale_model: Any | None
    dataset_sha256: str = ""
    schema_version: int = BUNDLE_SCHEMA_VERSION

    @classmethod
    def load(cls, path: str | Path) -> MarketPolicyBundle:
        payload = joblib.load(path)
        if isinstance(payload, cls):
            bundle = payload
        elif isinstance(payload, dict):
            bundle = cls(
                strategy=Strategy.from_dict(dict(payload["strategy"])),
                purchase_model=payload.get("purchase_model"),
                sale_model=payload.get("sale_model"),
                dataset_sha256=str(payload.get("dataset_sha256", "")),
                schema_version=int(payload.get("schema_version", 0)),
            )
        else:
            raise ValueError("unsupported RF market checkpoint payload")
        if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"RF market checkpoint schema {bundle.schema_version} is not "
                f"supported (expected {BUNDLE_SCHEMA_VERSION})"
            )
        return bundle

    def dump(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema_version": self.schema_version,
                "state_schema_version": STATE_SCHEMA_VERSION,
                "strategy": self.strategy.to_dict(),
                "purchase_model": self.purchase_model,
                "sale_model": self.sale_model,
                "dataset_sha256": self.dataset_sha256,
            },
            destination,
        )


def _carried(private: dict, names: tuple[str, ...]) -> dict[str, int]:
    result = {name: 0 for name in names}
    for inventory in private.get("inventories", []) or []:
        for name in names:
            result[name] += int((inventory or {}).get(name, 0))
    return result


def market_state_features(
    obs: Any,
    farm: dict,
    private: dict,
    plan: Plan,
    *,
    episode_steps: int = 720,
    turns_per_day: int = 24,
) -> np.ndarray:
    """Encode only information available to the acting player."""

    step = int(get(obs, "step", 0))
    day = int(get(obs, "day", step // max(1, turns_per_day)))
    hour = int(get(obs, "hour", step % max(1, turns_per_day)))
    player = int(get(obs, "player", 0))
    farms = get(obs, "farms", []) or []
    opponent_money = max(
        (float(other.get("money", 0)) for index, other in enumerate(farms) if index != player),
        default=0.0,
    )
    money = float(farm.get("money", 0))
    shed = private.get("shed", {}) or {}
    seeds = private.get("seeds", {}) or {}
    carried = _carried(private, (*PRODUCTS, *ANIMALS))
    market = get(obs, "market", {}) or {}
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}
    crops, animals, _ = count_assets(farm)
    opponent = opponent_supply(obs, player)
    demand = town_counts(obs)
    total_days = max(1.0, episode_steps / max(1, turns_per_day))
    shed_total = sum(int(value) for value in shed.values())

    values: list[float] = [
        day / total_days,
        hour / max(1, turns_per_day),
        step / max(1, episode_steps),
        max(0, episode_steps - step) / max(1, episode_steps),
        np.log1p(max(0.0, money)) / 10.0,
        money / 10_000.0,
        opponent_money / 10_000.0,
        (money - opponent_money) / 10_000.0,
    ]
    values.extend(int(seeds.get(name, 0)) / 25.0 for name in CROPS)
    values.extend(int(shed.get(name, 0)) / 100.0 for name in PRODUCTS)
    values.extend(carried[name] / 50.0 for name in PRODUCTS)
    values.extend(carried[name] / 10.0 for name in ANIMALS)
    for name in PRODUCTS:
        base = (
            float(CROP_DATA[name]["base"])
            if name in CROP_DATA
            else next(
                (float(data["base"]) for data in ANIMAL_DATA.values() if data["product"] == name),
                100.0,
            )
        )
        values.append(float(prices.get(name, base)) / max(1.0, base))
    values.extend((float(inventory.get(name, 1000)) - 1000.0) / 1000.0 for name in PRODUCTS)
    values.extend(
        (
            len(farm.get("hands", []) or []) / 24.0,
            int(farm.get("hires_today", 0)) / 24.0,
            len(farm.get("unlocked_quadrants", []) or ["NW"]) / 4.0,
            len(unlocked_empty_tiles(farm)) / 100.0,
            shed_total / 100.0,
        )
    )
    values.extend(crops[name] / 100.0 for name in CROPS)
    values.extend(animals[name] / 50.0 for name in ANIMALS)
    values.extend(opponent[name] / 100.0 for name in PRODUCTS)
    values.extend(demand[name] / 4.0 for name in PRODUCTS)
    values.extend(plan.target_crops[name] / 100.0 for name in CROPS)
    values.extend(plan.target_animals[name] / 50.0 for name in ANIMALS)
    values.extend(plan.crop_scores[name] / 10_000.0 for name in CROPS)
    values.extend(plan.animal_scores[name] / 10_000.0 for name in ANIMALS)
    values.extend(
        (
            plan.target_hands / 24.0,
            plan.wheat_reserve / 100.0,
            plan.fertilizer_reserve / 20.0,
            float(plan.buy_land),
        )
    )
    return np.asarray(values, dtype=np.float32)


def _scaled_targets(
    targets: dict[str, int],
    names: tuple[str, ...],
    current: dict[str, int],
    scale: float,
    emphasis: str | None,
) -> tuple[int, ...]:
    if scale == 0.0:
        return tuple(int(current[name]) for name in names)
    desired = {
        name: max(int(current[name]), round(int(current[name]) + scale * max(0, int(targets[name]) - int(current[name]))))
        for name in names
    }
    if emphasis is not None and sum(desired.values()) > 0:
        total = sum(desired.values())
        desired[emphasis] *= 2
        denominator = sum(desired.values())
        exact = {name: total * desired[name] / denominator for name in names}
        desired = {name: int(np.floor(exact[name])) for name in names}
        for name in sorted(names, key=lambda item: (-(exact[item] - desired[item]), item))[: total - sum(desired.values())]:
            desired[name] += 1
        for name in names:
            desired[name] = max(int(current[name]), desired[name])
    return tuple(desired[name] for name in names)


def enumerate_purchase_candidates(
    obs: Any,
    farm: dict,
    private: dict,
    plan: Plan,
    strategy: Strategy,
    *,
    episode_steps: int = 720,
    turns_per_day: int = 24,
) -> tuple[PurchasePlan, ...]:
    step = int(get(obs, "step", 0))
    remaining_days = max(0.0, (episode_steps - step - 1) / max(1, turns_per_day))
    crops, animals, _ = count_assets(farm)
    seeds = private.get("seeds", {}) or {}
    for crop in CROPS:
        crops[crop] += int(seeds.get(crop, 0))
    shed = private.get("shed", {}) or {}
    carried = _carried(private, ANIMALS)
    for animal in ANIMALS:
        animals[animal] += int(shed.get(animal, 0)) + carried[animal]
    viable_crops = [name for name in CROPS if remaining_days >= float(CROP_DATA[name]["first"])]
    viable_animals = [
        name for name in ANIMALS if remaining_days >= float(ANIMAL_DATA[name]["first"]) + 1.0
    ]
    current_land = max(1, len(farm.get("unlocked_quadrants", []) or ["NW"]))
    land_levels = [current_land]
    if current_land < min(4, strategy.max_quadrants):
        land_levels.append(current_land + 1)
    owned_animals = sum(animals.values())
    survival = owned_animals * min(strategy.feed_reserve_days, max(0, int(np.ceil(remaining_days))))
    wheat_levels = sorted({survival, max(survival, plan.wheat_reserve), max(survival, plan.wheat_reserve + 2 * owned_animals)})

    shapes: list[tuple[float, str | None, str | None]] = [(0.0, None, None)]
    shapes.extend((scale, crop, None) for scale in (0.5, 1.0, 1.25) for crop in (None, *viable_crops))
    shapes.extend((1.0, None, animal) for animal in viable_animals)
    candidates: dict[tuple[Any, ...], PurchasePlan] = {}
    for scale, crop_emphasis, animal_emphasis in shapes:
        crop_targets = _scaled_targets(plan.target_crops, CROPS, crops, scale, crop_emphasis)
        animal_targets = _scaled_targets(plan.target_animals, ANIMALS, animals, scale, animal_emphasis)
        for target_land in land_levels:
            for wheat_reserve in wheat_levels:
                candidate = PurchasePlan(crop_targets, animal_targets, target_land, wheat_reserve, scale)
                key = (*crop_targets, *animal_targets, target_land, wheat_reserve)
                candidates.setdefault(key, candidate)
    return tuple(candidates.values())


def purchase_action_features(candidate: PurchasePlan) -> np.ndarray:
    values = [value / 100.0 for value in candidate.target_crops]
    values.extend(value / 50.0 for value in candidate.target_animals)
    values.extend((candidate.target_land / 4.0, candidate.wheat_reserve / 100.0, candidate.scale / 1.25))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (PURCHASE_ACTION_WIDTH,):
        raise AssertionError("purchase feature width changed without a schema bump")
    return result


def purchase_feature_matrix(state: np.ndarray, candidates: tuple[PurchasePlan, ...]) -> np.ndarray:
    actions = np.stack([purchase_action_features(candidate) for candidate in candidates])
    return np.concatenate((np.broadcast_to(state, (len(candidates), len(state))), actions), axis=1)


def reserve_candidates(item: str, available: int, survival_floor: int = 0) -> tuple[int, ...]:
    floor = min(max(0, survival_floor if item == "WHEAT" else 0), available)
    return tuple(sorted({max(floor, round(available * fraction)) for fraction in RESERVE_FRACTIONS}))


def sale_action_features(
    state: np.ndarray,
    item: str,
    available: int,
    reserve: int,
    price_ratio: float,
) -> np.ndarray:
    item_row = [float(name == item) for name in PRODUCTS]
    action = np.asarray(
        [*item_row, available / 100.0, reserve / 100.0, reserve / max(1, available), price_ratio],
        dtype=np.float32,
    )
    if action.shape != (SALE_ACTION_WIDTH,):
        raise AssertionError("sale feature width changed without a schema bump")
    return np.concatenate((state, action))


def _choose_index(model: Any | None, matrix: np.ndarray, epsilon: float, rng: random.Random, fallback: int) -> int:
    if epsilon > 0.0 and rng.random() < epsilon:
        return rng.randrange(len(matrix))
    if model is None:
        return fallback
    predictions = np.asarray(model.predict(matrix), dtype=float)
    return int(np.argmax(predictions))


def choose_purchase(
    state: np.ndarray,
    candidates: tuple[PurchasePlan, ...],
    model: Any | None,
    *,
    epsilon: float,
    rng: random.Random,
    fallback: int = 0,
) -> tuple[PurchasePlan, int, np.ndarray]:
    matrix = purchase_feature_matrix(state, candidates)
    index = _choose_index(model, matrix, epsilon, rng, min(fallback, len(candidates) - 1))
    return candidates[index], index, matrix[index]


def choose_reserves(
    state: np.ndarray,
    shed: dict,
    prices: dict,
    model: Any | None,
    *,
    survival_wheat: int,
    epsilon: float,
    rng: random.Random,
) -> tuple[dict[str, int], list[tuple[str, int, np.ndarray]]]:
    selected: dict[str, int] = {}
    records: list[tuple[str, int, np.ndarray]] = []
    for item in PRODUCTS:
        available = max(0, int(shed.get(item, 0)))
        if available <= 0:
            selected[item] = 0
            continue
        base = (
            float(CROP_DATA[item]["base"])
            if item in CROP_DATA
            else next(
                (float(data["base"]) for data in ANIMAL_DATA.values() if data["product"] == item),
                100.0,
            )
        )
        ratio = float(prices.get(item, base)) / max(1.0, base)
        levels = reserve_candidates(item, available, survival_wheat)
        matrix = np.stack([sale_action_features(state, item, available, reserve, ratio) for reserve in levels])
        # The heuristic fallback retains survival wheat and otherwise sells only
        # above the base quote.  Learned checkpoints replace this choice.
        fallback = 0 if ratio >= 1.0 else len(levels) - 1
        index = _choose_index(model, matrix, epsilon, rng, fallback)
        selected[item] = levels[index]
        records.append((item, index, matrix[index]))
    return selected, records
