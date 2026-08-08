"""Serializable strategy parameters for the configurable crop agent."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import Any

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")


class CropSelection(StrEnum):
    PROFIT_PER_ACTION = "PROFIT_PER_ACTION"
    PROFIT_PER_DAY = "PROFIT_PER_DAY"
    MARKET_ADJUSTED = "MARKET_ADJUSTED"


class HarvestPolicy(StrEnum):
    EARLIEST = "EARLIEST"
    MAX_YIELD = "MAX_YIELD"
    PRICE_AWARE = "PRICE_AWARE"


class SellPolicy(StrEnum):
    IMMEDIATE = "IMMEDIATE"
    PRICE_THRESHOLD = "PRICE_THRESHOLD"
    SHED_PRESSURE = "SHED_PRESSURE"


class WorkerAssignment(StrEnum):
    NEAREST_JOB = "NEAREST_JOB"
    PRIORITY_FIRST = "PRIORITY_FIRST"
    VALUE_PER_DISTANCE = "VALUE_PER_DISTANCE"


@dataclass(frozen=True, slots=True)
class Strategy:
    """All tunable behavior for the v1 crop-only agent."""

    max_hands: int = 3
    hire_until_hour: int = 2
    cash_reserve: int = 300
    target_planted_tiles: int = 12

    wheat_weight: float = 1.0
    carrot_weight: float = 1.0
    tomato_weight: float = 0.6
    strawberry_weight: float = 0.4
    melon_weight: float = 0.5
    crop_selection: CropSelection = CropSelection.MARKET_ADJUSTED

    water_priority: float = 2.0
    emergency_water_multiplier: float = 8.0
    harvest_policy: HarvestPolicy = HarvestPolicy.MAX_YIELD

    sell_policy: SellPolicy = SellPolicy.IMMEDIATE
    sell_price_ratio: float = 1.0
    shed_pressure_fraction: float = 0.65

    worker_assignment: WorkerAssignment = WorkerAssignment.VALUE_PER_DISTANCE
    movement_cost_weight: float = 0.75

    use_town_forecast: bool = True
    town_demand_weight: float = 0.5
    use_opponent_supply: bool = True
    opponent_supply_weight: float = 0.3
    endgame_liquidation_turns: int = 48

    def __post_init__(self) -> None:
        enum_fields = {
            "crop_selection": CropSelection,
            "harvest_policy": HarvestPolicy,
            "sell_policy": SellPolicy,
            "worker_assignment": WorkerAssignment,
        }
        for name, enum_type in enum_fields.items():
            value = getattr(self, name)
            if not isinstance(value, enum_type):
                object.__setattr__(self, name, enum_type(value))
        self.validate()

    def validate(self) -> None:
        ranges: dict[str, tuple[float, float]] = {
            "max_hands": (0, 8),
            "hire_until_hour": (0, 6),
            "cash_reserve": (0, 1500),
            "target_planted_tiles": (4, 25),
            "water_priority": (0.5, 3.0),
            "emergency_water_multiplier": (2.0, 12.0),
            "sell_price_ratio": (0.7, 1.5),
            "shed_pressure_fraction": (0.25, 0.9),
            "movement_cost_weight": (0.0, 2.0),
            "town_demand_weight": (0.0, 2.0),
            "opponent_supply_weight": (0.0, 2.0),
            "endgame_liquidation_turns": (12, 72),
        }
        for crop in CROPS:
            ranges[f"{crop.lower()}_weight"] = (0.0, 2.0)
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        if sum(self.crop_weights.values()) <= 0:
            raise ValueError("at least one crop weight must be positive")

    @property
    def crop_weights(self) -> dict[str, float]:
        return {crop: float(getattr(self, f"{crop.lower()}_weight")) for crop in CROPS}

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, StrEnum):
                values[key] = value.value
        return values

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def strategy_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    def to_features(self, prefix: str = "") -> dict[str, Any]:
        return {f"{prefix}{key}": value for key, value in self.to_dict().items()}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Strategy:
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown strategy parameters: {sorted(unknown)}")
        return cls(**values)


def sample_strategy(rng: random.Random) -> Strategy:
    """Draw one strategy from the versioned v1 search space."""

    weights = {f"{crop.lower()}_weight": round(rng.uniform(0, 2), 6) for crop in CROPS}
    if not any(weights.values()):
        weights["wheat_weight"] = 1.0
    return Strategy(
        max_hands=rng.randint(0, 8),
        hire_until_hour=rng.randint(0, 6),
        cash_reserve=rng.randint(0, 1500),
        target_planted_tiles=rng.randint(4, 25),
        **weights,
        crop_selection=rng.choice(list(CropSelection)),
        water_priority=round(rng.uniform(0.5, 3.0), 6),
        emergency_water_multiplier=round(rng.uniform(2.0, 12.0), 6),
        harvest_policy=rng.choice(list(HarvestPolicy)),
        sell_policy=rng.choice(list(SellPolicy)),
        sell_price_ratio=round(rng.uniform(0.7, 1.5), 6),
        shed_pressure_fraction=round(rng.uniform(0.25, 0.9), 6),
        worker_assignment=rng.choice(list(WorkerAssignment)),
        movement_cost_weight=round(rng.uniform(0.0, 2.0), 6),
        use_town_forecast=bool(rng.getrandbits(1)),
        town_demand_weight=round(rng.uniform(0.0, 2.0), 6),
        use_opponent_supply=bool(rng.getrandbits(1)),
        opponent_supply_weight=round(rng.uniform(0.0, 2.0), 6),
        endgame_liquidation_turns=rng.randint(12, 72),
    )


def generate_population(count: int, seed: int) -> list[Strategy]:
    rng = random.Random(seed)
    unique: dict[str, Strategy] = {}
    while len(unique) < count:
        strategy = sample_strategy(rng)
        unique[strategy.strategy_id] = strategy
    return list(unique.values())
