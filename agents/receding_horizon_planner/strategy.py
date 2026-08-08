"""Serializable tuning parameters for the receding-horizon planner."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class Strategy:
    max_hands: int = 4
    cash_reserve: int = 450
    movement_cost: float = 0.8
    deadline_weight: float = 5.0
    capacity_reserve: float = 0.18
    risk_discount: float = 0.9
    terminal_asset_discount: float = 0.7
    town_demand_weight: float = 0.7
    opponent_supply_weight: float = 0.6
    land_hurdle: float = 1.15
    animal_risk_premium: float = 1.0
    care_value_weight: float = 0.8
    fertilizer_value_weight: float = 0.8
    sell_price_ratio: float = 1.0
    liquidation_turns: int = 72

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        ranges: dict[str, tuple[float, float]] = {
            "max_hands": (1, 8),
            "cash_reserve": (0, 1500),
            "movement_cost": (0.1, 3.0),
            "deadline_weight": (1.0, 10.0),
            "capacity_reserve": (0.05, 0.4),
            "risk_discount": (0.75, 1.0),
            "terminal_asset_discount": (0.3, 1.0),
            "town_demand_weight": (0.0, 2.0),
            "opponent_supply_weight": (0.0, 2.0),
            "land_hurdle": (0.8, 2.0),
            "animal_risk_premium": (0.8, 1.5),
            "care_value_weight": (0.0, 2.0),
            "fertilizer_value_weight": (0.0, 2.0),
            "sell_price_ratio": (0.7, 1.4),
            "liquidation_turns": (24, 96),
        }
        for name, (low, high) in ranges.items():
            value = float(getattr(self, name))
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def strategy_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    def to_features(self, prefix: str = "") -> dict[str, Any]:
        return {f"{prefix}{key}": value for key, value in self.to_dict().items()}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Strategy:
        declared = {field.name: field for field in fields(cls)}
        unknown = set(values) - set(declared)
        if unknown:
            raise ValueError(f"unknown planner parameters: {sorted(unknown)}")
        defaults = cls()
        converted: dict[str, Any] = {}
        for name, value in values.items():
            current = getattr(defaults, name)
            converted[name] = int(value) if isinstance(current, int) else float(value)
        return cls(**converted)


def sample_strategy(rng: random.Random) -> Strategy:
    return Strategy(
        max_hands=rng.randint(1, 8),
        cash_reserve=rng.randint(0, 1500),
        movement_cost=round(rng.uniform(0.1, 3.0), 6),
        deadline_weight=round(rng.uniform(1.0, 10.0), 6),
        capacity_reserve=round(rng.uniform(0.05, 0.4), 6),
        risk_discount=round(rng.uniform(0.75, 1.0), 6),
        terminal_asset_discount=round(rng.uniform(0.3, 1.0), 6),
        town_demand_weight=round(rng.uniform(0.0, 2.0), 6),
        opponent_supply_weight=round(rng.uniform(0.0, 2.0), 6),
        land_hurdle=round(rng.uniform(0.8, 2.0), 6),
        animal_risk_premium=round(rng.uniform(0.8, 1.5), 6),
        care_value_weight=round(rng.uniform(0.0, 2.0), 6),
        fertilizer_value_weight=round(rng.uniform(0.0, 2.0), 6),
        sell_price_ratio=round(rng.uniform(0.7, 1.4), 6),
        liquidation_turns=rng.randint(24, 96),
    )


def generate_population(count: int, seed: int) -> list[Strategy]:
    rng = random.Random(seed)
    unique: dict[str, Strategy] = {}
    while len(unique) < count:
        strategy = sample_strategy(rng)
        unique[strategy.strategy_id] = strategy
    return list(unique.values())
