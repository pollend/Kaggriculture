"""Serializable search space for the adaptive route planner."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class Strategy:
    """Route-level goals and turn-level scheduling weights.

    Defaults are centered on the useful shape of the Tetsutani route: expand
    twice, carry a large daily workforce, operate cows and sheep, and rotate
    wheat, strawberry, and melon production before a decisive closeout.
    """

    max_hands: int = 14
    cash_reserve: int = 250
    first_land_day: int = 7
    second_land_day: int = 11
    cow_target: int = 8
    sheep_target: int = 6
    target_crop_tiles: int = 32
    wheat_weight: float = 1.0
    strawberry_weight: float = 0.72
    melon_weight: float = 0.58
    feed_stock_days: float = 1.6
    water_priority: float = 4.0
    feed_priority: float = 5.0
    care_priority: float = 2.2
    fertilizer_priority: float = 1.1
    movement_cost: float = 0.65
    commitment_turns: int = 6
    sell_price_ratio: float = 0.92
    shed_pressure_fraction: float = 0.68
    liquidation_turns: int = 60

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        ranges: dict[str, tuple[float, float]] = {
            "max_hands": (4, 14),
            "cash_reserve": (0, 1500),
            "first_land_day": (4, 10),
            "second_land_day": (8, 16),
            "cow_target": (0, 8),
            "sheep_target": (0, 8),
            "target_crop_tiles": (8, 60),
            "wheat_weight": (0.1, 2.0),
            "strawberry_weight": (0.0, 2.0),
            "melon_weight": (0.0, 2.0),
            "feed_stock_days": (0.5, 3.0),
            "water_priority": (1.0, 8.0),
            "feed_priority": (1.0, 10.0),
            "care_priority": (0.0, 5.0),
            "fertilizer_priority": (0.0, 4.0),
            "movement_cost": (0.1, 2.0),
            "commitment_turns": (1, 12),
            "sell_price_ratio": (0.7, 1.4),
            "shed_pressure_fraction": (0.3, 0.9),
            "liquidation_turns": (24, 96),
        }
        for name, (low, high) in ranges.items():
            value = float(getattr(self, name))
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        if self.second_land_day <= self.first_land_day:
            raise ValueError("second_land_day must follow first_land_day")
        if self.cow_target + self.sheep_target > 16:
            raise ValueError("combined animal target cannot exceed 16")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_features(self, prefix: str = "") -> dict[str, Any]:
        return {f"{prefix}{key}": value for key, value in self.to_dict().items()}

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def strategy_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Strategy:
        declared = {field.name for field in fields(cls)}
        unknown = set(values) - declared
        if unknown:
            raise ValueError(f"unknown adaptive route parameters: {sorted(unknown)}")
        defaults = cls()
        converted: dict[str, Any] = {}
        for name, value in values.items():
            current = getattr(defaults, name)
            converted[name] = int(value) if isinstance(current, int) else float(value)
        return cls(**converted)


def sample_strategy(rng: random.Random) -> Strategy:
    """Mix route-centered exploration with broad valid exploration."""

    centered = rng.random() < 0.7

    def integer(center: int, low: int, high: int, radius: int) -> int:
        if centered:
            return min(high, max(low, center + rng.randint(-radius, radius)))
        return rng.randint(low, high)

    def real(center: float, low: float, high: float, radius: float) -> float:
        value = rng.uniform(center - radius, center + radius) if centered else rng.uniform(low, high)
        return round(min(high, max(low, value)), 6)

    first_land = integer(7, 4, 10, 2)
    second_land = integer(11, max(8, first_land + 1), 16, 2)
    cows = integer(8, 0, 8, 2)
    sheep = integer(6, 0, min(8, 16 - cows), 2)
    return Strategy(
        max_hands=integer(14, 4, 14, 3),
        cash_reserve=integer(250, 0, 1500, 250),
        first_land_day=first_land,
        second_land_day=second_land,
        cow_target=cows,
        sheep_target=sheep,
        target_crop_tiles=integer(32, 8, 60, 10),
        wheat_weight=real(1.0, 0.1, 2.0, 0.35),
        strawberry_weight=real(0.72, 0.0, 2.0, 0.35),
        melon_weight=real(0.58, 0.0, 2.0, 0.35),
        feed_stock_days=real(1.6, 0.5, 3.0, 0.5),
        water_priority=real(4.0, 1.0, 8.0, 1.5),
        feed_priority=real(5.0, 1.0, 10.0, 2.0),
        care_priority=real(2.2, 0.0, 5.0, 1.0),
        fertilizer_priority=real(1.1, 0.0, 4.0, 0.7),
        movement_cost=real(0.65, 0.1, 2.0, 0.35),
        commitment_turns=integer(6, 1, 12, 3),
        sell_price_ratio=real(0.92, 0.7, 1.4, 0.18),
        shed_pressure_fraction=real(0.68, 0.3, 0.9, 0.15),
        liquidation_turns=integer(60, 24, 96, 18),
    )

