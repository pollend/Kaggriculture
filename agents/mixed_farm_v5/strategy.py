"""Serializable parameters for the configurable mixed-farm policy."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class Strategy:
    """Serializable search space for crops, livestock, land, and labor."""

    max_hands: int = 8
    hire_until_hour: int = 3
    cash_reserve: int = 600
    target_crop_tiles: int = 32
    target_goose: int = 1
    target_cow: int = 6
    target_sheep: int = 2
    max_quadrants: int = 4
    feed_reserve_days: int = 2

    # Version-three operating cadence. Defaults preserve the v2 policy when an
    # older profile is decoded.
    minimum_hands: int = 0
    feed_carry_target: int = 8
    sell_batch_size: int = 100
    land_expansion_day: int = 30

    # Learnable operating limits. These are policy choices, not game rules.
    crop_target_floor: int = 16
    crop_tiles_per_animal: float = 1.0
    opening_crop_tranche: int = 8
    animal_unlock_crop_tiles: int = 8
    opening_animal_batch: int = 4
    workload_per_hand: float = 7.0
    land_commitment_fraction: float = 0.8

    wheat_weight: float = 0.9
    carrot_weight: float = 0.5
    tomato_weight: float = 0.6
    strawberry_weight: float = 0.8
    melon_weight: float = 1.4
    goose_weight: float = 0.8
    cow_weight: float = 1.3
    sheep_weight: float = 1.0

    water_priority: float = 2.0
    emergency_multiplier: float = 8.0
    movement_cost: float = 0.8
    deadline_weight: float = 5.0
    town_demand_weight: float = 0.7
    opponent_supply_weight: float = 0.5
    land_hurdle: float = 1.15
    animal_risk_premium: float = 1.0
    care_value_weight: float = 0.8
    fertilizer_value_weight: float = 0.8
    sell_price_ratio: float = 1.0
    shed_pressure_fraction: float = 0.75
    liquidation_turns: int = 72

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        ranges: dict[str, tuple[float, float]] = {
            # Broad validity limits protect execution and serialization. The
            # narrower, shaped distributions in sample_strategy are the search
            # prior and may evolve independently of these legal limits.
            "max_hands": (0, 24),
            "hire_until_hour": (0, 23),
            "cash_reserve": (0, 10_000),
            "target_crop_tiles": (0, 100),
            "target_goose": (0, 50),
            "target_cow": (0, 50),
            "target_sheep": (0, 50),
            "max_quadrants": (1, 4),
            "feed_reserve_days": (0, 30),
            "minimum_hands": (0, 16),
            "feed_carry_target": (1, 24),
            "sell_batch_size": (1, 100),
            "land_expansion_day": (0, 30),
            "crop_target_floor": (0, 100),
            "crop_tiles_per_animal": (0.0, 4.0),
            "opening_crop_tranche": (0, 25),
            "animal_unlock_crop_tiles": (0, 100),
            "opening_animal_batch": (0, 25),
            "workload_per_hand": (1.0, 24.0),
            "land_commitment_fraction": (0.0, 1.0),
            "water_priority": (0.5, 3.0),
            "emergency_multiplier": (2.0, 12.0),
            "movement_cost": (0.1, 3.0),
            "deadline_weight": (1.0, 10.0),
            "town_demand_weight": (0.0, 2.0),
            "opponent_supply_weight": (0.0, 2.0),
            "land_hurdle": (0.8, 2.0),
            "animal_risk_premium": (0.8, 1.5),
            "care_value_weight": (0.0, 2.0),
            "fertilizer_value_weight": (0.0, 2.0),
            "sell_price_ratio": (0.7, 1.5),
            "shed_pressure_fraction": (0.25, 0.95),
            "liquidation_turns": (0, 720),
        }
        for item in ("wheat", "carrot", "tomato", "strawberry", "melon", "goose", "cow", "sheep"):
            ranges[f"{item}_weight"] = (0.0, 2.0)
        for name, (low, high) in ranges.items():
            value = float(getattr(self, name))
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        if sum(self.crop_weights.values()) <= 0:
            raise ValueError("at least one crop weight must be positive")
        if self.minimum_hands > self.max_hands:
            raise ValueError("minimum_hands must not exceed max_hands")

    @property
    def crop_weights(self) -> dict[str, float]:
        return {
            name: float(getattr(self, f"{name.lower()}_weight"))
            for name in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
        }

    @property
    def animal_weights(self) -> dict[str, float]:
        return {
            name: float(getattr(self, f"{name.lower()}_weight"))
            for name in ("GOOSE", "COW", "SHEEP")
        }

    @property
    def animal_targets(self) -> dict[str, int]:
        return {
            name: int(getattr(self, f"target_{name.lower()}"))
            for name in ("GOOSE", "COW", "SHEEP")
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_features(self, prefix: str = "") -> dict[str, Any]:
        values = self.to_dict()
        features = {f"{prefix}{name}": value for name, value in values.items()}
        animal_total = sum(self.animal_targets.values())
        productive_total = self.target_crop_tiles + animal_total
        capacity = self.max_quadrants * 25
        minimum_labor = max(1, self.minimum_hands)
        maximum_labor = max(1, self.max_hands)
        crop_total = sum(self.crop_weights.values())
        animal_weight_total = sum(self.animal_weights.values())
        derived = {
            "derived_target_animals": animal_total,
            "derived_target_productive_assets": productive_total,
            "derived_capacity_tiles": capacity,
            "derived_capacity_utilization": productive_total / max(1, capacity),
            "derived_animals_per_minimum_hand": animal_total / minimum_labor,
            "derived_animals_per_maximum_hand": animal_total / maximum_labor,
            "derived_crops_per_maximum_hand": self.target_crop_tiles / maximum_labor,
            "derived_opening_work_per_minimum_hand": (
                self.opening_crop_tranche + 4 * self.opening_animal_batch
            ) / minimum_labor,
            "derived_cash_reserve_ratio": self.cash_reserve / 3000.0,
            "derived_crop_weight_total": crop_total,
            "derived_animal_weight_total": animal_weight_total,
        }
        derived.update(
            {
                f"derived_{name.lower()}_share": weight / max(crop_total, 1e-9)
                for name, weight in self.crop_weights.items()
            }
        )
        derived.update(
            {
                f"derived_{name.lower()}_share": weight / max(animal_weight_total, 1e-9)
                for name, weight in self.animal_weights.items()
            }
        )
        features.update({f"{prefix}{name}": value for name, value in derived.items()})
        return features

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def strategy_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Strategy:
        declared = {field.name: field for field in fields(cls)}
        unknown = set(values) - set(declared)
        if unknown:
            raise ValueError(f"unknown mixed-farm parameters: {sorted(unknown)}")
        defaults = cls()
        converted: dict[str, Any] = {}
        for name, value in values.items():
            current = getattr(defaults, name)
            converted[name] = int(value) if isinstance(current, int) else float(value)
        return cls(**converted)


def sample_strategy(rng: random.Random) -> Strategy:
    weights = {
        f"{name}_weight": round(rng.uniform(0.0, 2.0), 6)
        for name in ("wheat", "carrot", "tomato", "strawberry", "melon", "goose", "cow", "sheep")
    }
    if not any(weights[f"{name}_weight"] for name in ("wheat", "carrot", "tomato", "strawberry", "melon")):
        weights["wheat_weight"] = 1.0
    minimum_hands = round(rng.triangular(0, 16, 6))
    return Strategy(
        max_hands=round(rng.triangular(minimum_hands, 24, max(minimum_hands, 10))),
        hire_until_hour=round(rng.triangular(0, 23, 5)),
        cash_reserve=round(rng.triangular(0, 5_000, 400)),
        target_crop_tiles=round(rng.triangular(0, 100, 48)),
        target_goose=round(rng.triangular(0, 30, 4)),
        target_cow=round(rng.triangular(0, 30, 8)),
        target_sheep=round(rng.triangular(0, 30, 5)),
        max_quadrants=rng.randint(1, 4),
        feed_reserve_days=round(rng.triangular(0, 12, 2)),
        minimum_hands=minimum_hands,
        feed_carry_target=round(rng.triangular(1, 24, 4)),
        sell_batch_size=round(rng.triangular(1, 100, 12)),
        land_expansion_day=round(rng.triangular(0, 30, 18)),
        crop_target_floor=round(rng.triangular(0, 60, 20)),
        crop_tiles_per_animal=round(rng.triangular(0.0, 4.0, 1.0), 6),
        opening_crop_tranche=round(rng.triangular(0, 25, 10)),
        animal_unlock_crop_tiles=round(rng.triangular(0, 40, 8)),
        opening_animal_batch=round(rng.triangular(0, 25, 6)),
        workload_per_hand=round(rng.triangular(1.0, 24.0, 7.0), 6),
        land_commitment_fraction=round(rng.triangular(0.0, 1.0, 0.7), 6),
        **weights,
        water_priority=round(rng.uniform(0.5, 3.0), 6),
        emergency_multiplier=round(rng.uniform(2.0, 12.0), 6),
        movement_cost=round(rng.uniform(0.1, 3.0), 6),
        deadline_weight=round(rng.uniform(1.0, 10.0), 6),
        town_demand_weight=round(rng.uniform(0.0, 2.0), 6),
        opponent_supply_weight=round(rng.uniform(0.0, 2.0), 6),
        land_hurdle=round(rng.uniform(0.8, 2.0), 6),
        animal_risk_premium=round(rng.uniform(0.8, 1.5), 6),
        care_value_weight=round(rng.uniform(0.0, 2.0), 6),
        fertilizer_value_weight=round(rng.uniform(0.0, 2.0), 6),
        sell_price_ratio=round(rng.uniform(0.7, 1.5), 6),
        shed_pressure_fraction=round(rng.uniform(0.25, 0.95), 6),
        liquidation_turns=round(rng.triangular(0, 180, 60)),
    )


def generate_population(count: int, seed: int) -> list[Strategy]:
    rng = random.Random(seed)
    unique: dict[str, Strategy] = {}
    while len(unique) < count:
        strategy = sample_strategy(rng)
        unique[strategy.strategy_id] = strategy
    return list(unique.values())
