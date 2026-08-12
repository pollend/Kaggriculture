"""Bounded economic preferences for the v5 mixed-farm controller."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields
from typing import Any

from agents.mixed_farm_v5.strategy import Strategy as LegacyStrategy

# `startingMoney` defaults to 3000; half of it is the most a sampled reserve may
# withhold and still leave a season's worth of working capital.
STARTING_MONEY = 3000
MAX_SAMPLED_CASH_RESERVE = STARTING_MONEY // 2


@dataclass(frozen=True, slots=True)
class Strategy:
    # Labor fields are preferences.  labor_v5 owns the inviolable ceilings,
    # reserve checks, Fibonacci pricing and useful-work requirements.
    max_hands: int = 10
    hire_until_hour: int = 8
    cash_reserve: int = 900
    target_hand_utilization: float = 0.72
    labor_roi_hurdle: float = 1.35
    max_hire_spend_fraction: float = 0.10

    target_crop_tiles: int = 32
    target_goose: int = 1
    target_cow: int = 6
    target_sheep: int = 2
    max_quadrants: int = 4
    feed_reserve_days: int = 2
    feed_carry_target: int = 8
    sell_batch_size: int = 100
    crop_target_floor: int = 16
    crop_tiles_per_animal: float = 1.0
    opening_crop_tranche: int = 8
    animal_unlock_crop_tiles: int = 8
    opening_animal_batch: int = 4
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
        ranges = {
            "max_hands": (0, 16), "hire_until_hour": (0, 12),
            "cash_reserve": (300, 5000), "target_hand_utilization": (0.55, 0.90),
            "labor_roi_hurdle": (1.0, 3.0), "max_hire_spend_fraction": (0.05, 0.20),
        }
        for name, (low, high) in ranges.items():
            value = float(getattr(self, name))
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
        # Reuse the mature validation for all non-labor economic preferences.
        self.to_legacy()

    def to_legacy(self) -> LegacyStrategy:
        values = self.to_dict()
        for name in (
            "target_hand_utilization", "labor_roi_hurdle", "max_hire_spend_fraction"
        ):
            values.pop(name)
        values.update(minimum_hands=0, land_expansion_day=30, workload_per_hand=7.0)
        return LegacyStrategy.from_dict(values)

    @property
    def crop_weights(self) -> dict[str, float]:
        return self.to_legacy().crop_weights

    @property
    def animal_weights(self) -> dict[str, float]:
        return self.to_legacy().animal_weights

    @property
    def animal_targets(self) -> dict[str, int]:
        return self.to_legacy().animal_targets

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_features(self, prefix: str = "") -> dict[str, Any]:
        values = self.to_dict()
        features = {f"{prefix}{name}": value for name, value in values.items()}
        animal_total = sum(self.animal_targets.values())
        # Exact cumulative Fibonacci cost at both selected and hard crew limits.
        fib = [1, 1]
        while len(fib) < 16:
            fib.append(fib[-1] + fib[-2])
        max_hands = max(1, self.max_hands)
        derived = {
            "derived_target_animals": animal_total,
            "derived_target_productive_assets": self.target_crop_tiles + animal_total,
            "derived_capacity_tiles": self.max_quadrants * 25,
            "derived_capacity_utilization": (self.target_crop_tiles + animal_total) / max(1, self.max_quadrants * 25),
            "derived_fibonacci_cost_at_max_hands": sum(fib[: self.max_hands]),
            "derived_fibonacci_cost_at_normal_ceiling": sum(fib[:10]),
            "derived_hands_capacity_per_day": max_hands * 24 * self.target_hand_utilization,
            "derived_target_work_per_hand": (self.target_crop_tiles + 3 * animal_total) / max_hands,
            "derived_cash_reserve_ratio": self.cash_reserve / 3000.0,
            "derived_hire_budget_at_start": 3000 * self.max_hire_spend_fraction,
        }
        features.update({f"{prefix}{name}": value for name, value in derived.items()})
        return features

    @property
    def strategy_id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:16]

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Strategy":
        values = dict(raw)
        # Explicit schema-v3/v4 migration.  The unsafe floors are discarded,
        # never translated into a softer version of the same behavior.
        values.pop("minimum_hands", None)
        values.pop("land_expansion_day", None)
        workload = values.pop("workload_per_hand", None)
        if "target_hand_utilization" not in values and workload is not None:
            values["target_hand_utilization"] = max(0.55, min(0.90, 7.0 / float(workload)))
        values.setdefault("labor_roi_hurdle", 1.35)
        values.setdefault("max_hire_spend_fraction", 0.10)
        values["max_hands"] = max(0, min(16, int(values.get("max_hands", cls.max_hands))))
        values["hire_until_hour"] = max(0, min(12, int(values.get("hire_until_hour", cls.hire_until_hour))))
        values["cash_reserve"] = max(300, min(5000, int(values.get("cash_reserve", cls.cash_reserve))))
        declared = {field.name for field in fields(cls)}
        unknown = set(values) - declared
        if unknown:
            raise ValueError(f"unknown mixed-farm v5 parameters: {sorted(unknown)}")
        defaults = cls()
        converted = {}
        for name, value in values.items():
            current = getattr(defaults, name)
            converted[name] = int(value) if isinstance(current, int) else float(value)
        return cls(**converted)


def sample_strategy(rng: random.Random) -> Strategy:
    legacy = LegacyStrategy.from_dict({
        **LegacyStrategy().to_dict(),
        "minimum_hands": 0,
        "max_hands": rng.randint(4, 16),
        "hire_until_hour": rng.randint(0, 12),
        "cash_reserve": rng.randint(300, 5000),
        "land_expansion_day": 30,
        "workload_per_hand": 7.0,
    })
    values = legacy.to_dict()
    for name in ("minimum_hands", "land_expansion_day", "workload_per_hand"):
        values.pop(name)
    values.update(
        target_hand_utilization=round(rng.uniform(0.55, 0.90), 6),
        labor_roi_hurdle=round(rng.uniform(1.0, 3.0), 6),
        max_hire_spend_fraction=round(rng.uniform(0.05, 0.20), 6),
    )
    # Diversify the mature economic fields through the existing bounded prior.
    donor = __import__("agents.mixed_farm_v5.strategy", fromlist=["sample_strategy"]).sample_strategy(rng)
    for name in values:
        if name in donor.to_dict() and rng.random() < 0.75:
            values[name] = donor.to_dict()[name]
    values["max_hands"] = min(16, int(values["max_hands"]))
    values["hire_until_hour"] = min(12, int(values["hire_until_hour"]))
    # A reserve near the opening bank leaves no working capital: the buy budget is
    # `money - cash_reserve`, so money settles at the reserve and the strategy can
    # never fund another seed. The validator still accepts up to 5000 so stored
    # revisions keep loading; only the prior is capped.
    values["cash_reserve"] = max(300, min(MAX_SAMPLED_CASH_RESERVE, int(values["cash_reserve"])))
    return Strategy.from_dict(values)
