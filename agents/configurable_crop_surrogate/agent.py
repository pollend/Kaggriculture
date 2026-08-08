"""Submission-safe configurable crop agent.

This module intentionally imports only the Python standard library so it can be
bundled with ``main.py`` without the experiment stack.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agents.configurable_crop_surrogate.strategy import (
    CROPS,
    CropSelection,
    HarvestPolicy,
    SellPolicy,
    Strategy,
    WorkerAssignment,
)

CROP_DATA: dict[str, dict[str, float | int | bool]] = {
    "WHEAT": {"seed": 10, "base": 25, "first": 2, "peak": 4, "yield": 4, "ongoing": False},
    "CARROT": {"seed": 20, "base": 35, "first": 2, "peak": 3, "yield": 3, "ongoing": False},
    "TOMATO": {"seed": 50, "base": 60, "first": 8, "peak": 11, "yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "base": 120, "first": 10, "peak": 16, "yield": 4, "ongoing": True},
    "MELON": {"seed": 80, "base": 250, "first": 10, "peak": 12, "yield": 6, "ongoing": False},
}

SHOP_DEMAND = {
    "BAKERY": {"WHEAT"},
    "PIZZA_SHOP": {"WHEAT", "TOMATO"},
    "BRUNCH_SPOT": {"WHEAT", "STRAWBERRY"},
    "ICE_CREAM_SHOP": {"WHEAT", "STRAWBERRY"},
    "PET_CAFE": {"CARROT"},
    "SMOOTHIE_SHOP": {"STRAWBERRY"},
    "FARMERS_MARKET": {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"},
}

MOVE_FOR_DELTA = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}


@dataclass(frozen=True, slots=True)
class Job:
    target: tuple[int, int]
    action: tuple[str, ...]
    value: float
    kind: str


@dataclass(slots=True)
class EpisodeState:
    """Memory that is valid only within one episode."""

    player: int | None = None
    last_step: int = -1
    unit_targets: dict[int, tuple[int, int]] = field(default_factory=dict)
    last_positions: dict[int, tuple[int, int]] = field(default_factory=dict)
    crop_plans: dict[tuple[int, int], str] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)

    def reset(self, player: int) -> None:
        self.player = player
        self.last_step = -1
        self.unit_targets.clear()
        self.last_positions.clear()
        self.crop_plans.clear()
        self.action_counts.clear()


def _get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step_toward(position: tuple[int, int], target: tuple[int, int]) -> list[str]:
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    # Horizontal-first is deterministic and gets workers off locked east-side spawn tiles quickly.
    delta = (1 if dx > 0 else -1, 0) if dx else (0, 1 if dy > 0 else -1)
    return [MOVE_FOR_DELTA[delta]]


def _shed_tiles(board_size: int) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _town_crop_counts(obs: Any) -> dict[str, int]:
    counts = {crop: 0 for crop in CROPS}
    town = _get(obs, "town", {}) or {}
    for shop in _get(town, "unlocked_shops", []) or []:
        for crop in SHOP_DEMAND.get(shop, ()):
            counts[crop] += 1
    return counts


def _opponent_crop_counts(obs: Any, player: int) -> dict[str, int]:
    counts = {crop: 0 for crop in CROPS}
    farms = _get(obs, "farms", []) or []
    for index, farm in enumerate(farms):
        if index == player:
            continue
        for row in farm.get("tiles", []):
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    crop = tile.get("crop")
                    if crop in counts:
                        counts[crop] += 1
    return counts


class ConfigurableAgent:
    """Deterministic crop scheduler driven by a :class:`Strategy`."""

    def __init__(self, strategy: Strategy, episode_steps: int = 720, turns_per_day: int = 24):
        self.strategy = strategy
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        self.state = EpisodeState()

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        farms = _get(obs, "farms", []) or []
        player = int(_get(obs, "player", 0))
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}

        farm = farms[player]
        private = _get(obs, "private", {}) or {}
        step = int(_get(obs, "step", 0))
        if step == 0 or self.state.player != player or step <= self.state.last_step:
            self.state.reset(player)
        day = int(_get(obs, "day", 0))
        hour = int(_get(obs, "hour", step % self.turns_per_day))
        episode_steps = int(_get(configuration, "episodeSteps", self.episode_steps))
        turns_per_day = int(_get(configuration, "turnsPerDay", self.turns_per_day))
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        endgame = step >= episode_steps - self.strategy.endgame_liquidation_turns

        positions = [tuple(farm["farmer"]), *(tuple(pos) for pos in farm.get("hands", []))]
        inventories = list(private.get("inventories", []))
        while len(inventories) < len(positions):
            inventories.append({})

        market = self._market_orders(obs, farm, private, step, hour, endgame)
        jobs = self._jobs(obs, farm, private, day, hour, endgame, inventories)
        unit_actions = self._assign(positions, inventories, jobs, len(farm["tiles"]), endgame)
        self.state.last_step = step
        self.state.last_positions = {index: position for index, position in enumerate(positions)}
        for action in unit_actions:
            op = action[0] if action else "PASS"
            self.state.action_counts[op] = self.state.action_counts.get(op, 0) + 1
        return {"farmer": unit_actions[0], "hands": unit_actions[1:], "market": market[:10]}

    def _crop_score(self, crop: str, obs: Any, player: int) -> float:
        data = CROP_DATA[crop]
        prices = (_get(obs, "market", {}) or {}).get("prices", {})
        price = float(prices.get(crop, data["base"]))
        gross = price * float(data["yield"]) - float(data["seed"])
        actions = 3.0 + float(data["peak"])  # plant + care days + harvest + rough travel
        days = float(data["peak"]) + 1.0
        if self.strategy.crop_selection == CropSelection.PROFIT_PER_DAY:
            economic = gross / days
        else:
            economic = gross / actions
        if self.strategy.crop_selection == CropSelection.MARKET_ADJUSTED:
            economic *= price / float(data["base"])

        score = max(0.01, economic) * self.strategy.crop_weights[crop]
        if self.strategy.use_town_forecast:
            score *= 1 + self.strategy.town_demand_weight * _town_crop_counts(obs)[crop] / 4
        if self.strategy.use_opponent_supply:
            supply = _opponent_crop_counts(obs, player)[crop]
            score /= 1 + self.strategy.opponent_supply_weight * supply / 25
        return score

    def _chosen_crops(self, obs: Any, player: int) -> list[str]:
        return sorted(CROPS, key=lambda crop: (-self._crop_score(crop, obs, player), crop))

    def _jobs(
        self,
        obs: Any,
        farm: dict,
        private: dict,
        day: int,
        hour: int,
        endgame: bool,
        inventories: list[dict],
    ) -> list[Job]:
        tiles = farm["tiles"]
        board_size = len(tiles)
        jobs: list[Job] = []
        plant_count = 0
        empty: list[tuple[int, int]] = []

        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                target = (x, y)
                if tile is None:
                    empty.append(target)
                    continue
                if not isinstance(tile, dict):
                    continue
                kind = tile.get("kind")
                if kind == "WEED":
                    jobs.append(Job(target, ("DIG",), 25.0, "DIG"))
                    continue
                if kind != "PLANT":
                    continue
                plant_count += 1
                crop = tile["crop"]
                data = CROP_DATA[crop]
                age = day - int(tile["planted_day"])
                watered = bool(tile.get("watered_today"))

                if not watered:
                    emergency = int(tile.get("consecutive_unwatered", 0)) >= 1 or hour >= 18
                    value = 100.0 * self.strategy.water_priority
                    if emergency:
                        value *= self.strategy.emergency_water_multiplier
                    jobs.append(Job(target, ("WATER",), value, "WATER"))
                    continue

                if self._should_harvest(tile, data, age, obs, endgame):
                    yield_value = float(tile.get("yield_units", 0)) * float(
                        (_get(obs, "market", {}) or {}).get("prices", {}).get(crop, data["base"])
                    )
                    jobs.append(Job(target, ("HARVEST",), 200.0 + yield_value, "HARVEST"))

        for index, inventory in enumerate(inventories):
            carried = sum(int(n) for n in inventory.values())
            if carried <= 0:
                continue
            # A separate uniquely keyed shed tile avoids assigning every carrier to one square.
            position = tuple(farm["farmer"] if index == 0 else farm.get("hands", [])[index - 1])
            target = min(_shed_tiles(board_size), key=lambda tile: (_distance(position, tile), tile))
            value = 350.0 + carried * 10
            if endgame:
                value += 10_000
            jobs.append(Job(target, ("DROP",), value, f"DROP:{index}"))

        remaining = max(0, self.strategy.target_planted_tiles - plant_count)
        seeds = dict(private.get("seeds", {}))
        # Planting needs at least one later action on the same day to water the new crop.
        if not endgame and hour < self.turns_per_day - 1 and remaining:
            crop_order = self._chosen_crops(obs, int(_get(obs, "player", 0)))
            empty.sort(key=lambda pos: (_distance(pos, _shed_tiles(board_size)[0]), pos))
            for target in empty:
                if remaining <= 0:
                    break
                crop = next((name for name in crop_order if seeds.get(name, 0) > 0), None)
                if crop is None:
                    break
                seeds[crop] -= 1
                remaining -= 1
                jobs.append(Job(target, ("PLANT", crop), 40.0 + self._crop_score(crop, obs, int(_get(obs, "player", 0))), "PLANT"))
        return jobs

    def _should_harvest(
        self, tile: dict, data: dict[str, float | int | bool], age: int, obs: Any, endgame: bool
    ) -> bool:
        units = int(tile.get("yield_units", 0))
        if units <= 0 or age < int(data["first"]):
            return False
        if endgame:
            return True
        ongoing = bool(data["ongoing"])
        if self.strategy.harvest_policy == HarvestPolicy.EARLIEST:
            return True
        if self.strategy.harvest_policy == HarvestPolicy.MAX_YIELD:
            return units >= int(data["yield"]) if ongoing else age >= int(data["peak"])
        crop = str(tile["crop"])
        price = float((_get(obs, "market", {}) or {}).get("prices", {}).get(crop, data["base"]))
        attractive = price >= float(data["base"]) * self.strategy.sell_price_ratio
        return attractive or units >= int(data["yield"]) or (not ongoing and age >= int(data["peak"]))

    def _assign(
        self,
        positions: list[tuple[int, int]],
        inventories: list[dict],
        jobs: list[Job],
        board_size: int,
        endgame: bool,
    ) -> list[list[str]]:
        actions = [["PASS"] for _ in positions]
        free_units = set(range(len(positions)))
        available = list(jobs)

        while free_units and available:
            pairs: list[tuple[float, int, int]] = []
            for unit in free_units:
                for job_index, job in enumerate(available):
                    # A DROP job created for one carrier must stay with that carrier.
                    if job.kind.startswith("DROP:") and job.kind != f"DROP:{unit}":
                        continue
                    distance = _distance(positions[unit], job.target)
                    if self.strategy.worker_assignment == WorkerAssignment.NEAREST_JOB:
                        rank = -distance + job.value * 1e-5
                    elif self.strategy.worker_assignment == WorkerAssignment.PRIORITY_FIRST:
                        rank = job.value - distance * 1e-5
                    else:
                        rank = job.value / (1 + self.strategy.movement_cost_weight * distance)
                    pairs.append((rank, unit, job_index))
            if not pairs:
                break
            _, unit, job_index = max(pairs, key=lambda item: (item[0], -item[1], -item[2]))
            job = available.pop(job_index)
            free_units.remove(unit)
            self.state.unit_targets[unit] = job.target
            if positions[unit] == job.target:
                actions[unit] = list(job.action)
            else:
                actions[unit] = _step_toward(positions[unit], job.target)

        if endgame:
            shed = _shed_tiles(board_size)
            for unit in free_units:
                if sum(int(n) for n in inventories[unit].values()) > 0:
                    target = min(shed, key=lambda tile: (_distance(positions[unit], tile), tile))
                    actions[unit] = ["DROP"] if positions[unit] == target else _step_toward(positions[unit], target)
        return actions

    def _market_orders(
        self, obs: Any, farm: dict, private: dict, step: int, hour: int, endgame: bool
    ) -> list[list[Any]]:
        orders: list[list[Any]] = []
        shed = private.get("shed", {})
        prices = (_get(obs, "market", {}) or {}).get("prices", {})
        shed_total = sum(int(n) for n in shed.values())

        for crop in CROPS:
            count = int(shed.get(crop, 0))
            if count <= 0:
                continue
            ratio = float(prices.get(crop, CROP_DATA[crop]["base"])) / float(CROP_DATA[crop]["base"])
            sell = endgame or self.strategy.sell_policy == SellPolicy.IMMEDIATE
            sell |= self.strategy.sell_policy == SellPolicy.PRICE_THRESHOLD and ratio >= self.strategy.sell_price_ratio
            sell |= self.strategy.sell_policy == SellPolicy.SHED_PRESSURE and (
                shed_total >= 100 * self.strategy.shed_pressure_fraction or ratio >= self.strategy.sell_price_ratio
            )
            if sell:
                orders.append(["SELL", crop, count])

        if endgame:
            return orders[:10]

        money = float(farm.get("money", 0))
        spendable = max(0.0, money - self.strategy.cash_reserve)
        hires = len(farm.get("hands", []))
        if hour <= self.strategy.hire_until_hour:
            fib_a, fib_b = 1, 1
            for _ in range(int(farm.get("hires_today", hires))):
                fib_a, fib_b = fib_b, fib_a + fib_b
            while hires < self.strategy.max_hands and spendable >= fib_a and len(orders) < 10:
                orders.append(["HIRE"])
                spendable -= fib_a
                fib_a, fib_b = fib_b, fib_a + fib_b
                hires += 1

        active = sum(
            1
            for row in farm.get("tiles", [])
            for tile in row
            if isinstance(tile, dict) and tile.get("kind") == "PLANT"
        )
        seeds = private.get("seeds", {})
        needed = max(0, self.strategy.target_planted_tiles - active - sum(int(n) for n in seeds.values()))
        crop_order = self._chosen_crops(obs, int(_get(obs, "player", 0)))
        while needed and len(orders) < 10:
            bought = False
            for crop in crop_order:
                cost = int(CROP_DATA[crop]["seed"])
                remaining_days = (self.episode_steps - step - 1) / self.turns_per_day
                if remaining_days < int(CROP_DATA[crop]["first"]) or spendable < cost:
                    continue
                orders.append(["BUY_SEED", crop, 1])
                spendable -= cost
                needed -= 1
                bought = True
                break
            if not bought:
                break
        return orders[:10]


def make_agent(strategy: Strategy | dict[str, Any] | None = None, **runtime: int) -> Callable[[Any], dict]:
    """Build a one-argument Kaggle agent callable."""

    if strategy is None:
        strategy = Strategy()
    elif isinstance(strategy, dict):
        values = dict(strategy)
        for key in ("episode_steps", "turns_per_day", "strategy_id"):
            if key in values:
                if key != "strategy_id":
                    runtime[key] = int(values[key])
                del values[key]
        strategy = Strategy.from_dict(values)
    return ConfigurableAgent(strategy, **runtime)


def strategy_from_environment() -> Strategy:
    raw = os.environ.get("KAGG_AGENT_CONFIG")
    if not raw:
        return Strategy()
    values = json.loads(raw)
    # Runner-only keys may share the environment variable with strategy fields.
    runtime_keys = {"episode_steps", "turns_per_day", "strategy_id"}
    return Strategy.from_dict({key: value for key, value in values.items() if key not in runtime_keys})
