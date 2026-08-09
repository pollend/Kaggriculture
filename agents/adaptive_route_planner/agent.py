"""Submission-safe adaptive route planner.

The season plan is intentionally coarse; every turn is scheduled from the live
farm so weeds, delayed work, changed prices, and different seats do not derail a
719-action script.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from agents.adaptive_route_planner.strategy import Strategy

CROPS = ("WHEAT", "STRAWBERRY", "MELON")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
CROP_DATA = {
    "WHEAT": {"seed": 10, "base": 25, "first": 2, "peak": 4, "yield": 4, "ongoing": False},
    "STRAWBERRY": {"seed": 100, "base": 120, "first": 10, "peak": 16, "yield": 4, "ongoing": True},
    "MELON": {"seed": 80, "base": 250, "first": 10, "peak": 12, "yield": 6, "ongoing": False},
}
ANIMAL_DATA = {
    "COW": {"cost": 400, "first": 8, "product": "MILK"},
    "SHEEP": {"cost": 500, "first": 6, "product": "WOOL"},
}
LAND_PRICES = (1000, 2000, 4000)
PRICE_PARAMS = {
    "WHEAT": (25, 400, "sqrt", 0.8, "log", 0.2),
    "CARROT": (35, 450, "log", 0.2, "sqrt", 0.7),
    "TOMATO": (60, 200, "linear", 0.4, "sqrt", 0.6),
    "STRAWBERRY": (120, 100, "sqrt", 0.7, "linear", 1.6),
    "MELON": (250, 300, "log", 0.2, "sq", 3.6),
    "EGG": (50, 332, "linear", 0.4, "log", 0.2),
    "MILK": (160, 122, "sqrt", 0.6, "linear", 1.6),
    "WOOL": (200, 105, "log", 0.2, "sq", 3.2),
    "FERTILIZER": (100, 200, "linear", 0.4, "linear", 0.4),
}
MOVE = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}


def get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def shed_tiles(size: int) -> tuple[tuple[int, int], ...]:
    half = size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def step_toward(position: tuple[int, int], target: tuple[int, int]) -> list[str]:
    dx, dy = target[0] - position[0], target[1] - position[1]
    if abs(dx) >= abs(dy) and dx:
        delta = (1 if dx > 0 else -1, 0)
    else:
        delta = (0, 1 if dy > 0 else -1)
    return [MOVE[delta]]


def shape(name: str, value: float) -> float:
    value = max(0.0, value)
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    return math.log1p(value)


def modeled_price(item: str, inventory: int) -> int:
    base, scale, below_fn, below_target, above_fn, above_target = PRICE_PARAMS[item]
    if inventory < 10_000:
        amplitude = below_target * base / shape(below_fn, scale)
        price = base + amplitude * shape(below_fn, 10_000 - inventory)
    else:
        amplitude = above_target * base / shape(above_fn, scale)
        price = base - amplitude * shape(above_fn, inventory - 10_000)
    return max(1, int(round(price)))


def hire_cost(number_already_hired: int) -> int:
    a, b = 1, 1
    for _ in range(number_already_hired):
        a, b = b, a + b
    return a


@dataclass(frozen=True, slots=True)
class Task:
    kind: str
    target: tuple[int, int]
    action: tuple[Any, ...]
    value: float
    owner: int | None = None

    @property
    def key(self) -> tuple[Any, ...]:
        return self.kind, self.target, self.action, self.owner


@dataclass(slots=True)
class EpisodeState:
    player: int | None = None
    last_step: int = -1
    commitments: dict[int, tuple[tuple[Any, ...], int]] = field(default_factory=dict)
    crop_cursor: int = 0

    def reset(self, player: int) -> None:
        self.player = player
        self.last_step = -1
        self.commitments.clear()
        self.crop_cursor = 0


class AdaptiveRouteAgent:
    def __init__(self, strategy: Strategy, episode_steps: int = 720, turns_per_day: int = 24):
        self.strategy = strategy
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        self.state = EpisodeState()

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        farms = get(obs, "farms", []) or []
        player = int(get(obs, "player", 0))
        if player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        farm = farms[player]
        private = get(obs, "private", {}) or {}
        step = int(get(obs, "step", 0))
        day = int(get(obs, "day", step // self.turns_per_day))
        hour = int(get(obs, "hour", step % self.turns_per_day))
        self.episode_steps = int(get(configuration, "episodeSteps", self.episode_steps))
        self.turns_per_day = int(get(configuration, "turnsPerDay", self.turns_per_day))
        if step == 0 or self.state.player != player or step <= self.state.last_step:
            self.state.reset(player)

        positions = [tuple(farm.get("farmer", (0, 0)))]
        positions.extend(tuple(item) for item in farm.get("hands", []) or [])
        inventories = list(private.get("inventories", []) or [])
        while len(inventories) < len(positions):
            inventories.append({})
        terminal = step >= self.episode_steps - self.strategy.liquidation_turns
        tasks = self._tasks(obs, farm, private, positions, inventories, day, hour, terminal)
        actions = self._schedule(tasks, positions, inventories, step)
        orders = self._market(obs, farm, private, day, hour, terminal)
        self.state.last_step = step
        return {"farmer": actions[0], "hands": actions[1:], "market": orders[:10]}

    def _counts(self, farm: dict, private: dict) -> tuple[dict[str, int], dict[str, int], int]:
        crops = {name: 0 for name in CROPS}
        animals = {name: 0 for name in ANIMAL_DATA}
        empty_pastures = 0
        for row in farm.get("tiles", []) or []:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                if tile.get("kind") == "PLANT" and tile.get("crop") in crops:
                    crops[str(tile["crop"])] += 1
                animal = tile.get("animal")
                if animal in animals:
                    animals[str(animal)] += 1
                elif tile.get("kind") == "PASTURE":
                    empty_pastures += 1
        shed = private.get("shed", {}) or {}
        for animal in animals:
            animals[animal] += int(shed.get(animal, 0))
            animals[animal] += sum(int((slot or {}).get(animal, 0)) for slot in private.get("inventories", []) or [])
        return crops, animals, empty_pastures

    def _crop_targets(self, obs: Any, terminal: bool) -> dict[str, int]:
        if terminal:
            return {crop: 0 for crop in CROPS}
        day = int(get(obs, "day", 0))
        remaining_days = (self.episode_steps - int(get(obs, "step", 0))) / self.turns_per_day
        weights = {
            "WHEAT": self.strategy.wheat_weight * (1.5 if day < self.strategy.first_land_day else 1.0),
            "STRAWBERRY": self.strategy.strawberry_weight if remaining_days >= 11 else 0.0,
            "MELON": self.strategy.melon_weight if remaining_days >= 11 else 0.0,
        }
        total = sum(weights.values()) or 1.0
        targets = {crop: int(self.strategy.target_crop_tiles * weight / total) for crop, weight in weights.items()}
        targets["WHEAT"] += self.strategy.target_crop_tiles - sum(targets.values())
        return targets

    def _tasks(
        self,
        obs: Any,
        farm: dict,
        private: dict,
        positions: list[tuple[int, int]],
        inventories: list[dict],
        day: int,
        hour: int,
        terminal: bool,
    ) -> list[Task]:
        tiles = farm.get("tiles", []) or []
        size = len(tiles)
        shed = shed_tiles(size)
        step = int(get(obs, "step", 0))
        hard_close = step >= self.episode_steps - 12
        tasks: list[Task] = []
        empty: list[tuple[int, int]] = []
        empty_pastures: list[tuple[int, int]] = []
        occupied_pastures = 0
        crops, animals, _ = self._counts(farm, private)

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
                    tasks.append(Task("DIG", target, ("DIG",), 10_000.0))
                elif kind == "PLANT":
                    crop = str(tile.get("crop"))
                    age = day - int(tile.get("planted_day", day))
                    if not tile.get("watered_today", False):
                        emergency = int(tile.get("consecutive_unwatered", 0)) >= 1 or hour >= 18
                        value = 1200.0 * self.strategy.water_priority * (3.0 if emergency else 1.0)
                        tasks.append(Task("WATER", target, ("WATER",), value))
                    units = int(tile.get("yield_units", 0))
                    data = CROP_DATA.get(crop)
                    if data and units > 0 and (terminal or age >= int(data["peak"]) or bool(data["ongoing"]) and units >= int(data["yield"])):
                        price = float((get(obs, "market", {}) or {}).get("prices", {}).get(crop, data["base"]))
                        tasks.append(Task("HARVEST", target, ("HARVEST",), 6500.0 + units * price))
                    fertilizer = int((private.get("shed", {}) or {}).get("FERTILIZER", 0))
                    if fertilizer and day >= max(1, int(data["peak"]) // 2) and int(tile.get("fertilized_until_day", -1)) < day:
                        tasks.append(Task("FERTILIZE", target, ("FERTILIZE",), 700.0 * self.strategy.fertilizer_priority))
                elif kind == "PASTURE":
                    animal = tile.get("animal")
                    if not animal:
                        empty_pastures.append(target)
                    else:
                        occupied_pastures += 1
                        if not tile.get("fed_today", False):
                            emergency = int(tile.get("consecutive_unfed", 0)) >= 1 or hour >= 18
                            value = 1500.0 * self.strategy.feed_priority * (3.0 if emergency else 1.0)
                            tasks.append(Task("FEED", target, ("FEED",), value))
                        if not tile.get("cared_today", False):
                            tasks.append(Task("CARE", target, ("CARE",), 800.0 * self.strategy.care_priority))
                        if tile.get("fertilizer_available", False):
                            tasks.append(Task("COLLECT_FERTILIZER", target, ("COLLECT_FERTILIZER",), 650.0 * self.strategy.fertilizer_priority))
                        if int(tile.get("yield_units", 0)) > 0:
                            tasks.append(Task("HARVEST_ANIMAL", target, ("HARVEST",), 2200.0))

        for owner, inventory in enumerate(inventories):
            position = positions[owner]
            carried = sum(int(value) for value in (inventory or {}).values())
            if carried:
                # Inventory actions at their destination outrank returning generic cargo.
                for animal in ANIMAL_DATA:
                    if int((inventory or {}).get(animal, 0)):
                        for target in empty_pastures:
                            tasks.append(Task(f"PLACE_{animal}", target, ("PLACE", animal, 1), 5000.0, owner))
                sellable = sum(
                    int((inventory or {}).get(item, 0))
                    for item in PRODUCTS
                    if item not in {"WHEAT", "FERTILIZER"}
                )
                sellable += max(
                    0,
                    int((inventory or {}).get("WHEAT", 0))
                    - sum(1 for task in tasks if task.kind == "FEED"),
                )
                if terminal:
                    target = min(shed, key=lambda item: (distance(position, item), item))
                    tasks.append(Task("DROP", target, ("DROP",), 20_000.0, owner))
                elif sellable and not any((inventory or {}).get(a, 0) for a in ANIMAL_DATA):
                    target = min(shed, key=lambda item: (distance(position, item), item))
                    tasks.append(Task("DROP", target, ("DROP",), 3000.0 + carried * 20, owner))

        if terminal:
            if hard_close:
                return [task for task in tasks if task.kind == "DROP"]
            return tasks

        # Shed pickup transactions are owner-specific and emerge from live inventory.
        shed_inventory = private.get("shed", {}) or {}
        hungry = sum(1 for task in tasks if task.kind == "FEED")
        wheat_carriers = sum(int((slot or {}).get("WHEAT", 0)) for slot in inventories)
        if hungry > wheat_carriers and int(shed_inventory.get("WHEAT", 0)):
            pickups = min(hungry - wheat_carriers, int(shed_inventory["WHEAT"]))
            for owner, (position, inventory) in enumerate(zip(positions, inventories)):
                if pickups <= 0:
                    break
                if sum(int(v) for v in (inventory or {}).values()) == 0:
                    target = min(shed, key=lambda item: (distance(position, item), item))
                    tasks.append(Task("PICKUP_WHEAT", target, ("PICKUP", "WHEAT", 1), 4500.0, owner))
                    pickups -= 1
        if any(task.kind == "FERTILIZE" for task in tasks) and int(shed_inventory.get("FERTILIZER", 0)):
            for owner, (position, inventory) in enumerate(zip(positions, inventories)):
                if sum(int(v) for v in (inventory or {}).values()) == 0:
                    target = min(shed, key=lambda item: (distance(position, item), item))
                    tasks.append(Task("PICKUP_FERTILIZER", target, ("PICKUP", "FERTILIZER", min(3, int(shed_inventory["FERTILIZER"]))), 1800.0, owner))
        for animal in ANIMAL_DATA:
            if int(shed_inventory.get(animal, 0)) and empty_pastures:
                pickups = min(int(shed_inventory[animal]), len(empty_pastures))
                for owner, (position, inventory) in enumerate(zip(positions, inventories)):
                    if pickups <= 0:
                        break
                    if sum(int(v) for v in (inventory or {}).values()) == 0:
                        target = min(shed, key=lambda item: (distance(position, item), item))
                        tasks.append(Task(f"PICKUP_{animal}", target, ("PICKUP", animal, 1), 4800.0, owner))
                        pickups -= 1

        desired_animals = self.strategy.cow_target + self.strategy.sheep_target
        existing_structures = occupied_pastures + len(empty_pastures)
        needed_structures = max(0, desired_animals - existing_structures)
        empty.sort(key=lambda target: (min(distance(target, item) for item in shed), target))
        for target in empty[:needed_structures]:
            tasks.append(Task("BUILD_PASTURE", target, ("BUILD_PASTURE",), 1800.0))

        targets = self._crop_targets(obs, False)
        seeds = dict(private.get("seeds", {}) or {})
        available_for_crops = empty[needed_structures:]
        if hour < self.turns_per_day - 1:
            for crop in CROPS:
                deficit = max(0, targets[crop] - crops[crop])
                quantity = min(deficit, int(seeds.get(crop, 0)))
                for _ in range(quantity):
                    if not available_for_crops:
                        break
                    target = available_for_crops.pop(0)
                    tasks.append(Task(f"PLANT_{crop}", target, ("PLANT", crop), 1000.0 + CROP_DATA[crop]["base"]))
        return tasks

    def _schedule(
        self,
        tasks: list[Task],
        positions: list[tuple[int, int]],
        inventories: list[dict],
        step: int,
    ) -> list[list[str]]:
        actions = [["PASS"] for _ in positions]
        available = list(tasks)
        free = set(range(len(positions)))
        by_key = {task.key: task for task in available}

        # Honor a still-valid short commitment first.
        for owner, (key, expires) in list(self.state.commitments.items()):
            task = by_key.get(key)
            if owner not in free or expires < step or task is None or task.owner not in (None, owner):
                self.state.commitments.pop(owner, None)
                continue
            actions[owner] = list(task.action) if positions[owner] == task.target else step_toward(positions[owner], task.target)
            free.remove(owner)
            available.remove(task)

        while free and available:
            candidates: list[tuple[float, int, int]] = []
            for owner in free:
                for index, task in enumerate(available):
                    if task.owner not in (None, owner):
                        continue
                    # FEED and FERTILIZE require matching carried inventory.
                    if task.kind == "FEED" and int((inventories[owner] or {}).get("WHEAT", 0)) <= 0:
                        continue
                    if task.kind == "FERTILIZE" and int((inventories[owner] or {}).get("FERTILIZER", 0)) <= 0:
                        continue
                    travel = distance(positions[owner], task.target)
                    score = task.value / (1.0 + self.strategy.movement_cost * travel)
                    candidates.append((score, -owner, -index))
            if not candidates:
                break
            _, neg_owner, neg_index = max(candidates)
            owner, index = -neg_owner, -neg_index
            task = available.pop(index)
            free.remove(owner)
            actions[owner] = list(task.action) if positions[owner] == task.target else step_toward(positions[owner], task.target)
            self.state.commitments[owner] = (task.key, step + self.strategy.commitment_turns)
        return actions

    def _market(self, obs: Any, farm: dict, private: dict, day: int, hour: int, terminal: bool) -> list[list[Any]]:
        shed = private.get("shed", {}) or {}
        market = get(obs, "market", {}) or {}
        prices = market.get("prices", {}) or {}
        inventories = market.get("inventory", {}) or {}
        shed_total = sum(int(v) for v in shed.values())
        sales: list[tuple[float, list[Any]]] = []
        animal_count = sum(
            1 for row in farm.get("tiles", []) or [] for tile in row
            if isinstance(tile, dict) and tile.get("animal") in ANIMAL_DATA
        )
        wheat_reserve = 0 if terminal else math.ceil(animal_count * self.strategy.feed_stock_days)
        for item in PRODUCTS:
            quantity = int(shed.get(item, 0)) - (wheat_reserve if item == "WHEAT" else 0)
            if quantity <= 0:
                continue
            ratio = float(prices.get(item, PRICE_PARAMS[item][0])) / PRICE_PARAMS[item][0]
            if not terminal and shed_total < 100 * self.strategy.shed_pressure_fraction and ratio < self.strategy.sell_price_ratio:
                continue
            inventory = int(inventories.get(item, 10_000))
            impact = quantity * max(0, float(prices.get(item, modeled_price(item, inventory))) - modeled_price(item, inventory + quantity))
            sales.append((impact, ["SELL", item, quantity]))
        orders = [order for _, order in sorted(sales, key=lambda row: (-row[0], row[1][1]))]
        if terminal:
            return orders[:10]

        budget = max(0, int(farm.get("money", 0)) - self.strategy.cash_reserve)
        unlocked = len(farm.get("unlocked_quadrants", []) or [])
        land_due = unlocked == 1 and day >= self.strategy.first_land_day or unlocked == 2 and day >= self.strategy.second_land_day
        if land_due and unlocked - 1 < len(LAND_PRICES) and budget >= LAND_PRICES[unlocked - 1] and len(orders) < 10:
            orders.append(["BUY_LAND"])
            budget -= LAND_PRICES[unlocked - 1]

        crops, animals, _ = self._counts(farm, private)
        hands = len(farm.get("hands", []) or [])
        active_assets = sum(crops.values()) + sum(animals.values())
        desired_hands = min(self.strategy.max_hands, max(2, math.ceil(active_assets / 3)))
        hires_today = int(farm.get("hires_today", hands))
        while hands < desired_hands and hour <= 3 and len(orders) < 10:
            cost = hire_cost(hires_today)
            if cost > budget:
                break
            orders.append(["HIRE"])
            budget -= cost
            hands += 1
            hires_today += 1

        carried_wheat = sum(int((slot or {}).get("WHEAT", 0)) for slot in private.get("inventories", []) or [])
        wheat_shortfall = max(0, wheat_reserve - int(shed.get("WHEAT", 0)) - carried_wheat)
        if wheat_shortfall and len(orders) < 10:
            price = max(1, int(prices.get("WHEAT", 25)))
            quantity = min(wheat_shortfall, budget // price)
            if quantity:
                orders.append(["BUY_PRODUCT", "WHEAT", quantity])
                budget -= quantity * price

        remaining_days = (self.episode_steps - int(get(obs, "step", 0))) / self.turns_per_day
        for animal, target in (("COW", self.strategy.cow_target), ("SHEEP", self.strategy.sheep_target)):
            if remaining_days < int(ANIMAL_DATA[animal]["first"]):
                continue
            deficit = max(0, target - animals[animal])
            cost = int(ANIMAL_DATA[animal]["cost"])
            quantity = min(deficit, budget // cost)
            if quantity and len(orders) < 10:
                orders.append(["BUY_ANIMAL", animal, quantity])
                budget -= quantity * cost

        targets = self._crop_targets(obs, False)
        seeds = private.get("seeds", {}) or {}
        for crop in CROPS:
            if remaining_days < int(CROP_DATA[crop]["first"]):
                continue
            deficit = max(0, targets[crop] - crops[crop] - int(seeds.get(crop, 0)))
            cost = int(CROP_DATA[crop]["seed"])
            quantity = min(deficit, budget // cost)
            if quantity and len(orders) < 10:
                orders.append(["BUY_SEED", crop, quantity])
                budget -= quantity * cost
        return orders[:10]


def make_agent(
    strategy: Strategy | dict[str, Any] | None = None,
    *,
    episode_steps: int | str = 720,
    turns_per_day: int | str = 24,
    **parameters: Any,
) -> AdaptiveRouteAgent:
    if strategy is not None and parameters:
        raise ValueError("pass either a Strategy or keyword parameters, not both")
    if isinstance(strategy, Strategy):
        selected = strategy
    elif isinstance(strategy, dict):
        values = dict(strategy)
        episode_steps = values.pop("episode_steps", episode_steps)
        turns_per_day = values.pop("turns_per_day", turns_per_day)
        selected = Strategy.from_dict(values)
    else:
        selected = Strategy.from_dict(parameters) if parameters else Strategy()
    return AdaptiveRouteAgent(selected, int(episode_steps), int(turns_per_day))


def strategy_from_environment() -> tuple[Strategy, int, int]:
    raw = os.environ.get("KAGG_AGENT_CONFIG")
    if not raw:
        return Strategy(), 720, 24
    values = json.loads(raw)
    episode_steps = int(values.pop("episode_steps", 720))
    turns_per_day = int(values.pop("turns_per_day", 24))
    values.pop("strategy_id", None)
    return Strategy.from_dict(values), episode_steps, turns_per_day
