"""Deadline-aware task construction and joint unit assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.receding_horizon_planner.planner import Plan
from agents.receding_horizon_planner.rules import (
    ANIMAL_DATA,
    ANIMAL_ORDER,
    CROP_DATA,
    CROPS,
    PRODUCTS,
    distance,
    get,
    shed_tiles,
    step_toward,
    unlocked_empty_tiles,
)
from agents.receding_horizon_planner.strategy import Strategy

MAX_COMMIT_TURNS = 20
IMPOSSIBLE = -1e12


@dataclass(frozen=True, slots=True)
class Task:
    kind: str
    target: tuple[int, int]
    action: tuple[Any, ...]
    value: float
    deadline: int
    owner: int | None = None
    required_item: str | None = None
    reason: str = ""

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, self.target, self.action, self.owner, self.required_item)


def _prices(obs: Any) -> dict[str, float]:
    market = get(obs, "market", {}) or {}
    raw = market.get("prices", {}) or {}
    defaults = {crop: float(CROP_DATA[crop]["base"]) for crop in CROPS}
    defaults.update({str(data["product"]): float(data["base"]) for data in ANIMAL_DATA.values()})
    defaults["FERTILIZER"] = 100.0
    return {item: float(raw.get(item, defaults[item])) for item in PRODUCTS}


def _nearest_shed(position: tuple[int, int], board_size: int) -> tuple[int, int]:
    return min(shed_tiles(board_size), key=lambda tile: (distance(position, tile), tile))


def _inventory_count(inventories: list[dict], item: str) -> int:
    return sum(int((slot or {}).get(item, 0)) for slot in inventories)


def _task(
    kind: str,
    target: tuple[int, int],
    action: tuple[Any, ...],
    value: float,
    deadline: int,
    *,
    owner: int | None = None,
    required_item: str | None = None,
    reason: str = "",
) -> Task:
    return Task(kind, target, action, value, deadline, owner, required_item, reason)


def enumerate_tasks(
    obs: Any,
    farm: dict,
    private: dict,
    plan: Plan,
    positions: list[tuple[int, int]],
    inventories: list[dict],
    strategy: Strategy,
    episode_steps: int = 720,
    turns_per_day: int = 24,
) -> list[Task]:
    """Build the currently executable frontier of the macro plan."""

    step = int(get(obs, "step", 0))
    day = int(get(obs, "day", 0))
    hour = int(get(obs, "hour", step % max(1, turns_per_day)))
    end_of_day = min(episode_steps - 1, (day + 1) * turns_per_day - 1)
    prices = _prices(obs)
    tiles = farm.get("tiles") or []
    board_size = len(tiles) or 10
    shed = private.get("shed", {}) or {}
    tasks: list[Task] = []
    crop_counts = {crop: 0 for crop in CROPS}
    animal_counts = {animal: 0 for animal in ANIMAL_ORDER}
    empty_structures: dict[str, list[tuple[int, int]]] = {"COOP": [], "PASTURE": []}
    hungry: list[tuple[tuple[int, int], dict]] = []
    fertilizable: list[tuple[tuple[int, int], dict]] = []

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            target = (x, y)
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "WEED":
                tasks.append(_task("DIG", target, ("DIG",), 25.0, episode_steps - 1, reason="weed"))
                continue
            if kind == "PLANT":
                crop = str(tile.get("crop", ""))
                if crop not in CROP_DATA:
                    continue
                crop_counts[crop] += 1
                data = CROP_DATA[crop]
                age = day - int(tile.get("planted_day", day))
                emergency = int(tile.get("consecutive_unwatered", 0)) >= 1
                if not tile.get("watered_today"):
                    value = 12_000.0 if emergency else max(35.0, prices[crop] * 0.8)
                    tasks.append(
                        _task(
                            "WATER", target, ("WATER",), value, end_of_day, reason="survival" if emergency else "yield"
                        )
                    )
                if int(tile.get("yield_units", 0)) > 0 and age >= int(data["first"]):
                    value = int(tile.get("yield_units", 0)) * prices[crop]
                    tasks.append(_task("HARVEST", target, ("HARVEST",), value, episode_steps - 1, reason="crop_yield"))
                if int(tile.get("fertilized_until_day", -1)) < day:
                    fertilizable.append((target, tile))
                continue

            animal = tile.get("animal")
            if animal in ANIMAL_DATA:
                animal = str(animal)
                animal_counts[animal] += 1
                data = ANIMAL_DATA[animal]
                emergency = int(tile.get("consecutive_unfed", 0)) >= 1
                if not tile.get("fed_today"):
                    hungry.append((target, tile))
                    tasks.append(
                        _task(
                            "FEED",
                            target,
                            ("FEED",),
                            14_000.0 if emergency else prices[str(data["product"])] * 1.4,
                            end_of_day,
                            required_item="WHEAT",
                            reason="animal_survival" if emergency else "animal_feed",
                        )
                    )
                if not tile.get("cared_today"):
                    tasks.append(
                        _task(
                            "CARE",
                            target,
                            ("CARE",),
                            prices[str(data["product"])] * strategy.care_value_weight,
                            end_of_day,
                            reason="care_bonus",
                        )
                    )
                if int(tile.get("yield_units", 0)) > 0:
                    tasks.append(
                        _task(
                            "HARVEST_ANIMAL",
                            target,
                            ("HARVEST",),
                            int(tile.get("yield_units", 0)) * prices[str(data["product"])],
                            episode_steps - 1,
                            reason="animal_yield",
                        )
                    )
                if tile.get("fertilizer_available"):
                    tasks.append(
                        _task(
                            "COLLECT_FERTILIZER",
                            target,
                            ("COLLECT_FERTILIZER",),
                            prices["FERTILIZER"] * strategy.fertilizer_value_weight,
                            end_of_day,
                            reason="fertilizer",
                        )
                    )
            elif kind in empty_structures:
                empty_structures[str(kind)].append(target)

    # Feed logistics: the closest suitable unit picks up enough wheat for the
    # current hungry set, then later turns expose the FEED frontier.
    carried_wheat = _inventory_count(inventories, "WHEAT")
    wheat_needed = max(0, len(hungry) - carried_wheat)
    if wheat_needed and int(shed.get("WHEAT", 0)) > 0 and positions:
        eligible = [
            (distance(position, _nearest_shed(position, board_size)), unit, position)
            for unit, position in enumerate(positions)
            if sum(int(value) for value in (inventories[unit] or {}).values()) < 20
        ]
        if eligible:
            _, owner, position = min(eligible)
            amount = min(wheat_needed, int(shed.get("WHEAT", 0)), 8)
            tasks.append(
                _task(
                    "PICKUP_WHEAT",
                    _nearest_shed(position, board_size),
                    ("PICKUP", "WHEAT", amount),
                    8_000.0 if any(int(tile.get("consecutive_unfed", 0)) >= 1 for _, tile in hungry) else 500.0,
                    end_of_day,
                    owner=owner,
                    reason="feed_logistics",
                )
            )

    # Fertilizer is useful only when the acting unit actually carries it.
    carried_fertilizer = _inventory_count(inventories, "FERTILIZER")
    for target, tile in fertilizable:
        crop = str(tile.get("crop", "WHEAT"))
        tasks.append(
            _task(
                "FERTILIZE",
                target,
                ("FERTILIZE",),
                prices[crop] * strategy.fertilizer_value_weight,
                end_of_day,
                required_item="FERTILIZER",
                reason="crop_bonus",
            )
        )
    if fertilizable and carried_fertilizer == 0 and int(shed.get("FERTILIZER", 0)) > 0 and positions:
        owner, position = min(
            enumerate(positions), key=lambda pair: distance(pair[1], _nearest_shed(pair[1], board_size))
        )
        tasks.append(
            _task(
                "PICKUP_FERTILIZER",
                _nearest_shed(position, board_size),
                ("PICKUP", "FERTILIZER", 1),
                160.0 * strategy.fertilizer_value_weight,
                end_of_day,
                owner=owner,
                reason="fertilizer_logistics",
            )
        )

    empties = sorted(
        unlocked_empty_tiles(farm),
        key=lambda target: (min((distance(position, target) for position in positions), default=0), target),
    )

    # Structures and animals are a two-stage frontier: build first, then pick up
    # and place the matching purchased animal on later turns.
    reserved_empty: set[tuple[int, int]] = set()
    desired_structures = {
        "COOP": plan.target_animals.get("GOOSE", 0),
        "PASTURE": plan.target_animals.get("COW", 0) + plan.target_animals.get("SHEEP", 0),
    }
    existing_structures = {
        "COOP": animal_counts["GOOSE"] + len(empty_structures["COOP"]),
        "PASTURE": animal_counts["COW"] + animal_counts["SHEEP"] + len(empty_structures["PASTURE"]),
    }
    for structure in ("COOP", "PASTURE"):
        deficit = max(0, desired_structures[structure] - existing_structures[structure])
        action = "BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"
        for target in [tile for tile in empties if tile not in reserved_empty][:deficit]:
            reserved_empty.add(target)
            tasks.append(_task(action, target, (action,), 350.0, end_of_day, reason="planned_structure"))

    available_animals = {
        animal: int(shed.get(animal, 0)) + _inventory_count(inventories, animal) for animal in ANIMAL_ORDER
    }
    structure_slots = {"COOP": list(empty_structures["COOP"]), "PASTURE": list(empty_structures["PASTURE"])}
    for animal in ANIMAL_ORDER:
        deficit = max(0, plan.target_animals.get(animal, 0) - animal_counts[animal])
        structure = str(ANIMAL_DATA[animal]["structure"])
        slots = structure_slots[structure][:deficit]
        structure_slots[structure] = structure_slots[structure][len(slots) :]
        for target in slots:
            for owner, inventory in enumerate(inventories):
                if int((inventory or {}).get(animal, 0)) > 0:
                    tasks.append(
                        _task(
                            "PLACE_ANIMAL",
                            target,
                            ("PLACE", animal, 1),
                            600.0,
                            end_of_day,
                            owner=owner,
                            required_item=animal,
                            reason="planned_animal",
                        )
                    )
        if slots and available_animals[animal] and _inventory_count(inventories, animal) == 0 and positions:
            owner, position = min(
                enumerate(positions), key=lambda pair: distance(pair[1], _nearest_shed(pair[1], board_size))
            )
            tasks.append(
                _task(
                    "PICKUP_ANIMAL",
                    _nearest_shed(position, board_size),
                    ("PICKUP", animal, 1),
                    650.0,
                    end_of_day,
                    owner=owner,
                    reason="animal_logistics",
                )
            )

    # Plant only the portfolio deficit and reserve structure tiles first.
    plant_tiles = [tile for tile in empties if tile not in reserved_empty]
    seeds = private.get("seeds", {}) or {}
    cursor = 0
    for crop in sorted(CROPS, key=lambda name: (-prices[name] * CROP_DATA[name]["yield"], name)):
        if hour >= turns_per_day - 1:
            continue
        remaining_days = (episode_steps - step - 1) / max(1, turns_per_day)
        if remaining_days < int(CROP_DATA[crop]["first"]):
            continue
        deficit = max(0, plan.target_crops.get(crop, 0) - crop_counts[crop])
        count = min(deficit, int(seeds.get(crop, 0)), len(plant_tiles) - cursor)
        for target in plant_tiles[cursor : cursor + count]:
            value = prices[crop] * float(CROP_DATA[crop]["yield"]) - float(CROP_DATA[crop]["seed"])
            tasks.append(_task("PLANT", target, ("PLANT", crop), value, end_of_day - 1, reason="planned_crop"))
        cursor += count

    # Carried sale products should reach the shed. Preserve wheat, fertilizer,
    # and animals while there is a matching planned use.
    terminal = step >= episode_steps - strategy.liquidation_turns
    useful = {*ANIMAL_ORDER} if terminal else {"WHEAT", "FERTILIZER", *ANIMAL_ORDER}
    for owner, inventory in enumerate(inventories):
        sale_value = sum(
            int(count) * prices.get(item, 0.0) for item, count in (inventory or {}).items() if item not in useful
        )
        if sale_value <= 0:
            continue
        tasks.append(
            _task(
                "DROP",
                _nearest_shed(positions[owner], board_size),
                ("DROP",),
                sale_value,
                episode_steps - 1,
                owner=owner,
                reason="liquid_inventory",
            )
        )

    # Deterministic order is important for reproducibility and stable task keys.
    tasks.sort(key=lambda item: (item.deadline, -item.value, item.kind, item.target, item.owner or -1))
    return tasks


def _allowed(task: Task, unit: int, inventory: dict) -> bool:
    if task.owner is not None and task.owner != unit:
        return False
    return task.required_item is None or int((inventory or {}).get(task.required_item, 0)) > 0


def task_utility(
    task: Task, unit: int, position: tuple[int, int], inventory: dict, step: int, strategy: Strategy
) -> float:
    if not _allowed(task, unit, inventory):
        return IMPOSSIBLE
    travel = distance(position, task.target)
    slack = task.deadline - step - travel
    deadline_bonus = strategy.deadline_weight * (1000.0 if slack < 0 else 24.0 / (1.0 + slack))
    return task.value + deadline_bonus - strategy.movement_cost * travel


def hungarian_maximize(utilities: list[list[float]]) -> list[int]:
    """Return one unique column per row for a rectangular utility matrix."""

    rows = len(utilities)
    if rows == 0:
        return []
    columns = len(utilities[0])
    if columns < rows:
        raise ValueError("Hungarian assignment requires at least as many columns as rows")
    finite = [value for row in utilities for value in row if value > IMPOSSIBLE / 2]
    ceiling = max(finite, default=0.0)
    costs = [[ceiling - value if value > IMPOSSIBLE / 2 else 1e15 for value in row] for row in utilities]

    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row_index in range(1, rows + 1):
        p[0] = row_index
        min_value = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        column0 = 0
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [-1] * rows
    for column in range(1, columns + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


class TaskScheduler:
    def __init__(self) -> None:
        self.commitments: dict[int, tuple[tuple[Any, ...], int]] = {}

    def reset(self) -> None:
        self.commitments.clear()

    def assign(
        self,
        tasks: list[Task],
        positions: list[tuple[int, int]],
        inventories: list[dict],
        step: int,
        strategy: Strategy,
    ) -> list[list[Any]]:
        task_by_key = {task.key: task for task in tasks}
        chosen: dict[int, Task] = {}
        reserved: set[tuple[Any, ...]] = set()

        # Live commitments are hard assignments until completion or invalidation.
        for unit in range(len(positions)):
            entry = self.commitments.get(unit)
            if entry is None:
                continue
            key, age = entry
            task = task_by_key.get(key)
            if task is None or key in reserved or age > MAX_COMMIT_TURNS or not _allowed(task, unit, inventories[unit]):
                self.commitments.pop(unit, None)
                continue
            chosen[unit] = task
            reserved.add(key)

        free_units = [unit for unit in range(len(positions)) if unit not in chosen]
        free_tasks = [task for task in tasks if task.key not in reserved]
        pass_tasks = [
            Task("PASS", positions[unit], ("PASS",), 0.0, step, owner=unit, reason="idle") for unit in free_units
        ]
        columns = [*free_tasks, *pass_tasks]
        if free_units:
            utilities = [
                [task_utility(task, unit, positions[unit], inventories[unit], step, strategy) for task in columns]
                for unit in free_units
            ]
            for row, column in enumerate(hungarian_maximize(utilities)):
                unit = free_units[row]
                task = columns[column]
                if utilities[row][column] <= IMPOSSIBLE / 2:
                    task = pass_tasks[row]
                chosen[unit] = task

        actions: list[list[Any]] = []
        for unit, position in enumerate(positions):
            task = chosen.get(unit, Task("PASS", position, ("PASS",), 0.0, step, owner=unit))
            if task.kind == "PASS":
                actions.append(["PASS"])
                self.commitments.pop(unit, None)
            elif position == task.target:
                actions.append(list(task.action))
                self.commitments.pop(unit, None)
            else:
                actions.append(step_toward(position, task.target))
                age = self.commitments.get(unit, (None, -1))[1] + 1
                self.commitments[unit] = (task.key, age)
        return actions
