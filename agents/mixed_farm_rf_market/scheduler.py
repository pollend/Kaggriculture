"""Deadline-aware mixed-farm tasks and deterministic joint assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.mixed_farm_rf_market.portfolio import Plan
from agents.mixed_farm_rf_market.rules import (
    ANIMAL_DATA,
    ANIMALS,
    CROP_DATA,
    CROPS,
    distance,
    get,
    nearest_shed,
    unlocked_empty_tiles,
)
from agents.mixed_farm_rf_market.strategy import Strategy


@dataclass(frozen=True, slots=True)
class Task:
    kind: str
    target: tuple[int, int]
    action: tuple[Any, ...]
    value: float
    deadline: int
    owner: int | None = None
    required_item: str | None = None

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.kind, self.target, self.action, self.owner, self.required_item)


def _prices(obs: Any) -> dict[str, float]:
    raw = (get(obs, "market", {}) or {}).get("prices", {}) or {}
    values = {name: float(raw.get(name, CROP_DATA[name]["base"])) for name in CROPS}
    for data in ANIMAL_DATA.values():
        product = str(data["product"])
        values[product] = float(raw.get(product, data["base"]))
    values["FERTILIZER"] = float(raw.get("FERTILIZER", 100))
    return values


def _inventory_count(inventories: list[dict], item: str) -> int:
    return sum(int((inventory or {}).get(item, 0)) for inventory in inventories)


def enumerate_tasks(
    obs: Any,
    farm: dict,
    private: dict,
    plan: Plan,
    positions: list[tuple[int, int]],
    inventories: list[dict],
    strategy: Strategy,
    *,
    episode_steps: int = 720,
    turns_per_day: int = 24,
) -> list[Task]:
    step = int(get(obs, "step", 0))
    day = int(get(obs, "day", 0))
    hour = int(get(obs, "hour", step % max(1, turns_per_day)))
    terminal = step >= episode_steps - strategy.liquidation_turns
    end_of_day = min(episode_steps - 1, (day + 1) * turns_per_day - 1)
    board_size = len(farm.get("tiles", []) or []) or 10
    prices = _prices(obs)
    shed = private.get("shed", {}) or {}
    tasks: list[Task] = []
    crop_counts = {name: 0 for name in CROPS}
    animal_counts = {name: 0 for name in ANIMALS}
    empty_structures: dict[str, list[tuple[int, int]]] = {"COOP": [], "PASTURE": []}
    hungry: list[tuple[int, int]] = []
    fertilizable: list[tuple[tuple[int, int], str, float]] = []

    for y, row in enumerate(farm.get("tiles", []) or []):
        for x, tile in enumerate(row):
            target = (x, y)
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "WEED":
                tasks.append(Task("DIG", target, ("DIG",), 900.0, episode_steps - 1))
                continue
            if kind == "PLANT" and tile.get("crop") in CROP_DATA:
                crop = str(tile["crop"])
                crop_counts[crop] += 1
                data = CROP_DATA[crop]
                age = day - int(tile.get("planted_day", day))
                if not tile.get("watered_today"):
                    emergency = int(tile.get("consecutive_unwatered", 0)) >= 1 or hour >= turns_per_day - 5
                    value = 25_000.0 if emergency else 500.0 * strategy.water_priority
                    tasks.append(Task("WATER", target, ("WATER",), value, end_of_day))
                units = int(tile.get("yield_units", 0))
                harvestable = units > 0 and age >= int(data["first"])
                if harvestable:
                    ready = terminal or bool(data["ongoing"]) or age >= int(data["peak"])
                    if ready:
                        tasks.append(
                            Task("HARVEST_CROP", target, ("HARVEST",), 700.0 + units * prices[crop], episode_steps - 1)
                        )
                if (
                    not terminal
                    and int(tile.get("fertilized_until_day", -1)) < day
                    and not bool(data["ongoing"])
                    and age <= int(data["peak"])
                ):
                    added_value = prices[crop] * 2.0 * strategy.fertilizer_value_weight
                    opportunity_cost = prices["FERTILIZER"] * strategy.sell_price_ratio
                    if added_value > opportunity_cost:
                        fertilizable.append((target, crop, added_value - opportunity_cost))
                continue

            animal = tile.get("animal")
            if animal in ANIMAL_DATA:
                animal = str(animal)
                animal_counts[animal] += 1
                data = ANIMAL_DATA[animal]
                product = str(data["product"])
                if not tile.get("fed_today"):
                    hungry.append(target)
                    emergency = int(tile.get("consecutive_unfed", 0)) >= 1 or hour >= turns_per_day - 5
                    tasks.append(
                        Task(
                            "FEED",
                            target,
                            ("FEED",),
                            30_000.0 if emergency else 900.0 + prices[product] * 2.0,
                            end_of_day,
                            required_item="WHEAT",
                        )
                    )
                if not tile.get("cared_today") and strategy.care_value_weight > 0.05:
                    tasks.append(Task(
                        "CARE", target, ("CARE",),
                        800.0 + prices[product] * strategy.care_value_weight, end_of_day,
                    ))
                units = int(tile.get("yield_units", 0))
                if units > 0:
                    tasks.append(
                        Task("HARVEST_ANIMAL", target, ("HARVEST",), 850.0 + units * prices[product], episode_steps - 1)
                    )
                if tile.get("fertilizer_available") and strategy.fertilizer_value_weight > 0.05:
                    tasks.append(
                        Task(
                            "COLLECT_FERTILIZER",
                            target,
                            ("COLLECT_FERTILIZER",),
                            700.0 + prices["FERTILIZER"] * strategy.fertilizer_value_weight,
                            end_of_day,
                        )
                    )
            elif kind in empty_structures:
                empty_structures[str(kind)].append(target)

    # Item logistics precede the tasks that consume those items on later turns.
    carried_wheat = _inventory_count(inventories, "WHEAT")
    wheat_shortfall = max(0, len(hungry) - carried_wheat)
    shed_wheat = int(shed.get("WHEAT", 0))
    if wheat_shortfall and shed_wheat and positions:
        remaining = min(wheat_shortfall, shed_wheat)
        candidates = sorted(
            (
                distance(position, nearest_shed(position, board_size)),
                owner,
                position,
            )
            for owner, position in enumerate(positions)
            if int((inventories[owner] or {}).get("WHEAT", 0)) == 0
        )
        for _, owner, position in candidates:
            if remaining <= 0:
                break
            amount = min(strategy.feed_carry_target, remaining)
            remaining -= amount
            tasks.append(
                Task(
                    "PICKUP_WHEAT",
                    nearest_shed(position, board_size),
                    ("PICKUP", "WHEAT", amount),
                    28_000.0 if any(
                        isinstance(farm["tiles"][y][x], dict)
                        and int(farm["tiles"][y][x].get("consecutive_unfed", 0)) >= 1
                        for x, y in hungry
                    ) else 2_000.0,
                    end_of_day,
                    owner=owner,
                )
            )

    carried_fertilizer = _inventory_count(inventories, "FERTILIZER")
    for target, _, value in fertilizable:
        tasks.append(
            Task("FERTILIZE", target, ("FERTILIZE",), value, end_of_day, required_item="FERTILIZER")
        )
    if fertilizable and carried_fertilizer == 0 and int(shed.get("FERTILIZER", 0)) > 0 and positions:
        owner, position = min(
            enumerate(positions),
            key=lambda pair: (distance(pair[1], nearest_shed(pair[1], board_size)), pair[0]),
        )
        tasks.append(
            Task(
                "PICKUP_FERTILIZER",
                nearest_shed(position, board_size),
                ("PICKUP", "FERTILIZER", min(3, int(shed["FERTILIZER"]))),
                600.0,
                end_of_day,
                owner=owner,
            )
        )

    empties = sorted(
        unlocked_empty_tiles(farm),
        key=lambda target: (min((distance(position, target) for position in positions), default=0), target),
    )
    reserved: set[tuple[int, int]] = set()
    if not terminal:
        owned_animals = {
            name: animal_counts[name]
            + int(shed.get(name, 0))
            + _inventory_count(inventories, name)
            for name in ANIMALS
        }
        existing_structures = {
            "COOP": animal_counts["GOOSE"] + len(empty_structures["COOP"]),
            "PASTURE": animal_counts["COW"] + animal_counts["SHEEP"] + len(empty_structures["PASTURE"]),
        }
        desired_structures = {
            "COOP": min(plan.target_animals["GOOSE"], owned_animals["GOOSE"]),
            "PASTURE": min(
                plan.target_animals["COW"] + plan.target_animals["SHEEP"],
                owned_animals["COW"] + owned_animals["SHEEP"],
            ),
        }
        for structure in ("COOP", "PASTURE"):
            deficit = max(0, desired_structures[structure] - existing_structures[structure])
            action = "BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"
            available = [target for target in empties if target not in reserved][:deficit]
            for target in available:
                reserved.add(target)
                tasks.append(Task(action, target, (action,), 1_100.0, episode_steps - 1))

        # Assign each empty structure to the highest-value outstanding compatible animal.
        slots = {name: sorted(values) for name, values in empty_structures.items()}
        deficits = {name: max(0, plan.target_animals[name] - animal_counts[name]) for name in ANIMALS}
        placements: list[tuple[str, tuple[int, int]]] = []
        for structure in ("COOP", "PASTURE"):
            compatible = [
                name for name in ANIMALS
                if ANIMAL_DATA[name]["structure"] == structure and deficits[name] > 0
            ]
            for target in slots[structure]:
                if not compatible:
                    break
                animal = max(compatible, key=lambda name: (plan.animal_scores[name], name))
                placements.append((animal, target))
                deficits[animal] -= 1
                compatible = [name for name in compatible if deficits[name] > 0]

        for animal, target in placements:
            carrying = [
                owner for owner, inventory in enumerate(inventories)
                if int((inventory or {}).get(animal, 0)) > 0
            ]
            for owner in carrying:
                tasks.append(
                    Task("PLACE_ANIMAL", target, ("PLACE", animal, 1), 1_500.0, episode_steps - 1, owner, animal)
                )
            if not carrying and int(shed.get(animal, 0)) > 0 and positions:
                owner, position = min(
                    enumerate(positions),
                    key=lambda pair: (distance(pair[1], nearest_shed(pair[1], board_size)), pair[0]),
                )
                tasks.append(
                    Task(
                        "PICKUP_ANIMAL",
                        nearest_shed(position, board_size),
                        ("PICKUP", animal, 1),
                        1_600.0,
                        episode_steps - 1,
                        owner=owner,
                    )
                )

        seeds = private.get("seeds", {}) or {}
        plant_tiles = [target for target in empties if target not in reserved]
        cursor = 0
        for crop in sorted(CROPS, key=lambda name: (-plan.crop_scores[name], name)):
            if hour >= turns_per_day - 1:
                break
            deficit = max(0, plan.target_crops[crop] - crop_counts[crop])
            count = min(deficit, int(seeds.get(crop, 0)), len(plant_tiles) - cursor)
            for target in plant_tiles[cursor : cursor + count]:
                tasks.append(
                    Task("PLANT", target, ("PLANT", crop), 500.0 + plan.crop_scores[crop], end_of_day - 1)
                )
            cursor += count

    # Any carried output should return to the shed. DROP is owner-specific.
    final_day = step >= episode_steps - turns_per_day
    useful = {"WHEAT", "FERTILIZER", *ANIMALS} if not final_day else set(ANIMALS)
    for owner, inventory in enumerate(inventories):
        liquid_value = sum(
            int(quantity) * prices.get(item, 0.0)
            for item, quantity in (inventory or {}).items()
            if item not in useful
        )
        if liquid_value > 0:
            tasks.append(
                Task(
                    "DROP",
                    nearest_shed(positions[owner], board_size),
                    ("DROP",),
                    liquid_value + (20_000.0 if terminal else 1_000.0),
                    episode_steps - 1,
                    owner=owner,
                )
            )

    tasks.sort(key=lambda task: (task.deadline, -task.value, task.kind, task.target, task.owner or -1))
    return tasks


def _allowed(task: Task, unit: int, inventory: dict) -> bool:
    if task.owner is not None and task.owner != unit:
        return False
    return task.required_item is None or int((inventory or {}).get(task.required_item, 0)) > 0


def assign_actions(
    tasks: list[Task],
    positions: list[tuple[int, int]],
    inventories: list[dict],
    step: int,
    strategy: Strategy,
    commitments: dict[int, tuple[Any, ...]] | None = None,
) -> tuple[list[list[Any]], dict[int, tuple[Any, ...]]]:
    """Greedily maximize joint task utility while keeping tasks unique."""

    # Task values are expressed in approximate coins, while Manhattan distance
    # is measured in tiles.  Scale the latter into the same neighborhood so a
    # small value difference does not send two workers past each other on
    # opposite cross-farm errands.  Survival emergencies remain dominant.
    travel_price = 125.0 * strategy.movement_cost

    actions: list[list[Any]] = [["PASS"] for _ in positions]
    free = set(range(len(positions)))
    available = list(tasks)
    retained: dict[int, tuple[Any, ...]] = {}
    commitments = commitments or {}

    # First reserve only as many workers as survival emergencies need. Other
    # workers keep travelling toward stable setup/logistics commitments.
    while free:
        choices: list[tuple[float, int, int]] = []
        for unit in free:
            for task_index, task in enumerate(available):
                if task.value < 20_000.0 or not _allowed(task, unit, inventories[unit]):
                    continue
                travel = distance(positions[unit], task.target)
                choices.append((task.value - travel_price * travel, unit, task_index))
        if not choices:
            break
        _, unit, task_index = max(choices, key=lambda row: (row[0], -row[1], -row[2]))
        task = available.pop(task_index)
        free.remove(unit)
        actions[unit] = (
            list(task.action)
            if positions[unit] == task.target
            else step_toward_safe(positions[unit], task.target)
        )
        if positions[unit] != task.target:
            retained[unit] = task.key

    by_key = {task.key: index for index, task in enumerate(available)}
    for unit, key in sorted(commitments.items()):
        task_index = by_key.get(key)
        if unit not in free or task_index is None or task_index >= len(available):
            continue
        task = available[task_index]
        if not _allowed(task, unit, inventories[unit]):
            continue
        actions[unit] = (
            list(task.action)
            if positions[unit] == task.target
            else step_toward_safe(positions[unit], task.target)
        )
        if positions[unit] != task.target:
            retained[unit] = task.key
        free.remove(unit)
        available.pop(task_index)
        by_key = {candidate.key: index for index, candidate in enumerate(available)}
    while free and available:
        choices: list[tuple[float, int, int]] = []
        for unit in free:
            for task_index, task in enumerate(available):
                if not _allowed(task, unit, inventories[unit]):
                    continue
                travel = distance(positions[unit], task.target)
                slack = task.deadline - step - travel
                urgency = strategy.deadline_weight * (2_000.0 if slack < 0 else 24.0 / (1.0 + slack))
                utility = task.value + urgency - travel_price * travel
                choices.append((utility, unit, task_index))
        if not choices:
            break
        _, unit, task_index = max(choices, key=lambda row: (row[0], -row[1], -row[2]))
        task = available.pop(task_index)
        free.remove(unit)
        actions[unit] = list(task.action) if positions[unit] == task.target else step_toward_safe(positions[unit], task.target)
        if positions[unit] != task.target:
            retained[unit] = task.key
    return actions, retained


def step_toward_safe(position: tuple[int, int], target: tuple[int, int]) -> list[str]:
    # Kept local to make the assignment API easy to unit test.
    dx, dy = target[0] - position[0], target[1] - position[1]
    if dx:
        return ["EAST" if dx > 0 else "WEST"]
    if dy:
        return ["SOUTH" if dy > 0 else "NORTH"]
    return ["PASS"]
