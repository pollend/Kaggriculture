"""Enumerate legal crop, animal, logistics, and terrain jobs for the hybrid DQN."""

from __future__ import annotations

from dataclasses import dataclass

from agents.dqn_job_scorer.hybrid import HybridPlan
from agents.dqn_job_scorer.rules import ANIMAL_DATA, ANIMALS, CROP_DATA, CROPS, distance, shed_tiles

# Append-only within schema v2. Reordering invalidates weights.
JOB_KINDS = (
    "PASS", "WATER", "HARVEST", "PLANT", "DIG", "FERTILIZE", "DROP",
    "BUILD_COOP", "BUILD_PASTURE", "PICKUP_WHEAT", "FEED", "CARE",
    "COLLECT_FERTILIZER", "PICKUP_FERTILIZER", "PICKUP_ANIMAL", "PLACE_ANIMAL",
)
JOB_KIND_INDEX = {kind: index for index, kind in enumerate(JOB_KINDS)}
ASSET_ORDER = (*CROPS, *ANIMALS, "WHEAT_ITEM", "FERTILIZER_ITEM")
ASSET_INDEX = {asset: index for index, asset in enumerate(ASSET_ORDER)}
MAX_JOBS = 96


@dataclass(frozen=True, slots=True)
class Job:
    kind: str
    target: tuple[int, int]
    action: tuple[str, ...]
    asset: str | None = None
    tile: dict | None = None
    owner: int | None = None
    urgent: bool = False
    value: float = 0.0

    @property
    def crop(self) -> str | None:  # compatibility for commitment keys/tests
        return self.asset if self.asset in CROPS else None


def _harvestable(tile: dict, day: int) -> bool:
    if int(tile.get("yield_units", 0)) <= 0:
        return False
    if tile.get("kind") != "PLANT":
        return True
    data = CROP_DATA.get(str(tile.get("crop", "")))
    return bool(data and day - int(tile.get("planted_day", day)) >= int(data["first"]))


def _nearest_shed(position: tuple[int, int], sheds) -> tuple[int, int]:
    return min(sheds, key=lambda tile: (distance(position, tile), tile))


def enumerate_jobs(farm: dict, private: dict, day: int, positions: list[tuple[int, int]], inventories: list[dict], plan: HybridPlan | None = None) -> tuple[list[Job], int]:
    tiles = farm.get("tiles") or []
    sheds = shed_tiles(len(tiles)) if tiles else ((0, 0),)
    anchor = positions[0] if positions else sheds[0]
    jobs = [Job("PASS", anchor, ("PASS",))]
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    empty: list[tuple[int, int]] = []
    empty_structures = {"COOP": [], "PASTURE": []}
    animal_counts = {name: 0 for name in ANIMALS}
    hungry: list[tuple[tuple[int, int], dict]] = []

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            target = (x, y)
            if tile is None:
                empty.append(target)
                continue
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", ""))
            if kind == "WEED":
                jobs.append(Job("DIG", target, ("DIG",), tile=tile, value=25.0))
            elif kind == "PLANT":
                crop = str(tile.get("crop", ""))
                fresh = int(tile.get("planted_day", -1)) == day
                emergency = fresh or int(tile.get("consecutive_unwatered", 0)) >= 1
                if not tile.get("watered_today"):
                    jobs.append(Job("WATER", target, ("WATER",), crop, tile, urgent=emergency, value=10_000.0 if emergency else 50.0))
                if _harvestable(tile, day):
                    jobs.append(Job("HARVEST", target, ("HARVEST",), crop, tile, value=float(tile.get("yield_units", 0))))
                if int(tile.get("fertilized_until_day", -1)) < day:
                    for unit, inventory in enumerate(inventories):
                        if int((inventory or {}).get("FERTILIZER", 0)) > 0:
                            jobs.append(Job("FERTILIZE", target, ("FERTILIZE",), crop, tile, owner=unit))
            elif kind in empty_structures and not tile.get("animal"):
                empty_structures[kind].append(target)
            animal = tile.get("animal")
            if animal in ANIMAL_DATA:
                animal = str(animal)
                animal_counts[animal] += 1
                emergency = int(tile.get("consecutive_unfed", 0)) >= 1
                if not tile.get("fed_today"):
                    hungry.append((target, tile))
                    for unit, inventory in enumerate(inventories):
                        if int((inventory or {}).get("WHEAT", 0)) > 0:
                            jobs.append(Job("FEED", target, ("FEED",), animal, tile, owner=unit, urgent=emergency, value=12_000.0 if emergency else 100.0))
                if not tile.get("cared_today"):
                    jobs.append(Job("CARE", target, ("CARE",), animal, tile, value=80.0))
                if _harvestable(tile, day):
                    jobs.append(Job("HARVEST", target, ("HARVEST",), animal, tile, value=float(tile.get("yield_units", 0))))
                if tile.get("fertilizer_available"):
                    jobs.append(Job("COLLECT_FERTILIZER", target, ("COLLECT_FERTILIZER",), animal, tile, value=100.0))

    carried_wheat = sum(int((slot or {}).get("WHEAT", 0)) for slot in inventories)
    if hungry and carried_wheat < len(hungry) and int(shed.get("WHEAT", 0)) > 0 and positions:
        unit = min(range(len(positions)), key=lambda i: distance(positions[i], _nearest_shed(positions[i], sheds)))
        emergency = any(int(tile.get("consecutive_unfed", 0)) >= 1 for _, tile in hungry)
        amount = min(8, len(hungry) - carried_wheat, int(shed.get("WHEAT", 0)))
        jobs.append(Job("PICKUP_WHEAT", _nearest_shed(positions[unit], sheds), ("PICKUP", "WHEAT", amount), "WHEAT_ITEM", owner=unit, urgent=emergency, value=9_000.0 if emergency else 200.0))

    carried_fertilizer = sum(int((slot or {}).get("FERTILIZER", 0)) for slot in inventories)
    if carried_fertilizer == 0 and int(shed.get("FERTILIZER", 0)) > 0 and positions:
        unit = min(range(len(positions)), key=lambda i: distance(positions[i], _nearest_shed(positions[i], sheds)))
        jobs.append(Job("PICKUP_FERTILIZER", _nearest_shed(positions[unit], sheds), ("PICKUP", "FERTILIZER", 1), "FERTILIZER_ITEM", owner=unit))

    desired = plan.target_animals if plan else 0
    preferred = plan.animal if plan else "GOOSE"
    structures = animal_counts["GOOSE"] + len(empty_structures["COOP"]), animal_counts["COW"] + animal_counts["SHEEP"] + len(empty_structures["PASTURE"])
    total_owned = sum(animal_counts.values()) + sum(int(shed.get(name, 0)) for name in ANIMALS) + sum(int((slot or {}).get(name, 0)) for slot in inventories for name in ANIMALS)
    structure = str(ANIMAL_DATA[preferred]["structure"])
    structure_count = structures[0] if structure == "COOP" else structures[1]
    if total_owned and structure_count < min(desired, total_owned):
        kind = "BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"
        for target in empty[: min(desired, total_owned) - structure_count]:
            jobs.append(Job(kind, target, (kind,), preferred, value=300.0))

    carried_animals = [(unit, animal) for unit, inv in enumerate(inventories) for animal in ANIMALS if int((inv or {}).get(animal, 0)) > 0]
    for unit, animal in carried_animals:
        for target in empty_structures[str(ANIMAL_DATA[animal]["structure"])]:
            jobs.append(Job("PLACE_ANIMAL", target, ("PLACE", animal, 1), animal, owner=unit, value=500.0))
    for animal in ANIMALS:
        if int(shed.get(animal, 0)) <= 0 or not empty_structures[str(ANIMAL_DATA[animal]["structure"])] or not positions:
            continue
        unit = min(range(len(positions)), key=lambda i: distance(positions[i], _nearest_shed(positions[i], sheds)))
        jobs.append(Job("PICKUP_ANIMAL", _nearest_shed(positions[unit], sheds), ("PICKUP", animal, 1), animal, owner=unit, value=450.0))

    target_crops = plan.target_crops if plan else 12
    planted = sum(1 for row in tiles for tile in row if isinstance(tile, dict) and tile.get("kind") == "PLANT")
    room = len(empty) if plan is None else max(0, target_crops - planted)
    # Keep the cultivated area compact around the shed across days. Unit
    # distance is only a tie-breaker: hands are temporary, while every crop
    # repeatedly needs service and its harvest ultimately returns to the shed.
    empty.sort(key=lambda target: (
        min(distance(target, shed) for shed in sheds),
        min((distance(position, target) for position in positions), default=0),
        target,
    ))
    for crop in CROPS:
        if int(seeds.get(crop, 0)) > 0:
            for target in empty[:room]:
                jobs.append(Job("PLANT", target, ("PLANT", crop), crop))

    useful = {"FERTILIZER", *ANIMALS}
    if plan is not None and plan.target_animals:
        useful.add("WHEAT")
    for unit, inventory in enumerate(inventories):
        if not any(item not in useful and int(count) > 0 for item, count in (inventory or {}).items()):
            continue
        jobs.append(Job("DROP", _nearest_shed(positions[unit], sheds), ("DROP",), owner=unit))

    if len(jobs) <= MAX_JOBS:
        return jobs, 0
    pinned = [job for job in jobs if job.kind == "PASS" or job.owner is not None or job.urgent]
    rest = [job for job in jobs if job not in pinned]
    rank = {"HARVEST": 0, "WATER": 1, "FEED": 1, "CARE": 2, "COLLECT_FERTILIZER": 3, "DIG": 4, "FERTILIZE": 4, "PLACE_ANIMAL": 4, "BUILD_COOP": 5, "BUILD_PASTURE": 5, "PLANT": 6}
    rest.sort(key=lambda job: (rank.get(job.kind, 5), -job.value, min(distance(pos, job.target) for pos in positions), job.target))
    kept = pinned[:MAX_JOBS] + rest[: max(0, MAX_JOBS - len(pinned))]
    order = {id(job): index for index, job in enumerate(jobs)}
    kept.sort(key=lambda job: order[id(job)])
    return kept, len(jobs) - len(kept)
