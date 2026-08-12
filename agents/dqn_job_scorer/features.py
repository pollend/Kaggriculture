"""The featurizer. Single-sourced so training and inference cannot drift.

Both the torch trainer and the numpy submission import this module. Any change
here invalidates trained weights, which is why the layout is pinned by explicit
dimension constants and a parity test.

The split matters for memory. Global features are identical for every unit at a
step, and job features are identical for every unit *except* the distance term.
So the replay buffer stores globals and job statics once per step and recomputes
the two distance columns at sample time -- roughly a tenfold saving over storing
a full candidate matrix per unit-transition.
"""

from __future__ import annotations

import numpy as np

from agents.dqn_job_scorer.hybrid import MODES, HybridPlan
from agents.dqn_job_scorer.jobs import ASSET_INDEX, ASSET_ORDER, JOB_KIND_INDEX, JOB_KINDS, Job
from agents.dqn_job_scorer.rules import (
    ANIMAL_DATA,
    ANIMALS,
    CROP_DATA,
    CROPS,
    MARKET_I0,
    MAX_MANHATTAN,
    PRODUCT_BASE,
    SHOP_DEMAND,
    distance,
    get,
    shed_tiles,
)

GLOBAL_DIM = 75
JOB_STATIC_DIM = 44
JOB_DISTANCE_DIM = 3
FEATURE_DIM = GLOBAL_DIM + JOB_STATIC_DIM + JOB_DISTANCE_DIM


def global_features(
    obs,
    farm: dict,
    private: dict,
    player: int,
    episode_steps: int = 720,
    turns_per_day: int = 24,
    plan: HybridPlan | None = None,
) -> np.ndarray:
    """Per-step state features, shared by every unit and every candidate job."""
    step = int(get(obs, "step", 0))
    day = int(get(obs, "day", 0))
    hour = int(get(obs, "hour", step % turns_per_day))
    total_days = max(1.0, episode_steps / turns_per_day)
    days_left = max(0.0, (episode_steps - step) / turns_per_day)

    money = float(farm.get("money", 0.0))
    farms = get(obs, "farms", []) or []
    opponent = max(
        (float(other.get("money", 0.0)) for index, other in enumerate(farms) if index != player),
        default=0.0,
    )

    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    market = get(obs, "market", {}) or {}
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}

    planted = empty = weeds = unwatered = harvestable = 0
    animal_counts = {animal: 0 for animal in ANIMALS}
    opponent_animals = {animal: 0 for animal in ANIMALS}
    empty_structures = {"COOP": 0, "PASTURE": 0}
    hungry = emergency_hungry = uncared = fertilizer_ready = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if tile is None:
                empty += 1
            elif isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "WEED":
                    weeds += 1
                elif kind == "PLANT":
                    planted += 1
                    if not tile.get("watered_today"):
                        unwatered += 1
                    if int(tile.get("yield_units", 0)) > 0:
                        harvestable += 1
                elif kind in empty_structures and not tile.get("animal"):
                    empty_structures[kind] += 1
                animal = tile.get("animal")
                if animal in animal_counts:
                    animal_counts[animal] += 1
                    if not tile.get("fed_today"):
                        hungry += 1
                    if int(tile.get("consecutive_unfed", 0)) >= 1:
                        emergency_hungry += 1
                    if not tile.get("cared_today"):
                        uncared += 1
                    if tile.get("fertilizer_available"):
                        fertilizer_ready += 1
    for index, other in enumerate(farms):
        if index == player:
            continue
        for row in other.get("tiles", []) or []:
            for tile in row:
                if isinstance(tile, dict) and tile.get("animal") in opponent_animals:
                    opponent_animals[str(tile["animal"])] += 1

    shed_total = sum(int(count) for count in shed.values())
    carried = sum(int(count) for slot in inventories for count in (slot or {}).values())
    quadrants = set(farm.get("unlocked_quadrants") or [])

    values = [
        day / total_days,
        hour / turns_per_day,
        step / max(1, episode_steps),
        days_left / total_days,
        np.log1p(max(0.0, money)) / 10.0,
        money / 10_000.0,
        opponent / 10_000.0,
        money / (money + opponent + 1.0),
    ]
    values += [int(seeds.get(crop, 0)) / 10.0 for crop in CROPS]
    values += [int(shed.get(crop, 0)) / 50.0 for crop in CROPS]
    values += [
        shed_total / 100.0,
        int(shed.get("FERTILIZER", 0)) / 20.0,
    ]
    values += [float(prices.get(crop, CROP_DATA[crop]["base"])) / float(CROP_DATA[crop]["base"]) for crop in CROPS]
    values += [(float(inventory.get(crop, MARKET_I0)) - MARKET_I0) / 1000.0 for crop in CROPS]
    values += [
        (1 + len(farm.get("hands") or [])) / 14.0,
        int(farm.get("hires_today", 0)) / 14.0,
        planted / 100.0,
        empty / 100.0,
        weeds / 100.0,
        unwatered / 25.0,
        harvestable / 25.0,
    ]
    values += [1.0 if name in quadrants else 0.0 for name in ("NW", "NE", "SW", "SE")]
    values += [
        carried / 50.0,
        max(0, 100 - shed_total) / 100.0,
    ]
    animal_products = ("EGG", "MILK", "WOOL", "FERTILIZER")
    values += [int(shed.get(item, 0)) / 25.0 for item in ("EGG", "MILK", "WOOL")]
    values += [float(prices.get(item, PRODUCT_BASE[item])) / PRODUCT_BASE[item] for item in animal_products]
    values += [(float(inventory.get(item, MARKET_I0)) - MARKET_I0) / 1000.0 for item in animal_products]
    values += [animal_counts[name] / 10.0 for name in ANIMALS]
    values += [opponent_animals[name] / 10.0 for name in ANIMALS]
    values += [empty_structures[name] / 10.0 for name in ("COOP", "PASTURE")]
    values += [hungry / 10.0, emergency_hungry / 10.0, uncared / 10.0, fertilizer_ready / 10.0]
    demand = {item: 0 for item in animal_products}
    for shop in (get(obs, "town", {}) or {}).get("unlocked_shops", []) or []:
        for item in SHOP_DEMAND.get(shop, ()):
            if item in demand:
                demand[item] += 1
    values += [demand[item] / 4.0 for item in animal_products]
    mode = plan.mode if plan else "CROP"
    values += [1.0 if mode == name else 0.0 for name in MODES]
    values += [sum(animal_counts.values()) * 3.0 / max(1.0, (1 + len(farm.get("hands") or [])) * 24.0)]
    values += [sum(int((slot or {}).get("WHEAT", 0)) for slot in inventories) / 20.0]

    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] != GLOBAL_DIM:  # pragma: no cover -- guards a silent layout change
        raise AssertionError(f"GLOBAL_DIM is {GLOBAL_DIM} but built {array.shape[0]} features")
    return array


def job_static_features(
    jobs: list[Job],
    obs,
    farm: dict,
    day: int,
    step: int,
    episode_steps: int = 720,
    turns_per_day: int = 24,
    plan: HybridPlan | None = None,
) -> np.ndarray:
    """Per-job features that do not depend on which unit is being assigned."""
    prices = ((get(obs, "market", {}) or {}).get("prices", {}) or {})
    board_size = len(farm.get("tiles") or []) or 10
    sheds = shed_tiles(board_size)
    days_left = max(0.0, (episode_steps - step) / turns_per_day)

    out = np.zeros((len(jobs), JOB_STATIC_DIM), dtype=np.float32)
    for index, job in enumerate(jobs):
        row = out[index]
        row[JOB_KIND_INDEX[job.kind]] = 1.0

        asset = job.asset
        encoded_asset = asset
        if asset == "WHEAT":
            encoded_asset = "WHEAT"
        if encoded_asset in ASSET_INDEX:
            row[len(JOB_KINDS) + ASSET_INDEX[encoded_asset]] = 1.0
            row[len(JOB_KINDS) + len(ASSET_ORDER)] = 1.0

        cursor = len(JOB_KINDS) + len(ASSET_ORDER) + 1
        tile = job.tile
        if tile is not None:
            age = day - int(tile.get("planted_day", tile.get("placed_day", day)))
            data = CROP_DATA.get(str(tile.get("crop", "")), None)
            first = float(data["first"]) if data else 0.0
            peak = float(data["peak"]) if data else 1.0
            row[cursor] = min(age, 12) / 12.0
            row[cursor + 1] = 1.0 if age >= first else 0.0
            row[cursor + 2] = 1.0 if age >= peak else 0.0
            row[cursor + 3] = min(int(tile.get("yield_units", 0)), 6) / 6.0
            row[cursor + 4] = min(int(tile.get("consecutive_unwatered", 0)), 2) / 2.0
            row[cursor + 5] = 1.0 if tile.get("watered_today") else 0.0
            # Bonus window opens at ceil(peak / 2) for one-time crops.
            row[cursor + 6] = 1.0 if age >= -(-peak // 2) else 0.0
            row[cursor + 7] = 1.0 if int(tile.get("fertilized_until_day", -1)) >= day else 0.0
            row[cursor + 8] = min(int(tile.get("consecutive_unfed", 0)), 2) / 2.0
            row[cursor + 9] = 1.0 if tile.get("fed_today") else 0.0
            row[cursor + 10] = 1.0 if tile.get("cared_today") else 0.0
            row[cursor + 11] = min(int(tile.get("pending_care_bonus", 0)), 6) / 6.0
        cursor += 12
        base = PRODUCT_BASE.get(str(ANIMAL_DATA.get(str(asset), {}).get("product", asset)), 100)
        product = str(ANIMAL_DATA.get(str(asset), {}).get("product", asset))
        row[cursor] = float(prices.get(product, base)) / max(1.0, float(base))
        first = CROP_DATA.get(str(asset), ANIMAL_DATA.get(str(asset), {})).get("first", 0)
        row[cursor + 1] = 1.0 if days_left >= float(first) else 0.0
        row[cursor + 2] = 1.0 if job.urgent else 0.0
        row[cursor + 3] = min(job.value, 12_000.0) / 12_000.0
        row[cursor + 4] = min(distance(job.target, sheds[0]), MAX_MANHATTAN) / MAX_MANHATTAN
        if cursor + 5 != JOB_STATIC_DIM:  # pragma: no cover
            raise AssertionError(f"JOB_STATIC_DIM is {JOB_STATIC_DIM} but built {cursor + 5}")
    return out


def distance_features(
    job_targets: np.ndarray, position: tuple[int, int], committed: int = -1
) -> np.ndarray:
    """The per-unit part of a job feature: distance, and whether it is committed.

    `job_targets` is an int array of shape (n_jobs, 2). Recomputed at sample
    time in the replay buffer rather than stored.

    `committed` is the index of the job this unit is already walking to, or -1.
    Without it the state is not Markov: the same board can be one turn into a
    walk east or one turn into a walk west, and the network cannot tell -- which
    is what let the policy oscillate between two targets and never arrive.
    """
    targets = np.asarray(job_targets, dtype=np.float32)
    distance = np.abs(targets[:, 0] - position[0]) + np.abs(targets[:, 1] - position[1])
    out = np.zeros((targets.shape[0], JOB_DISTANCE_DIM), dtype=np.float32)
    out[:, 0] = np.minimum(distance, MAX_MANHATTAN) / MAX_MANHATTAN
    out[:, 1] = (distance == 0).astype(np.float32)  # can act on this tile right now
    if 0 <= committed < targets.shape[0]:
        out[committed, 2] = 1.0
    return out


def assemble(global_feats: np.ndarray, job_static: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Broadcast globals across the candidate set and concatenate. (n_jobs, FEATURE_DIM)."""
    count = job_static.shape[0]
    tiled = np.broadcast_to(global_feats.astype(np.float32), (count, GLOBAL_DIM))
    return np.concatenate([tiled, job_static.astype(np.float32), distance], axis=1)


def job_targets(jobs: list[Job]) -> np.ndarray:
    return np.asarray([job.target for job in jobs], dtype=np.int16)
