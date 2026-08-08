"""Enumerate the legal job set the Q-network chooses from.

This is deliberately *not* ``ConfigurableAgent._jobs``. That method applies the
heuristic's own filters -- ``target_planted_tiles`` caps how much may be planted,
``_should_harvest`` decides when a plant is ripe enough -- which would trap a
learned policy inside the heuristic we are trying to beat. Here the rule is
"emit what the rules allow, attach no opinion", and the network supplies the
opinion.

Geometry and crop constants come from this agent's own ``rules`` module. The
duplication is deliberate: an agent directory is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.dqn_job_scorer.rules import CROP_DATA, distance, shed_tiles

# Job kinds, in a fixed order -- the index is the one-hot slot in `features.py`,
# so appending is safe but reordering silently invalidates trained weights.
JOB_KINDS = ("PASS", "WATER", "HARVEST", "PLANT", "DIG", "FERTILIZE", "DROP")
JOB_KIND_INDEX = {kind: index for index, kind in enumerate(JOB_KINDS)}

CROP_ORDER = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
CROP_INDEX = {crop: index for index, crop in enumerate(CROP_ORDER)}

# Per-unit candidate cap. Truncation is reported, never silent -- see `enumerate_jobs`.
MAX_JOBS = 64


@dataclass(frozen=True, slots=True)
class Job:
    """One legal thing a unit could be sent to do.

    `owner` binds a job to a single unit (only DROP uses it, since a carry job
    belongs to whoever is carrying). `None` means any unit may take it.
    """

    kind: str
    target: tuple[int, int]
    action: tuple[str, ...]
    crop: str | None = None
    tile: dict | None = None
    owner: int | None = None


def _plant_is_harvestable(tile: dict, day: int) -> bool:
    if int(tile.get("yield_units", 0)) <= 0:
        return False
    crop = str(tile.get("crop", ""))
    data = CROP_DATA.get(crop)
    if data is None:
        return False
    return day - int(tile.get("planted_day", day)) >= int(data["first"])


def enumerate_jobs(
    farm: dict,
    private: dict,
    day: int,
    positions: list[tuple[int, int]],
    inventories: list[dict],
) -> tuple[list[Job], int]:
    """Return (jobs, truncated_count) for the current turn.

    PASS is always index 0 so "do nothing" is a real, always-available choice
    rather than a fallback the network can never express.
    """
    tiles = farm.get("tiles") or []
    board_size = len(tiles)
    sheds = shed_tiles(board_size) if board_size else ((0, 0),)
    anchor = positions[0] if positions else sheds[0]

    jobs: list[Job] = [Job("PASS", anchor, ("PASS",))]

    seeds = private.get("seeds", {}) or {}
    plantable = [crop for crop in CROP_ORDER if int(seeds.get(crop, 0)) > 0]
    fertilizer_units = [
        unit
        for unit, inventory in enumerate(inventories)
        if int((inventory or {}).get("FERTILIZER", 0)) > 0
    ]

    empty: list[tuple[int, int]] = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            target = (x, y)
            if tile is None:
                empty.append(target)
                continue
            if not isinstance(tile, dict):
                continue  # "LOCKED" -- passable but nothing can be done there

            kind = tile.get("kind")
            if kind == "WEED":
                jobs.append(Job("DIG", target, ("DIG",), tile=tile))
                continue
            if kind != "PLANT":
                continue

            if not tile.get("watered_today"):
                jobs.append(Job("WATER", target, ("WATER",), crop=tile.get("crop"), tile=tile))
            if _plant_is_harvestable(tile, day):
                jobs.append(Job("HARVEST", target, ("HARVEST",), crop=tile.get("crop"), tile=tile))
            if int(tile.get("fertilized_until_day", -1)) < day:
                for unit in fertilizer_units:
                    jobs.append(
                        Job(
                            "FERTILIZE",
                            target,
                            ("FERTILIZE",),
                            crop=tile.get("crop"),
                            tile=tile,
                            owner=unit,
                        )
                    )

    for crop in plantable:
        for target in empty:
            jobs.append(Job("PLANT", target, ("PLANT", crop), crop=crop))

    for unit, inventory in enumerate(inventories):
        if sum(int(count) for count in (inventory or {}).values()) <= 0:
            continue
        position = positions[unit] if unit < len(positions) else anchor
        target = min(sheds, key=lambda tile: (distance(position, tile), tile))
        jobs.append(Job("DROP", target, ("DROP",), owner=unit))

    if len(jobs) <= MAX_JOBS:
        return jobs, 0

    # Keep PASS and every unit-owned job. Dropping an owned DROP strands a carrier.
    pinned = [job for job in jobs if job.kind == "PASS" or job.owner is not None]
    rest = [job for job in jobs if job.kind != "PASS" and job.owner is None]
    room = max(0, MAX_JOBS - len(pinned))

    # Cut by what is cheapest to lose, not by distance alone. Only PLANT
    # multiplies -- it is crops-in-hand x empty tiles, so it is what overflows
    # the cap -- while maintenance is bounded by the tiles already planted.
    # Ranking purely on distance let a nearby PLANT evict a distant plant that
    # dies tonight without water, which is a far worse trade.
    def priority(job: Job) -> int:
        if job.kind == "WATER" and int((job.tile or {}).get("consecutive_unwatered", 0)) >= 1:
            return 0  # unwatered again tonight means a weed tomorrow
        if job.kind == "HARVEST":
            return 1
        if job.kind == "WATER":
            return 2
        if job.kind in ("DIG", "FERTILIZE"):
            return 3
        return 4  # PLANT

    nearest = {id(job): min(distance(pos, job.target) for pos in positions) for job in rest}
    rest.sort(key=lambda job: (priority(job), nearest[id(job)], job.target))
    kept = pinned + rest[:room]

    # Restore enumeration order so the job list stays a deterministic function
    # of the observation rather than of the sort.
    order = {id(job): index for index, job in enumerate(jobs)}
    kept.sort(key=lambda job: order[id(job)])
    return kept, len(jobs) - len(kept)
