"""Let held seed inventory steer vacant donor crop roles.

The donor already knows how to route workers to plant, water, and harvest every
crop in its ``CROPS`` table. Its fixed role map is the only reason a raw market
decision such as ``BUY_SEED TOMATO`` cannot reach that machinery. This adapter
changes only vacant or weed-covered crop roles; live plants and every animal
role remain the donor's responsibility.
"""

from __future__ import annotations

from collections import Counter

from agents.dqn_market_free import donor


def _seed_allocations(seeds: Counter[str], slots: int) -> Counter[str]:
    """Allocate scarce vacant slots from the market's persistent seed signal.

    Every held crop gets one slot before bulk inventory gets additional slots.
    This prevents one large exploratory purchase from making a smaller, valid
    market choice permanently invisible while still letting quantities control
    most of the resulting crop mix.
    """

    allocation: Counter[str] = Counter()
    remaining = Counter({crop: count for crop, count in seeds.items() if count > 0})
    crop_order = {crop: index for index, crop in enumerate(donor.CROPS)}

    for crop in sorted(remaining, key=lambda item: (-remaining[item], crop_order[item])):
        if slots <= 0:
            return allocation
        allocation[crop] += 1
        remaining[crop] -= 1
        slots -= 1

    while slots > 0:
        available = [crop for crop, count in remaining.items() if count > 0]
        if not available:
            break
        crop = max(available, key=lambda item: (remaining[item], -crop_order[item]))
        allocation[crop] += 1
        remaining[crop] -= 1
        slots -= 1
    return allocation


def market_influenced_roles(
    obs: dict,
    farm: dict,
    private: dict,
    donor_roles: dict,
) -> dict:
    """Overlay held, still-plantable seeds onto vacant donor crop slots."""

    roles = dict(donor_roles)
    day = int(obs.get("day", 0) or 0)
    tiles = farm.get("tiles", []) or []
    candidates: list[tuple[int, int]] = []
    for position, (kind, _crop) in roles.items():
        if kind != "CROP":
            continue
        x, y = position
        tile = tiles[y][x]
        if tile is None or (isinstance(tile, dict) and tile.get("kind") == "WEED"):
            candidates.append(position)

    if not candidates:
        return roles

    raw_seeds = private.get("seeds", {}) or {}
    seeds = Counter(
        {
            crop: max(0, int(raw_seeds.get(crop, 0) or 0))
            for crop, rule in donor.CROPS.items()
            if day <= int(rule["last_plant"])
        }
    )
    allocation = _seed_allocations(seeds, len(candidates))
    if not allocation:
        return roles

    # Prefer the donor's original location for a crop when one is available;
    # otherwise use the next stable donor-ordered vacant crop slot.
    unassigned = list(candidates)
    for crop in donor.CROPS:
        count = allocation[crop]
        for _ in range(count):
            matching = next(
                (position for position in unassigned if roles[position] == ("CROP", crop)),
                None,
            )
            position = matching if matching is not None else unassigned[0]
            roles[position] = ("CROP", crop)
            unassigned.remove(position)
    return roles
