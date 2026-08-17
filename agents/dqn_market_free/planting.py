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

    # Two passes, and the split is the whole point.
    #
    # One pass taking `unassigned[0]` whenever a crop had no planned tile of its
    # own cost the agent most of its income. `MELON` is last in `donor.CROPS`, and
    # `CROP_MIX` plans no TOMATO or STRAWBERRY tile anywhere, so every tomato or
    # strawberry seed fell through to the fallback and took a tile the donor had
    # planned for melon -- the one crop worth 142 per tile-day against 24-40 for
    # everything else. Seats buying a 5-40% melon share finished on a median of
    # 64 against 2,568 for seats buying none at all: the mixed purchase was worse
    # than either pure strategy, because it paid for melon tiles and then planted
    # carrots on them.
    unassigned = list(candidates)
    leftover: Counter[str] = Counter()

    # Pass one: every crop claims the tiles the donor already planned for it,
    # before any crop is allowed to look at a tile planned for something else.
    for crop in donor.CROPS:
        for _ in range(allocation[crop]):
            matching = next(
                (position for position in unassigned if roles[position] == ("CROP", crop)),
                None,
            )
            if matching is None:
                leftover[crop] += 1
                continue
            roles[matching] = ("CROP", crop)
            unassigned.remove(matching)

    # Pass two: a crop with no planned tile left may still use a *spare* one, but
    # only where the planned crop has no seed held for it. Holding melon seed now
    # protects every melon tile, and the leftover is dropped rather than planted
    # -- an unplanted melon tile is worth more than a tomato growing on it, and a
    # seed left in the shed is recoverable next turn while a converted tile is not.
    for crop in donor.CROPS:
        for _ in range(leftover[crop]):
            spare = next(
                (
                    position
                    for position in unassigned
                    if seeds.get(roles[position][1], 0) <= 0
                ),
                None,
            )
            if spare is None:
                break
            roles[spare] = ("CROP", crop)
            unassigned.remove(spare)
    return roles
