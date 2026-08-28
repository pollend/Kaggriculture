"""Own the animal roles: structures follow committed animals, for every species.

The donor already knows how to build a structure, place an animal on it, feed it,
and harvest it -- every one of those jobs reads ``ANIMALS[role_item]`` rather than
a species name, so a ``("ANIMAL", "GOOSE")`` role drives the whole chain with no
donor change. What the donor gets wrong is *when* a role exists.

``donor._herd_targets`` floors its targets at ``CORE_HERD_SEQUENCE.count(animal)``
-- COW 3, SHEEP 1 -- **whether or not an animal was ever bought**, so pasture is
built in every episode. ``_role_plan`` then only ever emits COW and SHEEP, so a
goose gets a role from nobody. The two halves of that asymmetry are what this
module removes:

* measured over v1's traces, 52% of episodes bought zero animals and still built
  14.6 pastures each, and those episodes ended 16% poorer;
* ``BUY_ANIMAL COW`` landed on standing pasture and started paying, while
  ``BUY_ANIMAL GOOSE`` paid 300 and then waited on a ``BUILD_COOP`` and a
  ``PLACE``. The learner priced that delay correctly and unlearned the goose --
  goose purchases fall from 10.3 an episode at generation 1 to 0.05 at 53 with the
  column unmasked ~500 turns an episode the whole way.

So the rule here is *structures follow committed animals*, applied to all three
species symmetrically, plus exactly one warm slot so that the first purchase of a
species is a one-turn commitment rather than a three-turn gamble.

**The repair pass is not cosmetic, and it is the empty structure that needs it
most.** ``BUILD_*`` and ``PLACE`` are different turns, so every animal passes
through a bare structure, and a COW role standing on a ``COOP`` with no
``"animal"`` is the exact condition for the priority-3
``replace_incompatible_structure`` dig. Without the repair below the donor
demolishes the coop in that gap, every time, and the bird never lands. The
mismatch dig spares a tile that already carries an animal, so a *placed* goose is
not destroyed by a wrong role -- but the role is what ``_seed_needs``, the state's
``role_targets`` block and the liquidation path read, so it is repaired too.

That dig is also the warm slot's whole cost: guess the species wrong and the donor
spends one hand-turn replacing the structure. Guessing is bounded to one tile, and
the alternative -- the donor's unconditional four -- is what this module exists to
stop.
"""

from __future__ import annotations

from agents.dqn_market_free_v3e import donor
from agents.dqn_market_free_v3e.planting import market_influenced_roles

#: Every animal this module houses, in `donor.ANIMALS` order.
#:
#: Named rather than inlined because `order_space.FIELD_ANIMALS` has to agree with
#: it exactly: an action the head can emit but this module never houses strands the
#: purchase, and a role this module assigns that the head cannot buy never fills.
ANIMALS: tuple[str, ...] = tuple(donor.ANIMALS)

#: The one animal outside the donor's own herd sequence. Kept as a name because
#: `order_space.FIELD_ANIMALS` reads it and because the coop is the structure the
#: donor's planner has no path to at all.
GOOSE = "GOOSE"

#: What a goose stands on, out of the simulator's two structures.
COOP = str(donor.ANIMALS[GOOSE]["structure"])

#: Bare structures kept ahead of demand, across all species.
#:
#: One, not zero and not one per species. Zero is what makes a goose cost three
#: turns where a cow costs one, which is the asymmetry this module removes; one per
#: species is the donor's unconditional herd again in a smaller costume, and pays
#: for two structures the policy may never fill.
WARM_SLOTS = 1


def _tile_at(tiles: list, position: tuple[int, int]) -> object:
    """The tile under a board position, or None when the board does not reach it.

    `_reserved_animal_slots` reads the same `farm` this does, so an out-of-range
    position should be impossible; returning None rather than raising keeps a
    malformed observation costing one turn's herd rather than the episode.
    """

    x, y = position
    if not 0 <= y < len(tiles):
        return None
    row = tiles[y] or []
    if not 0 <= x < len(row):
        return None
    return row[x]


def _animal_on(tile: object) -> str | None:
    if isinstance(tile, dict):
        animal = tile.get("animal")
        if animal in donor.ANIMALS:
            return str(animal)
    return None


def _bare_structure(tile: object) -> str | None:
    """The kind of a structure with nothing living on it, or None.

    Reached two ways: the turn after `BUILD_*` and before `PLACE`, and after the
    simulator's escape rule -- two consecutive unfed days revert the tile to a bare
    structure and the animal is gone. Both want the same role.
    """

    if isinstance(tile, dict) and "animal" not in tile:
        kind = tile.get("kind")
        if kind in ("COOP", "PASTURE"):
            return str(kind)
    return None


def _is_claimable(tile: object) -> bool:
    """Whether a reserved animal slot can be turned over to a waiting animal.

    Empty ground and weeds only. A live animal is never displaced -- the donor
    would dig its pasture out from under it, which is the same bug this module
    exists to prevent, pointed the other way.
    """

    if tile is None:
        return True
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def _species_for(structure: str, waiting: dict[str, int]) -> str:
    """Which species a bare structure should be roled for.

    A structure is species-ambiguous only for the pasture, which COW and SHEEP
    share. Prefer whichever is actually waiting in the shed; fall back to the first
    species of that structure in `donor.ANIMALS` order so the role is never empty.
    """

    candidates = [
        animal
        for animal in ANIMALS
        if str(donor.ANIMALS[animal]["structure"]) == structure
    ]
    for animal in candidates:
        if waiting.get(animal, 0) > 0:
            return animal
    return candidates[0]


def remaining_yield_events(animal: str, day: int) -> int:
    """Production events an animal bought today can still deliver this season.

    The same arithmetic `lots.unplaced_events` prices the waste term with, repeated
    here rather than imported: `lots` is training-side accounting and this module
    ships inside the Kaggle bundle. `tests/test_dqn_market_free_v3e.py` pins the two
    equal.
    """

    rule = donor.ANIMALS[animal]
    offset = int(day) + int(rule["first"])
    interval = max(1, int(rule["interval"]))
    last = donor.TOTAL_DAYS - 1
    low = max(0, int(day) + 1 - offset)
    high = last - offset
    if high < low:
        return 0
    return high // interval - -(-low // interval) + 1


def warm_slot_species(
    obs: dict, day: int, owned: dict[str, int] | None = None
) -> str | None:
    """Which species to keep a bare structure for, or None to keep none.

    Gross coins the season can still pay per coin of purchase price, priced at the
    *midpoint* of what its own output does to the market rather than at the current
    quote. That is the whole difference between the three animals and the reason the
    quote alone is the wrong scorer: a goose's 26 remaining events and a cow's 11 are
    within 2% of each other at base price, but EGG is `T=332, above log 0.20` and
    MILK is `T=122, above linear 1.60`, so the eggs sell into a market that barely
    moves and the milk sells into one that halves.

    Deliberately not a market strategy and deliberately not the network's job: the
    head decides *whether* to buy, this decides only which empty structure is
    standing when it does.

    `owned` is what makes it self-correcting rather than a constant. The price
    impact is priced over the *whole* herd's remaining output, one animal larger, so
    each cow added pushes MILK further down its own steep curve while EGG's flat one
    barely notices -- a single animal of each species scores within 4%, and by the
    third cow the goose wins outright. That is the fleet effect the Kaggle loss was
    made of, and it cannot be seen one animal at a time.
    """

    market = (obs or {}).get("market", {}) or {}
    inventory = market.get("inventory", {}) or {}
    best: tuple[float, str] | None = None
    for animal in ANIMALS:
        rule = donor.ANIMALS[animal]
        product = str(rule["product"])
        units = float(remaining_yield_events(animal, day))
        if units <= 0:
            continue
        herd = 1.0 + float((owned or {}).get(animal, 0))
        stock = float(inventory.get(product, donor.MARKET_I0) or donor.MARKET_I0)
        price = float(donor._price_at(product, stock + units * herd / 2.0, obs))
        score = units * price / float(rule["cost"])
        if best is None or score > best[0]:
            best = (score, animal)
    return None if best is None else best[1]


def herd_influenced_roles(
    obs: dict,
    farm: dict,
    private: dict,
    donor_roles: dict,
) -> dict:
    """Rewrite the donor's animal roles to follow the animals actually committed."""

    roles = dict(donor_roles)
    tiles = farm.get("tiles", []) or []
    day = int(obs.get("day", 0) or 0)

    placed = donor._farm_animal_counts(farm)
    waiting = {
        animal: int(donor._private_item_total(private, animal)) for animal in ANIMALS
    }
    owned_total = sum(int(placed.get(animal, 0)) for animal in ANIMALS) + sum(
        waiting.values()
    )

    housed: set[tuple[int, int]] = set()
    #: Bare structures pass one handed to an animal that is actually waiting, as
    #: opposed to a speculative role on a structure nothing is queued for. The
    #: second sweep of pass three may take the latter for a species of another
    #: kind; never the former.
    demanded: set[tuple[int, int]] = set()

    # Pass one: every animal already on the board, and every structure already
    # built, keeps a role of the matching species. This runs over the tiles rather
    # than over `roles`, because a structure the donor never planned is absent from
    # `roles` entirely and its `_unit_actions` back-fill only rescues tiles that
    # already carry an "animal" -- a built-but-unfilled structure is invisible to it
    # and would never be placed on.
    for y, row in enumerate(tiles or []):
        for x, tile in enumerate(row or []):
            animal = _animal_on(tile)
            if animal is not None:
                roles[(x, y)] = ("ANIMAL", animal)
                housed.add((x, y))
                continue
            structure = _bare_structure(tile)
            if structure is not None:
                species = _species_for(structure, waiting)
                roles[(x, y)] = ("ANIMAL", species)
                housed.add((x, y))
                if waiting.get(species, 0) > 0:
                    waiting[species] -= 1
                    demanded.add((x, y))

    # Pass two used to strip every speculative role here -- a role on a tile with no
    # structure and no animal -- on the grounds that it is a `BUILD_` job for a herd
    # that was never bought, the donor emits four an episode, and 52% of episodes
    # never bought the animal. It is gone, for two reasons that are both measurable.
    #
    # The protection was already one layer down. `donor._field_jobs` emits the
    # `BUILD_` only when `left >= animal_rule["first"] + 2` and not at all during
    # liquidation, so a speculative role the season can no longer pay back produces
    # no job and costs nothing. Stripping it bought nothing the donor's own build
    # window was not already refusing; a payback gate on this loop was measured to
    # be bit-identical to deleting it outright, over 48 episodes, for that reason.
    #
    # And the 52% was measured on a learner that was not buying animals, which makes
    # it circular: the strip deletes the housing, the housing is what makes a
    # purchase land, and the purchase is what the statistic then fails to see. Both
    # sides of the fork buy now -- `market-free-splitladder-v1` averages 9.7 cows,
    # 7.4 sheep and 8.9 geese an episode -- so the premise no longer holds.
    #
    # What it cost, `sim.arena` over 24 seeds in both orientations, everything else
    # equal and the donor's own market rules on both sides:
    #
    #     field role plan            ours   pilkwang   win
    #     overlay with the strip   50,240    122,515  0.00
    #     overlay without it       83,525     86,111  0.35
    #
    # The strip was the whole of this module's cost. Passes one, three and four --
    # which are what opens the coop the donor's planner cannot reach at all -- are
    # untouched and are what `docs/dqn_market_free_animal_products.md` is about.

    liquidating = bool(day >= donor.TOTAL_DAYS - 1)
    pending = sum(max(0, count) for count in waiting.values())
    slots, _zones = donor._reserved_animal_slots(farm)

    # Pass three: an animal sitting in the shed or a worker's hands has nowhere to go
    # until a slot is assigned to it, and the donor only ever assigns COW and SHEEP.
    # Nearest-to-shed first, which is `_reserved_animal_slots`' own ordering. It
    # matters more for a goose than for the rest of the herd: `interval` 1 against
    # `max_held` 4 is a harvest visit every three days on top of the daily feed.
    for position in slots:
        if pending <= 0:
            break
        if position in housed or not _is_claimable(_tile_at(tiles, position)):
            continue
        species = next(
            (animal for animal in ANIMALS if waiting.get(animal, 0) > 0), None
        )
        if species is None:
            break
        roles[position] = ("ANIMAL", species)
        housed.add(position)
        waiting[species] -= 1
        pending -= 1

    # Pass three, second sweep: a waiting animal with no claimable ground left takes
    # over a bare structure of the wrong kind. Pass one roled every bare structure
    # for a species of its own kind -- a bare PASTURE is COW or SHEEP, never GOOSE --
    # and the sweep above only claims empty dirt and weeds, so a goose bought onto a
    # board whose reserved slots are all standing pasture got a role from nobody
    # and sat in the shed for the rest of the season. Measured on
    # `market-free-crop-v1` generation 12: 5.0 geese bought an episode, at most
    # 1.3 ever on the board, while `order_space.animal_room` -- which counts a bare
    # structure of any kind as housing -- kept the `BUY_ANIMAL GOOSE` rows open.
    #
    # Re-roling is what starts the donor's own chain: a GOOSE role on a bare
    # PASTURE is the exact condition for its priority-3
    # `replace_incompatible_structure` dig, then `BUILD_COOP`, then `PLACE`. Only a
    # bare structure is taken -- the mismatch dig spares a tile with an animal on it,
    # and so does this -- and only one that no waiting animal of the right kind is
    # already queued for: pass one decremented `waiting` for those, so anything
    # still positive here is a species the board's structures cannot serve.
    if pending > 0:
        for position in slots:
            if pending <= 0:
                break
            if position in demanded:
                continue
            tile = _tile_at(tiles, position)
            structure = _bare_structure(tile)
            if structure is None:
                continue
            species = next(
                (
                    animal
                    for animal in ANIMALS
                    if waiting.get(animal, 0) > 0
                    and str(donor.ANIMALS[animal]["structure"]) != structure
                ),
                None,
            )
            if species is None:
                break
            roles[position] = ("ANIMAL", species)
            housed.add(position)
            waiting[species] -= 1
            pending -= 1

    # Pass four: the warm slot. One bare structure ahead of demand, so the first
    # purchase of a species lands the turn after it is bought rather than three turns
    # later. Suppressed once nothing bought today could still produce, and during
    # liquidation, when a structure is a hand-turn spent on an asset that cannot pay.
    if liquidating:
        return roles
    # Free capacity is a bare structure with nothing already queued to fill it. A
    # slot claimed in pass three is a build for an animal that is bought and waiting,
    # so it is demand, not headroom, and must not count here.
    bare = sum(
        1
        for row in (tiles or [])
        for tile in (row or [])
        if _bare_structure(tile) is not None
    )
    if bare - sum(max(0, count) for count in waiting.values()) >= WARM_SLOTS:
        return roles
    species = warm_slot_species(
        obs,
        day,
        {
            animal: int(placed.get(animal, 0))
            + int(donor._private_item_total(private, animal))
            for animal in ANIMALS
        },
    )
    if species is None:
        return roles
    for position in slots:
        if position in housed or not _is_claimable(_tile_at(tiles, position)):
            continue
        roles[position] = ("ANIMAL", species)
        break

    # An animal bought with no slot left, or bought too late for the donor's
    # `left >= first + 2` build window, stays in the shed as dead capital. That is
    # deliberate and not masked away: the action space prices its own mistakes.
    return roles


def field_roles(obs: dict, farm: dict, private: dict) -> dict:
    """The whole role plan the field layer runs on, in the one order that is safe.

    Animals first, seeds second. `herd_influenced_roles` may claim a reserved animal
    slot the donor left bare, and the planting overlay only ever rewrites `CROP`
    roles, so running it after cannot take the slot back -- the reverse order would
    let a seed claim ground a bought animal needs.

    Every caller that needs roles goes through here: `main.py` for the live turn,
    `training/clone.py` when it rebuilds a demonstrator's turn, and
    `order_space`'s fallbacks when no plan was threaded in. A caller that reached
    for `donor._role_plan` directly would rebuild the donor's four unconditional
    pastures and disagree with the board the agent is actually standing on.
    """

    return market_influenced_roles(
        obs,
        farm,
        private,
        herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm)),
    )
