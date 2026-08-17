"""Let a bought goose steer the donor's animal roles onto a coop.

The donor already knows how to build a structure, place an animal on it, feed it,
and harvest it -- every one of those jobs reads ``ANIMALS[role_item]`` rather than
a species name, so a ``("ANIMAL", "GOOSE")`` role drives the whole chain with no
donor change. Its ``_role_plan`` is the only thing that never emits one:
``assigned = {"COW": 0, "SHEEP": 0}`` and the two-element candidate tuples around
it mean the sequence is COW/SHEEP and nothing else.

This adapter is demand-driven and owns no policy of its own. It never decides that
a goose is worth having; it reacts to one the network has already bought, and to
coops that already exist on the board. Every crop role and every live COW or SHEEP
remains the donor's responsibility.

**The repair pass is not cosmetic, and it is the empty coop that needs it most.**
``BUILD_COOP`` and ``PLACE GOOSE`` are different turns, so every goose passes
through a bare coop, and a COW role standing on a ``COOP`` with no ``"animal"`` is
the exact condition for the priority-3 ``replace_incompatible_structure`` dig.
Without the repair below the donor demolishes the coop in that gap, every time,
and the bird never lands. The mismatch dig spares a tile that already carries an
animal, so a *placed* goose is not destroyed by a wrong role -- but the role is
what ``_seed_needs``, the state's ``role_targets`` block and the liquidation path
read, so it is repaired too.
"""

from __future__ import annotations

from agents.dqn_market_free import donor

#: The one animal this agent reaches outside the donor's herd sequence.
#:
#: Named rather than inlined because `order_space.FIELD_ANIMALS` has to agree with
#: it exactly: an action the head can emit but this module never houses strands the
#: purchase, and a role this module assigns that the head cannot buy never fills.
GOOSE = "GOOSE"

#: What a goose stands on, out of the simulator's two structures.
COOP = str(donor.ANIMALS[GOOSE]["structure"])


def _tile_at(tiles: list, position: tuple[int, int]) -> object:
    """The tile under a board position, or None when the board does not reach it.

    `_reserved_animal_slots` reads the same `farm` this does, so an out-of-range
    position should be impossible; returning None rather than raising keeps a
    malformed observation costing one turn's goose rather than the episode.
    """

    x, y = position
    if not 0 <= y < len(tiles):
        return None
    row = tiles[y] or []
    if not 0 <= x < len(row):
        return None
    return row[x]


def _is_goose_tile(tile: object) -> bool:
    return isinstance(tile, dict) and tile.get("animal") == GOOSE


def _is_bare_coop(tile: object) -> bool:
    """A coop with nothing living on it.

    Reached two ways: the turn after `BUILD_COOP` and before `PLACE GOOSE`, and
    after the simulator's escape rule -- two consecutive unfed days revert the tile
    to a bare structure and the bird is gone. Both want the same role.
    """

    return isinstance(tile, dict) and tile.get("kind") == COOP and "animal" not in tile


def _is_claimable(tile: object) -> bool:
    """Whether a reserved animal slot can be turned over to a goose.

    Empty ground and weeds only. A live animal is never displaced -- the donor
    would dig its pasture out from under it, which is the same bug this module
    exists to prevent, pointed the other way.
    """

    if tile is None:
        return True
    return isinstance(tile, dict) and tile.get("kind") == "WEED"


def goose_influenced_roles(
    obs: dict,
    farm: dict,
    private: dict,
    donor_roles: dict,
) -> dict:
    """Overlay goose roles onto the donor's plan, driven by held birds and coops."""

    roles = dict(donor_roles)
    tiles = farm.get("tiles", []) or []
    housed: set[tuple[int, int]] = set()

    # Pass one: every goose already on the board, and every coop already built,
    # keeps a goose role. This runs over the tiles rather than over `roles`, because
    # a coop the donor never planned is absent from `roles` entirely and its
    # `_unit_actions` back-fill only rescues tiles that already carry an "animal" --
    # a built-but-unfilled coop is invisible to it and would never be placed on.
    for y, row in enumerate(tiles or []):
        for x, tile in enumerate(row or []):
            if _is_goose_tile(tile) or _is_bare_coop(tile):
                roles[(x, y)] = ("ANIMAL", GOOSE)
                housed.add((x, y))

    # Pass two: a goose sitting in the shed or a worker's hands has nowhere to go
    # until a slot is assigned to it, and the donor will never assign one.
    waiting = donor._private_item_total(private, GOOSE) - sum(
        1 for position in housed if _is_bare_coop(_tile_at(tiles, position))
    )
    if waiting <= 0:
        return roles

    # Nearest-to-shed first, which is `_reserved_animal_slots`' own ordering. It
    # matters more for a goose than for the rest of the herd: `interval` 1 against
    # `max_held` 4 is a harvest visit every three days on top of the daily feed.
    slots, _zones = donor._reserved_animal_slots(farm)
    for position in slots:
        if waiting <= 0:
            break
        if position in housed:
            continue
        if not _is_claimable(_tile_at(tiles, position)):
            continue
        roles[position] = ("ANIMAL", GOOSE)
        housed.add(position)
        waiting -= 1

    # A goose bought with no slot left, or bought too late for the donor's
    # `left >= first + 2` build window, stays in the shed as dead capital. That is
    # deliberate and not masked away: the action space prices its own mistakes.
    return roles
