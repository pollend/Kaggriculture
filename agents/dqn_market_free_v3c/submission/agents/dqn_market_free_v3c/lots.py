"""FIFO lot matching: which purchase paid for which sale, and how long it took.

**Measurement only.** `reward.py` needs no attribution at all: it pays the delta of
money plus held cost basis, and a basis is a sum over what is held rather than a
question about which purchase paid for which sale. Six reward legs used to read this
ledger -- `SEED_WASTE_WEIGHT` and its five siblings, each a per-lot reconstruction of
coins the stream dropped before the delta was paid -- and they are gone. What the
matching still answers is every column an update logs: `seed_holding_turns`,
`seed_margin`, `unrecovered_seed_share`, `stock_resale_share` and the rest, which are
now how a run *checks* the reward rather than an input to it.

It reconstructs the buy-to-sale chain by accounting rather than by discounting:
read a trace of what was held and what the wallet did, work out which seed lot
each sale realised, report it.

**FIFO is a convention, not a measurement.** Shed units are fungible, so
attributing the oldest sale to the oldest lot is a choice -- the right one here
because the quantity being placed is a *turn index*, and the oldest lot is the one
whose purchase decision is furthest from its payoff.

Animals are matched too, on the same two legs. A seed is consumed into a plant and
a plant has a `max_yield`; an animal is a durable asset with no lifetime cap, so its
capacity is a *deadline* rather than a ceiling -- how many end-of-day production
events the rest of the season can still deliver. `remaining_events` is that count and
`animal_census` is the per-turn trace it reads, so `MILK` and `WOOL` sales attribute
to the COW and SHEEP that produced them instead of vanishing into
`unattributed_proceeds`.

One blind spot left: the last recorded turn's effect, because a diff needs a
following sample.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from agents.dqn_market_free_v3c import donor

#: Simulator turns in one in-game day.
#:
#: Fallback only, for the land potential's day when an observation carries none. The
#: live value comes off `configuration.turnsPerDay` in `main.py`; this is the
#: simulator's default and matches `donor.TOTAL_DAYS * 24 == 720` episode steps. It
#: lives here rather than in `collect` because the three trace builders that need it
#: now share one.
TURNS_PER_DAY = 24

#: The crops a seed purchase can create, in `donor.CROPS` order. Both trace
#: builders index their seed arrays by this, so the ledger never sees a name.
CROP_ITEMS: tuple[str, ...] = tuple(donor.CROPS)

#: Every item that can sit in the shed and be sold. Wider than `CROP_ITEMS`
#: because `BUY_PRODUCT` opens WHEAT and FERTILIZER as bought stock with their
#: own cost basis, and because a sale of something this ledger has no lot for
#: must still be *seen* -- an unattributed sale is a diagnostic, not a silence.
STOCK_ITEMS: tuple[str, ...] = (
    *CROP_ITEMS,
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)

#: How many sold units one purchased unit is allowed to account for.
#:
#: A seed becomes a plant and a plant yields up to `max_yield` units, so one bought
#: seed can legitimately be the origin of six melons. Without a cap the first lot of
#: the season absorbs every sale in it and every later purchase looks unpaid.
#:
#: **A loose upper bound.** Realised yield is routinely a fraction of the cap, and
#: two sinks are invisible to a wallet diff: end-of-day shed overflow is discarded,
#: and unharvested non-ongoing crops rot. Both are "stock down, money flat", which
#: matches no rule below, so destroyed units would keep their room forever.
#: `_reconcile` closes that against the trace rather than guessing.
#:
#: Bought stock (`BUY_PRODUCT`) resells one for one, so its capacity is its size.
_YIELD_CAP: dict[str, int] = {
    crop: max(1, int(rule["max_yield"])) for crop, rule in donor.CROPS.items()
}

#: Where each item sits in `STOCK_ITEMS`, so the animal ceiling can read a product's
#: shed count without re-deriving the index every turn.
_STOCK_INDEX: dict[str, int] = {item: index for index, item in enumerate(STOCK_ITEMS)}

#: The animals a `BUY_ANIMAL` can create, in `donor.ANIMALS` order.
#:
#: Declared before `_TRADED_ONLY` below, which subtracts the products from
#: `STOCK_ITEMS` to find the items no yield ceiling covers.
#:
#: All three, not the two the action space happens to reach: `order_space` filters
#: GOOSE out through `FIELD_ANIMALS`, which is a policy decision and not a rule of the
#: simulator. A third column costs nothing and survives someone re-enabling it.
ANIMAL_ITEMS: tuple[str, ...] = tuple(donor.ANIMALS)

#: What each animal produces. The lot is opened under the *product* name so the sale
#: attribution in `_attribute` needs no animal-specific path: a MILK sale walks
#: `queues["MILK"]`, which now holds COW lots.
_PRODUCT_FOR_ANIMAL: dict[str, str] = {
    animal: str(rule["product"]) for animal, rule in donor.ANIMALS.items()
}

#: Shed items whose lots no yield ceiling reconciles: FERTILIZER, and anything else
#: that is neither a crop nor an animal's product.
#:
#: The two ceilings in `match` iterate `CROP_ITEMS` and `ANIMAL_ITEMS`, so a
#: FERTILIZER lot -- which only a `BUY_PRODUCT` can open -- was reconciled by
#: neither. Units spread on the tiles are "stock down, money flat", matching no rule
#: in `match`, so the lot kept its full room forever, stayed at the head of the FIFO
#: and absorbed a later sale a later purchase actually paid for. That is exactly the
#: bias `_reconcile` exists to close, left uncovered for this one item.
#:
#: Derived rather than spelled `("FERTILIZER",)` so a new tradeable that is not grown
#: and not milked is covered the day it is added to `STOCK_ITEMS`.
_TRADED_ONLY: tuple[str, ...] = tuple(
    item
    for item in STOCK_ITEMS
    if item not in _YIELD_CAP and item not in set(_PRODUCT_FOR_ANIMAL.values())
)

#: The last in-game day on which an animal can still produce.
#:
#: **29, not `TOTAL_DAYS`.** Production happens in `_end_of_day`, which the simulator
#: runs when `(step + 1) % turnsPerDay == 0`, and DONE fires at `episodeSteps - 2` =
#: 718. The last boundary at or below that is step 695, whose `next_day` is 29, so
#: day 29's own end-of-day never runs. Anchoring on 30 would say a cow bought on day
#: 22 can still milk; it cannot, and `order_space` has no day gate on `BUY_ANIMAL`, so
#: that purchase is reachable and needs to read as the dead weight it is.
LAST_YIELD_DAY = donor.TOTAL_DAYS - 1


def remaining_events(animal: str, placed_day: int, day: int) -> int:
    """How many production events a placed animal can still deliver after `day`.

    The animal analogue of `_YIELD_CAP`, and necessarily a function of the calendar
    rather than a constant: an animal has no lifetime yield cap, so what bounds it is
    the season ending. The arithmetic is `donor._animal_produces_tonight`'s, solved
    for the count instead of evaluated one day at a time -- an event lands on every
    day `n` with `n - placed_day - first >= 0` and `(n - placed_day - first) %
    interval == 0`, for `n` from `day + 1` through `LAST_YIELD_DAY`.

    `max_held` is not in this and should not be: a tile sitting on six unharvested
    units stops accumulating, so counting the clamped events over-states, and an
    over-statement can only ever fail to trim a claim that was already impossible --
    the same direction `_reconcile` requires.
    """

    rule = donor.ANIMALS[animal]
    offset = int(placed_day) + int(rule["first"])
    interval = max(1, int(rule["interval"]))
    low = max(0, int(day) + 1 - offset)
    high = LAST_YIELD_DAY - offset
    if high < low:
        return 0
    return high // interval - -(-low // interval) + 1


def unplaced_events(animal: str, day: int) -> int:
    """`remaining_events` for an animal bought but not yet standing on a structure.

    Buying and placing are always different turns -- `BUY_ANIMAL` lands in the shed,
    and `PICKUP` then `PLACE` are separate unit actions -- so an animal's capacity has
    to be opened against the best case, which is being placed the same day. Counting
    only placed animals would trim a lot to nothing on its own purchase turn, and
    `_reconcile` never gives room back.

    Zero from day 22 for a cow and day 24 for a sheep, which is what makes a purchase
    past the deadline read as fully unrecovered on the turn it happens.
    """

    return remaining_events(animal, day, day)


def holdings(private: dict) -> tuple[list[int], list[int]]:
    """`(seeds, stock)` counts for one turn, in `CROP_ITEMS`/`STOCK_ITEMS` order.

    Shed and carried are summed, because the ledger only cares whether a unit is
    held -- `market_state` keeps them apart as separate features, so the two
    readings are deliberately not shared.

    Both the live recorder and the replay reader build their trace from this: a
    cloned episode and a self-collected one are only the same kind of sample if the
    trace is built the same way. A missing key reads as zero, so a malformed
    observation shows no trade rather than a phantom one.
    """

    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    carried = private.get("inventories", []) or []
    return (
        [int(seeds.get(crop, 0) or 0) for crop in CROP_ITEMS],
        [
            int(shed.get(item, 0) or 0)
            + sum(int((slot or {}).get(item, 0) or 0) for slot in carried)
            for item in STOCK_ITEMS
        ],
    )


def standing_plants(farm: dict) -> list[int]:
    """Per-crop count of tiles currently carrying a plant, in `CROP_ITEMS` order.

    The third leg of the held-cost-basis trace. Without it a seed leaving
    `private["seeds"]` for a tile reads as the cost basis vanishing, and the reward
    potential would charge for planting -- the one action the term exists to make
    cheaper.

    Read off `farm["tiles"]`, public for both seats, so a replay and a live rollout
    recover the identical number. Only `kind == "PLANT"` counts, and only for a crop
    this module knows, so a sixth crop reads as zero rather than raising.
    """

    counts = dict.fromkeys(CROP_ITEMS, 0)
    for row in farm.get("tiles") or []:
        for tile in row or []:
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue
            crop = tile.get("crop")
            if crop in counts:
                counts[crop] += 1
    return [counts[crop] for crop in CROP_ITEMS]


def animal_census(farm: dict, private: dict, day: int) -> tuple[list[int], list[int]]:
    """`(owned, produce_room)` for one turn, both in `ANIMAL_ITEMS` order.

    `owned` counts an animal wherever it is -- on a tile, in the shed, carried by a
    hand -- because all three are the same purchase and the ledger opens a lot off the
    count rising. It is the only sound read of *how many* were delivered: a market
    order is a request, and `_process_market` commits unit by unit and abandons the
    order on the first unit it cannot fill, so the op the policy chose says nothing
    about what arrived.

    `produce_room` is the ceiling `_reconcile` trims against, indexed by animal but
    counted in units of its product: what is already standing on the tile
    unharvested, plus every event the rest of the season can still deliver, plus the
    best case for anything still unplaced. A ceiling and not an estimate, so it can
    only remove a claim that was already impossible.

    **`pending_care_bonus` is counted, future ones are not.** A production event pops
    the whole accumulated bonus, so one event can deliver `1 + interval` units; a
    strict bound would be that multiple of the event count, which is three to four
    times loose and would leave `_reconcile` doing nothing. Counting the bonus in hand
    and under-counting the rest is the trade, and `unattributed_by_item` is the alarm
    that says it went wrong -- a MILK or WOOL entry there means the ceiling is tight,
    where before this it just meant "an animal sold something".
    """

    placed = donor._farm_animal_counts(farm)
    owned = []
    room = []
    for animal in ANIMAL_ITEMS:
        unplaced = int(donor._private_item_total(private, animal))
        owned.append(int(placed.get(animal, 0)) + unplaced)
        room.append(unplaced * unplaced_events(animal, day))

    index_of = {animal: index for index, animal in enumerate(ANIMAL_ITEMS)}
    for row in farm.get("tiles") or []:
        for tile in row or []:
            if not isinstance(tile, dict):
                continue
            animal = tile.get("animal")
            if animal not in index_of:
                continue
            # A missing `placed_day` reads as today, which is the same best case an
            # unplaced animal gets rather than a claim on the whole season.
            room[index_of[animal]] += (
                int(tile.get("yield_units", 0) or 0)
                + int(tile.get("pending_care_bonus", 0) or 0)
                + remaining_events(animal, int(tile.get("placed_day", day) or 0), day)
            )
    return owned, room


class Sample(NamedTuple):
    """One turn of every trace `episode_rewards` and this ledger read.

    Built here rather than three times over. `collect.Recorder.record_turn`,
    `clone._episode_record` and `diagnostics._lot_use` each used to open-code this,
    and they had already drifted: `collect` fell back to `turn // TURNS_PER_DAY` for
    a missing `day` while `clone` fell back to 0, which reads as "the whole season is
    still ahead" and feeds `reward.land_potential`'s remaining-usage term. A cloned
    episode and a self-collected one are only the same kind of sample if the trace is
    built the same way, so there is one builder.
    """

    quadrants: int
    day: int
    seeds: list[int]
    standing: list[int]
    stock: list[int]
    owned: list[int]
    produce_room: list[int]


def seat_view(observation: dict, seat: int = 0) -> tuple[dict, dict] | None:
    """`(farm, private)` for the seat an observation belongs to, or None.

    None means the observation carries no farm for its own player index, which is
    the malformed case every caller already skips rather than scores. `seat` is the
    fallback when the observation names no player -- the replay readers know which
    seat they asked for, so they pass it rather than defaulting to 0.
    """

    if not isinstance(observation, dict):
        return None
    farms = observation.get("farms") or []
    player = int(observation.get("player", seat) or seat)
    if not farms or not 0 <= player < len(farms):
        return None
    farm = farms[player]
    private = observation.get("private") or {}
    if not isinstance(farm, dict):
        return None
    return farm, private if isinstance(private, dict) else {}


def sample(observation: dict, farm: dict, private: dict, turn: int) -> Sample:
    """Every per-turn trace, from one observation.

    `unlocked_quadrants` defaults to what a fresh farm carries, so a malformed
    observation contributes no land potential rather than a spurious one, and the day
    falls back to this turn's own index the way `market_state` does -- a missing day
    must not read as "the whole season is still ahead".
    """

    held_seeds, held_stock = holdings(private)
    day = int(observation.get("day", turn // TURNS_PER_DAY) or 0)
    owned, produce_room = animal_census(farm, private, day)
    return Sample(
        quadrants=len(farm.get("unlocked_quadrants", []) or ["NW"]),
        day=day,
        seeds=held_seeds,
        standing=standing_plants(farm),
        stock=held_stock,
        owned=owned,
        produce_room=produce_room,
    )


@dataclass
class _Lot:
    """One purchase, and how much of its production is still unaccounted for."""

    item: str
    buy_turn: int
    units: int
    unit_cost: float
    capacity: int
    absorbed: int = 0
    proceeds: float = 0.0
    first_sale_turn: int | None = None
    #: The last turn a sale was attributed to this lot, None if it never sold.
    #:
    #: `first_sale_turn` answers "how long was this held", which is a diagnostic.
    #: This answers "when was this lot's outcome finally known", which is where the
    #: resolution half of the reward's split credit is paid. They differ on any lot
    #: sold across several turns, which is most of them.
    last_sale_turn: int | None = None
    #: True for a `BUY_PRODUCT` lot. Kept apart from a seed lot because the two
    #: answer different questions and used to be summed into one: `seed_spend`
    #: means coins committed to growing something, and wheat bought off the market
    #: to resell is not that.
    bought: bool = False
    #: The animal a `BUY_ANIMAL` lot was opened for, None for a seed or bought-stock
    #: lot. Kept as the name rather than a flag because it is what `summary` needs to
    #: say which species the coins went to, and it doubles as the `animal_lots`
    #: predicate -- which excludes bought lots for free, since nothing bought is ever
    #: an animal.
    animal: str | None = None

    @property
    def cost(self) -> float:
        return self.unit_cost * self.units

    @property
    def room(self) -> int:
        return max(0, self.capacity - self.absorbed)


@dataclass
class LotLedger:
    """The matched result: what to pay where, and what it says about behaviour."""

    #: `(buy_turn, sale_turn, coins)`. Coins are the sale proceeds attributed to
    #: that lot, always positive, always with `buy_turn <= sale_turn`. The ledger's
    #: audit trail: what `summary` aggregates, and the only place the matching
    #: itself can be inspected.
    contributions: list[tuple[int, int, float]] = field(default_factory=list)
    lots: list[_Lot] = field(default_factory=list)
    #: Sale coins no open lot could account for, which now always means the ledger
    #: missed a purchase or trimmed a lot too hard.
    #:
    #: This changed meaning. Animal produce used to land here by design, because
    #: nothing opened a lot for a cow; a MILK or WOOL entry now means the animal
    #: ceiling in `animal_census` was tighter than what the herd really delivered,
    #: which is the failure the care bonus can cause.
    unattributed_proceeds: float = 0.0
    #: The same coins split by item, because "an animal sold something" and "the
    #: melon ceiling is wrong" are different bugs and the scalar cannot tell them
    #: apart.
    unattributed_by_item: dict[str, float] = field(default_factory=dict)

    def seed_lots(self) -> list[_Lot]:
        """The lots a seed purchase opened, which is what every column here means.

        `not lot.bought` is load-bearing: WHEAT is both a crop and a `BUY_PRODUCT`
        item, so without it a bought-wheat lot lands in `seed_spend` while
        `seed_lots_matched` leaves it out of the count.
        """

        return [lot for lot in self.lots if lot.item in _YIELD_CAP and not lot.bought]

    def unrecovered(self) -> list[tuple[int, float]]:
        """`(buy_turn, coins)` for every seed lot whose cost did not come back.

        One definition of "unrecovered", read by both `summary`'s
        `unrecovered_seed_share` and `reward.SEED_WASTE_WEIGHT`. The diagnostic and
        the reward have to describe the same matching or a run reads one attribution
        and trains on another.

        Clamped at zero per lot, so a lot sold above cost contributes nothing rather
        than offsetting a lot that was wasted -- the charge is a deterrent on waste,
        not a running P&L.
        """

        return _unrecovered(self.seed_lots())

    def surplus(self) -> list[tuple[int, float]]:
        """`(buy_turn, coins)` for every seed lot that came back above its cost.

        The mirror of `unrecovered()`, read by `summary`'s `recovered_seed_surplus`
        and by `reward.SEED_GAIN_WEIGHT`. It exists because `unrecovered()` clamps at
        zero per lot, so on its own it scores a lot sold at 1.1x cost and one sold at
        6x cost identically: the reward could see a seed's loss and never its profit.

        **Capped at the lot's own cost**, which is what keeps the pair symmetric --
        the worst a lot can do to its buy turn is `-cost` and the best is `+cost`, so
        the two legs together are bounded by the season's seed spend. Uncapped, one
        melon lot sold at six times cost would pay five times what wasting it charges,
        and a linear-coin term that size would displace the compressive `outcome` as
        the objective.
        """

        return _surplus(self.seed_lots())

    def animal_lots(self) -> list[_Lot]:
        """The lots a `BUY_ANIMAL` opened.

        `lot.animal is not None` rather than a membership test on the product name:
        MILK and WOOL are in `STOCK_ITEMS` so that a sale is *seen*, and were an
        animal ever buyable off the market this predicate would still keep the two
        apart the way `not lot.bought` does for wheat.
        """

        return [lot for lot in self.lots if lot.animal is not None]

    def animal_unrecovered(self) -> list[tuple[int, float]]:
        """`(buy_turn, coins)` for every animal whose cost did not come back.

        The animal half of `unrecovered()`, read by `reward.ANIMAL_WASTE_WEIGHT`, and
        clamped per lot for the same reason. What it catches that the seed version
        cannot: an animal bought and never placed, and an animal bought after its
        deadline -- day 21 for a cow, day 23 for a sheep -- both of which are
        unpriced today and neither of which the action space forbids.
        """

        return _unrecovered(self.animal_lots())

    def animal_surplus(self) -> list[tuple[int, float]]:
        """`(buy_turn, coins)` for every animal that came back above its cost.

        Mirrors `surplus()` exactly, cap included. The cap binds less often here than
        it does on seeds -- a milked cow grosses several times its 400 coins -- so
        nearly every animal that produced at all clips at `+cost`, which is what keeps
        the pair bounded by the season's herd spend.
        """

        return _surplus(self.animal_lots())

    def bought_lots(self) -> list[_Lot]:
        """The lots a `BUY_PRODUCT` opened: stock bought off the market, not grown.

        The third sibling of `seed_lots()` and `animal_lots()`, and until it existed
        these lots were opened, queued, matched -- and then read by nothing. They are
        not in `seed_spend` (excluded on purpose, see `_Lot.bought`), not in
        `animal_spend`, and no reward leg saw them, while measurement over 20 stored
        `market-free-splitladder-v1` seats put 89% of all turnover through exactly
        these two items: 2,964 WHEAT and 1,758 FERTILIZER units bought an episode,
        84% and 86% of them resold.

        `lot.bought` rather than a membership test on the item name, for the reason
        `seed_lots` states from the other side: WHEAT is both a crop and a
        `BUY_PRODUCT` item, and only the flag separates the two.
        """

        return [lot for lot in self.lots if lot.bought]

    def stock_loss(self) -> list[tuple[int, float]]:
        """`(buy_turn, coins)` a bought lot lost on the units it actually resold.

        **Realised resale margin, not `unrecovered()`, and the difference is
        consumption.** Bought WHEAT eaten by the herd and bought FERTILIZER spread on
        the tiles both leave `proceeds == 0.0`, so the seed and animal definition --
        `cost - proceeds` over the whole lot -- would score *using the input you
        bought it for* identically to wasting it, and charge the farm for feeding its
        animals. `_cost_of_sold` counts only `unit_cost * absorbed`, so units that
        were never sold contribute nothing either way and only the round trip is
        priced.

        Clamped at zero per lot, exactly as `_unrecovered` is: the charge is a
        deterrent on a losing round trip, not a running P&L that a profitable lot can
        offset. `STOCK_GAIN_WEIGHT` is the other half of the pair.
        """

        return [
            (lot.buy_turn, max(0.0, _cost_of_sold(lot) - lot.proceeds))
            for lot in self.bought_lots()
        ]

    def stock_gain(self) -> list[tuple[int, float]]:
        """`(buy_turn, coins)` a bought lot earned above the cost of what it resold.

        The mirror of `stock_loss()`, capped at that same cost so the pair is bounded
        by resale turnover either way -- the invariant `surplus()`'s cap buys for the
        seed pair, and the reason a linear-coin term cannot displace `outcome` as the
        objective.

        It exists for the reason `surplus()` does: with only the loss leg, expected
        credit is zero for every purchase and only the expected charge differs, so
        the local optimum is to buy nothing -- a NOOP attractor, which is the failure
        mode this family has collapsed into repeatedly.
        """

        return [
            (
                lot.buy_turn,
                min(_cost_of_sold(lot), max(0.0, lot.proceeds - _cost_of_sold(lot))),
            )
            for lot in self.bought_lots()
        ]

    def split_legs(self, final_turn: int) -> dict[str, list[tuple[int, int, float]]]:
        """The six reward legs as `(buy_turn, resolution_turn, coins)`.

        Same matching and the same clamps as `unrecovered`/`surplus`, their animal
        twins and the bought-stock pair -- this only adds where each lot's outcome
        became known, so the reward can pay part of the charge at the buy and part
        where the state can see it. One method for all six so a caller cannot pair a
        seed charge computed here with an animal credit computed the old way.
        """

        seeds, animals = self.seed_lots(), self.animal_lots()
        bought = self.bought_lots()
        return {
            "unrecovered": _split(_unrecovered(seeds), seeds, final_turn),
            "surplus": _split(_surplus(seeds), seeds, final_turn),
            "animal_unrecovered": _split(_unrecovered(animals), animals, final_turn),
            "animal_surplus": _split(_surplus(animals), animals, final_turn),
            "stock_loss": _split(self.stock_loss(), bought, final_turn),
            "stock_gain": _split(self.stock_gain(), bought, final_turn),
        }

    def summary(self) -> dict[str, float]:
        """Per-episode behaviour columns, aggregated the way the land ones are."""

        seed_lots = self.seed_lots()
        spend = sum(lot.cost for lot in seed_lots)
        recovered = sum(lot.proceeds for lot in seed_lots)
        matched = [lot for lot in seed_lots if lot.first_sale_turn is not None]
        holding = sorted(
            float(lot.first_sale_turn - lot.buy_turn) for lot in matched  # type: ignore[operator]
        )
        animal_lots = self.animal_lots()
        animal_spend = sum(lot.cost for lot in animal_lots)
        animal_recovered = sum(lot.proceeds for lot in animal_lots)
        animal_matched = [lot for lot in animal_lots if lot.first_sale_turn is not None]
        animal_holding = sorted(
            float(lot.first_sale_turn - lot.buy_turn)  # type: ignore[operator]
            for lot in animal_matched
        )
        bought_lots = self.bought_lots()
        stock_spend = sum(lot.cost for lot in bought_lots)
        stock_recovered = sum(lot.proceeds for lot in bought_lots)
        stock_units = sum(lot.units for lot in bought_lots)
        stock_resold = sum(lot.absorbed for lot in bought_lots)
        return {
            "animal_lots": float(len(animal_lots)),
            "animal_lots_matched": float(len(animal_matched)),
            "animal_spend": float(animal_spend),
            "animal_proceeds": float(animal_recovered),
            "animal_margin": float(animal_recovered - animal_spend),
            # The distance the two animal legs exist to close, and much longer than
            # the seed one: a cow bought on day 3 first milks on day 11.
            "animal_holding_turns": median(animal_holding),
            "unrecovered_animal_cost": float(
                sum(coins for _, coins in self.animal_unrecovered())
            ),
            "recovered_animal_surplus": float(
                sum(coins for _, coins in self.animal_surplus())
            ),
            "seed_lots": float(len(seed_lots)),
            "seed_lots_matched": float(len(matched)),
            "seed_spend": float(spend),
            "seed_proceeds": float(recovered),
            "seed_margin": float(recovered - spend),
            # Median rather than mean: one lot bought on day 1 and sold on day 29
            # would otherwise set the number by itself.
            "seed_holding_turns": median(holding),
            # Summed rather than shared, because the reward charges coins per lot
            # and a share cannot be checked against it.
            "unrecovered_seed_cost": float(sum(coins for _, coins in self.unrecovered())),
            # The other leg the reward acts on, capped per lot the same way. Both
            # sides have to be visible or a run cannot tell a working pair from one
            # where only the charge ever fires.
            "recovered_seed_surplus": float(sum(coins for _, coins in self.surplus())),
            # Cost that never came back. Rises when the policy buys more than it can
            # plant, work or sell. Clamped at zero, because a negative share would
            # really be the margin, which `seed_margin` already carries.
            "unrecovered_seed_share": float(
                max(0.0, 1.0 - recovered / spend) if spend > 0 else 0.0
            ),
            "stock_lots": float(len(bought_lots)),
            # Coins put through `BUY_PRODUCT`, which until now appeared in no column
            # anywhere: not `seed_spend`, not `animal_spend`, and no reward leg. It is
            # the larger half of this agent's turnover by a wide margin.
            "stock_spend": float(stock_spend),
            "stock_proceeds": float(stock_recovered),
            "stock_margin": float(stock_recovered - stock_spend),
            # The realised round trip alone -- proceeds against the cost of the units
            # that actually left through a SELL. `stock_margin` mixes that with stock
            # bought and then consumed, which is a different decision.
            "stock_resale_margin": float(
                stock_recovered - sum(_cost_of_sold(lot) for lot in bought_lots)
            ),
            "stock_units": float(stock_units),
            "stock_resold_units": float(stock_resold),
            "stock_consumed_units": float(max(0, stock_units - stock_resold)),
            # **The number this pair was added for.** Bought units that were resold
            # rather than used. 0.84 for WHEAT and 0.86 for FERTILIZER on
            # `market-free-splitladder-v1`; the ladder champion buys wheat to feed a
            # herd and resells almost none of it.
            "stock_resale_share": float(
                stock_resold / stock_units if stock_units > 0 else 0.0
            ),
            "unattributed_proceeds": float(self.unattributed_proceeds),
        }


def _cost_of_sold(lot: _Lot) -> float:
    """What the units this lot actually resold had cost.

    `unit_cost * absorbed`, and the whole point is that it is not `lot.cost`. A bought
    lot resells one for one -- `capacity == units` when it opens -- so `absorbed` is
    the unit count that left through a SELL, and the units still in the shed, eaten by
    the herd or spread on the tiles are simply not in the number.

    Only bought lots use this. A seed lot's `absorbed` counts *harvested* units, whose
    basis is `seed / max_yield` rather than `unit_cost`, so the same expression would
    mean something else there; that is why the seed and animal legs keep `lot.cost`.
    """

    return float(lot.unit_cost) * float(lot.absorbed)


def _unrecovered(lots: Sequence[_Lot]) -> list[tuple[int, float]]:
    """`(buy_turn, coins)` a set of lots did not earn back, clamped at zero per lot.

    Shared by the seed and animal legs so the two cannot drift into meaning different
    things -- the clamp is what makes the charge a deterrent on waste rather than a
    running P&L, and one of the two silently dropping it would be invisible.
    """

    return [(lot.buy_turn, max(0.0, lot.cost - lot.proceeds)) for lot in lots]


def _surplus(lots: Sequence[_Lot]) -> list[tuple[int, float]]:
    """`(buy_turn, coins)` a set of lots earned above cost, capped at that cost.

    The cap is the half that bounds the pair: with both legs clamped to `+/- cost`,
    a season's two legs together cannot exceed what it spent either way.
    """

    return [
        (lot.buy_turn, min(lot.cost, max(0.0, lot.proceeds - lot.cost))) for lot in lots
    ]


def resolution_turn(lot: _Lot, final_turn: int) -> int:
    """The turn a lot's outcome stopped being open.

    A lot that sold resolves on its last attributed sale. A lot that never sold has
    no such turn -- it stayed sellable right up to the buzzer -- so it resolves at
    the end of the episode, which is the first moment "never sold" became a fact
    rather than a possibility.

    Read by `reward`'s split credit. `first_sale_turn` deliberately not used: it is
    when the lot *started* paying back, which is a holding-time diagnostic, not a
    resolution.
    """

    turn = lot.last_sale_turn
    return final_turn if turn is None else int(turn)


def _split(
    entries: Sequence[tuple[int, float]],
    lots: Sequence[_Lot],
    final_turn: int,
) -> list[tuple[int, int, float]]:
    """`(buy_turn, resolution_turn, coins)`, pairing each leg entry with its lot.

    Positional, not keyed: `_unrecovered` and `_surplus` both emit one entry per lot
    in lot order, so the zip is exact. A keyed join would have to key on `buy_turn`,
    which is not unique -- one turn can open lots for several items.
    """

    return [
        (buy_turn, resolution_turn(lot, final_turn), coins)
        for (buy_turn, coins), lot in zip(entries, lots)
    ]


def median(values: Sequence[float]) -> float:
    """Median of an already-sorted-or-not sequence, 0.0 when empty.

    Shared rather than local: `updater` had three inline copies of exactly this
    (`sort()` then `divmod(len, 2)`) behind the LR plateau, the benchmark ratchet and
    the population score, and a windowed median that disagrees with itself between
    two of those is a gate reading a different quantity than the diagnostic.
    """

    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle, odd = divmod(len(ordered), 2)
    if odd:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def match(
    seeds: Sequence[Sequence[int]],
    stock: Sequence[Sequence[int]],
    money: Sequence[float],
    ops: Sequence[str | None] | None = None,
    standing: Sequence[Sequence[int]] | None = None,
    owned: Sequence[Sequence[int]] | None = None,
    produce_room: Sequence[Sequence[int]] | None = None,
    days: Sequence[int] | None = None,
    #: Per-turn market orders as `(op, item, quantity, cash)`, cash signed the way
    #: the wallet moves. From `PendingTurn`, which quoted every order as it queued
    #: it. Optional: a caller that cannot recover the split -- a replay read by
    #: `clone.py` before it walks the turn -- gets v1's net-delta attribution.
    events: Sequence[Sequence[tuple[str, str | None, int, float]]] | None = None,
) -> LotLedger:
    """Reconstruct lots from per-turn holdings and the wallet.

    All of them are indexed by turn and sampled *before* that turn's action, which
    is what `Recorder.record_turn` stores and what a replay's step `t-1` carries.
    The effect of the action at turn `t` is therefore the difference between
    samples `t` and `t + 1`, which is exactly how `reward.py` indexes `raw`.

    `seeds[t]` and `standing[t]` are counted over `CROP_ITEMS`, `stock[t]` over
    `STOCK_ITEMS`. `ops[t]` is the market op the policy chose on turn `t` --
    `ACTIONS[chosen].op`, or None where the caller cannot recover it.

    One market order per turn (`main.py` puts a single `decision.wire()` on the
    wire), and the sim has no passive drains -- money moves only at BUY_SEED,
    BUY_PRODUCT, BUY_ANIMAL, BUY_LAND, HIRE and SELL -- so a turn's wallet
    movement belongs to that turn's order:

        seeds up                        BUY_SEED, cost from the crop table
        stock up,   op is BUY_PRODUCT   bought stock, cost is what was paid
        stock up,   otherwise           harvest, no cost of its own
        stock down, money up            SELL, proceeds are what arrived
        seeds down, stock flat          planted, still the same lot's cost
        owned up                        BUY_ANIMAL, cost from the animal table

    `owned[t]`, `produce_room[t]` and `days[t]` are the animal group and are read
    together or not at all; without them MILK and WOOL sales open no lot and fall
    through to `unattributed_by_item` exactly as they did before animals were matched.
    The group is deliberately *not* gated on `ops`: an animal purchase is the one
    thing a wallet diff cannot get wrong, because `owned` rising is the delivered
    count and the op is only ever the request. A purchase and an escape inside one
    step net to zero and the lot is missed, which is the ledger being allowed to miss
    a lot rather than to invent one.

    **The op is read, never inferred from the sign of the wallet move.** Field
    actions run before `_process_market` in the same step and `main.py` emits
    HARVEST every turn, so a harvest landing on a HIRE, BUY_LAND or BUY_ANIMAL turn
    reads as "stock up, money down" and opens a phantom lot priced at that outlay.
    `ops=None` falls back to the sign heuristic for a caller that cannot recover the
    decision; every caller in the tree passes it.

    Supplying the sequence is enough: a short `ops`, or a None sitting inside one, is
    read as "no purchase on that turn" rather than as a licence to fall back to the
    heuristic for it. The fallback is all-or-nothing, because a heuristic applied to
    the turns a caller happened not to resolve is exactly the silent phantom the
    parameter exists to prevent.

    Anything else -- stock falling with no money, a diff spanning a skipped turn --
    is left alone. The ledger is allowed to miss a lot; it is not allowed to invent
    one.
    """

    ledger = LotLedger()
    queues: dict[str, deque[_Lot]] = {item: deque() for item in STOCK_ITEMS}
    have_ops = ops is not None
    have_animals = owned is not None and produce_room is not None and days is not None
    turns = min(len(seeds), len(stock), len(money))
    if standing is not None:
        turns = min(turns, len(standing))
    if have_animals:
        turns = min(turns, len(owned), len(produce_room), len(days))  # type: ignore[arg-type]

    for turn in range(turns - 1):
        seed_delta = [
            int(seeds[turn + 1][index]) - int(seeds[turn][index])
            for index in range(len(CROP_ITEMS))
        ]
        stock_delta = [
            int(stock[turn + 1][index]) - int(stock[turn][index])
            for index in range(len(STOCK_ITEMS))
        ]
        money_delta = float(money[turn + 1]) - float(money[turn])
        op = ops[turn] if ops is not None and turn < len(ops) else None
        # Per-item cash for the turn, when the caller can supply it. v2 turns hold
        # several orders, so `money_delta` is a *net* and the two places below that
        # divide by it are wrong the moment a turn buys and sells at once: a
        # BUY_PRODUCT beside a larger SELL prices the bought lot at a negative basis,
        # and a sale on a spending turn is attributed whatever is left of the net
        # rather than what it earned. `events` splits the turn back into orders.
        bought_cash: dict[str, float] = {}
        sold_cash: dict[str, float] = {}
        if events is not None and turn < len(events):
            for event in events[turn] or ():
                event_op, event_item, _quantity, cash = event
                if event_op == "BUY_PRODUCT":
                    bought_cash[str(event_item)] = bought_cash.get(str(event_item), 0.0) + float(
                        cash
                    )
                elif event_op == "SELL":
                    sold_cash[str(event_item)] = sold_cash.get(str(event_item), 0.0) + float(cash)

        for index, crop in enumerate(CROP_ITEMS):
            bought = seed_delta[index]
            if bought <= 0:
                continue
            lot = _Lot(
                item=crop,
                buy_turn=turn,
                units=bought,
                unit_cost=float(donor.CROPS[crop]["seed"]),
                capacity=bought * _YIELD_CAP[crop],
            )
            ledger.lots.append(lot)
            queues[crop].append(lot)

        if have_animals:
            # Opened before the turn's sales for the same reason a seed lot is, even
            # though an animal cannot produce on the day it was bought: the two
            # openings then read the same way and neither has a special case.
            for index, animal in enumerate(ANIMAL_ITEMS):
                bought = int(owned[turn + 1][index]) - int(owned[turn][index])  # type: ignore[index]
                if bought <= 0:
                    continue
                product = _PRODUCT_FOR_ANIMAL[animal]
                lot = _Lot(
                    item=product,
                    buy_turn=turn,
                    units=bought,
                    unit_cost=float(donor.ANIMALS[animal]["cost"]),
                    # Best case: placed the same day it was bought. The per-turn
                    # ceiling below takes back whatever the policy actually gave up
                    # by leaving it in the shed.
                    capacity=bought * unplaced_events(animal, int(days[turn + 1])),  # type: ignore[index]
                    animal=animal,
                )
                ledger.lots.append(lot)
                queues[product].append(lot)

        for index, item in enumerate(STOCK_ITEMS):
            moved = stock_delta[index]
            # With `events` the question "was this bought" is answered per item by
            # the order list itself, and the op-vs-heuristic argument above does not
            # arise: a harvest landing on a HIRE turn has no BUY_PRODUCT event for
            # its item, so it cannot open a phantom lot.
            purchased = (
                item in bought_cash
                if events is not None
                else _is_bought_stock(op, money_delta, seed_delta, have_ops)
            )
            if moved > 0 and purchased:
                # Bought stock rather than harvested: it carries its own basis and
                # resells one for one.
                # `cash` is signed the way the wallet moves, so a purchase arrives
                # negative and the basis is its negation.
                outlay = -bought_cash[item] if events is not None else -money_delta
                lot = _Lot(
                    item=item,
                    buy_turn=turn,
                    units=moved,
                    unit_cost=outlay / moved,
                    capacity=moved,
                    bought=True,
                )
                ledger.lots.append(lot)
                queues[item].append(lot)
            elif moved < 0:
                # The sale's own proceeds where the events know them, the turn's net
                # otherwise -- and the net is only trustworthy when it is positive,
                # which is the v1 test kept for the v1 caller.
                proceeds = sold_cash.get(item, 0.0) if events is not None else money_delta
                if proceeds > 0:
                    _attribute(ledger, queues[item], -moved, proceeds, turn, item)

        if standing is not None:
            # Retire production that can no longer happen. Done after the turn's
            # sales, so a lot is still holding room for stock it is about to be
            # asked to account for.
            #
            # Every turn and every crop, against the crop's own remaining ceiling
            # -- not only once the crop is exhausted. The exhaustion-only rule was
            # the special case `seeds == standing == 0` of this one, and on a farm
            # that replants continuously it essentially never fired: measured over
            # 20 stored episodes, 60.7% of seed-lot yield capacity was never
            # absorbed, so the season's first lots held room for units nobody grew
            # and FIFO handed them every later sale. 83% of lots then read fully
            # unrecovered while the seeds had in aggregate returned 2.4x, which
            # `reward.SEED_WASTE_WEIGHT` bills as a charge on their buy turns.
            #
            # A ceiling rather than an estimate, so it can only ever remove a claim
            # that was already impossible: what a crop can still yield is bounded by
            # the units already in hand plus `max_yield` for every seed unplanted
            # and every plant standing. An ongoing crop mid-yield counts its plant
            # at the full cap again, which over-states and therefore never trims a
            # lot that could still produce.
            for index, crop in enumerate(CROP_ITEMS):
                ceiling = int(stock[turn + 1][index]) + _YIELD_CAP[crop] * (
                    int(seeds[turn + 1][index]) + int(standing[turn + 1][index])
                )
                _reconcile(queues[crop], ceiling)

        if have_animals:
            # The animal ceiling, on its own guard rather than the crops': without the
            # trace there is no room figure at all, and falling back to `stock` alone
            # would trim every animal lot to nothing on the turn it opened.
            #
            # Whatever the herd can still deliver, plus what is standing on the tiles
            # unharvested, plus what is already in the shed. The season ending is what
            # closes an animal lot -- there is no `max_yield` to exhaust -- so this is
            # also what charges a purchase made past the deadline: `unplaced_events`
            # is zero from day 22 for a cow, so the lot opens with no capacity and
            # reads fully unrecovered.
            for index, animal in enumerate(ANIMAL_ITEMS):
                product = _PRODUCT_FOR_ANIMAL[animal]
                ceiling = int(stock[turn + 1][_STOCK_INDEX[product]]) + int(
                    produce_room[turn + 1][index]  # type: ignore[index]
                )
                _reconcile(queues[product], ceiling)

        # The items neither ceiling above covers -- FERTILIZER today. Only a
        # `BUY_PRODUCT` opens a lot here and bought stock resells one for one, so the
        # units it can still account for are bounded by what is in the shed: there is
        # no yield to come and no herd to deliver it.
        #
        # Unconditional, unlike the two above. Those need a trace the caller may not
        # have (`standing`, `produce_room`); this reads `stock`, which `match`
        # requires. Fertilizer the farm *produced* opens no lot -- `COLLECT_FERTILIZER`
        # raises stock with no `BUY_PRODUCT` event -- so outstanding lot room is never
        # more than the shed count on its own account, and the ceiling can only trim a
        # claim that consumption already made impossible.
        for item in _TRADED_ONLY:
            _reconcile(queues[item], int(stock[turn + 1][_STOCK_INDEX[item]]))

    return ledger


def _is_bought_stock(
    op: str | None, money_delta: float, seed_delta: Sequence[int], have_ops: bool
) -> bool:
    """Whether stock arriving this turn was purchased rather than harvested.

    `have_ops` is whether the caller supplied an `ops` sequence at all, not whether
    this turn's entry is set. Keying on `op is not None` instead would silently
    reinstate the heuristic for any turn a supplied sequence left None -- and a
    phantom lot does more than mis-label itself: it joins the FIFO queue and absorbs
    sales a real seed lot is owed, so the real lot reads as unrecovered and
    `reward.SEED_WASTE_WEIGHT` bills a purchase that did come back.
    """

    if have_ops:
        return op == "BUY_PRODUCT"
    # No ops sequence at all: the pre-fix sign heuristic, kept only for that caller.
    return money_delta < 0 and not seed_delta_touches(seed_delta)


def _reconcile(queue: deque[_Lot], ceiling: int) -> None:
    """Cap a crop's outstanding room at what it can still deliver.

    `ceiling` is the units the crop could yet account for: stock in hand, plus
    `max_yield` for each seed and standing plant. Room above it is a claim on
    units that were never grown, rotted in the field, or overflowed the shed --
    all of which are "stock down, money flat" and match no rule in `match`, so
    without this they would keep their room forever. Leaving them open lets the
    oldest lot absorb sales a later purchase actually paid for, which is the bias
    `_YIELD_CAP` documents and which measurement found dominant.

    Trimmed from the back so the front of the queue keeps its room: FIFO hands
    the next sale to the oldest lot, and the oldest lot is the one whose purchase
    turn is furthest from its payoff.
    """

    room = sum(lot.room for lot in queue)
    excess = room - max(0, ceiling)
    for lot in reversed(queue):
        if excess <= 0:
            break
        trim = min(excess, lot.room)
        lot.capacity -= trim
        excess -= trim


def seed_delta_touches(seed_delta: Sequence[int]) -> bool:
    """Whether this turn also moved seeds, which makes the cash a seed purchase.

    Buying seed raises `seeds` and lowers money on the same turn, and WHEAT is
    both a crop and a `BUY_PRODUCT` item, so without this the same coins would
    open two lots.
    """

    return any(delta > 0 for delta in seed_delta)


def _attribute(
    ledger: LotLedger,
    queue: deque[_Lot],
    units: int,
    proceeds: float,
    turn: int,
    item: str,
) -> None:
    """Hand one sale's coins to the oldest lots with room left.

    `unit_price` is the turn's average, not each unit's own price. SELL walks the
    market curve down one unit at a time inside a single order
    (`kaggriculture.py:640-649`), so units within one sale really do fetch
    different coins. The average is exact for the total, which is what
    `seed_proceeds` and `seed_margin` aggregate, and wrong only for how a sale
    that straddles two lots splits between them.
    """

    unit_price = proceeds / max(1, units)
    for _ in range(units):
        while queue and queue[0].room == 0:
            queue.popleft()
        if not queue:
            ledger.unattributed_proceeds += unit_price
            ledger.unattributed_by_item[item] = (
                ledger.unattributed_by_item.get(item, 0.0) + unit_price
            )
            continue
        lot = queue[0]
        lot.absorbed += 1
        lot.proceeds += unit_price
        if lot.first_sale_turn is None:
            lot.first_sale_turn = turn
        lot.last_sale_turn = turn
        ledger.contributions.append((lot.buy_turn, turn, unit_price))
