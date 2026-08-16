"""FIFO lot matching: which purchase paid for which sale, and how long it took.

The reward stream pays nothing on the turn a purchase happens. `reward.py` builds
`shaped` from zeros and never adds the money delta to it -- `raw` is returned for
diagnostics and `replay.py` trains on `reward` alone -- so the per-turn ordering
is exactly the three lines `IDLE_PENALTY` records: an accepted order scores 0, a
NOOP -0.0002, a refused order -0.001. Buying a melon seed and buying nothing are
the same reward, and the only thing that ever distinguishes them is the terminal
outcome, reached through a bootstrap chain the length of a crop cycle.

`CROPS[crop]["first_yield_day"]` sizes that chain, and it is longer than this
module's first draft claimed. At `turnsPerDay = 24` the gap from BUY_SEED to the
first *legal* harvest is 48 turns for WHEAT and CARROT, 192 for TOMATO, and 240
for MELON and STRAWBERRY -- and `BUY_SEED` is processed after the turn's field
actions (`kaggriculture.py:927`), so the seed is not plantable until `t + 1`, and
the coins arrive later still, through harvest, the shed, and a SELL. Against
`n_step = 24` that is two bootstraps for wheat and ten for the melon
`planting.py` actually optimises for.

This module reconstructs that chain by accounting rather than by discounting. It
reads a trace of what was held and what the wallet did, works out which seed lot
each sale realised, and reports it. **The reward no longer consumes this**: the
credit `reward.py` pays for a purchase is `LOT_POTENTIAL_WEIGHT`, a potential
over held cost basis, which is policy-invariant and needs no attribution. What
the ledger is for now is measurement -- `seed_holding_turns`, `seed_margin` and
`unrecovered_seed_share`, the columns an update logs and a run is judged on.

**FIFO is a convention, not a measurement.** Units in the shed are fungible --
nothing in the observation says which seed produced which melon -- so attributing
the oldest sale to the oldest lot is a choice. It is the right choice here for
one reason: the quantity being placed is a *turn index*, and the oldest lot is
the one whose purchase decision is furthest from its payoff and therefore the one
the credit was missing from.

What it cannot see: the last recorded turn's effect, because a diff needs a
following sample; and anything the traces do not carry, which is why animals are
absent -- EGG, MILK and WOOL are produced by a bought animal rather than by a
consumed seed, and their cost basis belongs to a purchase this ledger would have
to model differently.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

from agents.dqn_market_free import donor

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
#: A seed becomes a plant and a plant yields up to `max_yield` units, so one
#: bought seed can legitimately be the origin of six melons. Without a cap the
#: first lot of the season would absorb every sale in it and every later purchase
#: would look unpaid; with the cap, a lot fills and the queue advances.
#:
#: **This is an upper bound and a loose one.** A non-ongoing crop starts at
#: `yield_units = 1` and only climbs toward `max_yield` through WATER inside a
#: window (`kaggriculture.py:418-430`), so realised yield is routinely a fraction
#: of the cap and the front lot would still never fill. Two sinks make it worse
#: and are invisible to a wallet diff: end-of-day shed overflow above
#: `shedCapacity` is discarded (`kaggriculture.py:830-845`) and unharvested
#: non-ongoing crops rot (`_decay_plants`, `:739-754`). Both are "stock down,
#: money flat", which matches no rule below, so destroyed units would keep their
#: room forever and every later lot would read unpaid.
#:
#: `_reconcile` closes all three against the trace instead of guessing: once a
#: crop has no seed and no standing plant, its queue can produce nothing more, so
#: the room left across it is exactly what is still in stock.
#:
#: Bought stock (`BUY_PRODUCT`) resells one for one, so its capacity is its size.
_YIELD_CAP: dict[str, int] = {
    crop: max(1, int(rule["max_yield"])) for crop, rule in donor.CROPS.items()
}


def holdings(private: dict) -> tuple[list[int], list[int]]:
    """`(seeds, stock)` counts for one turn, in `CROP_ITEMS`/`STOCK_ITEMS` order.

    Shed and carried are summed, because the ledger only cares whether a unit is
    held. `market_state` keeps them apart -- it feeds them as separate features --
    so the two readings are deliberately not shared, and this one is the shorter
    of the two.

    Both callers build the trace from this: `collect.Recorder.record_turn` live,
    and `clone.py` from a stored replay. A cloned episode and a self-collected one
    have to be the same kind of sample, which they can only be if the trace they
    carry is built the same way.

    A missing key reads as zero, so a malformed observation shows no trade on the
    diff rather than a phantom one.
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

    The third leg of the held-cost-basis trace, alongside `holdings`' two. Without
    it a seed leaving `private["seeds"]` for a tile reads as the cost basis
    vanishing, and the reward potential would charge for planting -- the one
    action the whole term exists to make cheaper.

    Read off `farm["tiles"]`, which is public state for both seats, so a replay
    and a live rollout recover the identical number. A weed or an empty tile is
    not a plant; only `kind == "PLANT"` counts, and only for a crop this module
    knows, so a schema that grows a sixth crop reads as zero rather than raising.
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
    #: True for a `BUY_PRODUCT` lot. Kept apart from a seed lot because the two
    #: answer different questions and used to be summed into one: `seed_spend`
    #: means coins committed to growing something, and wheat bought off the market
    #: to resell is not that.
    bought: bool = False

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
    #: that lot, always positive, always with `buy_turn <= sale_turn`.
    #:
    #: No longer read by the reward -- `LOT_POTENTIAL_WEIGHT` replaced the
    #: reallocation that consumed this. Kept as the ledger's audit trail: it is
    #: what `summary`'s aggregates are derived from and the only place the
    #: matching itself can be inspected.
    contributions: list[tuple[int, int, float]] = field(default_factory=list)
    lots: list[_Lot] = field(default_factory=list)
    #: Sale coins no open lot could account for. Animal produce lands here by
    #: design; anything else here means the ledger missed a purchase.
    unattributed_proceeds: float = 0.0

    def summary(self) -> dict[str, float]:
        """Per-episode behaviour columns, aggregated the way the land ones are."""

        seed_lots = [
            lot for lot in self.lots if lot.item in _YIELD_CAP and not lot.bought
        ]
        spend = sum(lot.cost for lot in seed_lots)
        recovered = sum(lot.proceeds for lot in seed_lots)
        matched = [lot for lot in seed_lots if lot.first_sale_turn is not None]
        holding = sorted(
            float(lot.first_sale_turn - lot.buy_turn) for lot in matched  # type: ignore[operator]
        )
        return {
            "seed_lots": float(len(seed_lots)),
            "seed_lots_matched": float(len(matched)),
            "seed_spend": float(spend),
            "seed_proceeds": float(recovered),
            "seed_margin": float(recovered - spend),
            # The headline: how many turns of bootstrap the credit used to have to
            # cross. Median rather than mean because a single lot bought on day 1
            # and sold on day 29 would otherwise set the number by itself.
            "seed_holding_turns": _median(holding),
            # Cost that never came back. Rises when the policy buys more than it
            # can plant, work or sell -- the failure a purchase bonus causes and
            # this term must not.
            # Clamped at zero: a season that sells for more than it spent has
            # recovered all of its cost, and a negative "share unrecovered" would
            # read as a ratio when it is really the margin, which
            # `seed_margin` above already carries.
            "unrecovered_seed_share": float(
                max(0.0, 1.0 - recovered / spend) if spend > 0 else 0.0
            ),
            "unattributed_proceeds": float(self.unattributed_proceeds),
        }


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2.0)


def match(
    seeds: Sequence[Sequence[int]],
    stock: Sequence[Sequence[int]],
    money: Sequence[float],
    ops: Sequence[str | None] | None = None,
    standing: Sequence[Sequence[int]] | None = None,
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

    **The op is read rather than inferred from the sign, and that is a fix rather
    than a refinement.** Field actions run before `_process_market` in the same
    step (`kaggriculture.py:927`) and `main.py` emits HARVEST every turn
    independently of the market decision, so a harvest landing on a HIRE,
    BUY_LAND or BUY_ANIMAL turn reads as "stock up, money down" and used to open
    a phantom lot -- dated to the *harvest* turn, and priced at the hire or land
    outlay divided by the units harvested. A 4,000-coin quadrant taken on a
    three-melon harvest booked melon seed at 1,333 against a real price of 80,
    straight into `seed_spend` and `unrecovered_seed_share`.

    With `ops=None` the old sign heuristic is kept, so a caller that cannot
    recover the decision degrades to the previous reading rather than losing the
    branch entirely. Every caller in the tree passes it.

    Anything else -- stock falling with no money, a diff that spans a skipped
    turn -- is left alone rather than guessed at. The ledger is allowed to miss a
    lot; it is not allowed to invent one.
    """

    ledger = LotLedger()
    queues: dict[str, deque[_Lot]] = {item: deque() for item in STOCK_ITEMS}
    turns = min(len(seeds), len(stock), len(money))
    if standing is not None:
        turns = min(turns, len(standing))

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

        for index, item in enumerate(STOCK_ITEMS):
            moved = stock_delta[index]
            if moved > 0 and _is_bought_stock(op, money_delta, seed_delta):
                # Bought stock rather than harvested: it carries its own basis and
                # resells one for one.
                lot = _Lot(
                    item=item,
                    buy_turn=turn,
                    units=moved,
                    unit_cost=-money_delta / moved,
                    capacity=moved,
                    bought=True,
                )
                ledger.lots.append(lot)
                queues[item].append(lot)
            elif moved < 0 and money_delta > 0:
                _attribute(ledger, queues[item], -moved, money_delta, turn)

        if standing is not None:
            # Retire production that can no longer happen. Done after the turn's
            # sales, so a lot is still holding room for stock it is about to be
            # asked to account for.
            for index, crop in enumerate(CROP_ITEMS):
                if int(seeds[turn + 1][index]) or int(standing[turn + 1][index]):
                    continue
                _reconcile(queues[crop], int(stock[turn + 1][index]))

    return ledger


def _is_bought_stock(
    op: str | None, money_delta: float, seed_delta: Sequence[int]
) -> bool:
    """Whether stock arriving this turn was purchased rather than harvested."""

    if op is not None:
        return op == "BUY_PRODUCT"
    # No op recoverable: the pre-fix sign heuristic, kept only for that caller.
    return money_delta < 0 and not seed_delta_touches(seed_delta)


def _reconcile(queue: deque[_Lot], held: int) -> None:
    """Cap a finished crop's remaining room at what is actually still in stock.

    Called once a crop has neither seed nor standing plant, so nothing in its
    queue can produce another unit. Whatever room is left is therefore a claim on
    units that were never grown, rotted in the field, or overflowed the shed --
    and leaving it open would let the oldest lot keep absorbing sales that a
    later purchase actually paid for, which is the bias `_YIELD_CAP` documents.

    Trimmed from the back so the front of the queue keeps its room: FIFO hands
    the next sale to the oldest lot, and the oldest lot is the one whose purchase
    turn is furthest from its payoff.
    """

    room = sum(lot.room for lot in queue)
    excess = room - max(0, held)
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
            continue
        lot = queue[0]
        lot.absorbed += 1
        lot.proceeds += unit_price
        if lot.first_sale_turn is None:
            lot.first_sale_turn = turn
        ledger.contributions.append((lot.buy_turn, turn, unit_price))
