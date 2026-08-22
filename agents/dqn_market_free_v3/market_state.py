"""Market-state encoder for the unscaffolded market learner.

229 dims: the scorer's 117-dim snapshot plus 101 of the economics its donor market
layer already knew and this agent has to learn, plus block E's 11 intra-turn
columns.

The network is a plain action head with no action features, so anything about
*quantity* -- what selling 45 melons does to the melon price -- has to arrive
through the state or not at all. Blocks A through C below are that arrival.

**Block D is gone, and that is the v3 fork's whole premise.** v2 spent 30 columns
on a hand-built cross-turn memory: a 24-turn lagged difference over prices,
market inventory, own money and opponent money, a one-hot of the last op, and
three refusal counters. Those columns were one fixed summary of the past, chosen
in advance -- one lag, four series, three latches. v3 replaces them with a GRU
that ticks once per turn (`policy.GruCell`), so the recurrence learns what to
remember instead of being told.

`MarketHistory` therefore still exists here and still runs, but **nothing reads
it**. `main.py` and `clone.py` call `observe`/`record_decision` and no caller
anywhere consults the result: the diagnostics do not touch its latches, and
`order_refused` -- which `training/reward.py` charges the refusal penalty from --
is a free function in this module with no dependency on the class. It is kept
only so the class is here to re-adopt if a latch is ever wanted again, and so the
`observe` call keeps marking the point in the turn where last turn's order has
settled, which is the anchor `main.py` and `clone.py` both site their tick
against. `market_state_features` no longer takes a `history` argument.

If nothing has re-adopted it by the next fork, delete it and the two call sites
rather than carrying four deques a turn for a value no one reads.

`donor.py` is a byte-identical vendored copy and is never edited. Where its helper
is wrong for the installed engine, or scalar where a vector is needed, the
corrected version lives here and the divergence is commented where it happens.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agents.dqn_market_free_v3 import donor, herd

CROPS: tuple[str, ...] = tuple(donor.CROPS)
ANIMALS: tuple[str, ...] = tuple(donor.ANIMALS)
PRODUCTS: tuple[str, ...] = tuple(donor.PRODUCTS)
PHASES: tuple[str, ...] = ("BOOTSTRAP", "COMPOUND", "REALIZE", "CRISIS", "LIQUIDATE")

#: Products the market will sell to a farm. Mirrors `order_space`'s set of the
#: same name, duplicated rather than imported because `order_space` is about what
#: the agent may *do* and importing it here would make the state schema depend on
#: the action schema.
BUYABLE: tuple[str, ...] = ("WHEAT", "FERTILIZER")

#: Fill depths probed by the price-curve block, in units.
#:
#: Two points, not the whole quantity ladder. The curve between them is smooth
#: and monotone, so a shallow fill and the deepest one the ladder can reach
#: (`order_space.QUANTITIES[-1]` is 45) pin its slope; eleven probes would cost
#: eleven times the arithmetic for a shape the network can interpolate.
FILL_PROBES: tuple[int, ...] = (8, 45)

#: Days ahead the supply projection looks, for the opponent and for own tiles.
OPPONENT_HORIZONS: tuple[int, ...] = (1, 7)
OWN_HORIZONS: tuple[int, ...] = (1, 3)

#: Turns of lag the history block differences over. One in-game day.
HISTORY_LAG = 24

#: Ceiling on the per-crop held-seed feature, in units.
#:
#: Every crop tile `donor.CROP_MIX` describes across all four quadrants, which is the
#: most tiles any single crop could occupy after `planting.market_influenced_roles`
#: has reassigned the plan. Seed held past that cannot be planted this season under
#: any allocation.
#:
#: The seed columns were the one unbounded block in this vector -- shed counts are
#: capped by `shedCapacity`, animals by the tile count, but `private["seeds"]` has no
#: ceiling in the simulator, so a policy holding 300 wheat fed a 30.0 into a network
#: whose other inputs sit in [0, 2]. Over-buying pushed the state out of distribution,
#: degrading the Q estimate for exactly the rows that caused it.
#:
#: Deliberately sized to be *inert*: with `order_space.feasible_actions` capping seed
#: orders at the plan's deficit, no reachable state reaches this clip, so the column's
#: values do not move and a resumed checkpoint sees no distribution shift. It is a
#: guard rail against a regression reopening the unbounded path, not a transform.
#:
#: If this clip ever binds, the demand cap has a hole and `seed_lots` per episode
#: shows it first.
SEED_FEATURE_CAP = sum(
    count for quadrant in donor.CROP_MIX.values() for count in quadrant.values()
)

#: Operations the last-order one-hot spans, in `order_space.ACTIONS` order.
ORDER_OPERATIONS: tuple[str, ...] = (
    "NOOP",
    "SELL",
    "BUY_PRODUCT",
    "BUY_SEED",
    "BUY_ANIMAL",
    "BUY_LAND",
    "HIRE",
)

#: Decisions one turn may take, for scaling block E's counter only.
#:
#: Duplicated from `order_space.MAX_DECISIONS_PER_TURN` for the same reason
#: `PRODUCTS` is duplicated at the top of this module: the state schema must not
#: depend on the action schema. A drift between the two costs a slightly miscalibrated
#: column, never a wrong mask, and `tests/test_dqn_market_free_v3.py` pins them equal.
ORDER_LOOP_CAP = int(donor.MAX_MARKET_ORDERS) + 1

#: Columns block E adds over v1's 238: four scalars plus one one-hot over
#: `ORDER_OPERATIONS` for the ops already queued this turn. Named rather than
#: folded into the total so the two halves of the arithmetic stay legible.
INTRA_TURN_DIM = 4 + len(ORDER_OPERATIONS)

#: Width of the cross-turn history block v2 carried and v3 does not: 9 price
#: deltas, 9 market-inventory deltas, 2 clipped money deltas, a 7-wide last-op
#: one-hot, and 3 refusal scalars. Named rather than folded into the subtraction
#: below so the arithmetic says which 30 columns left, and why.
HISTORY_BLOCK_DIM = 20 + len(ORDER_OPERATIONS) + 3

#: Columns the herd block adds: one affordability column per animal instead of the
#: two `("COW", "SHEEP")` this vector used to carry, plus three species-wide triples.
#:
#: The missing goose column was not a rounding error. Block C prices *how many units
#: the cash reaches* for every crop seed and both buyable products, and priced it for
#: the cow and the sheep only, so the one animal the donor's planner never houses was
#: also the one animal the network could not see the price of. The other three
#: triples are the quantities `ANIMAL_WASTE_WEIGHT` charges on and the state did not
#: carry at all: what is bought and still homeless, how many production events a
#: purchase made *today* can still deliver before the season ends, and what those
#: events are worth against the purchase price at the market's own depth.
#:
#: Measured over v1: goose purchases fall from 10.3 an episode at generation 1 to
#: 0.05 at 53 with the column unmasked throughout. A policy cannot price a deadline
#: it cannot see, and every animal in this game is a deadline.
HERD_VIABILITY_DIM = 1 + 3 * len(ANIMALS)

#: 238 in v1, 259 in v2, 229 here. Two forks' worth of width changes, and each is
#: a deliberate cold start: `policy.load` rejects a width mismatch rather than
#: reading it as a shape match, so neither a v1 nor a v2 checkpoint can be loaded
#: into this agent by accident.
#:
#: The GRU's input is this same vector -- not a separate encoding -- so there is
#: one width, one builder and one set of shape checks. `main.py` encodes it once
#: at the top of the turn with block E carrying the *previous* turn's committed
#: orders, which is what replaces the refusal signal block D used to carry: the
#: recurrence sees that an order was emitted and sees the money that followed it,
#: and can difference the two itself.
MARKET_STATE_DIM = 238 - HISTORY_BLOCK_DIM + INTRA_TURN_DIM + HERD_VIABILITY_DIM

#: Where block E sits in the finished vector. `main.py` overwrites exactly this
#: span when it reuses the turn-opening encode for the turn's first sub-decision,
#: which is the optimisation that keeps v3's encoder cost equal to v2's.
INTRA_TURN_SLICE = slice(MARKET_STATE_DIM - INTRA_TURN_DIM, MARKET_STATE_DIM)


#: Most production events any purchase can still have ahead of it: a goose bought on
#: day 0, yielding once a day from day 4 through the last day the simulator runs an
#: end-of-day. The scale, not a clip -- nothing exceeds it, so the column lands in
#: [0, 1] with 0 meaning "cannot produce again this season".
_MAX_YIELD_EVENTS = max(herd.remaining_yield_events(animal, 0) for animal in ANIMALS)

#: Clip on gross coins per coin of purchase price. A cow that gets placed early
#: grosses on the order of 4x its cost, so 8 is roughly double the best real case and
#: binds only on a market that has been driven far below equilibrium -- exactly where
#: the difference between 12x and 20x carries no decision.
_GROSS_PER_COIN_CAP = 8.0


#: Share of a turn's quoted cash movement that has to land before the turn counts
#: as filled, when the quote is known.
#:
#: Not 1.0, and the gap is not slack: orders interleave one unit at a time with the
#: opponent's, so a quote we priced at the observation's inventory can be taken at a
#: worse one, and a sale into a market the opponent is also dumping into fills below
#: what `PendingTurn` quoted. A tenth is well inside that noise and well outside a
#: refusal, which moves exactly nothing.
FILL_FRACTION = 0.1


def order_refused(emitted, money_delta, expected=None):
    """Whether an emitted order was refused by the simulator.

    One definition, shared by the reward (`training/reward.py`, which charges
    `REJECTED_ORDER_PENALTY` for exactly this) and by `MarketHistory`, which
    encodes it into the state. They must not be able to drift: a state feature
    that says "refused" on turns the reward does not charge for, or the reverse,
    teaches the network to predict a penalty that is not there.

    Without `expected`, this is v1's test and v1's reasoning: every order the
    simulator accepts moves cash by at least a coin -- prices floor at 1 and
    quantities start at 1 -- and nothing but the market phase moves cash at all, so
    a zero delta on a turn that emitted an order is a refusal.

    **That test is wrong for a turn that emits several orders**, which is every v2
    turn that wants to be. A `SELL WHEAT 6` beside a `BUY_PRODUCT FERTILIZER 4` can
    net to a delta of zero with both orders filled, and reads there as a refusal;
    worse, a turn whose buy was refused while its sale filled reads as accepted. So
    where the caller knows what the turn was quoted at -- `PendingTurn` priced every
    order as it queued it -- the test becomes "did a tenth of the quoted movement
    arrive", which asks about the orders rather than about their sum.

    Accepts scalars or arrays.
    """

    delta = np.asarray(money_delta)
    if expected is None:
        return np.logical_and(emitted, delta == 0.0)
    quoted = np.asarray(expected, dtype=np.float64)
    # A turn quoted at nothing falls back to v1's test rather than being waved
    # through.
    #
    # v2 failed this branch *open* -- `np.where(silent, False, missing)` -- on the
    # reasoning that a same-item round trip is deliberate market-making and
    # charging it would price the behaviour rather than the mistake. Measured, that
    # was wrong twice over. The wash was not a tactic, it was the policy: 47% of
    # v2's turns were `BUY_PRODUCT WHEAT 45 | SELL WHEAT 45` five times over for
    # exactly zero P&L, and this exemption was one of the three things that made it
    # reward-optimal. And the stated cost -- "a genuinely refused wash going
    # uncharged, which is rare" -- described the dominant case, not a rare one.
    #
    # `order_space.feasible_actions` now shuts same-item round trips outright, so
    # the case this branch was widened for cannot be reached. What remains at a
    # zero quote is a turn whose legs coincidentally cancel, and for that the
    # wallet is exactly as informative as it was in v1: a filled pair almost never
    # nets to precisely zero, while a refused pair always does.
    silent = np.abs(quoted) <= 0.0
    missing = np.abs(delta) < FILL_FRACTION * np.abs(quoted)
    return np.logical_and(emitted, np.where(silent, delta == 0.0, missing))


# --- price curve -------------------------------------------------------------

_SHAPE_FUNCS = {
    "linear": lambda values: values,
    "sq": lambda values: values * values,
    "sqrt": np.sqrt,
    "log": np.log1p,
    "log10": lambda values: np.log10(1.0 + values),
}


def _shape_array(name: str, values: np.ndarray) -> np.ndarray:
    """Vectorised `donor._shape`. The negative clamp is the donor's, kept."""

    values = np.maximum(0.0, np.asarray(values, dtype=np.float64))
    return _SHAPE_FUNCS.get(name, _SHAPE_FUNCS["linear"])(values)


def price_curve(item: str, inventories, obs=None) -> np.ndarray:
    """`donor._price_at` over a whole array of inventory levels.

    A fill probe walks 45 inventory levels per product, which is 495
    `_price_at` calls a turn against a 1-second act timeout and 719 turns an
    episode. This is the same arithmetic in one numpy pass, reading its
    parameters through `donor._market_parameters` so a `market["params"]`
    override is honoured exactly as the donor honours it.

    `test_price_curve_matches_donor` pins it to `donor._price_at` element by
    element; if that test fails this function is wrong, not the donor.
    """

    (
        base,
        throughput,
        below_fn,
        below_move,
        above_fn,
        above_move,
        equilibrium,
    ) = donor._market_parameters(obs, item)
    inventory = np.asarray(inventories, dtype=np.float64)
    below_amplitude = below_move * base / max(1e-9, donor._shape(below_fn, throughput))
    above_amplitude = above_move * base / max(1e-9, donor._shape(above_fn, throughput))
    # Both branches evaluate; `_shape_array` clamps its argument at zero, so the
    # unused one cannot produce a nan out of sqrt or log of a negative.
    values = np.where(
        inventory < equilibrium,
        base + below_amplitude * _shape_array(below_fn, equilibrium - inventory),
        base - above_amplitude * _shape_array(above_fn, inventory - equilibrium),
    )
    return np.maximum(1.0, np.rint(values))


_FILL_OFFSETS = np.arange(max(FILL_PROBES), dtype=np.float64)


def _fill_averages(item: str, inventory: float, obs, *, side: str) -> list[float]:
    """Mean realised unit price of filling `q` units, for each `q` in FILL_PROBES.

    Straight off the engine's per-unit lockstep loop: a SELL's k-th unit is
    quoted at `market_price(item, inventory + k)` because each accepted sale
    raises supply, and a BUY_PRODUCT's k-th unit at `inventory - k - 1` because
    buys are quoted post-fill so a buy/sell round trip against an unchanged
    market nets zero.

    Two known simplifications, both harmless as a feature: a sale clearing at
    price 1 does not actually raise supply, and the opponent's units interleave
    with these in the same loop. Neither is knowable at decision time.
    """

    if side == "SELL":
        curve = price_curve(item, inventory + _FILL_OFFSETS, obs)
    else:
        curve = price_curve(item, inventory - _FILL_OFFSETS - 1.0, obs)
    cumulative = np.cumsum(curve)
    return [float(cumulative[depth - 1] / depth) for depth in FILL_PROBES]


# --- corrected town demand ---------------------------------------------------


def town_demand_per_day(obs, item: str) -> float:
    """Units of `item` the town removes from the market per in-game day.

    Deliberately not `donor._town_demand_per_day`, which models the town centre
    as `2 * (4 if day >= 20 else 2 if day >= 10 else 1)` units a day. The
    installed engine's `_town_consume` removes exactly one unit of each
    non-fertilizer product every `townCenterSellInterval` (24) steps, flat, with
    no ramp -- so the donor overstates late-season demand by up to eight times,
    on precisely the days the agent has to decide whether to hold stock or dump
    it. The shop half of the donor's formula is right and is kept: a shop
    instance sells every `townShopSellInterval` (4) steps, doubled when it
    stocks a single product, which is 6 or 12 a day.
    """

    center = 0.0 if item == "FERTILIZER" else 1.0
    shop = 0.0
    for name in (((obs or {}).get("town", {}) or {}).get("unlocked_shops", []) or []):
        products = donor.SHOPS.get(name, ())
        if item in products:
            shop += 12.0 if len(products) == 1 else 6.0
    return center + shop


# --- supply projection -------------------------------------------------------


#: Which product each livestock animal yields, inverted from the per-item test
#: the projection used to run once per item.
_PRODUCT_FOR_ANIMAL = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}


def projected_supply_map(
    farm: dict, day: int, horizons: tuple[int, ...]
) -> dict[int, dict[str, int]]:
    """Every product's projected supply for one farm, in a single pass.

    A tile feeds at most one product, so asking per (item, horizon) rescanned the
    whole 10x10 board to read one crop off each tile and discard the other 99
    answers. The feature vector wants nine products across two horizons for both
    farms, which was 36 board scans -- 3,600 tile visits -- per decision, against
    the 200 the two boards actually contain. Measured at 51,768 calls and 0.54s
    of a ~2.9s episode, the single largest cost in the agent.
    """

    totals: dict[int, dict[str, int]] = {horizon: {} for horizon in horizons}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            crop = tile.get("crop")
            if tile.get("kind") == "PLANT" and crop in donor.CROPS:
                rule = donor.CROPS[crop]
                planted = tile.get("planted_day")
                planted_day = day if planted is None else int(planted)
                age = day - planted_day
                held = int(tile.get("yield_units", 0) or 0)
                ripe = held > 0 and age >= rule["first"]
                for horizon, bucket in totals.items():
                    if ripe:
                        bucket[crop] = bucket.get(crop, 0) + held
                    elif age + horizon >= rule["ripe"]:
                        bucket[crop] = bucket.get(crop, 0) + max(
                            1, min(rule["max_yield"], held + horizon)
                        )
            animal = tile.get("animal")
            product = _PRODUCT_FOR_ANIMAL.get(str(animal))
            if product is not None:
                held = int(tile.get("yield_units", 0) or 0)
                rule = donor.ANIMALS.get(str(animal), {"max_held": 4})
                for horizon, bucket in totals.items():
                    grown = min(2 * horizon, int(rule["max_held"])) if horizon > 0 else 0
                    bucket[product] = bucket.get(product, 0) + held + grown
    return totals


def projected_supply(farm: dict, item: str, day: int, horizon: int) -> int:
    """Units of `item` this farm's tiles will hold `horizon` days from now.

    A mirror of `donor._opponent_visible_supply`'s projection, lifted out so own
    and opponent supply are measured the same way and can be compared. The donor
    version reads only the *other* seat's tiles, which is why it cannot be called
    for both; `test_projected_supply_matches_donor` pins this one against it on
    an opponent farm -- and so, through this wrapper, pins `projected_supply_map`
    that the feature vector actually calls.
    """

    return projected_supply_map(farm, day, (horizon,))[horizon].get(item, 0)


# --- cross-turn history ------------------------------------------------------


class MarketHistory:
    """A short rolling window of the market, and how the last order went.

    The simulator's observation is a pure snapshot: no price series, no previous
    market inventory, and -- the one that costs most -- no sign that last turn's
    order was refused. `REJECTED_ORDER_PENALTY` charges for a refusal, but without
    this the state cannot tell the turn a policy entered a refusal loop from the
    five hundredth turn inside it, so no policy defined on the state can climb out.

    Owned by the live agent and rebuilt step by step from stored replays. Both must
    push in the same order -- observe the turn, encode the state, record the
    decision -- which `test_clone_matches_live_features` holds them to.
    """

    __slots__ = (
        "lag",
        "_prices",
        "_inventory",
        "_money",
        "_opponent_money",
        "last_op",
        "last_filled",
        "turns_since_fill",
        "consecutive_refusals",
        "_emitted",
        "_last_money",
    )

    def __init__(self, lag: int = HISTORY_LAG):
        self.lag = int(lag)
        self._prices: deque = deque(maxlen=self.lag + 1)
        self._inventory: deque = deque(maxlen=self.lag + 1)
        self._money: deque = deque(maxlen=self.lag + 1)
        self._opponent_money: deque = deque(maxlen=self.lag + 1)
        self.reset()

    def reset(self) -> None:
        self._prices.clear()
        self._inventory.clear()
        self._money.clear()
        self._opponent_money.clear()
        self.last_op: str | None = None
        self.last_filled = False
        self.turns_since_fill = 0
        self.consecutive_refusals = 0
        self._emitted = False
        self._last_money: float | None = None

    def observe(self, obs, farm: dict) -> None:
        """Take in this turn's market, and settle the previous turn's order.

        Called before the state is encoded. The money in `farm` has already been
        through the previous turn's market phase, so this is the first moment the
        previous decision's fate is knowable.
        """

        money = float(farm.get("money", 0) or 0)
        if self._last_money is not None:
            refused = bool(order_refused(self._emitted, money - self._last_money))
            filled = self._emitted and not refused
            self.last_filled = filled
            if filled:
                self.turns_since_fill = 0
                self.consecutive_refusals = 0
            else:
                self.turns_since_fill += 1
                if refused:
                    self.consecutive_refusals += 1
        self._last_money = money
        self._emitted = False

        market = obs.get("market", {}) or {}
        prices = market.get("prices", {}) or {}
        inventory = market.get("inventory", {}) or {}
        self._prices.append(
            {item: float(prices.get(item, np.nan) or np.nan) for item in PRODUCTS}
        )
        self._inventory.append(
            {item: float(inventory.get(item, donor.MARKET_I0) or donor.MARKET_I0) for item in PRODUCTS}
        )
        self._money.append(money)
        self._opponent_money.append(_opponent_money(obs))

    def record_decision(self, order) -> None:
        """Remember what was just emitted, so the next turn can encode it."""

        self.last_op = str(getattr(order, "op", "NOOP"))
        self._emitted = self.last_op != "NOOP"

    @property
    def ready(self) -> bool:
        return len(self._money) > self.lag

    def features(self, obs) -> list[float]:
        """The history block. Zero-filled until a full lag is buffered.

        Zero rather than a "no data" flag: a zero delta is what a still market
        looks like, and the first day of an episode is a still market by
        construction -- nothing has been sold into it yet.
        """

        values = [0.0] * 20
        if self.ready:
            old_prices, new_prices = self._prices[0], self._prices[-1]
            old_inventory, new_inventory = self._inventory[0], self._inventory[-1]
            deltas: list[float] = []
            for item in PRODUCTS:
                base = float(donor._market_parameters(obs, item)[0])
                previous = old_prices.get(item)
                current = new_prices.get(item)
                if previous is None or current is None or np.isnan(previous) or np.isnan(current):
                    deltas.append(0.0)
                else:
                    deltas.append((current - previous) / max(1.0, base))
            values = deltas
            values += [
                (new_inventory[item] - old_inventory[item]) / 1000.0 for item in PRODUCTS
            ]
            # Clipped, because a coin-scaled delta is unbounded and the opponent's
            # leg is where that bites: training opponents hold 0-3,000 and the
            # promotion benchmark ~142,000, so an unclipped opponent delta is a
            # feature the network only ever meets at evaluation time. Both legs
            # symmetrically, so the own leg cannot become the same problem in
            # reverse if the policy ever gets rich.
            values += [
                float(np.clip((self._money[-1] - self._money[0]) / 10_000.0, -1.0, 1.0)),
                float(
                    np.clip(
                        (self._opponent_money[-1] - self._opponent_money[0]) / 10_000.0,
                        -1.0,
                        1.0,
                    )
                ),
            ]

        last = self.last_op or "NOOP"
        values += [1.0 if last == name else 0.0 for name in ORDER_OPERATIONS]
        values += [
            1.0 if self.last_filled else 0.0,
            min(self.turns_since_fill, 4 * self.lag) / float(4 * self.lag),
            min(self.consecutive_refusals, 8) / 8.0,
        ]
        return values


def _opponent_money(obs) -> float:
    player = int((obs or {}).get("player", 0) or 0)
    return max(
        (
            float(other.get("money", 0) or 0)
            for index, other in enumerate((obs or {}).get("farms", []) or [])
            if index != player
        ),
        default=0.0,
    )


# --- board census ------------------------------------------------------------


def _owned_animals(farm: dict, private: dict) -> dict[str, int]:
    placed = donor._farm_animal_counts(farm)
    return {
        animal: int(placed.get(animal, 0)) + int(donor._private_item_total(private, animal))
        for animal in ANIMALS
    }


def _structures(farm: dict) -> dict[str, int]:
    counts = {"COOP": 0, "PASTURE": 0}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") in counts and "animal" not in tile:
                counts[str(tile["kind"])] += 1
    return counts


def _tile_census(obs, player: int) -> tuple[int, int, int, dict[str, int], dict[str, int], dict[str, int]]:
    planted = empty = weeds = 0
    opponent_crops = {name: 0 for name in CROPS}
    opponent_animals = {name: 0 for name in ANIMALS}
    pending = {item: 0 for item in PRODUCTS}
    for index, other in enumerate(obs.get("farms", []) or []):
        for row in other.get("tiles", []) or []:
            for tile in row:
                if index == player:
                    if tile is None:
                        empty += 1
                    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                        weeds += 1
                    elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                        planted += 1
                        crop = tile.get("crop")
                        if crop in pending:
                            pending[str(crop)] += int(tile.get("yield_units", 0) or 0)
                    if isinstance(tile, dict) and tile.get("animal") in donor.ANIMALS:
                        rule = donor.ANIMALS[str(tile["animal"])]
                        pending[str(rule["product"])] += int(tile.get("yield_units", 0) or 0)
                        pending["FERTILIZER"] += int(bool(tile.get("fertilizer_available")))
                elif isinstance(tile, dict):
                    if tile.get("kind") == "PLANT" and tile.get("crop") in opponent_crops:
                        opponent_crops[str(tile["crop"])] += 1
                    if tile.get("animal") in opponent_animals:
                        opponent_animals[str(tile["animal"])] += 1
    return planted, empty, weeds, opponent_crops, opponent_animals, pending


def intra_turn_block(turn) -> list[float]:
    """Block E: what the current turn has already committed.

    Extracted from `market_state_features` rather than inlined, for two callers
    that need it on its own. `main.py` reuses the turn-opening encode for a turn's
    first sub-decision -- legal because `PendingTurn.view` is the identity before
    the first `apply` -- and overwrites exactly this span, which is what keeps v3
    from paying for two full encodes on every single-decision turn. The parity
    test encodes both ways and compares.

    Block E is also the whole of the cross-turn signal the GRU receives, because
    `main.py` builds the recurrent input with the *previous* turn's finished
    `PendingTurn`. Read that way, the four scalars say how hard last turn traded
    and which direction its money went, and the one-hot says what it traded.

    `turn` is optional so a caller with no turn in flight (the feature tests, and
    `clone.py` before it starts its walk) gets the zero block, which is exactly
    what the first sub-decision of an episode's first turn looks like.
    """

    if turn is None:
        return [0.0] * INTRA_TURN_DIM
    start_money = max(1.0, float(turn.turn_start_money))
    queued = set(turn.ops)
    values = [
        turn.slots / float(max(1, turn.max_slots)),
        min(1.0, turn.decisions / float(max(1, ORDER_LOOP_CAP))),
        # Below 1.0 once the turn has spent, above once it has sold into the
        # market, so the sign carries the direction of the turn so far.
        min(2.0, turn.money / start_money),
        1.0 if turn.land else 0.0,
    ]
    values += [1.0 if operation in queued else 0.0 for operation in ORDER_OPERATIONS]
    return values


def market_state_features(
    obs,
    farm: dict,
    private: dict,
    roles: dict,
    field: dict,
    *,
    config=None,
    episode_steps: int = 720,
    turns_per_day: int = 24,
    turn=None,
) -> np.ndarray:
    """Public market state, the donor's planning state, and the market's shape.

    The donor block is the point of the original design: the network is choosing
    how much of the donor's stated demand to fund, so it has to see that demand.
    Blocks A-C that follow it are the market's own economics -- the price curve,
    where supply is going, and what the cash actually buys -- none of which the
    snapshot alone carries.

    There is no `history` argument and no block D. "What just happened" is the
    GRU's job in v3, and the GRU is fed *this function's own output*, encoded once
    at the top of the turn with `turn` set to the previous turn's finished
    `PendingTurn`. That is why block E is the only cross-turn signal left in the
    vector and why it must not be zeroed on the recurrent path: it is what tells
    the GRU an order was emitted at all.

    One vector, two readers, and they see different things only through `turn`:
    the head gets it per sub-decision off `PendingTurn.view`, the GRU gets it once
    per turn off the untouched observation.
    """

    step = int(obs.get("step", 0) or 0)
    day = int(obs.get("day", step // max(1, turns_per_day)) or 0)
    hour = int(obs.get("hour", step % max(1, turns_per_day)) or 0)
    total_days = max(1.0, episode_steps / max(1, turns_per_day))
    money = float(farm.get("money", 0) or 0)
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", []) or []
    opponent_money = _opponent_money(obs)
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    carried = {item: 0 for item in (*PRODUCTS, *ANIMALS)}
    for slot in private.get("inventories", []) or []:
        for item in carried:
            carried[item] += int((slot or {}).get(item, 0) or 0)
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}
    planted, empty, weeds, opponent_crops, opponent_animals, pending = _tile_census(obs, player)
    owned = _owned_animals(farm, private)
    structures = _structures(farm)
    shed_total = sum(max(0, int(value or 0)) for value in shed.values())
    bases = {item: float(donor._market_parameters(obs, item)[0]) for item in PRODUCTS}
    stock = {
        item: float(inventory.get(item, donor.MARKET_I0) or donor.MARKET_I0)
        for item in PRODUCTS
    }
    quotes = {item: float(prices.get(item, bases[item]) or bases[item]) for item in PRODUCTS}

    values = [
        day / total_days,
        hour / max(1, turns_per_day),
        step / max(1, episode_steps),
        max(0, episode_steps - step) / max(1, episode_steps),
        float(np.log1p(max(0.0, money))) / 10.0,
        money / 10_000.0,
        # The opponent's wealth on the same log scale as our own two lines above,
        # and the share taken on logs as well.
        #
        # **Both must stay on logs.** Read raw, these are where the gate and the
        # buffer stop describing the same policy: every training opponent holds
        # 0-3,000, while the promotion benchmark -- which appears in no rollout --
        # finishes near 142,000. That is fifty times outside the buffer, on two of
        # the eight wealth features, at exactly the moment the run is judged. One
        # unchanged checkpoint read a `trading_rate` of 0.805 against the benchmark
        # and 0.055 against its own incumbent in the same update: same weights, two
        # behaviours, because one arena is in distribution and the other is not.
        #
        # log1p compresses that 47x gap to 1.5x. The share is a ratio of the same
        # logs, so it stays monotone in relative wealth and bounded well away from
        # 0 and 1.
        float(np.log1p(max(0.0, opponent_money))) / 10.0,
        float(np.log1p(max(0.0, money)))
        / (float(np.log1p(max(0.0, money))) + float(np.log1p(max(0.0, opponent_money))) + 1.0),
    ]
    values += [min(int(seeds.get(item, 0) or 0), SEED_FEATURE_CAP) / 10.0 for item in CROPS]
    values += [int(shed.get(item, 0) or 0) / 50.0 for item in PRODUCTS]
    values += [carried[item] / 25.0 for item in PRODUCTS]
    values += [carried[item] / 10.0 for item in ANIMALS]
    # Normalised against the *effective* base, not `donor.MARKET[item][0]`: a
    # `marketParams` override moves the base and the raw constant would then
    # report a price change that never happened.
    values += [quotes[item] / bases[item] for item in PRODUCTS]
    values += [(stock[item] - donor.MARKET_I0) / 1000.0 for item in PRODUCTS]
    values += [
        len(farm.get("hands", []) or []) / float(donor.MAX_HANDS),
        int(farm.get("hires_today", 0) or 0) / float(donor.MAX_HANDS),
        planted / 100.0,
        empty / 100.0,
        weeds / 100.0,
        len(farm.get("unlocked_quadrants", []) or ["NW"]) / 4.0,
        shed_total / 100.0,
        max(0, 100 - shed_total) / 100.0,
    ]
    values += [owned[item] / 10.0 for item in ANIMALS]
    values += [structures[item] / 10.0 for item in ("COOP", "PASTURE")]
    # Bought and homeless, per species. `owned` above sums placed and held, and the
    # carried block at the top covers a worker's hands, so the one state the reward
    # actually charges for -- an animal in the shed with no structure under it -- was
    # the one state no column named. `herd.py` clears it in a turn when a slot is
    # free; when none is, this is the column that says so.
    values += [
        int(donor._private_item_total(private, item) or 0) / 10.0 for item in ANIMALS
    ]
    # Production events a purchase made today could still deliver, and what they are
    # worth per coin spent at the depth this farm's own herd would sell into. Zero on
    # the events column is a hard deadline -- past day 21 a cow cannot milk once, and
    # `order_space` has no day gate on `BUY_ANIMAL`, so the row stays open and the
    # waste term charges the full 400 coins. `herd.warm_slot_species` scores with the
    # same two quantities, which is deliberate: the structure that gets built ahead of
    # demand and the purchase the head chooses are then reading one story.
    events = {item: herd.remaining_yield_events(item, day) for item in ANIMALS}
    values += [events[item] / float(_MAX_YIELD_EVENTS) for item in ANIMALS]
    values += [
        min(
            _GROSS_PER_COIN_CAP,
            events[item]
            * float(
                donor._price_at(
                    str(donor.ANIMALS[item]["product"]),
                    stock[str(donor.ANIMALS[item]["product"])]
                    + events[item] * (owned[item] + 1) / 2.0,
                    obs,
                )
            )
            / float(donor.ANIMALS[item]["cost"]),
        )
        / _GROSS_PER_COIN_CAP
        for item in ANIMALS
    ]
    values += [opponent_crops[item] / 25.0 for item in CROPS]
    values += [opponent_animals[item] / 10.0 for item in ANIMALS]
    values += [pending[item] / 25.0 for item in PRODUCTS]
    demand = {item: town_demand_per_day(obs, item) for item in PRODUCTS}
    values += [demand[item] / 24.0 for item in PRODUCTS]

    # --- donor planning state -------------------------------------------------
    needs = donor._seed_needs(obs, farm, private, roles)
    values += [int(needs.get(item, 0)) / 10.0 for item in CROPS]
    role_targets = {animal: 0 for animal in ANIMALS}
    for kind, item in roles.values():
        if kind == "ANIMAL" and item in role_targets:
            role_targets[item] += 1
    values += [max(0, role_targets[item] - owned[item]) / 8.0 for item in ANIMALS]
    values.append(donor._target_hands(obs, farm, private, roles) / float(donor.MAX_HANDS))

    summary = donor._survey(farm, private, roles, day)
    phase = donor._policy_phase(obs, farm, private, summary)
    values += [1.0 if phase == name else 0.0 for name in PHASES]
    values += [
        summary["shed_load"] / 100.0,
        summary["carried_load"] / 100.0,
        summary["at_risk_animals"] / 8.0,
        summary["at_risk_crops"] / 16.0,
        summary["open_structures"] / 8.0,
        summary["structures_todo"] / 16.0,
        summary["plantable"] / 32.0,
        summary["unfed"] / 16.0,
    ]

    shed_capacity = int(donor._cfg(config, "shedCapacity", donor.SHED_CAPACITY))
    post_shed, post_carried = donor._post_field_storage(private, field, shed_capacity)
    occupancy = sum(max(0, int(value or 0)) for value in post_shed.values())
    values += [
        occupancy / max(1.0, float(shed_capacity)),
        max(0, shed_capacity - occupancy) / max(1.0, float(shed_capacity)),
    ]
    values += [
        max(0, donor.TOTAL_DAYS - day) / float(donor.TOTAL_DAYS),
        1.0 if field.get("liquidation") else 0.0,
    ]

    # --- A: price curve -------------------------------------------------------
    #
    # The snapshot gives the current quote and the current inventory and expects
    # the network to derive the rest, but "the rest" is a per-product asymmetric
    # log/sqrt/linear/square response integrated over the fill -- and with no
    # action features, each of the eleven quantity heads per product would have
    # to derive it independently from reward. These are the answer, computed.
    sell_fills = {item: _fill_averages(item, stock[item], obs, side="SELL") for item in PRODUCTS}
    buy_fills = {item: _fill_averages(item, stock[item], obs, side="BUY") for item in BUYABLE}
    for depth in range(len(FILL_PROBES)):
        values += [sell_fills[item][depth] / bases[item] for item in PRODUCTS]
    for depth in range(len(FILL_PROBES)):
        values += [buy_fills[item][depth] / bases[item] for item in BUYABLE]

    # --- B: forward projection ------------------------------------------------
    #
    # Where supply is going, which is what decides whether to sell now or hold.
    # The opponent's harvested stock is private, but their tiles are not, and a
    # ripening field is a price move that has not happened yet.
    opponent_farms = [other for index, other in enumerate(farms) if index != player]
    opponent_scans = [
        projected_supply_map(other, day, OPPONENT_HORIZONS) for other in opponent_farms
    ]
    opponent_supply = {
        horizon: {
            item: sum(scan[horizon].get(item, 0) for scan in opponent_scans)
            for item in PRODUCTS
        }
        for horizon in OPPONENT_HORIZONS
    }
    own_scan = projected_supply_map(farm, day, OWN_HORIZONS)
    own_supply = {
        horizon: {item: own_scan[horizon].get(item, 0) for item in PRODUCTS}
        for horizon in OWN_HORIZONS
    }
    for horizon in OPPONENT_HORIZONS:
        values += [opponent_supply[horizon][item] / 25.0 for item in PRODUCTS]
    for horizon in OWN_HORIZONS:
        values += [own_supply[horizon][item] / 25.0 for item in PRODUCTS]
    projected_stock = {
        item: stock[item] - demand[item] + own_supply[1][item] + opponent_supply[1][item]
        for item in PRODUCTS
    }
    values += [(projected_stock[item] - donor.MARKET_I0) / 1000.0 for item in PRODUCTS]
    values += [
        float(price_curve(item, projected_stock[item], obs)) / max(1.0, quotes[item])
        for item in PRODUCTS
    ]

    # --- C: constraint magnitudes --------------------------------------------
    #
    # `order_space.feasible_actions` answers yes/no. These are the magnitudes
    # behind that answer: how many units the cash actually reaches, and what the
    # two nonlinear prices -- the fibonacci hire ladder and the land staircase --
    # currently cost.
    values += [
        min(45.0, money / float(donor.CROPS[crop]["seed"])) / 45.0 for crop in CROPS
    ]
    # Every animal, not the donor's two. See `HERD_VIABILITY_DIM`.
    values += [
        min(4.0, money / float(donor.ANIMALS[animal]["cost"])) / 4.0
        for animal in ANIMALS
    ]
    values += [
        min(45.0, money / max(1.0, float(donor._price_at(item, stock[item] - 1, obs)))) / 45.0
        for item in BUYABLE
    ]
    hire_mult = float(donor._cfg(config, "farmHandCostMult", 1))
    hire_cost = hire_mult * float(donor._fib(int(farm.get("hires_today", 0) or 0)))
    extra_land = max(0, len(farm.get("unlocked_quadrants", ["NW"]) or ["NW"]) - 1)
    land_cost = (
        float(donor.LAND_PRICES[extra_land]) if extra_land < len(donor.LAND_PRICES) else 0.0
    )
    carried_total = sum(
        max(0, int(count or 0)) for slot in post_carried for count in (slot or {}).values()
    )
    values += [
        min(1.0, hire_cost / max(1.0, money)),
        1.0 if money >= hire_cost else 0.0,
        # 1.0 when there is no land left to buy: unaffordable and unbuyable read
        # the same way to the mask, so they should read the same way here.
        min(1.0, land_cost / max(1.0, money)) if land_cost else 1.0,
        # Overflow at the nightly auto-drop is discarded outright by the engine,
        # not held over. This is the size of tonight's loss if nothing is sold.
        max(0, occupancy + carried_total - shed_capacity) / 100.0,
        max(0, turns_per_day - 1 - hour) / float(max(1, turns_per_day)),
        max(0, (episode_steps - 1 - donor.LIQUIDATION_TURNS) - step) / float(max(1, episode_steps)),
    ]

    # --- D: cross-turn history --------------------------------------------------
    #
    # Deliberately absent. v2 spent 30 columns here on a 24-turn lagged difference
    # and three refusal latches; `policy.GruCell` carries that now. `MarketHistory`
    # still runs in `main.py` and `clone.py` but nothing reads what it accumulates;
    # see this module's docstring.

    # --- E: this turn, so far -------------------------------------------------
    values += intra_turn_block(turn)

    result = np.asarray(values, dtype=np.float32)
    if result.shape != (MARKET_STATE_DIM,):
        raise AssertionError(f"MARKET_STATE_DIM is {MARKET_STATE_DIM}, built {result.shape[0]}")
    return result
