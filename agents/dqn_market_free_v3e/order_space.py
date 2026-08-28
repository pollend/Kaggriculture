"""Flat market-order action space for the unscaffolded market learner.

`dqn_market_scorer` learns a *modulation* of the donor's market layer: the
donor decides what to buy and how much, and the network scales it. This agent
removes that layer entirely. The network emits the order itself.

The action is a single order drawn from a flat list of structurally possible
ones. Game-invalid product buys and inventory that this frozen field policy can
never create or consume are not exploration, so they are not in the list at all.
State-dependent choices -- affordability, sale quantities, land timing, the feed
floor, and terminal liquidation -- remain entirely for the network to learn from
reward.

One order per turn, not a list of ten, which keeps the value function an ordinary
Q(s, a). At 720 decisions an episode the throughput is ample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.dqn_market_free_v3e import donor, herd, labor

# The state half lives in `market_state.py`. This module is only about what the
# agent may *do*.

PRODUCTS: tuple[str, ...] = tuple(donor.PRODUCTS)
CROPS: tuple[str, ...] = tuple(donor.CROPS)
ANIMALS: tuple[str, ...] = tuple(donor.ANIMALS)

#: Quantity ladder for bulk goods: roughly geometric, since price impact makes the
#: interesting decisions order-of-magnitude rather than per-unit, but dense at the
#: bottom. The range is set by what stored replays actually contain -- every
#: quantity from 1 to 18 and then sparsely to 45 -- so a shorter ladder would
#: distort the behaviour `training/clone.py` reads out of them.
QUANTITIES: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 45)

#: How far a single sell order may walk its own price down, as a fraction of the
#: quote it was priced at.
#:
#: **This is the one preference in the mask, and it passes the proof rule above by
#: aliasing rather than by measurement.** `SELL WOOL 45` and four `SELL WOOL 12`s in
#: the same turn reach the same inventory through the same per-unit walk for the same
#: coins -- the simulator prices each unit off the running inventory, so splitting an
#: order changes nothing about what it fills at. The rungs above the cap are therefore
#: routes to an outcome that stays reachable, not outcomes; what they cost is wire
#: slots, and `MAX_MARKET_ORDERS` is ten.
#:
#: What shutting them buys is the whole reason this exists. The ladder tops out at 45
#: and was applied identically to WHEAT (`T=400`, `above log 0.20`, where 45 units
#: move the price 12%) and to WOOL (`T=105`, `above sq 3.20`, where they move it 25%
#: and the next 45 move it 40%). v1 dumped roughly 320 sell orders an episode into
#: that, sold WOOL in 4.2% of episodes and EGG in 0.9%, and drove MELON from 250 to 89
#: -- a 64% self-inflicted crash on the item it produced most. A policy cannot learn
#: "this market is thin" from a ladder that offers the same eleven rungs everywhere;
#: it has to learn it from 45-unit mistakes, once per episode, against a terminal
#: reward.
#:
#: 0.15 rather than something tighter: at the market's equilibrium inventory it leaves
#: WHEAT and EGG the whole ladder (65 and 76 units clear it), CARROT 22, WOOL 22,
#: TOMATO 13, MILK 11 and STRAWBERRY 9. The premium animal products keep enough depth
#: to be worth farming and lose the rung that craters them.
#:
#: Read against `above_func` in `donor.MARKET`, not against `T` alone: `sq` is flat
#: near equilibrium and vicious past it, which is why MELON's cap looks generous per
#: order and its season-long crash still happened. The cap is per order and reprices
#: itself as the inventory it sold into rises, so the cumulative case is handled by
#: the same arithmetic rather than by a second budget.
MAX_SELL_SLIPPAGE = 0.15

#: Animals are indivisible capital at 300-500 each against a few thousand in
#: cash, so the ladder is short.
ANIMAL_QUANTITIES: tuple[int, ...] = (1, 2, 4)

#: `HIRE` carries a hand count, and it is the one operation that has to. Hiring N
#: hands means N separate `["HIRE"]` orders on the *same* turn: the fibonacci cost
#: ladder is keyed on `hires_today`, hands last only the day they were hired, and a
#: hand hired late has lost part of its shift. Spread over consecutive turns it is
#: a different, worse action.
HIRE_QUANTITIES: tuple[int, ...] = tuple(range(1, int(donor.MAX_HANDS) + 1))


def nearest_quantity(quantity: int, ladder: tuple[int, ...] = QUANTITIES) -> int:
    """Snap an observed order quantity onto the ladder.

    Used only when reading demonstrations out of a replay, where quantities are
    unconstrained. Ties break low: over-ordering costs money the agent may not
    have, under-ordering only forgoes a fill.
    """

    return min(ladder, key=lambda level: (abs(level - quantity), level))


@dataclass(frozen=True, slots=True)
class Order:
    """One action, and the simulator order list it expands to.

    Almost every action is one order. `NOOP` is none -- it is the agent's way of
    emitting nothing. `HIRE` is `quantity` of them, because the simulator has no
    hand count: N hands means N identical `["HIRE"]` orders on the same turn.
    """

    op: str
    item: str | None = None
    quantity: int = 0

    def to_wire(self) -> list[list]:
        if self.op == "NOOP":
            return []
        if self.op == "HIRE":
            return [["HIRE"] for _ in range(max(1, int(self.quantity)))]
        if self.op == "BUY_LAND":
            return [["BUY_LAND"]]
        return [[self.op, self.item, int(self.quantity)]]


MARKET_BUYABLE_PRODUCTS = frozenset(("WHEAT", "FERTILIZER"))
FIELD_CROPS = frozenset(donor.CROPS)

#: Animals this agent's field layer can house, which is not the donor's herd.
#:
#: **Do not put `donor.CORE_HERD_SEQUENCE` back here on its own.** Deriving the head
#: from a frozen donor constant is what silently made EGG unsellable: the donor's
#: sequence is COW/SHEEP, so `REACHABLE_PRODUCTS` lost EGG, so `SELL EGG` was filtered
#: out of `ACTIONS` below and `order_index("SELL", "EGG", n)` returned None -- there
#: was no gate to inspect and no masked row to see, the column did not exist.
#:
#: `herd.py` houses all three species now -- it strips the donor's unconditional
#: COW/SHEEP pastures and assigns a structure only to an animal that was actually
#: bought -- so this set is that module's roster, read from it rather than restated.
#: The two have to agree exactly: an animal here that `herd` cannot house strands the
#: purchase, and one `herd` houses that is missing here can never be bought.
FIELD_ANIMALS = frozenset(herd.ANIMALS)
REACHABLE_PRODUCTS = frozenset(
    {
        *FIELD_CROPS,
        *(str(donor.ANIMALS[animal]["product"]) for animal in FIELD_ANIMALS),
        *MARKET_BUYABLE_PRODUCTS,
    }
)


def order_is_structurally_possible(order: Order) -> bool:
    """Whether an order can ever affect a standard episode for this agent.

    This is deliberately not a market strategy. It does not check price,
    quantity, cash, current stock, timing, or demand. It only removes actions
    rejected by the game and items the frozen field planner has no way to use
    or produce.
    """

    if order.op in ("NOOP", "HIRE"):
        return True
    # `BUY_LAND` is not an action here at all. `land_score` decides it and
    # `should_buy_land` emits it, so the head carries no column for it and there is
    # no Q value to read -- see `LAND_BUY_SET_POINT`.
    if order.op == "BUY_LAND":
        return False
    if order.op == "BUY_PRODUCT":
        return order.item in MARKET_BUYABLE_PRODUCTS
    if order.op == "BUY_SEED":
        return order.item in FIELD_CROPS
    if order.op == "BUY_ANIMAL":
        return order.item in FIELD_ANIMALS
    if order.op == "SELL":
        return order.item in REACHABLE_PRODUCTS
    return False


def _enumerate_actions() -> tuple[Order, ...]:
    """Every order the network may emit, NOOP first.

    Index 0 must stay NOOP. A network that scores every action equally breaks ties
    to the lowest index, so an untrained policy trades nothing rather than emitting
    a random order every turn -- doing nothing is survivable and a random policy is
    not.

    Structurally impossible orders are filtered out here rather than masked at score
    time, so the head carries no permanently dead rows. `BUY_LAND` is absent for the
    same reason from the other direction: the row is reachable, but since v43 nothing
    reads its Q, so a column for it is dead weight the learner still has to fit.
    """

    rows: list[Order] = [Order("NOOP")]
    for item in PRODUCTS:
        rows += [Order("SELL", item, quantity) for quantity in QUANTITIES]
    for item in PRODUCTS:
        rows += [Order("BUY_PRODUCT", item, quantity) for quantity in QUANTITIES]
    for crop in CROPS:
        rows += [Order("BUY_SEED", crop, quantity) for quantity in QUANTITIES]
    for animal in ANIMALS:
        rows += [Order("BUY_ANIMAL", animal, quantity) for quantity in ANIMAL_QUANTITIES]
    rows += [Order("HIRE", None, quantity) for quantity in HIRE_QUANTITIES]
    return tuple(order for order in rows if order_is_structurally_possible(order))


#: The network's output head, one column per entry. `ACTIONS[0]` is NOOP.
ACTIONS: tuple[Order, ...] = _enumerate_actions()
ACTION_COUNT = len(ACTIONS)

#: Reverse index, so a demonstration order becomes an action index in O(1).
ACTION_INDEX: dict[tuple[str, str | None, int], int] = {
    (order.op, order.item, order.quantity): index for index, order in enumerate(ACTIONS)
}


def order_index(op: str, item: str | None, quantity: int) -> int | None:
    """Index of the closest emittable order, or None if there is none.

    Quantity is snapped to the ladder; operation and item are not, because there
    is no meaningful nearest neighbour for those. An order this agent cannot
    emit at all -- unnameable, or structurally impossible -- returns None and the
    caller drops the demonstration slot rather than guessing.

    `BUY_LAND` returns None like anything else off the head. It is not a
    demonstration fault, though, so `training/clone.py` counts it on its own rather
    than as an unreadable order: the demonstrator's quadrant is a decision this
    agent's `land_score` makes for itself.
    """

    if op == "NOOP":
        return ACTION_INDEX.get((op, None, 0))
    if op == "HIRE":
        hands = min(max(1, int(quantity)), HIRE_QUANTITIES[-1])
        return ACTION_INDEX.get((op, None, hands))
    if item is None:
        return None
    ladder = ANIMAL_QUANTITIES if op == "BUY_ANIMAL" else QUANTITIES
    return ACTION_INDEX.get((op, item, nearest_quantity(int(quantity), ladder)))


def _indices_by_item(op: str) -> dict[str | None, np.ndarray]:
    """Action indices for one operation, grouped by item.

    Built once at import so a feasibility mask costs a handful of numpy slice
    writes rather than a 199-iteration Python loop. This runs on every turn of
    every rollout episode.
    """

    groups: dict[str | None, list[int]] = {}
    for index, order in enumerate(ACTIONS):
        if order.op == op:
            groups.setdefault(order.item, []).append(index)
    return {item: np.asarray(rows, dtype=np.intp) for item, rows in groups.items()}


_SELL_ROWS = _indices_by_item("SELL")
_BUY_PRODUCT_ROWS = _indices_by_item("BUY_PRODUCT")
_BUY_SEED_ROWS = _indices_by_item("BUY_SEED")
_BUY_ANIMAL_ROWS = _indices_by_item("BUY_ANIMAL")
_HIRE_ROWS = _indices_by_item("HIRE").get(None, np.empty(0, dtype=np.intp))
NOOP_INDEX = 0

#: The land purchase, as a wire order rather than as an action.
#:
#: Off the head deliberately -- see `LAND_BUY_SET_POINT`. It still has to be an
#: `Order`, because `MarketHistory.record_decision` encodes the last op into the
#: state and "the turn I bought a quadrant" is worth a feature even when it is not
#: worth a column.
LAND_ORDER = Order("BUY_LAND")

#: Each row's ladder quantity, so the seed affordability test is one vector compare
#: rather than a Python loop on the per-turn path. Zero for NOOP, which carries no
#: quantity.
_ROW_QUANTITY = np.asarray([int(order.quantity) for order in ACTIONS], dtype=np.intp)

#: Units of a crop the mask keeps open when none of it is held, whatever the plan says.
#:
#: Without this the cap is self-locking, and the lock is not hypothetical --
#: `planting.market_influenced_roles` rewrites every vacant crop role from *held*
#: seed in its second pass, so a farm hoarding wheat has its melon-planned tiles
#: reassigned to wheat, `donor._seed_needs` then reports no melon demand, a strict cap
#: shuts the melon rows, and no melon can ever be bought again. The plan follows the
#: inventory, so a cap read only off the plan ratchets the agent into monoculture.
#: Keying the floor on `held == 0` breaks the cycle: buy a few of a crop you hold none
#: of, pass two plants them on spare tiles, the plan then wants that crop, and the
#: full ladder unlocks on the deficit.
#:
#: 4 is the top of `QUANTITIES`' dense bottom (1, 2, 3, 4), it matches CARROT and
#: TOMATO's `max_yield` and the donor plan's smallest per-crop allocation, and at
#: melon's 80 coins it is 320 -- roughly a tenth of the stake, so a probe can never
#: be a wallet event on its own.
#:
#: Watch `distinct_crops` and `top_crop_share`: crops collapsing toward 1 with the
#: share toward 0.9 is this floor being too small. `trading_rate` falling toward the
#: viability floor is the same thing seen from the other side -- raise this before
#: reverting the cap.
SEED_PROBE_UNITS = 4


def seed_room(
    obs: dict,
    farm: dict,
    private: dict,
    roles: dict | None = None,
    config=None,
) -> dict[str, int]:
    """Seed units per crop the mask will open, by crop, omitting the shut ones.

    `donor._seed_needs` is the base: vacant tiles the plan wants, still plantable
    today under `last_plant`, net of the seed already held. `SEED_PROBE_UNITS` is the
    floor that keeps a crop reachable when none is held, which is what stops the cap
    ratcheting into monoculture.

    Deliberately *not* what `land_score` reads. That veto asks "does the plan still
    want seed", which is the raw deficit; this asks "what may be bought", which
    includes a probe for crops the plan has not asked for yet. Feeding the probe into
    the land veto would leave it permanently vetoed, since some crop is almost always
    held at zero.
    """

    if roles is None:
        try:
            roles = herd.field_roles(obs, farm, private)
        except Exception:
            # `feasible_actions` calls this every turn and its own contract is that it
            # never raises -- `main.agent` swallows four exceptions an episode and the
            # mask must not spend one. No plan is no proof of a deficit, and the proof
            # is what opens the rows, so a malformed observation shuts them rather
            # than reopening the old unbounded path.
            return {}
    try:
        needs = donor._seed_needs(obs, farm, private, roles)
    except Exception:
        return {}

    day = int(obs.get("day", 0) or 0)
    held = private.get("seeds", {}) or {}
    room: dict[str, int] = {}
    for crop in CROPS:
        allowance = int(needs.get(crop, 0) or 0)
        # The probe rides the same `last_plant` gate `_seed_needs` applies, or it
        # would leave "buy melon on day 25" open -- an order the game accepts and
        # that can never mature, which is the class of divergence this cap exists
        # to close.
        if int(held.get(crop, 0) or 0) == 0 and day <= int(donor.CROPS[crop]["last_plant"]):
            allowance = max(allowance, int(SEED_PROBE_UNITS))
        if allowance > 0:
            room[crop] = allowance
    return room


def animal_room(
    obs: dict,
    farm: dict,
    private: dict,
    roles: dict | None = None,
    config=None,
) -> dict[str, int]:
    """Head per species the mask will open, by species, omitting the shut ones.

    Two proofs, both of an outcome that is already unreachable rather than of a
    route this cap dislikes -- the rule at `feasible_actions`' preamble.

    **Housing.** `donor.ANIMAL_SLOTS` is 15 tiles board-wide and 4 in the opening
    quadrant, while `shedCapacity` is 100, so the affordability test alone let a
    turn buy ~100 head that `herd.herd_influenced_roles` can never find ground for.
    `herd.py:318-321` calls that deliberate -- "the action space prices its own
    mistakes" -- and it is, per lot. What it does not price is the *rate*: a league
    slot spends 37,000 coins an episode discovering it, and five of ten slots were
    still discovering it at generation 17.

    Aliasing is what makes this feasibility rather than policy, and it is exact
    here in a way the sell cap's is not. Animals are flat-priced -- no inventory
    walk, no `_price_at` -- so `BUY_ANIMAL COW 2` costs precisely what one now plus
    one after a slot frees costs, and the deferred route keeps the second 400 coins
    liquid meanwhile. The rungs above free housing are routes, not outcomes.

    **The calendar.** `remaining_yield_events` is the same arithmetic
    `lots.unplaced_events` charges the waste term with, so a species it reads zero
    for is an order the game accepts and that can never mature -- "buy melon on day
    25" for the herd, and the class `seed_room`'s `last_plant` gate already closes.

    All three species share one number because they share the ground. COW and SHEEP
    split the pasture outright, and a bare structure of either kind is re-roled by
    `herd._species_for` toward whatever is actually waiting, so a free tile is free
    for the whole herd rather than for one column of it.
    """

    day = int(obs.get("day", 0) or 0)
    try:
        slots, _zones = donor._reserved_animal_slots(farm)
        tiles = farm.get("tiles", []) or []
        # Ground a waiting animal could actually be placed on this turn: bare
        # structures first (built and empty), then claimable dirt and weeds. A tile
        # already carrying an animal is neither, which is what keeps a full board
        # reading zero rather than reading its own herd back as headroom.
        capacity = sum(
            1
            for position in slots
            if herd._bare_structure(herd._tile_at(tiles, position)) is not None
            or herd._is_claimable(herd._tile_at(tiles, position))
        )
        # Head already bought and not yet placed. Pass three of
        # `herd_influenced_roles` will consume this much of the capacity above
        # before any purchase made now can claim a slot, so it is demand, not
        # headroom -- the same subtraction the warm slot makes at `herd.py:299`.
        waiting = sum(
            max(0, int(donor._private_item_total(private, animal)))
            for animal in herd.ANIMALS
        )
        room = max(0, capacity - waiting)
    except Exception:
        # `feasible_actions` calls this every turn and its own contract is that it
        # never raises. No board is no proof of free housing, and the proof is what
        # opens the rows, so a malformed observation shuts them.
        return {}

    if room <= 0:
        return {}
    return {
        animal: room
        for animal in herd.ANIMALS
        if herd.remaining_yield_events(animal, day) > 0
    }


#: `HIRE n` costs `sum(_fib(hires_today) .. _fib(hires_today + n - 1))` times the
#: hand multiplier, because `_do_hire` charges per hand and increments
#: `hires_today` between them. Prefix sums turn that into one subtraction:
#: `_FIB_PREFIX[h + n] - _FIB_PREFIX[h]`.
#:
#: The simulator puts no ceiling on `hires_today`, only affordability, so the table
#: is sized well past `MAX_HANDS`. It need not cover every reachable value --
#: `_fib(40)` is 165 million, and falling off the end leaves the rows masked, which
#: is the right answer.
_FIB_PREFIX = np.concatenate(
    ([0.0], np.cumsum([float(donor._fib(index)) for index in range(64)]))
)
#: `_HIRE_ROWS` is built in `HIRE_QUANTITIES` order, so these line up positionally.
_HIRE_HANDS = np.asarray(HIRE_QUANTITIES, dtype=np.intp)


#: Score given to a masked action before an argmax, in both numpy and torch.
#:
#: Finite rather than -inf, for two reasons: a masked score can reach a multiply,
#: and `-inf * 0.0` is nan, which takes a whole minibatch down through the backward
#: pass; and a cross-entropy landing on a masked class would return inf rather than
#: a large, visible number.
#:
#: Shared by numpy and torch because the two argmaxes must break ties identically
#: for identical weights, which the `policy.py` / `training/network.py` parity test
#: pins. -1e4 against single-digit Q values leaves the ordering in no doubt.
MASKED_SCORE = -1e4


#: Earliest day a land purchase is offered. Day 0 is 29% of every purchase on record
#: and the whole of the bankruptcy tail -- 1,000 coins against the 3,000 stake,
#: spent before a single crop has been sold. Opens at 1 rather than the donor's 5
#: because days 1-4 and 5-9 are indistinguishable in the data.
LAND_MIN_DAY = 1

#: Days that must remain for a quadrant to pay for itself. Land has no salvage value
#: -- the episode reward is cash on hand -- so a late purchase is a pure write-off.
LAND_MIN_DAYS_LEFT = 12


#: Tiles a quadrant unlocks. The board is 10x10 in four 5x5 quadrants, and
#: `_do_buy_land` flips every `"LOCKED"` cell in the one it sells to `None` at once.
QUADRANT_TILES = 25


#: Extra quadrants this agent will buy, out of the three the simulator sells
#: (`LAND_ORDER = ["NE", "SW", "SE"]` at 1,000 / 2,000 / 4,000, on top of a free NW).
#: There is no fourth: four quadrants is the whole board.
#:
#: **The donor's `MAX_EXTRA_LAND = 2` is not this agent's number and was wrong
#: here.** It is read out of a frozen file -- `donor.py` is byte-identical to the
#: scorer's copy and license-pinned, so it cannot be edited -- and the reason given
#: for it, "nothing on record has come close to working two", is false on the data:
#: `land_accepted / land_buying_episodes` reaches 1.72-1.81 on `animallots_04/08`,
#: `widecap_06` and `lotbasis_08`, so most buyers on those slots already take both
#: and the cap binds. The leaderboard settles the direction: its best performers own
#: all four quadrants.
#:
#: Raising it is safe *because the score is a score*. The third quadrant is 4,000
#: coins and adds 25 tiles to a farm whose crew is already at the simulator's
#: `MAX_HANDS = 13`, so it buys area with no labour to work it -- and that shows up
#: directly in `land_utilisation`, which falls as soon as the second quadrant goes
#: unworked and refuses the third on its own. A hard cap forbids the purchase the
#: leaderboard says wins; the score prices it and declines it when the farm is not
#: the kind that can use it.
#:
#: Watch `land_accepted / land_buying_episodes` against
#: `land_tiles_planted_per_purchase`. Buyers moving past 2.0 with tiles per purchase
#: holding is the change working; tiles per purchase falling as the ratio rises is
#: the third quadrant being bought and abandoned, and the answer is to put this back
#: to 2 rather than to touch the set point.
LAND_MAX_EXTRA = len(donor.LAND_PRICES)


#: Days of the expanded farm's wage bill to hold back on top of the land price.
#:
#: Hands are wiped at every day boundary and re-hired on a fibonacci ladder keyed to
#: `hires_today`, so a day's wage is the whole prefix rather than the marginal hand:
#: 143 coins at 10 hands, 609 at 13. A quadrant adds `QUADRANT_TILES` tiles, which
#: raises due jobs, which raises `_target_hands` -- so the reserve is a function of
#: the tile count, through the crew those tiles need.
#:
#: **Where that factor bites, measured rather than assumed.** `_target_hands` floors
#: at 10 while `day <= 27` and the simulator caps at `MAX_HANDS = 13`, so the first
#: quadrant moves the crew 10 -> 13 and the daily bill 143 -> 609, a 4.3x jump off a
#: doubling of area. The second saturates against that cap and costs no more in
#: wages -- it costs more in price instead, 2,000 against 1,000. So the tile term is
#: live exactly once, which is also the only purchase anything on record has worked.
#:
#: **One day, chosen as the risk-taking end of a real trade.** One day is what
#: "can still staff the farm tomorrow" means literally, and it is the smallest
#: reserve that is still a reserve: hands are wiped nightly, so a farm that cannot
#: cover one day's crew cannot work the tiles it just bought at all.
#:
#: The two directions this can be wrong, and the column that says which:
#:
#: * Too low, and expansion is bought out of the working capital the season runs
#:   on. That is the v33 shape -- purchases landing on day 10.3 holding 1,965 coins
#:   and finishing on a 304 median. At 609 a ready farm clears the set point around
#:   1,767 on day 5, so that wallet does still buy.
#: * Too high, and it is an off switch. Reserving until the new tiles pay back was
#:   the alternative and is arithmetically dead: melon's `first_yield_day` is 10,
#:   which at 609 a day is 6,090 coins against a run whose `mean_final_money` sits
#:   near 3,500 -- the v10 day-gate outcome `roadmap.md` records, where the policy
#:   stopped buying land at all and money fell.
#:
#: Read `buy_land_rate` against `land_tiles_planted_per_purchase` and the solvent
#: fraction. Purchases rising while tiles worked stay flat is the first failure and
#: this constant is the answer to it; `land_gate_open_rate` near zero is the second.
LAND_RESERVE_DAYS = 1.0


#: The set point on the land score. Land is bought when the score clears it.
#:
#:     horizon = clip((days_left - LAND_MIN_DAYS_LEFT) / (TOTAL_DAYS - LAND_MIN_DAYS_LEFT), 0, 1)
#:     worked  = land_utilisation(farm)
#:     pressure = land_expansion_pressure(obs, farm, roles)
#:     runway  = clip((money - price - reserve) / reserve, 0, 1)
#:     score   = horizon * worked * pressure * runway
#:
#: Land is decided by that score and the network is not consulted for it at all --
#: which is why, since v44, there is no `BUY_LAND` column on the head to consult.
#: A row nothing ranks is a row the learner still fits, still noises, still counts in
#: `selection_entropy`, and still renumbers every index after it; leaving it in place
#: cost all of that to represent a decision made somewhere else. `should_buy_land`
#: emits `LAND_ORDER` directly and `choose_order` records the turn as NOOP, which is
#: what the network actually contributed to it.
#:
#: v42 asked `Q(BUY_LAND) - Q(NOOP)` instead, and the row it read was dead:
#: `buy_land_rate` ran 0.000-0.050 across all ten slots over batches 351-390, one
#: purchase per twenty to two hundred episodes, against v33's 0.37. A gated row is
#: starved of gradient, so its Q is arbitrary exactly when the gate opens, and every
#: exploration lever aimed at the problem has been refused by measurement -- 212
#: seats at sigma 3.159 bought zero quadrants, and every purchase on record happened
#: at sigma <= 0.871. Land is chosen greedily or not at all.
#:
#: Why an argmax was the wrong home for it in the first place, kept because it is
#: still the argument against putting it back:
#:
#: * **It is a capital decision priced on a marginal scale.** Every other row is a
#:   turn's worth of coins; this one commits 1,000-4,000 against a payback 10-25
#:   days out. `roadmap.md` measures the consequence -- `BUY_LAND` sits at *median
#:   rank 33 of 104 feasible actions*. It does not need to be rank 1. It needs to be
#:   "yes" at most twice in 720 turns, and an argmax has no way to say that.
#: * **The competition it loses is one the simulator never staged.** `BUY_LAND` is a
#:   market op resolved atomically against `maxMarketOrdersPerTurn = 10`, so it does
#:   not contend with a farmer op at all. Ranking it against SELL and BUY_SEED is an
#:   artefact of this module emitting one order per decision.
#:
#: The product is the point: a factor at zero is a veto, so the three cannot trade
#: each other away. In particular `horizon` alone peaks on day 0 -- the timing that
#: produced 649 of 707 purchases in the 0-3,000 money bucket -- and `worked` is what
#: forbids it, because nothing is planted on turn 0 whatever the wallet says. The
#: purchase therefore lands when the farm's own progress says so rather than on a
#: hardcoded day like the donor's `LAND_OPEN_DAYS = (5, 9)`.
#:
#: 0.15 is the first measurement of this knob. It is the single dial: `land_gate_open_rate`
#: near zero means it or `LAND_RESERVE_DAYS` is too tight, and this is the one to
#: move first. Setting it to 0.0 buys on the vetoes alone, which is the loosest this
#: knob goes; the pre-v42 revert -- put the row back in the argmax -- is no longer one
#: constant away, because the column is gone.
LAND_BUY_SET_POINT: float = 0.15


def _daily_wage(hands: int, hire_mult: float) -> float:
    """Coins to put `hands` hands in the field for one day, from a standing start.

    The whole fibonacci prefix, not the marginal hand: `hires_today` resets at every
    day boundary, so tomorrow's crew is bought again from rung zero.
    """

    index = min(max(0, int(hands)), _FIB_PREFIX.shape[0] - 1)
    return float(hire_mult) * float(_FIB_PREFIX[index])


def land_reserve(
    obs: dict,
    farm: dict,
    private: dict,
    roles: dict,
    config=None,
) -> float:
    """Cash to hold back for the hands the *expanded* farm will need.

    `_role_plan` builds no roles for a locked quadrant, so the post-purchase hand
    target cannot be read off the plan -- it is scaled by tile count instead, which
    is the same thing `_target_hands` is ultimately counting.
    """

    quadrants = max(1, len(farm.get("unlocked_quadrants", ["NW"]) or ["NW"]))
    tiles_now = QUADRANT_TILES * quadrants
    tiles_after = tiles_now + QUADRANT_TILES
    target_now = max(0, int(donor._target_hands(obs, farm, private, roles)))
    target_after = min(
        int(donor.MAX_HANDS), -(-target_now * tiles_after // tiles_now)
    )
    hire_mult = float(donor._cfg(config, "farmHandCostMult", 1))
    return LAND_RESERVE_DAYS * _daily_wage(target_after, hire_mult)


def land_utilisation(farm: dict) -> float:
    """Share of the unlocked board this farm is actually working, in [0, 1].

    Counted off the tiles rather than off the role plan, and that is not a detail.
    `planting.market_influenced_roles` rewrites every vacant crop role from the
    seed actually held, so a farm holding seed for three tiles has a plan wanting
    three tiles -- and a role-based utilisation reads it as nearly full while
    twenty tiles sit bare. Measured: that read opened the gate on day 2 at 3,000
    untouched coins, which is the day-0 purchase this whole score exists to refuse.

    A tile counts as worked if something is on it that is not a weed: a standing
    plant, an animal, or a structure waiting for one. `"LOCKED"` cells are not
    tiles yet and are excluded on both sides of the ratio.
    """

    worked = idle = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if tile == "LOCKED":
                continue
            if tile is None:
                idle += 1
            elif isinstance(tile, dict):
                if tile.get("kind") == "WEED":
                    idle += 1
                else:
                    worked += 1
    return worked / max(1, worked + idle)


def land_expansion_pressure(obs: dict, farm: dict, roles: dict) -> float:
    """Share of unlocked tiles the current field plan has a productive claim on.

    `land_utilisation` says whether the ground already owned is physically worked.
    This says whether the *plan* is space-constrained. A board can be full of live
    crops and still not need the next quadrant if the role map has contracted; it
    can also have a few empty tiles and still be ready if the plan has seed-covered
    roles waiting to fill them. The seed-covered part is enforced by `land_score`'s
    `_seed_needs` veto; this helper deliberately measures only planned tile demand.
    """

    day = int(obs.get("day", 0) or 0)
    tiles = farm.get("tiles", []) or []
    unlocked = 0
    for row in tiles:
        for tile in row:
            if tile != "LOCKED":
                unlocked += 1
    if unlocked <= 0:
        return 0.0

    demanded: set[tuple[int, int]] = set()
    for position, role in (roles or {}).items():
        if not isinstance(position, tuple) or len(position) != 2:
            continue
        x, y = position
        if not 0 <= y < len(tiles):
            continue
        row = tiles[y] or []
        if not 0 <= x < len(row) or row[x] == "LOCKED":
            continue
        kind, item = role
        if kind == "ANIMAL":
            demanded.add((x, y))
        elif kind == "CROP" and (
            isinstance(row[x], dict) and row[x].get("kind") == "PLANT"
            or day <= int(donor.CROPS.get(item, {}).get("last_plant", -1))
        ):
            demanded.add((x, y))
    return min(1.0, len(demanded) / max(1, unlocked))


def land_score(
    obs: dict,
    farm: dict,
    private: dict,
    money: float,
    extra: int,
    roles: dict | None,
    config=None,
) -> float:
    """How much a quadrant is worth buying *now*, in [0, 1]. Zero means no.

    The single exception to this module's "feasibility, not preference" rule, and
    since v43 it is a preference *score* rather than a readiness bool: the network
    has no say in this row. Affordability alone offers 1,000 coins against the 3,000
    stake on turn 0, and **49.4% of all accepted purchases planted zero of their 25
    new tiles** -- the money bought area the farm had neither the seed nor the hands
    to use.

    Three vetoes and four factors. The vetoes are the old readiness test:

    * mid-season, with `LAND_MIN_DAYS_LEFT` still to run and fewer than
      `LAND_MAX_EXTRA` quadrants already bought;
    * the seed already held covers every vacant crop role the plan wants filled, so
      the new tiles are not competing for seed the old ones still need.

    The factors are lateness, utilisation, expansion pressure and runway,
    multiplied so that any one of them at zero refuses the purchase on its own.
    `worked`, `pressure` and the seed veto are complementary rather than redundant:
    the veto catches a farm whose *plan* is short of seed, `worked` catches bare
    ground however the plan describes it, and `pressure` catches a plan that is not
    actually constrained by the current acreage.

    Watch `land_tiles_planted_per_purchase` against `buy_land_rate` and
    `land_gate_open_rate`. A rate near zero with the gate open is impossible now --
    the score *is* the decision -- so a rate near zero means the score never clears
    `LAND_BUY_SET_POINT`, which is a constant to move rather than a policy to wait
    on.
    """

    day = int(obs.get("day", 0) or 0)
    if day < LAND_MIN_DAY:
        return 0.0
    days_left = donor.TOTAL_DAYS - day
    if days_left < LAND_MIN_DAYS_LEFT:
        return 0.0
    if extra >= LAND_MAX_EXTRA:
        return 0.0

    # `main.py` has already built this for the turn; recomputing it is only for
    # callers that do not, and it is reached only once the cheap tests above pass.
    if roles is None:
        roles = herd.field_roles(obs, farm, private)

    # Seed the plan you have before buying a plan you do not.
    #
    # The raw deficit, deliberately not `seed_room`: that one carries a probe floor
    # for crops held at zero, and since some crop almost always is, reading it here
    # would veto land for the whole season. This veto also used to be gameable in the
    # wrong direction -- driving the deficit to zero opened the land row, and
    # over-buying was the cheapest way to do it. With the seed rows capped at the
    # deficit, the only way to clear it is a purchase the plan can absorb, which is
    # what the veto always meant.
    if sum(donor._seed_needs(obs, farm, private, roles).values()) > 0:
        return 0.0

    span = float(donor.TOTAL_DAYS - LAND_MIN_DAYS_LEFT)
    horizon = min(1.0, max(0.0, (days_left - LAND_MIN_DAYS_LEFT) / span))

    worked = land_utilisation(farm)
    pressure = land_expansion_pressure(obs, farm, roles)

    reserve = land_reserve(obs, farm, private, roles, config)
    price = float(donor.LAND_PRICES[extra])
    runway = min(1.0, max(0.0, (float(money) - price - reserve) / max(1.0, reserve)))

    return horizon * worked * pressure * runway


def liquid_cash(obs: dict, farm: dict, private: dict) -> float:
    """Wallet plus what the shed would fetch if this turn sold it.

    The wallet alone is the wrong quantity to price a quadrant against, because the
    simulator applies a turn's market orders *in list order* and a SELL earlier in
    the list funds a BUY_LAND later in it. Measured over 1,050 Kaggle episodes and
    4,464 accepted purchases: 72.6% are preceded by a SELL in the same turn, the
    median first quadrant is bought holding 1,034 coins against a 1,000 price, and
    **47.2% of first quadrants are outright unaffordable on the pre-turn wallet**.
    Reading `farm["money"]` refused every one of them.

    Priced at the current inventory rather than walking each unit forward, which is
    what `PendingTurn.sale_proceeds` does for an order actually being placed. The
    walk is the exact number and this is deliberately not it: a sale depresses its
    own price, so the flat quote is an over-estimate of a large shed's worth, and the
    cost of being wrong here is one turn's delay on a purchase the score still has to
    justify. What it buys is O(items) instead of O(units) on a gate that runs every
    turn -- a full shed is 100 units, and `main.py`'s encoder budget has no room for
    another per-unit walk.

    Seed is excluded because the simulator will not buy it back at any price; only
    `private["shed"]` is sellable. Livestock is excluded for the same reason and is
    the sharper edge: `donor.ANIMALS` lives in the shed alongside produce but carries
    no `donor.MARKET` row, so pricing the shed indiscriminately raises `KeyError` on
    the first sheep. `donor.MARKET` *is* the sellable set -- keying off it rather than
    off the shed is what makes that structural instead of a guard.
    """

    money = float(farm.get("money", 0) or 0)
    shed = (private or {}).get("shed", {}) or {}
    if not shed:
        return money
    inventories = (obs.get("market", {}) or {}).get("inventory", {}) or {}
    for item, count in shed.items():
        held = int(count or 0)
        if held <= 0 or item not in donor.MARKET:
            continue
        raw = inventories.get(item)
        inventory = donor.MARKET_I0 if raw is None else int(raw)
        money += held * float(donor._price_at(item, inventory, obs))
    return money


def should_buy_land(
    obs: dict,
    farm: dict,
    private: dict,
    roles: dict | None = None,
    config=None,
) -> bool:
    """Whether to buy the next quadrant this turn. The whole land decision.

    Affordability first, because `land_score`'s runway term needs a price to be
    quoted against and there is none once the board is sold out.

    Both the affordability test and the runway factor read `liquid_cash` rather than
    the wallet, and they read the same number deliberately: the two used to disagree
    about what "can afford" meant, and the runway term was the stricter of them by an
    order of magnitude. `main.py` emits `LAND_ORDER` after the head's rows so the wire
    matches what this function assumed -- moving the emission without moving this
    read, or the reverse, puts them back out of step.

    This is the only decision in this module the network has no vote in, and it
    deliberately does not go through the mask: a mask entry is a claim about what the
    simulator would accept, and this is a claim about what is worth doing. Keeping
    the two apart is what let the `BUY_LAND` column leave the head without the
    decision leaving with it.
    """

    money = liquid_cash(obs, farm, private)
    extra = max(0, len(farm.get("unlocked_quadrants", ["NW"]) or ["NW"]) - 1)
    if extra >= len(donor.LAND_PRICES) or money < float(donor.LAND_PRICES[extra]):
        return False
    score = land_score(obs, farm, private, money, extra, roles, config)
    return score > 0.0 and score >= float(LAND_BUY_SET_POINT)


#: Cash the reserve floor keeps out of every buy, whichever is larger.
#:
#: v1 spent 18.4% of its ladder steps under 100 coins against 8.5% for its
#: opponents (`docs/dqn_market_free_kaggle_episode_review.md`), because nothing in
#: the mask distinguishes "can pay for this" from "can pay for this and still run a
#: farm". The fraction is what does the work in the mid-game, where a wheat flip is
#: small beside the wallet; the absolute floor is what does it at the start, where
#: 3,000 coins times any sane fraction is less than a single melon seed.
#:
#: A floor, not a strategy: it shuts *buy* rows only, never SELL, HIRE or the
#: terminator, so it can only stop the policy spending -- and the reserve is
#: measured against the money the turn opened with, so a turn that sells first
#: raises its own ceiling rather than being locked out by its opening balance.
#:
#: Watch `broke_frac` in the batch metadata. If it does not fall below v1's 0.184,
#: this is too small; if `orders_per_turn` collapses toward 1, it is too large.
RESERVE_ABS = 250.0
RESERVE_FRAC = 0.08

#: Sub-decisions one turn may take, terminator included.
#:
#: `maxMarketOrdersPerTurn` bounds the *wire*, and `HIRE n` spends n slots for one
#: decision, so the wire cap alone does not bound the loop -- eleven single-slot
#: sells would. This is the loop's own bound and is deliberately one above the wire
#: cap so a turn that fills all ten slots still gets to choose its terminator
#: rather than having the loop cut it off, which would leave the STOP row untrained
#: on exactly the turns that reach the cap.
MAX_DECISIONS_PER_TURN = int(donor.MAX_MARKET_ORDERS) + 1


class PendingTurn:
    """What this turn has already committed, as the next sub-decision sees it.

    v1 decided one order per turn, so the observation was the whole story. v2
    decides up to `MAX_DECISIONS_PER_TURN` times against a market that its own
    earlier orders have already moved, and *nothing in the observation moves* --
    the simulator applies the whole list at once, after the agent returns. So the
    wallet, the shed, the market inventory, the seed allowance and the hire ladder
    all have to be carried forward here or the second order of a turn is priced
    against a world that no longer exists.

    The accounting mirrors `donor._market_actions`, which walks exactly this state
    while building its own multi-order list. Two rules from the README's market
    section are load-bearing and easy to get backwards:

      * a sale is quoted at the **pre-sell** inventory, unit by unit, rising
      * a buy is quoted at the **post-buy** inventory, unit by unit, falling

    which is why `_price_at(item, inventory + offset)` runs forward for SELL and
    `_price_at(item, inventory - offset)` runs backward for BUY_PRODUCT. Get them
    the same way round and a buy-then-sell of one item shows a profit that the
    simulator will not pay.

    `expected_delta` is the sum of those quotes, signed. It is an estimate, not a
    promise: orders interleave with the opponent's, so a price we quoted may be
    taken first. `market_state.order_refused` compares it against the realised
    delta to tell a refused order from a filled one, which is a strictly better test
    than v1's money-delta sign -- that one cannot read a turn holding both a buy and
    a sell. The reward no longer reads it.
    """

    __slots__ = (
        "money",
        "turn_start_money",
        "shed",
        "occupancy",
        "capacity",
        "market_inventory",
        "seed_room",
        "animal_room",
        "seeds_bought",
        "hires_today",
        "hire_cap",
        "slots",
        "max_slots",
        "ops",
        "events",
        "land",
        "expected_delta",
        "obs",
        "config",
    )

    def __init__(
        self,
        obs: dict,
        farm: dict,
        private: dict,
        field: dict,
        config=None,
        plantable: dict[str, int] | None = None,
        housing: dict[str, int] | None = None,
        roles: dict | None = None,
    ):
        capacity = int(donor._cfg(config, "shedCapacity", donor.SHED_CAPACITY))
        post_shed, _ = donor._post_field_storage(private, field or {}, capacity)
        raw_shed = private.get("shed", {}) or {}
        # The permissive view of storage, exactly as `feasible_actions` takes it:
        # `_post_field_storage` is a model of the simulator, so where the two
        # disagree the one that leaves an action open wins.
        self.shed = {
            item: max(int(raw_shed.get(item, 0) or 0), int(post_shed.get(item, 0) or 0))
            for item in {*raw_shed, *post_shed}
        }
        self.occupancy = min(
            sum(max(0, int(value or 0)) for value in raw_shed.values()),
            sum(max(0, int(value or 0)) for value in post_shed.values()),
        )
        self.capacity = capacity
        self.money = float(farm.get("money", 0) or 0)
        self.turn_start_money = self.money
        self.market_inventory = dict((obs.get("market", {}) or {}).get("inventory", {}) or {})
        self.seed_room = dict(
            plantable if plantable is not None else seed_room(obs, farm, private, roles)
        )
        #: Free animal tiles this turn may still commit head to, per species. One
        #: number shared across species because they share the ground, so `apply`
        #: decrements all of them together.
        self.animal_room = dict(
            housing if housing is not None else animal_room(obs, farm, private, roles)
        )
        self.seeds_bought: dict[str, int] = {}
        self.hires_today = int(farm.get("hires_today", 0) or 0)
        #: Most hands this turn may end the day with -- `labor.hire_cap`, read once
        #: off the opening board. `feasible_actions` shuts the `HIRE` rungs that
        #: would carry `hires_today` past it.
        self.hire_cap = labor.hire_cap(
            obs, farm, private, roles if roles is not None else herd.field_roles(obs, farm, private)
        )
        self.slots = 0
        self.max_slots = int(donor._cfg(config, "maxMarketOrdersPerTurn", donor.MAX_MARKET_ORDERS))
        self.ops: list[str] = []
        #: `(op, item, quantity, cash)` per queued order, cash signed the way the
        #: wallet moves. `lots.match` splits a turn back into its orders with this;
        #: without it a turn that buys and sells prices both against the net.
        self.events: list[tuple[str, str | None, int, float]] = []
        self.land = False
        self.expected_delta = 0.0
        self.obs = obs
        self.config = config

    @property
    def slots_left(self) -> int:
        return max(0, self.max_slots - self.slots)

    @property
    def decisions(self) -> int:
        """Sub-decisions taken, terminator excluded -- one per queued op."""

        return len(self.ops)

    @property
    def reserve(self) -> float:
        return max(RESERVE_ABS, RESERVE_FRAC * self.turn_start_money)

    def _inventory(self, item: str) -> int:
        raw = self.market_inventory.get(item)
        return donor.MARKET_I0 if raw is None else int(raw)

    def sale_proceeds(self, item: str, quantity: int) -> float:
        inventory = self._inventory(item)
        return float(
            sum(donor._price_at(item, inventory + offset, self.obs) for offset in range(quantity))
        )

    def _commit_sale(self, item: str, quantity: int) -> tuple[int, float, int]:
        inventory = self._inventory(item)
        proceeds = 0.0
        committed = 0
        for _ in range(max(0, int(quantity))):
            price = float(donor._price_at(item, inventory, self.obs))
            proceeds += price
            committed += 1
            # Floor-price sales pay, but they do not add supply in the simulator.
            if price > 1.0:
                inventory += 1
        return committed, proceeds, inventory

    def purchase_cost(self, item: str, quantity: int) -> float:
        inventory = self._inventory(item)
        return float(
            sum(
                donor._price_at(item, inventory - offset, self.obs)
                for offset in range(1, quantity + 1)
            )
        )

    def hire_cost(self, hands: int) -> float:
        multiplier = float(donor._cfg(self.config, "farmHandCostMult", 1))
        return multiplier * float(
            _FIB_PREFIX[min(self.hires_today + hands, _FIB_PREFIX.shape[0] - 1)]
            - _FIB_PREFIX[min(self.hires_today, _FIB_PREFIX.shape[0] - 1)]
        )

    def commit_land(self) -> None:
        """`should_buy_land` fired. One slot, and the wallet knows about it.

        In v1 the quadrant took the whole turn because the turn was one order. Here
        it is one slot of ten, so the rest of the turn has to see the money leave.
        """

        extra = self.land_price_index()
        if extra is not None:
            price = float(donor.LAND_PRICES[extra])
            self.money -= price
            self.expected_delta -= price
            self.events.append(("BUY_LAND", None, 1, -price))
        self.land = True
        self.slots += 1

    def land_price_index(self) -> int | None:
        farms = self.obs.get("farms", []) or []
        player = int(self.obs.get("player", 0) or 0)
        farm = farms[player] if len(farms) > player else {}
        extra = max(0, len(farm.get("unlocked_quadrants", ["NW"]) or ["NW"]) - 1)
        return extra if extra < len(donor.LAND_PRICES) else None

    def view(self, obs: dict, farm: dict, private: dict) -> tuple[dict, dict, dict]:
        """`(obs, farm, private)` as they will read *after* what this turn queued.

        Without this the encoder is blind inside the turn. The simulator applies the
        whole order list at once, after the agent returns, so `farm["money"]`,
        `private["shed"]` and the market inventory are all frozen at the turn's
        opening values however many orders have been queued -- and every block of
        `market_state_features` except block E reads them straight. A policy that has
        just queued `SELL WHEAT 30` would still be shown 30 wheat in the shed and the
        old wallet, and would rank a second identical sale on stock it has already
        promised away. The mask would refuse the repeat; nothing would stop the
        argmax reaching for it, and the Q value it learned would be fitted to a state
        that never existed.

        Shallow copies over the two dicts the market moves, so the tile grid, the
        hands and the opponent's farm stay shared -- none of them move within a turn.

        `farms` is rebuilt too: the opponent-facing columns index it by player, and a
        farm dict that disagreed with its own entry in `farms` is exactly the kind of
        split-brain that shows up later as an unreproducible feature.
        """

        player = int(obs.get("player", 0) or 0)
        farms = list(obs.get("farms", []) or [])
        farm_view = {**farm, "money": self.money, "hires_today": self.hires_today}
        if len(farms) > player:
            farms[player] = farm_view
        obs_view = {
            **obs,
            "farms": farms,
            "market": {
                **(obs.get("market", {}) or {}),
                "inventory": self.market_inventory,
            },
        }
        seeds = dict(private.get("seeds", {}) or {})
        for crop, bought in self.seeds_bought.items():
            seeds[crop] = int(seeds.get(crop, 0) or 0) + bought
        private_view = {**private, "shed": self.shed, "seeds": seeds}
        return obs_view, farm_view, private_view

    def apply(self, order: Order) -> None:
        """Fold one accepted order into the running state."""

        quantity = max(0, int(order.quantity))
        item = str(order.item) if order.item is not None else None
        if order.op == "SELL" and item is not None:
            quantity = min(quantity, max(0, int(self.shed.get(item, 0) or 0)))
            if quantity <= 0:
                return
            quantity, proceeds, inventory = self._commit_sale(item, quantity)
            self.money += proceeds
            self.expected_delta += proceeds
            self.shed[item] = max(0, int(self.shed.get(item, 0) or 0) - quantity)
            self.occupancy = max(0, self.occupancy - quantity)
            self.market_inventory[item] = inventory
            self.events.append((order.op, item, quantity, proceeds))
        elif order.op == "BUY_PRODUCT" and item is not None:
            quantity = min(quantity, max(0, self.capacity - self.occupancy))
            if quantity <= 0:
                return
            cost = self.purchase_cost(item, quantity)
            self.money -= cost
            self.expected_delta -= cost
            self.shed[item] = int(self.shed.get(item, 0) or 0) + quantity
            self.occupancy += quantity
            self.market_inventory[item] = max(0, self._inventory(item) - quantity)
            self.events.append((order.op, item, quantity, -cost))
        elif order.op == "BUY_SEED" and item is not None:
            cost = float(donor.CROPS[item]["seed"]) * quantity
            self.money -= cost
            self.expected_delta -= cost
            # Seed lives in its own slot and never touches shed occupancy.
            self.seed_room[item] = max(0, int(self.seed_room.get(item, 0) or 0) - quantity)
            self.seeds_bought[item] = int(self.seeds_bought.get(item, 0) or 0) + quantity
            self.events.append((order.op, item, quantity, -cost))
        elif order.op == "BUY_ANIMAL" and item is not None:
            quantity = min(
                quantity,
                max(0, self.capacity - self.occupancy),
                max(0, int(self.animal_room.get(item, 0) or 0)),
            )
            if quantity <= 0:
                return
            cost = float(donor.ANIMALS[item]["cost"]) * quantity
            self.money -= cost
            self.expected_delta -= cost
            self.shed[item] = int(self.shed.get(item, 0) or 0) + quantity
            self.occupancy += quantity
            # Every species, not just the one bought: COW and SHEEP share the
            # pasture and a bare COOP is re-roled by `herd._species_for`, so the
            # tile this head just claimed is gone for all of them. Without this a
            # turn buys four `BUY_ANIMAL COW 4` into two free tiles and the mask
            # never notices, which is the whole failure one turn down.
            for species in self.animal_room:
                self.animal_room[species] = max(0, self.animal_room[species] - quantity)
            self.events.append((order.op, item, quantity, -cost))
        elif order.op == "HIRE":
            hands = max(1, quantity)
            cost = self.hire_cost(hands)
            self.money -= cost
            self.expected_delta -= cost
            self.hires_today += hands
            self.events.append((order.op, None, hands, -cost))
        self.ops.append(order.op)
        self.slots += len(order.to_wire())


#: Memo for `sell_depth_cap`, keyed on the two things the answer depends on.
#:
#: Market inventory is an integer that moves a few units a turn, and the mask is
#: rebuilt up to eleven times within one turn, so the hit rate is high and the miss
#: costs at most `len(QUANTITIES)` price evaluations. Bounded and cleared wholesale
#: rather than evicted: the keys of one episode are the keys of the next.
_DEPTH_CAP_CACHE: dict[tuple[str, int, int], int] = {}
_DEPTH_CAP_CACHE_MAX = 8192


def stock_ceiling(held: int, ladder: tuple[int, ...] = QUANTITIES) -> int:
    """Largest rung worth offering against `held` units: the smallest rung >= it.

    A sale is bounded by the shed, so `SELL WHEAT 45` on three wheat and
    `SELL WHEAT 4` on three wheat put the same three units on the wire for the same
    coins. Every rung above this one is that same action spelled differently, so
    shutting them removes no reachable outcome -- which is the *proof* this
    function's own contract asks for, rather than a measurement that the big rungs
    are usually a bad idea.

    It is not cosmetic, and the cost was on the reward rather than on the fill.
    `PendingTurn.apply` quotes the order at its full quantity, so an over-sized rung
    inflates `expected_delta`; `market_state.order_refused` then tests the **whole
    turn** against `FILL_FRACTION` of that quote and reads a turn whose legs all
    filled as refused. Measured on `market-free-splitladder-v1`, 2,113 of 7,910 sell
    units offered per episode had no stock behind them, which on
    `market-free-scale-v1` came to `REJECTED_ORDER_PENALTY` firing on nine times the
    simulator's true refusal rate. That charge is gone (see `training/reward.py`);
    capping the rung here is still worth it, because the inflated quote also latches
    the `order_refused` state feature on a turn that traded fine.

    Returns the top rung when `held` runs past the ladder, which is the ordinary
    case for a full shed.
    """

    return next((level for level in ladder if level >= int(held)), ladder[-1])


def sell_depth_cap(item: str, inventory: int, obs: dict | None = None) -> int:
    """Largest ladder rung whose last unit still fills within `MAX_SELL_SLIPPAGE`.

    Always at least `QUANTITIES[0]`: a market can be crashed far enough that even one
    unit is past the bound, and refusing to sell at all is not what the cap is for --
    holding stock the season will liquidate at nothing is strictly worse than selling
    it cheap.
    """

    stock = int(inventory)
    quote = int(donor._price_at(item, stock, obs))
    key = (item, stock, quote)
    cached = _DEPTH_CAP_CACHE.get(key)
    if cached is not None:
        return cached
    floor = (1.0 - MAX_SELL_SLIPPAGE) * float(quote)
    cap = QUANTITIES[0]
    for quantity in QUANTITIES:
        if float(donor._price_at(item, stock + quantity, obs)) < floor:
            break
        cap = quantity
    if len(_DEPTH_CAP_CACHE) >= _DEPTH_CAP_CACHE_MAX:
        _DEPTH_CAP_CACHE.clear()
    _DEPTH_CAP_CACHE[key] = cap
    return cap


def feed_floor(farm: dict, private: dict) -> int:
    """WHEAT the herd will eat over `donor.FEED_STOCK_DAYS`, in units.

    `FEED` consumes one WHEAT a day per head (`kaggriculture.py:491-499`), and two
    consecutive unfed days makes the animal escape and leave the structure behind
    (`:801-806`). Upkeep is therefore **in kind, never in cash**: starving the herd
    costs assets and shows up in no money column, which is why nothing in this agent
    noticed it. The donor guards it at `donor.py:1768` -- `animal_pipeline *
    FEED_STOCK_DAYS`, held out of what it is willing to sell -- but that lives in
    `donor._market_actions`, which v3d never calls.

    Counted the way `animal_room` counts: placed head off the board plus head bought
    and not yet placed, since both eat.

    Returns 0 on a malformed board. `feasible_actions` never raises, and the
    permissive reading is the safe one for a rule that *shuts* rows.
    """

    try:
        pipeline = sum(
            1
            for row in (farm or {}).get("tiles", []) or []
            for tile in (row if isinstance(row, list) else [row])
            if herd._animal_on(tile) is not None
        )
        pipeline += sum(
            max(0, int(donor._private_item_total(private, animal)))
            for animal in herd.ANIMALS
        )
    except Exception:
        return 0
    return int(pipeline) * int(donor.FEED_STOCK_DAYS)


def feasible_actions(
    obs: dict,
    farm: dict,
    private: dict,
    field: dict,
    config=None,
    roles: dict | None = None,
    plantable: dict[str, int] | None = None,
    housing: dict[str, int] | None = None,
    pending: PendingTurn | None = None,
) -> np.ndarray:
    """[ACTION_COUNT] bool: True where the simulator would commit at least one unit.

    Feasibility, not preference. Every rule is read straight off `_commit_unit`,
    `_do_hire` and `_do_buy_land`, so a False means the game would refuse the order
    outright -- never that it is unwise.

    **Forbidding an order the game would accept requires a proof that the order
    cannot be worth taking.** Not a measurement that it usually is not: a proof. One
    column now meets that bar, and no second should be added on statistics alone.

    * `BUY_SEED`, past the plan's deficit. Seed is quoted at the constant
      `CROPS[item]["seed"]` and commits into `private["seeds"]`, while `SELL` reads
      `private["shed"]` only -- so seed can never be sold back, at any price, in any
      state. `planting.py` drops what the tile plan cannot absorb, held seed does not
      expire, and the reward stream liquidates it at nothing. A coin spent past the
      deficit is therefore worth strictly less than the coin, always. See
      `SEED_HEADROOM_TILES`.

    Four rules that decide the shape:

    * An action is feasible when *one* unit would commit. `SELL WHEAT 45` against a
      single wheat fills partially, which is what the simulator does, so quantity
      never affects feasibility. `HIRE` and `BUY_SEED` are the exceptions and for the
      same reason: both spend on the whole quantity, so both are tested on it.
    * `field` is this turn's unit plan and is required: the environment applies
      every DROP, PLACE and PICKUP *before* the market phase, so a hand dropping a
      harvested melon this turn makes `SELL MELON` commit even though the
      observation still shows an empty shed.
    * Where the raw and post-field views disagree the permissive one wins.
      `_post_field_storage` is a model of the simulator, so an error there can only
      leave an extra action open -- which the game then refuses -- and never remove
      a real one.
    * That permissiveness runs one way only. Where this function *does* forbid an
      accepted order it is under the proof rule above, never under a model of the
      simulator that might be wrong.
    """

    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    # Doing nothing is always available, so the mask can never be empty and the
    # argmax always has something to land on. In v2 it is also the terminator: the
    # inner loop stops on it, so it is the one row that must never be shut.
    mask[NOOP_INDEX] = True

    capacity = int(donor._cfg(config, "shedCapacity", donor.SHED_CAPACITY))
    if pending is None:
        # The v1 path, unchanged. `tests/test_dqn_market_free_v3e.py` pins it
        # element-wise against v1's mask, which is what makes the fork's inherited
        # feasibility tests mean anything.
        money = float(farm.get("money", 0) or 0)
        raw_shed = private.get("shed", {}) or {}
        post_shed, _ = donor._post_field_storage(private, field or {}, capacity)

        def _held(item: str) -> int:
            return max(int(raw_shed.get(item, 0) or 0), int(post_shed.get(item, 0) or 0))

        occupancy = min(
            sum(max(0, int(value or 0)) for value in raw_shed.values()),
            sum(max(0, int(value or 0)) for value in post_shed.values()),
        )
        inventory = (obs.get("market", {}) or {}).get("inventory", {}) or {}
        hires_today = int(farm.get("hires_today", 0) or 0)
        spendable = money
        # No wire budget without a turn to spend it against. The cap below is a v2
        # concept -- it exists because ten slots are now shared between several
        # orders -- and applying it here would move the v1 mask this branch exists
        # to reproduce. v2 never reaches inference through this branch.
        slots_left = None
        liquidating = False
    else:
        # Everything the turn has already committed. `spendable` is what separates
        # the two paths: affordability is still tested against `money`, because that
        # is what the simulator will refuse on, while the reserve floor rides on top
        # as a policy limit and only ever narrows the buy rows.
        money = pending.money
        shed = pending.shed

        def _held(item: str) -> int:
            return int(shed.get(item, 0) or 0)

        occupancy = pending.occupancy
        inventory = pending.market_inventory
        hires_today = pending.hires_today
        spendable = max(0.0, money - pending.reserve)
        slots_left = pending.slots_left
        # The donor's own endgame flag, not a day threshold of our own: it already
        # knows when the field has stopped planting and started carrying produce to
        # the market, and a second definition would drift from it.
        liquidating = bool((field or {}).get("liquidation"))

    has_room = occupancy < capacity

    # Items this turn has already been on the other side of. A turn that buys an
    # item and sells it back moves the market inventory down and then straight back
    # up the same price levels, so the round trip is worth *exactly* zero coins at
    # every quantity -- verified against `donor._price_at`.
    #
    # v2 converged on precisely that: 47% of its turns were
    # `BUY_PRODUCT WHEAT 45 | SELL WHEAT 45` five times over, filling the whole wire
    # for no P&L. It was the reward-optimal move under the shaped reward of the
    # day: the wash is free, it counted as "not idle" so it dodged the idle
    # penalty, and `order_refused`'s zero-quote branch exempted it from the
    # rejected-order charge. It strictly dominated NOOP. (Both charges are gone.)
    #
    # Shut structurally rather than priced. A penalty has to be tuned against the
    # behaviour it suppresses and can be learned around at the margin; a mask
    # cannot. v1 could not wash at all -- one order per turn -- and v1 is the
    # lineage this fork has to beat.
    sold_this_turn: set[str] = set()
    bought_this_turn: set[str] = set()
    if pending is not None:
        for op, event_item, _quantity, _cash in pending.events:
            if event_item is None:
                continue
            if op == "SELL":
                sold_this_turn.add(str(event_item))
            elif op == "BUY_PRODUCT":
                bought_this_turn.add(str(event_item))

    for item, rows in _SELL_ROWS.items():
        if _held(item) <= 0:
            continue
        if item in bought_this_turn:
            # Bought this turn already; selling it back is the wash.
            continue
        if item in sold_this_turn:
            # One SELL per item per turn. Unlike the wash rule and the stock
            # ceiling this is a *preference* and not a proof -- 45 + 45 units is a
            # reachable outcome no single rung reproduces, because `QUANTITIES`
            # stops at 45 -- so it is carried on a measured ceiling instead.
            #
            # Across 300 ladder seats a second same-item SELL in one turn is 1.9%
            # of sell orders, 1.4% of units and 2.3% of selling turns: the agents
            # that win do not need it. Against that, `market-free-splitladder-v1`
            # put **72.7%** of its sell orders there -- 843 repeats of 1,160 -- and
            # each one spends a wire slot on a fill the previous rung could have
            # carried, because `pending.apply` has already moved both the shed and
            # the quote the next rung is priced against.
            #
            # The bound that makes the preference safe is the ladder above: 45 units
            # a turn is 1,080 a day, against a ladder seat's 78. And the precedent
            # that says preferences are dangerous is v10's land gate -- the policy
            # stopped taking the action rather than re-timing it. The equivalent
            # failure here is `sell_units_per_episode` collapsing rather than
            # concentrating; watch it beside `orders_per_decision`.
            continue
        mask[rows] = True
        # Both narrowings below are v2-path only. The unpended branch reproduces
        # v1's mask element-wise and `tests/test_dqn_market_free_v3e.py` pins it
        # there, so they ride with the turn that made v2 a sequence of orders --
        # which is also the only branch that can reprice as its own sales move the
        # inventory.
        if pending is None:
            continue
        # Deduplication, not a preference, which is what lets it past this
        # function's own bar: every rung above `stock_ceiling` sells the identical
        # units for the identical coins, because the shed is what bounds the fill.
        # They are the same action under a different name.
        #
        # Kept on through liquidation, unlike the depth cap: this is not about
        # price. It costs nothing there and the endgame is the turn most short of
        # wire slots.
        mask[rows] &= _ROW_QUANTITY[rows] <= stock_ceiling(_held(item))
        # Exempt while liquidating: the endgame is short of wire slots, not of
        # price, and stock the season ends holding is worth zero however cheaply it
        # went.
        if liquidating:
            continue
        # The herd's feed, held back from what may be sold.
        #
        # **This is a policy rule, not a proof, and it is the second one in this
        # function** -- `labor.hire_cap` below is the first. It does not meet the
        # preamble's bar: selling wheat while holding animals is not provably
        # worthless, because the agent may intend to buy it back, or to let the herd
        # go. It is here because the outcome it prices is invisible everywhere else.
        # `FEED` eats one WHEAT a head a day, two unfed days and the animal escapes
        # (`kaggriculture.py:491-499`, `:801-806`), and the loss lands entirely in
        # assets -- no money column moves, so neither the reward, the gate, nor any
        # diagnostic reports it. The agent cannot learn from a signal nothing emits.
        #
        # What prompted it: `stock_resale_share` -- the share of bought WHEAT and
        # FERTILIZER flipped rather than used -- climbed 0.470 -> 0.685 -> 0.765 ->
        # 0.799 over the first four updates of `soloanchor-v2`, against a ladder
        # champion `diagnostics.py:356-359` describes as buying "wheat to feed a herd
        # and resells almost none of it".
        #
        # It is the donor's own arithmetic, and it caps rather than shuts: only the
        # rungs above the spare wheat close, so selling the surplus stays open at
        # every size it was open at before. `donor.py:1778` additionally lifts it in
        # the last turns; the `liquidating` exemption above already does that here.
        #
        # The failure mode to watch is the one v10's land gate produced -- the policy
        # abandoning the action instead of re-timing it. Here that is
        # `stock_units_per_episode` or the WHEAT share of sell volume collapsing
        # rather than the resale share falling. Watch it beside `orders_per_decision`.
        if item == "WHEAT":
            spare = max(0, _held(item) - feed_floor(farm, private))
            mask[rows] &= _ROW_QUANTITY[rows] <= spare
        stock = int(inventory.get(item, donor.MARKET_I0) or donor.MARKET_I0)
        mask[rows] &= _ROW_QUANTITY[rows] <= sell_depth_cap(item, stock, obs)


    # Buys are shut for the rest of the season once the field is liquidating.
    # Terminal stock is worth nothing -- v1 finished holding 3,572 coins of shed
    # against its opponents' 275 -- and seed bought now cannot reach a harvest.
    # SELL and the terminator stay open, which is the whole endgame.
    if not liquidating:
        for item, rows in _BUY_PRODUCT_ROWS.items():
            # Quoted at the post-buy inventory, which is what `_commit_unit` does --
            # not the sale price the observation carries at current inventory.
            stock = int(inventory.get(item, donor.MARKET_I0) or donor.MARKET_I0)
            if not has_room or item in sold_this_turn:
                # Sold this turn already; buying it back is the wash. See the
                # `sold_this_turn` comment above.
                continue
            # Per *rung* against the wallet, not one unit against the wallet.
            #
            # This used to be `spendable >= price_at(stock - 1)` opening the whole
            # row set, so a single affordable unit unlocked every quantity up to 45.
            # The simulator does not refuse an order it cannot fill -- it part-fills
            # it, draining the wallet to zero -- so the policy could and did pick a
            # rung it could not pay for. Measured on the Kaggle ladder
            # (`docs/dqn_market_free_kaggle_episode_review.md`): 31.5% of
            # `BUY_PRODUCT` orders were sized past the wallet against opponents'
            # 3.9%, with 0.0% refused, and cash starvation followed -- 18.4% of
            # steps under 100 coins. v2 then ran 330 batches with
            # `rejected_order_rate` flat at 0.22-0.25 and never learned its way out.
            #
            # `BUY_SEED` below has always masked per rung for exactly this reason;
            # the two are now symmetric. This is a *policy* change, not a
            # correctness fix -- the rungs it shuts were legal orders -- so it
            # belongs to v3's lineage and not to a patch of v2's.
            quantities = _ROW_QUANTITY[rows]
            unit_prices = np.asarray(
                [
                    float(donor._price_at(item, stock - offset, obs))
                    for offset in range(1, int(quantities.max()) + 1)
                ],
                dtype=np.float64,
            )
            # `cost[n - 1]` is what buying `n` units costs, matching
            # `PendingTurn.purchase_cost` term for term so the mask and the wallet
            # accounting cannot disagree about the price of the same order.
            cost = np.cumsum(unit_prices)
            room = max(0, capacity - occupancy)
            mask[rows] = (quantities <= room) & (cost[quantities - 1] <= spendable)

    # Seed rows open only where the plan has a deficit to fill, and only at rungs the
    # wallet covers *after* `choose_order` clamps them. Both halves are load-bearing:
    # without the deficit test the 55 seed rows are open on essentially every turn of
    # every episode, and without the affordability test on the clamped quantity a
    # 3,600-coin order is offered against 80 coins and the simulator part-fills it by
    # draining the wallet.
    if pending is not None:
        room = pending.seed_room
    else:
        room = plantable if plantable is not None else seed_room(obs, farm, private, roles)
    if liquidating:
        room = {}
    # The animal half of the same idea, resolved the same three ways so a turn in
    # progress reads what it has already committed rather than the turn's opening
    # board. `liquidating` needs no branch here: the `BUY_ANIMAL` loop below is
    # already inside `if not liquidating`.
    if pending is not None:
        stalls = pending.animal_room
    else:
        stalls = housing if housing is not None else animal_room(obs, farm, private, roles)
    for crop, rows in _BUY_SEED_ROWS.items():
        available = int(room.get(crop, 0))
        if available <= 0:
            continue
        price = float(donor.CROPS[crop]["seed"])
        quantities = _ROW_QUANTITY[rows]
        # Rungs above the allowance are shut rather than clamped down onto it. Both
        # bound the spend identically, but clamping would leave every rung above the
        # allowance aliased onto the same emitted order -- eleven rows the argmax can
        # pick between for one outcome, which is the row-count bias this cap is also
        # meant to relieve. `_BUY_SEED_ROWS[crop]` is built in `QUANTITIES` order, so
        # these line up positionally exactly as `_HIRE_HANDS` does.
        mask[rows] = (quantities <= available) & (quantities * price <= spendable)

    if not liquidating:
        for animal, rows in _BUY_ANIMAL_ROWS.items():
            if not has_room:
                continue
            # Housing and the calendar, from `animal_room`. Rungs above free
            # housing are shut rather than clamped, for the reason `BUY_SEED`
            # states above: clamping aliases every rung above the allowance onto
            # one emitted order, which is row-count bias rather than relief from
            # it. A species past its last productive day drops out of `stalls`
            # entirely and takes its whole ladder with it.
            stall = min(int(stalls.get(animal, 0)), max(0, capacity - occupancy))
            if stall <= 0:
                continue
            # Per rung, as `BUY_PRODUCT` and `BUY_SEED` already are. This was the
            # last place a single affordable unit opened the whole ladder: animals
            # are flat-priced, so `n` head cost `n * cost` and a wallet covering one
            # was letting `BUY_ANIMAL 4` through. `ANIMAL_QUANTITIES` is (1, 2, 4)
            # so the overshoot was bounded at 4x rather than `BUY_PRODUCT`'s 45x,
            # but the simulator part-fills either way and drains the wallet doing it.
            unit = float(donor.ANIMALS[animal]["cost"])
            quantities = _ROW_QUANTITY[rows]
            mask[rows] = (quantities <= stall) & (quantities * unit <= spendable)

    # Every hand in the order has to be affordable, not just the first. This is
    # deliberately stricter than `_do_hire`, which commits what it can and stops
    # when the money runs out -- do not "fix" the asymmetry. It is still
    # feasibility: `HIRE 13` against 700 coins partially commits as four hands, and
    # `HIRE 4` is its own index on the same ladder, so no reachable outcome is lost,
    # only the mislabelled route to it.
    #
    # Hiring is spending, but it is not a *buy*: hands are the only thing the wallet
    # converts into field work, and shutting them behind the reserve floor would
    # starve the farm to protect a cash pile that exists to fund the farm. So this
    # row tests `money`, like v1, and the floor above does not reach it.
    hire_mult = float(donor._cfg(config, "farmHandCostMult", 1))
    if hires_today + int(_HIRE_HANDS[-1]) < _FIB_PREFIX.shape[0]:
        costs = hire_mult * (
            _FIB_PREFIX[hires_today + _HIRE_HANDS] - _FIB_PREFIX[hires_today]
        )
        mask[_HIRE_ROWS] = money >= costs

    # One hire action per turn. `HIRE 7` then `HIRE 3` reaches the wire as ten
    # identical `["HIRE"]` entries, indistinguishable from the single `HIRE 10` row
    # the ladder already carries -- the same aliasing the per-hand affordability test
    # above exists to remove, rebuilt one sub-decision later. It also makes the turn
    # unrecoverable from a stored replay: `clone._chosen_sequence` reads ten entries
    # and can only map them to one row, so a cloned episode came up short by exactly
    # the split hires and every later `order_step` slid.
    if pending is not None and "HIRE" in pending.ops:
        mask[_HIRE_ROWS] = False

    # Hands beyond what the field's jobs can use. A policy cap rather than a
    # feasibility one, and the only one besides the reserve floor: `labor.py` has
    # the measurement. The v1 branch is left alone -- it is pinned element-wise
    # against v1's mask and inference never reaches it.
    if pending is not None:
        mask[_HIRE_ROWS] &= (hires_today + _HIRE_HANDS) <= int(pending.hire_cap)

    # Nothing may be queued that the wire cannot carry. One slot per order, except
    # `HIRE n`, which is n separate `["HIRE"]` entries on the same turn -- so the
    # hire ladder is capped by the slots left rather than shut wholesale.
    if slots_left is not None:
        if slots_left <= 0:
            mask[:] = False
            mask[NOOP_INDEX] = True
        else:
            mask[_HIRE_ROWS] &= _HIRE_HANDS <= slots_left

    return mask


@dataclass(frozen=True, slots=True)
class OrderDecision:
    state: np.ndarray  # float32 [STATE_DIM]
    order: Order
    chosen: int  # index into ACTIONS
    mask: np.ndarray | None = None  # bool [ACTION_COUNT], feasibility at `state`
    #: `should_buy_land` fired this turn, so `order` is `LAND_ORDER` and `chosen` is
    #: NOOP. The two disagree on purpose: the wire carries a quadrant, and the row the
    #: learner is credited with is the one the network actually contributed, which was
    #: nothing.
    buy_land: bool = False

    def wire(self) -> list[list]:
        """The order list as the simulator wants it; empty on NOOP."""

        return self.order.to_wire()


def _resolve(chosen: int, plantable: dict[str, int] | None) -> Order:
    """The order a row emits, which for seed is the row clamped to the deficit.

    `OrderDecision.chosen` stays the raw row index and only `.order` moves. That
    split is load-bearing: `collect.Recorder.record_decision` stores the index into
    `order_chosen`, `training/rare.py` bincounts it, and the replay buffer's masks are
    keyed on it, so a clamped index would silently re-label which action the learner
    took. The clamp is a property of the fill, not of the choice.
    """

    order = ACTIONS[chosen]
    if plantable is None or order.op != "BUY_SEED":
        return order
    available = int(plantable.get(str(order.item), 0))
    if available <= 0 or available >= int(order.quantity):
        return order
    return Order(order.op, order.item, available)


def choose_order(
    state: np.ndarray,
    network,
    mask: np.ndarray | None = None,
    plantable: dict[str, int] | None = None,
    buy_land: bool = False,
) -> OrderDecision:
    """Greedy argmax over the feasible action head, with land taken out of it.

    Exploration lives in the *weights*, and neither NoisyNet nor the directed
    widening in `training/rare.py` appears here: both are zero-mean perturbations of
    the network, so this stays a plain argmax over whatever it just said. Keeping it
    that way is what stops exploration forcing a purchase the policy has learned is
    ruinous -- 99 of the 199 actions spend money.

    `mask` restricts the argmax to what the simulator would actually accept.

    `buy_land` is `should_buy_land`'s answer and is not a row: the head has no
    `BUY_LAND` column to rank, so the caller supplies the decision and the network is
    not consulted. The quadrant takes the whole turn -- one market order is what this
    module emits and `lots.py` reconciles the wallet against -- so the returned
    decision carries `LAND_ORDER` on the wire and NOOP as the row the learner is
    credited with. See `LAND_BUY_SET_POINT`.
    """

    if mask is None:
        mask = np.ones(ACTION_COUNT, dtype=np.bool_)

    if buy_land:
        # Resolved before the ranking, not inside it. The network still scored the
        # state and the score is still discarded, which is the whole point of taking
        # the row off the head rather than masking it.
        return OrderDecision(state, LAND_ORDER, NOOP_INDEX, mask, buy_land=True)

    scores = np.asarray(network(state)).reshape(ACTION_COUNT)
    chosen = int(np.argmax(np.where(mask, scores, MASKED_SCORE)))
    return OrderDecision(state, _resolve(chosen, plantable), chosen, mask)
