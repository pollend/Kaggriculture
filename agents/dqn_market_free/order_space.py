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

One order per turn, not a list of ten. The earlier design factored the action
into `MAX_MARKET_ORDERS` independent slots scoring a shared candidate grid, with
total Q as the sum over slots. That sum is the only thing the TD target ever
saw, so no slot received its own credit: once the network valued an order, every
slot's argmax picked *that same order*, and one good buy became a tenfold spend.
Measured over fourteen updates, every candidate produced that way finished a
3,000-coin season near zero. Decisions here are turn-scale -- 720 an episode --
so a single order per turn is ample throughput, and it makes the value function
an ordinary Q(s, a) again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.dqn_market_free import donor

# The state half lives in `market_state.py` and is imported directly by whoever
# needs it. It uses the same schema as the scaffolded agent: prices, market
# inventory, the shed, the herd, and -- the part that matters here -- the donor's
# own planning state, its seed deficit, its animal deficit, its hand target, its
# phase. This module is only about what the agent may *do*.

PRODUCTS: tuple[str, ...] = tuple(donor.PRODUCTS)
CROPS: tuple[str, ...] = tuple(donor.CROPS)
ANIMALS: tuple[str, ...] = tuple(donor.ANIMALS)

#: Quantity ladder for bulk goods. Roughly geometric -- the marginal value of one
#: more unit falls off with the price impact of the fill, so the interesting
#: decisions are at the order of magnitude, not the unit -- but dense at the
#: bottom and stretched to 45.
#:
#: The range is set by what real agents actually do. Stored league replays place
#: orders at every quantity from 1 to 18 and then sparsely up to 45, so a
#: shorter ladder cannot represent the behaviour `training/clone.py` reads out of
#: them: snapping 13 to 16 and 45 to 25 distorts exactly the decisions being
#: cloned. Eleven levels covers the observed range with a worst-case snap error
#: under 20%.
QUANTITIES: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 45)

#: Animals are indivisible capital at 300-500 each against a few thousand in
#: cash, so the ladder is short.
ANIMAL_QUANTITIES: tuple[int, ...] = (1, 2, 4)

#: `HIRE` carries a hand count, and it is the one operation that has to.
#:
#: Hiring N hands means N separate `["HIRE"]` orders on the *same* turn: the
#: fibonacci cost ladder is keyed on `hires_today`, hands last only the day they
#: were hired, and a hand hired at hour 5 has lost a fifth of its shift. Spread
#: over consecutive turns it is a different, worse action.
#:
#: Measured on a full donor episode, this is the difference between a usable
#: demonstration set and a crippled one: of the 505 orders a one-order-per-turn
#: agent had to drop, 314 were repeated HIREs, and the donor issues up to 9 in a
#: single turn at each day boundary.
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
FIELD_ANIMALS = frozenset(donor.CORE_HERD_SEQUENCE)
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

    if order.op in ("NOOP", "BUY_LAND", "HIRE"):
        return True
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

    Index 0 must stay NOOP. A network that scores every action equally breaks
    ties to the lowest index, so an untrained policy trades nothing rather than
    emitting a random order every turn. Doing nothing is a genuinely survivable
    policy; a random one is not, and the difference decides whether the first few
    hundred episodes carry any signal at all.

    Structurally impossible orders are filtered out here rather than masked at
    score time. They were once retained so persisted action indices stayed
    stable across the eligibility rule; nothing persisted uses the old numbering
    any more, and carrying 91 permanently `-inf` rows through every forward pass
    bought nothing.
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
    rows.append(Order("BUY_LAND"))
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
    """

    if op in ("NOOP", "BUY_LAND"):
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
    writes rather than a 186-iteration Python loop. This runs on every turn of
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
_BUY_LAND_ROW = ACTION_INDEX[("BUY_LAND", None, 0)]
NOOP_INDEX = 0

#: `HIRE n` costs `sum(_fib(hires_today) .. _fib(hires_today + n - 1))` times the
#: hand multiplier, because `_do_hire` charges per hand and increments
#: `hires_today` between them. Prefix sums turn that into one subtraction:
#: `_FIB_PREFIX[h + n] - _FIB_PREFIX[h]`.
#:
#: The simulator puts no ceiling on `hires_today` -- only affordability stops it
#: -- so the table is sized well past `MAX_HANDS`. It does not need to cover
#: every reachable value: `_fib(40)` is already 165 million, so a farm that far
#: into a day cannot afford another hand under any money the game produces, and
#: falling off the end leaves the rows masked, which is the right answer.
_FIB_PREFIX = np.concatenate(
    ([0.0], np.cumsum([float(donor._fib(index)) for index in range(64)]))
)
#: `_HIRE_ROWS` is built in `HIRE_QUANTITIES` order, so these line up positionally.
_HIRE_HANDS = np.asarray(HIRE_QUANTITIES, dtype=np.intp)


#: Score given to a masked action before an argmax, in both numpy and torch.
#:
#: Finite rather than -inf, and the same constant on both sides. Finite because
#: a masked score can reach a multiply -- `-inf * 0.0` is nan, not zero, and one
#: nan takes a whole minibatch down through the backward pass -- and because a
#: cross-entropy whose target somehow landed on a masked class would return inf
#: rather than a large, visible number. Shared because the numpy argmax and the
#: torch argmax must break ties identically for identical weights, which is what
#: the parity test between `policy.py` and `training/network.py` pins.
#:
#: -1e4 against Q values of single digits (`reward.MONEY_SCALE` sizes a season's
#: return that way) leaves the ordering in no doubt.
MASKED_SCORE = -1e4


def feasible_actions(
    obs: dict, farm: dict, private: dict, field: dict, config=None
) -> np.ndarray:
    """[ACTION_COUNT] bool: True where the simulator would commit at least one unit.

    This is feasibility, not preference. Every rule below is read straight off
    `_commit_unit`, `_do_hire` and `_do_buy_land` in the environment, so a False
    here means the game would refuse the order outright -- never that the order
    is unwise. Choosing among what remains is left entirely to the network, the
    same way `order_is_structurally_possible` already removes orders the game can
    never accept without expressing a view on the rest.

    `BUY_LAND` was gated on the day and a cash reserve for one session and is
    not any more, so the claim above is unqualified again. The measurement that
    motivated it stands and is recorded in the roadmap: land-buying seats finish
    far behind the rest because `LAND_PRICES` opens at 1,000 against a 3,000
    stake, so the purchase is offered on turn 0 before any revenue exists. What
    the gate could not show is that forbidding it *helps* -- under it the policy
    stopped buying land entirely rather than buying it later. Revisit with that
    evidence rather than from scratch.

    It exists because roughly half the action space is unusable until the farm
    produces something. Measured on a live run: SELL is 88 of the 186 actions,
    the policy emitted an order on 84% of turns, and the simulator accepted 1% of
    them -- it was selling goods it did not own, and could not stop, because
    nothing it could sell would exist until it first bought seed.

    An action counts as feasible when *one* unit would commit. `SELL WHEAT 45`
    against a single wheat in the shed is feasible and fills partially, which is
    exactly what the simulator does, so quantity never affects feasibility.

    `field` is this turn's unit plan and is required. The environment applies
    every DROP, PLACE and PICKUP *before* the market phase (`interpreter` calls
    `_apply_unit_action` for both seats, then `_process_market`), so a hand
    dropping a harvested melon this turn makes `SELL MELON` commit even though
    the observation still shows an empty shed. Reading the raw shed alone would
    forbid an order the game accepts -- the one direction this function must
    never take.

    Where the raw and post-field views disagree the permissive one wins:
    `_post_field_storage` is a model of the simulator, not the simulator, so a
    modelling error can only ever leave an extra action open -- which the game
    then refuses exactly as it does today -- and never remove a real one.
    """

    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    # Doing nothing is always available, so the mask can never be empty and the
    # argmax always has something to land on.
    mask[NOOP_INDEX] = True

    money = float(farm.get("money", 0) or 0)
    capacity = int(donor._cfg(config, "shedCapacity", donor.SHED_CAPACITY))
    raw_shed = private.get("shed", {}) or {}
    post_shed, _ = donor._post_field_storage(private, field or {}, capacity)

    def _held(item: str) -> int:
        return max(int(raw_shed.get(item, 0) or 0), int(post_shed.get(item, 0) or 0))

    occupancy = min(
        sum(max(0, int(value or 0)) for value in raw_shed.values()),
        sum(max(0, int(value or 0)) for value in post_shed.values()),
    )
    has_room = occupancy < capacity
    inventory = (obs.get("market", {}) or {}).get("inventory", {}) or {}

    for item, rows in _SELL_ROWS.items():
        if _held(item) > 0:
            mask[rows] = True

    for item, rows in _BUY_PRODUCT_ROWS.items():
        # Quoted at the post-buy inventory, which is what `_commit_unit` does --
        # not the sale price the observation carries at current inventory.
        stock = int(inventory.get(item, donor.MARKET_I0) or donor.MARKET_I0)
        if has_room and money >= float(donor._price_at(item, stock - 1, obs)):
            mask[rows] = True

    for crop, rows in _BUY_SEED_ROWS.items():
        if money >= float(donor.CROPS[crop]["seed"]):
            mask[rows] = True

    for animal, rows in _BUY_ANIMAL_ROWS.items():
        if has_room and money >= float(donor.ANIMALS[animal]["cost"]):
            mask[rows] = True

    extra = max(0, len(farm.get("unlocked_quadrants", ["NW"]) or ["NW"]) - 1)
    if extra < len(donor.LAND_PRICES) and money >= float(donor.LAND_PRICES[extra]):
        mask[_BUY_LAND_ROW] = True

    # Every hand in the order has to be affordable, not just the first.
    #
    # This is the one rule here that is stricter than `_do_hire`, which commits
    # what it can and stops when the money runs out. It is still feasibility and
    # not preference: `HIRE 13` against 700 coins partially commits as four
    # hands, and `HIRE 4` is its own index on the same ladder, so no outcome the
    # simulator would produce becomes unreachable -- only the mislabelled route
    # to it does. Measured on `market-free-simple-v2`: hire cost is fibonacci in
    # `hires_today`, so one `HIRE 13` is 609 coins against a 3,000 stake, and the
    # policy that learned to emit it was bankrupt by turn four of 720 and spent
    # the rest of the episode in a state where nothing but NOOP was feasible.
    hire_mult = float(donor._cfg(config, "farmHandCostMult", 1))
    hires_today = int(farm.get("hires_today", 0) or 0)
    if hires_today + int(_HIRE_HANDS[-1]) < _FIB_PREFIX.shape[0]:
        costs = hire_mult * (
            _FIB_PREFIX[hires_today + _HIRE_HANDS] - _FIB_PREFIX[hires_today]
        )
        mask[_HIRE_ROWS] = money >= costs

    return mask


@dataclass(frozen=True, slots=True)
class OrderDecision:
    state: np.ndarray  # float32 [STATE_DIM]
    order: Order
    chosen: int  # index into ACTIONS
    mask: np.ndarray | None = None  # bool [ACTION_COUNT], feasibility at `state`

    def wire(self) -> list[list]:
        """The order list as the simulator wants it; empty on NOOP."""

        return self.order.to_wire()


def choose_order(
    state: np.ndarray,
    network,
    mask: np.ndarray | None = None,
) -> OrderDecision:
    """Greedy argmax over the feasible action head.

    Exploration lives in the *weights*, not here. This took an `epsilon` and an
    rng until v13, drawing a uniformly random real order on that fraction of
    turns, and it was retired because the rate is the wrong thing to control: 97
    of the 186 actions spend money, so every exploring turn was a forced purchase
    chosen without reference to what the network believed anything was worth.
    Measured at per-turn rates of 1.0, 0.1, 0.02 and even 0.005, the agent
    finished the season on 5, 185, 48 and 4 coins from a 3,000 float while the
    donor finished above 100,000. NoisyNet perturbs the weights instead, so the
    argmax moves only among actions already near-tied in Q.

    Directed exploration does not appear here either, for the same reason. A
    structurally rare action is reached by widening *its own* sigma row in the
    head -- see `training/rare.py` -- which keeps the mechanism zero-mean and
    keeps this function a plain argmax over whatever the network just said.

    `mask` restricts the argmax to what the simulator would actually accept.
    """

    if mask is None:
        mask = np.ones(ACTION_COUNT, dtype=np.bool_)

    scores = np.asarray(network(state)).reshape(ACTION_COUNT)
    chosen = int(np.argmax(np.where(mask, scores, MASKED_SCORE)))
    return OrderDecision(state, ACTIONS[chosen], chosen, mask)
