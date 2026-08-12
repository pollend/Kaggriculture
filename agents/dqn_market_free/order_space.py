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


@dataclass(frozen=True, slots=True)
class OrderDecision:
    state: np.ndarray  # float32 [STATE_DIM]
    order: Order
    chosen: int  # index into ACTIONS

    def wire(self) -> list[list]:
        """The order list as the simulator wants it; empty on NOOP."""

        return self.order.to_wire()


def choose_order(
    state: np.ndarray,
    network,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
) -> OrderDecision:
    """Greedy argmax over the action head, with epsilon-greedy exploration.

    `epsilon` is the probability that this turn emits a uniformly random *real*
    order. It absorbs what used to be two knobs: a per-turn exploration coin and
    a per-slot act rate, whose product was the only thing that mattered. Keeping
    them separate made the effective rate a number nobody read off the config.

    That effective rate has to stay small, and the reason is measured rather
    than assumed. Exploring at every turn is fatal: at per-turn rates of 1.0,
    0.1, 0.02 and even 0.005 the agent finishes the season on 5, 185, 48 and 4
    coins from a 3,000 float while the donor finishes above 100,000. Random
    capital purchases are what does it -- BUY_LAND is 1,000-4,000 and BUY_ANIMAL
    300-500 -- so the buffer fills with nothing but bankruptcy and the only
    policy consistent with it is NOOP. Which order to emit when exploring is
    still uniform, including state-invalid choices such as unaffordable ones.
    """

    if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
        chosen = int(rng.integers(1, ACTION_COUNT))
    else:
        chosen = int(np.argmax(np.asarray(network(state)).reshape(ACTION_COUNT)))
    return OrderDecision(state, ACTIONS[chosen], chosen)
