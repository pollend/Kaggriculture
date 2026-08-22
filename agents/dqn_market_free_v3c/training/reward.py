"""Reward and its scales for the free-market learner.

Local rather than shared with `dqn_market_scorer`: the scorer's market layer
decides land and cash reserves in code, so sharing the constants would mean
changing the scorer's reward to fix this one.

**The reward is the season's net worth, paid where it moves.** One term carries the
whole objective:

    shaped[t] = d(money + held cost basis)[t] / MONEY_SCALE

The wallet delta prices every fill at the coins it actually moved. The basis delta
beside it is what makes a *purchase* score about nothing instead of its full price:
an 80-coin melon seed is cash -80 and basis +80, and the loss appears on the turn the
lot fails to sell rather than on the turn it was bought. The basis is written off at
the terminal step, where the simulator pays for `farm["money"]` and nothing else, so
the stream telescopes to

    sum(shaped) == (final_money - stake) / MONEY_SCALE

exactly. That is the quantity the league ranks on, with per-turn credit assignment:
maximising what each sale earns and maximising end-of-season capital are the same
objective here. There is no terminal term to keep aligned with the promotion gate,
because the stream already sums to what the gate reads.

Two per-turn charges sit alongside it, and only because the objective is silent
about the difference between them:

    accepted order    priced by the coins it moved
    NOOP             -IDLE_PENALTY
    refused order    -REJECTED_ORDER_PENALTY

A refused order and a NOOP both move zero coins, so without the second charge
emitting junk would score *above* sitting still, which is backwards. Neither is a
view about which trade is good.

**What this replaced, and why so little is left.** Through v3c the stream paid a
compressive terminal `outcome` -- `tanh` of log growth on the seat's own stake --
plus a per-step annuity of it, a blend against mean net worth, a land potential, an
insolvency charge with a floor and a grace, and six non-invariant lot P&L legs
matched through a `lots.match` FIFO ledger and split across the buy and resolution
turns. Twenty-one constants, five of them inert.

The six legs were hand-rolled Monte Carlo reconstructions of coins this stream
dropped, and they existed only because the money delta had been computed since v1
and never paid. They only ever covered lots: HIRE wages, land outlay and shed
overflow were in none of them. Measured over 40 `market-free-splitladder-v1` seats
at generations >= 8 against the winning seat of 120 stored ladder replays, per
episode-seat:

                   gross in   gross out       net   final   margin
    old stream      213,280     211,437     1,843   4,843     0.9%
    ladder winner   147,119      48,081    99,038 102,038    67.3%

The old stream *out-grossed* the ladder champion by 45% and kept 0.9% of it, because
89% of both sides was WHEAT and FERTILIZER round-tripped through the market at a
loss no term had an opinion about. Paying the delta prices every round trip at its
true cost and reaches every flow no lot covers, so the reconstructions have nothing
left to reconstruct.

Three terms were cut for their own reasons rather than as duplicates:

  * **The land potential.** `BUY_LAND` is not a row on the head -- `order_space`
    keeps it off and `land_score` decides it (`order_space.py:156`) -- so the term
    was shaping a transition the policy does not choose.
  * **The insolvency charge, its floor and its grace.** Three constants for one
    guard, and this file's own measurement said it was billing ordinary operation:
    across `market-free-splitladder-v1` generations 8-10 the wallet is under 100
    coins on 40.8% of turns, and `split_02` paid about -1.11 of a possible -1.44
    while still finishing with 2,379 coins in hand. Its argument was that a spent
    wallet reads as one small debit while everything it forecloses arrives only as
    the absence of later fills; a paid money delta prices the outlay where it
    happens, so that argument is gone.
  * **`OPPONENT_WEIGHT`.** Inert since it landed. The opponent's effect on us is
    already in our own money -- one market, and every unit either seat sells
    depresses the price for both. What absolute money cannot see is denial, and
    `corr(final_money, opponent_money)` is 0.006, so nothing cancels and the
    behaviour is measurably absent.

**What was given up, stated plainly.** Linear cash is not compressive. Finishing at
zero costs `-3,000 / MONEY_SCALE = -0.03` where the old `tanh` charged -0.42:
downside is bounded by the stake while upside is unbounded, and with the insolvency
charge gone nothing else opposes a gamble. That is what maximising capital linearly
means. `IDLE_PENALTY` is now the only thing opposing NOOP -- an idle turn scores
exactly 0 where a losing trade scores negative -- and it is the one constant to
re-read if `own_orders_per_decision` collapses. Watch it beside the money either
way: a silent slot converges on the stake and raises `lr_score` mechanically.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agents.dqn_market_free_v3c import donor
from agents.dqn_market_free_v3c.market_state import order_refused
from agents.dqn_market_free_v3c import lots

#: Reward units per coin, sized so a whole season's return lands around 1.0.
#:
#: **100,000, up from 20,000 when the outcome term paid the objective.** At the old
#: scale a ladder-grade season is `99,038 / 20,000 = 4.95`, well outside `SmoothL1`'s
#: beta of 1.0 -- the linear regime, where a 5-unit and a 50-unit TD error give the
#: same gradient. At 100,000 the same season is 0.99 and a correct Q sits inside the
#: quadratic zone, which is the band `updater`'s `max_abs_q` divergence detector was
#: calibrated against.
#:
#: Scaled rather than clipped. Per-step clipping to [-1, 1] is the usual DQN answer
#: and is wrong here: the magnitude of a fill *is* the signal.
#:
#: The cash half and the basis half of the delta divide by this same constant, which
#: is what keeps them in one unit. There is nothing to rebalance if it moves.
MONEY_SCALE = 100_000.0

#: Charged on every turn the policy emits no order at all.
#:
#: **The only term opposing NOOP.** Under a net-worth delta an idle turn scores
#: exactly 0 while any losing trade scores negative, so idling is otherwise weakly
#: dominant -- the attractor this family has collapsed into repeatedly. Everything
#: that used to oppose it lived outside the reward, so the machinery was fighting the
#: objective instead of stating it.
#:
#: Three constraints on the shape and the size:
#:
#:   * Decays linearly to zero at the deadline, so a fully idle season pays ~0.072,
#:     7% of what a ladder-grade season pays. An idle turn on day 1 forfeits a season
#:     of compounding; one on day 29 forfeits almost nothing.
#:   * **Predictable from the state.** `day / total_days` is already a feature, so
#:     the agent can see this turn's charge. A streak or cumulative-count curve would
#:     vary it on something the state does not carry -- a new feature, a changed
#:     `MARKET_STATE_DIM`, and every checkpoint invalidated.
#:   * Strictly below `REJECTED_ORDER_PENALTY` at t=0, so a policy that cannot afford
#:     anything is not pushed into emitting junk to escape a charge.
IDLE_PENALTY = 0.0002

#: Charged on a turn that emitted an order the simulator then refused.
#:
#: A refused order moves no coins, so the objective scores it identically to a NOOP.
#: Without this the `IDLE_PENALTY` above would make emitting junk strictly better
#: than sitting still, which is the wrong way round. It expresses no view on *which*
#: trade is good: an impossible order falls below NOOP, a real purchase is judged by
#: the coins it moved, and the choice among real purchases is untouched.
REJECTED_ORDER_PENALTY = 0.001

#: Coins of committed cost per unit held, by `lots.STOCK_ITEMS` index.
#:
#: A harvested unit carries `seed / max_yield`, so a maximal harvest conserves the
#: basis and a poor one writes the difference off on the turn it lands.
#:
#: **Zero for EGG, MILK, WOOL and FERTILIZER.** The conserving version of the crop
#: rule would put `cost / remaining_events` on each unit and decrement the animal's
#: own basis as it produces. It is not usable: it needs the count of *placed*
#: animals, and an animal in the shed is not a feature (`market_state` covers
#: `PRODUCTS` only), so the basis would be a function of something the network cannot
#: see -- and fitting `Q - PHI` where `PHI` is unrepresentable is strictly worse than
#: fitting `Q`. The consequence is that an animal purchase is an uncushioned outlay
#: in the delta and its produce is pure credit when it sells: the true cash story,
#: just paid 200 turns apart.
_UNIT_BASIS: tuple[float, ...] = tuple(
    float(donor.CROPS[item]["seed"]) / max(1, int(donor.CROPS[item]["max_yield"]))
    if item in donor.CROPS
    else 0.0
    for item in lots.STOCK_ITEMS
)

#: Coins of committed cost per seed held or plant standing, by `CROP_ITEMS` index.
_SEED_BASIS: tuple[float, ...] = tuple(
    float(donor.CROPS[crop]["seed"]) for crop in lots.CROP_ITEMS
)


def _held(
    trace: Sequence,
    width: int,
    span: int,
    columns: int | None = None,
) -> np.ndarray:
    """One per-turn trace widened to `money_array`'s length, holding its last value.

    Padded rather than truncated when the trace is short: holding the last known
    state is the reading that cannot invent a purchase, and the trailing entries are
    written off at the terminal step anyway.

    `span` is passed in rather than derived per trace because the group is padded in
    lockstep, so a short member cannot slide the others' indexing and leave a turn's
    basis read off a different turn's holdings.
    """

    shape = (width,) if columns is None else (width, columns)
    padded = np.zeros(shape, dtype=np.float64)
    if span:
        padded[:span] = np.asarray(trace[:span], dtype=np.float64)
        padded[span:] = padded[span - 1]
    return padded


def held_basis(
    seeds: Sequence[Sequence[int]],
    standing: Sequence[Sequence[int]],
    stock: Sequence[Sequence[int]],
    width: int,
) -> np.ndarray:
    """Coins of committed cost the seat is holding, per entry of `money_array`.

    Counts a purchased seed at its price, a standing plant at the price of the seed
    that became it, and a harvested unit at that price over the crop's `max_yield`.

    **Cost basis, never market price.** Every price is a fixed constant out of
    `donor.CROPS` and nothing here reads `obs.market`, so there is no quote the policy
    can pump by marking its own shed up.

    Coins rather than reward units, and no terminal write-off: both of those belong
    to the caller, since the seat genuinely does hold this stock on the last turn even
    though the simulator pays nothing for it.
    """

    span = min(width, len(seeds), len(standing), len(stock))
    seed_basis = np.asarray(_SEED_BASIS, dtype=np.float64)
    crops = len(lots.CROP_ITEMS)
    return (
        _held(seeds, width, span, columns=crops) @ seed_basis
        + _held(standing, width, span, columns=crops) @ seed_basis
        + _held(stock, width, span, columns=len(lots.STOCK_ITEMS))
        @ np.asarray(_UNIT_BASIS, dtype=np.float64)
    )


def net_worth_delta(
    money_array: np.ndarray, basis: np.ndarray, gamma: float
) -> np.ndarray:
    """The objective: what the seat's net worth did on each turn, in reward units.

    `basis` is the held cost basis in coins, indexed like `money_array` -- one entry
    longer than the reward stream. Its last entry is written off here rather than by
    the caller: unsold stock, standing crops and unplanted seed all liquidate to
    nothing at the last step, where the simulator pays `farm["money"]` and only that.
    That write-off is also what makes the stream telescope to `final_money - stake`,
    so what a season is paid for is the money it actually ends with.

    The basis half is discounted by `gamma` where the cash half is not, which leaves a
    real `(1 - gamma) * basis` holding cost per turn -- 4e-7 for a melon seed at
    `gamma = 0.999`. Small, the correct sign, and the one thing opposing the cash
    delta's lean toward early liquidation: at 0.999 a sale deferred 240 turns is
    discounted to 0.79, so without it the objective quietly prefers selling now.

    At `gamma = 1.0` this is exactly `diff(money + basis) / MONEY_SCALE`.
    """

    held = np.asarray(basis, dtype=np.float64) / MONEY_SCALE
    held[-1] = 0.0
    cash = (money_array[1:] - money_array[:-1]) / MONEY_SCALE
    return cash + np.float32(gamma) * held[1:] - held[:-1]


def episode_rewards(
    money: list[float],
    final_money: float,
    gamma: float,
    emitted: Sequence[bool] | None = None,
    #: What each turn's order list was quoted at, from `PendingTurn.expected_delta`.
    #: Optional: without it the refusal test falls back to v1's zero-delta rule, which
    #: is what a demonstration read out of a replay can offer.
    expected: Sequence[float] | None = None,
    seeds: Sequence[Sequence[int]] | None = None,
    standing: Sequence[Sequence[int]] | None = None,
    stock: Sequence[Sequence[int]] | None = None,
    idle_penalty: float = IDLE_PENALTY,
    rejected_penalty: float = REJECTED_ORDER_PENALTY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one episode's money trace into `(shaped, raw, money)`.

    Shared by live rollouts and by demonstrations read out of stored replays, so a
    cloned episode and a self-collected one are the same kind of sample and cannot
    drift apart.

    `emitted[t]` says whether the policy put an order on the wire that turn; a turn
    that emitted one and moved no money had it refused.

    `seeds`/`standing`/`stock` are one all-or-nothing group. With any member absent
    the stream falls back to a *cash* delta, which is not a safe default -- it scores
    every purchase as a pure loss. A caller that can recover the traces must pass
    them; the fallback exists for a demonstration that cannot.

    `raw` -- the bare cash delta -- is returned as well as paid, so money diagnostics
    keep meaning what they meant when it was computed and discarded.
    """

    money_array = np.asarray([*money, final_money], dtype=np.float32)

    raw = (money_array[1:] - money_array[:-1]) / MONEY_SCALE
    if seeds is not None and standing is not None and stock is not None:
        shaped = net_worth_delta(
            money_array,
            held_basis(seeds, standing, stock, money_array.shape[0]),
            gamma,
        )
    else:
        shaped = raw.astype(np.float64)

    if emitted is not None:
        tried = np.zeros(raw.shape[0], dtype=bool)
        # `emitted` can be shorter than the reward stream -- the agent skips a
        # malformed observation -- and an unknown turn must not be charged for
        # silence it was never observed to have.
        known = np.zeros(raw.shape[0], dtype=bool)
        width = min(len(emitted), tried.shape[0])
        tried[:width] = np.asarray(emitted[:width], dtype=bool)
        known[:width] = True
        if rejected_penalty:
            # `order_refused` lives in `market_state` because the state encodes the
            # same fact, so a feature saying "refused" and a reward charging for it
            # cannot disagree.
            #
            # `expected` arrives in coins, straight off `PendingTurn.expected_delta`,
            # and `raw` is already divided by `MONEY_SCALE`. `order_refused`'s test is
            # `|delta| < FILL_FRACTION * |quoted|`, so handing it the two in different
            # units makes it true for every quote above a hundredth of a coin --
            # measured on `market-free-networth-v1` generation 19, 50.1% of turns were
            # charged as refused against a simulator refusal rate of 0.3%, and that
            # -0.36 an episode was 74% of the gap between the shaped return and the
            # cash it telescopes to. Scaled here rather than in `order_refused`, which
            # `market_state` calls in coins on both sides.
            quoted = None
            if expected is not None:
                quoted = np.zeros(raw.shape[0], dtype=np.float64)
                span = min(len(expected), quoted.shape[0])
                quoted[:span] = (
                    np.asarray(expected[:span], dtype=np.float64) / MONEY_SCALE
                )
            shaped[order_refused(tried, raw, quoted)] -= float(rejected_penalty)
        if idle_penalty:
            # `turns` is the recorded length, so a truncated trace decays over what
            # it actually played -- the only thing it can know.
            turns = raw.shape[0]
            falloff = 1.0 - np.arange(turns, dtype=np.float64) / max(1, turns)
            charge = float(idle_penalty) * falloff
            idle_turns = known & ~tried
            shaped[idle_turns] -= charge[idle_turns]

    return shaped.astype(np.float32), raw.astype(np.float32), money_array
