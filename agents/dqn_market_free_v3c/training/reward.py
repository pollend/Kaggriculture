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
the terminal step, where the simulator pays for `farm["money"]` and nothing else.

`gamma * basis[t+1] - basis[t]` with a terminal potential of zero is **potential-based
shaping** in the textbook form, so it leaves the optimal policy for the discounted
objective exactly where the bare cash delta puts it, and shifts every `Q` by
`-basis[s] / MONEY_SCALE`. That is the whole argument for the basis half: it moves
credit from the turn a lot fails to the turn it was bought without changing what is
worth doing.

What it does not do is telescope. Earlier revisions of this file claimed

    sum(shaped) == (final_money - stake) / MONEY_SCALE      # false

and the exact identity is

    sum(shaped) == (final_money - stake) / MONEY_SCALE
                   - basis[0] / MONEY_SCALE
                   - (1 - gamma) * sum(basis[1:-1]) / MONEY_SCALE

Both correction terms are shaping, not objective -- they cancel out of the discounted
control problem and out of `Q` differences -- but they do not cancel out of an
*undiscounted* sum, which is what `own_mean_return` reports. Measured on
`market-free-scale-v1` they are worth 0.0033-0.0058 an episode against a money term
of -0.0032 to +0.0171, so a policy holding inventory reads worse there than a policy
that holds none, at equal money. See `net_worth_delta` for the measurement and for
what to read instead.

One per-turn charge sits alongside it, and only because the objective is silent
about the difference it draws:

    accepted order    priced by the coins it moved
    refused order     priced by the coins it moved, which is none
    NOOP             -IDLE_PENALTY

It is not a view about which trade is good.

**`REJECTED_ORDER_PENALTY` is gone.** It existed for one reason -- a refused order
and a NOOP both move zero coins, so without it `IDLE_PENALTY` made emitting junk
score above sitting still -- and on `market-free-scale-v1` it was buying nothing at
all. The simulator refuses 0.37% of emitting turns: `own_order_diagnostics` on
`scale_00` batch 20 reads 138,470 emitting decisions and 137,619 effective, of
230,080. `feasible_actions` is why -- it is a superset of what the game accepts, but
a close one. 0.37% of turns is not a hole worth a constant.

**An earlier revision of this block deleted the charge on a different ground -- that
it fired on ~3.4% of turns, nine times the refusal rate -- and that ground does not
survive.** The claim was arithmetic on two aggregates: penalty mass of 0.021 to 0.028
against an idle mass "of about 0.011 at the measured 38% idle rate". The idle input
was wrong. `own_order_diagnostics.emitted_rate` was read as the emitting fraction of
*turns*; recomputed per turn off the stored `order_chosen` / `order_step`,
`market-free-scale-v1` emits on 0.484 to 0.592 of turns under its declared constants,
so idle runs 41-52% and the idle charge is ~1.5x the mass that subtraction allowed
it. What is left over is 26-30% of the drain, not 45%, and that ceiling puts the
charge at **at most** 2.3-2.4% of turns rather than 3.4%.

**What settles it is an accident.** The constants in this file were edited while
`scale-v1` was collecting, and its workers picked the edit up between g0002 and
g0003: the stored traces read `IDLE_PENALTY` 8e-5 with this charge live through
g0002, and 8e-6 with it deleted from g0003 on. That is an A/B nobody designed. The
leftover residual, the quantity the 3.4% was inferred from, **does not move across
it** -- 0.0067 to 0.0065 an episode on `scale_00`, 0.0069 to 0.0042 on `scale_05`,
with the charge present on one side and absent on the other. A term that survives its
own deletion was never that term. It is the `(1 - gamma)` basis holding drag in
`net_worth_delta`, which had gone unsized.

So the charge goes because it is measurably ~0, and the residual it was blamed for
belongs to the shaping identity above rather than to any constant this file names --
which is why no amount of re-tuning the charges was ever going to remove it. The
mechanism
`order_space.sell_rung` names -- an over-sized sell rung inflating
`PendingTurn.expected_delta`, so `order_refused` tests the *whole turn* against
`FILL_FRACTION` of an inflated quote -- is a claim about `order_space` that this
measurement neither confirms nor refutes. What it does say is that whatever that path
was doing, it was not costing the objective anything. The 10x `IDLE_PENALTY` cut
below is the only change of size in this revision, and should be read as such.

The refusal *state* block stays (`market_state.REFUSAL_BLOCK_DIM`). Nothing about it
depended on the charge -- it is three integrator columns a GRU cannot hold across 720
ticks -- and deleting it would move `MARKET_STATE_DIM` and invalidate every
checkpoint.

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
zero costs `-3,000 / MONEY_SCALE = -0.012` where the old `tanh` charged -0.42:
downside is bounded by the stake while upside is unbounded, and with the insolvency
charge gone nothing else opposes a gamble. That is what maximising capital linearly
means, and `market-free-scale-v1` shows the shape of it -- 3,246 coins of animals
bought an episode against a 3,000 stake, 98% of them left to rot.

`IDLE_PENALTY` is now the only term besides the delta, and the only thing opposing
NOOP: an idle turn scores exactly 0 where a losing trade scores negative. It is the
one constant to re-read if `own_orders_per_decision` collapses -- but re-read it
against the money term of the policy *in front of you*, not against a ladder season.
Getting that comparison wrong is what made it the objective. Watch it beside the
money either way: a silent slot converges on the stake and raises `lr_score`
mechanically.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from agents.dqn_market_free_v3c import donor
from agents.dqn_market_free_v3c import lots

#: Reward units per coin, sized so a whole season's return lands inside
#: `SmoothL1`'s quadratic zone -- not merely near its edge.
#:
#: **250,000, up from 100,000.** Both moves are the same argument: past `SmoothL1`'s
#: beta of 1.0 the loss is linear, and a 5-unit and a 50-unit TD error give the same
#: gradient. What changed is the number the bound is taken against. 100,000 was
#: fitted to one ladder season, `99,038 / 100,000 = 0.99`, which is not a bound at
#: all -- it is the median sitting exactly on the limit. Over the 321 stored
#: `ladder-001` demonstration seasons the final-money distribution is
#:
#:     median   85,867      -> 0.86 at 100,000, 0.34 here
#:     p99     149,431      -> 1.49 at 100,000, 0.60 here
#:     max     155,637      -> 1.56 at 100,000, 0.62 here
#:
#: so the top decile of *demonstrated* play was already in the linear regime, and
#: the target this run is actually chasing -- pilkwang's 131,000 against us -- was
#: outside it before the policy ever got there. At 250,000 the whole demonstrated
#: distribution fits with room for a policy that beats both.
#:
#: The largest single wallet move bounds the per-step reward the same way: 14,513
#: coins over those demonstrations, 27,670 over 1,920 stored `networth_*` rollout
#: episodes, so 0.06 and 0.11 here against 0.15 and 0.28 before.
#:
#: Scaled rather than clipped. Per-step clipping to [-1, 1] is the usual DQN answer
#: and is wrong here: the magnitude of a fill *is* the signal.
#:
#: **`IDLE_PENALTY` has to move with it.** The cash half and the basis half of the
#: delta divide by this constant and stay in one unit, but `IDLE_PENALTY` is an
#: absolute reward unit and does not. Left alone it would get 2.5x heavier against
#: money -- on a policy finishing on 2,185 coins the anti-idle term already outweighed
#: the profit signal, and that would be the wrong direction to push it. It was divided
#: by the same 2.5 to hold its coin-equivalent.
#:
#: That co-scaling was correct and insufficient: holding a ratio fixed cannot fix a
#: ratio that was already wrong, and it was. See `IDLE_PENALTY`, which has since been
#: cut a further 10x on its own terms rather than on this constant's.
MONEY_SCALE = 250_000.0

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
#:   * Decays linearly to zero at the deadline, so a fully idle season pays ~0.0288,
#:     7% of what a ladder-grade season pays. That ratio is what the constant is
#:     sized on and it is scale-invariant, so the 2.5x cut alongside `MONEY_SCALE`
#:     holds it exactly: 0.072/0.99 and 0.0288/0.396 are the same 7.3%. An idle turn
#:     on day 1 forfeits a season of compounding; one on day 29 forfeits almost
#:     nothing.
#:   * **Predictable from the state.** `day / total_days` is already a feature, so
#:     the agent can see this turn's charge. A streak or cumulative-count curve would
#:     vary it on something the state does not carry -- a new feature, a changed
#:     `MARKET_STATE_DIM`, and every checkpoint invalidated.
#: **0.000008, down 10x from 0.00008.** The previous move -- 0.0002 to 0.00008
#: alongside `MONEY_SCALE`'s 2.5x -- held every ratio fixed by construction and so
#: changed nothing for the policy. This one changes the ratio, because the ratio was
#: the bug.
#:
#: The constant was sized as 7% of a *ladder-grade* season, 99,038 coins. The policy
#: is not there and has never been there. Across all 30 `market-free-scale-v1`
#: updates `own_mean_return` -- what the network actually fits -- is **negative in
#: every one**, while the money term it telescopes to spans -0.007 to +0.018.
#:
#: Decomposed per episode over every stored `episode_traces` npz, in that run's own
#: reward units. `drain` is `sum(raw_reward) - sum(reward)`, everything the shaping
#: takes off the money term; `idle` recomputes this charge from `order_chosen` /
#: `order_step` as `IDLE_PENALTY * falloff` on turns carrying no non-NOOP row.
#: `scale-v1` is split at g0003, where its workers picked up an edit to this file
#: mid-run and switched constants:
#:
#:     slot / regime                 eps   final$   money   shaped   drain    idle  share  resid
#:     networth_00  2e-4, bug      3,360    9,688  +0.0669  -0.4413  0.5082  0.0371   7.3% 0.4711
#:     networth_05                 3,360    4,690  +0.0169  -0.4509  0.4678  0.0433   9.3% 0.4245
#:     quote_00     2e-4, fixed      512    5,496  +0.0250  -0.0322  0.0572  0.0411  71.9% 0.0161
#:     quote_05                       496    7,685  +0.0469  -0.0392  0.0861  0.0458  53.2% 0.0403
#:     scale_00     8e-5, g0-g2      480    4,317  +0.0053  -0.0174  0.0227  0.0160  70.5% 0.0067
#:     scale_05                       480    7,262  +0.0171  -0.0091  0.0261  0.0193  73.8% 0.0069
#:     scale_00     8e-6, g3+        912    2,206  -0.0032  -0.0116  0.0084  0.0019  22.5% 0.0065
#:     scale_05                       912    3,552  +0.0022  -0.0045  0.0067  0.0025  37.5% 0.0042
#:
#: `idle` is an **upper** bound and `resid` therefore a lower one: the reconstruction
#: reads `chosen != 0` and misses the `buy_land` turns `collect.finish` also marks
#: emitted, so it scores some trading turns as idle.
#:
#: Read down `shaped`: negative on every slot of every run and every regime, while
#: `money` is positive on all but one. Under `scale-v1`'s declared constants this
#: charge is 70-74% of everything subtracted from what the season made. At 7% of a
#: ladder season that size is honest; at 240% of the season this policy plays it is
#: the gradient.
#:
#: The last two rows are the accidental A/B described in the module docstring, and
#: they are why `REJECTED_ORDER_PENALTY` is deleted rather than re-scaled: `resid`
#: barely moves when that charge is removed (0.0067 -> 0.0065, 0.0069 -> 0.0042), so
#: it was never carrying the residual. They are also a warning about the rest of that
#: regime -- money, emit rate and `shaped` all fall with it -- but it landed on
#: checkpoints, buffers and a self-play pool fitted to the other objective, so it is
#: evidence, not a run.
#:
#: `networth-v1` is in the table as the contrast, not as evidence for this constant.
#: Its drain is 7-28x its money term while idle is under a tenth of it -- that is
#: `REJECTED_ORDER_PENALTY` under the unit bug `quote-v1` fixed, charged on ~61% of
#: turns. It is why a run can look like this one without the idle charge being at
#: fault, and why the two have to be decomposed rather than inferred from a total.
#:
#: Two sanity checks it also fails at the old value. A fully idle season paid 0.0288
#: while **losing the entire 3,000 stake costs 0.012** -- bankruptcy was 2.4x cheaper
#: than sitting still, which is the wrong way round for a term whose whole job is to
#: be the small one. And `own_solvent_fraction` measured 0.013 to 0.29: the wallet is
#: empty on 71% to 99% of turns, so most idling was not a choice being priced.
#:
#: At 0.000008 a fully idle season pays 0.00288, 0.7% of a ladder season and 16% of
#: the best money term measured here. Still strictly the smaller term, which is the
#: one ordering that has to hold; still 2 coins a turn at t=0, which is a price and
#: not a rounding error.
#:
#: **This is the last constant in the stream that can distort the objective.** Unlike
#: the basis half it is not potential-based: it is a flat charge on a state the policy
#: chooses, so it moves the argmax. That is what made it worth cutting and it is also
#: what makes it the only term here worth arguing about again.
#:
#: The g3+ rows above are this constant at 8e-6 with the refusal charge gone. The cut
#: does what it was meant to -- idle falls to 0.0019-0.0025 -- and `own_mean_return`
#: stays negative anyway, at 0.0033-0.0058 of residual against a money term of -0.0032
#: to +0.0022. That residual is the shaping identity in `net_worth_delta`, which
#: cancels out of the policy and not out of an undiscounted sum, so **the sign of
#: `own_mean_return` was never going to flip and is the wrong thing to have promised.**
#: A slot holding 2,000 coins of stock reads 0.0058 worse than an identical slot
#: holding none. Read `raw_reward`'s sum, or `own_mean_final_money`, which is the same
#: quantity in coins.
IDLE_PENALTY = 0.000008

#: Coins of committed cost per unit held, by `lots.STOCK_ITEMS` index.
#:
#: A harvested unit carries `seed / max_yield`, so a maximal harvest conserves the
#: basis and a poor one writes the difference off on the turn it lands.
#:
#: **Zero for EGG, MILK, WOOL and FERTILIZER.** Animal basis lives on the animal
#: itself, not on the product. That keeps the purchase neutral when it is made,
#: lets produce sales pay cash when they happen, and writes off any unrecovered
#: herd cost at the terminal step where the simulator pays only banked money.
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

#: Coins of committed cost per owned animal, by `lots.ANIMAL_ITEMS` index.
_ANIMAL_BASIS: tuple[float, ...] = tuple(
    float(donor.ANIMALS[animal]["cost"]) for animal in lots.ANIMAL_ITEMS
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
    owned: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Coins of committed cost the seat is holding, per entry of `money_array`.

    Counts a purchased seed at its price, a standing plant at the price of the seed
    that became it, a harvested unit at that price over the crop's `max_yield`, and
    an owned animal at its fixed purchase cost.

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
    basis = (
        _held(seeds, width, span, columns=crops) @ seed_basis
        + _held(standing, width, span, columns=crops) @ seed_basis
        + _held(stock, width, span, columns=len(lots.STOCK_ITEMS))
        @ np.asarray(_UNIT_BASIS, dtype=np.float64)
    )
    if owned is not None:
        animal_span = min(width, len(owned))
        basis += _held(owned, width, animal_span, columns=len(lots.ANIMAL_ITEMS)) @ np.asarray(
            _ANIMAL_BASIS, dtype=np.float64
        )
    return basis


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

    The basis half is discounted by `gamma` where the cash half is not. That is not a
    holding-cost design decision -- it is what makes the term potential-based, and
    potential-based shaping is the only kind that provably leaves the discounted
    optimal policy alone. The price of the guarantee is that the undiscounted sum
    picks up `basis[0] + (1 - gamma) * sum(basis[1:-1])`.

    **Small per seed is not small per season.** 4e-7 for a melon seed at
    `gamma = 0.999`, and 0.0033 to 0.0058 an episode once the whole portfolio is
    integrated over 719 turns. Recovered by inverting the stream --
    `basis[t] = gamma * basis[t+1] - (shaped[t] - raw[t] + charge[t])` backwards from
    `basis[T] = 0` -- over 960 `market-free-scale-v1` episodes a side, and the
    inversion returns a non-negative basis on 100% of turns, which a wrong charge
    model does not:

        slot      mean held   median    p95   season peak   basis[0]   (1-g) drag
        scale_00      1,999    1,780  3,844      turns 286-572    199      0.00575
        scale_05      1,132      872  3,267      turns 286-572    197      0.00325

    The drag is exactly proportional to held basis across the two slots (1.77x and
    1.77x), which is the signature of the identity rather than of anything the policy
    is being charged for.

    **So do not chase this with `gamma`.** The term is policy-invariant by
    construction; what it breaks is the *read*. `own_mean_return` is an undiscounted
    sum of a shaped stream, so it is biased low in proportion to inventory held and
    cannot be compared across policies that hold different amounts of stock -- and a
    pass condition phrased on its sign is asking a question the quantity cannot
    answer. `raw_reward` is stored beside it and telescopes to
    `(final_money - stake) / MONEY_SCALE` exactly. Read that.

    `basis[0]` is a second, smaller problem and a real one: it averages 199 coins but
    carries a standard deviation of 1,204, because a minority of episodes open with
    thousands of coins of stock already held -- `money[0]` is 3,000 on every episode,
    so this is the recorder's first turn not always being the season's first turn.
    That injects a per-episode nuisance of ~0.005 into the shaped return, the same
    size as the money term itself, unrelated to anything the policy did.

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
    seeds: Sequence[Sequence[int]] | None = None,
    standing: Sequence[Sequence[int]] | None = None,
    stock: Sequence[Sequence[int]] | None = None,
    owned: Sequence[Sequence[int]] | None = None,
    idle_penalty: float = IDLE_PENALTY,
    basis_required: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one episode's money trace into `(shaped, raw, money)`.

    Shared by live rollouts and by demonstrations read out of stored replays, so a
    cloned episode and a self-collected one are the same kind of sample and cannot
    drift apart.

    `emitted[t]` says whether the policy put an order on the wire that turn, and is
    read for one thing only: a turn that emitted nothing pays `IDLE_PENALTY`. A turn
    whose order the simulator refused is priced by the coins it moved, which is none
    -- there is no separate charge for it, and no `expected` quote to test against.
    See the module docstring for what that measured.

    `seeds`/`standing`/`stock` are the crop-basis group, and `owned` adds the animal
    basis. A caller that can recover all traces must pass them and set
    `basis_required`; otherwise the stream uses the traces it was given and falls
    back to cash only when the crop group is incomplete.

    `raw` -- the bare cash delta -- is returned as well as paid, so money diagnostics
    keep meaning what they meant when it was computed and discarded.
    """

    money_array = np.asarray([*money, final_money], dtype=np.float32)

    raw = (money_array[1:] - money_array[:-1]) / MONEY_SCALE
    crop_traces = (seeds, standing, stock)
    if basis_required and owned is None:
        raise ValueError("basis_required=True but reward traces are missing: owned")
    if all(trace is not None for trace in crop_traces):
        shaped = net_worth_delta(
            money_array,
            held_basis(seeds, standing, stock, money_array.shape[0], owned=owned),
            gamma,
        )
    elif basis_required:
        missing = [
            name
            for name, trace in zip(("seeds", "standing", "stock"), crop_traces)
            if trace is None
        ]
        raise ValueError(
            "basis_required=True but reward traces are missing: " + ", ".join(missing)
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
        if idle_penalty:
            # `turns` is the recorded length, so a truncated trace decays over what
            # it actually played -- the only thing it can know.
            turns = raw.shape[0]
            falloff = 1.0 - np.arange(turns, dtype=np.float64) / max(1, turns)
            charge = float(idle_penalty) * falloff
            idle_turns = known & ~tried
            shaped[idle_turns] -= charge[idle_turns]

    return shaped.astype(np.float32), raw.astype(np.float32), money_array
