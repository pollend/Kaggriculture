"""Per-turn cash delta for the free-market learner.

The learner fits one stream: the cash banked or spent on *this turn*. There is no
inventory basis shaping, idle penalty, or order-refusal charge. Each turn stores
the change in its own cash position:

    reward[t] = (money[t + 1] - money[t]) / MONEY_SCALE

`money[t]` is the cash the recorder saw *before* turn `t` acted, so an outlay at
turn `t` is charged to turn `t` and a sale is credited on the turn it lands. The
stream telescopes: `reward.sum()` is the season's profit, which is what
`priority_stats` reports as `own_mean_return`.

Replay turns the stream into an `n_step`-turn discounted return and bootstraps
from the turn `n_step` ahead (`replay._nstep_tables`), so the Bellman backup
connects turns. `members-v2` ran the undiscounted reward-to-go
`(final_money - money[t])` as a Monte Carlo label with no cross-turn bootstrap:
the closing row's target did not depend on which action closed the turn, the
intra-turn max propagated one constant back, and every member finished the
season insolvent (`own_solvent_fraction` 0.00-0.03, benchmark money 334-974
against a 3,000 stake). `gamma` is accepted here for the call path and unused:
discounting is replay's, applied over the horizon, not baked into the label.

`raw_reward` is the cash stream alone; `reward` carries two shaping terms on top of
it, and the two are stored separately so the diagnostics keep reading cash.

**The animal potential.** A cow costs 400 on the turn it is bought and pays its
first milk eight days -- 192 turns -- later, eight n-step windows past the charge.
Under the bare cash delta the purchase is a `-0.016` with nothing inside the
horizon to answer it, and the value has to arrive through the bootstrap alone. The
ladder pack says that purchase is the whole game: winners put 8 cows and 6 sheep an
episode on the board and sell 379 MILK and 226 WOOL; the learner bought 2.6 cows
and sold 6 MILK. So the label carries the textbook potential-based term

    F[t] = gamma * phi[t + 1] - phi[t]
    phi[t] = sum over species of placed[t] * cost / MONEY_SCALE

with `phi` written off at the terminal step. *Placed* -- on a structure, from the
state's own animal columns (`market_state.PLACED_ANIMAL_BASIS`) -- not bought: an
animal in the shed with no housing under it is exactly the dead capital
`herd.py` exists to prevent, and it stays charged. The term moves the purchase's
cost from the turn it was bought to the turn the animal leaves the board (or the
season ends), which under potential shaping leaves the optimal policy where the
cash delta puts it and shifts every `Q` by `-phi[s]`.

**The stock mark.** The second potential, and the reason this fork exists. Under
the cash delta a harvest sitting in the shed is worth exactly nothing until the
turn it sells, so the only place "sell" is crisp in the label is the last `n_step`
turns, where replay bootstraps with discount zero and the target is realised cash
alone. Measured on v3d's `ruin-v2` / `open-v1` (60 seats each, learner against
pilkwang): shed value at quotes on day 27 of 12,925 / 10,548 against pilkwang's
9,808 / 8,974, a *median* 1,540 / 2,449 coins still unsold at the terminal against
pilkwang's 0, and late-season sales at 0.92-1.03x the opening quote where pilkwang
takes 1.14-1.26x. It hoards, dumps into the prices it has already depressed, and
still leaves stock on the table. So the label also carries

    phi_stock[t] = STOCK_MARK_WEIGHT * sum over products of held[t] * base / MONEY_SCALE

with `held` the shed plus carried count from the state's own columns
(`market_state.HELD_STOCK_BASIS`) and `base` the product's constant base price,
`donor.MARKET[item][0]`. What that does to each event, at weight 1.0:

  * a harvest landing in the shed is credited `+base` the turn it lands;
  * a sale at quote `q` pays `q - base`, so a dump below base is negative on the
    turn it happens and a premium sale is positive on the turn it happens;
  * unsold produce is written off at the terminal step at full base value;
  * holding costs `(1 - gamma) * phi` a turn -- ~4e-4 a turn on a 10k shed, ~0.12
    over 300 turns -- a deliberate anti-hoard drag.

`base` is a constant, so unlike the mark-to-quote v2 rejected there is no price to
pump: buying product to lift the quote moves the cash and the mark by the same
coins. Still potential-based, so the optimum is where the cash delta puts it.

**The labor charge.** Hands live one day and are paid once, at hire, on the
fibonacci ladder; they do nothing on their own -- the field planner hands each one
a job or a PASS. A hand the job list cannot use is a wage with no return anywhere
in the 24-turn window, and the cash delta alone cannot tell it from a hand that
harvests. Through mark-v5, and again in mark-v9/v10, a demand-plus-two mask did
that job; mark-v6 and mark-v11 expose the full ladder and leave it to this gradient.
mark-v7/v8
accidentally put that gradient on the decision *after* HIRE and scaled it until its
mean was 4.886 reward units against -0.050 of cash. The corrected term looks from
the opening state of turn `t` to the opening state of `t + 1`, and charges the
excess wage on reward row `t`, where the causing HIRE decision lives. Read off the
state's own labor columns so every stored pack relabels. **Not potential-based**:
the nightly reset refunds nothing.

Neither potential telescopes: `reward.sum()` is the season's profit less
`phi[0] + (1 - gamma) * sum(phi[1:-1])`. `phi[0]` is zero (nothing is placed or
held on turn 0); the drag is ~0.005 an episode for a 14-head herd and 0.24 on a
ladder season that held 12.7k of produce at its peak (measured on
`ladder-top25-296` part 0, episode 0, against a 4.8 season). `replay.priority_stats` reports that shaped sum as `own_mean_return`
and the cash alone -- `raw_reward`, still exactly the season's profit over
`MONEY_SCALE` -- as `own_mean_cash_return`; grade money on the second.
"""

from __future__ import annotations

import numpy as np

from agents.dqn_market_free_v3e import donor
from agents.dqn_market_free_v3e.market_state import (
    HAND_COUNT_INDEX,
    HAND_COUNT_SCALE,
    HELD_STOCK_BASIS,
    HIRE_DEMAND_INDEX,
    PLACED_ANIMAL_BASIS,
)

#: Discount the potential is paid under when a caller passes none. Replay's own
#: `gamma`; the identity above is only exact when the two agree.
DEFAULT_GAMMA = 0.999

# A 10k season sums to 0.28, a 50k season to 1.9, and losing the whole stake to
# -0.12. Was 250_000, which put a 5k-to-10k season spread inside a 0.02-wide band
# -- SmoothL1's quadratic zone is 1.0 wide, so the gradient on that band was the
# residual itself, and `mean_grad_norm` read 0.01-0.03 for the whole run. Ladder
# seasons (50k-155k) land in the linear zone by design: they are outliers to this
# policy and Huber treats them as such.
MONEY_SCALE = 25_000.0

#: The stake, and the line below which a season did worse than not playing.
#: Matches `evaluation.STARTING_MONEY`; restated rather than imported because this
#: module is what a pinned pack relabels through and must stay import-light.
RUIN_THRESHOLD = 3_000.0

#: Reward charged at T for destroying the whole stake, prorated on the shortfall.
#:
#: Ruin was free before this. The cash stream telescopes, so losing everything from
#: the 3,000 stake sums to -0.12 against +0.28 for a 10k season, +1.9 for a 50k one
#: and +6.1 for the best pinned ladder season -- spread over 720 turns that is
#: ~1.7e-4 a turn, inside SmoothL1's quadratic zone and under the bootstrap noise.
#: The simulator agrees: money floors at zero and part-fills
#: (`kaggriculture.py:650-704`), nothing terminates early on bankruptcy (`:947-950`),
#: and there are no passive drains at all. Downside capped at the stake, upside
#: uncapped, so a risk-neutral maximiser *should* gamble, and the arms on record did.
#:
#: 1.0 puts total ruin at -1.12 all-in: worse than three 10k seasons are worth, and
#: still under a single 50k one, so a genuinely good bet stays worth taking. This is
#: the one number here to tune if the policy turns timid -- watch `trading_rate`
#: against the 0.30-0.47 band, because suppressing trade is how every arm that
#: scored the bare stake failed.
RUIN_PENALTY = 1.0

#: Weight on the held-produce potential (the stock mark, module docstring).
#:
#: 1.0 marks a unit at exactly its base price, which makes a sale at base reward
#: neutral and a sale at `q` worth `q - base`: the price-quality signal the learner
#: was missing. Potential-based, so any value leaves the optimum alone; what the
#: weight sizes is how loudly the label speaks about *when* to sell against the
#: bootstrap that used to be the only voice.
#:
#: The knob to turn, and which way: `trading_rate` collapsing out of the 0.30-0.47
#: band, or `sale_price_over_base` climbing while `unsold_terminal_value` climbs with
#: it (refusing every sale under base and eating the write-off instead), means this
#: is too loud -- try 0.5. A learner that gambles the stake is `RUIN_PENALTY`'s
#: problem, not this one's. 0.0 restores v3d's label exactly.
STOCK_MARK_WEIGHT = 1.0

#: Multiplier on the terminal write-off of unsold produce. 1.0 is the potential's
#: own write-off and nothing more; the excess over 1.0 is charged on `target` only,
#: on the last step, beside `ruin_penalty`.
#:
#: **Not potential-based, deliberately**, for the same reason `RUIN_PENALTY` is
#: not: under the potential alone a hoarded unit is reward-neutral (credited `+base`
#: on landing, written off `-base` at T), so the label says nothing against holding
#: produce to the terminal beyond the ~1e-3 discount drag. mark-v1 at weight 1.0
#: held 2.0-4.5k of produce to the terminal on every rejected update from seq 500 on
#: against pilkwang's 0, while `sale_price_over_base` climbed 1.17 -> 1.39: it
#: learned the premium and kept the hoard, which is the failure the module
#: docstring's "too loud" clause names. At 1.5 an unsold unit costs half its base
#: over the neutral write-off, so a dump at 0.5x base breaks even against holding;
#: pilkwang's late dumps sit at 1.14-1.26x. The optimum moves, and it is meant to.
UNSOLD_WRITE_OFF_WEIGHT = 1.5

#: Small auxiliary charge on a HIRE above `labor.hire_demand`. At 0.1, relabelling
#: mark-v8's first 640 episodes turns the old 4.886 / 122k-equivalent mean charge
#: into about 0.122 / 3.05k: large enough to distinguish a wasted full crew, but no
#: longer two orders of magnitude louder than the cash objective.
LABOR_WASTE_WEIGHT = 0.1

#: Wage of a crew of n from a standing start: `sum(fib(i) for i in range(n))`, the
#: simulator's own ladder (`kaggriculture._do_hire`, `farmHandCostMult` 1). Long
#: enough for any crew the wire can carry in a day.
_FIB_PREFIX = np.concatenate(
    ([0.0], np.cumsum([float(donor._fib(index)) for index in range(64)]))
)


def ruin_penalty(final_money: float) -> float:
    """Terminal charge for finishing under the stake, in reward units.

    **Not potential-based, deliberately.** The animal term above is shaped in the
    Ng sense precisely so it leaves the optimal policy alone -- it is a
    credit-assignment aid. This one has to move the optimum, because the optimum
    under the bare cash stream is to gamble the stake. A policy-invariant ruin
    penalty would be a contradiction in terms.

    Linear in the shortfall so it is a gradient rather than a cliff: a season that
    ends at 2,999 is charged almost nothing and one that ends at 0 pays it all. A
    step at the stake would make the difference between 2,999 and 3,001 worth more
    than the difference between 3,000 and 28,000.

    Charged on `target` only. `raw` stays exactly the cash delta, so
    `own_mean_return` remains season profit over `MONEY_SCALE` and every money read
    that grades against it is unaffected.
    """

    shortfall = max(0.0, RUIN_THRESHOLD - float(final_money))
    return -RUIN_PENALTY * (shortfall / RUIN_THRESHOLD)


def _basis(stock_weight: float) -> np.ndarray:
    """The one vector both potentials read through: animals at cost, produce at base."""

    return PLACED_ANIMAL_BASIS + float(stock_weight) * HELD_STOCK_BASIS


def potential(
    turn_state: np.ndarray | None, stock_weight: float = STOCK_MARK_WEIGHT
) -> np.ndarray | None:
    """`phi` per turn, in reward units, from the stored turn states.

    Read off `turn_state` rather than recorded separately so a demonstration pack
    written before either term existed carries both on relabel: the state columns
    are the one per-turn record every artifact already holds. None when there is no
    state to read, which is the bare cash delta.
    """

    if turn_state is None:
        return None
    states = np.asarray(turn_state, dtype=np.float32)
    if states.ndim != 2 or states.shape[0] == 0:
        return None
    return (states @ _basis(stock_weight).astype(np.float32)) / np.float32(MONEY_SCALE)


def animal_basis(turn_state: np.ndarray | None) -> np.ndarray | None:
    """The animal potential alone -- `potential` with the stock mark off."""

    return potential(turn_state, stock_weight=0.0)


def stock_mark(turn_state: np.ndarray | None) -> np.ndarray | None:
    """The stock mark alone, at `STOCK_MARK_WEIGHT`, for tests and diagnostics."""

    if turn_state is None:
        return None
    states = np.asarray(turn_state, dtype=np.float32)
    if states.ndim != 2 or states.shape[0] == 0:
        return None
    basis = (STOCK_MARK_WEIGHT * HELD_STOCK_BASIS).astype(np.float32)
    return (states @ basis) / np.float32(MONEY_SCALE)


def labor_waste(turn_state: np.ndarray | None) -> np.ndarray | None:
    """Per-turn over-hire charge in reward units, at `LABOR_WASTE_WEIGHT`.

    Reward row `t` belongs to the action taken from state `t`; its resulting crew is
    visible in state `t + 1`. Both sides are valued at `demand[t]`, so a demand
    change on the following observation cannot manufacture a wage. The final row
    has no following state and receives no inferred auxiliary charge; its real cash
    wage is still present in `raw_reward`.
    """

    if turn_state is None:
        return None
    states = np.asarray(turn_state, dtype=np.float32)
    if states.ndim != 2 or states.shape[0] == 0:
        return None
    hands = np.rint(states[:, HAND_COUNT_INDEX] * HAND_COUNT_SCALE).astype(np.intp)
    demand = np.rint(states[:, HIRE_DEMAND_INDEX] * HAND_COUNT_SCALE).astype(np.intp)
    limit = _FIB_PREFIX.shape[0] - 1
    hands = np.clip(hands, 0, limit)
    demand = np.clip(demand, 0, limit)

    def excess_wage(crew: np.ndarray, needed: np.ndarray) -> np.ndarray:
        return _FIB_PREFIX[crew] - _FIB_PREFIX[np.minimum(crew, needed)]

    charge = np.zeros(states.shape[0], dtype=np.float64)
    if states.shape[0] > 1:
        before = excess_wage(hands[:-1], demand[:-1])
        after = excess_wage(hands[1:], demand[:-1])
        charge[:-1] = np.maximum(0.0, after - before)
    return (LABOR_WASTE_WEIGHT * charge / MONEY_SCALE).astype(np.float32)


def episode_rewards(
    money: list[float],
    final_money: float,
    gamma: float | None = None,
    turn_state: np.ndarray | None = None,
    **_ignored,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn an episode money trace into `(target, raw, money_array)`.

    `raw` is the per-turn cash delta; `target` is `raw` plus the animal potential,
    the stock mark and the labor charge when `turn_state` is given, and `raw` again when it is not. Keywords the older
    shaping signature took are accepted and ignored so live rollouts, cloned
    replays and stored packs keep one call path.
    """

    del _ignored

    money_array = np.asarray([*money, final_money], dtype=np.float32)
    raw = (
        (money_array[1:].astype(np.float64) - money_array[:-1].astype(np.float64))
        / MONEY_SCALE
    ).astype(np.float32)
    # Terminal, so it lands on the last step and nowhere else. `raw` is left alone.
    charge = ruin_penalty(final_money)
    phi = potential(turn_state)
    if phi is None:
        target = raw.copy()
        if charge:
            target[-1] = np.float32(target[-1] + charge)
        return target, raw, money_array
    turns = raw.shape[0]
    if phi.shape[0] != turns:
        raise ValueError(
            f"turn_state holds {phi.shape[0]} rows for {turns} reward steps"
        )
    discount = DEFAULT_GAMMA if gamma is None else float(gamma)
    # `phi[T]` is zero: the basis is written off where the simulator pays for
    # `farm["money"]` and nothing else.
    padded = np.concatenate([phi.astype(np.float64), [0.0]])
    shaping = discount * padded[1:] - padded[:-1]
    shaped = raw.astype(np.float64) + shaping
    waste = labor_waste(turn_state)
    if waste is not None:
        shaped -= waste.astype(np.float64)
    # The excess write-off on whatever is still held at the last decision. `target`
    # only, like the ruin charge; `raw` stays the cash delta.
    excess = UNSOLD_WRITE_OFF_WEIGHT - 1.0
    if excess:
        held = stock_mark(turn_state)
        if held is not None:
            shaped[-1] -= excess * float(held[-1])
    if charge:
        shaped[-1] += charge
    target = shaped.astype(np.float32)
    return target, raw, money_array


def relabel(record, gamma: float | None = None):
    """Recompute `record.reward` / `record.raw_reward` from its stored money trace.

    For records read off disk -- demonstration packs, old rollout artifacts --
    whose `reward` was written by an earlier reward function. The money trace is
    the one array every record carries that the reward is a pure function of, so
    relabelling here is what makes an edit to this file apply to a pinned pack
    without a rebuild. `ladder-001` was built under the shaped stream at
    `MONEY_SCALE = 20_000` and sat in a quarter of every TD batch for a whole
    session before anyone read a shard.
    """

    money = np.asarray(record.money, dtype=np.float32)
    target, raw, _ = episode_rewards(
        list(money[:-1]),
        float(money[-1]),
        gamma,
        turn_state=getattr(record, "turn_state", None),
    )
    record.reward = target
    record.raw_reward = raw
    return record
