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

`raw_reward` is the cash stream alone; `reward` carries one shaping term on top of
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

It does not telescope: `reward.sum()` is the season's profit less
`phi[0] + (1 - gamma) * sum(phi[1:-1])`. `phi[0]` is zero (no animal is placed on
turn 0) and the holding drag at `gamma = 0.999` on a 14-head herd is ~0.005 an
episode -- small against a 0.28 season, but it is there, so `own_mean_return`
reads `raw_reward`, which is still exactly the cash.
"""

from __future__ import annotations

import numpy as np

from agents.dqn_market_free_v3d.market_state import PLACED_ANIMAL_BASIS

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


def animal_basis(turn_state: np.ndarray | None) -> np.ndarray | None:
    """`phi` per turn, in reward units, from the stored turn states.

    Read off `turn_state` rather than recorded separately so a demonstration pack
    written before this term existed carries it on relabel: the state columns are
    the one per-turn record every artifact already holds. None when there is no
    state to read, which is the bare cash delta.
    """

    if turn_state is None:
        return None
    states = np.asarray(turn_state, dtype=np.float32)
    if states.ndim != 2 or states.shape[0] == 0:
        return None
    return (states @ PLACED_ANIMAL_BASIS.astype(np.float32)) / np.float32(MONEY_SCALE)


def episode_rewards(
    money: list[float],
    final_money: float,
    gamma: float | None = None,
    turn_state: np.ndarray | None = None,
    **_ignored,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn an episode money trace into `(target, raw, money_array)`.

    `raw` is the per-turn cash delta; `target` is `raw` plus the animal potential
    when `turn_state` is given, and `raw` again when it is not. Keywords the older
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
    phi = animal_basis(turn_state)
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
