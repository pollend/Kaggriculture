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

`raw_reward` is the same stream; it is returned separately because the older
shaping signature distinguished the training target from the cash delta and the
recorder still stores both.
"""

from __future__ import annotations

import numpy as np

# A 10k season sums to 0.28, a 50k season to 1.9, and losing the whole stake to
# -0.12. Was 250_000, which put a 5k-to-10k season spread inside a 0.02-wide band
# -- SmoothL1's quadratic zone is 1.0 wide, so the gradient on that band was the
# residual itself, and `mean_grad_norm` read 0.01-0.03 for the whole run. Ladder
# seasons (50k-155k) land in the linear zone by design: they are outliers to this
# policy and Huber treats them as such.
MONEY_SCALE = 25_000.0


def episode_rewards(
    money: list[float],
    final_money: float,
    gamma: float | None = None,
    **_ignored,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn an episode money trace into `(target, raw, money_array)`.

    `target` and `raw` are the same per-turn cash delta. `gamma` and any keyword
    the older shaping signature took are accepted and ignored so live rollouts,
    cloned replays and stored packs keep one call path.
    """

    del gamma, _ignored

    money_array = np.asarray([*money, final_money], dtype=np.float32)
    raw = (
        (money_array[1:].astype(np.float64) - money_array[:-1].astype(np.float64))
        / MONEY_SCALE
    ).astype(np.float32)
    return raw.copy(), raw, money_array


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
    target, raw, _ = episode_rewards(list(money[:-1]), float(money[-1]), gamma)
    record.reward = target
    record.raw_reward = raw
    return record
