"""A bounded visit-count widening of the noise on structurally rare actions.

Value-proportional exploration cannot reach an action the policy has never had a
reason to try. NoisyNet moves the argmax only among actions already near-tied in
Q, which is exactly what makes it safe -- it is why the epsilon purchase pump
went at v13 -- and it is also why a row that is feasible, cheap in Q terms, and
simply never sampled stays never sampled.

Measured on `market-free-falloff-v18`'s live checkpoint, over 23,008 states drawn
from its own rollout batches:

    BUY_LAND median rank        33 of 104 feasible actions
    Q gap to the greedy pick    median 0.031, mean 0.113
    greedy picks BUY_LAND       0.0000% of states
    flip destinations at x0.5   HIRE 1/2/3/11 35%, BUY_SEED 20%

A 0.031 gap is well inside what the perturbation crosses routinely, so the row is
not out of reach on value. It is out of reach on *direction*: head-only
factorized noise correlates all 186 perturbations through one 128-dim `f_in`
(`agent.toml` says so where it chooses "head"), so the flips concentrate on a
handful of rows regardless of state. Across sessions v15-v18 -- 7,266 league
seats -- land was bought 89 times, never twice by the same seat, and never at all
by `evaluation_opponent`; `hot_v1` had it affordable on 88% of turns and bought
none.

So the fix is aimed at the correlation rather than at the magnitude: widen the
*sigma row of the rare action itself*. `epsilon_out` is already the per-output
half of the factored pair in both the torch head and its numpy mirror, so a
per-action multiplier there is a one-line change to each and needs no new noise
machinery at all.

What this module is *not*: an intrinsic reward, and not a score bonus either.
Nothing is added to the Q values anywhere -- the perturbation stays zero-mean,
the argmax stays a plain argmax, and the TD target never sees any of it, so the
learned values keep describing the environment rather than how often this trainer
has looked at something. `main:agent` -- the arena, the gates, the Kaggle entry
point -- never draws noise at all, so the inference path is untouched.

The family history here is the epsilon purchase pump, so the shape is chosen
against it: the multiplier is clipped, it is bounded below at 1.0 so it can only
ever add variance, and a widened row still has to win an argmax against a Q
function that has an opinion. A rare row the policy has learned is ruinous stays
unreachable, which is the property `choose_order` has defended since v13.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from agents.dqn_market_free.order_space import ACTION_COUNT, NOOP_INDEX

#: Filename stem written beside `rollout_policies/`, keyed the same way.
COUNTS_SUFFIX = ".npz"


def visit_counts(records: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """`(chosen, opportunities)` over `ACTION_COUNT`, from recorded episodes.

    `chosen[a]` counts the turns action `a` was selected; `opportunities[a]` the
    turns it was feasible. Both come straight off what a rollout already stores,
    so nothing new has to be recorded to keep this fed.

    The pair is what makes the novelty honest. Counting selections alone cannot
    tell "the policy declines this" from "the mask forbids this", and the second
    kind is not explorable at any gain -- see `sigma_scale_vector`.
    """

    chosen = np.zeros(ACTION_COUNT, dtype=np.float64)
    opportunities = np.zeros(ACTION_COUNT, dtype=np.float64)
    for record in records:
        indices = np.asarray(record.order_chosen, dtype=np.int64).reshape(-1)
        chosen += np.bincount(indices, minlength=ACTION_COUNT)[:ACTION_COUNT]
        opportunities += np.asarray(record.order_mask, dtype=np.float64).sum(axis=0)
    return chosen, opportunities


def fold(
    previous: tuple[np.ndarray, np.ndarray] | None,
    update: tuple[np.ndarray, np.ndarray],
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Age the running counts by `decay` and add this round's.

    Without the decay the widening freezes on the first rounds' distribution: an
    action sampled heavily while the policy was still random stays permanently
    narrow long after the policy stopped choosing it, and the counts stop
    describing anything current. `decay = 1.0` keeps every count forever and
    is the setting to name when that is actually wanted.
    """

    chosen, opportunities = update
    if previous is None:
        return chosen.copy(), opportunities.copy()
    scale = float(decay)
    return previous[0] * scale + chosen, previous[1] * scale + opportunities


def sigma_scale_vector(
    chosen: np.ndarray,
    opportunities: np.ndarray,
    gain: float,
    cap: float,
) -> np.ndarray:
    """Per-action multiplier on the head's noise draw: `1 + gain * novelty`.

    Returned as a multiplier on `epsilon_out`, which is already the per-output
    half of the factored perturbation, so entry `a` widens exactly action `a`'s
    row and leaves the other 185 alone. Every entry is at least 1.0: this only
    ever adds variance to a row, never removes it, so a run cannot quietly
    switch exploration off for the actions it happens to have sampled.

    A *bonus added to the scores* was the obvious alternative and is the worse
    one. It is a deterministic tilt, so a rare row whose Q gap sits under the
    coefficient does not get explored occasionally -- it wins every single turn
    it is feasible, on every state, which is a policy change wearing an
    exploration costume and the exact shape that bankrupted the epsilon runs.
    Widening sigma is zero-mean: the row comes up sometimes, in states where the
    perturbation happens to favour it, and the argmax stays a plain argmax over
    what the network said.

    `novelty = sqrt(log(1 + total) / (1 + n_a))` -- the UCB shape, so a row
    decays out of the widening as it accumulates visits and the whole thing anneals
    itself without a schedule to tune.

    Pinned to exactly 1.0 in two places, both of which would otherwise widen a
    row to no purpose:

    `NOOP` is the most-visited action by a wide margin, so its novelty is near
    zero anyway -- but it is also the collapse attractor this family keeps
    falling into, and extra probability mass on it is not worth the argument.

    A row that was never feasible in the window keeps maximum novelty forever,
    because the mask means it can never accumulate a visit to decay it. Harmless
    while the mask still refuses it, wrong the moment the mask opens: the row
    would arrive at full width rather than at the novelty it has actually earned.
    """

    total = float(np.sum(chosen))
    novelty = np.sqrt(np.log1p(total) / (1.0 + chosen))
    scale = 1.0 + float(gain) * novelty
    scale = np.clip(scale, 1.0, max(1.0, float(cap)))
    scale[NOOP_INDEX] = 1.0
    scale[opportunities <= 0.0] = 1.0
    return scale.astype(np.float32)


def save(path: Path | str, chosen: np.ndarray, opportunities: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, chosen=chosen, opportunities=opportunities)


def load(path: Path | str) -> tuple[np.ndarray, np.ndarray] | None:
    """The stored counts, or None when the file is absent or a stale width.

    A missing file is the ordinary first round rather than an error. A wrong
    width means the action space changed under a resumed run, and the counts are
    then meaningless per-row -- dropping them costs one round of novelty and is
    the only reading that cannot silently misattribute a widening.
    """

    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path) as data:
        chosen = np.asarray(data["chosen"], dtype=np.float64)
        opportunities = np.asarray(data["opportunities"], dtype=np.float64)
    if chosen.shape != (ACTION_COUNT,) or opportunities.shape != (ACTION_COUNT,):
        return None
    return chosen, opportunities


def selection_entropy(chosen: np.ndarray) -> float:
    """Shannon entropy of the selection distribution, in nats.

    The one number that says whether directed exploration is doing anything
    without naming an action in advance: a policy pouring every turn into three
    rows reads low here whatever its money curve does.
    """

    total = float(np.sum(chosen))
    if total <= 0.0:
        return 0.0
    p = chosen[chosen > 0] / total
    return float(-np.sum(p * np.log(p)))
