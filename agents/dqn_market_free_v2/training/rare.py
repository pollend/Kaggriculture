"""A bounded visit-count widening of the noise on structurally rare actions.

NoisyNet moves the argmax only among actions already near-tied in Q, which is what
makes it safe and also why a feasible, cheap, never-sampled row stays never
sampled. The blocker is direction, not magnitude: head-only factorized noise
correlates all 199 perturbations through one 128-dim `f_in`, so flips concentrate
on a handful of rows regardless of state. So widen the *sigma row of the rare
action itself* -- `epsilon_out` is already the per-output half of the factored
pair, so one entry moves one action.

What this is not: an intrinsic reward, and not a score bonus. Nothing is added to
the Q values, the perturbation stays zero-mean, and the TD target never sees any
of it. `main:agent` -- the arena, the gates, the Kaggle entry point -- never draws
noise at all, so the inference path is untouched.

Shaped against this family's epsilon purchase pump: the multiplier is clipped,
bounded below at 1.0 so it can only add variance, and a widened row still has to
win a plain argmax. A row the policy has learned is ruinous stays unreachable.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from agents.dqn_market_free_v2.order_space import ACTION_COUNT, NOOP_INDEX

#: Filename stem written beside `rollout_policies/`, keyed the same way.
COUNTS_SUFFIX = ".npz"


def visit_counts(records: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """`(chosen, opportunities)` over `ACTION_COUNT`, from recorded episodes.

    The pair is what makes the novelty honest: selections alone cannot tell "the
    policy declines this" from "the mask forbids this", and the second is not
    explorable at any gain.
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

    Without the decay the widening freezes on the first rounds' distribution, when
    the policy was still random. `decay = 1.0` keeps every count forever.
    """

    chosen, opportunities = update
    if previous is None:
        return chosen.copy(), opportunities.copy()
    scale = float(decay)
    return previous[0] * scale + chosen, previous[1] * scale + opportunities


#: Which statistic `sigma_scale_vector` reads. See its docstring.
NOVELTY_KINDS = ("count", "rate")


def _count_novelty(chosen: np.ndarray, opportunities: np.ndarray) -> np.ndarray:
    """`sqrt(log(1 + T) / (1 + n_a))` -- the plain UCB shape, on raw selections.

    Kept so the sessions that played it stay reproducible and so the rate arm has a
    control. Its blind spot: the numerator is nearly constant on a long run, so the
    whole spread lives in `1 + n_a`. It can see "never tried" but not "tried, then
    abandoned" -- a crop with a five-figure historical count keeps its novelty near
    zero forever, however long ago the policy stopped choosing it.
    """

    total = float(np.sum(chosen))
    return np.sqrt(np.log1p(total) / (1.0 + chosen))


def _rate_novelty(chosen: np.ndarray, opportunities: np.ndarray) -> np.ndarray:
    """`log(reference_rate / rate_a)` -- how far under uniform attention a row sits.

    `rate_a` is the share of the turns a row was *feasible* on which it was chosen;
    `reference` is that ratio over the whole action space. Floored at zero, so a row
    at or above the reference earns nothing.

    Log-ratio rather than the UCB square root because the run length must not enter:
    `chosen` grows without bound, so the count arm's calibration drifts across a run
    while a ratio of two rates over the same window does not.

    `rate_a` floors at one hypothetical selection over the row's own opportunities
    rather than a fixed epsilon, so a never-chosen feasible row is treated as chosen
    exactly once -- the largest claim the data supports, and no constant to tune.
    """

    opportunity = np.maximum(opportunities, 1.0)
    total_opportunity = float(np.sum(opportunity))
    reference = float(np.sum(chosen)) / total_opportunity if total_opportunity > 0.0 else 0.0
    if reference <= 0.0:
        # Nothing has been selected yet, so nothing has been *avoided* yet either.
        # The first round has no evidence to widen on and says so.
        return np.zeros_like(chosen)
    rate = np.maximum(chosen / opportunity, 1.0 / opportunity)
    return np.maximum(np.log(reference / rate), 0.0)


_NOVELTY = {"count": _count_novelty, "rate": _rate_novelty}


def sigma_scale_vector(
    chosen: np.ndarray,
    opportunities: np.ndarray,
    gain: float,
    cap: float,
    novelty: str = "count",
) -> np.ndarray:
    """Per-action multiplier on the head's noise draw: `1 + gain * novelty`.

    `novelty` picks the statistic; the two are not calibrated alike, so
    `rare_sigma_gain` has to move with it. Either one anneals itself as a row
    accumulates selections, so there is no schedule to tune.

    Multiplies `epsilon_out`, the per-output half of the factored perturbation, so
    entry `a` widens action `a` and leaves the other 185 alone. Every entry is at
    least 1.0, so this can only add variance, never remove it.

    A *bonus added to the scores* was the obvious alternative and is worse: a
    deterministic tilt makes a rare row whose Q gap sits under the coefficient win
    every turn it is feasible, which is a policy change wearing an exploration
    costume and the shape that bankrupted the epsilon runs.

    Pinned to exactly 1.0 in two places:

      * `NOOP`, the collapse attractor -- extra mass on it is not worth the
        argument, and its novelty is near zero anyway.
      * A row never feasible in the window, which can never accumulate a visit to
        decay it and would otherwise arrive at full width the moment the mask opens.

    `BUY_LAND` used to be pinned here as a third case, because it was the rarest row
    there is and sat at the `rare_sigma_cap` on every measurement v30 and v31 took --
    the whole widening budget spent on a column that could not change a decision, and
    `selection_entropy` distorted while it did so. Since v44 it is not a column at
    all, so there is nothing left to pin.
    """

    if novelty not in _NOVELTY:
        raise ValueError(f"rare_sigma_novelty must be one of {list(NOVELTY_KINDS)}: {novelty}")
    scale = 1.0 + float(gain) * _NOVELTY[novelty](chosen, opportunities)
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

    A missing file is the ordinary first round. A wrong width means the action space
    changed under a resumed run, so the counts are meaningless per-row -- dropping
    them costs one round of novelty and cannot misattribute a widening.
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

    Says whether directed exploration is doing anything without naming an action in
    advance: a policy pouring every turn into three rows reads low here.
    """

    total = float(np.sum(chosen))
    if total <= 0.0:
        return 0.0
    p = chosen[chosen > 0] / total
    return float(-np.sum(p * np.log(p)))
