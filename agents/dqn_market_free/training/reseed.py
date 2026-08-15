"""One-shot weight-space perturbation applied when a learner slot is reseeded.

A learner that has decayed to `lr_graduate_cut` has stopped moving, and the slot
restarts from the best mature member. Copying those weights unchanged would
restart the slot *inside the basin the mature member is already sitting in*, at a
learning rate warming back up -- which is the single-lineage run this replaced,
with extra steps. The perturbation is what makes the restart a different search
rather than a continuation of the same one.

Weight-space dropout, not an `nn.Dropout` layer, and the distinction is
load-bearing here: `policy.py` (numpy, ships in the Kaggle bundle) and
`training/network.py` (torch) must produce bit-identical outputs for identical
weights, and the parity test in `tests/test_dqn_market_free.py` pins it. A layer
added to one side and not the other breaks that; a function that rewrites the
weights *before* either side loads them cannot. Nothing in this module runs at
inference and neither network module knows it exists.

Shrink-and-perturb (Ash & Adams) rather than plain reinitialisation, with the
shrink defaulted off. Reinitialising a fraction of the weights is the part that
recovers plasticity; multiplying the survivors is the part that costs
performance, and on this family the mature member's score is the whole reason it
was chosen as the source. `reseed_shrink = 1.0` keeps the survivors exactly as
they were and is the setting a run should start from.
"""

from __future__ import annotations

import numpy as np

from agents.dqn_market_free.policy import OrderQNetwork


def perturb(
    network: OrderQNetwork,
    rate: float,
    shrink: float = 1.0,
    seed: int = 0,
    sigma_reset: float | None = None,
) -> OrderQNetwork:
    """A copy of `network` with `rate` of its weights reinitialised.

    The mask is drawn per weight rather than per unit. Dropping whole rows would
    delete an action's entire readout, and at 186 outputs over a 128-wide trunk
    that is a much larger intervention than the same expected count of scattered
    entries -- a row of zeros has no gradient of its own to recover through
    except via the bias.

    Reinitialised entries are drawn from the layer's own initialisation
    distribution, `U(-1/sqrt(fan_in), +1/sqrt(fan_in))`, matching
    `NoisyLinear.reset_parameters` and torch's default `nn.Linear`. A fresh entry
    therefore sits at the scale the layer was born at rather than at the scale a
    trained layer happens to have drifted to, which is what makes it explorable
    again instead of immediately dominated.

    `sigma_reset` restores every noisy layer's sigma to that same bound scaled by
    `sigma0`. Sigma anneals toward zero over a learner's life -- the standing
    NoisyNet failure mode `agent.toml` records -- so a slot that inherited a
    collapsed sigma would restart with no exploration budget at all and could
    only ever rediscover what the mature member already knows. `None` leaves
    sigma exactly as the source carried it.

    `rate = 0.0` returns an unperturbed copy rather than the same object, so a
    caller can always treat the result as its own to write.
    """

    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"reseed_dropout must be in [0, 1]: {rate}")
    if shrink < 0.0:
        raise ValueError(f"reseed_shrink must not be negative: {shrink}")

    rng = np.random.default_rng(seed)
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    sigma: list[tuple[np.ndarray, np.ndarray] | None] = []
    for (weight, bias), pair in zip(network.layers, network.sigma):
        # Numpy stores this transposed relative to torch, so the input dimension
        # is axis 0. Reading it off the array rather than off a passed-in size
        # keeps this correct for whatever `HIDDEN` becomes.
        bound = 1.0 / np.sqrt(weight.shape[0])
        layers.append(
            (
                _mix(weight, rate, shrink, bound, rng),
                _mix(bias, rate, shrink, bound, rng),
            )
        )
        if pair is None or sigma_reset is None:
            sigma.append(pair)
        else:
            value = np.float32(sigma_reset * bound)
            sigma.append(
                (
                    np.full_like(pair[0], value),
                    np.full_like(pair[1], value),
                )
            )
    return OrderQNetwork(layers, sigma)


def _mix(
    values: np.ndarray,
    rate: float,
    shrink: float,
    bound: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """`rate` of `values` redrawn at the layer's init scale, the rest shrunk."""

    kept = np.asarray(values, dtype=np.float32) * np.float32(shrink)
    if rate <= 0.0:
        return kept
    fresh = rng.uniform(-bound, bound, size=values.shape).astype(np.float32)
    return np.where(rng.random(values.shape) < rate, fresh, kept).astype(np.float32)
