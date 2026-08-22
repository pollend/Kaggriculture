"""One-shot weight-space perturbation applied when a learner slot is reseeded.

Copying a mature member's weights unchanged restarts the slot *inside the basin
that member is already in*, at a learning rate warming back up -- the
single-lineage run this replaced, with extra steps. The perturbation is what makes
the restart a different search.

**Weight-space dropout, not an `nn.Dropout` layer**, and that is load-bearing:
`policy.py` (numpy, ships in the Kaggle bundle) and `training/network.py` (torch)
must produce bit-identical outputs for identical weights, which a parity test
pins. A layer added to one side breaks that; a function rewriting the weights
*before* either side loads them cannot. Nothing here runs at inference.

Shrink-and-perturb (Ash & Adams) with the shrink defaulted off: reinitialising a
fraction of the weights recovers plasticity, multiplying the survivors is the half
that costs performance.
"""

from __future__ import annotations

import numpy as np

from agents.dqn_market_free_v3c.policy import GruCell, OrderQNetwork


def perturb(
    network: OrderQNetwork,
    rate: float,
    shrink: float = 1.0,
    seed: int = 0,
    sigma_reset: float | None = None,
) -> OrderQNetwork:
    """A copy of `network` with `rate` of its weights reinitialised.

    Per weight rather than per unit: dropping whole rows would delete an action's
    readout, and a row of zeros has no gradient of its own to recover through except
    via the bias.

    Reinitialised entries come from the layer's own `U(-1/sqrt(fan_in), +1/sqrt
    (fan_in))`, matching `NoisyLinear.reset_parameters` and torch's `nn.Linear`, so
    a fresh entry sits at the scale the layer was born at and is explorable again
    rather than immediately dominated.

    `sigma_reset` restores every noisy layer's sigma to that bound scaled by
    `sigma0`. Sigma anneals toward zero over a learner's life, so an inherited one
    leaves the restart with no exploration budget. `None` keeps the source's sigma.

    `rate = 0.0` returns an unperturbed *copy*, so a caller can always treat the
    result as its own to write.

    **The recurrence is perturbed too, and it needs its own arm.** `network.layers`
    is the head alone -- the GRU is a separate attribute, for the reasons in
    `policy.GruCell` -- so a loop over `layers` silently leaves the recurrence
    byte-identical. That would half-defeat the whole mechanism: on a recurrent
    policy the GRU is the part that most encodes the basin, so a slot reseeded with
    the incumbent's memory intact is restarting in the same place with a fresher
    head. Nothing would report it.

    The GRU's bound is also not the head's. `nn.GRUCell.reset_parameters` uses one
    `stdv = 1/sqrt(hidden_size)` for **all four** of its tensors, `weight_ih`
    included -- even though that tensor's fan-in is the state width. Deriving the
    bound from `weight.shape[0]` the way the head arm does would redraw `wx`
    entries at `1/sqrt(229)` instead of `1/sqrt(128)`, about 30% too small, so the
    redrawn weights would be quieter than the ones they replaced and the
    perturbation would be weaker than the configured rate says.
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
    gru = None
    if network.gru is not None:
        bound = 1.0 / np.sqrt(network.gru.hidden_size)
        gru = GruCell(
            *(
                _mix(tensor, rate, shrink, bound, rng)
                for tensor in (
                    network.gru.wx,
                    network.gru.wh,
                    network.gru.bx,
                    network.gru.bh,
                )
            )
        )
    return OrderQNetwork(layers, sigma, gru=gru)


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
