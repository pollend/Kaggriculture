"""Numpy inference for the free-market Q-network.

An ordinary action-head MLP: state in, one Q value per emittable order out.
Torch-free for the same reason `agents/dqn_market_scorer/policy.py` is -- the
submission bundle may not depend on torch -- but it carries its own shape, and
`load` rejects a checkpoint whose input or output width disagrees with the
current schema rather than silently reading it as a shape match.

One flat head, so scoring is roughly 53k multiply-accumulates a turn rather than
the 51M a candidate-scoring network cost. That is why nothing here is chunked and
why per-turn decisions are affordable.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import ACTION_COUNT

HIDDEN = (128, 128)
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.npz"

_LAYER_KEY = re.compile(r"w\d+")

Pair = tuple[np.ndarray, np.ndarray]


def _factorized(rng: np.random.Generator, size: int) -> np.ndarray:
    """`f(x) = sign(x) * sqrt(|x|)` over a standard normal draw.

    float32 explicitly: `standard_normal` defaults to float64, and computing the
    noise in double while the weights are single widens the gap against the torch
    mirror by roughly the square root of the layer width -- straight through the
    parity test's 1e-5 budget.
    """

    draw = rng.standard_normal(size, dtype=np.float32)
    return np.sign(draw) * np.sqrt(np.abs(draw))


class OrderQNetwork:
    """MLP mapping (..., STATE_DIM) -> (..., ACTION_COUNT) Q values.

    `layers` is the mean (mu) weight of every layer and is what `__call__` uses
    on its own. `sigma` carries the NoisyNet standard deviations, one slot per
    layer and `None` where the layer is deterministic, so a checkpoint written
    before NoisyNet loads as an ordinary network and scores identically.

    Noise is *state on the network*, drawn by `sample_noise` and held until replaced,
    rather than an argument to `__call__`. That keeps `order_space.choose_order`
    unaware it exists, and makes evaluation determinism structural: the arena, the
    gates and the Kaggle entry point all reach this class through `main.agent`,
    which never calls `sample_noise`.
    """

    def __init__(
        self,
        layers: list[Pair],
        sigma: list[Pair | None] | None = None,
    ):
        self.layers = [(w.astype(np.float32), b.astype(np.float32)) for w, b in layers]
        self.sigma: list[Pair | None] = (
            [
                None if pair is None else (pair[0].astype(np.float32), pair[1].astype(np.float32))
                for pair in sigma
            ]
            if sigma is not None
            else [None] * len(self.layers)
        )
        if len(self.sigma) != len(self.layers):
            raise ValueError("sigma must carry one slot per layer")
        self._noise: list[Pair | None] = [None] * len(self.layers)

    @property
    def is_noisy(self) -> bool:
        return any(pair is not None for pair in self.sigma)

    def sample_noise(
        self,
        rng: np.random.Generator,
        scale: float | np.ndarray = 1.0,
    ) -> None:
        """Draw one factorized epsilon pair per noisy layer, held until replaced.

        Factorized rather than independent: `p + q` draws per layer instead of
        `p * q`, which at this width is 314 against 23,994.

        `scale` multiplies the draw, never `sigma`. `_linear` is bilinear in the two
        epsilons, so a factor on either scales the perturbation exactly; keeping it
        off `sigma` means it cannot compound across calls or reach `save`.

        `scale` may also be a *vector*, one entry per output unit -- directed
        exploration, widening one action's row. It applies only to layers whose
        output width it matches, so the same vector is correct whether the run
        perturbs the head alone or the whole trunk.
        """

        factor = np.asarray(scale, dtype=np.float32)
        self._noise = [
            None
            if pair is None
            else (
                _factorized(rng, weight.shape[0]),
                _factorized(rng, weight.shape[1])
                * (factor if factor.ndim == 0 or factor.shape == (weight.shape[1],) else 1.0),
            )
            for (weight, _bias), pair in zip(self.layers, self.sigma)
        ]

    def clear_noise(self) -> None:
        """Back to mu only. The evaluation and submission mode."""

        self._noise = [None] * len(self.layers)

    @classmethod
    def zeros(
        cls,
        state_dim: int = STATE_DIM,
        hidden: tuple[int, ...] = HIDDEN,
        actions: int = ACTION_COUNT,
    ) -> OrderQNetwork:
        """All-zero network. Inference fallback only, never a training start.

        Scores every action identically, so the argmax falls to NOOP at index 0 --
        survivable, and a defined behaviour for a missing weights file.

        **Never initialize training with this.** An all-zero MLP has zero gradient
        everywhere but its output bias, since both hidden activations are zero for
        any input and `relu'(0) == 0` closes the path back to the biases. The
        symmetry never breaks and the run stays at NOOP forever while reporting a
        healthy loss. Use `quiet_start`.
        """

        sizes = (state_dim, *hidden, actions)
        return cls(
            [
                (np.zeros((a, b), dtype=np.float32), np.zeros(b, dtype=np.float32))
                for a, b in zip(sizes, sizes[1:])
            ]
        )

    @classmethod
    def quiet_start(
        cls,
        seed: int = 0,
        state_dim: int = STATE_DIM,
        hidden: tuple[int, ...] = HIDDEN,
        actions: int = ACTION_COUNT,
    ) -> OrderQNetwork:
        """Random hidden layers, zero output layer: NOOP behaviour, live gradients.

        Every action scores exactly 0.0, so the argmax ties to index 0 and the
        greedy policy is silent -- the same cold start `zeros` gives. Unlike
        `zeros` it trains: `dL/dw_out` is proportional to a non-zero activation,
        so the output layer moves on the first step and the hidden layers get
        gradient through a non-zero `w_out` from the second onward.
        """

        rng = np.random.default_rng(seed)
        sizes = (state_dim, *hidden)
        layers = [
            (
                rng.standard_normal((a, b)).astype(np.float32) / np.sqrt(a),
                rng.standard_normal(b).astype(np.float32) * 0.01,
            )
            for a, b in zip(sizes, sizes[1:])
        ]
        layers.append(
            (
                np.zeros((sizes[-1], actions), dtype=np.float32),
                np.zeros(actions, dtype=np.float32),
            )
        )
        return cls(layers)

    @classmethod
    def random(
        cls,
        seed: int = 0,
        state_dim: int = STATE_DIM,
        hidden: tuple[int, ...] = HIDDEN,
        actions: int = ACTION_COUNT,
        sigma0: float = 0.0,
        noisy_layers: str = "head",
    ) -> OrderQNetwork:
        """`sigma0 > 0` mints a noisy network directly; the default is deterministic.

        Only a test constructor. Training never starts a noisy network from here:
        sigma is born in torch (`training/network.py`) and reaches numpy through
        `to_numpy`, because sigma = 0 is a stationary point of the loss and a
        network exported with it could never grow sigma back.
        """

        rng = np.random.default_rng(seed)
        sizes = (state_dim, *hidden, actions)
        layers = [
            (
                rng.standard_normal((a, b)).astype(np.float32) / np.sqrt(a),
                rng.standard_normal(b).astype(np.float32) * 0.01,
            )
            for a, b in zip(sizes, sizes[1:])
        ]
        if sigma0 <= 0.0:
            return cls(layers)
        last = len(layers) - 1
        sigma: list[Pair | None] = [
            (
                np.full(weight.shape, sigma0 / np.sqrt(weight.shape[0]), dtype=np.float32),
                np.full(bias.shape, sigma0 / np.sqrt(weight.shape[0]), dtype=np.float32),
            )
            if noisy_layers == "all" or index == last
            else None
            for index, (weight, bias) in enumerate(layers)
        ]
        return cls(layers, sigma)

    @classmethod
    def load(cls, path: Path | str = WEIGHTS_PATH) -> OrderQNetwork:
        with np.load(str(path), allow_pickle=False) as data:
            # Anchored, not `startswith("w")`. The sigma keys are named `sw{i}`
            # precisely so they cannot inflate this count, but a loose prefix
            # test is one badly-chosen key away from making `data[f"w{count-1}"]`
            # raise KeyError -- which `main.load_network` swallows, silently
            # falling back to `zeros()`. Challenger and incumbent then play as
            # one policy and every paired margin comes out exactly 0.0.
            count = sum(1 for key in data.files if _LAYER_KEY.fullmatch(key))
            layers = [(data[f"w{index}"], data[f"b{index}"]) for index in range(count)]
            sigma: list[Pair | None] = []
            for index, (weight, bias) in enumerate(layers):
                names = (f"sw{index}", f"sb{index}")
                present = [name in data.files for name in names]
                if any(present) and not all(present):
                    raise ValueError(f"checkpoint layer {index} carries half a sigma pair")
                if not all(present):
                    sigma.append(None)
                    continue
                sigma_w, sigma_b = data[names[0]], data[names[1]]
                if sigma_w.shape != weight.shape or sigma_b.shape != bias.shape:
                    raise ValueError(
                        f"checkpoint layer {index} sigma shape {sigma_w.shape} disagrees "
                        f"with mu shape {weight.shape}"
                    )
                sigma.append((sigma_w, sigma_b))
            network = cls(layers, sigma)
        if not network.layers:
            raise ValueError("checkpoint contains no layers")
        observed_state = network.layers[0][0].shape[0]
        observed_actions = network.layers[-1][0].shape[1]
        if observed_state != STATE_DIM or observed_actions != ACTION_COUNT:
            raise ValueError(
                f"checkpoint shape {observed_state}->{observed_actions} does not match "
                f"schema {STATE_DIM}->{ACTION_COUNT}"
            )
        return network

    def save(self, path: Path | str = WEIGHTS_PATH) -> None:
        """Write mu always, sigma only where a layer carries it.

        A deterministic network therefore writes exactly the key set this agent
        wrote before NoisyNet existed, so runs that leave `noisy` off churn no
        checkpoints -- and an inference bundle built before NoisyNet, which knows
        only `w{i}`/`b{i}`, reads a noisy checkpoint as its own mu-only policy
        rather than failing.
        """

        payload: dict[str, np.ndarray] = {
            "state_dim": np.asarray([STATE_DIM], dtype=np.int32),
            "action_count": np.asarray([ACTION_COUNT], dtype=np.int32),
        }
        for index, (weight, bias) in enumerate(self.layers):
            payload[f"w{index}"] = weight
            payload[f"b{index}"] = bias
            pair = self.sigma[index]
            if pair is not None:
                payload[f"sw{index}"], payload[f"sb{index}"] = pair
        np.savez_compressed(str(path), **payload)

    def _linear(self, index: int, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        out = x @ weight + bias
        sigma = self.sigma[index]
        noise = self._noise[index]
        if sigma is None or noise is None:
            return out
        sigma_w, sigma_b = sigma
        f_in, f_out = noise
        # Deliberately the factored form, and deliberately this association
        # order: it is what `training/network.py` computes, and materializing
        # `mu + sigma * outer(f_out, f_in)` on one side only would round
        # differently in float32 and eat the parity test's tolerance.
        return out + ((x * f_in) @ sigma_w) * f_out + sigma_b * f_out

    def __call__(self, state: np.ndarray) -> np.ndarray:
        activation = np.asarray(state, dtype=np.float32)
        last = len(self.layers) - 1
        for index, (weight, bias) in enumerate(self.layers):
            activation = self._linear(index, activation, weight, bias)
            if index != last:
                activation = np.maximum(activation, 0.0)
        return activation
