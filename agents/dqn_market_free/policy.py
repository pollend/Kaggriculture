"""Numpy inference for the free-market Q-network.

An ordinary action-head MLP: state in, one Q value per emittable order out.
Torch-free for the same reason `agents/dqn_market_scorer/policy.py` is -- the
submission bundle may not depend on torch -- but it carries its own shape, and
`load` rejects a checkpoint whose input or output width disagrees with the
current schema rather than silently reading it as a shape match.

This replaced a candidate-scoring network that took `state ⊕ action_features`
and returned a single score, evaluated over a 2,650-row block every turn. The
head costs one `(117, 128) @ (128, 128) @ (128, 186)` chain instead -- roughly
53k multiply-accumulates against 51M -- so scoring is no longer anywhere near the
critical path of a rollout, and the training update no longer has to chunk a
410 MB candidate expansion to fit in memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import ACTION_COUNT

HIDDEN = (128, 128)
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.npz"


class OrderQNetwork:
    """MLP mapping (..., STATE_DIM) -> (..., ACTION_COUNT) Q values."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = [(w.astype(np.float32), b.astype(np.float32)) for w, b in layers]

    @classmethod
    def zeros(
        cls,
        state_dim: int = STATE_DIM,
        hidden: tuple[int, ...] = HIDDEN,
        actions: int = ACTION_COUNT,
    ) -> OrderQNetwork:
        """All-zero network. Inference fallback only, never a training start.

        Scores every action identically, so the argmax falls to index 0, which is
        NOOP: the agent trades nothing. Survivable, and a defined behaviour for a
        missing weights file.

        It must never initialize training. An all-zero MLP has zero gradient
        everywhere but its output bias -- both hidden activations are
        identically zero for any input, and `relu'(0) == 0` closes the path back
        to the biases -- so the symmetry never breaks and the run stays at NOOP
        forever while reporting a healthy loss. Use `quiet_start`.
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
    ) -> OrderQNetwork:
        rng = np.random.default_rng(seed)
        sizes = (state_dim, *hidden, actions)
        return cls(
            [
                (
                    rng.standard_normal((a, b)).astype(np.float32) / np.sqrt(a),
                    rng.standard_normal(b).astype(np.float32) * 0.01,
                )
                for a, b in zip(sizes, sizes[1:])
            ]
        )

    @classmethod
    def load(cls, path: Path | str = WEIGHTS_PATH) -> OrderQNetwork:
        with np.load(str(path), allow_pickle=False) as data:
            count = len([key for key in data.files if key.startswith("w")])
            network = cls([(data[f"w{index}"], data[f"b{index}"]) for index in range(count)])
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
        payload: dict[str, np.ndarray] = {
            "state_dim": np.asarray([STATE_DIM], dtype=np.int32),
            "action_count": np.asarray([ACTION_COUNT], dtype=np.int32),
        }
        for index, (weight, bias) in enumerate(self.layers):
            payload[f"w{index}"] = weight
            payload[f"b{index}"] = bias
        np.savez_compressed(str(path), **payload)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        activation = np.asarray(state, dtype=np.float32)
        for weight, bias in self.layers[:-1]:
            activation = np.maximum(activation @ weight + bias, 0.0)
        weight, bias = self.layers[-1]
        return activation @ weight + bias
