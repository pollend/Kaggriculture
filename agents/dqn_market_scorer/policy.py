"""Numpy inference for the market Q-network.

Deliberately torch-free: the submission ships `weights.npz` and evaluates two
small matmuls per market decision. The candidate set is ~38 modifier rows plus
9x7 reserve rows, a (100, 137) @ (137, 128) product, microseconds against the
1 s `actTimeout` -- and it only runs twice a day, not every turn.

Layer shapes and activation order mirror `training/network.py`; a parity test
pins the two together.

Unlike `dqn_job_scorer` there is no bundle: this agent has exactly one network,
which is the whole point of the design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agents.dqn_market_scorer.market_policy import MARKET_FEATURE_DIM

HIDDEN = (128, 128)
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.npz"


class MarketQNetwork:
    """MLP mapping (n_candidates, MARKET_FEATURE_DIM) -> (n_candidates,) Q values."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = [(w.astype(np.float32), b.astype(np.float32)) for w, b in layers]

    @classmethod
    def zeros(
        cls, feature_dim: int = MARKET_FEATURE_DIM, hidden: tuple[int, ...] = HIDDEN
    ) -> MarketQNetwork:
        """An untrained network.

        Outputs all-zero Q, so every argmax falls to index 0 -- which is the
        donor-default modifier and a zero reserve. A zero net therefore plays
        the unmodified donor policy, not a random one.
        """
        sizes = (feature_dim, *hidden, 1)
        return cls(
            [
                (np.zeros((a, b), dtype=np.float32), np.zeros(b, dtype=np.float32))
                for a, b in zip(sizes, sizes[1:])
            ]
        )

    @classmethod
    def random(
        cls, seed: int = 0, feature_dim: int = MARKET_FEATURE_DIM, hidden: tuple[int, ...] = HIDDEN
    ) -> MarketQNetwork:
        rng = np.random.default_rng(seed)
        sizes = (feature_dim, *hidden, 1)
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
    def load(cls, path: Path | str = WEIGHTS_PATH) -> MarketQNetwork:
        with np.load(str(path), allow_pickle=False) as data:
            count = len([key for key in data.files if key.startswith("w")])
            network = cls([(data[f"w{index}"], data[f"b{index}"]) for index in range(count)])
        observed = network.layers[0][0].shape[0] if network.layers else 0
        if observed != MARKET_FEATURE_DIM:
            raise ValueError(
                f"checkpoint feature dimension {observed} does not match schema {MARKET_FEATURE_DIM}"
            )
        return network

    def save(self, path: Path | str = WEIGHTS_PATH) -> None:
        payload: dict[str, np.ndarray] = {
            "feature_dim": np.asarray([MARKET_FEATURE_DIM], dtype=np.int32)
        }
        for index, (weight, bias) in enumerate(self.layers):
            payload[f"w{index}"] = weight
            payload[f"b{index}"] = bias
        np.savez_compressed(str(path), **payload)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        activation = np.asarray(features, dtype=np.float32)
        for weight, bias in self.layers[:-1]:
            activation = np.maximum(activation @ weight + bias, 0.0)
        weight, bias = self.layers[-1]
        return (activation @ weight + bias).reshape(-1)
