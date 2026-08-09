"""Numpy inference for the job-scoring Q-network.

Deliberately torch-free: the submission ships `weights.npz` and evaluates two
small matmuls per turn. A full 64-job candidate set is a (64, 71) @ (71, 128)
product, which is microseconds against the 1 s `actTimeout`.

Layer shapes and activation order mirror `agents/dqn_job_scorer/training/network.py`; a
parity test pins the two together.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents.dqn_job_scorer.features import FEATURE_DIM
from agents.dqn_job_scorer.market_policy import MARKET_FEATURE_DIM

HIDDEN = (128, 128)
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.npz"


class QNetwork:
    """MLP mapping (n_jobs, FEATURE_DIM) -> (n_jobs,) Q values."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        self.layers = [(w.astype(np.float32), b.astype(np.float32)) for w, b in layers]

    @classmethod
    def zeros(cls, feature_dim: int = FEATURE_DIM, hidden: tuple[int, ...] = HIDDEN) -> QNetwork:
        """An untrained network. Outputs all-zero Q, so selection falls to the
        index tiebreak and the agent plays a deterministic, legal, useless game."""
        sizes = (feature_dim, *hidden, 1)
        return cls([(np.zeros((a, b), dtype=np.float32), np.zeros(b, dtype=np.float32))
                    for a, b in zip(sizes, sizes[1:])])

    @classmethod
    def random(cls, seed: int = 0, feature_dim: int = FEATURE_DIM, hidden: tuple[int, ...] = HIDDEN) -> QNetwork:
        rng = np.random.default_rng(seed)
        sizes = (feature_dim, *hidden, 1)
        return cls([
            (rng.standard_normal((a, b)).astype(np.float32) / np.sqrt(a), rng.standard_normal(b).astype(np.float32) * 0.01)
            for a, b in zip(sizes, sizes[1:])
        ])

    @classmethod
    def load(cls, path: Path | str = WEIGHTS_PATH) -> QNetwork:
        with np.load(str(path)) as data:
            count = len([key for key in data.files if key.startswith("w")])
            network = cls([(data[f"w{index}"], data[f"b{index}"]) for index in range(count)])
        if not network.layers or network.layers[0][0].shape[0] != FEATURE_DIM:
            observed = network.layers[0][0].shape[0] if network.layers else 0
            raise ValueError(f"checkpoint feature dimension {observed} does not match schema {FEATURE_DIM}")
        return network

    def save(self, path: Path | str = WEIGHTS_PATH) -> None:
        payload = {}
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


@dataclass(frozen=True, slots=True)
class PolicyBundle:
    """Submission artifact containing the field and optional market networks."""

    job: QNetwork
    market: QNetwork | None = None

    @classmethod
    def random(cls, seed: int = 0, *, with_market: bool = True) -> PolicyBundle:
        return cls(
            QNetwork.random(seed=seed),
            QNetwork.random(seed=seed + 1, feature_dim=MARKET_FEATURE_DIM) if with_market else None,
        )

    @classmethod
    def load(cls, path: Path | str = WEIGHTS_PATH) -> PolicyBundle:
        with np.load(str(path), allow_pickle=False) as data:
            if "bundle_version" not in data.files:
                return cls(QNetwork.load(path), None)
            job_count = len([key for key in data.files if key.startswith("job_w")])
            market_count = len([key for key in data.files if key.startswith("market_w")])
            job = QNetwork([(data[f"job_w{i}"], data[f"job_b{i}"]) for i in range(job_count)])
            market = QNetwork([(data[f"market_w{i}"], data[f"market_b{i}"]) for i in range(market_count)])
        if not job.layers or job.layers[0][0].shape[0] != FEATURE_DIM:
            raise ValueError("joint checkpoint has an incompatible job feature schema")
        if not market.layers or market.layers[0][0].shape[0] != MARKET_FEATURE_DIM:
            raise ValueError("joint checkpoint has an incompatible market feature schema")
        return cls(job, market)

    def save(self, path: Path | str = WEIGHTS_PATH) -> None:
        if self.market is None:
            self.job.save(path)
            return
        payload: dict[str, np.ndarray] = {"bundle_version": np.asarray([1], dtype=np.int16)}
        for prefix, network in (("job", self.job), ("market", self.market)):
            for index, (weight, bias) in enumerate(network.layers):
                payload[f"{prefix}_w{index}"] = weight
                payload[f"{prefix}_b{index}"] = bias
        np.savez_compressed(str(path), **payload)


def select(
    q_values: np.ndarray,
    allowed: np.ndarray | None = None,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
) -> int:
    """Epsilon-greedy over the allowed candidates.

    Ties break to the lowest index, so epsilon=0 play is a deterministic
    function of the observation -- which the arena's reproducibility checks
    depend on.
    """
    if allowed is None:
        allowed = np.ones(q_values.shape[0], dtype=bool)
    indices = np.flatnonzero(allowed)
    if indices.size == 0:
        return 0  # PASS is always job 0
    if epsilon > 0.0 and rng is not None and rng.random() < epsilon:
        return int(rng.choice(indices))
    return int(indices[np.argmax(q_values[indices])])
