"""Submission-safe SP-PSRO policy artifacts.

A population member is either one joint job/market policy or a root-level
mixture of such policies.  Mixtures choose one component at episode start;
they never average Q values within an episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents.dqn_job_scorer.policy import PolicyBundle, QNetwork

MIXTURE_VERSION = 1


@dataclass(frozen=True, slots=True)
class PopulationArtifact:
    components: tuple[PolicyBundle, ...]
    weights: np.ndarray

    @property
    def job(self) -> QNetwork:
        if len(self.components) != 1:
            raise AttributeError("mixture artifacts do not have one job network")
        return self.components[0].job

    @property
    def market(self) -> QNetwork | None:
        if len(self.components) != 1:
            raise AttributeError("mixture artifacts do not have one market network")
        return self.components[0].market

    @classmethod
    def single(cls, bundle: PolicyBundle) -> PopulationArtifact:
        return cls((bundle,), np.asarray([1.0], dtype=np.float64))

    @classmethod
    def load(cls, path: str | Path) -> PopulationArtifact:
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            if "sp_mixture_version" not in data.files:
                return cls.single(PolicyBundle.load(source))
            if int(data["sp_mixture_version"][0]) != MIXTURE_VERSION:
                raise ValueError("unsupported SP-PSRO mixture artifact")
            count = int(data["component_count"][0])
            weights = data["weights"].astype(np.float64)
            components = []
            for component in range(count):
                job_count = int(data[f"c{component}_job_count"][0])
                market_count = int(data[f"c{component}_market_count"][0])
                job = QNetwork(
                    [
                        (data[f"c{component}_job_w{i}"], data[f"c{component}_job_b{i}"])
                        for i in range(job_count)
                    ]
                )
                market = QNetwork(
                    [
                        (
                            data[f"c{component}_market_w{i}"],
                            data[f"c{component}_market_b{i}"],
                        )
                        for i in range(market_count)
                    ]
                )
                components.append(PolicyBundle(job, market))
        if not components or weights.shape != (len(components),):
            raise ValueError("invalid SP-PSRO mixture artifact")
        if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("invalid SP-PSRO mixture weights")
        return cls(tuple(components), weights / weights.sum())

    def save(self, path: str | Path) -> None:
        if len(self.components) == 1:
            self.components[0].save(path)
            return
        weights = np.asarray(self.weights, dtype=np.float64)
        if (
            weights.shape != (len(self.components),)
            or np.any(weights < 0)
            or not np.isfinite(weights).all()
            or weights.sum() <= 0
        ):
            raise ValueError("invalid SP-PSRO mixture weights")
        payload: dict[str, np.ndarray] = {
            "sp_mixture_version": np.asarray([MIXTURE_VERSION], dtype=np.int16),
            "component_count": np.asarray([len(self.components)], dtype=np.int16),
            "weights": weights / weights.sum(),
        }
        for component, bundle in enumerate(self.components):
            if bundle.market is None:
                raise ValueError("SP-PSRO mixture components require market networks")
            payload[f"c{component}_job_count"] = np.asarray([len(bundle.job.layers)], dtype=np.int16)
            payload[f"c{component}_market_count"] = np.asarray(
                [len(bundle.market.layers)], dtype=np.int16
            )
            for prefix, network in (("job", bundle.job), ("market", bundle.market)):
                for index, (weight, bias) in enumerate(network.layers):
                    payload[f"c{component}_{prefix}_w{index}"] = weight
                    payload[f"c{component}_{prefix}_b{index}"] = bias
        np.savez_compressed(path, **payload)

    def choose(self, rng: np.random.Generator) -> PolicyBundle:
        index = int(rng.choice(len(self.components), p=self.weights))
        return self.components[index]
