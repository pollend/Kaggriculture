"""Shared episode-scoped runtime for SP-PSRO inference and training."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.dqn_job_scorer.main import JobScorerAgent
from agents.sp_psro_job_scorer.policy import PopulationArtifact


class SPPSROAgent:
    def __init__(
        self,
        artifact: PopulationArtifact,
        seed: int | None,
        *,
        component_index: int | None = None,
        epsilon: float = 0.0,
        market_epsilon: float = 0.0,
        recorder: Any | None = None,
    ):
        self.artifact = artifact
        self.rng = np.random.default_rng(seed)
        self.component_index = component_index
        self.epsilon = epsilon
        self.market_epsilon = market_epsilon
        self.recorder = recorder
        self.inner: JobScorerAgent | None = None

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        step = int(obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0))
        if self.inner is None or step == 0:
            bundle = (
                self.artifact.choose(self.rng)
                if self.component_index is None
                else self.artifact.component(self.component_index)
            )
            self.inner = JobScorerAgent(
                bundle.job,
                bundle.market,
                epsilon=self.epsilon,
                market_epsilon=self.market_epsilon,
                rng=self.rng,
                recorder=self.recorder,
            )
        return self.inner(obs, configuration)
