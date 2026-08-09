"""Submission-safe inference entry point for the SP-PSRO job scorer.

Training maintains a population, but releases exactly one selected champion.
The gameplay implementation is deliberately shared with dqn_job_scorer so the
experiment changes the training policy rather than the action representation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn_job_scorer.main import JobScorerAgent
from agents.dqn_job_scorer.policy import PolicyBundle, QNetwork
from agents.sp_psro_job_scorer.policy import PopulationArtifact

WEIGHTS_PATH = Path(__file__).with_name("weights.npz")


def _config() -> dict[str, Any]:
    raw = os.environ.get("KAGG_AGENT_CONFIG")
    return json.loads(raw) if raw else {}


def _load_artifact(config: dict[str, Any]) -> PopulationArtifact:
    override = config.get("weights")
    if override:
        return PopulationArtifact.load(override)
    if WEIGHTS_PATH.is_file():
        return PopulationArtifact.load(WEIGHTS_PATH)
    return PopulationArtifact.single(PolicyBundle(QNetwork.random(seed=0), None))


class SPPSROAgent:
    def __init__(self, artifact: PopulationArtifact, seed: int | None):
        self.artifact = artifact
        self.rng = np.random.default_rng(seed)
        self.inner: JobScorerAgent | None = None

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        step = int(obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0))
        if self.inner is None or step == 0:
            bundle = self.artifact.choose(self.rng)
            self.inner = JobScorerAgent(bundle.job, bundle.market)
        return self.inner(obs, configuration)


_CONFIG = _config()
_AGENT = SPPSROAgent(_load_artifact(_CONFIG), _CONFIG.get("policy_seed"))


def agent(obs: Any, configuration: Any | None = None) -> dict[str, list]:
    return _AGENT(obs, configuration)
