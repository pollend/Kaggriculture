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

from agents.dqn_job_scorer.policy import PolicyBundle, QNetwork
from agents.sp_psro_job_scorer.policy import PopulationArtifact
from agents.sp_psro_job_scorer.runtime import SPPSROAgent

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


_CONFIG = _config()
_AGENT = SPPSROAgent(
    _load_artifact(_CONFIG),
    _CONFIG.get("policy_seed"),
    component_index=(int(_CONFIG["component_index"]) if "component_index" in _CONFIG else None),
)


def agent(obs: Any, configuration: Any | None = None) -> dict[str, list]:
    return _AGENT(obs, configuration)
