"""Training-only manifest agent for persisted SP-PSRO matchups."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agents.dqn_job_scorer.rules import get
from agents.dqn_job_scorer.training.collect import Recorder, save_episode_batch
from agents.sp_psro_job_scorer.policy import PopulationArtifact
from agents.sp_psro_job_scorer.runtime import SPPSROAgent


class MatchPolicyAgent:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.role = str(config.get("role", "population"))
        self.policy_revision_id = str(config["policy_revision_id"])
        self.trace_path = Path(str(config["trace_path"])) if config.get("trace_path") else None
        self.gamma = float(config.get("gamma", 0.997))
        self.recorder = Recorder() if self.trace_path is not None else None
        seed = int(config.get("exploration_seed", 0))
        self.artifact = PopulationArtifact.load(str(config["weights"]))
        epsilon = float(self.config.get("epsilon", 0.0))
        self.inner = SPPSROAgent(
            self.artifact,
            seed,
            epsilon=epsilon,
            market_epsilon=float(self.config.get("market_epsilon", epsilon)),
            recorder=self.recorder,
        )

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        return self.inner(obs, configuration)

    def finalize_episode(self, steps: list, seat: int) -> dict[str, Any] | None:
        if self.recorder is None or self.trace_path is None:
            return None
        final = steps[-1]
        obs = final[seat].observation
        farms = get(obs, "farms", []) or []
        record = self.recorder.finish(
            float(farms[seat].get("money", 0.0)),
            float(farms[1 - seat].get("money", 0.0)),
            self.gamma,
        )
        save_episode_batch([record], self.trace_path)
        digest = hashlib.sha256(self.trace_path.read_bytes()).hexdigest()
        return {
            "role": self.role,
            "policy_revision_id": self.policy_revision_id,
            "artifact_path": str(self.trace_path),
            "artifact_sha256": digest,
            "sample_count": 1,
        }


def make_agent(config: dict[str, Any] | None = None) -> MatchPolicyAgent:
    return MatchPolicyAgent(dict(config or {}))
