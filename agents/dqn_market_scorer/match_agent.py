"""Training-only manifest agent for persisted market rollout matchups.

`main:agent` is the inference entry point: greedy, no exploration, no recording.
Collection needs the same policy with an epsilon schedule and a `Recorder`
attached, and it needs to be reachable as an ordinary agent spec so a rollout is
run by `sim.runner.run_match` and lands in `episodes`/`episode_players` like any
league match. That is what this wrapper is -- the exploration knobs arrive as
spec query parameters, and `finalize_episode` hands the recorded decisions back
to the lab as an `episode_training_artifacts` sidecar.

Never submitted: it imports `training/`, which is deliberately absent from the
inference allowlist in `agent.toml`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn_market_scorer.main import MarketScorerAgent
from agents.dqn_market_scorer.policy import MarketQNetwork
from agents.dqn_market_scorer.training.collect import Recorder, save_episode_batch


class MatchPolicyAgent:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.role = str(config.get("role", "learner"))
        # Optional so the identity spec -- weights and exploration rate, with
        # the per-episode trace path and revision id stripped -- still resolves.
        self.policy_revision_id = str(config.get("policy_revision_id", "market-learner"))
        self.trace_path = Path(str(config["trace_path"])) if config.get("trace_path") else None
        self.gamma = float(config.get("gamma", 0.997))
        self.recorder = Recorder() if self.trace_path is not None else None
        seed = int(config.get("exploration_seed", 0))
        self.inner = MarketScorerAgent(
            MarketQNetwork.load(str(config["weights"])),
            epsilon=float(config.get("epsilon", 0.0)),
            rng=np.random.default_rng(seed),
            recorder=self.recorder,
        )

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        return self.inner(obs, configuration)

    def finalize_episode(self, steps: list, seat: int) -> dict[str, Any] | None:
        if self.recorder is None or self.trace_path is None:
            return None
        final = steps[-1]
        obs = final[seat].observation
        farms = obs.get("farms", []) or []
        record = self.recorder.finish(
            float(farms[seat].get("money", 0.0)),
            float(farms[1 - seat].get("money", 0.0)),
            self.gamma,
        )
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
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
