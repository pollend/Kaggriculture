"""Training-only manifest agent for persisted free-market rollout matchups.

`main:agent` is the inference entry point: greedy, no exploration, no recording.
Collection needs the same policy with a per-actor noise scale and a
`Recorder` attached, reachable as an ordinary agent spec so a rollout is run by
`sim.runner.run_match` and lands in `episodes`/`episode_players` like any league
match.

Never submitted: it imports `training/`, which is deliberately absent from the
inference allowlist in `agent.toml`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn_market_free.main import MarketFreeAgent
from agents.dqn_market_free.policy import OrderQNetwork
from agents.dqn_market_free.training import rare
from agents.dqn_market_free.training.collect import Recorder, save_episode_batch


class MatchPolicyAgent:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.role = str(config.get("role", "learner"))
        # Optional so the identity spec -- weights and exploration knobs, with
        # the per-episode trace path and revision id stripped -- still resolves.
        self.policy_revision_id = str(config.get("policy_revision_id", "free-market-learner"))
        self.trace_path = Path(str(config["trace_path"])) if config.get("trace_path") else None
        self.gamma = float(config.get("gamma", 0.999))
        self.recorder = Recorder() if self.trace_path is not None else None
        seed = int(config.get("exploration_seed", 0))
        network = OrderQNetwork.load(str(config["weights"]))
        # Directed exploration: which of the head's 186 sigma rows this actor
        # draws wider than the rest. Resolved once here -- the counts are a whole
        # round old by the time an episode starts, so re-reading them per turn
        # would buy nothing and cost a file read on every decision.
        action_noise_scale = None
        rare_gain = float(config.get("rare_sigma_gain", 0.0))
        rare_path = config.get("rare_sigma_path")
        if rare_gain > 0.0 and rare_path:
            counts = rare.load(str(rare_path))
            if counts is not None:
                action_noise_scale = rare.sigma_scale_vector(
                    *counts,
                    gain=rare_gain,
                    cap=float(config.get("rare_sigma_cap", 1.0 + rare_gain)),
                )
        # Ape-X's per-actor temperature spread, transplanted onto the noise: one
        # scale per seed pair, so a single batch carries both an exploratory and
        # an effectively greedy read of the policy.
        #
        # Carried to the *draw* rather than multiplied into `network.sigma`,
        # which is what this used to do. Both are exact -- the perturbation is
        # bilinear in the two epsilons -- but scaling the checkpoint's sigma is
        # correct only for as long as it happens exactly once per load, and the
        # object gives no sign of that. The flip-rate probe in `training/updater`
        # scales the same way for the same reason, so the two paths cannot drift.
        self.inner = MarketFreeAgent(
            network,
            rng=np.random.default_rng(seed),
            recorder=self.recorder,
            noise_cadence=str(config.get("noise_cadence", "none")),
            noise_scale=float(config.get("sigma_scale", 1.0)),
            action_noise_scale=action_noise_scale,
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
