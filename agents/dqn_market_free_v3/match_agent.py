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

from agents.dqn_market_free_v3.main import MarketFreeAgent
from agents.dqn_market_free_v3.policy import OrderQNetwork
from agents.dqn_market_free_v3.training import rare
from agents.dqn_market_free_v3.training.collect import Recorder, save_episode_batch


class MatchPolicyAgent:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.role = str(config.get("role", "learner"))
        # Optional so the identity spec -- weights and exploration knobs, with
        # the per-episode trace path and revision id stripped -- still resolves.
        self.policy_revision_id = str(config.get("policy_revision_id", "free-market-learner"))
        self.trace_path = Path(str(config["trace_path"])) if config.get("trace_path") else None
        self.gamma = float(config.get("gamma", 0.999))
        # Defaults to 0.0 rather than to `OPPONENT_WEIGHT`: a spec saying nothing
        # about the relative leg predates it, and must label its rewards as it did
        # then.
        self.opponent_weight = float(config.get("opponent_weight", 0.0))
        self.recorder = Recorder() if self.trace_path is not None else None
        seed = int(config.get("exploration_seed", 0))
        network = OrderQNetwork.load(str(config["weights"]))
        # Directed exploration: which sigma rows this actor draws wider. Resolved
        # once, since the counts are a whole round old by the time an episode
        # starts.
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
                    # Absent on a spec written before the rate arm existed, which
                    # must keep reading the statistic it was measured with.
                    novelty=str(config.get("rare_sigma_novelty", "count")),
                )
        # Ape-X's per-actor temperature spread on the noise: one scale per seed
        # pair, so a batch carries both an exploratory and an effectively greedy
        # read. Scaled at the draw, never into `network.sigma` -- see `MarketFreeAgent`.
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
            self.opponent_weight,
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
