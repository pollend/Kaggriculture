"""DQN job-scoring agent -- strategy attempt #2.

The Q-network replaces only the hand-tuned `Job.value` scoring in
`agents/configurable_crop_surrogate/agent.py`. Job enumeration, unit routing, and every market
order (selling, hiring, seed buying) stay on the existing rules, so this is one
learning problem rather than four.

Submission-safe: standard library plus numpy. It must never import torch.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from agents.dqn_job_scorer import features as F
from agents.dqn_job_scorer.jobs import enumerate_jobs
from agents.dqn_job_scorer.market import MarketConfig, market_orders
from agents.dqn_job_scorer.policy import WEIGHTS_PATH, QNetwork, select
from agents.dqn_job_scorer.rules import get, step_toward

PASS_ACTION = ["PASS"]

# Longest walk worth honouring before re-deciding. The board diagonal is 18, so
# anything beyond this means the target moved or became unreachable.
MAX_COMMIT_TURNS = 20


class JobScorerAgent:
    """Scores candidate jobs with a Q-network, routes with the existing rules."""

    def __init__(
        self,
        network: QNetwork,
        market: MarketConfig | None = None,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
        recorder: Any | None = None,
        episode_steps: int = 720,
        turns_per_day: int = 24,
    ):
        self.network = network
        self.epsilon = epsilon
        self.rng = rng
        self.recorder = recorder
        self.market = market or MarketConfig()
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        # unit index -> (job key, turns spent walking to it)
        self._commitments: dict[int, tuple[tuple, int]] = {}
        self._day = -1

    @staticmethod
    def _key(job) -> tuple:
        return (job.kind, job.target, job.crop, job.owner)

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        farms = get(obs, "farms", []) or []
        player = int(get(obs, "player", 0))
        if not farms or player >= len(farms):
            return {"farmer": list(PASS_ACTION), "hands": [], "market": []}

        farm = farms[player]
        private = get(obs, "private", {}) or {}
        step = int(get(obs, "step", 0))
        day = int(get(obs, "day", 0))
        episode_steps = int(get(configuration, "episodeSteps", self.episode_steps))
        turns_per_day = int(get(configuration, "turnsPerDay", self.turns_per_day))
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        hour = int(get(obs, "hour", step % turns_per_day))

        if step == 0:
            self._commitments.clear()
            self._day = -1
            if self.recorder is not None:
                self.recorder.reset()
        if day != self._day:
            # Every unit teleports to the shed at day start and hands are
            # re-hired, so unit indices no longer mean the same worker.
            self._commitments.clear()
            self._day = day

        positions = [tuple(farm["farmer"]), *(tuple(pos) for pos in farm.get("hands", []) or [])]
        inventories = list(private.get("inventories", []) or [])
        while len(inventories) < len(positions):
            inventories.append({})

        jobs, truncated = enumerate_jobs(farm, private, day, positions, inventories)
        global_feats = F.global_features(obs, farm, private, player, episode_steps, turns_per_day)
        job_static = F.job_static_features(jobs, obs, farm, day, step, episode_steps, turns_per_day)
        targets = F.job_targets(jobs)

        taken = np.zeros(len(jobs), dtype=bool)
        owner = np.array([-1 if job.owner is None else job.owner for job in jobs], dtype=np.int32)
        index_of = {self._key(job): index for index, job in enumerate(jobs)}

        # Honour live commitments first, so a unit already walking somewhere keeps
        # its target reserved against a unit that has not chosen yet.
        held: dict[int, int] = {}
        for unit in range(len(positions)):
            entry = self._commitments.get(unit)
            if entry is None:
                continue
            key, age = entry
            index = index_of.get(key)
            if index is None or taken[index] or age > MAX_COMMIT_TURNS:
                self._commitments.pop(unit, None)  # job done, gone, or unreachable
                continue
            held[unit] = index
            if index != 0:
                taken[index] = True

        chosen: list[int] = []
        committed_indices: list[int] = []
        unit_actions: list[list[str]] = []

        for unit, position in enumerate(positions):
            committed = held.get(unit, -1)
            committed_indices.append(committed)
            distance = F.distance_features(targets, position, committed)
            q_values = self.network(F.assemble(global_feats, job_static, distance))

            if committed >= 0:
                # Sticky: no re-selection and no exploration mid-walk. Random
                # re-picks every turn are what made units oscillate instead of
                # arriving, and epsilon applied per step derailed almost every walk.
                index = committed
            else:
                # A job is available if nobody took it and it is not bound to
                # another unit. PASS (index 0) is never consumed.
                allowed = (~taken) & ((owner < 0) | (owner == unit))
                allowed[0] = True
                index = select(q_values, allowed, self.epsilon, self.rng)
                if index != 0:
                    taken[index] = True

            chosen.append(index)
            job = jobs[index]
            if position == job.target:
                unit_actions.append(list(job.action))
                self._commitments.pop(unit, None)  # arrived and acted
            else:
                unit_actions.append(step_toward(position, job.target))
                age = self._commitments.get(unit, (None, -1))[1] + 1
                self._commitments[unit] = (self._key(job), age)

        endgame = step >= episode_steps - self.market.endgame_liquidation_turns
        orders = market_orders(
            obs, farm, private, step, hour, endgame, self.market, episode_steps, turns_per_day
        )

        if self.recorder is not None:
            self.recorder.record(
                global_feats=global_feats,
                job_static=job_static,
                job_targets=targets,
                positions=positions,
                chosen=chosen,
                committed=committed_indices,
                money=float(farm.get("money", 0.0)),
                obs=obs,
                farm=farm,
                private=private,
                truncated=truncated,
            )

        return {"farmer": unit_actions[0], "hands": unit_actions[1:], "market": orders}


def _load_network() -> QNetwork:
    """Trained weights if present, otherwise a zero net.

    A `weights` key in KAGG_AGENT_CONFIG overrides the default path -- that is
    how self-play resolves a frozen snapshot as an opponent spec, e.g.
    ``agents/dqn_job_scorer/main.py?weights=runs/dqn/snapshots/round0100.npz``.
    The environment variable is read the same way
    `agents.configurable_crop_surrogate.agent.strategy_from_environment` reads it, so the agent
    stays dependency-free.

    A zero net plays legally and deterministically (every Q is 0, so selection
    falls to the index tiebreak), which keeps the agent importable and testable
    before the first training run finishes.
    """
    raw = os.environ.get("KAGG_AGENT_CONFIG")
    if raw:
        override = json.loads(raw).get("weights")
        if override:
            return QNetwork.load(override)
    if WEIGHTS_PATH.exists():
        return QNetwork.load(WEIGHTS_PATH)
    return QNetwork.zeros()


_AGENT = JobScorerAgent(_load_network())


def agent(obs: Any, configuration: Any | None = None) -> dict[str, list]:
    return _AGENT(obs, configuration)


def __getattr__(name: str) -> Any:
    """Expose the league's post-episode hook without importing it for inference.

    `agent.toml` points `episode_evaluator` at `main:evaluate_episode`, but a
    Kaggle submission only ever calls `agent()`. Resolving the evaluator lazily
    keeps the submission's import graph to numpy and this package.
    """
    if name == "evaluate_episode":
        from agents.dqn_job_scorer.evaluation import evaluate_episode

        return evaluate_episode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
