"""Frozen donor field policy, learned market layer.

`dqn_job_scorer` learns two Q-networks off one scalar reward -- a per-turn job
net and a daily market net -- and they share the credit for every dollar. This
agent removes that coupling by freezing a field policy that already wins and
learning only the market: one network, one credit-assignment problem.

The field policy is `donor.py`, an unmodified copy of the pilkwang competitor
snapshot (see NOTICE). It is used exactly as written: `_role_plan` then
`_unit_actions`. The donor's dependency runs field -> market and never back, and
it gates every market-supplied resource at runtime (`seeds.get(crop, 0) > 0`
before PLANT, live `farm["hands"]`, live `unlocked_quadrants`), so replacing its
market layer degrades gracefully instead of desynchronising.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from agents.dqn_market_scorer import donor, market_exec
from agents.dqn_market_scorer.market_policy import (
    DEFAULT_MODIFIER,
    MarketModifier,
    choose_market,
    enumerate_market_candidates,
)
from agents.dqn_market_scorer.policy import WEIGHTS_PATH, MarketQNetwork

PASS_ACTION = ("PASS",)

# Two decisions a day rather than one. The market net is the only learner here,
# so it needs the samples; the donor hires at hour <= 2, which the hour-0
# decision covers, and reserves matter all day, which the midday one revisits.
DECISION_HOURS = (0, 12)


def _reset_donor_state() -> None:
    """Clear the donor's module-level latches between episodes.

    `donor._schedule_signature` and `donor._schedule_market_adjustment` keep
    cross-turn state in module globals and self-reset on step 0. Rollouts run in
    a `spawn` pool so processes are not usually shared, but a second episode in
    one interpreter would otherwise inherit a latched opponent signature.
    """

    donor._SIGNATURE_LAST_STEP = -1
    donor._SIGNATURE_ACTIVE = False
    donor._SCHEDULE_WHEAT_REQUESTED = 0


class MarketScorerAgent:
    def __init__(
        self,
        network: MarketQNetwork | None = None,
        *,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
        recorder: Any | None = None,
        episode_steps: int = 720,
        turns_per_day: int = 24,
    ):
        self.network = network
        self.epsilon = float(epsilon)
        self.rng = rng
        self.recorder = recorder
        self.episode_steps = int(episode_steps)
        self.turns_per_day = int(turns_per_day)
        self._modifier: MarketModifier = DEFAULT_MODIFIER
        self._reserves: dict[str, int] = {}
        self._decided_step = -1

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0) or 0)
        if not farms or not (0 <= player < len(farms)):
            return {"farmer": list(PASS_ACTION), "hands": [], "market": []}

        farm = farms[player]
        private = obs.get("private", {}) or {}
        episode_steps = int(donor._cfg(configuration, "episodeSteps", self.episode_steps))
        turns_per_day = int(donor._cfg(configuration, "turnsPerDay", self.turns_per_day))
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        step = int(obs.get("step", 0) or 0)
        hour = int(obs.get("hour", step % max(1, turns_per_day)) or 0)

        if step == 0:
            _reset_donor_state()
            self._modifier = DEFAULT_MODIFIER
            self._reserves = {}
            self._decided_step = -1
            if self.recorder is not None:
                self.recorder.reset()

        roles = donor._role_plan(obs, farm)
        field = donor._unit_actions(obs, configuration, farm, private, roles)

        if self.network is not None and (step == 0 or hour in DECISION_HOURS) and step != self._decided_step:
            candidates = enumerate_market_candidates(
                obs,
                farm,
                private,
                roles,
                field,
                config=configuration,
                episode_steps=episode_steps,
                turns_per_day=turns_per_day,
            )
            decision = choose_market(candidates, self.network, self.epsilon, self.rng)
            self._modifier = decision.modifier
            self._reserves = decision.reserves
            self._decided_step = step
            if self.recorder is not None:
                self.recorder.record_market(step=step, decision=decision)

        orders = market_exec.market_actions(
            obs,
            configuration,
            farm,
            private,
            roles,
            field,
            modifier=self._modifier,
            reserves=self._reserves,
        )
        orders = donor._schedule_market_adjustment(obs, configuration, farm, private, orders)

        if self.recorder is not None:
            self.recorder.record_turn(step=step, money=float(farm.get("money", 0) or 0), obs=obs, farm=farm, private=private)

        return {"farmer": field["farmer"], "hands": field["hands"], "market": orders}


# Bound at import, while `sim.registry.resolve` still has KAGG_AGENT_CONFIG
# exported for *this* spec. Reading it any later is too late: resolve() restores
# the variable in a finally block, and `sim.runner.run_match` resolves both seats
# before either one plays. Reading it lazily makes every `?weights=` silently
# fall back to the packaged weights, so challenger and incumbent play as one
# policy and every paired promotion margin comes out exactly 0.0.
_IMPORT_CONFIG = os.environ.get("KAGG_AGENT_CONFIG")


def load_network(raw: str | None = _IMPORT_CONFIG) -> MarketQNetwork:
    """Trained weights if present, otherwise a zero net.

    A `weights` key in KAGG_AGENT_CONFIG overrides the default path -- that is
    how self-play and the promotion gate resolve a frozen snapshot as an
    opponent spec, e.g.
    ``agents/dqn_market_scorer/main.py?weights=artifacts/.../abc123.npz``.

    A zero net scores every candidate equally, so each argmax takes index 0:
    the donor-default modifier and a zero reserve. The fallback is therefore
    the unmodified donor policy, which is a strong agent in its own right.
    """
    if raw:
        config = json.loads(raw)
        override = config.get("weights")
        if override:
            return MarketQNetwork.load(override)
        if config.get("network_seed") is not None:
            return MarketQNetwork.random(seed=int(config["network_seed"]))
    if WEIGHTS_PATH.exists():
        try:
            return MarketQNetwork.load(WEIGHTS_PATH)
        except ValueError:
            pass
    return MarketQNetwork.zeros()


_AGENT: MarketScorerAgent | None = None


def agent(obs: Any, configuration: Any | None = None) -> dict[str, list]:
    global _AGENT
    if _AGENT is None:
        _AGENT = MarketScorerAgent(load_network())
    try:
        return _AGENT(obs, configuration)
    except Exception:
        # Mirrors the donor's own guard: a thrown turn costs one turn, not the
        # episode. Training wrappers attach a recorder and re-raise instead, so
        # a broken rollout is dropped rather than silently trained on.
        if _AGENT is not None and _AGENT.recorder is not None:
            raise
        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0) or 0)
        hands = len(farms[player].get("hands", []) or []) if 0 <= player < len(farms) else 0
        return {"farmer": list(PASS_ACTION), "hands": [list(PASS_ACTION) for _ in range(hands)], "market": []}


def __getattr__(name: str) -> Any:
    """Expose the league's post-episode hook without importing it for inference.

    `agent.toml` points `episode_evaluator` at `main:evaluate_episode`, but a
    Kaggle submission only ever calls `agent()`. Resolving the evaluator lazily
    keeps the submission's import graph to numpy and this package.
    """
    if name == "evaluate_episode":
        from agents.dqn_market_scorer.evaluation import evaluate_episode

        return evaluate_episode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
