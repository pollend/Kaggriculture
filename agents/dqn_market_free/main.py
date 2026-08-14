"""Donor field scheduler with market-directed planting and raw market orders.

`dqn_market_scorer` keeps the donor's market executor and learns a modulation of
it: the donor decides what to buy and how much, the network scales the demand.
Every hard rule stays in code -- price thresholds, phase gates, the survival feed
floor, terminal liquidation, cash reserves.

This agent deletes all of it. There is no `market_exec`. Every turn the network
picks one order out of a fixed list, or nothing, and `donor._market_actions` is
never called; neither is `donor._schedule_market_adjustment`, the donor's
post-pass over the resulting orders.

What that means, stated plainly, is that the network has to learn from reward
alone that it cannot spend money it does not have, that animals starve without
WHEAT, that stock held past the last day is worthless, and that the field policy
idles if nobody buys seed. The scaffolded agent is handed all four. This one is
not, and the failure mode is quiet: a market policy that buys no seed scores near
zero while every episode completes without error. `training/collect.py` records
the diagnostics that separate that from a healthy run.

The donor still owns animal placement, field jobs, routing, crop care, harvest,
and liquidation. One narrow adapter lets seed inventory selected by the market
model replace vacant donor crop roles. Existing plants and animal roles are
never rewritten.
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import Any

import numpy as np

from agents.dqn_market_free import donor
from agents.dqn_market_free.market_state import MarketHistory, market_state_features
from agents.dqn_market_free.order_space import choose_order, feasible_actions
from agents.dqn_market_free.planting import market_influenced_roles
from agents.dqn_market_free.policy import WEIGHTS_PATH, OrderQNetwork

PASS_ACTION = ("PASS",)


def _reset_donor_state() -> None:
    """Clear the donor's module-level latches between episodes.

    The donor keeps cross-turn state in module globals and self-resets on step
    0. Rollouts run in a `spawn` pool so processes are not usually shared, but a
    second episode in one interpreter would otherwise inherit a latched opponent
    signature.
    """

    donor._SIGNATURE_LAST_STEP = -1
    donor._SIGNATURE_ACTIVE = False
    donor._SCHEDULE_WHEAT_REQUESTED = 0


class MarketFreeAgent:
    def __init__(
        self,
        network: OrderQNetwork | None = None,
        *,
        rng: np.random.Generator | None = None,
        recorder: Any | None = None,
        episode_steps: int = 720,
        turns_per_day: int = 24,
        noise_cadence: str = "none",
        noise_scale: float = 1.0,
        action_noise_scale: np.ndarray | None = None,
    ):
        self.network = network
        # Kept for the noise cadence redraw; the epsilon explore branch it also
        # served was retired at v13.
        self.rng = rng
        # Multiplier on the perturbation this actor plays at, applied to the
        # *draw* rather than to the checkpoint's sigma. Scaling sigma in place --
        # which is how the per-actor spread used to arrive -- works only because
        # it happened exactly once at load; a second application squares it, and
        # nothing about the object says so.
        self.noise_scale = float(noise_scale)
        # Directed exploration: a per-action multiplier folded into the same draw,
        # so this actor plays `noise_scale` everywhere and `noise_scale * rare[a]`
        # on the rows `training/rare.py` says it has barely sampled. Resolved to a
        # vector once here rather than per turn.
        #
        # Collection-only, and structurally so: `agent()` below leaves the cadence
        # at "none", and a network that never draws noise cannot be steered by
        # scaling one. Nothing on the arena, gate or Kaggle paths can reach it.
        self.action_noise_scale = (
            None
            if action_noise_scale is None
            else self.noise_scale * np.asarray(action_noise_scale, dtype=np.float32)
        )
        # How often a NoisyNet weight perturbation is redrawn: "none", or one of
        # "decision" / "day" / "episode". The default is what makes evaluation
        # determinism structural rather than a promise -- the arena, the
        # viability gates, the promotion head-to-head and the Kaggle entry point
        # all reach this class through `agent()` below, which never sets it, so
        # nothing on any of those paths can draw noise even from a checkpoint
        # that carries sigma.
        self.noise_cadence = str(noise_cadence)
        self.recorder = recorder
        self.episode_steps = int(episode_steps)
        self.turns_per_day = int(turns_per_day)
        # The simulator's observation is a snapshot, so the only place a price
        # trend or a refused order can be remembered is here.
        self.history = MarketHistory()

    def _sample_noise(self, step: int, turns_per_day: int) -> None:
        """Redraw the network's weight perturbation, if this turn is due one.

        Cadence is a free parameter, not a cost trade: factorized noise costs
        `in + out` draws per layer rather than `in * out` -- 314 for the 186-wide
        head -- which is tens of microseconds against a simulator turn measured
        in milliseconds. "decision" matches how the TD loop resamples, so the
        behaviour and training distributions agree; "episode" is the setting that
        can express a coherent multi-turn plan, because at 720 decisions
        per-decision noise is near-independent across turns and cannot hold a
        seed purchase and its harvest inside one perturbed policy.
        """

        if self.noise_cadence == "none" or self.rng is None:
            return
        if self.network is None or not getattr(self.network, "is_noisy", False):
            return
        due = (
            self.noise_cadence == "decision"
            or (self.noise_cadence == "episode" and step == 0)
            or (self.noise_cadence == "day" and turns_per_day > 0 and step % turns_per_day == 0)
        )
        if due:
            scale = (
                self.noise_scale if self.action_noise_scale is None else self.action_noise_scale
            )
            self.network.sample_noise(self.rng, scale)

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

        if step == 0:
            _reset_donor_state()
            self.history.reset()
            if self.recorder is not None:
                self.recorder.reset()

        # Before the state is encoded, and before this turn's decision: the money
        # in `farm` has already been through last turn's market phase, so this is
        # the first moment last turn's order is known to have filled or been
        # refused. `training/clone.py` pushes at the same point in its own loop.
        self.history.observe(obs, farm)

        roles = market_influenced_roles(
            obs,
            farm,
            private,
            donor._role_plan(obs, farm),
        )
        field = donor._unit_actions(obs, configuration, farm, private, roles)

        # Every turn, not twice a day. The scorer decides at hours 0 and 12
        # because its action is a standing posture -- a modifier and a set of
        # reserves that the donor's executor then applies all day. A raw order
        # is not a posture: it is filled this turn at this turn's quote and is
        # gone. Deciding twice a day would leave the network unable to react to
        # a price it can see, and would hand it 60 samples an episode to learn
        # an unscaffolded policy from. It is also what makes one order per turn
        # sufficient throughput.
        orders: list[list] = []
        if self.network is not None:
            state = market_state_features(
                obs,
                farm,
                private,
                roles,
                field,
                config=configuration,
                episode_steps=episode_steps,
                turns_per_day=turns_per_day,
                history=self.history,
            )
            # Feasibility only: orders the simulator would refuse outright are
            # withheld from both the argmax and the exploring draw. It decides
            # nothing about which of the remaining orders is worth making.
            mask = feasible_actions(obs, farm, private, field, configuration)
            self._sample_noise(step, turns_per_day)
            decision = choose_order(state, self.network, mask)
            orders = decision.wire()
            self.history.record_decision(decision.order)
            if self.recorder is not None:
                self.recorder.record_decision(decision=decision)

        # Deliberately no `donor._schedule_market_adjustment` here. In the
        # scorer it is the donor's last word on the order list; carrying it over
        # would leave a piece of donor market judgement in an agent whose whole
        # premise is that there is none.
        if self.recorder is not None:
            self.recorder.record_turn(
                money=float(farm.get("money", 0) or 0),
                obs=obs,
                farm=farm,
                private=private,
            )

        return {"farmer": field["farmer"], "hands": field["hands"], "market": orders}


# Bound at import, while `sim.registry.resolve` still has KAGG_AGENT_CONFIG
# exported for *this* spec. Reading it any later is too late: resolve() restores
# the variable in a finally block, and `sim.runner.run_match` resolves both seats
# before either one plays. Reading it lazily makes every `?weights=` silently
# fall back to the packaged weights, so challenger and incumbent play as one
# policy and every paired promotion margin comes out exactly 0.0.
_IMPORT_CONFIG = os.environ.get("KAGG_AGENT_CONFIG")


def load_network(raw: str | None = _IMPORT_CONFIG) -> OrderQNetwork:
    """Trained weights if present, otherwise a silent net.

    A `weights` key in KAGG_AGENT_CONFIG overrides the default path -- that is
    how self-play and the promotion gate resolve a frozen snapshot as an
    opponent spec.

    The fallback scores every action equally, so the argmax takes index 0, which
    is NOOP. Unlike the scorer -- whose fallback is a strong donor policy -- the
    fallback here is an agent that never trades. That is the honest cold start
    for an unscaffolded market and it is deliberately weak.
    """

    if raw:
        config = json.loads(raw)
        override = config.get("weights")
        if override:
            return OrderQNetwork.load(override)
        if config.get("network_seed") is not None:
            return OrderQNetwork.random(seed=int(config["network_seed"]))
    if WEIGHTS_PATH.exists():
        try:
            return OrderQNetwork.load(WEIGHTS_PATH)
        # Every way a packaged file can be unusable: a stale schema or garbage
        # header raises ValueError, a truncated archive BadZipFile -- which
        # descends from Exception, not OSError -- an unreadable file OSError, and
        # an archive missing `w0` a KeyError. All of them mean the same thing
        # here: fall back to the silent network rather than take the whole
        # submission down. The `?weights=` override above deliberately still
        # raises, because a silent fallback there makes challenger and incumbent
        # play as one policy and every paired margin come out exactly 0.0.
        except (ValueError, OSError, KeyError, zipfile.BadZipFile):
            pass
    return OrderQNetwork.zeros()


_AGENT: MarketFreeAgent | None = None

#: Thrown turns tolerated in one episode before the guard below gives up.
#:
#: A genuine one-off should cost one turn, not the episode. A systematic
#: breakage -- a feature-schema change, say -- must not be recorded as a
#: legitimate loss, because `_arena_evaluate` counts seat errors and a
#: challenger that passes every turn would otherwise score terribly and look
#: like evidence about the policy.
MAX_SWALLOWED_ERRORS = 4
_ERRORS = 0


def agent(obs: Any, configuration: Any | None = None) -> dict[str, list]:
    global _AGENT, _ERRORS
    if _AGENT is None:
        _AGENT = MarketFreeAgent(load_network())
    if int(obs.get("step", 0) or 0) == 0:
        _ERRORS = 0
    try:
        return _AGENT(obs, configuration)
    except Exception:
        # Rollouts bypass this function entirely -- `match_agent` drives
        # `MarketFreeAgent` directly -- so training data is never collected
        # through the guard.
        _ERRORS += 1
        if _ERRORS > MAX_SWALLOWED_ERRORS:
            raise
        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0) or 0)
        hands = len(farms[player].get("hands", []) or []) if 0 <= player < len(farms) else 0
        return {
            "farmer": list(PASS_ACTION),
            "hands": [list(PASS_ACTION) for _ in range(hands)],
            "market": [],
        }


def __getattr__(name: str) -> Any:
    """Expose the league's post-episode hook without importing it for inference."""

    if name == "evaluate_episode":
        from agents.dqn_market_free.evaluation import evaluate_episode

        return evaluate_episode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
