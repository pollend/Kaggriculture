"""Donor field scheduler with market-directed planting and raw market orders.

There is no `market_exec` here, unlike `dqn_market_scorer`, which keeps the
donor's market executor and learns a modulation of it. Every turn the network
picks one order out of a fixed list, or nothing; `donor._market_actions` and
`donor._schedule_market_adjustment` are never called.

So the network has to learn from reward alone that it cannot spend money it does
not have, that animals starve without WHEAT, that stock held past the last day is
worthless, and that the field policy idles if nobody buys seed. The failure mode
is quiet -- a market policy that buys no seed scores near zero while every episode
completes without error -- so `training/collect.py` records the diagnostics that
separate that from a healthy run.

The donor still owns field jobs, routing, crop care, harvest and liquidation. Two
narrow adapters route market purchases into its role map: `planting` lets bought
seed replace vacant crop roles, and `herd` gives a bought goose a coop the donor's
COW/SHEEP-only planner would never assign. Live plants, and every animal the donor
does plan, are never rewritten.
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import Any

import numpy as np

from agents.dqn_market_free import donor
from agents.dqn_market_free.herd import goose_influenced_roles
from agents.dqn_market_free.market_state import MarketHistory, market_state_features
from agents.dqn_market_free.order_space import (
    choose_order,
    feasible_actions,
    seed_room,
    should_buy_land,
)
from agents.dqn_market_free.planting import market_influenced_roles
from agents.dqn_market_free.policy import WEIGHTS_PATH, OrderQNetwork

PASS_ACTION = ("PASS",)


def _reset_donor_state() -> None:
    """Clear the donor's module-level latches between episodes.

    The donor keeps cross-turn state in module globals. Rollouts run in a `spawn`
    pool, but a second episode in one interpreter would otherwise inherit a latched
    opponent signature.
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
        # Kept for the noise cadence redraw.
        self.rng = rng
        # Multiplier on the perturbation this actor plays at, applied to the *draw*
        # rather than to the checkpoint's sigma. Scaling sigma in place works only
        # if it happens exactly once at load; a second application squares it, and
        # nothing about the object says so.
        self.noise_scale = float(noise_scale)
        # Directed exploration: a per-action multiplier folded into the same draw,
        # resolved to a vector once here rather than per turn. Collection-only, and
        # structurally so -- `agent()` leaves the cadence at "none", and a network
        # that never draws noise cannot be steered by scaling one.
        self.action_noise_scale = (
            None
            if action_noise_scale is None
            else self.noise_scale * np.asarray(action_noise_scale, dtype=np.float32)
        )
        # How often a NoisyNet perturbation is redrawn: "none", or one of
        # "decision" / "day" / "episode". The default makes evaluation determinism
        # structural rather than a promise: the arena, the viability gates, the
        # promotion head-to-head and the Kaggle entry point all reach this class
        # through `agent()`, which never sets it, so none of them can draw noise
        # even from a checkpoint carrying sigma.
        self.noise_cadence = str(noise_cadence)
        self.recorder = recorder
        self.episode_steps = int(episode_steps)
        self.turns_per_day = int(turns_per_day)
        # The simulator's observation is a snapshot, so the only place a price
        # trend or a refused order can be remembered is here.
        self.history = MarketHistory()

    def _sample_noise(self, step: int, turns_per_day: int) -> None:
        """Redraw the network's weight perturbation, if this turn is due one."""

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

        # Goose first, seeds second. `goose_influenced_roles` may claim a reserved
        # animal slot the donor left bare, and the planting overlay only ever
        # rewrites `CROP` roles, so running it after cannot take the slot back --
        # the reverse order would let a seed claim ground a bought bird needs.
        roles = market_influenced_roles(
            obs,
            farm,
            private,
            goose_influenced_roles(obs, farm, private, donor._role_plan(obs, farm)),
        )
        field = donor._unit_actions(obs, configuration, farm, private, roles)

        # Every turn, not twice a day like the scorer: a raw order is filled at
        # this turn's quote and is gone, not a standing posture. Deciding twice a
        # day would leave the network unable to react to a price it can see, on 60
        # samples an episode.
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
            # `roles` is passed rather than recomputed: the land gate reads it for
            # the seed and hand tests, and it must be the same map the field
            # actions were built from.
            #
            # `plantable` is threaded for the same reason and is a net saving --
            # the deficit was being derived up to three times a turn. The mask, the
            # emission clamp and the land veto now read one vector, so they cannot
            # disagree about how much seed the plan can absorb.
            plantable = seed_room(obs, farm, private, roles)
            mask = feasible_actions(
                obs, farm, private, field, configuration, roles=roles, plantable=plantable
            )
            self._sample_noise(step, turns_per_day)
            decision = choose_order(
                state,
                self.network,
                mask,
                plantable=plantable,
                buy_land=should_buy_land(obs, farm, private, roles, configuration),
            )
            orders = decision.wire()
            self.history.record_decision(decision.order)
            if self.recorder is not None:
                self.recorder.record_decision(decision=decision)

        # Deliberately no `donor._schedule_market_adjustment` here: it would leave
        # a piece of donor market judgement in an agent whose premise is that there
        # is none.
        if self.recorder is not None:
            self.recorder.record_turn(
                money=float(farm.get("money", 0) or 0),
                obs=obs,
                farm=farm,
                private=private,
            )

        return {"farmer": field["farmer"], "hands": field["hands"], "market": orders}


# Bound at import, while `sim.registry.resolve` still has KAGG_AGENT_CONFIG
# exported for *this* spec. Reading it lazily is too late -- resolve() restores the
# variable in a finally block and both seats are resolved before either plays -- so
# every `?weights=` would silently fall back to the packaged weights and every
# paired promotion margin would come out exactly 0.0.
_IMPORT_CONFIG = os.environ.get("KAGG_AGENT_CONFIG")


def load_network(raw: str | None = _IMPORT_CONFIG) -> OrderQNetwork:
    """Trained weights if present, otherwise a silent net.

    A `weights` key in KAGG_AGENT_CONFIG overrides the default path; that is how
    self-play and the promotion gate resolve a frozen snapshot as an opponent.

    The fallback scores every action equally, so the argmax takes NOOP at index 0 --
    an agent that never trades. Deliberately weak, and unlike the scorer, whose
    fallback is a strong donor policy.
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
        # Every way a packaged file can be unusable, all meaning the same thing:
        # fall back to the silent network rather than take the submission down. The
        # `?weights=` override above deliberately still raises -- see `_IMPORT_CONFIG`.
        except (ValueError, OSError, KeyError, zipfile.BadZipFile):
            pass
    return OrderQNetwork.zeros()


_AGENT: MarketFreeAgent | None = None

#: Thrown turns tolerated in one episode before the guard below gives up.
#:
#: A one-off should cost one turn, not the episode. A systematic breakage must not
#: be recorded as a legitimate loss: `_arena_evaluate` counts seat errors, so a
#: challenger passing every turn would score terribly and read as evidence about
#: the policy.
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
