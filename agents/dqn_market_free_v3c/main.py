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

from agents.dqn_market_free_v3c import donor
from agents.dqn_market_free_v3c.herd import field_roles
from agents.dqn_market_free_v3c.market_state import (
    INTRA_TURN_SLICE,
    MarketHistory,
    intra_turn_block,
    market_state_features,
)
from agents.dqn_market_free_v3c.order_space import (
    LAND_ORDER,
    MAX_DECISIONS_PER_TURN,
    PendingTurn,
    choose_order,
    feasible_actions,
    seed_room,
    should_buy_land,
)
from agents.dqn_market_free_v3c.policy import WEIGHTS_PATH, OrderQNetwork

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


def _shed_unmoved(pending: PendingTurn, private: dict) -> bool:
    """Whether `pending.view` still shows the observation's own shed.

    The precondition for reusing the turn-opening encode, and it is not the
    tautology it looks like. `PendingTurn.shed` is the *permissive* view --
    `max(raw, donor._post_field_storage(...))`, `order_space.py` -- so the moment
    the field plan has a unit DROP produce this turn, `pending.view` differs from
    the untouched observation before a single order has been applied.

    That divergence is the whole reason this test exists. `feasible_actions` sizes
    its SELL rows off the same post-field shed, so a first sub-decision scored
    against the raw one would rank orders under a state the mask disagrees with --
    the split-brain `PendingTurn.view`'s docstring says the encoder exists to
    prevent. Six columns move: the per-product shed column, the shed-load pair, and
    three block-C constraint columns downstream of `shed_total`.

    Compared by value over the union of both key sets rather than with `==`, so a
    zero-count key present on one side only is not mistaken for a delivery.
    """

    raw = private.get("shed", {}) or {}
    return all(
        int(pending.shed.get(item, 0) or 0) == int(raw.get(item, 0) or 0)
        for item in {*pending.shed, *raw}
    )


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
        # The simulator's observation is a snapshot. v2 remembered the past here,
        # in a hand-built 24-turn window; v3 remembers it in the network's hidden
        # state instead -- with one exception this fork exists for. The three
        # refusal latches are read again (`market_state.refusal_block`), so
        # `observe` is load-bearing twice over: it writes those latches, and it
        # marks the point in the turn where last turn's order has settled, which
        # is what the tick below sites against and what `clone.py` reproduces.
        self.history = MarketHistory()
        #: The previous turn's finished `PendingTurn`, or None on the first turn.
        #:
        #: This is part of what replaces block D. The GRU is fed the same 280-column vector
        #: the head sees, encoded once at the top of the turn -- and block E of that
        #: encoding is built from *this*, not from the empty turn just starting. So
        #: the recurrence sees how hard last turn traded, which ops it queued, and
        #: which way its money went, and can pair that against the money this turn
        #: opens with to work out whether the order filled. Leave it None and the
        #: GRU gets eleven dead zeros where the only cross-turn signal should be.
        self.last_turn = None
        # Not only at `step == 0`: `agent()` keeps a module-level `_AGENT` that
        # survives across episodes inside one interpreter, so a network built once
        # and never reset would carry the previous episode's memory into the next.
        if self.network is not None:
            self.network.reset_hidden()

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
            self.last_turn = None
            if self.network is not None:
                self.network.reset_hidden()
            if self.recorder is not None:
                self.recorder.reset()

        # Before the state is encoded, and before this turn's decision: the money
        # in `farm` has already been through last turn's market phase, so this is
        # the first moment last turn's order is known to have filled or been
        # refused. `training/clone.py` pushes at the same point in its own loop.
        self.history.observe(obs, farm)

        # One composer, shared with `clone.py` and the mask's fallbacks, so no two
        # callers can disagree about which tiles the herd holds.
        roles = field_roles(obs, farm, private)
        field = donor._unit_actions(obs, configuration, farm, private, roles)

        # Every turn, not twice a day like the scorer: a raw order is filled at
        # this turn's quote and is gone, not a standing posture. Deciding twice a
        # day would leave the network unable to react to a price it can see, on 60
        # samples an episode.
        orders: list[list] = []
        if self.network is not None:
            # `roles` is passed rather than recomputed: the land gate reads it for
            # the seed and hand tests, and it must be the same map the field
            # actions were built from.
            #
            # `plantable` is threaded for the same reason and is a net saving --
            # the deficit was being derived up to three times a turn. The mask, the
            # emission clamp and the land veto now read one vector, so they cannot
            # disagree about how much seed the plan can absorb.
            plantable = seed_room(obs, farm, private, roles)
            pending = PendingTurn(
                obs, farm, private, field, configuration, plantable=plantable, roles=roles
            )

            # --- the recurrent tick, once per turn ---------------------------
            #
            # Position matters three ways and all three are pinned by
            # `test_cloned_state_matches_the_live_agent`, because `clone.py` has to
            # reproduce this walk exactly from a stored replay:
            #
            #   * after `history.observe`, so the turn's money has already settled
            #     last turn's order;
            #   * before the decision loop, so every sub-decision of this turn
            #     scores against the same latent -- the turn is the unit the
            #     reward-to-go is indexed by;
            #   * on the untouched `obs/farm/private`, not on `pending.view`, so the
            #     recurrence sees the turn as it opened rather than as its own
            #     queued orders left it.
            #
            # `turn=self.last_turn` is the part that is easy to read as a mistake
            # and is not: the *previous* turn's finished `PendingTurn` fills block
            # E. See `self.last_turn`.
            turn_state = market_state_features(
                obs,
                farm,
                private,
                roles,
                field,
                config=configuration,
                episode_steps=episode_steps,
                turns_per_day=turns_per_day,
                history=self.history,
                turn=self.last_turn,
            )
            hidden_in = self.network.hidden
            self.network.tick(turn_state)
            if self.recorder is not None:
                # Same call site as the tick, deliberately on the following line:
                # the invariant the whole sequence sampler rests on is that the
                # recorder holds exactly one turn row per reward slot, and the
                # cheapest way to keep that true is to make the two impossible to
                # separate when someone edits this block.
                self.recorder.record_turn_input(turn_state, hidden_in)

            # Noise is drawn once per turn, not once per sub-decision. NoisyNet is a
            # perturbation of the *policy*, so redrawing it inside the turn would
            # make the second order the answer of a different policy than the first
            # -- and the turn is the unit the reward is paid on.
            self._sample_noise(step, turns_per_day)

            # Land decided first and off the head, exactly as in v1 (see
            # `LAND_BUY_SET_POINT`), but *emitted* last -- see the append below the
            # decision loop. It costs one wire slot instead of the whole turn, so the
            # rest of the loop still gets to trade against a wallet the quadrant has
            # left, and that is still true with the emission moved: `commit_land`
            # takes the price out of `pending` here, where the decision is made,
            # rather than where the row reaches the wire.
            # It is not a recorded decision here, which is the one place v2 departs
            # from v1's bookkeeping rather than extending it. v1 had to store a row
            # for the land turn -- NOOP against the real state -- because the turn
            # held no other decision. Here the turn goes on to decide orders whose
            # own rows carry `pending.land` in block E, so the network sees the
            # quadrant in the state it is actually credited with, and no row has to
            # be minted for a choice the network did not make.
            buy_land = should_buy_land(obs, farm, private, roles, configuration)
            if buy_land:
                pending.commit_land()
                self.history.record_decision(LAND_ORDER)

            # Up to `MAX_DECISIONS_PER_TURN` orders, each priced against what the
            # earlier ones already committed. NOOP terminates: the turn is finished,
            # and the row is a real choice the learner is credited with rather than a
            # loop that ran out.
            # Whether the turn-opening encode may stand in for the first
            # sub-decision's. See `_shed_unmoved`: `pending.view` is the identity
            # only when the field plan delivers nothing into the shed, and the land
            # commit above moves the wallet.
            reuse_turn_state = not pending.land and _shed_unmoved(pending, private)
            for index in range(MAX_DECISIONS_PER_TURN):
                if not pending.slots_left:
                    break
                if index == 0 and reuse_turn_state:
                    # `pending.view` is the identity here -- no `apply` has run, no
                    # land was committed, and the field plan moved nothing into the
                    # shed -- so the turn's first sub-decision sees exactly the
                    # vector already encoded above in every column except block E,
                    # which is the only one that reads `turn`.
                    #
                    # Reusing it is not a micro-optimisation. A second full encode
                    # costs a `projected_supply_map`, which is ~0.54s of a ~2.9s
                    # episode, and the common turn holds one decision -- so without
                    # this, adding the recurrence would double the encoder bill on
                    # the majority of turns while the matmuls it was added for cost
                    # microseconds. With it, v3 pays what v2 paid.
                    #
                    # `test_the_reused_turn_state_equals_a_fresh_encode` encodes
                    # both ways and asserts exact equality, on a turn that delivers
                    # produce as well as one that does not.
                    state = turn_state.copy()
                    state[INTRA_TURN_SLICE] = intra_turn_block(pending)
                else:
                    # The world as this turn's queued orders have already left it.
                    # Block E alone would say *that* a sale is queued; the view is
                    # what makes the shed, the wallet and the quote say what it was
                    # a sale *of*.
                    obs_view, farm_view, private_view = pending.view(obs, farm, private)
                    state = market_state_features(
                        obs_view,
                        farm_view,
                        private_view,
                        roles,
                        field,
                        config=configuration,
                        episode_steps=episode_steps,
                        turns_per_day=turns_per_day,
                        history=self.history,
                        turn=pending,
                    )
                mask = feasible_actions(
                    obs,
                    farm,
                    private,
                    field,
                    configuration,
                    roles=roles,
                    plantable=pending.seed_room,
                    pending=pending,
                )
                decision = choose_order(
                    state,
                    self.network,
                    mask,
                    plantable=pending.seed_room,
                )
                if self.recorder is not None:
                    self.recorder.record_decision(decision=decision)
                if decision.order.op == "NOOP":
                    break
                orders += decision.wire()
                pending.apply(decision.order)
                # The cross-turn history block wants the turn's *last* order, which
                # is what `record_decision` keeps, so every accepted order pushes.
                self.history.record_decision(decision.order)

            # Emitted after the head's rows so the simulator resolves this turn's
            # SELLs before the quadrant's debit. The wire is applied in list order,
            # so land placed first could only ever be funded by money already banked
            # -- and `should_buy_land` prices the purchase against `liquid_cash`,
            # which counts the shed the head is *about to* sell. Placing the row
            # first while testing it against that number is the one combination that
            # gets the order rejected by the simulator rather than by the gate.
            #
            # `pending.money` is the guard on the other half of that bargain, and it
            # is not optional. `liquid_cash` assumes the shed gets sold; the head is
            # under no obligation to sell it, and when it does not, the gate has
            # bought a quadrant with money that never arrived. Because `commit_land`
            # already took the price out above, the projected end-of-turn wallet
            # going negative is exactly the case where the sells did not cover it --
            # measured at 1.0 rejected emission per episode without this test, each
            # one burning a market slot out of ten for an order that cannot resolve.
            if buy_land and pending.money >= 0.0:
                orders += LAND_ORDER.to_wire()

            if self.recorder is not None:
                self.recorder.record_events(pending.events)

            # Block E of the *next* turn's recurrent input. Set last, after the
            # loop has finished committing, so it describes the turn as it ended.
            self.last_turn = pending

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
        from agents.dqn_market_free_v3c.evaluation import evaluate_episode

        return evaluate_episode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
