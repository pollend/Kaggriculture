"""Roll out episodes and record the raw order decisions as compact training data.

Transitions cannot be reconstructed from `env.steps` after the fact -- the replay
stores filled orders, not the action set the policy chose from -- so the acting
agent records its own decisions and this module owns that recorder.

A decision here is one index into a fixed list, so it is fully described by the
state vector and a single integer. That is what makes a per-turn rate affordable:
720 decisions an episode costs less memory than the scorer's 60, which have to
store a whole rebuilt candidate grid each.

The reward stream and its scales live in `training/reward.py`.
"""

from __future__ import annotations

import traceback
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import numpy as np

from agents.dqn_market_free_v3c import lots
from agents.dqn_market_free_v3c.diagnostics import OrderDiagnostics, diagnose_order_lists
from agents.dqn_market_free_v3c.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free_v3c.market_state import ORDER_LOOP_CAP
from agents.dqn_market_free_v3c.order_space import (
    ACTION_COUNT,
    ACTIONS,
    LAND_ORDER,
    OrderDecision,
)
from agents.dqn_market_free_v3c.policy import HIDDEN_SIZE as GRU_HIDDEN
from agents.dqn_market_free_v3c.training.reward import episode_rewards
from sim.workers import (
    MAP_TIMEOUT_SECONDS,
    discard_pool,
    single_threaded_workers,
    worker_pool,
)

if TYPE_CHECKING:
    from sim.runner import MatchResult


#: Turns within which a sale of an item bought earlier still counts as a round trip.
#:
#: One in-game day (`donor.TURNS_PER_DAY` is 24), which is also
#: `townCenterSellInterval`: a purchase and the sale that closes it inside a day are
#: one decision, and beyond it the two stop being the same trade. Measured over 20 stored seats, 57% of resold
#: wheat units and 89% of resold fertilizer clear inside this window.
LAGGED_WASH_WINDOW = 24


@dataclass
class EpisodeRecord:
    """One episode: the per-turn reward stream and the per-turn order decisions."""

    reward: np.ndarray  # float32 [T]    shaped, ready for training
    raw_reward: np.ndarray  # float32 [T]    unshaped money delta, for diagnostics
    money: np.ndarray  # float32 [T+1]
    order_state: np.ndarray  # float32 [D, STATE_DIM]
    order_chosen: np.ndarray  # int16   [D]  index into ACTIONS
    order_step: np.ndarray  # int16   [D]  index into `reward` each decision is paid from
    #: bool [D, ACTION_COUNT]. Which actions the simulator would have accepted at
    #: each decision. Stored rather than recomputed from `order_state`, which would
    #: tie the mask to the exact scaling of six `market_state` columns and corrupt
    #: the Double-DQN target silently if either drifted. Compresses to almost
    #: nothing beside `order_state`.
    order_mask: np.ndarray

    #: float32 [T, STATE_DIM]. The GRU's input on each turn, indexed exactly like
    #: `reward` -- the same compacted index `order_step` points into, and for the
    #: same reason spelled out in `Recorder.record_decision`. A turn the agent
    #: skipped contributes no reward slot and must contribute no tick either, or
    #: every chunk after it carries a hidden state one turn out of phase with the
    #: reward it is being trained against. Nothing downstream can detect that: the
    #: sampler clamps at the episode end and the loss looks entirely healthy.
    turn_state: np.ndarray

    #: float16 [ceil(T / hidden_stride), GRU_HIDDEN]. The hidden state *fed into*
    #: turn `k * hidden_stride`, as the acting policy produced it -- R2D2's stored
    #: state. Row 0 is exactly zero by construction, so a chunk starting at the
    #: episode's first turn carries no staleness at all.
    #:
    #: Stored at a stride rather than every turn, and not to save memory: every
    #: turn in fp16 is only 184 KB an episode. A chunk may only *start* where a
    #: hidden state was stored, so storing at turns no chunk can begin at is dead
    #: weight, and constraining starts to a grid is what lets the sampler take a
    #: plain integer draw. fp16 is deliberate too: if three decimal digits of
    #: rounding survived the burn-in, the burn-in would be too short -- which is
    #: exactly what `hidden_recovery_l2` exists to detect, and storing at full
    #: precision would hide it.
    turn_hidden: np.ndarray

    #: int16 [1]. The stride `turn_hidden` was written at. Stored rather than
    #: assumed: a `hidden_stride` change between the session that wrote an artifact
    #: and the one reading it misaligns every chunk start, so `load_episode_batch`
    #: fails on it the way it already fails on a state width.
    hidden_stride: np.ndarray

    #: Ephemeral training-only multiplier. Not serialized in `EPISODE_ARRAY_FIELDS`:
    #: it is derived from the batch/generation metadata of the session currently
    #: being loaded, so old artifacts remain byte-compatible and a different run can
    #: choose a different weighting law without rewriting rollout files.
    episode_weight: float = 1.0

    @property
    def steps(self) -> int:
        return self.reward.shape[0]

    @property
    def decisions(self) -> int:
        return self.order_step.shape[0]

    @property
    def orders_placed(self) -> int:
        """Trading decisions across the episode. The headline health diagnostic.

        An unscaffolded market policy has two degenerate attractors and this
        number separates them: zero means it collapsed to NOOP, and something
        near `decisions` means it is ordering every single turn and is almost
        certainly bankrupt.

        Counted in decisions rather than wire orders, so a `HIRE 8` counts once.
        """

        return int(np.count_nonzero(self.order_chosen))

    @property
    def order_diagnostics(self) -> OrderDiagnostics:
        """Decision and order counts for the orders actually emitted.

        Grouped by *turn*, not by sub-decision, which is what keeps this number
        comparable with v1 and with the gate that reads it: a v2 turn holding a sale,
        a seed buy and a terminator is one decision that emitted two orders, exactly
        as an opponent emitting the same list would be counted by
        `diagnose_order_lists`. Counting sub-decisions instead would divide by the
        terminator and report a policy trading well as barely trading --
        `viability_min_orders_per_decision` sits on this.
        """

        by_turn: dict[int, list] = {}
        for choice, step in zip(self.order_chosen, self.order_step):
            by_turn.setdefault(int(step), []).extend(ACTIONS[int(choice)].to_wire())
        turns = sorted(by_turn)
        return diagnose_order_lists(
            [by_turn[turn] for turn in turns],
            [float(self.raw_reward[turn]) for turn in turns],
        )

    @property
    def decisions_per_turn(self) -> float:
        """Sub-decisions per turn, terminator included.

        v2's health number, and the one that separates its two degenerate
        attractors: 1.0 is the policy that stops before ordering anything, and the
        loop cap is the policy stuffing the wire. Neither is visible in
        `orders_placed`, which counts the same either way.

        **Read `decisions_per_turn_histogram` beside it, never this alone.** v2
        finished its run reporting 5.71 here, which sounds like the middle of the
        range and describes no turn it ever played: the real distribution was 40%
        on 1 and 47% on the cap. A mean over a bimodal split reports the empty
        space between the two modes.
        """

        turns = len(set(int(step) for step in self.order_step))
        return float(self.decisions) / float(max(1, turns))

    @property
    def wash_turns(self) -> int:
        """Turns that both bought and sold the same item.

        The zero-P&L round trip v2 converged on: buying an item walks the market
        inventory down and selling it back walks the same price levels up, so the
        pair nets exactly nothing at every quantity while consuming the wire.

        `order_space.feasible_actions` shuts it structurally now, so this is a
        guard rather than a dial -- it must read zero, and anything else means the
        mask has a hole. Counted from the recorded actions rather than from the
        mask, so the two are independent: a counter derived from the thing it is
        checking would agree with it by construction.
        """

        if not self.decisions:
            return 0
        washes = 0
        start = 0
        steps = self.order_step
        for index in range(1, len(steps) + 1):
            if index < len(steps) and steps[index] == steps[start]:
                continue
            sold: set[str] = set()
            bought: set[str] = set()
            for choice in self.order_chosen[start:index]:
                action = ACTIONS[int(choice)]
                if action.item is None:
                    continue
                if action.op == "SELL":
                    sold.add(str(action.item))
                elif action.op == "BUY_PRODUCT":
                    bought.add(str(action.item))
            if sold & bought:
                washes += 1
            start = index
        return washes

    @property
    def lagged_wash_turns(self) -> int:
        """Turns that sold an item bought within `LAGGED_WASH_WINDOW` turns before.

        **The dial `wash_turns` is not.** That property counts the *same-turn* round
        trip and reads zero because `order_space.feasible_actions` masks it, which is
        exactly what it is there to confirm. What it cannot see is the policy placing
        the two legs on different turns, and measurement over 20 stored
        `market-free-splitladder-v1` seats says that is what happened: 0.0% of resold
        units at lag 0, 17.3% of wheat and 49.8% of fertilizer at lag 1, 57% and 89%
        within one in-game day. The mask held and the behaviour walked around it.

        A window rather than the whole episode, because a purchase held for a fortnight
        and then sold is inventory management and not a wash. One in-game day, because
        a round trip that clears inside a day is one decision.

        Counted per turn, not per unit, and from the recorded actions rather than the
        mask -- a counter derived from the thing it checks would agree with it by
        construction. The reward prices the trip at the coins it moved and says
        nothing about whether it was a wash; this is how a run watches for one.
        """

        if not self.decisions:
            return 0
        #: Last turn each item was bought on, so the check is O(1) per sale.
        last_buy: dict[str, int] = {}
        washes = 0
        start = 0
        steps = self.order_step
        for index in range(1, len(steps) + 1):
            if index < len(steps) and steps[index] == steps[start]:
                continue
            turn = int(steps[start])
            sold: set[str] = set()
            bought: set[str] = set()
            for choice in self.order_chosen[start:index]:
                action = ACTIONS[int(choice)]
                if action.item is None:
                    continue
                if action.op == "SELL":
                    sold.add(str(action.item))
                elif action.op == "BUY_PRODUCT":
                    bought.add(str(action.item))
            # Sales scored against buys strictly *before* this turn: a same-turn pair
            # is `wash_turns`' business and double-counting it here would make the two
            # columns move together and mean nothing apart.
            if any(turn - last_buy.get(item, -LAGGED_WASH_WINDOW - 1) <= LAGGED_WASH_WINDOW
                   for item in sold):
                washes += 1
            for item in bought:
                last_buy[item] = turn
            start = index
        return washes

    @property
    def decisions_per_turn_histogram(self) -> list[int]:
        """Turns holding exactly 1, 2, ... `ORDER_LOOP_CAP` sub-decisions.

        The shape, not the mean, is what says whether the multi-order MDP is being
        used or merely survived. Both of v2's attractors are ends of this array and
        both are invisible in the average of it.
        """

        if not self.decisions:
            return [0] * ORDER_LOOP_CAP
        _turns, counts = np.unique(self.order_step, return_counts=True)
        histogram = np.bincount(np.clip(counts, 1, ORDER_LOOP_CAP), minlength=ORDER_LOOP_CAP + 1)
        return [int(value) for value in histogram[1:]]

    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in EPISODE_ARRAY_FIELDS)


EPISODE_ARRAY_FIELDS = (
    "reward",
    "raw_reward",
    "money",
    "order_state",
    "order_chosen",
    "order_step",
    "order_mask",
    "turn_state",
    "turn_hidden",
    "hidden_stride",
)

#: Turns between stored hidden states, and therefore the grid a chunk may start
#: on. `training/replay.py` requires `burn_in % HIDDEN_STRIDE == 0` so that
#: `train_start - burn_in` lands on a stored row, and validates it rather than
#: assuming it.
#:
#: Half the training window is R2D2's 50% sequence overlap, so a turn in the
#: interior is covered by exactly two chunks and no turn is over- or under-trained
#: against its neighbours. Two caveats worth stating, because both are silent:
#:
#:   * The ends are not interior. Turns `0 .. HIDDEN_STRIDE - 1` are reachable from
#:     one chunk start only, and the last `(T - train) % HIDDEN_STRIDE` turns of an
#:     episode are reachable from none -- an episode of 719 turns at `train = 40`
#:     never puts a gradient row on turns 700-718. That matters most for cloned
#:     demonstrations, where `T` is `len(steps) - 1` and `reward.py` puts the whole
#:     terminal money lump on the last turn; a live 720-turn episode divides evenly.
#:   * "Half the training window" is a fact about `train_turns = 40`, and
#:     `train_turns` is a config key while this is a module constant baked into
#:     every stored artifact. Change one without the other and the overlap silently
#:     stops being 50%; only `burn_in % HIDDEN_STRIDE` is actually enforced.
HIDDEN_STRIDE = 20


def save_episode_batch(records: list[EpisodeRecord], path: str | Path) -> Path:
    """Persist variable-width episode records without pickle/object arrays."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "episode_count": np.asarray([len(records)], dtype=np.int32),
    }
    for index, record in enumerate(records):
        for name in EPISODE_ARRAY_FIELDS:
            payload[f"e{index}_{name}"] = getattr(record, name)
    np.savez_compressed(destination, **payload)
    return destination


def load_episode_batch(path: str | Path) -> list[EpisodeRecord]:
    """Read a rollout artifact written by this run.

    Rollout artifacts do not outlive the session that wrote them, so there is no
    format negotiation here -- only a width check, so an artifact left over from
    a differently-shaped state encoder fails on the spot rather than training
    against misaligned columns.
    """

    with np.load(Path(path), allow_pickle=False) as payload:
        count = int(payload["episode_count"][0])
        records = []
        for index in range(count):
            try:
                values = {
                    name: payload[f"e{index}_{name}"].copy() for name in EPISODE_ARRAY_FIELDS
                }
            except KeyError as missing:
                raise ValueError(
                    f"{path}: episode {index} is missing {missing}, so it predates a "
                    "schema change. Rollout artifacts do not outlive the session that "
                    "wrote them -- start a new training instance rather than reading it."
                ) from missing
            width = values["order_state"].shape[-1]
            if width != STATE_DIM:
                raise ValueError(
                    f"{path}: episode {index} has a {width}-wide state, expected {STATE_DIM}"
                )
            actions = values["order_mask"].shape[-1]
            if actions != ACTION_COUNT:
                raise ValueError(
                    f"{path}: episode {index} has a {actions}-wide action mask, "
                    f"expected {ACTION_COUNT}"
                )
            turn_width = values["turn_state"].shape[-1]
            if turn_width != STATE_DIM:
                raise ValueError(
                    f"{path}: episode {index} has a {turn_width}-wide turn state, "
                    f"expected {STATE_DIM}"
                )
            latent = values["turn_hidden"].shape[-1]
            if latent != GRU_HIDDEN:
                raise ValueError(
                    f"{path}: episode {index} stored a {latent}-wide hidden state, "
                    f"expected {GRU_HIDDEN}"
                )
            stride = int(values["hidden_stride"][0])
            if stride != HIDDEN_STRIDE:
                raise ValueError(
                    f"{path}: episode {index} stored a hidden state every {stride} turns "
                    f"and this session chunks at {HIDDEN_STRIDE}; every chunk start "
                    "would be misaligned with the state it is supposed to resume from"
                )
            records.append(EpisodeRecord(**values))
        return records


@dataclass
class Recorder:
    """Accumulates one episode of decisions as the agent makes them."""

    money: list[float] = field(default_factory=list)
    # The land potential's two inputs, indexed exactly like `money` because the
    # reward stream is built from that list and `order_step` is a compacted index
    # into it. Kept as parallel lists rather than folded into one tuple list so a
    # trace that goes missing fails at `episode_rewards`' length check rather
    # than silently pairing a quadrant count with the wrong day.
    quadrants: list[int] = field(default_factory=list)
    days: list[int] = field(default_factory=list)
    # The lot potential's three inputs, counts rather than dicts over
    # `lots.CROP_ITEMS` and `lots.STOCK_ITEMS`. `standing` is the one not in
    # `private`: without the tile count `PHI` falls on the turn the agent plants,
    # charging for the action the term exists to make cheaper.
    seeds: list[list[int]] = field(default_factory=list)
    standing: list[list[int]] = field(default_factory=list)
    stock: list[list[int]] = field(default_factory=list)
    # The animal legs' two inputs, over `lots.ANIMAL_ITEMS`. `owned` counts an animal
    # wherever it is, because all three places are one purchase; `produce_room` is what
    # the season can still get out of the herd, which is what bounds an animal lot
    # where `max_yield` bounds a seed one.
    owned: list[list[int]] = field(default_factory=list)
    produce_room: list[list[int]] = field(default_factory=list)
    order_steps: list[int] = field(default_factory=list)
    decisions: list[OrderDecision] = field(default_factory=list)
    #: Per-turn order events from `PendingTurn.events`, indexed like `money`.
    events: list[list] = field(default_factory=list)
    #: The GRU's input each turn and the hidden state it was fed, both indexed
    #: like `money`. Pushed together by `record_turn_input`.
    turn_inputs: list[np.ndarray] = field(default_factory=list)
    turn_hiddens: list[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self.money.clear()
        self.quadrants.clear()
        self.days.clear()
        self.seeds.clear()
        self.standing.clear()
        self.stock.clear()
        self.owned.clear()
        self.produce_room.clear()
        self.order_steps.clear()
        self.decisions.clear()
        self.events.clear()
        self.turn_inputs.clear()
        self.turn_hiddens.clear()

    def record_decision(self, *, decision: OrderDecision) -> None:
        """Store a decision against the reward index it will be paid from.

        The simulator's turn number is deliberately **not** what lands in
        `order_step`. The reward stream is built from `self.money`, and the agent
        skips both on a malformed observation, so a turn the simulator counted may
        contribute no reward. Indexing by simulator step then slides every later
        decision's n-step window forward *silently*, because `replay.sample` clamps
        with `min(end_step, record.steps)`. `clone.py` records the same compacted
        index; the two record kinds share one buffer and must agree on it.

        `record_decision` runs before `record_turn`, so `len(self.money)` is exactly
        the slot this turn's reward will take.
        """

        self.order_steps.append(len(self.money))
        self.decisions.append(decision)

    def record_turn_input(self, turn_input, hidden) -> None:
        """The GRU's input for this turn, and the state it was fed.

        Indexed exactly like `money`, and for the identical reason spelled out in
        `record_decision`: the reward stream is built from `self.money`, the agent
        skips a turn on a malformed observation, and a hidden state stored against
        the simulator's turn number would slide every later chunk one turn out of
        phase with the reward it trains against -- silently, because the sampler
        clamps at the episode end and never sees the shift.

        Called from `main.py` on the line after the tick itself. `finish` asserts
        the two lists came out the same length as `money`, which is the only thing
        standing between a desynchronised recurrence and a training run that looks
        completely healthy.
        """

        # Copied, not referenced. `OrderQNetwork.hidden` hands back the live array
        # and `np.asarray` on an already-float32 array returns that same object, so
        # this is only safe today because `GruCell.step` rebinds rather than writing
        # in place. An `out=` buffer added there later would retroactively rewrite
        # every hidden state already sitting in this list.
        self.turn_inputs.append(np.array(turn_input, dtype=np.float32, copy=True))
        self.turn_hiddens.append(np.array(hidden, dtype=np.float32, copy=True))

    def record_events(self, events) -> None:
        """The turn's queued orders, as `(op, item, quantity, cash)`.

        `lots.match` splits a turn back into its orders with these. One call per
        turn, from the agent, alongside `record_turn`.
        """

        self.events.append(list(events))

    def record_turn(self, *, money: float, obs: Any, farm: dict, private: dict) -> None:
        self.money.append(float(money))
        # Every trace comes off one `lots.sample`, shared with `clone` and
        # `diagnostics`: a cloned episode and a self-collected one are only the same
        # kind of sample if the trace is built the same way, and the three copies this
        # replaced had already drifted on the `day` fallback.
        turn = len(self.money) - 1
        held = lots.sample(obs if isinstance(obs, dict) else {}, farm, private, turn)
        self.quadrants.append(held.quadrants)
        self.days.append(held.day)
        self.seeds.append(held.seeds)
        self.standing.append(held.standing)
        self.stock.append(held.stock)
        self.owned.append(held.owned)
        self.produce_room.append(held.produce_room)

    def finish(self, final_money: float, gamma: float) -> EpisodeRecord:
        steps = len(self.money)
        if steps == 0:
            raise ValueError("recorder captured no steps")
        if not self.decisions:
            raise ValueError("recorder captured no order decisions")

        # Which turns put an order on the wire, indexed the same way the reward
        # stream is. A turn that emitted one and moved no money had it refused.
        emitted = [False] * len(self.money)
        # The op each turn chose, indexed the same way. The reward no longer reads it
        # -- `diagnostics` does, to tell a BUY_PRODUCT from a harvest that landed on a
        # spending turn -- and `ACTIONS[chosen].op` rather than the wire row is the
        # *mapped* action the learner actually saw, which is what `clone` records too.
        ops: list[str | None] = [None] * len(self.money)
        #: Sub-decisions the turn took, terminator included. The health diagnostic
        #: that separates v2's two new failure modes -- instant STOP at 1.0, slot
        #: stuffing at the cap -- and it is free here.
        turn_decisions = [0] * len(self.money)
        for step, decision in zip(self.order_steps, self.decisions):
            if not 0 <= step < len(emitted):
                continue
            turn_decisions[step] += 1
            # `buy_land`, not just `chosen`. A land turn puts `[["BUY_LAND"]]` on the
            # wire while `chosen` stays NOOP, because `OrderDecision` splits the two
            # so the learner is credited with the row it actually contributed --
            # which, for a decision `should_buy_land` made without consulting the
            # network, is nothing. `emitted` is not credit: it is whether an order
            # went out. Reading `chosen` alone charged `IDLE_PENALTY` on every turn
            # the agent spent a quadrant's price, and told `order_refused` to skip a
            # turn that did emit.
            if decision.chosen or decision.buy_land:
                emitted[step] = True
            # The wire op on a land turn, for the same reason. `lots.match` reads this
            # only through `_is_bought_stock`'s `op == "BUY_PRODUCT"` test, so the
            # label cannot open or close a lot either way -- but a turn that moved a
            # quadrant's price is not a NOOP, and the trace should not say it was.
            op = LAND_ORDER.op if decision.buy_land else ACTIONS[int(decision.chosen)].op
            # The terminator does not overwrite the op the turn actually traded on:
            # every v2 turn ends on a NOOP, so keeping it here would label every
            # trading turn a NOOP and blind every lot diagnostic to all of them.
            if op != "NOOP" or ops[step] is None:
                ops[step] = op

        shaped, raw, money = episode_rewards(
            self.money,
            final_money,
            gamma,
            emitted,
            seeds=self.seeds,
            standing=self.standing,
            stock=self.stock,
            owned=self.owned,
            basis_required=True,
        )

        # One turn row per reward slot, or the recurrence is out of phase with the
        # reward stream and every chunk drawn from this episode trains a hidden
        # state against the wrong turn's return. A hard failure, not a warning:
        # the same class of error `episode_rewards`' own length check already
        # refuses, and nothing downstream can detect it.
        turns = len(shaped)
        if len(self.turn_inputs) != turns or len(self.turn_hiddens) != turns:
            raise ValueError(
                f"recorder holds {len(self.turn_inputs)} turn inputs and "
                f"{len(self.turn_hiddens)} hidden states for {turns} reward steps; "
                "the GRU tick and `record_turn` are out of phase"
            )
        turn_state = (
            np.asarray(self.turn_inputs, dtype=np.float32)
            if turns
            else np.zeros((0, STATE_DIM), dtype=np.float32)
        )
        # Only the rows a chunk may start at. `[::HIDDEN_STRIDE]` keeps row 0,
        # which is the zero state every episode opens on.
        turn_hidden = (
            np.asarray(self.turn_hiddens[::HIDDEN_STRIDE], dtype=np.float16)
            if turns
            else np.zeros((0, GRU_HIDDEN), dtype=np.float16)
        )

        count = len(self.decisions)
        order_state = np.zeros((count, STATE_DIM), dtype=np.float32)
        order_chosen = np.zeros(count, dtype=np.int16)
        order_mask = np.ones((count, ACTION_COUNT), dtype=np.bool_)
        for index, decision in enumerate(self.decisions):
            order_state[index] = decision.state
            order_chosen[index] = decision.chosen
            if decision.mask is not None:
                order_mask[index] = decision.mask

        return EpisodeRecord(
            reward=shaped.astype(np.float32),
            raw_reward=raw.astype(np.float32),
            money=money,
            order_state=order_state,
            order_chosen=order_chosen,
            order_step=np.asarray(self.order_steps, dtype=np.int16),
            order_mask=order_mask,
            turn_state=turn_state,
            turn_hidden=turn_hidden,
            hidden_stride=np.asarray([HIDDEN_STRIDE], dtype=np.int16),
        )


#: Spec target of the training-only wrapper in `match_agent.py`.
LEARNER_ENTRY_POINT = "agents.dqn_market_free_v3c.match_agent:make_agent"

#: Spec target of this agent's ordinary inference entry point.
#:
#: Here rather than in `training/updater`, where it used to live, because the
#: mirror gate on the opponent-relative reward leg has to read it and `updater`
#: already imports this module. Re-exported there under the same name, so nothing
#: that referred to `updater.SELF_SPEC` has to move.
SELF_SPEC = "agents/dqn_market_free_v3c/main.py"


@dataclass(frozen=True)
class OrderRollout:
    """One recorded rollout episode plus the match the league will store."""

    record: EpisodeRecord
    opponent_spec: str
    learner_seat: int
    match: MatchResult
    seed: int
    decisions: int
    #: Index of the seed pair this episode belongs to. Both orientations of one
    #: seed share it, which is what lets the league settle a seat-balanced
    #: margin for the pairing rather than two unrelated one-sided results.
    pair_index: int = 0


def learner_spec(
    weights_path: Path | str,
    *,
    gamma: float,
    exploration_seed: int,
    trace_path: Path | str | None = None,
    policy_revision_id: str | None = None,
    noise_cadence: str = "none",
    sigma_scale: float = 1.0,
    rare_sigma_gain: float = 0.0,
    rare_sigma_cap: float = 0.0,
    rare_sigma_novelty: str = "count",
    rare_sigma_path: Path | str | None = None,
) -> str:
    """Address the exploring learner as an ordinary agent spec.

    The trace path and revision id are separable: the spec without them is the
    *policy identity*, and folding a per-episode trace into it would mint one agent
    revision per episode.

    `gamma` is in the identity, although only `Recorder.finish` reads it, because a
    run that relabels its rewards is worth telling apart from one that does not.
    """

    query: dict[str, Any] = {
        "weights": str(weights_path),
        "gamma": f"{gamma:.6f}",
        "exploration_seed": str(int(exploration_seed)),
    }
    # Emitted only on departure from the default, for the reason spelled out
    # above: a run with NoisyNet off leaves every spec string byte-identical to
    # what this agent recorded before it existed, so no revision churns.
    if noise_cadence != "none":
        query["noise_cadence"] = str(noise_cadence)
    if sigma_scale != 1.0:
        query["sigma_scale"] = f"{sigma_scale:.6f}"
    # Same rule, and it matters more here: the counts file is content-addressed
    # per checkpoint, so emitting these unconditionally would mint a fresh policy
    # identity every round even for a run that leaves the widening off.
    if rare_sigma_gain > 0.0 and rare_sigma_path is not None:
        query["rare_sigma_gain"] = f"{rare_sigma_gain:.6f}"
        query["rare_sigma_cap"] = f"{rare_sigma_cap:.6f}"
        query["rare_sigma_path"] = str(rare_sigma_path)
        # Same departure-from-the-default rule as the keys above: `"count"` is
        # what every spec before this key meant, so a run on that arm still emits
        # byte-identical spec strings and churns no revisions.
        if rare_sigma_novelty != "count":
            query["rare_sigma_novelty"] = str(rare_sigma_novelty)
    if trace_path is not None:
        query["trace_path"] = str(trace_path)
    if policy_revision_id is not None:
        query["policy_revision_id"] = str(policy_revision_id)
    return f"{LEARNER_ENTRY_POINT}?{urlencode(query)}"


def collect_episode(
    weights_path: Path,
    opponent_spec: str,
    seed: int,
    learner_seat: int,
    gamma: float,
    trace_dir: Path,
    exploration_seed: int,
    pair_index: int = 0,
    policy_revision_id: str | None = None,
    config: dict[str, Any] | None = None,
    noise_cadence: str = "none",
    sigma_scale: float = 1.0,
    rare_sigma_gain: float = 0.0,
    rare_sigma_cap: float = 0.0,
    rare_sigma_novelty: str = "count",
    rare_sigma_path: Path | str | None = None,
) -> OrderRollout | None:
    """Play one episode with an exploring learner and record it."""

    from kaggriculture_lab.episode_metrics import episode_metrics
    from sim.runner import OK_STATUSES, run_match

    trace_path = Path(trace_dir) / f"seed-{seed:010d}-seat-{learner_seat}.npz"
    spec = learner_spec(
        weights_path,
        gamma=gamma,
        exploration_seed=exploration_seed,
        trace_path=trace_path,
        policy_revision_id=policy_revision_id,
        noise_cadence=noise_cadence,
        sigma_scale=sigma_scale,
        rare_sigma_gain=rare_sigma_gain,
        rare_sigma_cap=rare_sigma_cap,
        rare_sigma_novelty=rare_sigma_novelty,
        rare_sigma_path=rare_sigma_path,
    )
    seats = (spec, opponent_spec) if learner_seat == 0 else (opponent_spec, spec)
    result = run_match(seats, seed, config=config, analyzer=episode_metrics)
    # The learner's own seat is what decides whether this episode is trainable.
    # `result.failed` is a match-level flag and would also discard episodes the
    # opponent lost by erroring, which are perfectly good training data.
    if result.seats[learner_seat].status not in OK_STATUSES or not trace_path.is_file():
        return None
    records = load_episode_batch(trace_path)
    if not records:
        return None
    record = records[0]
    return OrderRollout(
        record=record,
        opponent_spec=opponent_spec,
        learner_seat=learner_seat,
        match=result,
        seed=seed,
        decisions=record.decisions,
        pair_index=pair_index,
    )


def _job(args: tuple) -> OrderRollout | None:
    try:
        return collect_episode(*args)
    except Exception:
        # Swallowed so one bad seed cannot take down a whole batch, but never
        # silently: a worker that dies without a trace turns "collection
        # produced nothing" into an unfalsifiable guess. Worker stderr flows to
        # the trainer's stderr, which is where a round hook's failure is read.
        traceback.print_exc()
        return None


#: The BLAS pin and the rollout pool both moved to `sim.workers`, because the
#: promotion gate's arena pays exactly the same two costs on exactly the same
#: episodes. Re-exported under the old names: `training.updater` imports the pin
#: from here to wrap `play_series`.
_single_threaded_workers = single_threaded_workers


def actor_sigma_scales(actors: int, alpha: float = 32.0) -> list[float]:
    """One NoisyNet sigma multiplier per actor, spread geometrically.

    One temperature for a whole batch fills the buffer with near-copies of one
    trajectory: on the spread this replaced, 81% mean pairwise action agreement
    across a batch and 38% of turns identical on all sixteen episodes.

    Actor `i` of `N` runs at `alpha ** (-i/(N-1))`, so actor 0 plays the learned
    sigma and the last is effectively greedy -- which is what keeps `trading_rate`
    and the money diagnostics a clean read of the policy. Ape-X's per-actor
    temperature trick transplanted, *not* principled NoisyNet: the paper has one
    learned sigma and no spread.

    Downward only. Widening the whole policy does not reach rare behaviour, it
    bankrupts the wallet before the behaviour becomes affordable; the retired hot
    rungs measured a land rate indistinguishable from the ordinary ones.
    `rare.sigma_scale_vector` widens single rows instead.
    """

    if actors <= 1:
        return [1.0]
    return [alpha ** (-(index / (actors - 1))) for index in range(actors)]


def collect_batch(
    weights_path: Path,
    opponents: list[str],
    seeds: list[int],
    gamma: float,
    trace_dir: Path,
    jobs: int,
    exploration_seed: int = 0,
    policy_revision_id: str | None = None,
    config: dict[str, Any] | None = None,
    noise_cadence: str = "none",
    sigma_scales: list[float] | None = None,
    rare_sigma_gain: float = 0.0,
    rare_sigma_cap: float = 0.0,
    rare_sigma_novelty: str = "count",
    rare_sigma_path: Path | str | None = None,
    rare_sigma_pairs: int = 0,
) -> list[OrderRollout]:
    """Play `len(seeds)` episodes, one opponent drawn per consecutive seed pair.

    Seeds arrive in consecutive pairs -- the same seed twice -- so the learner takes
    seat 0 in one orientation and seat 1 in the other. Both the opponent and
    `sigma_scales` are indexed by *pair*, never by episode: if the two orientations
    were played by different policies or at different temperatures, the paired
    margin the league settles would compare two different agents.

    `rare_sigma_gain` goes to the first `rare_sigma_pairs` pairs -- the widest rungs
    -- at full gain. The near-greedy tail stays a clean read of the policy, so the
    money series remains comparable across the change.
    """

    scales = sigma_scales or [1.0]
    payloads = []
    for index, seed in enumerate(seeds):
        pair_index = index // 2
        opponent = opponents[pair_index % len(opponents)]
        scale = scales[pair_index % len(scales)]
        pair_gain = rare_sigma_gain if pair_index < max(0, rare_sigma_pairs) else 0.0
        payloads.append(
            (
                weights_path,
                opponent,
                seed,
                index % 2,
                gamma,
                trace_dir,
                exploration_seed + index,
                pair_index,
                policy_revision_id,
                config,
                noise_cadence,
                scale,
                pair_gain,
                rare_sigma_cap,
                rare_sigma_novelty,
                rare_sigma_path,
            )
        )
    if jobs <= 1:
        return [rollout for rollout in (_job(payload) for payload in payloads) if rollout]
    with _single_threaded_workers(), worker_pool(jobs) as pool:
        try:
            return [
                rollout
                for rollout in pool.map(_job, payloads, timeout=MAP_TIMEOUT_SECONDS)
                if rollout
            ]
        except TimeoutError:
            # Same reason as the arena's: a stalled pool that goes back on the
            # free-list hands the next caller the same wedged workers.
            discard_pool(pool)
            raise
        except BrokenProcessPool:
            # A worker died hard (OOM kill, segfault in a BLAS backend). The
            # pool is unusable for good, so drop it rather than returning it to
            # the free-list, and replay the batch once on a fresh one. A second
            # failure is the caller's to see.
            discard_pool(pool)
            with worker_pool(jobs) as retry:
                return [
                    rollout
                    for rollout in retry.map(_job, payloads, timeout=MAP_TIMEOUT_SECONDS)
                    if rollout
                ]
