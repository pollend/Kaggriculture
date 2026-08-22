"""Read demonstration transitions out of stored league replays.

Uniform exploration cannot bootstrap this MDP: a random order policy finishes a
3,000-coin season on single digits at every rate measured, because the openings
that matter are a needle in a wide list of ways to spend the stake. A buffer
filled that way contains nothing but bankruptcy, and the only policy consistent
with it is NOOP.

`episode_payloads.replay_gzip` keeps every league episode of agents that do *not*
go bankrupt. This module turns those into transitions in this agent's action
space, so the buffer has something positive in it from the first update.

Nothing here touches the inference path: an offline dataset, not a policy
component.

Replay layout, which is easy to get wrong:

    steps[i][seat]["action"]       the action taken ON step i
    steps[i][seat]["observation"]  the state AFTER that action

so a transition pairs `steps[i - 1]`'s observation with `steps[i]`'s action.
`steps[0]` is the environment's own reset and carries an empty market.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from agents.dqn_market_free_v3b import donor, lots
from agents.dqn_market_free_v3b.herd import field_roles
from agents.dqn_market_free_v3b.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free_v3b.market_state import MarketHistory, market_state_features
from agents.dqn_market_free_v3b.order_space import (
    ACTIONS,
    LAND_ORDER,
    MAX_DECISIONS_PER_TURN,
    PendingTurn,
    feasible_actions,
    order_index,
)
from agents.dqn_market_free_v3b.policy import HIDDEN_SIZE as GRU_HIDDEN
from agents.dqn_market_free_v3b.policy import GruCell
from agents.dqn_market_free_v3b.training.collect import HIDDEN_STRIDE, EpisodeRecord
from agents.dqn_market_free_v3b.training.reward import episode_rewards


@dataclass(frozen=True, slots=True)
class CloneStats:
    """What a conversion kept and what it had to drop.

    Reported rather than swallowed: a set that silently dropped most of its orders
    would train the network toward NOOP, the exact failure demonstrations exist to
    prevent. `orders_unrepresentable` means the action list cannot name the order at
    all, and reads as a schema problem.

    v1 carried an `orders_over_budget` here for the demonstrator's second and later
    orders on a turn, which it structurally could not emit. v2 emits them, so the
    counter is gone rather than left reporting zero -- a live zero and an impossible
    zero look the same in a metrics table.
    """

    episodes: int = 0
    decisions: int = 0
    orders_seen: int = 0
    orders_mapped: int = 0
    orders_unrepresentable: int = 0
    #: Turns where the feasibility mask forbade the order the demonstrator
    #: actually placed. Should be ~0: a solvent donor's accepted orders are
    #: feasible by definition, so anything here is the mask being wrong.
    #:
    #: `BUY_LAND` cannot appear here at all -- it is not a row -- and is counted in
    #: `orders_land` instead. The demonstrator buying a quadrant this agent's
    #: `should_buy_land` declines is the two policies disagreeing, which is the point
    #: of having a heuristic, not a fault in the mask.
    orders_masked_infeasible: int = 0
    #: `BUY_LAND` rows seen in demonstrations. Not a fault and not a loss: nothing
    #: maps them because land is decided off the head.
    orders_land: int = 0

    @property
    def mapped_fraction(self) -> float:
        return self.orders_mapped / self.orders_seen if self.orders_seen else 0.0


def decode_replay(blob: bytes) -> dict[str, Any]:
    """Inflate a `replay_gzip` payload."""

    return json.loads(gzip.decompress(blob))


def _chosen_sequence(orders: Iterable[Any]) -> tuple[list[int], int, int]:
    """Map one turn's order list onto the sequence of actions this agent emits.

    Returns `(indices, unrepresentable, land)`. Repeated `HIRE` rows fold into one
    `HIRE n` first, as they do for the live agent.

    v1 had a fourth counter, `over_budget`: it could emit one order a turn, so every
    order after the first was loss. v2 emits the whole list, so there is no budget to
    be over until the wire cap -- which the demonstrator was held to as well, since it
    is the simulator's -- and the counter has no meaning left. What was the dominant
    source of demonstration loss is now simply the data.

    Order is preserved, because it is a decision the demonstrator made: selling
    before buying funds the buy, and the replayed `PendingTurn` walk has to see the
    same sequence to reproduce the wallet the demonstrator was actually spending.

    Orders the action list cannot name are skipped rather than snapped, so the
    search continues instead of turning a demonstration into a guess.

    `BUY_LAND` is skipped without being counted as unnameable, and does not claim the
    turn. Since v44 it is not on the head, so there is no row to map it to -- but the
    demonstrator buying a quadrant is not a hole in this agent's action space, it is
    a decision `order_space.should_buy_land` makes on its own. Lumping it into
    `unrepresentable` would make that counter unreadable, which is the same argument
    the old `orders_masked_land` was carrying before the row left.
    """

    rows = [order for order in orders if order]
    hires = sum(1 for order in rows if str(order[0]) == "HIRE")
    if hires:
        # Collapse to the first HIRE's position, carrying the whole count.
        folded: list[Any] = []
        for order in rows:
            if str(order[0]) != "HIRE":
                folded.append(order)
            elif not any(str(row[0]) == "HIRE" for row in folded):
                folded.append(["HIRE", None, hires])
        rows = folded

    chosen: list[int] = []
    unrepresentable = 0
    land = 0
    for order in rows:
        if str(order[0]) == "BUY_LAND":
            land += 1
            continue
        op = str(order[0])
        item = str(order[1]) if len(order) > 2 and order[1] is not None else None
        quantity = int(order[2]) if len(order) > 2 else 0
        index = order_index(op, item, quantity)
        if index is None:
            unrepresentable += _wire_count(order)
            continue
        chosen.append(index)
    return chosen, unrepresentable, land


def _wire_count(order: Any) -> int:
    """How many simulator orders this row stands for, after HIRE folding."""

    if str(order[0]) == "HIRE":
        return max(1, int(order[2])) if len(order) > 2 else 1
    return 1


def record_from_replay(
    replay: dict[str, Any],
    seat: int,
    gamma: float = 0.999,
    opponent_weight: float = 0.0,
    gru: GruCell | None = None,
) -> tuple[EpisodeRecord | None, CloneStats]:
    """Convert one seat of one replay into a trainable episode record.

    The reward comes from the same `episode_rewards` a live rollout uses, so a
    cloned episode and a self-collected one share a buffer without bootstrapping
    across two reward functions.

    `opponent_weight` is the one part the caller supplies, because the mirror gate
    is a fact about who played. Defaults to 0.0: a demonstration whose provenance is
    unknown is scored the absolute way, the only labelling correct regardless of who
    the other seat was.
    """

    steps = replay.get("steps") or []
    if len(steps) < 2:
        return None, CloneStats()

    money: list[float] = []
    quadrants: list[int] = []
    days: list[int] = []
    seeds: list[list[int]] = []
    standing: list[list[int]] = []
    stock: list[list[int]] = []
    owned: list[list[int]] = []
    produce_room: list[list[int]] = []
    ops: list[str | None] = []
    # Accumulated rather than preallocated: a v2 turn holds one decision per order
    # plus a terminator, so the row count is a property of the demonstrator's
    # behaviour and not of the episode length. Preallocating the worst case --
    # every turn filling the wire -- would reserve 11x the memory for a shape no
    # demonstration reaches.
    state_rows: list[np.ndarray] = []
    chosen_rows: list[int] = []
    step_rows: list[int] = []
    mask_rows: list[np.ndarray] = []
    #: Per-turn, for the reward stream: what the turn's orders were quoted at, and
    #: the orders themselves. `PendingTurn` computes both while walking the turn, so
    #: a cloned episode reaches `episode_rewards` carrying exactly what a live one
    #: does -- which is what keeps the two the same kind of sample.
    events: list[list] = []
    expected: list[float] = []
    configuration = replay.get("configuration")

    seen = mapped = unrepresentable = masked_infeasible = land_seen = 0
    decisions = 0
    turn_index = 0
    # Run in lockstep with `main.py` -- observe the turn, encode the state, record
    # the decision -- and `history` is live here: block D' reads the latches
    # `observe` maintains, so the walk is load-bearing rather than bookkeeping. It is the walk that has to match, and `observe` is the
    # marker the tick below sites against; drift in that ordering silently poisons
    # every demonstration, which is what
    # `test_cloned_state_matches_the_live_agent` exists to catch.
    history = MarketHistory()
    # The recurrent half of the same lockstep. `hidden` starts at exactly zero,
    # as `OrderQNetwork.reset_hidden` does; `last_turn` starts at None, as
    # `MarketFreeAgent` does at `step == 0`.
    #
    # `gru` is required rather than defaulted to a no-op. Without one every row of
    # `turn_hidden` would be the same zero vector, and that record passes every
    # check `load_episode_batch` makes -- the widths and the stride are all
    # correct -- before feeding `hidden0 = 0` to chunks that start deep inside the
    # episode. That is the desynchronised-resume failure this whole file is
    # written to avoid, arriving silently through the default argument.
    if gru is None:
        raise ValueError(
            "record_from_replay needs the GRU that will train on the record: "
            "without it every stored hidden state is zero and mid-episode chunks "
            "resume from a state the policy never occupied"
        )
    hidden = gru.zeros()
    last_turn = None
    turn_inputs: list[np.ndarray] = []
    turn_hiddens: list[np.ndarray] = []
    for index in range(1, len(steps)):
        previous = steps[index - 1][seat]
        current = steps[index][seat]
        obs = previous.get("observation") or {}
        view = lots.seat_view(obs, seat)
        if view is None:
            continue
        farm, private = view

        # Every trace off one `lots.sample`, appended in lockstep with `money` so a
        # cloned episode and a self-collected one carry the same trace on the same
        # index. `collect.Recorder.record_turn` is the other half of this, and it is
        # now literally the same builder -- the two hand-written copies had already
        # drifted on the `day` fallback. `standing` comes off `farm["tiles"]` rather
        # than `private`, which is what makes it recoverable from a stored replay.
        held = lots.sample(obs, farm, private, len(money))
        money.append(float(farm.get("money", 0) or 0))
        quadrants.append(held.quadrants)
        days.append(held.day)
        seeds.append(held.seeds)
        standing.append(held.standing)
        stock.append(held.stock)
        owned.append(held.owned)
        produce_room.append(held.produce_room)
        history.observe(obs, farm)

        orders = (current.get("action") or {}).get("market") or []
        actions, dropped, land_orders = _chosen_sequence(orders)
        # The mapped action's op, not the raw wire row: `collect` records
        # `ACTIONS[chosen].op` too, and an order the space could not represent was
        # never the decision the learner is being trained on. The turn's *last*
        # traded op, matching what `collect.finish` keeps in `ops`.
        traded = [ACTIONS[index].op for index in actions if index]
        ops.append(traded[-1] if traded else ACTIONS[0].op)
        seen += len(orders)
        # Counted in simulator orders, not actions, so the totals add up to
        # `orders_seen`: one `HIRE 8` action preserves eight of them.
        mapped += sum(len(ACTIONS[index].to_wire()) for index in actions)
        unrepresentable += dropped
        # `land_orders` was unpacked and discarded here while its three siblings
        # above were accumulated, so `orders_land` reported 0 for every clone ever
        # taken and the totals came up short by exactly the land rows. `BUY_LAND`
        # left the action space, so these are neither mapped nor unrepresentable:
        # the head has no column to carry them and `should_buy_land` owns the
        # decision instead. They are their own bucket.
        land_seen += land_orders

        roles = field_roles(obs, farm, private)
        field = donor._unit_actions(obs, configuration, farm, private, roles)

        # The recurrent tick, in `main.py`'s position and with `main.py`'s
        # arguments: after `history.observe`, before the decision loop, on the
        # untouched observation, and with block E carrying the *previous* turn's
        # finished `PendingTurn`. Any drift between the two walks poisons every
        # state after it rather than degrading a feature, which is why
        # `test_cloned_state_matches_the_live_agent` compares the whole vector and
        # the hidden trajectory rather than one block.
        turn_state = market_state_features(
            obs,
            farm,
            private,
            roles,
            field,
            config=configuration,
            episode_steps=int(donor._cfg(configuration, "episodeSteps", len(steps))),
            turns_per_day=int(donor._cfg(configuration, "turnsPerDay", 24)),
            history=history,
            turn=last_turn,
        )
        turn_inputs.append(turn_state)
        turn_hiddens.append(hidden)
        hidden = gru.step(turn_state, hidden)
        # The turn is replayed the way the live agent plays it: one `PendingTurn`,
        # one decision per order in the demonstrator's own order, and a terminator.
        # Nothing else reproduces the state the demonstrator's second order was
        # chosen from -- its wallet had already paid for the first.
        pending = PendingTurn(obs, farm, private, field, configuration, roles=roles)
        if land_orders:
            pending.commit_land()
            # Paired with `commit_land` exactly as `main.py:284-287` pairs them.
            # Inert in v3, where nothing read what `history` accumulated; live here,
            # because `record_decision` sets `_emitted` and the *next* `observe`
            # differences that into block D's three latches. Without it, a
            # demonstrator turn whose only order was a land buy leaves the clone at
            # `_emitted = False` -- so the clone reads `turns_since_fill += 1` where
            # the live agent reads a fill and resets. Latches integrate, so the two
            # do not re-converge on the following turn; columns 218:221 stay apart
            # for the rest of the episode.
            history.record_decision(LAND_ORDER)
        # The terminator is a decision too. Without it the STOP row is never a
        # demonstrated action, and behaviour cloning would push every state toward
        # trading -- including the states the demonstrator deliberately left alone.
        #
        # But only where the live loop would have reached it. `main.py` breaks
        # *before* recording when the wire is full or the decision cap is hit, so a
        # turn that spends all ten slots records ten rows and no STOP. This walk
        # used to be a flat `[*actions, 0]`, which minted a terminator on every
        # turn: a decision the demonstrator never made, in a state where NOOP was
        # not necessarily what it would have chosen, and -- worse -- one extra row,
        # which slides every later `order_step` and every n-step window built off
        # it. Two turns in a 720-turn episode were enough to do it.
        #
        # The three stopping conditions below are `main.py`'s, in its order: the
        # iteration cap, the slot test before the row is recorded, and the break
        # after a recorded NOOP.
        taken = 0
        for action in [*actions, 0]:
            if taken >= MAX_DECISIONS_PER_TURN or not pending.slots_left:
                break
            obs_view, farm_view, private_view = pending.view(obs, farm, private)
            state = market_state_features(
                obs_view,
                farm_view,
                private_view,
                roles,
                field,
                config=configuration,
                episode_steps=int(donor._cfg(configuration, "episodeSteps", len(steps))),
                turns_per_day=int(donor._cfg(configuration, "turnsPerDay", 24)),
                history=history,
                turn=pending,
            )
            state_rows.append(state)
            chosen_rows.append(int(action))
            # The same feasibility the live agent is held to, built from the same
            # `field` that encoded the state so the two describe one turn, and from
            # the same pending walk so they describe the same point *within* it.
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
            # The simulator accepted this order, so it is feasible by demonstration
            # whatever the model says. Forced rather than dropped, because the learner
            # depends on a stored action being allowed by its stored mask -- otherwise
            # the clone's cross-entropy target lands on a masked class and the TD gather
            # on an action the argmax can never reach. Counted, since the count measures
            # how over-restrictive the mask is against an agent that really farms.
            if not mask[action]:
                masked_infeasible += 1
                mask[action] = True
            mask_rows.append(mask)
            # Index into this record's own compacted reward stream, not into the
            # replay -- see `collect.Recorder.record_decision` for why. Every
            # sub-decision of a turn carries the same index, which is what makes the
            # intra-turn transitions carry no reward and bootstrap to each other.
            step_rows.append(turn_index)
            decisions += 1
            taken += 1
            if not action:
                break
            history.record_decision(ACTIONS[action])
            pending.apply(ACTIONS[action])
        events.append(list(pending.events))
        expected.append(pending.expected_delta)
        last_turn = pending
        turn_index += 1

    if decisions == 0:
        return None, CloneStats()

    final = steps[-1]
    final_obs = final[seat].get("observation") or {}
    final_farms = final_obs.get("farms") or []
    # Read the terminal farm the same way the loop reads every other one --
    # through the observation's own `player` index rather than the seat number.
    # They agree in every replay seen so far, but disagreeing here would pair a
    # whole episode of one seat's states with the other seat's final money.
    # Not `or seat`: seat 1 reading `player: 0` is precisely the disagreement
    # named above, and 0 is falsy, so the guard would hand the seat back its own
    # index in the one case it exists for.
    _declared = final_obs.get("player", seat)
    final_player = int(seat if _declared is None else _declared)
    if not (0 <= final_player < len(final_farms)):
        final_player = seat
    final_money = (
        float(final_farms[final_player].get("money", 0) or 0)
        if 0 <= final_player < len(final_farms)
        else money[-1]
    )
    opponent_money = (
        float(final_farms[1 - final_player].get("money", 0) or 0)
        if len(final_farms) > 1 and 0 <= 1 - final_player < len(final_farms)
        else 0.0
    )

    order_state = np.asarray(state_rows, dtype=np.float32)
    order_chosen = np.asarray(chosen_rows, dtype=np.int16)
    order_step = np.asarray(step_rows, dtype=np.int16)
    order_mask = np.asarray(mask_rows, dtype=np.bool_)

    # Per *turn*, not per decision: `emitted` indexes the reward stream, and a turn
    # counts as having traded if any of its sub-decisions did.
    emitted = [False] * len(money)
    for choice, step in zip(chosen_rows, step_rows):
        if choice and step < len(emitted):
            emitted[step] = True
    shaped, raw, money_array = episode_rewards(
        money,
        final_money,
        opponent_money,
        gamma,
        emitted,
        opponent_weight=opponent_weight,
        quadrants=quadrants,
        days=days,
        seeds=seeds,
        standing=standing,
        stock=stock,
        ops=ops,
        events=events,
        expected=expected,
        owned=owned,
        produce_room=produce_room,
    )

    # Same phase check `Recorder.finish` makes, and for the same reason: the walk
    # above `continue`s past a malformed seat view, so a turn that contributed no
    # reward slot must have contributed no tick either.
    if len(turn_inputs) != len(shaped):
        raise ValueError(
            f"clone walked {len(turn_inputs)} turns for {len(shaped)} reward steps; "
            "the GRU tick and the money trace are out of phase"
        )
    record = EpisodeRecord(
        reward=shaped.astype(np.float32),
        raw_reward=raw.astype(np.float32),
        money=money_array,
        order_state=order_state,
        order_chosen=order_chosen,
        order_step=order_step,
        order_mask=order_mask,
        turn_state=np.asarray(turn_inputs, dtype=np.float32).reshape(-1, STATE_DIM),
        turn_hidden=np.asarray(
            turn_hiddens[::HIDDEN_STRIDE], dtype=np.float16
        ).reshape(-1, GRU_HIDDEN),
        hidden_stride=np.asarray([HIDDEN_STRIDE], dtype=np.int16),
    )
    stats = CloneStats(
        episodes=1,
        decisions=decisions,
        orders_seen=seen,
        orders_mapped=mapped,
        orders_unrepresentable=unrepresentable,
        orders_masked_infeasible=masked_infeasible,
        orders_land=land_seen,
    )
    return record, stats


def rehidden(record: EpisodeRecord, gru: GruCell) -> EpisodeRecord:
    """Re-roll a record's stored hidden states under `gru`.

    `record_from_replay` refuses a missing GRU because a `turn_hidden` column
    written by a different network is the desynchronised-resume failure: every
    width and stride still checks out, and chunks that start mid-episode resume
    from a state the policy never occupied. A *cached* demonstration has the same
    problem for a different reason -- the policy moves every update, so a pack
    converted once is stale by the second one.

    It is fixable rather than fatal, because the hidden trajectory is a pure
    function of `turn_state` and the GRU. `turn_state` is stored (`collect.py`'s
    `EPISODE_ARRAY_FIELDS`), so the expensive half of the conversion -- inflating
    30 MB of replay and walking 720 turns of feature encoding -- is cached and only
    the cheap recurrence is redone. This is the loop from the walk above, lifted
    verbatim: same start at `gru.zeros()`, same pre-step store, same stride, same
    fp16.
    """

    if record.turn_state.shape[1] != gru.input_size:
        raise ValueError(
            f"record carries {record.turn_state.shape[1]}-wide turn states and the "
            f"gru takes {gru.input_size}; the pack was written by another schema"
        )
    hidden = gru.zeros()
    hiddens: list[np.ndarray] = []
    for turn_state in record.turn_state:
        hiddens.append(hidden)
        hidden = gru.step(turn_state, hidden)
    return replace(
        record,
        turn_hidden=np.asarray(hiddens[::HIDDEN_STRIDE], dtype=np.float16).reshape(
            -1, GRU_HIDDEN
        ),
        hidden_stride=np.asarray([HIDDEN_STRIDE], dtype=np.int16),
    )


DEMONSTRATION_SQL = """
SELECT ep.match_id, ep.seat, pl.replay_gzip
FROM episode_players ep
JOIN episode_payloads pl USING (match_id)
JOIN agent_revisions ar ON ar.revision_id = ep.strategy_id
WHERE NOT ep.failed
  AND ep.final_money >= %s
  AND (CAST(%s AS varchar) IS NULL OR ar.family_id = %s)
ORDER BY ep.final_money DESC
LIMIT %s
"""


def load_demonstrations(
    reader,
    *,
    limit: int = 200,
    min_final_money: float = 10_000.0,
    family_id: str | None = None,
    gamma: float = 0.999,
    gru: GruCell | None = None,
) -> tuple[list[EpisodeRecord], CloneStats]:
    """Pull the best stored episodes and convert the demonstrator's seat.

    Ranked by `final_money` and floored at `min_final_money` because the point
    is to put *solvent* trajectories in the buffer. Cloning a bankruptcy teaches
    nothing the agent's own exploration will not find on its own.
    """

    rows = reader.fetchall(
        DEMONSTRATION_SQL, [min_final_money, family_id, family_id, int(limit)]
    )
    records: list[EpisodeRecord] = []
    totals = CloneStats()
    for _match_id, seat, blob in rows:
        try:
            replay = decode_replay(bytes(blob))
        except (OSError, ValueError):
            continue
        record, stats = record_from_replay(replay, int(seat), gamma, gru=gru)
        if record is None:
            continue
        records.append(record)
        totals = CloneStats(
            episodes=totals.episodes + stats.episodes,
            decisions=totals.decisions + stats.decisions,
            orders_seen=totals.orders_seen + stats.orders_seen,
            orders_mapped=totals.orders_mapped + stats.orders_mapped,
            orders_unrepresentable=totals.orders_unrepresentable + stats.orders_unrepresentable,
            orders_masked_infeasible=(
                totals.orders_masked_infeasible + stats.orders_masked_infeasible
            ),
            orders_land=totals.orders_land + stats.orders_land,
        )
    return records, totals
