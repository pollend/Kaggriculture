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
from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.dqn_market_free import donor
from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.market_state import MarketHistory, market_state_features
from agents.dqn_market_free.order_space import (
    ACTION_COUNT,
    ACTIONS,
    feasible_actions,
    order_index,
)
from agents.dqn_market_free import lots
from agents.dqn_market_free.training.collect import EpisodeRecord
from agents.dqn_market_free.training.reward import episode_rewards


@dataclass(frozen=True, slots=True)
class CloneStats:
    """What a conversion kept and what it had to drop.

    Reported rather than swallowed: a set that silently dropped most of its orders
    would train the network toward NOOP, the exact failure demonstrations exist to
    prevent. `orders_over_budget` is the structural loss -- the demonstrator's
    second and later orders on one turn -- while `orders_unrepresentable` means the
    action list cannot name the order at all, and reads as a schema problem.
    """

    episodes: int = 0
    decisions: int = 0
    orders_seen: int = 0
    orders_mapped: int = 0
    orders_unrepresentable: int = 0
    orders_over_budget: int = 0
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


def _chosen_index(orders: Iterable[Any]) -> tuple[int, int, int, int]:
    """Map one turn's order list onto the single action this agent can emit.

    Returns `(index, unrepresentable, over_budget, land)`. Repeated `HIRE` rows fold
    into one `HIRE n` first -- that is a single action here, and day-boundary hiring
    is otherwise the largest source of loss. Then the first nameable order wins and
    anything after it on the same turn is `over_budget`.

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

    chosen = 0
    unrepresentable = 0
    over_budget = 0
    land = 0
    for order in rows:
        if str(order[0]) == "BUY_LAND":
            land += 1
            continue
        if chosen:
            over_budget += _wire_count(order)
            continue
        op = str(order[0])
        item = str(order[1]) if len(order) > 2 and order[1] is not None else None
        quantity = int(order[2]) if len(order) > 2 else 0
        index = order_index(op, item, quantity)
        if index is None:
            unrepresentable += _wire_count(order)
            continue
        chosen = index
    return chosen, unrepresentable, over_budget, land


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
    order_state = np.zeros((len(steps) - 1, STATE_DIM), dtype=np.float32)
    order_chosen = np.zeros(len(steps) - 1, dtype=np.int16)
    order_step = np.zeros(len(steps) - 1, dtype=np.int16)
    order_mask = np.ones((len(steps) - 1, ACTION_COUNT), dtype=np.bool_)
    configuration = replay.get("configuration")

    seen = mapped = unrepresentable = over_budget = masked_infeasible = land_seen = 0
    decisions = 0
    # The history block of the state schema is the one part that cannot be read
    # off a single step, so it is replayed here exactly as `main.py` builds it
    # live: observe the turn, encode the state, record the decision. Any drift
    # between the two orderings silently poisons every demonstration, which is
    # what `test_clone_matches_live_features` exists to catch.
    history = MarketHistory()
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
        action, dropped, extra, land_orders = _chosen_index(orders)
        # The mapped action's op, not the raw wire row: `collect` records
        # `ACTIONS[chosen].op` too, and an order the space could not represent was
        # never the decision the learner is being trained on.
        ops.append(ACTIONS[action].op)
        seen += len(orders)
        # Counted in simulator orders, not actions, so the three totals add up
        # to `orders_seen`: one `HIRE 8` action preserves eight of them.
        mapped += len(ACTIONS[action].to_wire())
        unrepresentable += dropped
        over_budget += extra

        roles = donor._role_plan(obs, farm)
        field = donor._unit_actions(obs, configuration, farm, private, roles)
        order_state[decisions] = market_state_features(
            obs,
            farm,
            private,
            roles,
            field,
            config=configuration,
            episode_steps=int(donor._cfg(configuration, "episodeSteps", len(steps))),
            turns_per_day=int(donor._cfg(configuration, "turnsPerDay", 24)),
            history=history,
        )
        history.record_decision(ACTIONS[action])
        order_chosen[decisions] = action
        # The same feasibility the live agent is held to, built from the same
        # `field` that encoded the state so the two describe one turn.
        mask = feasible_actions(obs, farm, private, field, configuration, roles=roles)
        # The simulator accepted this order, so it is feasible by demonstration
        # whatever the model says. Forced rather than dropped, because the learner
        # depends on a stored action being allowed by its stored mask -- otherwise
        # the clone's cross-entropy target lands on a masked class and the TD gather
        # on an action the argmax can never reach. Counted, since the count measures
        # how over-restrictive the mask is against an agent that really farms.
        if not mask[action]:
            masked_infeasible += 1
            mask[action] = True
        order_mask[decisions] = mask
        # Index into this record's own compacted reward stream, not into the replay
        # -- see `collect.Recorder.record_decision` for why.
        order_step[decisions] = decisions
        decisions += 1

    if decisions == 0:
        return None, CloneStats()

    final = steps[-1]
    final_obs = final[seat].get("observation") or {}
    final_farms = final_obs.get("farms") or []
    # Read the terminal farm the same way the loop reads every other one --
    # through the observation's own `player` index rather than the seat number.
    # They agree in every replay seen so far, but disagreeing here would pair a
    # whole episode of one seat's states with the other seat's final money.
    final_player = int(final_obs.get("player", seat) or seat)
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

    emitted = [bool(order_chosen[index]) for index in range(decisions)]
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
        owned=owned,
        produce_room=produce_room,
    )

    record = EpisodeRecord(
        reward=shaped.astype(np.float32),
        raw_reward=raw.astype(np.float32),
        money=money_array,
        order_state=order_state[:decisions],
        order_chosen=order_chosen[:decisions],
        order_step=order_step[:decisions],
        order_mask=order_mask[:decisions],
    )
    stats = CloneStats(
        episodes=1,
        decisions=decisions,
        orders_seen=seen,
        orders_mapped=mapped,
        orders_unrepresentable=unrepresentable,
        orders_over_budget=over_budget,
        orders_masked_infeasible=masked_infeasible,
        orders_land=land_seen,
    )
    return record, stats


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
        record, stats = record_from_replay(replay, int(seat), gamma)
        if record is None:
            continue
        records.append(record)
        totals = CloneStats(
            episodes=totals.episodes + stats.episodes,
            decisions=totals.decisions + stats.decisions,
            orders_seen=totals.orders_seen + stats.orders_seen,
            orders_mapped=totals.orders_mapped + stats.orders_mapped,
            orders_unrepresentable=totals.orders_unrepresentable + stats.orders_unrepresentable,
            orders_over_budget=totals.orders_over_budget + stats.orders_over_budget,
            orders_masked_infeasible=(
                totals.orders_masked_infeasible + stats.orders_masked_infeasible
            ),
            orders_land=totals.orders_land + stats.orders_land,
        )
    return records, totals
