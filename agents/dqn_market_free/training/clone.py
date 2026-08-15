"""Read demonstration transitions out of stored league replays.

Uniform exploration cannot bootstrap this MDP. Measured: with a 3,000 starting
float, a random order policy ends the episode on 5 coins at a per-turn
exploration rate of 1.0, and still only 4 coins at 0.005, because BUY_LAND is
1,000-4,000 and BUY_ANIMAL 300-500 while the one opening that matters -- buy
seed, plant, sell -- is a specific needle in a wide list. A buffer filled that
way contains nothing but bankruptcy, and the only policy consistent with it is
NOOP.

The league has already played hundreds of episodes of agents that do not go
bankrupt, and `episode_payloads.replay_gzip` keeps every one of them. This module
turns those into transitions in *this* agent's action space, so the buffer has
something positive in it from the first update.

Nothing here touches the inference path. The agent still emits raw orders with
no market rules; these replays are an offline dataset, not a policy component.

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
from agents.dqn_market_free.training.collect import EpisodeRecord
from agents.dqn_market_free.training.reward import episode_rewards


@dataclass(frozen=True, slots=True)
class CloneStats:
    """What a conversion kept and what it had to drop.

    Reported rather than swallowed: a demonstration set that silently dropped
    most of its orders would train the network toward NOOP, which is the exact
    failure the demonstrations exist to prevent. `orders_over_budget` is the
    expected, structural loss -- the demonstrator's second and later orders on a
    turn where this agent may emit only one -- while `orders_unrepresentable`
    means the action list cannot name the order at all and is worth reading as a
    schema problem.
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
    orders_masked_infeasible: int = 0

    @property
    def mapped_fraction(self) -> float:
        return self.orders_mapped / self.orders_seen if self.orders_seen else 0.0


def decode_replay(blob: bytes) -> dict[str, Any]:
    """Inflate a `replay_gzip` payload."""

    return json.loads(gzip.decompress(blob))


def _chosen_index(orders: Iterable[Any]) -> tuple[int, int, int]:
    """Map one turn's order list onto the single action this agent can emit.

    Returns `(index, unrepresentable, over_budget)`. Repeated `HIRE` rows are
    folded into one `HIRE n` first, because that is a single action here and the
    demonstrator's day-boundary hiring is otherwise the largest source of loss
    by far. Then the first order the action list can name wins, and anything the
    demonstrator placed after it on the same turn is `over_budget` -- this agent
    emits one action per turn, and it has 720 turns to spread them over.

    Orders the action list cannot name at all are `unrepresentable` and are
    skipped rather than snapped, so the search continues to the next order
    instead of turning a demonstration into a guess.
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
    for order in rows:
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
    return chosen, unrepresentable, over_budget


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
    cloned episode and a self-collected one are the same kind of sample and can
    share a buffer -- a buffer holding demonstrations scored one way and rollouts
    scored another would be bootstrapping across two different reward functions.

    `opponent_weight` is the one part of that the caller has to supply, because
    the mirror gate is a fact about who played and a replay's seat specs are
    where that lives here. It defaults to 0.0 rather than to `OPPONENT_WEIGHT`
    for the same reason `match_agent` does: a demonstration whose provenance is
    unknown is scored the absolute way, which is the only labelling that is
    correct regardless of who the other seat was.
    """

    steps = replay.get("steps") or []
    if len(steps) < 2:
        return None, CloneStats()

    money: list[float] = []
    order_state = np.zeros((len(steps) - 1, STATE_DIM), dtype=np.float32)
    order_chosen = np.zeros(len(steps) - 1, dtype=np.int16)
    order_step = np.zeros(len(steps) - 1, dtype=np.int16)
    order_mask = np.ones((len(steps) - 1, ACTION_COUNT), dtype=np.bool_)
    configuration = replay.get("configuration")

    seen = mapped = unrepresentable = over_budget = masked_infeasible = 0
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
        farms = obs.get("farms") or []
        player = int(obs.get("player", seat) or seat)
        if not farms or not (0 <= player < len(farms)):
            continue
        farm = farms[player]
        private = obs.get("private") or {}

        money.append(float(farm.get("money", 0) or 0))
        history.observe(obs, farm)

        orders = (current.get("action") or {}).get("market") or []
        action, dropped, extra = _chosen_index(orders)
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
        mask = feasible_actions(obs, farm, private, field, configuration)
        # The demonstrator's order was accepted by the simulator, so it is
        # feasible by demonstration whatever the model of feasibility says.
        # Forced rather than dropped: the invariant the learner depends on is
        # that a stored action is allowed by its stored mask, and violating it
        # would put the clone's cross-entropy target on a masked class and the
        # TD gather on an action the argmax can never reach. Counted, because
        # the count is the direct measure of how over-restrictive the mask is
        # against an agent that really farms.
        if not mask[action]:
            masked_infeasible += 1
            mask[action] = True
        order_mask[decisions] = mask
        # Index into this record's own compacted reward stream, not into the
        # replay. A step the loop skipped above appends no money and so no
        # reward, and `replay.sample` walks `reward[order_step[d]:...]` -- with
        # the raw replay index the window would silently slide off the decision
        # it belongs to, guarded from crashing by the `min(..., record.steps)`
        # clamp and therefore invisible.
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
        )
    return records, totals
