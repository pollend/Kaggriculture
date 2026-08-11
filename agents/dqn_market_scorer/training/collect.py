"""Roll out episodes and record the market decisions as compact training data.

Transitions cannot be reconstructed after the fact from `env.steps`: the replay
stores the resulting orders, not the candidate set the policy chose from. So the
acting agent records its own decisions, and this module owns that recorder.

Unlike `dqn_job_scorer` there are no per-unit job arrays here -- the field policy
is frozen, so the only thing worth storing is the daily market decision plus the
per-turn reward stream those decisions are credited with. That makes an episode
record roughly two orders of magnitude smaller, which is why `buffer_episodes`
can be large without the replay leaving memory.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import numpy as np

from agents.dqn_market_scorer import donor
from agents.dqn_market_scorer.market_policy import (
    MARKET_ACTION_DIM,
    MARKET_STATE_DIM,
    PRODUCTS,
    MarketDecision,
)

if TYPE_CHECKING:
    from sim.runner import MatchResult

MONEY_SCALE = 1000.0
TERMINAL_MARGIN_SCALE = 20_000.0

FORMAT_VERSION = 1


@dataclass
class EpisodeRecord:
    """One episode: the per-turn reward stream and the daily market decisions."""

    reward: np.ndarray  # float32 [T]    shaped, ready for training
    raw_reward: np.ndarray  # float32 [T]    unshaped money delta, for diagnostics
    potential: np.ndarray  # float32 [T+1]  Phi(s_t); last entry is the terminal state
    money: np.ndarray  # float32 [T+1]
    market_state: np.ndarray  # float32 [D, MARKET_STATE_DIM]
    market_purchase_static: np.ndarray  # float16 [D, P, MARKET_ACTION_DIM]
    market_purchase_mask: np.ndarray  # bool    [D, P]
    market_reserve_static: np.ndarray  # float16 [D, len(PRODUCTS), R, MARKET_ACTION_DIM]
    market_reserve_mask: np.ndarray  # bool    [D, len(PRODUCTS), R]
    market_chosen: np.ndarray  # int16   [D, 1 + len(PRODUCTS)]
    market_step: np.ndarray  # int16   [D]    turn each decision was taken on

    @property
    def steps(self) -> int:
        return self.reward.shape[0]

    @property
    def decisions(self) -> int:
        return self.market_step.shape[0]

    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in EPISODE_ARRAY_FIELDS)


EPISODE_ARRAY_FIELDS = (
    "reward",
    "raw_reward",
    "potential",
    "money",
    "market_state",
    "market_purchase_static",
    "market_purchase_mask",
    "market_reserve_static",
    "market_reserve_mask",
    "market_chosen",
    "market_step",
)


class EpisodeFormatError(ValueError):
    """A rollout artifact cannot be trained with the current replay semantics."""


def save_episode_batch(records: list[EpisodeRecord], path: str | Path) -> Path:
    """Persist variable-width episode records without pickle/object arrays."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray([FORMAT_VERSION], dtype=np.int16),
        "episode_count": np.asarray([len(records)], dtype=np.int32),
    }
    for index, record in enumerate(records):
        for name in EPISODE_ARRAY_FIELDS:
            payload[f"e{index}_{name}"] = getattr(record, name)
    np.savez_compressed(destination, **payload)
    return destination


def load_episode_batch(path: str | Path) -> list[EpisodeRecord]:
    with np.load(Path(path), allow_pickle=False) as payload:
        version = int(payload["format_version"][0])
        if version != FORMAT_VERSION:
            raise EpisodeFormatError(
                f"unsupported market episode batch format {version}; expected {FORMAT_VERSION}"
            )
        count = int(payload["episode_count"][0])
        records = []
        for index in range(count):
            values = {name: payload[f"e{index}_{name}"].copy() for name in EPISODE_ARRAY_FIELDS}
            records.append(EpisodeRecord(**values))
        return records


def potential(farm: dict, private: dict, obs: Any) -> float:
    """Phi(s): liquidatable inventory value, excluding cash already in the reward.

    Used for potential-based reward shaping. Buying seeds, land and animals is
    immediately negative with a payoff dozens of turns later, so an unshaped
    money delta teaches the network to hoard cash and buy nothing -- which is
    exactly the degenerate market policy. Shaping of this form is
    policy-invariant when the terminal potential is forced to zero. Cash is
    deliberately absent because the base reward already records its delta.
    """

    prices = (obs.get("market", {}) or {}).get("prices", {}) or {}

    def price(item: str) -> float:
        return float(prices.get(item, donor.MARKET[item][0]) or donor.MARKET[item][0])

    value = 0.0
    shed = private.get("shed", {}) or {}
    for item in PRODUCTS:
        value += int(shed.get(item, 0) or 0) * price(item)
    seeds = private.get("seeds", {}) or {}
    for crop, rule in donor.CROPS.items():
        value += int(seeds.get(crop, 0) or 0) * float(rule["seed"])
    for animal, rule in donor.ANIMALS.items():
        value += int(shed.get(animal, 0) or 0) * float(rule["cost"])
    for slot in private.get("inventories", []) or []:
        for item in PRODUCTS:
            value += int((slot or {}).get(item, 0) or 0) * price(item)
        for animal, rule in donor.ANIMALS.items():
            value += int((slot or {}).get(animal, 0) or 0) * float(rule["cost"])

    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crop = str(tile.get("crop", ""))
                if crop in donor.CROPS:
                    value += int(tile.get("yield_units", 0) or 0) * price(crop)
            animal = tile.get("animal")
            if animal in donor.ANIMALS:
                product = str(donor.ANIMALS[str(animal)]["product"])
                value += int(tile.get("yield_units", 0) or 0) * price(product)
                value += int(bool(tile.get("fertilizer_available"))) * price("FERTILIZER")
                # A live animal keeps producing; half a unit of its product is a
                # cheap stand-in for that stream and keeps culling unattractive.
                value += 0.5 * price(product)
    return value / MONEY_SCALE


@dataclass
class Recorder:
    """Accumulates one episode of decisions as the agent makes them."""

    money: list[float] = field(default_factory=list)
    potential: list[float] = field(default_factory=list)
    market_steps: list[int] = field(default_factory=list)
    market_decisions: list[MarketDecision] = field(default_factory=list)

    def reset(self) -> None:
        self.money.clear()
        self.potential.clear()
        self.market_steps.clear()
        self.market_decisions.clear()

    def record_market(self, *, step: int, decision: MarketDecision) -> None:
        self.market_steps.append(int(step))
        self.market_decisions.append(decision)

    def record_turn(self, *, step: int, money: float, obs: Any, farm: dict, private: dict) -> None:
        self.money.append(float(money))
        self.potential.append(potential(farm, private, obs))

    def finish(self, final_money: float, opponent_money: float, gamma: float) -> EpisodeRecord:
        steps = len(self.money)
        if steps == 0:
            raise ValueError("recorder captured no steps")
        if not self.market_decisions:
            raise ValueError("recorder captured no market decisions")

        money = np.asarray([*self.money, final_money], dtype=np.float32)
        # The environment ends in an absorbing state. Forcing Phi(s_T)=0 keeps
        # shaping from assigning terminal value to inventory that has no score.
        phi = np.asarray([*self.potential, 0.0], dtype=np.float32)

        raw = (money[1:] - money[:-1]) / MONEY_SCALE
        shaped = raw + gamma * phi[1:] - phi[:-1]
        shaped[-1] += float(np.clip((final_money - opponent_money) / TERMINAL_MARGIN_SCALE, -1.0, 1.0))

        decisions = len(self.market_decisions)
        purchase_width = max(item.candidates.purchase_static.shape[0] for item in self.market_decisions)
        reserve_width = max(item.candidates.reserve_static.shape[1] for item in self.market_decisions)
        market_state = np.zeros((decisions, MARKET_STATE_DIM), dtype=np.float32)
        market_purchase_static = np.zeros((decisions, purchase_width, MARKET_ACTION_DIM), dtype=np.float16)
        market_purchase_mask = np.zeros((decisions, purchase_width), dtype=bool)
        market_reserve_static = np.zeros(
            (decisions, len(PRODUCTS), reserve_width, MARKET_ACTION_DIM), dtype=np.float16
        )
        market_reserve_mask = np.zeros((decisions, len(PRODUCTS), reserve_width), dtype=bool)
        market_chosen = np.zeros((decisions, 1 + len(PRODUCTS)), dtype=np.int16)
        for index, decision in enumerate(self.market_decisions):
            candidates = decision.candidates
            count = candidates.purchase_static.shape[0]
            market_state[index] = candidates.state
            market_purchase_static[index, :count] = candidates.purchase_static
            market_purchase_mask[index, :count] = candidates.purchase_mask
            width = candidates.reserve_static.shape[1]
            market_reserve_static[index, :, :width] = candidates.reserve_static
            market_reserve_mask[index, :, :width] = candidates.reserve_mask
            market_chosen[index] = decision.chosen

        return EpisodeRecord(
            reward=shaped.astype(np.float32),
            raw_reward=raw.astype(np.float32),
            potential=phi,
            money=money,
            market_state=market_state,
            market_purchase_static=market_purchase_static,
            market_purchase_mask=market_purchase_mask,
            market_reserve_static=market_reserve_static,
            market_reserve_mask=market_reserve_mask,
            market_chosen=market_chosen,
            market_step=np.asarray(self.market_steps, dtype=np.int16),
        )


#: Spec target of the training-only wrapper in `match_agent.py`. Rollouts play
#: through it so an exploring learner is an ordinary agent spec.
LEARNER_ENTRY_POINT = "agents.dqn_market_scorer.match_agent:make_agent"

#: Revision role recorded for the learner seat of a rollout pairing.
LEARNER_ROLE = "learner"


@dataclass(frozen=True)
class MarketRollout:
    """One recorded rollout episode plus the match the league will store."""

    record: EpisodeRecord
    opponent_spec: str
    learner_seat: int
    seed: int
    match: MatchResult
    trace_path: str
    # Which seed pair this episode belongs to. Carried rather than recomputed
    # from list position, because a dropped episode would shift every index
    # after it and split a pairing in two.
    pair_index: int = 0


def learner_spec(
    weights_path: str | Path,
    trace_path: str | Path,
    epsilon: float,
    gamma: float,
    exploration_seed: int,
    policy_revision_id: str,
) -> str:
    """The exploring learner as a resolvable agent spec.

    Weights cross the process boundary as a path rather than as arrays: the
    rollout workers are spawned interpreters, and a spec is the only thing
    `sim.registry` accepts.
    """
    query = urlencode(
        {
            "role": LEARNER_ROLE,
            "policy_revision_id": policy_revision_id,
            "weights": str(Path(weights_path).resolve()),
            "trace_path": str(Path(trace_path).resolve()),
            "epsilon": epsilon,
            "gamma": gamma,
            "exploration_seed": exploration_seed,
        }
    )
    return f"{LEARNER_ENTRY_POINT}?{query}"


def collect_episode(
    weights_path: str | Path,
    opponent_spec: str,
    seed: int,
    epsilon: float,
    gamma: float,
    learner_seat: int = 0,
    config: dict[str, Any] | None = None,
    trace_root: str | Path | None = None,
    policy_revision_id: str = "market-learner",
) -> MarketRollout | None:
    """Play one episode with an epsilon-greedy learner and record it.

    Goes through `sim.runner.run_match` rather than driving the environment
    directly, so a rollout yields a full `MatchResult` -- per-seat statuses,
    timings, the replay -- and can be committed to the league exactly like a
    scheduled match instead of vanishing into an `.npz`.

    Returns None if the *learner* errored, so a broken policy is dropped rather
    than silently trained on. An episode the opponent crashed out of is kept and
    recorded as failed.
    """
    from kaggriculture_lab.episode_metrics import episode_metrics
    from sim.runner import OK_STATUSES, run_match

    root = Path(trace_root) if trace_root is not None else Path()
    trace_path = root / f"rollout-{seed}-{learner_seat}.npz"
    specs = [
        learner_spec(
            weights_path,
            trace_path,
            epsilon,
            gamma,
            seed * 7919 + learner_seat,
            policy_revision_id,
        ),
        opponent_spec,
    ]
    if learner_seat == 1:
        specs.reverse()
    result = run_match(
        (specs[0], specs[1]),
        seed,
        config=config,
        analyzer=episode_metrics,
    )
    if result.seats[learner_seat].status not in OK_STATUSES:
        return None
    return MarketRollout(
        load_episode_batch(trace_path)[0],
        opponent_spec,
        learner_seat,
        seed,
        result,
        str(trace_path),
    )


def _job(args: tuple) -> MarketRollout | None:
    return collect_episode(*args)


def collect_batch(
    weights_path: str | Path,
    opponents: list[str],
    seeds: list[int],
    epsilon: float,
    gamma: float,
    jobs: int = 1,
    config: dict[str, Any] | None = None,
    trace_root: str | Path | None = None,
    policy_revision_id: str = "market-learner",
) -> list[MarketRollout]:
    """Collect one episode per seed, alternating which seat the learner takes.

    Seeds arrive in consecutive pairs and each pair draws one opponent, so every
    seed is played in both seat orientations against the same opponent. Both
    seats are used because the environment is symmetric and a policy that only
    ever plays seat 0 has seen half the states it will meet in the arena; the
    completed pair is also what `pairing_results` needs before it can cancel
    seat advantage.
    """
    if len(seeds) % 2:
        raise ValueError("rollout seeds must be paired for seat rotation")
    tasks = [
        (
            weights_path,
            opponents[(index // 2) % len(opponents)],
            seed,
            epsilon,
            gamma,
            index % 2,
            config,
            trace_root,
            policy_revision_id,
        )
        for index, seed in enumerate(seeds)
    ]
    if jobs == 1:
        rollouts = [_job(task) for task in tasks]
    else:
        # Torch is initialized before collection. Forking that process can
        # deadlock or terminate workers after the replay grows; a fresh
        # interpreter is slower to start but stable and cross-platform.
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            rollouts = list(pool.map(_job, tasks, chunksize=1))
    return [
        replace(rollout, pair_index=index // 2)
        for index, rollout in enumerate(rollouts)
        if rollout is not None
    ]
