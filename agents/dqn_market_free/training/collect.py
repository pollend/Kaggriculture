"""Roll out episodes and record the raw order decisions as compact training data.

Transitions cannot be reconstructed from `env.steps` after the fact -- the
replay stores filled orders, not the action set the policy chose from -- so
the acting agent records its own decisions and this module owns that recorder.

Records here are far smaller than the scorer's. Its candidate set is rebuilt per
decision from live shed contents, so it has to store the whole encoded grid per
decision. This agent's action is one index into a fixed list, so a decision is
fully described by the state vector and a single integer. That is what makes a
per-turn decision rate affordable: 720 decisions an episode costs less memory
than the scorer's 60.

Reward shaping, the potential function, and the terminal margin bonus live in
`training/reward.py`. They started as the scorer's and are no longer: this agent
decides land and cash reserves itself, so its potential has to price what the
scorer's market layer handled in code.
"""

from __future__ import annotations

import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import numpy as np

from agents.dqn_market_free.diagnostics import OrderDiagnostics, diagnose_order_lists
from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import ACTIONS, OrderDecision
from agents.dqn_market_free.training.reward import episode_rewards, potential

if TYPE_CHECKING:
    from sim.runner import MatchResult

@dataclass
class EpisodeRecord:
    """One episode: the per-turn reward stream and the per-turn order decisions."""

    reward: np.ndarray  # float32 [T]    shaped, ready for training
    raw_reward: np.ndarray  # float32 [T]    unshaped money delta, for diagnostics
    money: np.ndarray  # float32 [T+1]
    order_state: np.ndarray  # float32 [D, STATE_DIM]
    order_chosen: np.ndarray  # int16   [D]  index into ACTIONS
    order_step: np.ndarray  # int16   [D]  index into `reward` each decision is paid from

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
        """Decision and order counts for the orders actually emitted."""

        return diagnose_order_lists(
            [ACTIONS[int(choice)].to_wire() for choice in self.order_chosen],
            [float(self.raw_reward[int(step)]) for step in self.order_step],
        )

    def nbytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in EPISODE_ARRAY_FIELDS)


EPISODE_ARRAY_FIELDS = (
    "reward",
    "raw_reward",
    "money",
    "order_state",
    "order_chosen",
    "order_step",
)


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
            values = {name: payload[f"e{index}_{name}"].copy() for name in EPISODE_ARRAY_FIELDS}
            width = values["order_state"].shape[-1]
            if width != STATE_DIM:
                raise ValueError(
                    f"{path}: episode {index} has a {width}-wide state, expected {STATE_DIM}"
                )
            records.append(EpisodeRecord(**values))
        return records


@dataclass
class Recorder:
    """Accumulates one episode of decisions as the agent makes them."""

    money: list[float] = field(default_factory=list)
    potential: list[float] = field(default_factory=list)
    order_steps: list[int] = field(default_factory=list)
    decisions: list[OrderDecision] = field(default_factory=list)

    def reset(self) -> None:
        self.money.clear()
        self.potential.clear()
        self.order_steps.clear()
        self.decisions.clear()

    def record_decision(self, *, decision: OrderDecision) -> None:
        """Store a decision against the reward index it will be paid from.

        The simulator's turn number is deliberately not what lands in
        `order_step`. The reward stream is built from `self.money`, which is
        appended by `record_turn` -- and the agent skips both on a malformed
        observation, so a turn the simulator counted may contribute no reward at
        all. Indexing by simulator step then slides every later decision's n-step
        window forward, silently, because `replay.sample` clamps the window with
        `min(end_step, record.steps)`. `clone.py` records the same compacted
        index for the same reason; the two record kinds share one buffer and
        must mean the same thing by it.

        `record_decision` runs before `record_turn` in `main.py`, so
        `len(self.money)` is exactly the slot this turn's reward will take.
        """

        self.order_steps.append(len(self.money))
        self.decisions.append(decision)

    def record_turn(self, *, money: float, obs: Any, farm: dict, private: dict) -> None:
        self.money.append(float(money))
        self.potential.append(potential(farm, private, obs))

    def finish(self, final_money: float, opponent_money: float, gamma: float) -> EpisodeRecord:
        steps = len(self.money)
        if steps == 0:
            raise ValueError("recorder captured no steps")
        if not self.decisions:
            raise ValueError("recorder captured no order decisions")

        # Which turns put an order on the wire, indexed the same way the reward
        # stream is. A turn that emitted one and moved no money had it refused.
        emitted = [False] * len(self.money)
        for step, decision in zip(self.order_steps, self.decisions):
            if 0 <= step < len(emitted) and decision.chosen:
                emitted[step] = True

        shaped, raw, money = episode_rewards(
            self.money, self.potential, final_money, opponent_money, gamma, emitted
        )

        count = len(self.decisions)
        order_state = np.zeros((count, STATE_DIM), dtype=np.float32)
        order_chosen = np.zeros(count, dtype=np.int16)
        for index, decision in enumerate(self.decisions):
            order_state[index] = decision.state
            order_chosen[index] = decision.chosen

        return EpisodeRecord(
            reward=shaped.astype(np.float32),
            raw_reward=raw.astype(np.float32),
            money=money,
            order_state=order_state,
            order_chosen=order_chosen,
            order_step=np.asarray(self.order_steps, dtype=np.int16),
        )


#: Spec target of the training-only wrapper in `match_agent.py`.
LEARNER_ENTRY_POINT = "agents.dqn_market_free.match_agent:make_agent"


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
    epsilon: float,
    gamma: float,
    exploration_seed: int,
    trace_path: Path | str | None = None,
    policy_revision_id: str | None = None,
) -> str:
    """Address the exploring learner as an ordinary agent spec.

    The per-episode trace path and revision id are deliberately separable: the
    spec without them is the *policy identity*, and folding a per-episode trace
    into it would mint one agent revision per episode.
    """

    query: dict[str, Any] = {
        "weights": str(weights_path),
        "epsilon": f"{epsilon:.6f}",
        "gamma": f"{gamma:.6f}",
        "exploration_seed": str(int(exploration_seed)),
    }
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
    epsilon: float,
    gamma: float,
    trace_dir: Path,
    exploration_seed: int,
    pair_index: int = 0,
    policy_revision_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> OrderRollout | None:
    """Play one episode with an epsilon-greedy learner and record it."""

    from kaggriculture_lab.episode_metrics import episode_metrics
    from sim.runner import OK_STATUSES, run_match

    trace_path = Path(trace_dir) / f"seed-{seed:010d}-seat-{learner_seat}.npz"
    spec = learner_spec(
        weights_path,
        epsilon=epsilon,
        gamma=gamma,
        exploration_seed=exploration_seed,
        trace_path=trace_path,
        policy_revision_id=policy_revision_id,
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


#: Thread-count variables every BLAS backend numpy might be linked against reads.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextmanager
def _single_threaded_workers():
    """Pin spawned rollout workers to one BLAS thread each.

    Parallelism here is across episodes, not inside one. Left alone, every
    worker's BLAS opens a thread pool sized to the whole machine, so `jobs`
    workers ask for `jobs * cores` threads: measured at ~350% CPU per worker,
    eight workers wanting ~29 cores of a 32-core box, and a 16-episode batch
    taking 17 minutes against an ideal of well under one.

    Set in the parent rather than in an initializer because `spawn` children
    inherit the environment at launch, and the variables have to be set before
    numpy is imported in the child -- which happens while the task is being
    unpickled, ahead of any initializer running.
    """

    saved = {name: os.environ.get(name) for name in _THREAD_VARS}
    os.environ.update({name: "1" for name in _THREAD_VARS})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def actor_epsilons(base: float, actors: int, alpha: float = 4.0) -> list[float]:
    """One exploration rate per actor, spread geometrically around `base`.

    A single epsilon for the whole batch means every episode is the same policy
    at the same temperature, and the buffer fills with near-copies of one
    trajectory. Measured on this agent at a base of 0.014: 81% mean pairwise
    action agreement between the sixteen episodes of a batch, and on 38% of
    turns all sixteen chose the identical action.

    So spread it, as Ape-X does: actor `i` of `N` runs at
    `base ** (1 + i/(N-1) * alpha)`. The low-index actors explore hard enough to
    find something, the high-index ones are effectively greedy and show what the
    policy actually does, and every batch contains both rather than alternating
    between them as the schedule decays.
    """

    if actors <= 1:
        return [base]
    return [base ** (1.0 + (index / (actors - 1)) * alpha) for index in range(actors)]


def collect_batch(
    weights_path: Path,
    opponents: list[str],
    seeds: list[int],
    epsilons: list[float],
    gamma: float,
    trace_dir: Path,
    jobs: int,
    exploration_seed: int = 0,
    policy_revision_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> list[OrderRollout]:
    """Play `len(seeds)` episodes, one opponent drawn per consecutive seed pair.

    Seeds arrive in consecutive pairs -- the same seed twice -- so the learner
    takes seat 0 in one orientation and seat 1 in the other. Drawing one
    opponent per pair rather than per episode is what lets the resulting pairing
    settle a seat-balanced margin.

    `epsilons` is indexed by pair for the same reason the opponent is: both
    orientations of a seed must be played by the same policy at the same
    temperature, or the paired margin the league settles is a comparison between
    two different agents.
    """

    payloads = []
    for index, seed in enumerate(seeds):
        pair_index = index // 2
        opponent = opponents[pair_index % len(opponents)]
        payloads.append(
            (
                weights_path,
                opponent,
                seed,
                index % 2,
                epsilons[pair_index % len(epsilons)],
                gamma,
                trace_dir,
                exploration_seed + index,
                pair_index,
                policy_revision_id,
                config,
            )
        )
    if jobs <= 1:
        return [rollout for rollout in (_job(payload) for payload in payloads) if rollout]
    context = multiprocessing.get_context("spawn")
    with _single_threaded_workers():
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            return [rollout for rollout in pool.map(_job, payloads) if rollout]
