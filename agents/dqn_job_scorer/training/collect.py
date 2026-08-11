"""Roll out episodes and record them as compact training data.

Transitions cannot be reconstructed after the fact from `env.steps`: the replay
stores the resulting op, not the candidate job set the policy chose from. So the
acting agent records its own decisions, and this module owns that recorder.

Storage is a struct-of-arrays per episode rather than a flat transition list.
The naive layout -- a full (n_jobs, FEATURE_DIM) candidate matrix per
unit-transition -- costs ~30 MB per episode and makes any useful buffer size
impossible. Globals are identical across units at a step and job features are
identical except the distance term, so those are stored once per step and the
distance columns are recomputed when a minibatch is drawn.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import numpy as np

from agents.dqn_job_scorer import features as F
from agents.dqn_job_scorer.market_policy import MARKET_ACTION_DIM, MARKET_STATE_DIM, MarketDecision
from agents.dqn_job_scorer.rules import ANIMAL_DATA, CROP_DATA, CROPS, PRODUCT_BASE, PRODUCTS, get

if TYPE_CHECKING:
    from sim.runner import MatchResult

MONEY_SCALE = 1000.0
TERMINAL_MARGIN_SCALE = 20_000.0


@dataclass
class EpisodeRecord:
    """One episode, padded to a common job/unit width with masks."""

    global_feats: np.ndarray  # float32 [T, GLOBAL_DIM]
    job_static: np.ndarray  # float16 [T, J, JOB_STATIC_DIM]
    job_targets: np.ndarray  # int8    [T, J, 2]
    job_mask: np.ndarray  # bool    [T, J]
    unit_pos: np.ndarray  # int8    [T, U, 2]
    unit_mask: np.ndarray  # bool    [T, U]
    chosen: np.ndarray  # int16   [T, U]
    committed: np.ndarray  # int16   [T, U]  job index this unit was already walking to, -1 if none
    available: np.ndarray  # bool    [T, U, J] exact live action mask for each unit
    reward: np.ndarray  # float32 [T]   shaped, ready for training
    raw_reward: np.ndarray  # float32 [T]   unshaped money delta, for diagnostics
    potential: np.ndarray  # float32 [T+1] Phi(s_t); last entry is the terminal state
    money: np.ndarray  # float32 [T+1]
    truncated: np.ndarray  # int32   [T]   candidate-set overflow, counted not hidden
    market_state: np.ndarray | None = None
    market_purchase_static: np.ndarray | None = None
    market_purchase_mask: np.ndarray | None = None
    market_reserve_static: np.ndarray | None = None
    market_reserve_mask: np.ndarray | None = None
    market_chosen: np.ndarray | None = None
    market_step: np.ndarray | None = None

    @property
    def steps(self) -> int:
        return self.global_feats.shape[0]

    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.global_feats,
                self.job_static,
                self.job_targets,
                self.job_mask,
                self.unit_pos,
                self.unit_mask,
                self.chosen,
                self.committed,
                self.available,
                self.reward,
                self.raw_reward,
                self.potential,
                self.money,
                self.truncated,
                self.market_state,
                self.market_purchase_static,
                self.market_purchase_mask,
                self.market_reserve_static,
                self.market_reserve_mask,
                self.market_chosen,
                self.market_step,
            )
            if array is not None
        )


EPISODE_ARRAY_FIELDS = (
    "global_feats",
    "job_static",
    "job_targets",
    "job_mask",
    "unit_pos",
    "unit_mask",
    "chosen",
    "committed",
    "available",
    "reward",
    "raw_reward",
    "potential",
    "money",
    "truncated",
)

MARKET_ARRAY_FIELDS = (
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
        "format_version": np.asarray([4], dtype=np.int16),
        "episode_count": np.asarray([len(records)], dtype=np.int32),
    }
    for index, record in enumerate(records):
        for field_name in EPISODE_ARRAY_FIELDS:
            payload[f"e{index}_{field_name}"] = getattr(record, field_name)
        for field_name in MARKET_ARRAY_FIELDS:
            value = getattr(record, field_name)
            if value is not None:
                payload[f"e{index}_{field_name}"] = value
    np.savez_compressed(destination, **payload)
    return destination


def load_episode_batch(path: str | Path) -> list[EpisodeRecord]:
    with np.load(Path(path), allow_pickle=False) as payload:
        version = int(payload["format_version"][0])
        if version not in (3, 4):
            raise EpisodeFormatError(
                f"unsupported DQN episode batch format {version}; adaptive-hybrid features require format 3"
            )
        count = int(payload["episode_count"][0])
        records = []
        for index in range(count):
            values = {field_name: payload[f"e{index}_{field_name}"].copy() for field_name in EPISODE_ARRAY_FIELDS}
            for field_name in MARKET_ARRAY_FIELDS:
                key = f"e{index}_{field_name}"
                values[field_name] = payload[key].copy() if key in payload.files else None
            records.append(EpisodeRecord(**values))
        return records


def potential(farm: dict, private: dict, obs: Any) -> float:
    """Phi(s): liquidatable crop value, excluding cash already in the reward.

    Used for potential-based reward shaping. Buying seeds and hiring are
    immediately negative with a payoff dozens of turns later, so unshaped money
    delta teaches the network to sit still. Shaping of this form is
    policy-invariant when the terminal potential is forced to zero. Cash is
    deliberately absent because the base reward already records its delta.
    """
    prices = (get(obs, "market", {}) or {}).get("prices", {}) or {}
    value = 0.0

    shed = private.get("shed", {}) or {}
    for item in PRODUCTS:
        value += int(shed.get(item, 0)) * float(prices.get(item, PRODUCT_BASE[item]))
    for crop in CROPS:
        value += int((private.get("seeds", {}) or {}).get(crop, 0)) * float(CROP_DATA[crop]["seed"])
    for animal in ANIMAL_DATA:
        value += int(shed.get(animal, 0)) * float(ANIMAL_DATA[animal]["cost"])
    for slot in private.get("inventories", []) or []:
        for item in PRODUCTS:
            value += int((slot or {}).get(item, 0)) * float(prices.get(item, PRODUCT_BASE[item]))
        for animal in ANIMAL_DATA:
            value += int((slot or {}).get(animal, 0)) * float(ANIMAL_DATA[animal]["cost"])

    for row in farm.get("tiles") or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                crop = str(tile.get("crop", ""))
                if crop in CROP_DATA:
                    units = int(tile.get("yield_units", 0))
                    value += units * float(prices.get(crop, CROP_DATA[crop]["base"]))
            animal = tile.get("animal")
            if animal in ANIMAL_DATA:
                data = ANIMAL_DATA[str(animal)]
                product = str(data["product"])
                value += int(tile.get("yield_units", 0)) * float(prices.get(product, data["base"]))
                value += int(bool(tile.get("fertilizer_available"))) * float(
                    prices.get("FERTILIZER", PRODUCT_BASE["FERTILIZER"])
                )
                value += 0.5 * float(prices.get(product, data["base"]))
    return value / MONEY_SCALE


@dataclass
class Recorder:
    """Accumulates one episode of decisions as the agent makes them."""

    global_feats: list[np.ndarray] = field(default_factory=list)
    job_static: list[np.ndarray] = field(default_factory=list)
    job_targets: list[np.ndarray] = field(default_factory=list)
    unit_pos: list[np.ndarray] = field(default_factory=list)
    chosen: list[list[int]] = field(default_factory=list)
    committed: list[list[int]] = field(default_factory=list)
    available: list[np.ndarray] = field(default_factory=list)
    money: list[float] = field(default_factory=list)
    potential: list[float] = field(default_factory=list)
    truncated: list[int] = field(default_factory=list)
    market_steps: list[int] = field(default_factory=list)
    market_decisions: list[MarketDecision] = field(default_factory=list)

    def reset(self) -> None:
        self.global_feats.clear()
        self.job_static.clear()
        self.job_targets.clear()
        self.unit_pos.clear()
        self.chosen.clear()
        self.committed.clear()
        self.available.clear()
        self.money.clear()
        self.potential.clear()
        self.truncated.clear()
        self.market_steps.clear()
        self.market_decisions.clear()

    def record_market(self, *, step: int, decision: MarketDecision) -> None:
        self.market_steps.append(int(step))
        self.market_decisions.append(decision)

    def record(
        self,
        *,
        global_feats,
        job_static,
        job_targets,
        positions,
        chosen,
        committed,
        available,
        money,
        obs,
        farm,
        private,
        truncated,
    ) -> None:
        self.global_feats.append(np.asarray(global_feats, dtype=np.float32))
        self.job_static.append(np.asarray(job_static, dtype=np.float16))
        self.job_targets.append(np.asarray(job_targets, dtype=np.int8))
        self.unit_pos.append(np.asarray(positions, dtype=np.int8))
        self.chosen.append(list(chosen))
        self.committed.append(list(committed))
        self.available.append(np.asarray(available, dtype=bool))
        self.money.append(float(money))
        self.potential.append(potential(farm, private, obs))
        self.truncated.append(int(truncated))

    def finish(self, final_money: float, opponent_money: float, gamma: float) -> EpisodeRecord:
        """Pad to a common width and compute shaped rewards."""
        steps = len(self.global_feats)
        if steps == 0:
            raise ValueError("recorder captured no steps")

        width = max(array.shape[0] for array in self.job_static)
        units = max(array.shape[0] for array in self.unit_pos)

        job_static = np.zeros((steps, width, F.JOB_STATIC_DIM), dtype=np.float16)
        job_targets = np.zeros((steps, width, 2), dtype=np.int8)
        job_mask = np.zeros((steps, width), dtype=bool)
        unit_pos = np.zeros((steps, units, 2), dtype=np.int8)
        unit_mask = np.zeros((steps, units), dtype=bool)
        chosen = np.zeros((steps, units), dtype=np.int16)
        committed = np.full((steps, units), -1, dtype=np.int16)
        available = np.zeros((steps, units, width), dtype=bool)

        for t in range(steps):
            count = self.job_static[t].shape[0]
            job_static[t, :count] = self.job_static[t]
            job_targets[t, :count] = self.job_targets[t]
            job_mask[t, :count] = True
            n_units = self.unit_pos[t].shape[0]
            unit_pos[t, :n_units] = self.unit_pos[t]
            unit_mask[t, :n_units] = True
            chosen[t, :n_units] = self.chosen[t]
            committed[t, :n_units] = self.committed[t]
            available[t, :n_units, :count] = self.available[t]

        money = np.asarray([*self.money, final_money], dtype=np.float32)
        # The environment ends in an absorbing state. Forcing Phi(s_T)=0 keeps
        # shaping from assigning terminal value to inventory that has no score.
        phi = np.asarray([*self.potential, 0.0], dtype=np.float32)

        raw = (money[1:] - money[:-1]) / MONEY_SCALE
        shaped = raw + gamma * phi[1:] - phi[:-1]
        shaped[-1] += float(np.clip((final_money - opponent_money) / TERMINAL_MARGIN_SCALE, -1.0, 1.0))

        market_state = market_purchase_static = market_purchase_mask = None
        market_reserve_static = market_reserve_mask = market_chosen = market_step = None
        if self.market_decisions:
            decisions = len(self.market_decisions)
            purchase_width = max(len(item.candidates.purchase_plans) for item in self.market_decisions)
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
                market_purchase_mask[index, :count] = True
                width = candidates.reserve_static.shape[1]
                market_reserve_static[index, :, :width] = candidates.reserve_static
                market_reserve_mask[index, :, :width] = candidates.reserve_mask
                market_chosen[index] = decision.chosen
            market_step = np.asarray(self.market_steps, dtype=np.int16)

        return EpisodeRecord(
            global_feats=np.stack(self.global_feats),
            job_static=job_static,
            job_targets=job_targets,
            job_mask=job_mask,
            unit_pos=unit_pos,
            unit_mask=unit_mask,
            chosen=chosen,
            committed=committed,
            available=available,
            reward=shaped.astype(np.float32),
            raw_reward=raw.astype(np.float32),
            potential=phi,
            money=money,
            truncated=np.asarray(self.truncated, dtype=np.int32),
            market_state=market_state,
            market_purchase_static=market_purchase_static,
            market_purchase_mask=market_purchase_mask,
            market_reserve_static=market_reserve_static,
            market_reserve_mask=market_reserve_mask,
            market_chosen=market_chosen,
            market_step=market_step,
        )


#: Spec target of the training-only wrapper in `match_agent.py`. Rollouts play
#: through it so an exploring learner is an ordinary agent spec.
LEARNER_ENTRY_POINT = "agents.dqn_job_scorer.match_agent:make_agent"

#: Revision role recorded for the learner seat of a rollout pairing.
LEARNER_ROLE = "learner"


@dataclass(frozen=True)
class DQNRollout:
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
    market_epsilon: float,
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
            "market_epsilon": market_epsilon,
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
    market_epsilon: float | None = None,
    trace_root: str | Path | None = None,
    policy_revision_id: str = "dqn-learner",
) -> DQNRollout | None:
    """Play one episode with an epsilon-greedy learner and record it.

    Goes through `sim.runner.run_match` rather than driving the environment
    directly, so a rollout yields a full `MatchResult` -- per-seat statuses,
    timings, the replay -- and can be committed to the league exactly like a
    scheduled match instead of vanishing into an `.npz`.

    Returns None if the *learner* errored, so a broken policy is dropped rather
    than silently trained on. An episode the opponent crashed out of is kept and
    recorded as failed, which is what the pre-league collector did.
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
            epsilon if market_epsilon is None else market_epsilon,
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
    return DQNRollout(
        load_episode_batch(trace_path)[0],
        opponent_spec,
        learner_seat,
        seed,
        result,
        str(trace_path),
    )


def _job(args: tuple) -> DQNRollout | None:
    return collect_episode(*args)


def collect_batch(
    weights_path: str | Path,
    opponents: list[str],
    seeds: list[int],
    epsilon: float,
    gamma: float,
    jobs: int = 1,
    config: dict[str, Any] | None = None,
    market_epsilon: float | None = None,
    trace_root: str | Path | None = None,
    policy_revision_id: str = "dqn-learner",
) -> list[DQNRollout]:
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
            market_epsilon,
            trace_root,
            policy_revision_id,
        )
        for index, seed in enumerate(seeds)
    ]
    if jobs == 1:
        rollouts = [_job(task) for task in tasks]
    else:
        # Torch is initialized before collection in both trainers. Forking that
        # process can deadlock or terminate workers after the replay grows; a
        # fresh interpreter is slower to start but stable and cross-platform.
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            rollouts = list(pool.map(_job, tasks, chunksize=1))
    return [
        replace(rollout, pair_index=index // 2)
        for index, rollout in enumerate(rollouts)
        if rollout is not None
    ]
