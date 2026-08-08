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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn_job_scorer import features as F
from agents.dqn_job_scorer.jobs import CROP_ORDER
from agents.dqn_job_scorer.main import JobScorerAgent
from agents.dqn_job_scorer.policy import QNetwork
from agents.dqn_job_scorer.rules import CROP_DATA, get

MONEY_SCALE = 1000.0
WIN_BONUS = 1.0


@dataclass
class EpisodeRecord:
    """One episode, padded to a common job/unit width with masks."""

    global_feats: np.ndarray   # float32 [T, GLOBAL_DIM]
    job_static: np.ndarray     # float16 [T, J, JOB_STATIC_DIM]
    job_targets: np.ndarray    # int8    [T, J, 2]
    job_mask: np.ndarray       # bool    [T, J]
    unit_pos: np.ndarray       # int8    [T, U, 2]
    unit_mask: np.ndarray      # bool    [T, U]
    chosen: np.ndarray         # int16   [T, U]
    committed: np.ndarray      # int16   [T, U]  job index this unit was already walking to, -1 if none
    available: np.ndarray      # bool    [T, U, J] exact live action mask for each unit
    reward: np.ndarray         # float32 [T]   shaped, ready for training
    raw_reward: np.ndarray     # float32 [T]   unshaped money delta, for diagnostics
    potential: np.ndarray      # float32 [T+1] Phi(s_t); last entry is the terminal state
    money: np.ndarray          # float32 [T+1]
    truncated: np.ndarray      # int32   [T]   candidate-set overflow, counted not hidden

    @property
    def steps(self) -> int:
        return self.global_feats.shape[0]

    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.global_feats, self.job_static, self.job_targets, self.job_mask,
                self.unit_pos, self.unit_mask, self.chosen, self.committed, self.available, self.reward,
                self.raw_reward, self.potential, self.money, self.truncated,
            )
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


class EpisodeFormatError(ValueError):
    """A rollout artifact cannot be trained with the current replay semantics."""


def save_episode_batch(records: list[EpisodeRecord], path: str | Path) -> Path:
    """Persist variable-width episode records without pickle/object arrays."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray([2], dtype=np.int16),
        "episode_count": np.asarray([len(records)], dtype=np.int32),
    }
    for index, record in enumerate(records):
        for field_name in EPISODE_ARRAY_FIELDS:
            payload[f"e{index}_{field_name}"] = getattr(record, field_name)
    np.savez_compressed(destination, **payload)
    return destination


def load_episode_batch(path: str | Path) -> list[EpisodeRecord]:
    with np.load(Path(path), allow_pickle=False) as payload:
        version = int(payload["format_version"][0])
        if version != 2:
            raise EpisodeFormatError(
                f"unsupported DQN episode batch format {version}; "
                "exact live action masks require format 2"
            )
        count = int(payload["episode_count"][0])
        return [
            EpisodeRecord(
                **{
                    field_name: payload[f"e{index}_{field_name}"].copy()
                    for field_name in EPISODE_ARRAY_FIELDS
                }
            )
            for index in range(count)
        ]


def potential(farm: dict, private: dict, obs: Any) -> float:
    """Phi(s): liquidatable crop value, excluding cash already in the reward.

    Used for potential-based reward shaping. Buying seeds and hiring are
    immediately negative with a payoff dozens of turns later, so unshaped money
    delta teaches the network to sit still. Shaping of this form is
    policy-invariant when the terminal potential is forced to zero. Cash is
    deliberately absent because the base reward already records its delta.
    """
    prices = ((get(obs, "market", {}) or {}).get("prices", {}) or {})
    value = 0.0

    shed = private.get("shed", {}) or {}
    for crop in CROP_ORDER:
        value += int(shed.get(crop, 0)) * float(prices.get(crop, CROP_DATA[crop]["base"]))
    for slot in private.get("inventories", []) or []:
        for crop in CROP_ORDER:
            value += int((slot or {}).get(crop, 0)) * float(prices.get(crop, CROP_DATA[crop]["base"]))

    for row in farm.get("tiles") or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = str(tile.get("crop", ""))
                if crop in CROP_DATA:
                    units = int(tile.get("yield_units", 0))
                    value += units * float(prices.get(crop, CROP_DATA[crop]["base"]))
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

    def record(self, *, global_feats, job_static, job_targets, positions, chosen,
               committed, available, money, obs, farm, private, truncated) -> None:
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

    def finish(self, final_money: float, outcome: float, gamma: float) -> EpisodeRecord:
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
        shaped[-1] += (2.0 * float(outcome) - 1.0) * WIN_BONUS

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
        )


def collect_episode(
    weights: list[tuple[np.ndarray, np.ndarray]],
    opponent_spec: str,
    seed: int,
    epsilon: float,
    gamma: float,
    learner_seat: int = 0,
    config: dict[str, Any] | None = None,
) -> EpisodeRecord | None:
    """Play one episode with an epsilon-greedy learner and record it.

    Returns None if the learner errored, so a broken policy is dropped rather
    than silently trained on -- the same rule `sim/runner.py` applies to matches.
    """
    from kaggle_environments import make

    from sim import registry
    from sim.runner import OK_STATUSES

    recorder = Recorder()
    network = QNetwork(weights)
    learner = JobScorerAgent(
        network,
        epsilon=epsilon,
        rng=np.random.default_rng(seed * 7919 + learner_seat),
        recorder=recorder,
    )
    opponent = registry.resolve(opponent_spec)

    seats = [opponent, opponent]
    seats[learner_seat] = learner

    env = make("kaggriculture", configuration={**(config or {}), "seed": seed}, debug=False)
    env.run(seats)

    for state in env.steps:
        if state[learner_seat].status not in OK_STATUSES:
            return None

    final = env.steps[-1]
    obs = final[learner_seat].observation
    farms = get(obs, "farms", []) or []
    farm = farms[learner_seat]

    rewards = [state.reward or 0.0 for state in final]
    opponent_reward = max(r for i, r in enumerate(rewards) if i != learner_seat)
    outcome = (
        1.0
        if rewards[learner_seat] > opponent_reward
        else 0.0
        if rewards[learner_seat] < opponent_reward
        else 0.5
    )

    return recorder.finish(
        final_money=float(farm.get("money", 0.0)),
        outcome=outcome,
        gamma=gamma,
    )


def _job(args: tuple) -> EpisodeRecord | None:
    return collect_episode(*args)


def collect_batch(
    weights: list[tuple[np.ndarray, np.ndarray]],
    opponents: list[str],
    seeds: list[int],
    epsilon: float,
    gamma: float,
    jobs: int = 1,
    config: dict[str, Any] | None = None,
) -> list[EpisodeRecord]:
    """Collect one episode per seed, alternating which seat the learner takes.

    Both seats are used because the environment is symmetric and a policy that
    only ever plays seat 0 has seen half the states it will meet in the arena.
    """
    tasks = [
        (weights, opponents[index % len(opponents)], seed, epsilon, gamma, index % 2, config)
        for index, seed in enumerate(seeds)
    ]
    if jobs == 1:
        records = [_job(task) for task in tasks]
    else:
        # Torch is initialized before collection in both trainers. Forking that
        # process can deadlock or terminate workers after the replay grows; a
        # fresh interpreter is slower to start but stable and cross-platform.
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            records = list(pool.map(_job, tasks, chunksize=1))
    return [record for record in records if record is not None]
