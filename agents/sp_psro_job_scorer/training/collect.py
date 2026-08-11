"""Dual-seat population-versus-best-response rollout collection."""

from __future__ import annotations

import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np

from agents.dqn_job_scorer.training.collect import (
    EPISODE_ARRAY_FIELDS,
    MARKET_ARRAY_FIELDS,
    EpisodeFormatError,
    EpisodeRecord,
    load_episode_batch,
)
from kaggriculture_lab.episode_metrics import episode_metrics
from sim.runner import MatchResult, run_match

FORMAT_VERSION = 3


@dataclass(frozen=True)
class PopulationRollout:
    new_policy: EpisodeRecord
    best_response: EpisodeRecord
    behavior_kind: str
    behavior_member_id: str | None
    seed: int
    match: MatchResult | None = None
    beta_seat: int = 0
    sampling_probability: float = 1.0
    population_payoff: float = 0.0


@dataclass(frozen=True)
class RolloutArtifacts:
    population_path: str
    beta_path: str
    behavior_member_id: str
    seed: int
    match: MatchResult
    beta_seat: int


def collect_episode(
    behavior_weights: str | Path,
    beta_weights: str | Path,
    behavior_kind: str,
    behavior_member_id: str | None,
    seed: int,
    epsilon: float,
    gamma: float,
    beta_seat: int,
    config: dict[str, Any] | None = None,
    market_epsilon: float | None = None,
) -> RolloutArtifacts | None:
    """Run two manifest agents; workers exchange paths and IDs, never Q arrays."""
    trace_root = Path(str((config or {}).get("trace_root", ".")))
    token = f"{seed}-{beta_seat}-{behavior_member_id}"
    population_trace = trace_root / f"{token}-population.npz"
    beta_trace = trace_root / f"{token}-beta.npz"

    def spec(
        role: str,
        revision: str,
        weights: str | Path,
        trace: Path,
        salt: int,
        policy_epsilon: float,
    ) -> str:
        query = urlencode(
            {
                "role": role,
                "policy_revision_id": revision,
                "weights": str(Path(weights).resolve()),
                "trace_path": str(trace.resolve()),
                "epsilon": policy_epsilon,
                "market_epsilon": (
                    policy_epsilon if market_epsilon is None else (market_epsilon if policy_epsilon > 0 else 0.0)
                ),
                "gamma": gamma,
                "exploration_seed": seed * salt + beta_seat,
            }
        )
        return f"agents.sp_psro_job_scorer.match_agent:make_agent?{query}"

    behavior_epsilon = epsilon if behavior_kind == "new_policy" else 0.0
    population_spec = spec(
        behavior_kind,
        str(behavior_member_id),
        behavior_weights,
        population_trace,
        7919,
        behavior_epsilon,
    )
    beta_digest = hashlib.sha256(Path(beta_weights).read_bytes()).hexdigest()[:20]
    beta_spec = spec(
        "best_response",
        f"beta-{beta_digest}",
        beta_weights,
        beta_trace,
        104729,
        epsilon,
    )
    specs = [population_spec, beta_spec]
    if beta_seat == 0:
        specs.reverse()
    result = run_match(
        tuple(specs),
        seed,
        config={k: v for k, v in (config or {}).items() if k != "trace_root"},
        analyzer=episode_metrics,
    )
    if result.failed:
        return None
    return RolloutArtifacts(str(population_trace), str(beta_trace), str(behavior_member_id), seed, result, beta_seat)


def _job(args: tuple) -> RolloutArtifacts | None:
    return collect_episode(*args)


def collect_batch(
    population: list[tuple[str, str | Path]],
    nu_weights: str | Path,
    beta_weights: str | Path,
    seeds: list[int],
    epsilon: float,
    gamma: float,
    jobs: int = 1,
    offset: int = 0,
    config: dict[str, Any] | None = None,
    market_epsilon: float | None = None,
    mixture_weights: list[float] | None = None,
    sampling_seed: int = 0,
) -> list[PopulationRollout]:
    if not population:
        raise ValueError("SP-PSRO population cannot be empty")
    if len(seeds) % 2:
        raise ValueError("episodes_per_batch must be even for paired seat rotation")
    arms = [*population, ("live-nu", nu_weights)]
    if mixture_weights is None:
        # Backward-compatible scheduling for callers that do not provide a
        # learned restricted distribution: frozen population only.
        probabilities = np.asarray([1.0 / len(population)] * len(population) + [0.0])
        pair_arms = np.asarray([((offset // 2) + pair) % len(population) for pair in range(len(seeds) // 2)])
    else:
        probabilities = np.asarray(mixture_weights, dtype=np.float64)
        if probabilities.shape != (len(arms),) or np.any(probabilities < 0):
            raise ValueError("mixture weights must match population plus live nu")
        probabilities /= probabilities.sum()
        rng = np.random.default_rng(sampling_seed)
        pair_arms = rng.choice(len(arms), size=len(seeds) // 2, p=probabilities)
    tasks = []
    task_probabilities = []
    for index, seed in enumerate(seeds):
        arm_index = int(pair_arms[index // 2])
        member_id, behavior_weights = arms[arm_index]
        behavior_kind = "new_policy" if member_id == "live-nu" else "population"
        tasks.append(
            (
                behavior_weights,
                beta_weights,
                behavior_kind,
                member_id,
                seed,
                epsilon,
                gamma,
                (offset + index) % 2,
                config,
                market_epsilon,
            )
        )
        task_probabilities.append(float(probabilities[arm_index]))
    if jobs == 1:
        records = [_job(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            records = list(pool.map(_job, tasks, chunksize=1))
    # Keep the scheduling seam independently testable without constructing
    # full episode arrays in a monkeypatched worker.
    if any(record is not None and not isinstance(record, RolloutArtifacts) for record in records):
        return [record for record in records if record is not None]  # type: ignore[return-value]
    rollouts = []
    for artifact, sampling_probability in zip(records, task_probabilities):
        if artifact is None:
            continue
        population_record = load_episode_batch(artifact.population_path)[0]
        beta_record = load_episode_batch(artifact.beta_path)[0]
        margin = float(artifact.match.margin or 0.0)
        population_margin = -margin if artifact.beta_seat == 0 else margin
        rollouts.append(
            PopulationRollout(
                population_record,
                beta_record,
                "new_policy" if artifact.behavior_member_id == "live-nu" else "population",
                artifact.behavior_member_id,
                artifact.seed,
                artifact.match,
                artifact.beta_seat,
                sampling_probability,
                float(np.sign(population_margin)),
            )
        )
    return rollouts


def save_batch(records: list[PopulationRollout], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray([FORMAT_VERSION], dtype=np.int16),
        "episode_count": np.asarray([len(records)], dtype=np.int32),
        "behavior_kind": np.asarray([r.behavior_kind for r in records]),
        "behavior_member_id": np.asarray([r.behavior_member_id or "" for r in records]),
        "seeds": np.asarray([r.seed for r in records], dtype=np.int64),
        "beta_seats": np.asarray([r.beta_seat for r in records], dtype=np.int8),
        "sampling_probability": np.asarray([r.sampling_probability for r in records], dtype=np.float64),
        "population_payoff": np.asarray([r.population_payoff for r in records], dtype=np.float32),
    }
    for index, rollout in enumerate(records):
        for role, episode in (("nu", rollout.new_policy), ("beta", rollout.best_response)):
            for field_name in EPISODE_ARRAY_FIELDS:
                payload[f"e{index}_{role}_{field_name}"] = getattr(episode, field_name)
            for field_name in MARKET_ARRAY_FIELDS:
                value = getattr(episode, field_name)
                if value is not None:
                    payload[f"e{index}_{role}_{field_name}"] = value
    np.savez_compressed(destination, **payload)
    return destination


def load_batch(path: str | Path) -> list[PopulationRollout]:
    with np.load(Path(path), allow_pickle=False) as payload:
        version = int(payload["format_version"][0])
        if version not in (1, 2, FORMAT_VERSION):
            raise EpisodeFormatError(f"unsupported SP-PSRO rollout format {version}")
        count = int(payload["episode_count"][0])
        result = []
        for index in range(count):
            episodes = {}
            for role in ("nu", "beta"):
                values = {field: payload[f"e{index}_{role}_{field}"].copy() for field in EPISODE_ARRAY_FIELDS}
                for field in MARKET_ARRAY_FIELDS:
                    key = f"e{index}_{role}_{field}"
                    values[field] = payload[key].copy() if key in payload.files else None
                episodes[role] = EpisodeRecord(**values)
            member = str(payload["behavior_member_id"][index])
            result.append(
                PopulationRollout(
                    episodes["nu"],
                    episodes["beta"],
                    str(payload["behavior_kind"][index]),
                    member or None,
                    int(payload["seeds"][index]),
                    beta_seat=(int(payload["beta_seats"][index]) if "beta_seats" in payload.files else 0),
                    sampling_probability=(
                        float(payload["sampling_probability"][index])
                        if "sampling_probability" in payload.files
                        else 1.0
                    ),
                    population_payoff=(
                        float(payload["population_payoff"][index]) if "population_payoff" in payload.files else 0.0
                    ),
                )
            )
        return result


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
