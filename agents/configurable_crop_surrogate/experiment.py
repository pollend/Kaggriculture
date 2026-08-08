"""Deterministic staged tournament orchestration."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
import os
import random
import subprocess
import tomllib
import uuid
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from kaggriculture_lab.episode_metrics import episode_metrics
from kaggriculture_lab.storage import ExperimentStore
from agents.configurable_crop_surrogate.strategy import Strategy, generate_population
from sim.runner import MatchResult, run_match


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str = "experiment"
    seed: int = 42
    jobs: int = 1
    episode_steps: int = 720
    population_size: int = 1000
    replay_policy: str = "failures"
    stage1_seeds: int = 3
    stage1_keep: int = 500
    stage2_opponents: int = 20
    stage2_seeds: int = 2
    stage2_keep: int = 50
    stage3_seeds: int = 3
    candidate_pool: int = 100_000
    proposal_count: int = 1000

    def __post_init__(self) -> None:
        if self.replay_policy not in {"none", "failures", "all"}:
            raise ValueError("replay_policy must be none, failures, or all")
        positive = (
            "population_size",
            "episode_steps",
            "stage1_seeds",
            "stage1_keep",
            "stage2_seeds",
            "stage2_keep",
            "stage3_seeds",
            "candidate_pool",
            "proposal_count",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_toml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("rb") as stream:
            return cls(**tomllib.load(stream))


@dataclass(frozen=True, slots=True)
class MatchSpec:
    match_id: str
    run_id: str
    stage: str
    seed: int
    strategy_ids: tuple[str, str]
    specs: tuple[str, str]
    environment: dict[str, Any]
    keep_replay: bool


def strategy_spec(strategy: Strategy, episode_steps: int) -> str:
    values = {**strategy.to_dict(), "episode_steps": episode_steps}
    return f"agents.configurable_crop_surrogate.agent:make_agent?{urlencode(values)}"


def _worker(spec: MatchSpec) -> tuple[MatchSpec, MatchResult]:
    return spec, run_match(
        spec.specs, spec.seed, spec.environment, spec.keep_replay, analyzer=episode_metrics
    )


def _match_id(
    run_id: str, stage: str, seed: int, strategy_ids: tuple[str, str]
) -> str:
    raw = json.dumps([run_id, stage, seed, *strategy_ids], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _versions() -> dict[str, str]:
    result = {"python": os.sys.version.split()[0]}
    for package in ("kaggle-environments", "duckdb", "pandas", "scikit-learn"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        store: ExperimentStore,
        artifacts: str | Path = "artifacts",
        run_id: str | None = None,
    ):
        self.config = config
        self.store = store
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{timestamp}-{uuid.uuid4().hex[:8]}"
        self.artifacts = Path(artifacts)
        self.run_dir = self.artifacts / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> str:
        config_dict = asdict(self.config)
        self.store.begin_run(
            self.run_id,
            self.config.name,
            self.config.seed,
            config_dict,
            _versions(),
            _git_commit(),
        )
        (self.run_dir / "manifest.json").write_text(
            json.dumps(
                {"run_id": self.run_id, "config": config_dict, "versions": _versions()},
                indent=2,
                sort_keys=True,
            )
        )
        try:
            population = generate_population(self.config.population_size, self.config.seed)
            self.store.add_strategies(population, "generated", self.run_id)
            by_id = {strategy.strategy_id: strategy for strategy in population}

            self._execute(self._stage1(population))
            stage2_ids = self.store.ranked_strategy_ids(
                self.run_id, "stage1", min(self.config.stage1_keep, len(population))
            )
            self._execute(self._stage2([by_id[item] for item in stage2_ids]))
            stage3_ids = self.store.ranked_strategy_ids(
                self.run_id, "stage2", min(self.config.stage2_keep, len(stage2_ids))
            )
            self._execute(self._stage3([by_id[item] for item in stage3_ids]))
            self.store.finish_run(self.run_id)
            self.store.export(
                self.run_id, self.artifacts / "parquet" / self.run_id, csv=False
            )
        except Exception:
            self.store.finish_run(self.run_id, "FAILED")
            raise
        return self.run_id

    def _stage1(self, strategies: list[Strategy]) -> list[MatchSpec]:
        matches: list[MatchSpec] = []
        builtins = (
            ("builtin:pass", "pass"),
            ("builtin:random", "seeded_random"),
            ("builtin:starter", "starter"),
        )
        for strategy in strategies:
            own_spec = strategy_spec(strategy, self.config.episode_steps)
            for builtin_id, builtin_spec in builtins:
                for offset in range(self.config.stage1_seeds):
                    seed = self.config.seed + offset
                    actual_builtin = (
                        f"seeded_random?seed={seed}" if builtin_spec == "seeded_random" else builtin_spec
                    )
                    matches.extend(
                        self._paired_specs(
                            "stage1",
                            seed,
                            (strategy.strategy_id, builtin_id),
                            (own_spec, actual_builtin),
                        )
                    )
        return matches

    def _stage2(self, strategies: list[Strategy]) -> list[MatchSpec]:
        if len(strategies) < 2:
            return []
        rng = random.Random(self.config.seed + 20_000)
        pairs: set[tuple[str, str]] = set()
        by_id = {strategy.strategy_id: strategy for strategy in strategies}
        degree = min(self.config.stage2_opponents, len(strategies) - 1)
        ids = list(by_id)
        rng.shuffle(ids)
        # A ring graph gives every strategy the requested degree without the
        # near-doubling caused by independent per-strategy opponent sampling.
        for offset in range(1, degree // 2 + 1):
            for index, left_id in enumerate(ids):
                right_id = ids[(index + offset) % len(ids)]
                pairs.add(tuple(sorted((left_id, right_id))))
        if degree % 2:
            if len(ids) % 2 == 0:
                half = len(ids) // 2
                for index in range(half):
                    pairs.add(tuple(sorted((ids[index], ids[index + half]))))
            else:
                for index in range(0, len(ids) - 1, 2):
                    pairs.add(tuple(sorted((ids[index], ids[index + 1]))))
        matches: list[MatchSpec] = []
        for left_id, right_id in sorted(pairs):
            for offset in range(self.config.stage2_seeds):
                seed = self.config.seed + 100_000 + offset
                matches.extend(
                    self._paired_specs(
                        "stage2",
                        seed,
                        (left_id, right_id),
                        (
                            strategy_spec(by_id[left_id], self.config.episode_steps),
                            strategy_spec(by_id[right_id], self.config.episode_steps),
                        ),
                    )
                )
        return matches

    def _stage3(self, strategies: list[Strategy]) -> list[MatchSpec]:
        matches: list[MatchSpec] = []
        for left, right in itertools.combinations(sorted(strategies, key=lambda item: item.strategy_id), 2):
            for offset in range(self.config.stage3_seeds):
                seed = self.config.seed + 200_000 + offset
                matches.extend(
                    self._paired_specs(
                        "stage3",
                        seed,
                        (left.strategy_id, right.strategy_id),
                        (
                            strategy_spec(left, self.config.episode_steps),
                            strategy_spec(right, self.config.episode_steps),
                        ),
                    )
                )
        return matches

    def _paired_specs(
        self,
        stage: str,
        seed: int,
        strategy_ids: tuple[str, str],
        specs: tuple[str, str],
    ) -> list[MatchSpec]:
        environment = {"episodeSteps": self.config.episode_steps, "actTimeout": 1.0}
        keep = self.config.replay_policy != "none"
        result = []
        for orientation in (0, 1):
            ids = strategy_ids if orientation == 0 else (strategy_ids[1], strategy_ids[0])
            agent_specs = specs if orientation == 0 else (specs[1], specs[0])
            result.append(
                MatchSpec(
                    match_id=_match_id(self.run_id, stage, seed, ids),
                    run_id=self.run_id,
                    stage=stage,
                    seed=seed,
                    strategy_ids=ids,
                    specs=agent_specs,
                    environment=environment,
                    keep_replay=keep,
                )
            )
        return result

    def _execute(self, specs: Iterable[MatchSpec]) -> None:
        completed = self.store.completed_match_ids(self.run_id)
        pending = [spec for spec in specs if spec.match_id not in completed]
        if not pending:
            return
        jobs = self.config.jobs if self.config.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
        print(f"{self.run_id}: running {len(pending)} {pending[0].stage} matches with {jobs} worker(s)")
        if jobs == 1:
            outcomes = map(_worker, pending)
            pool = None
        else:
            pool = ProcessPoolExecutor(
                max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
            )
            outcomes = pool.map(_worker, pending, chunksize=1)
        try:
            for index, (spec, result) in enumerate(outcomes, 1):
                replay_path = self._save_replay(spec, result)
                self.store.add_match(
                    spec.match_id,
                    self.run_id,
                    spec.stage,
                    spec.strategy_ids,
                    spec.specs,
                    result,
                    replay_path,
                )
                if index % 100 == 0 or index == len(pending):
                    print(f"  {spec.stage}: {index}/{len(pending)}")
        finally:
            if pool is not None:
                pool.shutdown()

    def _save_replay(self, spec: MatchSpec, result: MatchResult) -> str | None:
        save = self.config.replay_policy == "all" or (
            self.config.replay_policy == "failures" and result.failed
        )
        if not save or result.replay is None:
            return None
        directory = self.run_dir / "replays"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{spec.match_id}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(result.replay, stream, separators=(",", ":"))
        return str(path)


def run_experiment(
    config: ExperimentConfig,
    database: str | Path = "artifacts/experiments.duckdb",
    artifacts: str | Path = "artifacts",
    run_id: str | None = None,
) -> str:
    with ExperimentStore(database) as store:
        return ExperimentRunner(config, store, artifacts, run_id).run()
