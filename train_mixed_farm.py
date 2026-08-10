#!/usr/bin/env python3
"""Run the isolated configurable mixed-farm surrogate league."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from kaggriculture_lab.database import database_target
from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.runner import ensure_session, run_generations

GENERATIONS = 48
POPULATION_SIZE = 20
ROUNDS_PER_GENERATION = 24
HALL_OF_FAME_ROUNDS = 12
EPISODE_STEPS = 720
TRAINING_JOBS = 0
TRAINING_SEED = 2718
CANDIDATE_POOL = 10_000
RUN_ACCEPTANCE_ARENA = True
ACCEPTANCE_SEEDS = 20
ACCEPTANCE_JOBS = 24
SYNC_ENVIRONMENT = True

SESSION_ID = "configurable-mixed-farm-v3-training"
ARTIFACTS = Path("artifacts")
LEAGUE_CONFIG = Path("configs/mixed_farm_league.toml")
TRAINING_AGENTS = [
    {
        "type": "configurable_mixed_farm_surrogate",
        "name": "v3",
        "primary": True,
        "train": True,
    }
]

REPO_ROOT = Path(__file__).resolve().parent


def _database():
    return database_target()


def _run(command: list[str], *, check: bool = True, quiet: bool = False):
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def _prepare_environment() -> None:
    if SYNC_ENVIRONMENT:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to synchronize the environment")
        _run([uv, "sync", "--locked", "--python", "3.12"])


def _effective_config() -> LeagueConfig:
    base = LeagueConfig.from_toml(REPO_ROOT / LEAGUE_CONFIG)
    return replace(
        base,
        name=SESSION_ID,
        population_size=POPULATION_SIZE,
        rounds_per_generation=ROUNDS_PER_GENERATION,
        hall_of_fame_rounds=HALL_OF_FAME_ROUNDS,
        episode_steps=EPISODE_STEPS,
        jobs=TRAINING_JOBS,
        seed=TRAINING_SEED,
        candidate_pool=CANDIDATE_POOL,
        training_agents=tuple(
            TrainingAgentSpec.from_dict(item) for item in TRAINING_AGENTS
        ),
    )


def _ensure_session() -> None:
    config = _effective_config()
    created, _ = ensure_session(
        config,
        _database(),
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_mixed_farm",
    )
    print(
        f"{'Creating' if created else 'Resuming'} mixed-farm session: {SESSION_ID}",
        flush=True,
    )


def _run_acceptance() -> None:
    _run(
        [
            sys.executable,
            "-m",
            "agents.configurable_mixed_farm_surrogate.analysis",
            "--seeds",
            str(ACCEPTANCE_SEEDS),
            "--jobs",
            str(ACCEPTANCE_JOBS),
            "--broad",
        ]
    )


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_mixed_farm.py accepts no arguments; edit its constants")
    positive = {
        "GENERATIONS": GENERATIONS,
        "POPULATION_SIZE": POPULATION_SIZE,
        "ROUNDS_PER_GENERATION": ROUNDS_PER_GENERATION,
        "EPISODE_STEPS": EPISODE_STEPS,
        "CANDIDATE_POOL": CANDIDATE_POOL,
        "ACCEPTANCE_SEEDS": ACCEPTANCE_SEEDS,
        "ACCEPTANCE_JOBS": ACCEPTANCE_JOBS,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid or TRAINING_JOBS < 0:
        raise ValueError(f"invalid training constants: {invalid or ['TRAINING_JOBS']}")

    _prepare_environment()
    _ensure_session()
    run_generations(
        session_id=SESSION_ID,
        database=_database(),
        artifacts=ARTIFACTS,
        generations=GENERATIONS,
        workspace=REPO_ROOT,
        on_generation=lambda generation, _: print(
            f"\n=== Mixed-farm generation {generation}/{GENERATIONS} complete ===",
            flush=True,
        ),
    )

    if RUN_ACCEPTANCE_ARENA:
        _run_acceptance()
    print(f"Mixed-farm training complete. Session: {SESSION_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
