#!/usr/bin/env python3
"""Run the isolated mixed-farm v5 league."""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from kaggriculture_lab.constants import ARTIFACTS_ROOT, DEFAULT_GENERATIONS, REPO_ROOT
from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.runner import ensure_session, run_generations

GENERATIONS = DEFAULT_GENERATIONS
POPULATION_SIZE = 20
ROUNDS_PER_GENERATION = 28
ANCHOR_ROUNDS = 12
HALL_OF_FAME_ROUNDS = 4
EPISODE_STEPS = 720
TRAINING_JOBS = 0
TRAINING_SEED = 2718
CANDIDATE_POOL = 10_000
SYNC_ENVIRONMENT = True

SESSION_ID = "configurable-mixed-farm-v5-training"
ARTIFACTS = ARTIFACTS_ROOT
LEAGUE_CONFIG = Path("configs/mixed_farm_league.toml")
TRAINING_AGENTS = [
    {
        "type": "mixed_farm_v5",
        "name": "v5",
        "primary": True,
        "train": True,
    }
]


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
        anchor_rounds=ANCHOR_ROUNDS,
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
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_mixed_farm",
    )
    print(
        f"{'Creating' if created else 'Resuming'} mixed-farm session: {SESSION_ID}",
        flush=True,
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
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid or TRAINING_JOBS < 0:
        raise ValueError(f"invalid training constants: {invalid or ['TRAINING_JOBS']}")

    _prepare_environment()
    _ensure_session()
    run_generations(
        session_id=SESSION_ID,
        artifacts=ARTIFACTS,
        generations=GENERATIONS,
        workspace=REPO_ROOT,
        on_generation=lambda generation, _: print(
            f"\n=== Mixed-farm generation {generation}/{GENERATIONS} complete ===",
            flush=True,
        ),
    )

    print(f"Mixed-farm training complete. Session: {SESSION_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
