#!/usr/bin/env python3
"""Run the resumable DQN self-play workflow.

One learner, one session. The learner improves by playing frozen snapshots of
its own promoted checkpoints; the fixed benchmarks exist only so promotion has
an external yardstick. Training curriculum and release gate are deliberately
separate knobs -- conflating them is how a policy losing every benchmark game by
~30,000 still scored `benchmark_non_regression: true` against an equally weak
incumbent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

from kaggriculture_lab.database import database_target
from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.model_admin import model_status
from kaggriculture_lab.runner import ensure_session, run_generations

# Edit these experiment constants. The runner intentionally accepts no flags.
GENERATIONS = 5
SYNC_ENVIRONMENT = True

SESSION_ID = "dqn-self-play-v1"
ARTIFACTS = Path("artifacts/dqn_self_play")
LEAGUE_CONFIG = Path("configs/dqn_league.toml")
MODEL_NAME = "default"

# Measurement, not curriculum. These never enter the rollout mix.
BENCHMARKS = (
    "starter",
    "agents/configurable_crop_surrogate/main.py",
)

SELF_PLAY_CONFIG = {
    "initialization": "random",
    # Cold-start floor only. `self_play_pool` is empty until the first
    # checkpoint is promoted, and `compose_opponents` falls back to this base
    # while it is.
    "opponents": ["starter"],
    "benchmarks": list(BENCHMARKS),
    # `benchmark` remains for older readers; promotion evaluates `benchmarks`.
    "benchmark": BENCHMARKS[1],
    "league_opponents": False,
    # No live peers and no cross-instance history: an empty `history_peer_types`
    # routes opponent selection to `self_play_pool`, i.e. this model's own
    # promoted checkpoints.
    "active_peer_types": [],
    "history_peer_types": [],
    "self_play_after": 0,
    "self_play_pool": 5,
    # Against a single-item base this resolves to three self-play slots per
    # base slot, so three of every four rollouts face a frozen snapshot.
    "self_play_ratio": 0.75,
    "episode_steps": 720,
    # Durable observation rounds, so progression is visible in `episodes`
    # rather than only in gate JSON.
    "progress_observation_seeds": 2,
    "progress_opponents": list(BENCHMARKS),
    "progress_seed_base": 1_500_000_000,
}

TRAINING_AGENTS = [
    {
        "type": "dqn_job_scorer",
        "name": MODEL_NAME,
        "train": True,
        "primary": True,
        "model_config": SELF_PLAY_CONFIG,
    }
]

REPO_ROOT = Path(__file__).resolve().parent
MAIN_CONFIG = REPO_ROOT / "main.toml"


def _database():
    return database_target()


def _device() -> str:
    with MAIN_CONFIG.open("rb") as stream:
        return str(tomllib.load(stream).get("compute", {}).get("device", "auto"))


def _prepare_environment() -> None:
    if SYNC_ENVIRONMENT:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to synchronize the environment")
        subprocess.run(
            [uv, "sync", "--locked", "--python", "3.12"],
            cwd=REPO_ROOT,
            check=True,
        )


def _effective_config() -> LeagueConfig:
    base = LeagueConfig.from_toml(REPO_ROOT / LEAGUE_CONFIG)
    return replace(
        base,
        training_agents=tuple(TrainingAgentSpec.from_dict(item) for item in TRAINING_AGENTS),
    )


def _ensure_session() -> None:
    config = _effective_config()
    created, _ = ensure_session(
        config,
        _database(),
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_dqn",
        experiment={"training_agents": TRAINING_AGENTS},
    )
    print(
        f"{'Creating' if created else 'Resuming'} DQN session: {SESSION_ID}",
        flush=True,
    )


def _model_status() -> None:
    print(
        model_status(
            REPO_ROOT / "agents/dqn_job_scorer/agent.toml",
            _database(),
            name=MODEL_NAME,
        )
    )


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_dqn.py accepts no arguments; edit its constants")
    if not isinstance(GENERATIONS, int) or isinstance(GENERATIONS, bool) or GENERATIONS < 1:
        raise ValueError(f"invalid DQN training constant: GENERATIONS={GENERATIONS!r}")

    os.environ["KAGGRICULTURE_TORCH_DEVICE"] = _device()
    _prepare_environment()
    _ensure_session()

    def report(generation: int, _: dict) -> None:
        print(f"\n=== DQN self-play generation {generation}/{GENERATIONS} complete ===", flush=True)
        _model_status()

    run_generations(
        session_id=SESSION_ID,
        database=_database(),
        artifacts=ARTIFACTS,
        generations=GENERATIONS,
        workspace=REPO_ROOT,
        on_generation=report,
    )
    print(f"DQN self-play training complete. Session: {SESSION_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
