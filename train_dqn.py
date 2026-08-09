#!/usr/bin/env python3
"""Run the isolated, resumable multi-DQN self-play workflow."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.model_admin import model_status
from kaggriculture_lab.runner import ensure_session, run_generations
from sim.arena import play_series
from sim.metrics import render, summarize

# Edit these experiment constants. The runner intentionally accepts no flags.
GENERATIONS = 5  # 3 collectors * 5 * 20 * 16 = 4,800 rollout games
RUN_FINAL_ARENA = False  # enable after explicitly exporting a reviewed active checkpoint
FINAL_ARENA_SEEDS = 100
FINAL_ARENA_JOBS = 8
SYNC_ENVIRONMENT = True

SESSION_ID = "dqn-self-play-training"
DATABASE = Path("artifacts/dqn_self_play_experiments.duckdb")
ARTIFACTS = Path("artifacts/dqn_self_play")
LEAGUE_CONFIG = Path("configs/dqn_league.toml")
FROZEN_DQN = (
    "agents.dqn_job_scorer.main:agent"
    "?network_seed=0"
)
DQN_SELF_PLAY_CONFIG = {
    "opponents": [FROZEN_DQN],
    "benchmark": FROZEN_DQN,
    "league_opponents": False,
    "active_peer_types": ["dqn_job_scorer"],
    "history_peer_types": ["dqn_job_scorer"],
    "self_play_after": 0,
    "self_play_pool": 5,
    "self_play_ratio": 0.3,
}
TRAINING_AGENTS = [
    {
        "type": "dqn_job_scorer",
        "name": name,
        "train": True,
        "model_config": DQN_SELF_PLAY_CONFIG,
    }
    for name in ("default", "peer_a", "peer_b")
]
REPO_ROOT = Path(__file__).resolve().parent
MAIN_CONFIG = REPO_ROOT / "main.toml"


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
        DATABASE,
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
    for name in ("default", "peer_a", "peer_b"):
        print(
            model_status(
                REPO_ROOT / "agents/dqn_job_scorer/agent.toml",
                DATABASE,
                name=name,
            )
        )


def _final_arena() -> None:
    if not RUN_FINAL_ARENA:
        return
    opponents = [FROZEN_DQN]
    agent = "agents/dqn_job_scorer/main.py"
    for opponent in opponents:
        results = play_series(
            agent,
            opponent,
            range(FINAL_ARENA_SEEDS),
            {},
            FINAL_ARENA_JOBS,
        )
        print(render(summarize(agent, opponent, results)))


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_dqn.py accepts no arguments; edit its constants")
    positive = {
        "GENERATIONS": GENERATIONS,
        "FINAL_ARENA_SEEDS": FINAL_ARENA_SEEDS,
        "FINAL_ARENA_JOBS": FINAL_ARENA_JOBS,
    }
    invalid = [name for name, value in positive.items() if not isinstance(value, int) or isinstance(value, bool) or value < 1]
    if invalid:
        raise ValueError(f"invalid DQN training constants: {invalid}")

    os.environ["KAGGRICULTURE_TORCH_DEVICE"] = _device()
    _prepare_environment()
    _ensure_session()

    def report(generation: int, _: dict) -> None:
        print(f"\n=== DQN self-play generation {generation}/{GENERATIONS} complete ===", flush=True)
        _model_status()

    run_generations(
        session_id=SESSION_ID,
        database=DATABASE,
        artifacts=ARTIFACTS,
        generations=GENERATIONS,
        workspace=REPO_ROOT,
        on_generation=report,
    )
    _final_arena()
    print(f"DQN self-play training complete. Session: {SESSION_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
