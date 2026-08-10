#!/usr/bin/env python3
"""Run stable fixed-opponent training for the joint job/market DQN."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

from agents.dqn_job_scorer.policy import PolicyBundle
from kaggriculture_lab.database import database_target
from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.runner import ensure_session, run_generations

GENERATIONS = 5
SESSION_ID = "fixed-opponent-job-market-v1-training"
ARTIFACTS = Path("artifacts/fixed_dqn_shared")
CONFIG = Path("configs/fixed_dqn_league.toml")
BOOTSTRAP_WEIGHTS = Path(
    "artifacts/sp_psro_v4/sp_psro_job_scorer/default/"
    "population_members/member-000001-beta.npz"
)
FIXED_OPPONENTS = (
    "starter",
    "agents/configurable_crop_surrogate/main.py",
    "agents/receding_horizon_planner/main.py",
)
REPO_ROOT = Path(__file__).resolve().parent


def _database():
    return database_target()


def _training_config() -> dict[str, object]:
    return {
        "opponents": list(FIXED_OPPONENTS),
        "benchmark": "agents/configurable_crop_surrogate/main.py",
        "league_opponents": False,
        "active_peer_types": [],
        "history_peer_types": [],
        "self_play_after": 0,
        "self_play_pool": 0,
        "self_play_ratio": 0.0,
        "initialization": "incumbent",
        "incumbent_weights": str(BOOTSTRAP_WEIGHTS),
    }


def _effective_config() -> LeagueConfig:
    base = LeagueConfig.from_toml(REPO_ROOT / CONFIG)
    instance = TrainingAgentSpec(
        type="dqn_job_scorer",
        name="fixed",
        primary=True,
        model_config=_training_config(),
    )
    return replace(base, training_agents=(instance,))


def _bootstrap_digest() -> str:
    source = REPO_ROOT / BOOTSTRAP_WEIGHTS
    if not source.is_file():
        raise FileNotFoundError(
            f"fixed DQN bootstrap checkpoint is missing: {source}; "
            "finish or export a checkpoint before starting this run"
        )
    PolicyBundle.load(source)
    return hashlib.sha256(source.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_fixed_dqn.py accepts no arguments; edit its constants")
    bootstrap_sha256 = _bootstrap_digest()
    config = _effective_config()
    _, status = ensure_session(
        config,
        _database(),
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_fixed_dqn.py",
        experiment={
            "algorithm": "fixed_opponent_joint_job_market_dqn_v1",
            "generations_per_invocation": GENERATIONS,
            "opponents": list(FIXED_OPPONENTS),
            "bootstrap_weights": str(BOOTSTRAP_WEIGHTS),
            "bootstrap_sha256": bootstrap_sha256,
        },
    )
    print(status)
    print(
        "[fixed-dqn] PSRO disabled: one learner, three fixed opponents, "
        "no population admission or self-play",
        flush=True,
    )
    try:
        run_generations(
            session_id=SESSION_ID,
            database=_database(),
            artifacts=ARTIFACTS,
            generations=GENERATIONS,
            continuous=True,
            workspace=REPO_ROOT,
        )
    except KeyboardInterrupt:
        print(f"\n[{SESSION_ID}] stopped; committed work remains resumable", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
