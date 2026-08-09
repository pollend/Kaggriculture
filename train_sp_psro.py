#!/usr/bin/env python3
"""Run the isolated resumable SP-PSRO training workflow."""

from __future__ import annotations

import sys
from pathlib import Path

from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.runner import ensure_session, run_generations

GENERATIONS = 5
SESSION_ID = "sp-psro-job-market-v4-training"
DATABASE = Path("artifacts/sp_psro_v4_experiments.duckdb")
ARTIFACTS = Path("artifacts/sp_psro_v4")
CONFIG = Path("configs/sp_psro_joint_league.toml")
REPO_ROOT = Path(__file__).resolve().parent


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_sp_psro.py accepts no arguments; edit its constants")
    config = LeagueConfig.from_toml(REPO_ROOT / CONFIG)
    _, status = ensure_session(
        config,
        DATABASE,
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_sp_psro.py",
        experiment={
            "generations_per_invocation": GENERATIONS,
            "algorithm": "sp_psro_joint_job_market_v4_trainer_only",
        },
    )
    print(status)
    try:
        run_generations(
            session_id=SESSION_ID,
            database=DATABASE,
            artifacts=ARTIFACTS,
            generations=GENERATIONS,
            continuous=True,
            workspace=REPO_ROOT,
            on_generation=lambda _, report: print(report),
        )
    except KeyboardInterrupt:
        print(f"\n[{SESSION_ID}] stopped; committed work remains resumable", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
