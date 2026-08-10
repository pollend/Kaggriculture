#!/usr/bin/env python3
"""Train the mixed-farm Random-Forest market policy."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from kaggriculture_lab.database import database_target
from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.runner import ensure_session, run_generations
from kaggriculture_lab.storage import ExperimentStore

GENERATIONS = 10
SESSION_ID = "mixed-farm-rf-market-v1-training"
CONFIG = Path("configs/mixed_farm_rf_market_league.toml")
ARTIFACTS = Path("artifacts/mixed_farm_rf_market")
BOOTSTRAP_SESSION = "configurable-mixed-farm-v4-training"
REPO_ROOT = Path(__file__).resolve().parent


def _database():
    return database_target()


def _effective_config() -> LeagueConfig:
    base = LeagueConfig.from_toml(REPO_ROOT / CONFIG)
    instance = TrainingAgentSpec(
        type="mixed_farm_rf_market",
        name="default",
        primary=True,
        train=True,
    )
    return replace(base, training_agents=(instance,))


def _validate_bootstrap() -> None:
    with ExperimentStore(_database()) as store:
        row = store.connection.execute(
            """
            SELECT count(*) FROM generations
            WHERE session_id = %s AND status = 'COMPLETE'
            """,
            [BOOTSTRAP_SESSION],
        ).fetchone()
    if row is None or int(row[0]) == 0:
        raise RuntimeError(
            f"RF market training needs at least one completed generation in "
            f"{BOOTSTRAP_SESSION!r}"
        )


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_rf_market.py accepts no arguments; edit its constants")
    if GENERATIONS < 1:
        raise ValueError("GENERATIONS must be positive")
    _validate_bootstrap()
    config = _effective_config()
    _, status = ensure_session(
        config,
        _database(),
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_rf_market.py",
        experiment={
            "algorithm": "mixed_farm_random_forest_market_v1",
            "bootstrap_session": BOOTSTRAP_SESSION,
            "frozen_field_policy": True,
            "learned_decisions": ["buys", "sales", "land"],
        },
    )
    print(status)
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
