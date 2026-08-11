#!/usr/bin/env python3
"""Train the mixed-farm Random-Forest market policy."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from kaggriculture_lab.constants import ARTIFACTS_ROOT, REPO_ROOT
from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.runner import ensure_session, run_generations

GENERATIONS = 10
SESSION_ID = "mixed-farm-rf-market-v1-training"
CONFIG = Path("configs/mixed_farm_rf_market_league.toml")
ARTIFACTS = ARTIFACTS_ROOT / "mixed_farm_rf_market"


def _effective_config() -> LeagueConfig:
    base = LeagueConfig.from_toml(REPO_ROOT / CONFIG)
    instance = TrainingAgentSpec(
        type="mixed_farm_rf_market",
        name="default",
        primary=True,
        train=True,
    )
    return replace(base, training_agents=(instance,))


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_rf_market.py accepts no arguments; edit its constants")
    if GENERATIONS < 1:
        raise ValueError("GENERATIONS must be positive")
    config = _effective_config()
    _, status = ensure_session(
        config,
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_rf_market.py",
        experiment={
            "algorithm": "mixed_farm_random_forest_market_v1",
            "bootstrap": "self_play_heuristic",
            "frozen_field_policy": True,
            "learned_decisions": ["buys", "sales", "land"],
        },
    )
    print(status)
    try:
        run_generations(
            session_id=SESSION_ID,
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
