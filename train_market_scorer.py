#!/usr/bin/env python3
"""Run the resumable market-scorer workflow.

One learner, one session, one Q-network. The field policy is frozen -- an
unmodified copy of the pilkwang snapshot -- so everything this session trains
is the market layer sitting on top of it.

That makes the measurement unusually clean, and the benchmark is chosen to keep
it that way: the challenger plays the donor, with the same field policy on both
sides of the board. A positive paired margin there is the learned market layer
and nothing else. Point it at a weaker benchmark and the donor's field play
carries the result, so promotion stops being evidence about the thing being
trained.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

from kaggriculture_lab.constants import ARTIFACTS_ROOT, REPO_ROOT
from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.model_admin import model_status
from kaggriculture_lab.runner import ensure_session, run_generations

# Edit these experiment constants. The runner intentionally accepts no flags.
GENERATIONS = 5
SYNC_ENVIRONMENT = True

SESSION_ID = "market-scorer-v1"
ARTIFACTS = ARTIFACTS_ROOT / "dqn_market_scorer"
LEAGUE_CONFIG = Path("configs/dqn_market_league.toml")
MANIFEST = Path("agents/dqn_market_scorer/agent.toml")
MODEL_NAME = "default"

DONOR = "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"

BENCHMARKS = (
    DONOR,
    "agents/configurable_crop_surrogate/main.py",
)

# Overrides only. `register_training_agent` materializes `agent.toml`'s
# `[models.config]` and then applies this on top, so every key here is a
# deliberate departure from the manifest and everything absent is the
# manifest's to decide. In particular the opponent curriculum -- `opponents`,
# `league_opponents`, `self_play_after`, `self_play_ratio` -- is deliberately
# absent: training on a distribution the gate never measures is how a run
# rejects update after update while looking healthy.
MARKET_CONFIG = {
    # A zero network makes every argmax fall to index 0, which is the donor
    # default modifier and a zero reserve. Round one therefore rolls out
    # unmodified pilkwang instead of a random market policy the learner would
    # have to climb back from.
    "initialization": "zeros",
    "benchmarks": list(BENCHMARKS),
    # `benchmark` remains for older readers; promotion evaluates `benchmarks`.
    "benchmark": DONOR,
    # No live peers and no cross-instance history: an empty `history_peer_types`
    # routes opponent selection to `self_play_pool`, i.e. this model's own
    # promoted checkpoints.
    "active_peer_types": [],
    "history_peer_types": [],
    "episode_steps": 720,
    # Durable observation rounds, so progression is visible in `episodes`
    # rather than only in gate JSON.
    "progress_observation_seeds": 2,
    "progress_opponents": list(BENCHMARKS),
    "progress_seed_base": 1_600_000_000,
}

TRAINING_AGENTS = [
    {
        "type": "dqn_market_scorer",
        "name": MODEL_NAME,
        "train": True,
        "primary": True,
        "model_config": MARKET_CONFIG,
    }
]

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
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_market_scorer",
        experiment={"training_agents": TRAINING_AGENTS},
    )
    print(
        f"{'Creating' if created else 'Resuming'} market-scorer session: {SESSION_ID}",
        flush=True,
    )


def _model_status() -> None:
    print(model_status(REPO_ROOT / MANIFEST, name=MODEL_NAME))


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_market_scorer.py accepts no arguments; edit its constants")
    if not isinstance(GENERATIONS, int) or isinstance(GENERATIONS, bool) or GENERATIONS < 1:
        raise ValueError(
            f"invalid market-scorer training constant: GENERATIONS={GENERATIONS!r}"
        )

    os.environ["KAGGRICULTURE_TORCH_DEVICE"] = _device()
    _prepare_environment()
    _ensure_session()

    def report(generation: int, _: dict) -> None:
        print(
            f"\n=== market-scorer generation {generation}/{GENERATIONS} complete ===",
            flush=True,
        )
        _model_status()

    run_generations(
        session_id=SESSION_ID,
        artifacts=ARTIFACTS,
        generations=GENERATIONS,
        workspace=REPO_ROOT,
        on_generation=report,
    )
    print(f"Market-scorer training complete. Session: {SESSION_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
