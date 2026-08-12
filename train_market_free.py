#!/usr/bin/env python3
"""Run the resumable free-market workflow.

Same shape as `train_market_scorer.py` -- one learner, one session, one
Q-network on top of a frozen pilkwang field policy -- and the same benchmark, so
the two are directly comparable. The difference is everything below the network:
this agent has no `market_exec`, no donor market rules, and no
`_schedule_market_adjustment`. Every turn it picks one order out of a flat list
of 186, or nothing, and has to learn affordability, the feed floor and terminal
liquidation from reward.

Expect this to be hard, and expect it to look like nothing is happening for a
long time. The two failure modes to watch, both recorded per batch in
`metadata_json` and per update in `metrics_json`:

    orders_per_decision ~ 0        the NOOP collapse: it stopped trading
    mean_final_money    ~ 0        the bankruptcy collapse: it traded itself broke

A healthy run has `orders_per_decision` somewhere in between and
`solvent_episodes` climbing off zero. Query them rather than guessing:

    SELECT batch_seq,
           metadata_json ->> 'orders_per_decision' AS orders,
           metadata_json ->> 'mean_final_money'    AS money,
           metadata_json ->> 'solvent_episodes'    AS solvent
    FROM training_batches
    WHERE model_id = 'dqn_market_free.raw_order_q_v1@simple_v1'
    ORDER BY batch_seq;
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
#
# The run is unbounded: generations are created and played one at a time until
# the process is interrupted. Exploration anneals over
# `epsilon_decay_transitions` (2,000,000) at roughly 11,500 transitions a batch
# and 20 batches a generation, so epsilon does not reach its floor until
# generation ~9. Any fixed stop below that answers only that the policy had not
# finished annealing, not whether it learns -- and a useful stopping point is a
# judgement about the metrics in the docstring above, not a number decided in
# advance. Ctrl-C after a generation closes; committed work stays resumable.
SYNC_ENVIRONMENT = True

# A fresh session and instance name. The action space changed from ten order
# slots to one, which changes both the rollout record format and the network
# shape, so no checkpoint or batch from `market-free-demo-v1` can be read. A new
# instance also means a clean artifact root rather than one where new
# `rollout-000000NN.npz` files land on top of the old run's.
SESSION_ID = "market-free-simple-v1"
ARTIFACTS = ARTIFACTS_ROOT / "dqn_market_free"
LEAGUE_CONFIG = Path("configs/dqn_market_free_league.toml")
MANIFEST = Path("agents/dqn_market_free/agent.toml")
MODEL_NAME = "simple_v1"

DONOR = "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"

BENCHMARKS = (
    DONOR,
    "agents/configurable_crop_surrogate/main.py",
)

# Overrides only. `register_training_agent` materializes `agent.toml`'s
# `[models.config]` and then applies this on top, so every key here is a
# deliberate departure from the manifest.
FREE_CONFIG = {
    "benchmarks": list(BENCHMARKS),
    "benchmark": DONOR,
    "episode_steps": 720,
    # This named instance pins solvent trajectories from the matching pilkwang
    # donor family and clones their order choices before Double-DQN fine-tuning.
    # The bare manifest (`demonstration_episodes = 0`) remains the reward-only
    # control.
    "demonstration_episodes": 200,
    "demonstration_min_money": 10_000.0,
    "demonstration_family": "competitor_pilkwang_structured_economic_policy",
    "behavior_clone_steps_per_update": 100,
    # A resume restores the optimizer's learning rate from the checkpoint, so
    # this key was previously inert on every generation after the first. It is
    # now re-applied from config each update.
    "learning_rate": 1e-4,
}

TRAINING_AGENTS = [
    {
        "type": "dqn_market_free",
        "name": MODEL_NAME,
        "train": True,
        "primary": True,
        "model_config": FREE_CONFIG,
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
        runner="train_market_free",
        experiment={"training_agents": TRAINING_AGENTS},
    )
    print(
        f"{'Creating' if created else 'Resuming'} market-free session: {SESSION_ID}",
        flush=True,
    )


def _model_status() -> None:
    print(model_status(REPO_ROOT / MANIFEST, name=MODEL_NAME))


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_market_free.py accepts no arguments; edit its constants")

    os.environ["KAGGRICULTURE_TORCH_DEVICE"] = _device()
    _prepare_environment()
    _ensure_session()

    def report(generation: int, _: dict) -> None:
        print(f"\n=== market-free generation {generation} complete ===", flush=True)
        _model_status()

    try:
        run_generations(
            session_id=SESSION_ID,
            artifacts=ARTIFACTS,
            # Continuous mode never queues: the next generation is created only
            # after the current one has closed, so `generations` is only the
            # positional floor the runner validates.
            generations=1,
            continuous=True,
            workspace=REPO_ROOT,
            on_generation=report,
        )
    except KeyboardInterrupt:
        print(f"\n[{SESSION_ID}] stopped; committed work remains resumable", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
