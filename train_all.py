#!/usr/bin/env python3
"""Run the resumable all-agent crop and DQN training workflow.

Machine-level settings live in root ``main.toml``. The named training roster
below remains code because it is part of the experiment definition.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

from kaggriculture_lab.constants import (
    ARTIFACTS_ROOT,
    DEFAULT_GENERATIONS,
    REPO_ROOT,
    default_database,
)
from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.model_admin import export_active_model, model_status
from kaggriculture_lab.runner import ensure_session, run_generations, run_round_robin
from sim.metrics import render

# ---------------------------------------------------------------------------
# Experiment defaults. Root main.toml overrides machine/workflow values.
# ---------------------------------------------------------------------------

GENERATIONS = DEFAULT_GENERATIONS
RUN_FINAL_ARENA = True
FINAL_ARENA_SEEDS = 20
FINAL_ARENA_JOBS = 24  # set to None to use the arena's automatic default

SESSION_ID = "all-agents-training"
ARTIFACTS = ARTIFACTS_ROOT
LEAGUE_CONFIG = Path("configs/league.toml")
INCLUDE_ALL_AGENT_ANCHORS = True

# Each (type, name) owns independent model history and artifacts. Set
# `train = False` to lock an instance at its active checkpoint while retaining
# it for status/export and, when primary, surrogate-guided breeding. Optional
# `seed` overrides the stable seed derived from type + name. Agents without a
# model are listed with `train = False` and participate through the automatically
# discovered anchor roster. The planner stays locked here because its surrogate
# trains in its own planner league (`train_planner.py`).
TRAINING_AGENTS = [
    {
        "type": "configurable_crop_surrogate",
        "name": "default",
        "primary": True,
        "train": False,
    },
    {
        "type": "adaptive_route_planner",
        "name": "default",
        "train": False,
    },
    {"type": "dqn_job_scorer", "name": "default", "train": True},
    {
        # Off in the shared roster: its benchmark is the pilkwang donor, so it
        # only means anything in its own trainer-only league
        # (`configs/dqn_market_league.toml`). Here it is an anchor.
        "type": "dqn_market_scorer",
        "name": "default",
        "train": False,
    },
    {
        # Same reasoning as the scorer above, and the same donor benchmark: it
        # trains in `configs/dqn_market_free_league.toml`. It is the
        # unscaffolded counterpart -- no donor market layer, raw orders learned
        # from reward -- so here it is only an anchor.
        "type": "dqn_market_free",
        "name": "default",
        "train": False,
    },
    {
        "type": "sp_psro_job_scorer",
        "name": "default",
        "train": False,
    },
    {
        "type": "receding_horizon_planner",
        "name": "default",
        "train": False,
    },
    {
        "type": "mixed_farm_v5",
        "name": "v3",
        "train": False,
    },
    {
        # Locked here for the same reason as the planner: its forests train in
        # their own trainer-only league (`train_rf_market.py`).
        "type": "mixed_farm_rf_market",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_alexandergremyakov_goose_dividend_v2",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_avikdas567_microeconomic_policy",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_boatlee_v13_r3_premium_control",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_degnonguidi_agent_builder",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_ektarr_diversified_scheduler",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_jek1wantaufik_building_agent",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_kaitofukami_v22_price_impact",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_lucifer19_night_harvest",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_pilkwang_structured_economic_policy",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_prvsiyan_moon_counts_melons",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_romantamrazov_kaggriculture_hamburger",
        "name": "default",
        "train": False,
    },
    {
        "type": "competitor_tetsutani_adaptive_farming",
        "name": "default",
        "train": False,
    },
]


SYNC_ENVIRONMENT = True
# Shipping is an explicit release action. A long-running trainer must never
# replace the reviewed submission checkpoint merely because a gate passed.
EXPORT_ACTIVE_DQN_WEIGHTS = False
DQN_WEIGHTS_OUTPUT = Path("agents/dqn_job_scorer/weights.npz")
DQN_EXPORT_NAME = "default"


# ---------------------------------------------------------------------------
# Workflow implementation.
# ---------------------------------------------------------------------------

MAIN_CONFIG = REPO_ROOT / "main.toml"


def _load_main_config(path: Path = MAIN_CONFIG) -> None:
    """Apply machine-level root settings to this constants-driven runner."""

    global ARTIFACTS, DQN_WEIGHTS_OUTPUT, EXPORT_ACTIVE_DQN_WEIGHTS
    global FINAL_ARENA_JOBS, FINAL_ARENA_SEEDS, GENERATIONS, INCLUDE_ALL_AGENT_ANCHORS
    global LEAGUE_CONFIG, RUN_FINAL_ARENA, SESSION_ID, SYNC_ENVIRONMENT, TRAINING_DEVICE

    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    training = raw.get("training", {})
    paths = raw.get("paths", {})
    compute = raw.get("compute", {})
    GENERATIONS = training.get("generations", GENERATIONS)
    SESSION_ID = training.get("session_id", SESSION_ID)
    SYNC_ENVIRONMENT = training.get("sync_environment", SYNC_ENVIRONMENT)
    RUN_FINAL_ARENA = training.get("run_final_arena", RUN_FINAL_ARENA)
    FINAL_ARENA_SEEDS = training.get("final_arena_seeds", FINAL_ARENA_SEEDS)
    FINAL_ARENA_JOBS = training.get("final_arena_jobs", FINAL_ARENA_JOBS)
    INCLUDE_ALL_AGENT_ANCHORS = training.get(
        "include_all_agent_anchors", INCLUDE_ALL_AGENT_ANCHORS
    )
    EXPORT_ACTIVE_DQN_WEIGHTS = training.get(
        "export_active_dqn_weights", EXPORT_ACTIVE_DQN_WEIGHTS
    )
    ARTIFACTS = Path(paths.get("artifacts", ARTIFACTS))
    LEAGUE_CONFIG = Path(paths.get("league_config", LEAGUE_CONFIG))
    DQN_WEIGHTS_OUTPUT = Path(paths.get("dqn_weights_output", DQN_WEIGHTS_OUTPUT))
    TRAINING_DEVICE = str(compute.get("device", "auto"))


_load_main_config()


def _validate_configuration() -> None:
    positive_values = {
        "GENERATIONS": GENERATIONS,
        "FINAL_ARENA_SEEDS": FINAL_ARENA_SEEDS,
    }
    for name, value in positive_values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if FINAL_ARENA_JOBS is not None and (
        not isinstance(FINAL_ARENA_JOBS, int) or isinstance(FINAL_ARENA_JOBS, bool) or FINAL_ARENA_JOBS < 1
    ):
        raise ValueError("FINAL_ARENA_JOBS must be None or a positive integer")


def _run(
    command: list[str],
    *,
    check: bool = True,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["KAGGRICULTURE_TORCH_DEVICE"] = TRAINING_DEVICE
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
        env=environment,
    )


def _prepare_environment() -> Path:
    if SYNC_ENVIRONMENT:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required: https://docs.astral.sh/uv/getting-started/installation/")
        print("Preparing the locked Python 3.12 environment...", flush=True)
        _run([uv, "sync", "--locked", "--python", "3.12"])

    if sys.platform == "win32":
        python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        python = REPO_ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"virtual-environment Python not found at {python}; run setup first")
    return python


def _effective_config() -> LeagueConfig:
    base = LeagueConfig.from_toml(REPO_ROOT / LEAGUE_CONFIG)
    anchors = list(base.anchor_manifests)
    if INCLUDE_ALL_AGENT_ANCHORS:
        bred_manifest = str(base.manifest)
        anchors = [
            str(path.relative_to(REPO_ROOT))
            for path in sorted((REPO_ROOT / "agents").glob("*/agent.toml"))
            if str(path.relative_to(REPO_ROOT)) != bred_manifest
        ]
    return replace(
        base,
        anchor_manifests=tuple(anchors),
        training_agents=tuple(TrainingAgentSpec.from_dict(item) for item in TRAINING_AGENTS),
    )


def _ensure_session() -> None:
    config = _effective_config()
    if not (REPO_ROOT / LEAGUE_CONFIG).is_file():
        raise FileNotFoundError(f"league config not found: {LEAGUE_CONFIG}")
    created, _ = ensure_session(
        config,
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_all",
        experiment={"training_agents": TRAINING_AGENTS},
    )
    print(f"{'Creating' if created else 'Resuming'} league session: {SESSION_ID}", flush=True)


def _after_training_iteration(iteration: int, _: dict) -> None:
    # One command. Collection and model updates ride the league's round schedule
    # through each agent's `train.py` hooks, so which models train, how much data
    # a round buys, and when a fit is attempted are all the agents' business --
    # declared in `configs/league.toml` and their manifests, not here.
    print(
        f"\n=== Training iteration {iteration}/{GENERATIONS}: league ===",
        flush=True,
    )
    if EXPORT_ACTIVE_DQN_WEIGHTS:
        try:
            export_active_model(
                REPO_ROOT / "agents/dqn_job_scorer/agent.toml",
                default_database(),
                DQN_WEIGHTS_OUTPUT,
                confirm_release=True,
                name=DQN_EXPORT_NAME,
            )
            print("Exported the active DQN inference weights.", flush=True)
        except ValueError:
            print(
                "warning: no DQN checkpoint has passed promotion yet",
                file=sys.stderr,
                flush=True,
            )

    for instance in TRAINING_AGENTS:
        if not (REPO_ROOT / "agents" / str(instance["type"]) / "cli.py").is_file():
            continue
        manifest = REPO_ROOT / "agents" / str(instance["type"]) / "agent.toml"
        if manifest.is_file():
            print(model_status(manifest, name=str(instance["name"])))


def _run_final_arena() -> None:
    if not RUN_FINAL_ARENA:
        return
    agent_specs = sorted((REPO_ROOT / "agents").glob("*/main.py"))
    if not agent_specs:
        raise FileNotFoundError("no agent entry points found under agents/*/main.py")

    print(
        f"\n=== Final all-agent round-robin ({FINAL_ARENA_SEEDS} seeds in both seats) ===",
        flush=True,
    )
    specs = [str(path.relative_to(REPO_ROOT)) for path in agent_specs]
    jobs = FINAL_ARENA_JOBS or max(1, (os.cpu_count() or 2) - 1)
    for summary in run_round_robin(
        [*specs, "starter"], seeds=FINAL_ARENA_SEEDS, jobs=jobs
    ):
        print(render(summary))


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_all.py does not accept arguments; edit its configuration constants")
    _validate_configuration()
    _prepare_environment()
    _ensure_session()
    run_generations(
        session_id=SESSION_ID,
        artifacts=ARTIFACTS,
        generations=GENERATIONS,
        workspace=REPO_ROOT,
        on_generation=_after_training_iteration,
    )
    _run_final_arena()
    print(f"\nTraining loop complete. Session: {SESSION_ID}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
