#!/usr/bin/env python3
"""Run the resumable all-agent crop and DQN training workflow.

This script intentionally has no command-line configuration. Edit the constants
below, then run `python train_all.py` from any working directory.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

# ---------------------------------------------------------------------------
# Training configuration -- edit these values.
# ---------------------------------------------------------------------------

GENERATIONS = 48
RUN_FINAL_ARENA = True
FINAL_ARENA_SEEDS = 20
FINAL_ARENA_JOBS = 24  # set to None to use the arena's automatic default

SESSION_ID = "all-agents-training"
DATABASE = Path("artifacts/experiments.duckdb")
ARTIFACTS = Path("artifacts")
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
        "train": True,
    },
    {"type": "dqn_job_scorer", "name": "default", "train": True},
    {
        "type": "receding_horizon_planner",
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

REPO_ROOT = Path(__file__).resolve().parent


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
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
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


def _agent_cli(python: Path, module: str, *arguments: str, check: bool = True, quiet: bool = False):
    """Model administration is agent-owned; `kaglab` has no `models` group."""

    return _run([str(python), "-m", module, *arguments], check=check, quiet=quiet)


def _kaglab(python: Path, *arguments: str, check: bool = True, quiet: bool = False):
    return _run(
        [str(python), "-m", "kaggriculture_lab.cli", *arguments],
        check=check,
        quiet=quiet,
    )


def _toml_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, dict):
        fields = ", ".join(f"{key} = {_toml_literal(item)}" for key, item in value.items())
        return f"{{ {fields} }}"
    return str(value)


def _effective_config() -> Path:
    with (REPO_ROOT / LEAGUE_CONFIG).open("rb") as stream:
        values = dict(tomllib.load(stream)["league"])
    if INCLUDE_ALL_AGENT_ANCHORS:
        bred_manifest = str(values["manifest"])
        values["anchor_manifests"] = [
            str(path.relative_to(REPO_ROOT))
            for path in sorted((REPO_ROOT / "agents").glob("*/agent.toml"))
            if str(path.relative_to(REPO_ROOT)) != bred_manifest
        ]
    values["training_agents"] = TRAINING_AGENTS
    lines = ["[league]"]
    for name, value in values.items():
        if isinstance(value, list):
            lines.append(f"{name} = [")
            lines.extend(f"  {_toml_literal(item)}," for item in value)
            lines.append("]")
        else:
            lines.append(f"{name} = {_toml_literal(value)}")
    destination = REPO_ROOT / ARTIFACTS / "train_all.effective.toml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return destination


def _ensure_session(python: Path) -> None:
    config = _effective_config()
    status = _kaglab(
        python,
        "league",
        "status",
        "--session-id",
        SESSION_ID,
        "--database",
        str(DATABASE),
        check=False,
        quiet=True,
    )
    if status.returncode == 0:
        print(f"Resuming league session: {SESSION_ID}", flush=True)
        _kaglab(
            python,
            "league",
            "trainers",
            "--session-id",
            SESSION_ID,
            "--database",
            str(DATABASE),
            "--config",
            str(config),
        )
        return

    if not (REPO_ROOT / LEAGUE_CONFIG).is_file():
        raise FileNotFoundError(f"league config not found: {LEAGUE_CONFIG}")
    print(f"Creating league session: {SESSION_ID}", flush=True)
    _kaglab(
        python,
        "league",
        "init",
        "--config",
        str(config),
        "--session-id",
        SESSION_ID,
        "--database",
        str(DATABASE),
    )


def _run_training_iteration(python: Path, iteration: int) -> None:
    # One command. Collection and model updates ride the league's round schedule
    # through each agent's `train.py` hooks, so which models train, how much data
    # a round buys, and when a fit is attempted are all the agents' business --
    # declared in `configs/league.toml` and their manifests, not here.
    print(
        f"\n=== Training iteration {iteration}/{GENERATIONS}: league ===",
        flush=True,
    )
    _kaglab(
        python,
        "league",
        "run",
        "--session-id",
        SESSION_ID,
        "--database",
        str(DATABASE),
        "--artifacts",
        str(ARTIFACTS),
    )

    if EXPORT_ACTIVE_DQN_WEIGHTS:
        exported = _agent_cli(
            python,
            "agents.dqn_job_scorer.cli",
            "model",
            "export-active",
            "--name",
            DQN_EXPORT_NAME,
            "--database",
            str(DATABASE),
            "--output",
            str(DQN_WEIGHTS_OUTPUT),
            "--confirm-release",
            check=False,
        )
        if exported.returncode == 0:
            print("Exported the active DQN inference weights.", flush=True)
        else:
            print(
                "warning: no DQN checkpoint has passed promotion yet",
                file=sys.stderr,
                flush=True,
            )

    for instance in TRAINING_AGENTS:
        module = f"agents.{instance['type']}.cli"
        if not (REPO_ROOT / "agents" / str(instance["type"]) / "cli.py").is_file():
            continue
        _agent_cli(
            python,
            module,
            "model",
            "status",
            "--name",
            str(instance["name"]),
            "--database",
            str(DATABASE),
        )

    # Always leave the next generation OPEN so every later invocation completes
    # exactly GENERATIONS new generations.
    _kaglab(
        python,
        "league",
        "next-generation",
        "--session-id",
        SESSION_ID,
        "--database",
        str(DATABASE),
    )


def _run_final_arena(python: Path) -> None:
    if not RUN_FINAL_ARENA:
        return
    agent_specs = sorted((REPO_ROOT / "agents").glob("*/main.py"))
    if not agent_specs:
        raise FileNotFoundError("no agent entry points found under agents/*/main.py")

    print(
        f"\n=== Final all-agent round-robin ({FINAL_ARENA_SEEDS} seeds in both seats) ===",
        flush=True,
    )
    command = [
        str(python),
        "-m",
        "sim.arena",
        "league",
        *(str(path.relative_to(REPO_ROOT)) for path in agent_specs),
        "starter",
        "--seeds",
        str(FINAL_ARENA_SEEDS),
    ]
    if FINAL_ARENA_JOBS is not None:
        command.extend(["--jobs", str(FINAL_ARENA_JOBS)])
    _run(command)


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_all.py does not accept arguments; edit its configuration constants")
    _validate_configuration()
    python = _prepare_environment()
    _ensure_session(python)
    for iteration in range(1, GENERATIONS + 1):
        _run_training_iteration(python, iteration)
    _run_final_arena(python)
    print(f"\nTraining loop complete. Session: {SESSION_ID}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
