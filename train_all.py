#!/usr/bin/env python3
"""Run the resumable all-agent crop and DQN training workflow.

This script intentionally has no command-line configuration. Edit the constants
below, then run `python train_all.py` from any working directory.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
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

SYNC_ENVIRONMENT = True
# Shipping is an explicit release action. A long-running trainer must never
# replace the reviewed submission checkpoint merely because a gate passed.
EXPORT_ACTIVE_DQN_WEIGHTS = False
DQN_WEIGHTS_OUTPUT = Path("agents/dqn_job_scorer/weights.npz")


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
        not isinstance(FINAL_ARENA_JOBS, int)
        or isinstance(FINAL_ARENA_JOBS, bool)
        or FINAL_ARENA_JOBS < 1
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
            raise RuntimeError(
                "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
            )
        print("Preparing the locked Python 3.12 environment...", flush=True)
        _run([uv, "sync", "--locked", "--python", "3.12"])

    if sys.platform == "win32":
        python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        python = REPO_ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(
            f"virtual-environment Python not found at {python}; run setup first"
        )
    return python


def _kaglab(python: Path, *arguments: str, check: bool = True, quiet: bool = False):
    return _run(
        [str(python), "-m", "kaggriculture_lab.cli", *arguments],
        check=check,
        quiet=quiet,
    )


def _ensure_session(python: Path) -> None:
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
        return

    if not (REPO_ROOT / LEAGUE_CONFIG).is_file():
        raise FileNotFoundError(f"league config not found: {LEAGUE_CONFIG}")
    print(f"Creating league session: {SESSION_ID}", flush=True)
    _kaglab(
        python,
        "league",
        "init",
        "--config",
        str(LEAGUE_CONFIG),
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
        exported = _kaglab(
            python,
            "models",
            "export-active",
            "dqn_job_scorer.q_network",
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

    _kaglab(python, "models", "status", "--database", str(DATABASE))

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
        f"\n=== Final all-agent round-robin "
        f"({FINAL_ARENA_SEEDS} seeds in both seats) ===",
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
        raise SystemExit(
            "train_all.py does not accept arguments; edit its configuration constants"
        )
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
