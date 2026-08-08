#!/usr/bin/env python3
"""Run the isolated receding-horizon planner tuning league."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

# Edit these constants; this runner intentionally accepts no arguments.
GENERATIONS = 48
POPULATION_SIZE = 20
ROUNDS_PER_GENERATION = 20
HALL_OF_FAME_ROUNDS = 4
EPISODE_STEPS = 720
TRAINING_JOBS = 0  # 0 uses all but one CPU.
TRAINING_SEED = 314
CANDIDATE_POOL = 10_000
RUN_FINAL_ARENA = True
FINAL_ARENA_SEEDS = 20
FINAL_ARENA_JOBS = 24
SYNC_ENVIRONMENT = True

SESSION_ID = "receding-horizon-planner-training"
DATABASE = Path("artifacts/planner_experiments.duckdb")
ARTIFACTS = Path("artifacts")
LEAGUE_CONFIG = Path("configs/planner_league.toml")
TRAINING_AGENTS = [
    {
        "type": "receding_horizon_planner",
        "name": "default",
        "primary": True,
        "train": True,
    }
]

REPO_ROOT = Path(__file__).resolve().parent


def _run(command: list[str], *, check: bool = True, quiet: bool = False):
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )


def _python() -> Path:
    if SYNC_ENVIRONMENT:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to synchronize the environment")
        _run([uv, "sync", "--locked", "--python", "3.12"])
    path = REPO_ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not path.is_file():
        raise FileNotFoundError("project Python is missing; run setup first")
    return path


def _kaglab(python: Path, *arguments: str, check: bool = True, quiet: bool = False):
    return _run(
        [str(python), "-m", "kaggriculture_lab.cli", *arguments],
        check=check,
        quiet=quiet,
    )


def _effective_config() -> Path:
    with (REPO_ROOT / LEAGUE_CONFIG).open("rb") as stream:
        values = dict(tomllib.load(stream)["league"])
    values.update(
        population_size=POPULATION_SIZE,
        rounds_per_generation=ROUNDS_PER_GENERATION,
        hall_of_fame_rounds=HALL_OF_FAME_ROUNDS,
        episode_steps=EPISODE_STEPS,
        jobs=TRAINING_JOBS,
        seed=TRAINING_SEED,
        candidate_pool=CANDIDATE_POOL,
        training_agents=TRAINING_AGENTS,
    )
    lines = ["[league]"]
    for name, value in values.items():
        if isinstance(value, list):
            lines.append(f"{name} = [")
            for item in value:
                if isinstance(item, dict):
                    fields = ", ".join(
                        f"{key} = {json.dumps(field) if isinstance(field, str) else str(field).lower() if isinstance(field, bool) else field}"
                        for key, field in item.items()
                    )
                    lines.append(f"  {{ {fields} }},")
                else:
                    lines.append(f"  {json.dumps(item)},")
            lines.append("]")
        elif isinstance(value, str):
            lines.append(f"{name} = {json.dumps(value)}")
        else:
            lines.append(f"{name} = {value}")
    destination = REPO_ROOT / ARTIFACTS / "planner_league.effective.toml"
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
        print(f"Resuming planner session: {SESSION_ID}", flush=True)
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


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_planner.py accepts no arguments; edit its constants")
    positive = {
        "GENERATIONS": GENERATIONS,
        "POPULATION_SIZE": POPULATION_SIZE,
        "ROUNDS_PER_GENERATION": ROUNDS_PER_GENERATION,
        "EPISODE_STEPS": EPISODE_STEPS,
        "CANDIDATE_POOL": CANDIDATE_POOL,
        "FINAL_ARENA_SEEDS": FINAL_ARENA_SEEDS,
        "FINAL_ARENA_JOBS": FINAL_ARENA_JOBS,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid or TRAINING_JOBS < 0:
        raise ValueError(f"invalid training constants: {invalid or ['TRAINING_JOBS']}")
    python = _python()
    _ensure_session(python)
    for generation in range(1, GENERATIONS + 1):
        print(f"\n=== Planner generation {generation}/{GENERATIONS} ===", flush=True)
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
        if generation < GENERATIONS:
            _kaglab(
                python,
                "league",
                "next-generation",
                "--session-id",
                SESSION_ID,
                "--database",
                str(DATABASE),
            )

    if RUN_FINAL_ARENA:
        agents = sorted((REPO_ROOT / "agents").glob("*/main.py"))
        command = [
            str(python),
            "-m",
            "sim.arena",
            "league",
            *(str(path.relative_to(REPO_ROOT)) for path in agents),
            "starter",
            "--seeds",
            str(FINAL_ARENA_SEEDS),
            "--jobs",
            str(FINAL_ARENA_JOBS),
        ]
        _run(command)
    print(f"Planner training complete. Session: {SESSION_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
