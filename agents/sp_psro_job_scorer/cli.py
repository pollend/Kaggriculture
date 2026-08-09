"""Training and release administration for the SP-PSRO job scorer."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import torch

from kaggriculture_lab.model_admin import add_model_commands

MANIFEST = Path(__file__).with_name("agent.toml")
DEFAULT_DATABASE = "artifacts/sp_psro_v4_experiments.duckdb"
DEFAULT_ARTIFACTS = "artifacts/sp_psro_v4"


def _default_model_id() -> str:
    with MANIFEST.open("rb") as stream:
        return str(tomllib.load(stream)["models"][0]["model_id"])


def _population_status(args: argparse.Namespace) -> int:
    from kaggriculture_lab.storage import ExperimentStore

    with ExperimentStore(args.database) as store:
        row = store.connection.execute(
            """
            SELECT mtc.trainer_artifact_path
            FROM training_models tm
            LEFT JOIN model_training_checkpoints mtc
              ON mtc.checkpoint_id = tm.active_checkpoint_id
            WHERE tm.model_id = ?
            """,
            [args.model_id],
        ).fetchone()
    if not row or not row[0]:
        print(json.dumps({"model_id": args.model_id, "population": 0, "champion": None}))
        return 0
    state = torch.load(row[0], map_location="cpu", weights_only=False)
    payload = {
        "model_id": args.model_id,
        "population": len(state.get("population", [])),
        "iteration": int(state.get("iteration", 0)),
        "champion": state.get("population_champion_id", state.get("champion_id")),
        "release_champion": state.get("release_champion", state.get("inference_champion")),
        "members": state.get("population", []),
        "last_admission": state.get("last_admission"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sp-psro-job-scorer")
    commands = parser.add_subparsers(dest="command", required=True)
    add_model_commands(
        commands,
        manifest_path=MANIFEST,
        default_database=DEFAULT_DATABASE,
        default_artifacts=DEFAULT_ARTIFACTS,
        allow_export=True,
    )
    population = commands.add_parser("population-status")
    population.add_argument("--database", default=DEFAULT_DATABASE)
    population.add_argument("--model-id", default=_default_model_id())
    population.set_defaults(func=_population_status)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
