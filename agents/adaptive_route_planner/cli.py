"""Model administration for the adaptive route planner."""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggriculture_lab.model_admin import add_model_commands

MANIFEST = Path(__file__).with_name("agent.toml")
DEFAULT_DATABASE = "artifacts/adaptive_route_experiments.duckdb"
DEFAULT_ARTIFACTS = "artifacts/adaptive_route_planner"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="adaptive-route-planner", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_model_commands(
        commands,
        manifest_path=MANIFEST,
        default_database=DEFAULT_DATABASE,
        default_artifacts=DEFAULT_ARTIFACTS,
    )
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

