"""Training administration for the receding-horizon planner.

The planner's league performs normal batch collection and model updates
automatically. This CLI only inspects state or retries an eligible update.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggriculture_lab.model_admin import add_model_commands

MANIFEST = Path(__file__).with_name("agent.toml")
DEFAULT_DATABASE = "artifacts/planner_experiments.duckdb"
DEFAULT_ARTIFACTS = "artifacts/planner"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="receding-horizon-planner", description=__doc__)
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
