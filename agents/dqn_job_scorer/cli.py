"""Training and release administration for the DQN job scorer.

Normal collection, fitting, and gating run automatically through ``league run``.
These commands inspect that state, retry an eligible update, or explicitly copy
a reviewed active checkpoint into a release path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggriculture_lab.model_admin import add_model_commands

MANIFEST = Path(__file__).with_name("agent.toml")
DEFAULT_ARTIFACTS = "artifacts"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dqn-job-scorer", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_model_commands(
        commands,
        manifest_path=MANIFEST,
        default_artifacts=DEFAULT_ARTIFACTS,
        allow_export=True,
    )
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
