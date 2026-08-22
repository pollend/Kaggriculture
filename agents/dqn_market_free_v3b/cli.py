"""Training and release administration for the market-free agent.

Normal collection, fitting, and gating run automatically through ``league run``.
These commands inspect that state, retry an eligible update, or explicitly copy
a reviewed active checkpoint into a release path.

Exporting matters more here than it does for the scorer. The scorer's packaged
fallback is the donor policy, so a missing `weights.npz` still plays a strong
agent; this agent's fallback never trades at all (`main.load_network`). Nothing
promotes a checkpoint into `weights.npz` on its own -- run `model export-active`
after reviewing a passed gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggriculture_lab.model_admin import add_model_commands

MANIFEST = Path(__file__).with_name("agent.toml")
DEFAULT_ARTIFACTS = "artifacts"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dqn-market-free", description=__doc__)
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
