"""Export a torch checkpoint to the numpy weights the submission loads.

    python -m agents.dqn_market_scorer.training.export \
        artifacts/.../checkpoints/<id>.pt agents/dqn_market_scorer/weights.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from agents.dqn_market_scorer.policy import WEIGHTS_PATH
from agents.dqn_market_scorer.training import network as net_module


def export(checkpoint: Path, destination: Path) -> Path:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Trainer checkpoints wrap the network alongside its optimizer and RNG
    # state; a bare state_dict is also accepted so a hand-saved module works.
    weights = state.get("market_online", state) if isinstance(state, dict) else state
    module = net_module.build()
    module.load_state_dict(weights)
    net_module.to_numpy(module).save(destination)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("destination", type=Path, nargs="?", default=WEIGHTS_PATH)
    args = parser.parse_args(argv)
    print(export(args.checkpoint, args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
