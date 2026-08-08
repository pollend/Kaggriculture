"""Export a torch checkpoint to the numpy weights the submission loads.

    python -m agents.dqn_job_scorer.training.export runs/dqn/online.pt agents/dqn_job_scorer/weights.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from agents.dqn_job_scorer.policy import WEIGHTS_PATH
from agents.dqn_job_scorer.training import network as net_module


def export(checkpoint: Path, destination: Path) -> Path:
    module = net_module.build()
    module.load_state_dict(torch.load(checkpoint, map_location="cpu"))
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
