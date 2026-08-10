"""Load a checked-in mixed-farm strategy profile."""

from __future__ import annotations

import json
from pathlib import Path

from agents.configurable_mixed_farm_surrogate.strategy import Strategy


def load_strategy(name: str = "balanced") -> Strategy:
    path = Path(__file__).with_name("profiles") / f"{name}.json"
    return Strategy.from_dict(json.loads(path.read_text(encoding="utf-8")))

