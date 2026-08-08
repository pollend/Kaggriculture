"""Load this strategy attempt's checked-in profiles."""

from __future__ import annotations

import json
from pathlib import Path

from agents.configurable_crop_surrogate.agent import make_agent
from agents.configurable_crop_surrogate.strategy import Strategy

PROFILE_DIR = Path(__file__).with_name("profiles")


def load_strategy(name: str) -> Strategy:
    path = PROFILE_DIR / f"{name}.json"
    if not path.is_file():
        available = ", ".join(sorted(item.stem for item in PROFILE_DIR.glob("*.json")))
        raise ValueError(f"unknown configurable-crop profile {name!r}; available: {available}")
    return Strategy.from_dict(json.loads(path.read_text()))


def named_agent(name: str):
    return make_agent(load_strategy(name))

