"""Balanced default for the configurable mixed-farm surrogate."""

from __future__ import annotations

from typing import Any

from agents.configurable_mixed_farm_surrogate._profile import load_strategy
from agents.configurable_mixed_farm_surrogate.agent import make_agent

agent = make_agent(load_strategy("balanced"))


def __getattr__(name: str) -> Any:
    if name == "evaluate_episode":
        from agents.configurable_mixed_farm_surrogate.evaluation import evaluate_episode

        return evaluate_episode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

