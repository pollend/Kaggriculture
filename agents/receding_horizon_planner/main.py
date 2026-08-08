"""Default entry point for the receding-horizon planner."""

from __future__ import annotations

from typing import Any

from agents.receding_horizon_planner.agent import make_agent, strategy_from_environment

_STRATEGY, _EPISODE_STEPS = strategy_from_environment()
agent = make_agent(_STRATEGY, episode_steps=_EPISODE_STEPS)


def __getattr__(name: str) -> Any:
    if name == "evaluate_episode":
        from agents.receding_horizon_planner.evaluation import evaluate_episode

        return evaluate_episode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
