"""Balanced route-informed default and Kaggle entry point."""

from __future__ import annotations

from typing import Any

from agents.adaptive_route_planner.agent import make_agent, strategy_from_environment

_STRATEGY, _EPISODE_STEPS, _TURNS_PER_DAY = strategy_from_environment()
agent = make_agent(
    _STRATEGY,
    episode_steps=_EPISODE_STEPS,
    turns_per_day=_TURNS_PER_DAY,
)


def __getattr__(name: str) -> Any:
    if name == "evaluate_episode":
        from agents.adaptive_route_planner.evaluation import evaluate_episode

        return evaluate_episode
    raise AttributeError(name)
