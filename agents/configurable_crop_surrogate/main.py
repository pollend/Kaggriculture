"""Balanced default for strategy attempt #1."""

from __future__ import annotations

from typing import Any

from agents.configurable_crop_surrogate._profile import named_agent

agent = named_agent("balanced")


def __getattr__(name: str) -> Any:
    """Expose the league's post-episode hook without importing it for inference.

    `agent.toml` points `episode_evaluator` at `main:evaluate_episode`, but a
    Kaggle submission only ever calls `agent`. Resolving it lazily keeps the
    submission from importing `kaggriculture_lab` at all.
    """
    if name == "evaluate_episode":
        from agents.configurable_crop_surrogate.evaluation import evaluate_episode

        return evaluate_episode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
