"""Balanced default entry point for mixed-farm v5."""

from agents.mixed_farm_v5.agent_v5 import make_agent
from agents.mixed_farm_v5.strategy_v5 import Strategy

agent = make_agent(Strategy())


def __getattr__(name: str):
    if name == "evaluate_episode":
        from agents.mixed_farm_v5.evaluation import evaluate_episode
        return evaluate_episode
    raise AttributeError(name)
