"""Deterministic configurable full-economy Kaggriculture agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agents.configurable_mixed_farm_surrogate.market import market_orders
from agents.configurable_mixed_farm_surrogate.portfolio import build_plan
from agents.configurable_mixed_farm_surrogate.rules import get
from agents.configurable_mixed_farm_surrogate.scheduler import assign_actions, enumerate_tasks
from agents.configurable_mixed_farm_surrogate.strategy import Strategy


@dataclass(slots=True)
class EpisodeState:
    player: int | None = None
    last_step: int = -1
    action_counts: dict[str, int] = field(default_factory=dict)
    commitments: dict[int, tuple[Any, ...]] = field(default_factory=dict)

    def reset(self, player: int) -> None:
        self.player = player
        self.last_step = -1
        self.action_counts.clear()
        self.commitments.clear()


class ConfigurableMixedFarmAgent:
    def __init__(self, strategy: Strategy, episode_steps: int = 720, turns_per_day: int = 24):
        self.strategy = strategy
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        self.state = EpisodeState()

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        farms = get(obs, "farms", []) or []
        player = int(get(obs, "player", 0))
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        farm = farms[player]
        private = get(obs, "private", {}) or {}
        step = int(get(obs, "step", 0))
        if step == 0 or self.state.player != player or step <= self.state.last_step:
            self.state.reset(player)

        self.episode_steps = int(get(configuration, "episodeSteps", self.episode_steps))
        self.turns_per_day = int(get(configuration, "turnsPerDay", self.turns_per_day))
        farm_hand_cost_mult = int(get(configuration, "farmHandCostMult", 1))
        positions = [tuple(farm["farmer"]), *(tuple(position) for position in farm.get("hands", []) or [])]
        inventories = list(private.get("inventories", []) or [])
        while len(inventories) < len(positions):
            inventories.append({})

        plan = build_plan(
            obs,
            farm,
            self.strategy,
            episode_steps=self.episode_steps,
            turns_per_day=self.turns_per_day,
        )
        orders = market_orders(
            obs,
            farm,
            private,
            plan,
            self.strategy,
            episode_steps=self.episode_steps,
            turns_per_day=self.turns_per_day,
            farm_hand_cost_mult=farm_hand_cost_mult,
        )
        tasks = enumerate_tasks(
            obs,
            farm,
            private,
            plan,
            positions,
            inventories,
            self.strategy,
            episode_steps=self.episode_steps,
            turns_per_day=self.turns_per_day,
        )
        actions, self.state.commitments = assign_actions(
            tasks, positions, inventories, step, self.strategy, self.state.commitments
        )
        self.state.last_step = step
        for action in actions:
            operation = str(action[0]) if action else "PASS"
            self.state.action_counts[operation] = self.state.action_counts.get(operation, 0) + 1
        return {"farmer": actions[0], "hands": actions[1:], "market": orders[:10]}


def make_agent(strategy: Strategy | dict[str, Any] | None = None, **runtime: int) -> Callable[[Any], dict]:
    if strategy is None:
        strategy = Strategy()
    elif isinstance(strategy, dict):
        values = dict(strategy)
        for key in ("episode_steps", "turns_per_day", "strategy_id"):
            if key in values:
                if key != "strategy_id":
                    runtime[key] = int(values[key])
                del values[key]
        strategy = Strategy.from_dict(values)
    return ConfigurableMixedFarmAgent(strategy, **runtime)
