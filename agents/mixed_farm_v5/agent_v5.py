"""V5 mixed-farm controller: value -> feasible work -> labor -> execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agents.mixed_farm_v5.labor_v5 import build_labor_plan
from agents.mixed_farm_v5.market_v5 import market_orders
from agents.mixed_farm_v5.portfolio_v5 import build_plan
from agents.mixed_farm_v5.rules import get
from agents.mixed_farm_v5.scheduler import assign_actions, enumerate_tasks
from agents.mixed_farm_v5.strategy_v5 import Strategy


@dataclass(slots=True)
class EpisodeState:
    player: int | None = None
    last_step: int = -1
    commitments: dict[int, tuple[Any, ...]] = field(default_factory=dict)


class ConfigurableMixedFarmAgentV5:
    def __init__(self, strategy: Strategy, episode_steps: int = 720, turns_per_day: int = 24):
        self.strategy = strategy
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        self.state = EpisodeState()

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        farms = get(obs, "farms", []) or []
        player = int(get(obs, "player", 0))
        if player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        farm = farms[player]
        private = get(obs, "private", {}) or {}
        step = int(get(obs, "step", 0))
        if step == 0 or self.state.player != player or step <= self.state.last_step:
            self.state = EpisodeState(player=player)
        self.episode_steps = int(get(configuration, "episodeSteps", self.episode_steps))
        self.turns_per_day = int(get(configuration, "turnsPerDay", self.turns_per_day))
        multiplier = int(get(configuration, "farmHandCostMult", 1))
        positions = [tuple(farm["farmer"]), *(tuple(value) for value in farm.get("hands", []) or [])]
        inventories = list(private.get("inventories", []) or [])
        inventories.extend({} for _ in range(max(0, len(positions) - len(inventories))))
        plan = build_plan(obs, farm, self.strategy, episode_steps=self.episode_steps, turns_per_day=self.turns_per_day)
        tasks = enumerate_tasks(obs, farm, private, plan, positions, inventories, self.strategy,
                                episode_steps=self.episode_steps, turns_per_day=self.turns_per_day)
        labor = build_labor_plan(obs, farm, tasks, plan.intents, self.strategy,
                                 turns_per_day=self.turns_per_day, farm_hand_cost_mult=multiplier)
        orders = market_orders(obs, farm, private, plan, labor, self.strategy,
                               episode_steps=self.episode_steps, turns_per_day=self.turns_per_day)
        actions, self.state.commitments = assign_actions(
            tasks, positions, inventories, step, self.strategy, self.state.commitments
        )
        self.state.last_step = step
        return {"farmer": actions[0], "hands": actions[1:], "market": orders}


def make_agent(strategy: Strategy | dict[str, Any] | None = None, **runtime: int) -> Callable[[Any], dict]:
    if strategy is None:
        strategy = Strategy()
    elif isinstance(strategy, dict):
        values = dict(strategy)
        for key in ("episode_steps", "turns_per_day", "strategy_id"):
            if key in values:
                if key != "strategy_id":
                    runtime[key] = int(values[key])
                values.pop(key)
        strategy = Strategy.from_dict(values)
    return ConfigurableMixedFarmAgentV5(strategy, **runtime)
