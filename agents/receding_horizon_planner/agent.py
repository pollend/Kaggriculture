"""Hybrid receding-horizon planner and joint task scheduler."""

from __future__ import annotations

import json
import os
from typing import Any

from agents.receding_horizon_planner.market import market_orders
from agents.receding_horizon_planner.planner import Plan, build_plan
from agents.receding_horizon_planner.rules import get
from agents.receding_horizon_planner.scheduler import TaskScheduler, enumerate_tasks
from agents.receding_horizon_planner.strategy import Strategy


class RecedingHorizonAgent:
    def __init__(self, strategy: Strategy, episode_steps: int = 720, turns_per_day: int = 24):
        self.strategy = strategy
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        self.scheduler = TaskScheduler()
        self.plan: Plan | None = None
        self._day = -1
        self._event_signature: tuple[Any, ...] | None = None

    def _reset(self) -> None:
        self.scheduler.reset()
        self.plan = None
        self._day = -1
        self._event_signature = None

    def _signature(self, obs: Any, farm: dict, terminal: bool) -> tuple[Any, ...]:
        shops = tuple((get(obs, "town", {}) or {}).get("unlocked_shops", []) or ())
        quadrants = tuple(farm.get("unlocked_quadrants") or ())
        return shops, quadrants, terminal

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        farms = get(obs, "farms", []) or []
        player = int(get(obs, "player", 0))
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}

        farm = farms[player]
        private = get(obs, "private", {}) or {}
        step = int(get(obs, "step", 0))
        day = int(get(obs, "day", 0))
        episode_steps = int(get(configuration, "episodeSteps", self.episode_steps))
        turns_per_day = int(get(configuration, "turnsPerDay", self.turns_per_day))
        farm_hand_cost_mult = int(get(configuration, "farmHandCostMult", 1))
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day

        if step == 0:
            self._reset()
        if day != self._day:
            self.scheduler.reset()
            self._day = day

        terminal = step >= episode_steps - self.strategy.liquidation_turns
        signature = self._signature(obs, farm, terminal)
        if self.plan is None or self.plan.day != day or signature != self._event_signature:
            self.plan = build_plan(
                obs,
                farm,
                private,
                player,
                self.strategy,
                episode_steps,
                turns_per_day,
            )
            self._event_signature = signature

        positions = [tuple(farm.get("farmer", (0, 0)))]
        positions.extend(tuple(position) for position in farm.get("hands", []) or [])
        inventories = list(private.get("inventories", []) or [])
        while len(inventories) < len(positions):
            inventories.append({})

        tasks = enumerate_tasks(
            obs,
            farm,
            private,
            self.plan,
            positions,
            inventories,
            self.strategy,
            episode_steps,
            turns_per_day,
        )
        unit_actions = self.scheduler.assign(tasks, positions, inventories, step, self.strategy)
        orders = market_orders(
            obs,
            farm,
            private,
            self.plan,
            self.strategy,
            episode_steps,
            turns_per_day,
            farm_hand_cost_mult,
        )
        return {"farmer": unit_actions[0], "hands": unit_actions[1:], "market": orders}


def make_agent(
    strategy: Strategy | dict[str, Any] | None = None,
    *,
    episode_steps: int | str = 720,
    **parameters: Any,
) -> RecedingHorizonAgent:
    if strategy is not None and parameters:
        raise ValueError("pass either a Strategy or keyword parameters, not both")
    if isinstance(strategy, Strategy):
        selected = strategy
    elif isinstance(strategy, dict):
        values = dict(strategy)
        episode_steps = values.pop("episode_steps", episode_steps)
        selected = Strategy.from_dict(values)
    else:
        selected = Strategy.from_dict(parameters) if parameters else Strategy()
    return RecedingHorizonAgent(selected, episode_steps=int(episode_steps))


def strategy_from_environment() -> tuple[Strategy, int]:
    raw = os.environ.get("KAGG_AGENT_CONFIG")
    if not raw:
        return Strategy(), 720
    values = json.loads(raw)
    episode_steps = int(values.pop("episode_steps", 720))
    return Strategy.from_dict(values), episode_steps
