"""Hard-bounded daily labor planning for v5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from agents.mixed_farm_v5.portfolio_v5 import InvestmentIntent
from agents.mixed_farm_v5.rules import distance, get, hire_cost
from agents.mixed_farm_v5.scheduler import Task
from agents.mixed_farm_v5.strategy_v5 import Strategy

NORMAL_CREW_CEILING = 10
ABSOLUTE_CREW_CEILING = 16
MIN_USEFUL_TURNS = 4


@dataclass(frozen=True, slots=True)
class LaborPlan:
    backlog: float
    desired_crew: int
    hire_count: int
    hire_cost: int
    useful_turns: int
    exceptional_hires: int
    reserve_after_hires: float
    safety_violations: tuple[str, ...] = ()


def fibonacci_hire_cost(start: int, count: int, multiplier: int = 1) -> int:
    return sum(hire_cost(start + offset, multiplier) for offset in range(count))


def build_labor_plan(obs: Any, farm: dict, tasks: Iterable[Task], intents: Iterable[InvestmentIntent], strategy: Strategy, *, turns_per_day: int = 24, farm_hand_cost_mult: int = 1) -> LaborPlan:
    step = int(get(obs, "step", 0))
    hour = int(get(obs, "hour", step % max(1, turns_per_day)))
    useful_turns = max(0, turns_per_day - hour - 1)
    hands = len(farm.get("hands", []) or [])
    positions = [tuple(farm.get("farmer", (0, 0))), *(tuple(value) for value in farm.get("hands", []) or [])]
    task_list = list(tasks)
    backlog = sum(1.0 + min((distance(position, task.target) for position in positions), default=0) for task in task_list)
    intent_list = [intent for intent in intents if intent.net_value > 0]
    accepted_setup = sum(intent.setup_work for intent in intent_list)
    backlog += accepted_setup
    if useful_turns < MIN_USEFUL_TURNS or hour > strategy.hire_until_hour or backlog <= 0:
        return LaborPlan(backlog, hands, 0, 0, useful_turns, 0, float(farm.get("money", 0)))
    capacity_per_unit = max(1.0, useful_turns * strategy.target_hand_utilization)
    desired_units = math.ceil(backlog / capacity_per_unit)
    desired_hands = min(strategy.max_hands, ABSOLUTE_CREW_CEILING, max(hands, desired_units - 1))
    money = float(farm.get("money", 0))
    investment_cost = sum(intent.cost for intent in intent_list)
    labor_cash = max(0.0, money - investment_cost)
    daily_cap = money * strategy.max_hire_spend_fraction
    hires_today = int(farm.get("hires_today", hands))
    chosen = 0
    spent = 0
    assigned_work = max(0.0, backlog - (hands + 1) * capacity_per_unit)
    while hands + chosen < desired_hands and assigned_work > 0:
        next_hand = hands + chosen + 1
        cost = hire_cost(hires_today + chosen, farm_hand_cost_mult)
        marginal_value = assigned_work * max((task.value for task in task_list), default=1.0) / max(1.0, backlog)
        hurdle = strategy.labor_roi_hurdle * (2.0 if next_hand > NORMAL_CREW_CEILING else 1.0)
        if spent + cost > daily_cap or labor_cash - spent - cost < strategy.cash_reserve:
            break
        if marginal_value < cost * hurdle:
            break
        chosen += 1
        spent += cost
        assigned_work -= capacity_per_unit
    return LaborPlan(backlog, hands + chosen, chosen, spent, useful_turns,
                     max(0, hands + chosen - NORMAL_CREW_CEILING), labor_cash - spent)
