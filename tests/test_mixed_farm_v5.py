from __future__ import annotations

from agents.mixed_farm_v5.labor_v5 import (
    ABSOLUTE_CREW_CEILING,
    build_labor_plan,
    fibonacci_hire_cost,
)
from agents.mixed_farm_v5.portfolio_v5 import InvestmentIntent, build_plan
from agents.mixed_farm_v5.scheduler import Task
from agents.mixed_farm_v5.strategy_v5 import Strategy
from tests.test_mixed_farm_v5_base import observation


def test_v4_migration_discards_unsafe_labor_and_land_floors():
    raw = Strategy().to_legacy().to_dict()
    raw.update(minimum_hands=15, land_expansion_day=2, workload_per_hand=2, max_hands=24)

    migrated = Strategy.from_dict(raw)

    assert migrated.max_hands == ABSOLUTE_CREW_CEILING
    assert "minimum_hands" not in migrated.to_dict()
    assert "land_expansion_day" not in migrated.to_dict()
    assert "workload_per_hand" not in migrated.to_dict()
    assert migrated.target_hand_utilization == 0.9


def test_plan_exposes_investment_intents_but_no_target_hands():
    obs = observation()
    plan = build_plan(obs, obs["farms"][0], Strategy())

    assert not hasattr(plan, "target_hands")
    assert plan.intents
    assert all(intent.value > intent.cost for intent in plan.intents)


def test_labor_refuses_hiring_without_useful_turns_or_work():
    obs = observation(step=21)
    strategy = Strategy(max_hands=16, hire_until_hour=12)
    assert build_labor_plan(obs, obs["farms"][0], [], [], strategy).hire_count == 0

    obs = observation(step=11)
    assert build_labor_plan(obs, obs["farms"][0], [], [], strategy).hire_count == 0


def test_labor_obeys_reserve_spend_cap_and_fibonacci_cost():
    obs = observation(step=0)
    obs["farms"][0]["money"] = 1000
    tasks = [Task("WORK", (0, 0), ("DIG",), 10_000, 23) for _ in range(100)]
    strategy = Strategy(max_hands=16, cash_reserve=900, max_hire_spend_fraction=0.05)

    labor = build_labor_plan(obs, obs["farms"][0], tasks, [], strategy)

    assert labor.desired_crew <= ABSOLUTE_CREW_CEILING
    assert labor.hire_cost == fibonacci_hire_cost(0, labor.hire_count)
    assert labor.hire_cost <= 50
    assert labor.reserve_after_hires >= strategy.cash_reserve


def test_setup_intent_can_create_labor_backlog_but_not_break_reserve():
    obs = observation(step=0)
    intent = InvestmentIntent("seed", "WHEAT", 10, 100, 10_000, 30, 10, 10, 719)
    labor = build_labor_plan(obs, obs["farms"][0], [], [intent], Strategy())

    assert labor.backlog == 30
    assert labor.reserve_after_hires >= Strategy().cash_reserve
