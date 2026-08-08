from __future__ import annotations

from copy import deepcopy

from agents.receding_horizon_planner.agent import make_agent
from agents.receding_horizon_planner.market import market_orders
from agents.receding_horizon_planner.planner import Plan, build_plan
from agents.receding_horizon_planner.scheduler import Task, enumerate_tasks, hungarian_maximize
from agents.receding_horizon_planner.strategy import Strategy, sample_strategy

PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
ANIMALS = ("GOOSE", "COW", "SHEEP")


def observation(*, step: int = 0, hands: list[list[int]] | None = None) -> dict:
    hands = hands or []
    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0,
        "tiles": tiles,
        "farmer": [4, 4],
        "hands": hands,
        "unlocked_quadrants": ["NW"],
        "hires_today": len(hands),
    }
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, deepcopy(farm)],
        "private": {
            "shed": {item: 0 for item in (*PRODUCTS, *ANIMALS)},
            "seeds": {crop: 0 for crop in PRODUCTS[:5]},
            "inventories": [{} for _ in range(1 + len(hands))],
        },
        "market": {
            "inventory": {item: 10_000 for item in PRODUCTS},
            "prices": {
                "WHEAT": 25,
                "CARROT": 35,
                "TOMATO": 60,
                "STRAWBERRY": 120,
                "MELON": 250,
                "EGG": 50,
                "MILK": 160,
                "WOOL": 200,
                "FERTILIZER": 100,
            },
        },
        "town": {"unlocked_shops": []},
    }


def empty_plan(day: int = 0) -> Plan:
    return Plan(
        day,
        {crop: 0 for crop in PRODUCTS[:5]},
        {animal: 0 for animal in ANIMALS},
        0,
        False,
        0,
        0.0,
        (),
    )


def test_hungarian_assignment_beats_row_greedy():
    utilities = [[10.0, 9.0], [9.0, 0.0]]
    assignment = hungarian_maximize(utilities)
    assert assignment == [1, 0]
    assert sum(utilities[row][column] for row, column in enumerate(assignment)) == 18.0


def test_strategy_sampling_stays_valid():
    import random

    for seed in range(20):
        sample_strategy(random.Random(seed)).validate()


def test_emergency_water_is_selected_on_the_current_tile():
    obs = observation(step=72)
    obs["farms"][0]["tiles"][4][4] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 4,
        "fertilized_until_day": -1,
    }
    action = make_agent(Strategy())(obs)
    assert action["farmer"] == ["WATER"]


def test_animal_frontier_contains_all_daily_work():
    obs = observation(step=120)
    obs["private"]["inventories"][0]["WHEAT"] = 1
    obs["farms"][0]["tiles"][4][4] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "placed_day": 0,
        "yield_units": 2,
        "consecutive_unfed": 1,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": True,
        "pending_care_bonus": 0,
    }
    tasks = enumerate_tasks(
        obs,
        obs["farms"][0],
        obs["private"],
        empty_plan(5),
        [(4, 4)],
        obs["private"]["inventories"],
        Strategy(),
    )
    assert {task.kind for task in tasks} >= {
        "FEED",
        "CARE",
        "HARVEST_ANIMAL",
        "COLLECT_FERTILIZER",
    }


def test_last_hour_never_exposes_a_plant_task():
    obs = observation(step=23)
    obs["private"]["seeds"]["WHEAT"] = 1
    base = empty_plan()
    plan = Plan(
        base.day,
        {**base.target_crops, "WHEAT": 1},
        base.target_animals,
        0,
        False,
        0,
        0.0,
        (),
    )
    tasks = enumerate_tasks(
        obs,
        obs["farms"][0],
        obs["private"],
        plan,
        [(4, 4)],
        [{}],
        Strategy(),
    )
    assert "PLANT" not in {task.kind for task in tasks}


def test_terminal_market_sells_without_buying():
    obs = observation(step=700)
    obs["private"]["shed"]["WHEAT"] = 5
    orders = market_orders(
        obs,
        obs["farms"][0],
        obs["private"],
        empty_plan(29),
        Strategy(),
    )
    assert ["SELL", "WHEAT", 5] in orders
    assert not any(order[0].startswith("BUY") for order in orders)


def test_terminal_scheduler_returns_carried_feed_to_the_shed():
    obs = observation(step=700)
    obs["private"]["inventories"][0]["WHEAT"] = 4
    tasks = enumerate_tasks(
        obs,
        obs["farms"][0],
        obs["private"],
        empty_plan(29),
        [(4, 4)],
        obs["private"]["inventories"],
        Strategy(),
    )
    assert any(task.kind == "DROP" and task.owner == 0 for task in tasks)


def test_plan_does_not_add_assets_that_cannot_mature():
    obs = observation(step=710)
    plan = build_plan(obs, obs["farms"][0], obs["private"], 0, Strategy())
    assert sum(plan.target_crops.values()) == 0
    assert sum(plan.target_animals.values()) == 0


def test_agent_returns_one_action_for_every_hand_and_resets():
    agent = make_agent(Strategy())
    first = agent(observation(hands=[[3, 4], [4, 3]]))
    assert len(first["hands"]) == 2
    assert len(first["market"]) <= 10
    agent.scheduler.commitments[0] = (("stale",), 2)
    agent(observation(step=0))
    assert (("stale",), 2) not in agent.scheduler.commitments.values()


def test_task_keys_include_owner_for_logistics():
    left = Task("DROP", (4, 4), ("DROP",), 1.0, 10, owner=0)
    right = Task("DROP", (4, 4), ("DROP",), 1.0, 10, owner=1)
    assert left.key != right.key
