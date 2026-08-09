from __future__ import annotations

import random
from copy import deepcopy

from agents.adaptive_route_planner.agent import make_agent, modeled_price
from agents.adaptive_route_planner.strategy import Strategy, sample_strategy

PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")


def observation(*, step: int = 0, hands: list[list[int]] | None = None) -> dict:
    hands = hands or []
    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000,
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
            "shed": {**{item: 0 for item in PRODUCTS}, "COW": 0, "SHEEP": 0},
            "seeds": {"WHEAT": 0, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0},
            "inventories": [{} for _ in range(1 + len(hands))],
        },
        "market": {
            "inventory": {item: 10_000 for item in PRODUCTS},
            "prices": {item: modeled_price(item, 10_000) for item in PRODUCTS},
        },
        "town": {"unlocked_shops": []},
    }


def test_route_strategy_sampling_is_valid_and_deterministic():
    for seed in range(100):
        left = sample_strategy(random.Random(seed))
        right = sample_strategy(random.Random(seed))
        left.validate()
        assert left == right
        assert left.second_land_day > left.first_land_day
        assert left.cow_target + left.sheep_target <= 16


def test_emergency_animal_feed_wins_on_current_tile():
    obs = observation(step=120)
    obs["private"]["inventories"][0]["WHEAT"] = 1
    obs["farms"][0]["tiles"][4][4] = {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": 0,
        "yield_units": 0,
        "fed_today": False,
        "consecutive_unfed": 1,
        "cared_today": False,
        "fertilizer_available": False,
    }
    action = make_agent(Strategy())(obs)
    assert action["farmer"] == ["FEED"]


def test_weed_is_repaired_before_route_construction():
    obs = observation()
    obs["farms"][0]["tiles"][4][4] = {"kind": "WEED"}
    action = make_agent(Strategy())(obs)
    assert action["farmer"] == ["DIG"]


def test_terminal_turn_sells_everything_and_never_invests():
    obs = observation(step=700)
    obs["private"]["shed"].update({"WHEAT": 7, "MILK": 4, "WOOL": 3})
    obs["private"]["inventories"][0]["MILK"] = 2
    action = make_agent(Strategy())(obs)
    assert {tuple(order) for order in action["market"]} >= {
        ("SELL", "WHEAT", 7),
        ("SELL", "MILK", 4),
        ("SELL", "WOOL", 3),
    }
    assert not any(order[0] in {"HIRE", "BUY_LAND"} or order[0].startswith("BUY_") for order in action["market"])
    assert action["farmer"] == ["DROP"]


def test_sale_slots_are_ranked_by_modeled_price_impact():
    obs = observation(step=200)
    obs["private"]["shed"].update({"STRAWBERRY": 20, "MELON": 20})
    action = make_agent(Strategy(sell_price_ratio=0.7))(obs)
    sales = [order for order in action["market"] if order[0] == "SELL"]
    impacts = []
    for _, item, quantity in sales:
        inventory = obs["market"]["inventory"][item]
        impacts.append(quantity * (obs["market"]["prices"][item] - modeled_price(item, inventory + quantity)))
    assert impacts == sorted(impacts, reverse=True)


def test_agent_aligns_hand_actions_and_resets_episode_state():
    agent = make_agent(Strategy())
    action = agent(observation(hands=[[3, 4], [4, 3]]))
    assert len(action["hands"]) == 2
    assert len(action["market"]) <= 10
    agent.state.commitments[0] = (("stale",), 20)
    agent(observation(step=0))
    assert all(key != ("stale",) for key, _ in agent.state.commitments.values())


def test_default_route_completes_and_liquidates_full_season():
    from kaggriculture_lab.episode_metrics import episode_metrics
    from sim.runner import run_match

    result = run_match(
            ("agents.adaptive_route_planner.main:agent", "pass"),
            70117,
            {"episodeSteps": 720},
            analyzer=episode_metrics,
        )
    route = result.seats[0]
    assert not result.failed
    assert route.reward is not None and route.reward > 3000
    assert route.metrics["final_shed_items"] == 0
    assert route.metrics["final_carried_items"] == 0
    assert route.metrics["market_orders"].get("BUY_LAND") == 2
    assert route.metrics["field_actions"].get("FEED", 0) > 0
    assert route.metrics["field_actions"].get("FERTILIZE", 0) > 0
