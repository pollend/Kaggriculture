from __future__ import annotations

import random
from copy import deepcopy

from agents.configurable_mixed_farm_surrogate.agent import make_agent
from agents.configurable_mixed_farm_surrogate.market import market_orders
from agents.configurable_mixed_farm_surrogate.portfolio import Plan, build_plan
from agents.configurable_mixed_farm_surrogate.rules import ANIMALS, CROPS, PRODUCTS
from agents.configurable_mixed_farm_surrogate.scheduler import enumerate_tasks
from agents.configurable_mixed_farm_surrogate.strategy import Strategy, sample_strategy
from sim.runner import run_match


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
            "seeds": {crop: 0 for crop in CROPS},
            "inventories": [{} for _ in range(1 + len(hands))],
        },
        "market": {
            "inventory": {item: 10_000 for item in PRODUCTS},
            "prices": {
                "WHEAT": 25, "CARROT": 35, "TOMATO": 60,
                "STRAWBERRY": 120, "MELON": 250,
                "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
            },
        },
        "town": {"unlocked_shops": []},
    }


def empty_plan() -> Plan:
    return Plan(
        {crop: 0 for crop in CROPS},
        {animal: 0 for animal in ANIMALS},
        {crop: 0.0 for crop in CROPS},
        {animal: 0.0 for animal in ANIMALS},
        0,
        0,
        0,
        False,
        None,
    )


def test_strategy_sampling_stays_valid_and_stable():
    learned_limits = {
        "crop_target_floor",
        "crop_tiles_per_animal",
        "opening_crop_tranche",
        "animal_unlock_crop_tiles",
        "opening_animal_batch",
        "workload_per_hand",
        "land_commitment_fraction",
    }
    for seed in range(20):
        strategy = sample_strategy(random.Random(seed))
        strategy.validate()
        assert 0 <= strategy.max_hands <= 24
        assert 0 <= strategy.target_crop_tiles <= 100
        assert 0 <= strategy.land_commitment_fraction <= 1
        assert learned_limits <= strategy.to_features().keys()
        assert Strategy.from_dict(strategy.to_dict()).strategy_id == strategy.strategy_id


def test_portfolio_includes_all_animals_and_self_supplied_feed():
    obs = observation()
    plan = build_plan(obs, obs["farms"][0], Strategy())
    assert plan.target_animals == {"GOOSE": 1, "COW": 6, "SHEEP": 2}
    assert plan.target_crops["WHEAT"] >= 8
    assert not plan.buy_land, "land waits until the current quadrant is substantially committed"


def test_opening_labor_tracks_committed_work_instead_of_season_target():
    obs = observation()
    strategy = Strategy(max_hands=7, target_crop_tiles=75, target_goose=1, target_cow=6, target_sheep=3)
    assert build_plan(obs, obs["farms"][0], strategy).target_hands == 0

    obs["private"]["seeds"]["WHEAT"] = 8
    obs["private"]["shed"]["SHEEP"] = 5
    assert build_plan(obs, obs["farms"][0], strategy).target_hands == 5


def test_opening_market_stages_crops_before_livestock_and_caps_labor():
    obs = observation()
    strategy = Strategy(max_hands=7, target_crop_tiles=75, target_goose=1, target_cow=6, target_sheep=3)
    plan = build_plan(obs, obs["farms"][0], strategy)
    orders = market_orders(obs, obs["farms"][0], obs["private"], plan, strategy)

    seed_orders = [order for order in orders if order[0] == "BUY_SEED"]
    animal_orders = [order for order in orders if order[0] == "BUY_ANIMAL"]
    assert sum(int(order[2]) for order in seed_orders) >= 8
    assert seed_orders and animal_orders
    assert max(orders.index(order) for order in seed_orders) < min(
        orders.index(order) for order in animal_orders
    )
    assert sum(order[0] == "HIRE" for order in orders) <= 4


def test_small_legacy_crop_target_does_not_deadlock_livestock_or_labor():
    obs = observation()
    strategy = Strategy(
        max_hands=5,
        cash_reserve=866,
        target_crop_tiles=7,
        target_goose=4,
        target_cow=8,
        target_sheep=3,
    )
    plan = build_plan(obs, obs["farms"][0], strategy)
    orders = market_orders(obs, obs["farms"][0], obs["private"], plan, strategy)

    assert sum(plan.target_crops.values()) == 31
    assert any(order[0] == "BUY_ANIMAL" for order in orders)
    assert sum(order[0] == "HIRE" for order in orders) >= strategy.max_hands - 1


def test_operating_caps_are_strategy_features():
    obs = observation()
    compact = Strategy(
        max_hands=3,
        target_crop_tiles=7,
        target_goose=4,
        target_cow=8,
        target_sheep=3,
        crop_target_floor=0,
        crop_tiles_per_animal=0,
        opening_crop_tranche=4,
        animal_unlock_crop_tiles=7,
        opening_animal_batch=0,
        workload_per_hand=24,
        land_commitment_fraction=1,
    )
    expansive = Strategy.from_dict(
        {
            **compact.to_dict(),
            "crop_target_floor": 16,
            "crop_tiles_per_animal": 1,
            "opening_crop_tranche": 12,
            "animal_unlock_crop_tiles": 4,
            "opening_animal_batch": 8,
            "workload_per_hand": 4,
            "land_commitment_fraction": 0.25,
        }
    )

    compact_plan = build_plan(obs, obs["farms"][0], compact)
    expansive_plan = build_plan(obs, obs["farms"][0], expansive)
    compact_orders = market_orders(
        obs, obs["farms"][0], obs["private"], compact_plan, compact
    )
    expansive_orders = market_orders(
        obs, obs["farms"][0], obs["private"], expansive_plan, expansive
    )

    assert sum(compact_plan.target_crops.values()) == 7
    assert sum(expansive_plan.target_crops.values()) == 31
    assert not any(order[0] == "BUY_ANIMAL" for order in compact_orders)
    assert any(order[0] == "BUY_ANIMAL" for order in expansive_orders)
    assert sum(order[0] == "HIRE" for order in expansive_orders) > sum(
        order[0] == "HIRE" for order in compact_orders
    )


def test_land_purchase_requires_committed_capacity_and_roi():
    obs = observation()
    obs["farms"][0]["money"] = 5000
    obs["private"]["seeds"]["MELON"] = 20
    plan = build_plan(obs, obs["farms"][0], Strategy())
    assert plan.buy_land
    assert plan.next_land_price == 1000


def test_owned_animal_exposes_structure_and_daily_logistics():
    obs = observation(step=48)
    obs["private"]["shed"]["COW"] = 1
    obs["private"]["shed"]["WHEAT"] = 2
    plan = build_plan(obs, obs["farms"][0], Strategy())
    tasks = enumerate_tasks(
        obs, obs["farms"][0], obs["private"], plan,
        [(4, 4)], obs["private"]["inventories"], Strategy(),
    )
    assert "BUILD_PASTURE" in {task.kind for task in tasks}
    assert not any(task.target[0] >= 5 or task.target[1] >= 5 for task in tasks)


def test_hungry_animal_gets_feed_and_collection_frontier():
    obs = observation(step=120)
    obs["private"]["inventories"][0]["WHEAT"] = 1
    obs["farms"][0]["tiles"][4][4] = {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 0,
        "yield_units": 2, "fed_today": False, "consecutive_unfed": 1,
        "cared_today": False, "fertilizer_available": True,
        "pending_care_bonus": 0,
    }
    plan = build_plan(obs, obs["farms"][0], Strategy())
    tasks = enumerate_tasks(
        obs, obs["farms"][0], obs["private"], plan,
        [(4, 4)], obs["private"]["inventories"], Strategy(),
    )
    assert {task.kind for task in tasks} >= {
        "FEED", "CARE", "HARVEST_ANIMAL", "COLLECT_FERTILIZER"
    }


def test_current_herd_can_buy_feed_below_cash_reserve():
    obs = observation(step=120)
    obs["farms"][0]["money"] = 500
    obs["farms"][0]["tiles"][4][4] = {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 0,
        "yield_units": 0, "fed_today": False, "consecutive_unfed": 0,
        "cared_today": False, "fertilizer_available": False,
    }
    orders = market_orders(obs, obs["farms"][0], obs["private"], empty_plan(), Strategy())
    assert ["BUY_PRODUCT", "WHEAT", 2] in orders


def test_terminal_market_sells_and_does_not_buy_assets():
    obs = observation(step=710)
    obs["private"]["shed"]["MELON"] = 4
    orders = market_orders(obs, obs["farms"][0], obs["private"], empty_plan(), Strategy())
    assert ["SELL", "MELON", 4] in orders
    assert not any(order[0] in {"BUY_LAND", "BUY_ANIMAL", "BUY_SEED"} for order in orders)


def test_agent_returns_one_action_per_hand_and_resets_commitments():
    agent = make_agent(Strategy())
    first = agent(observation(hands=[[3, 4], [4, 3]]))
    assert len(first["hands"]) == 2
    assert len(first["market"]) <= 10
    agent.state.commitments[0] = ("stale",)
    agent(observation(step=0))
    assert ("stale",) not in agent.state.commitments.values()


def test_balanced_agent_completes_a_full_season_profitably():
    result = run_match(("agents/configurable_mixed_farm_surrogate/main.py", "pass"), seed=17)
    assert not result.failed
    assert result.seats[0].reward is not None and result.seats[0].reward > 3000
