from __future__ import annotations

from copy import deepcopy

import numpy as np

from agents.mixed_farm_rf_market.agent import make_agent
from agents.mixed_farm_rf_market.market import learned_market_orders
from agents.mixed_farm_rf_market.policy import (
    MarketPolicyBundle,
    choose_purchase,
    choose_reserves,
    enumerate_purchase_candidates,
    market_state_features,
    purchase_feature_matrix,
)
from agents.mixed_farm_rf_market.portfolio import build_plan
from agents.mixed_farm_rf_market.rules import ANIMALS, CROPS, PRODUCTS
from agents.mixed_farm_rf_market.strategy import Strategy
from agents.mixed_farm_rf_market.training import DecisionDataset, fit_forests, split_by_episode
from kaggriculture_lab.identity import AgentManifest
from sim.runner import run_match


def observation(*, step: int = 0, money: float = 3000.0) -> dict:
    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": money,
        "tiles": tiles,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
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
            "inventories": [{}],
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


class ColumnModel:
    def __init__(self, column: int, multiplier: float = 1.0):
        self.column = column
        self.multiplier = multiplier

    def predict(self, rows):
        return np.asarray(rows)[:, self.column] * self.multiplier


def test_manifest_defines_an_isolated_trainer_only_family():
    manifest = AgentManifest.load("agents/mixed_farm_rf_market/agent.toml")
    assert manifest.family_id == "mixed_farm_rf_market"
    assert manifest.entry_point == "agents.mixed_farm_rf_market.agent:make_agent"
    assert manifest.models[0].model_id == "mixed_farm_rf_market.market_policy"


def test_purchase_candidates_and_features_are_stable():
    obs = observation()
    strategy = Strategy()
    plan = build_plan(obs, obs["farms"][0], strategy)
    state = market_state_features(obs, obs["farms"][0], obs["private"], plan)
    candidates = enumerate_purchase_candidates(
        obs, obs["farms"][0], obs["private"], plan, strategy
    )
    matrix = purchase_feature_matrix(state, candidates)

    assert len(candidates) > 10
    assert matrix.shape[0] == len(candidates)
    assert matrix.shape[1] > len(state)
    assert {candidate.target_land for candidate in candidates} == {1, 2}
    assert candidates == enumerate_purchase_candidates(
        obs, obs["farms"][0], obs["private"], plan, strategy
    )


def test_purchase_forest_can_select_land_expansion():
    obs = observation(money=5000)
    strategy = Strategy()
    plan = build_plan(obs, obs["farms"][0], strategy)
    state = market_state_features(obs, obs["farms"][0], obs["private"], plan)
    candidates = enumerate_purchase_candidates(
        obs, obs["farms"][0], obs["private"], plan, strategy
    )
    purchase, _, _ = choose_purchase(
        state,
        candidates,
        ColumnModel(-3),  # target_land is the third feature from the end
        epsilon=0,
        rng=__import__("random").Random(1),
    )
    reserves = {item: int(obs["private"]["shed"].get(item, 0)) for item in PRODUCTS}
    orders = learned_market_orders(
        obs, obs["farms"][0], obs["private"], plan, purchase, reserves, strategy
    )
    assert ["BUY_LAND"] in orders


def test_sale_forest_controls_what_is_retained():
    obs = observation(step=120)
    obs["private"]["shed"]["MELON"] = 20
    strategy = Strategy()
    plan = build_plan(obs, obs["farms"][0], strategy)
    state = market_state_features(obs, obs["farms"][0], obs["private"], plan)
    reserves, records = choose_reserves(
        state,
        obs["private"]["shed"],
        obs["market"]["prices"],
        ColumnModel(-3, -1.0),  # prefer the smallest reserve
        survival_wheat=0,
        epsilon=0,
        rng=__import__("random").Random(1),
    )
    assert reserves["MELON"] == 0
    assert any(item == "MELON" for item, _, _ in records)


def test_executor_preserves_emergency_feed_and_terminal_liquidation():
    obs = observation(step=710, money=500)
    obs["private"]["shed"]["MELON"] = 4
    obs["farms"][0]["tiles"][0][0] = {
        "kind": "COOP",
        "animal": "GOOSE",
        "fed_today": False,
        "cared_today": False,
        "yield_units": 0,
    }
    strategy = Strategy()
    plan = build_plan(obs, obs["farms"][0], strategy)
    candidate = enumerate_purchase_candidates(
        obs, obs["farms"][0], obs["private"], plan, strategy
    )[0]
    orders = learned_market_orders(
        obs,
        obs["farms"][0],
        obs["private"],
        plan,
        candidate,
        {item: 100 for item in PRODUCTS},
        strategy,
    )
    assert ["SELL", "MELON", 4] in orders
    assert not any(order[0] in {"BUY_SEED", "BUY_ANIMAL", "BUY_LAND"} for order in orders)

    operating = observation(step=120, money=500)
    operating["farms"][0]["tiles"][0][0] = obs["farms"][0]["tiles"][0][0]
    operating_plan = build_plan(operating, operating["farms"][0], strategy)
    operating_candidate = enumerate_purchase_candidates(
        operating,
        operating["farms"][0],
        operating["private"],
        operating_plan,
        strategy,
    )[0]
    operating_orders = learned_market_orders(
        operating,
        operating["farms"][0],
        operating["private"],
        operating_plan,
        operating_candidate,
        {item: 100 for item in PRODUCTS},
        strategy,
    )
    feed_orders = [order for order in operating_orders if order[:2] == ["BUY_PRODUCT", "WHEAT"]]
    assert feed_orders and feed_orders[0][2] >= 1


def test_forest_training_groups_rows_by_episode_and_bundle_round_trips(tmp_path):
    rng = np.random.default_rng(7)
    purchase_x = rng.normal(size=(60, 12)).astype(np.float32)
    sale_x = rng.normal(size=(80, 14)).astype(np.float32)
    purchase_y = (purchase_x[:, 0] * 10 + purchase_x[:, 1]).astype(np.float32)
    sale_y = (sale_x[:, 0] * -5 + sale_x[:, 2]).astype(np.float32)
    dataset = DecisionDataset(
        purchase_x,
        purchase_y,
        np.asarray([f"p{i // 6}" for i in range(60)]),
        sale_x,
        sale_y,
        np.asarray([f"s{i // 8}" for i in range(80)]),
    )
    parts = split_by_episode(dataset, 5)
    assert len(parts) == 5
    assert sum(len(part.purchase_y) for part in parts) == len(purchase_y)
    assert not (
        set(parts[0].purchase_group.tolist())
        & set(parts[1].purchase_group.tolist())
    )
    purchase, sale, metrics = fit_forests(
        dataset,
        {"n_estimators": 12, "min_samples_leaf": 2, "max_features": 0.7, "seed": 3},
    )
    assert metrics["purchase"]["episodes"] == 10
    assert metrics["sale"]["episodes"] == 10

    path = tmp_path / "market.joblib"
    MarketPolicyBundle(Strategy(), purchase, sale, "dataset").dump(path)
    restored = MarketPolicyBundle.load(path)
    np.testing.assert_allclose(
        restored.purchase_model.predict(purchase_x[:3]), purchase.predict(purchase_x[:3])
    )
    assert restored.strategy == Strategy()


def test_agent_fallback_and_short_episode_are_valid():
    action = make_agent()(observation())
    assert len(action["market"]) <= 10
    assert len(action["hands"]) == 0

    result = run_match(
        ("agents.mixed_farm_rf_market.agent:make_agent", "pass"),
        seed=17,
        config={"episodeSteps": 48},
    )
    assert not result.failed
