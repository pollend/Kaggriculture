from pathlib import Path

import numpy as np
import pytest

from agents.dqn_job_scorer.hybrid import choose_plan
from agents.dqn_job_scorer.main import JobScorerAgent
from agents.dqn_job_scorer.market import MarketConfig, market_orders
from agents.dqn_job_scorer.market_policy import (
    MARKET_FEATURE_DIM,
    MARKET_STATE_DIM,
    enumerate_market_candidates,
)
from agents.dqn_job_scorer.policy import QNetwork
from agents.dqn_job_scorer.rules import PRODUCTS


def _farm(tile=None):
    tiles = [[None] * 10 for _ in range(10)]
    if tile is not None:
        tiles[0][0] = tile
    return {
        "money": 3000,
        "tiles": tiles,
        "farmer": [0, 0],
        "hands": [],
        "hires_today": 0,
        "unlocked_quadrants": ["NW"],
    }


def _obs(farm, *, shops=(), prices=None, step=0):
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, _farm()],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"prices": prices or {}, "inventory": {}},
        "town": {"unlocked_shops": list(shops)},
    }


def test_yarn_demand_opens_mixed_sheep_regime():
    farm = _farm()
    plan = choose_plan(_obs(farm, shops=("YARN_STORE",), prices={"WOOL": 400, "WHEAT": 10}), farm)
    assert plan.mode == "MIXED"
    assert plan.animal == "SHEEP"
    assert plan.target_animals > 0


def test_emergency_feed_preempts_network_score():
    tile = {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": 0,
        "yield_units": 0,
        "fed_today": False,
        "consecutive_unfed": 1,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }
    farm = _farm(tile)
    obs = _obs(farm)
    obs["private"]["inventories"] = [{"WHEAT": 1}]
    action = JobScorerAgent(QNetwork.random(seed=7))(obs)
    assert action["farmer"] == ["FEED"]


def test_market_buys_feed_before_new_animals():
    farm = _farm()
    obs = _obs(farm, shops=("YARN_STORE",), prices={"WHEAT": 10, "WOOL": 400})
    plan = choose_plan(obs, farm)
    orders = market_orders(obs, farm, obs["private"], 0, 0, False, MarketConfig(), 720, 24, plan)
    feed = next(index for index, order in enumerate(orders) if order[:2] == ["BUY_PRODUCT", "WHEAT"])
    animal = next(index for index, order in enumerate(orders) if order[:2] == ["BUY_ANIMAL", "SHEEP"])
    assert feed < animal
    assert len(orders) <= 10


def test_market_features_include_private_and_public_inventory():
    farm = _farm()
    obs = _obs(farm, prices={"WHEAT": 40})
    obs["private"]["shed"] = {"WHEAT": 7, "EGG": 3}
    obs["private"]["seeds"] = {"MELON": 2}
    obs["private"]["inventories"] = [{"WOOL": 4}]
    obs["market"]["inventory"] = {"WHEAT": 9000}

    candidates = enumerate_market_candidates(obs, farm, obs["private"])

    assert candidates.state.shape == (MARKET_STATE_DIM,)
    assert candidates.purchase_static.shape[1] + candidates.state.shape[0] == MARKET_FEATURE_DIM
    assert candidates.reserve_values[PRODUCTS.index("WHEAT")][-1] == 7
    assert candidates.reserve_values[PRODUCTS.index("EGG")][-1] == 3


def test_learned_reserve_sells_only_excess_inventory():
    farm = _farm()
    obs = _obs(farm)
    obs["private"]["shed"] = {"WHEAT": 5}
    reserves = [0] * len(PRODUCTS)
    reserves[PRODUCTS.index("WHEAT")] = 2
    from agents.dqn_job_scorer.hybrid import HybridPlan

    plan = HybridPlan("CROP", "GOOSE", 0, 0, 0, "WHEAT", 0, 1, tuple(reserves))
    orders = market_orders(obs, farm, obs["private"], 0, 0, False, MarketConfig(), 720, 24, plan)
    assert ["SELL", "WHEAT", 3] in orders


def test_terminal_liquidation_overrides_learned_reserve():
    farm = _farm()
    obs = _obs(farm, step=719)
    obs["private"]["shed"] = {"MELON": 6}
    from agents.dqn_job_scorer.hybrid import HybridPlan

    plan = HybridPlan("CROP", "GOOSE", 0, 0, 0, "WHEAT", 0, 1, tuple([100] * len(PRODUCTS)))
    orders = market_orders(obs, farm, obs["private"], 719, 23, True, MarketConfig(), 720, 24, plan)
    assert orders == [["SELL", "MELON", 6]]


def test_schema_one_checkpoint_is_rejected(tmp_path: Path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, w0=np.zeros((72, 128), dtype=np.float32), b0=np.zeros(128, dtype=np.float32))
    with pytest.raises(ValueError, match="feature dimension 72"):
        QNetwork.load(path)
