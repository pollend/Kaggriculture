"""Coverage for the economic modules v5 wraps, plus its own controller.

`strategy.py` and `portfolio.py` are the v4 economics that `strategy_v5` and
`portfolio_v5` still delegate to, so they keep their own tests. Everything that
executes -- the controller, market orders -- is exercised through v5.
"""

from __future__ import annotations

import random
from copy import deepcopy

import pytest

from agents.mixed_farm_v5.agent_v5 import make_agent
from agents.mixed_farm_v5.labor_v5 import LaborPlan
from agents.mixed_farm_v5.market_v5 import market_orders
from agents.mixed_farm_v5.portfolio import build_plan
from agents.mixed_farm_v5.portfolio_v5 import build_plan as build_plan_v5
from agents.mixed_farm_v5.rules import ANIMALS, CROPS, PRODUCTS
from agents.mixed_farm_v5.scheduler import enumerate_tasks
from agents.mixed_farm_v5.strategy import Strategy, sample_strategy
from agents.mixed_farm_v5.strategy_v5 import Strategy as StrategyV5
from agents.mixed_farm_v5.training import (
    MOON_FAMILY_ID,
    MixedFarmSurrogateTrainer,
    adapter,
)
from sim.runner import run_match


def idle_labor(money: float = 3000.0) -> LaborPlan:
    """A labor plan that hires nothing, so market tests isolate trade orders."""

    return LaborPlan(0.0, 0, 0, 0, 0, 0, money)


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


def test_v2_strategy_defaults_and_v3_model_features_are_compatible():
    legacy = Strategy().to_dict()
    for name in (
        "minimum_hands",
        "feed_carry_target",
        "sell_batch_size",
        "land_expansion_day",
    ):
        legacy.pop(name)

    strategy = Strategy.from_dict(legacy)
    features = strategy.to_features()

    assert strategy.minimum_hands == 0
    assert strategy.feed_carry_target == 8
    assert strategy.sell_batch_size == 100
    assert strategy.land_expansion_day == 30
    assert features["derived_target_animals"] == 9
    assert features["derived_capacity_tiles"] == 100
    assert features["derived_capacity_utilization"] == pytest.approx(0.41)


def test_minimum_hands_must_fit_beneath_maximum():
    with pytest.raises(ValueError, match="minimum_hands"):
        Strategy(minimum_hands=5, max_hands=4)


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


def test_small_legacy_crop_target_does_not_deadlock_livestock():
    obs = observation()
    strategy = StrategyV5(
        max_hands=5,
        cash_reserve=866,
        target_crop_tiles=7,
        target_goose=4,
        target_cow=8,
        target_sheep=3,
    )
    plan = build_plan_v5(obs, obs["farms"][0], strategy)
    orders = market_orders(
        obs, obs["farms"][0], obs["private"], plan, idle_labor(), strategy
    )

    assert sum(plan.target_crops.values()) == 31
    assert any(order[0] == "BUY_ANIMAL" for order in orders)


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

    assert sum(compact_plan.target_crops.values()) == 7
    assert sum(expansive_plan.target_crops.values()) == 31

    # The same preferences drive which intents v5 is willing to fund.
    compact_v5 = build_plan_v5(obs, obs["farms"][0], StrategyV5.from_dict(compact.to_dict()))
    expansive_v5 = build_plan_v5(obs, obs["farms"][0], StrategyV5.from_dict(expansive.to_dict()))
    compact_kinds = {intent.kind for intent in compact_v5.intents}
    expansive_kinds = {intent.kind for intent in expansive_v5.intents}

    assert "animal" not in compact_kinds, "opening_animal_batch = 0 buys no livestock"
    assert "animal" in expansive_kinds


def test_land_purchase_requires_committed_capacity_and_roi():
    obs = observation()
    obs["farms"][0]["money"] = 5000
    obs["private"]["seeds"]["MELON"] = 20
    plan = build_plan(obs, obs["farms"][0], Strategy())
    assert plan.buy_land
    assert plan.next_land_price == 1000


def test_land_expansion_day_can_bypass_commitment_hurdle():
    obs = observation(step=18 * 24)
    obs["farms"][0]["money"] = 5000
    strategy = Strategy(
        max_quadrants=2,
        target_crop_tiles=50,
        land_commitment_fraction=1,
        land_hurdle=2,
        land_expansion_day=18,
    )

    assert build_plan(obs, obs["farms"][0], strategy).buy_land


def test_portfolio_caps_combined_assets_to_strategy_capacity():
    obs = observation()
    strategy = Strategy(
        max_quadrants=1,
        target_crop_tiles=56,
        crop_target_floor=47,
        target_goose=4,
        target_cow=14,
        target_sheep=9,
    )
    plan = build_plan(obs, obs["farms"][0], strategy)

    assert sum(plan.target_crops.values()) + sum(plan.target_animals.values()) <= 25


def test_capacity_clamp_fits_a_herd_a_large_crop_target_cannot_veto():
    """`target_crop_tiles` is a want, not a floor the herd must clear.

    Treating it as one zeroed a whole herd whenever the crop target alone exceeded
    the board, even though the crop side is clamped to the leftover tiles anyway.
    """

    obs = observation()
    strategy = Strategy(
        max_quadrants=1,
        target_crop_tiles=32,
        crop_target_floor=16,
        target_goose=13,
        target_cow=17,
        target_sheep=17,
    )
    plan = build_plan(obs, obs["farms"][0], strategy)

    assert sum(plan.target_animals.values()) > 0
    assert sum(plan.target_crops.values()) + sum(plan.target_animals.values()) <= 25


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


def test_feed_carry_target_controls_pickup_distribution():
    obs = observation(step=120, hands=[[3, 4], [2, 4]])
    obs["private"]["shed"]["WHEAT"] = 3
    for x in range(3):
        obs["farms"][0]["tiles"][0][x] = {
            "kind": "COOP", "animal": "GOOSE", "placed_day": 0,
            "yield_units": 0, "fed_today": False, "consecutive_unfed": 0,
            "cared_today": False, "fertilizer_available": False,
        }
    strategy = Strategy(feed_carry_target=2)
    plan = build_plan(obs, obs["farms"][0], strategy)
    tasks = enumerate_tasks(
        obs, obs["farms"][0], obs["private"], plan,
        [(4, 4), (3, 4), (2, 4)], obs["private"]["inventories"], strategy,
    )
    pickups = [task.action for task in tasks if task.kind == "PICKUP_WHEAT"]

    assert pickups == [("PICKUP", "WHEAT", 2), ("PICKUP", "WHEAT", 1)]


def test_current_herd_can_buy_feed_below_cash_reserve():
    obs = observation(step=120)
    obs["farms"][0]["money"] = 500
    obs["farms"][0]["tiles"][4][4] = {
        "kind": "COOP", "animal": "GOOSE", "placed_day": 0,
        "yield_units": 0, "fed_today": False, "consecutive_unfed": 0,
        "cared_today": False, "fertilizer_available": False,
    }
    strategy = StrategyV5()
    plan = build_plan_v5(obs, obs["farms"][0], strategy)
    orders = market_orders(
        obs, obs["farms"][0], obs["private"], plan, idle_labor(500.0), strategy
    )

    assert plan.wheat_reserve > 0, "a live herd must reserve feed"
    # Feeding is the one purchase allowed to cross the cash reserve.
    assert ["BUY_PRODUCT", "WHEAT", plan.wheat_reserve] in orders


def test_terminal_market_sells_and_does_not_buy_assets():
    obs = observation(step=710)
    obs["private"]["shed"]["MELON"] = 4
    strategy = StrategyV5()
    plan = build_plan_v5(obs, obs["farms"][0], strategy)
    orders = market_orders(
        obs, obs["farms"][0], obs["private"], plan, idle_labor(), strategy
    )

    assert ["SELL", "MELON", 4] in orders
    assert not any(order[0] in {"BUY_LAND", "BUY_ANIMAL", "BUY_SEED"} for order in orders)


def test_sell_batch_size_limits_cash_cycle_quantity():
    obs = observation(step=120)
    obs["private"]["shed"]["MELON"] = 30
    strategy = StrategyV5(sell_batch_size=7, sell_price_ratio=0.7)
    plan = build_plan_v5(obs, obs["farms"][0], strategy)
    orders = market_orders(
        obs, obs["farms"][0], obs["private"], plan, idle_labor(), strategy
    )

    assert ["SELL", "MELON", 7] in orders


def test_broke_at_the_reserve_still_liquidates_stock():
    """Money settles at exactly `cash_reserve`, never below it.

    The buy budget is `money - cash_reserve`, so once it reaches zero nothing is
    spent and the balance stops moving. A strict `money < cash_reserve` distress
    check therefore never fires, and a high `sell_price_ratio` sat on a full shed
    for the rest of the season.
    """

    obs = observation(step=432)
    obs["farms"][0]["money"] = 2523.0
    obs["private"]["shed"]["CARROT"] = 26
    strategy = StrategyV5(cash_reserve=2523, sell_price_ratio=1.25, shed_pressure_fraction=0.9)
    plan = build_plan_v5(obs, obs["farms"][0], strategy)
    orders = market_orders(
        obs, obs["farms"][0], obs["private"], plan, idle_labor(2523.0), strategy
    )

    assert ["SELL", "CARROT", 26] in orders


def test_hires_are_funded_before_discretionary_assets():
    obs = observation(step=1)
    strategy = StrategyV5()
    plan = build_plan_v5(obs, obs["farms"][0], strategy)
    labor = LaborPlan(12.0, 2, 2, 2, 22, 0, 2998.0)
    orders = market_orders(
        obs, obs["farms"][0], obs["private"], plan, labor, strategy
    )

    assert sum(order[0] == "HIRE" for order in orders) == 2
    assert len(orders) <= 10


def test_agent_returns_one_action_per_hand_and_resets_commitments():
    agent = make_agent(StrategyV5())
    first = agent(observation(hands=[[3, 4], [4, 3]]))
    assert len(first["hands"]) == 2
    assert len(first["market"]) <= 10
    agent.state.commitments[0] = ("stale",)
    agent(observation(step=0))
    assert ("stale",) not in agent.state.commitments.values()


def test_initial_population_is_fresh_samples_and_reads_no_history():
    class Reader:
        def fetchall(self, sql, parameters):  # pragma: no cover - must never run
            raise AssertionError("generation zero must not read seed sessions")

    initialized = adapter.initialize_population(20, random.Random(17), Reader(), ())

    assert [label for _, label in initialized] == ["random"] * 20
    assert len({strategy.strategy_id for strategy, _ in initialized}) == 20
    for strategy, _ in initialized:
        assert isinstance(strategy, StrategyV5)


def test_surrogate_proposals_mix_prediction_and_uncertainty(tmp_path):
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    from agents.mixed_farm_v5.strategy_v5 import sample_strategy as sample_v5

    training = [sample_v5(random.Random(seed)) for seed in range(60)]
    frame = pd.DataFrame([adapter.encode(strategy) for strategy in training])
    columns = list(frame.columns)
    model = RandomForestRegressor(n_estimators=12, min_samples_leaf=2, random_state=5)
    model.fit(frame[columns], [strategy.labor_roi_hurdle for strategy in training])
    productive = RandomForestRegressor(n_estimators=12, min_samples_leaf=2, random_state=6)
    idle = RandomForestRegressor(n_estimators=12, min_samples_leaf=2, random_state=7)
    productive.fit(frame[columns], [0.25] * len(training))
    idle.fit(frame[columns], [0.20] * len(training))
    artifact = tmp_path / "surrogate.joblib"
    joblib.dump({
        "core_model": model,
        "productive_model": productive,
        "idle_model": idle,
        "feature_columns": columns,
        "quality": {"rows": 60, "core_error": 0.5, "productive_mae": 0.02, "idle_mae": 0.02},
    }, artifact)

    class Reader:
        def fetchone(self, sql, parameters):
            return (str(artifact),)

    proposed = adapter.propose(
        count=4,
        parents=[],
        rng=random.Random(23),
        reader=Reader(),
        model_id="model",
        candidate_pool=100,
    )

    assert len(proposed) == 4
    assert len({strategy.strategy_id for strategy in proposed}) == 4


def test_training_records_use_shared_features_and_equal_anchor_rollups():
    base_rows = []
    anchor_rows = []
    for generation in range(2):
        generation_id = f"g{generation}"
        for index in range(4):
            strategy = sample_strategy(random.Random(generation * 10 + index))
            revision_id = f"r{generation}-{index}"
            base_rows.append(
                (
                    strategy.canonical_json(),
                    100.0 + index,
                    10.0 + index,
                    20_000.0 + 100 * index,
                    3_000.0,
                    1_200.0 + index,
                    1_500.0,
                    2.0,
                    100.0 + index,
                    generation,
                    revision_id,
                    generation_id,
                    "random",
                )
            )
            anchor_rows.extend(
                (
                    (generation_id, revision_id, MOON_FAMILY_ID, -50_000.0 + index * 1_000),
                    (generation_id, revision_id, "other", -10_000.0 - index * 500),
                )
            )

    class Reader:
        def fetchall(self, sql, parameters):
            return anchor_rows if "pairing_kind = 'anchor'" in sql else base_rows

    frame = MixedFarmSurrogateTrainer._records(Reader(), ["g0", "g1"])

    assert len(frame) == 8
    assert set(frame["anchor_opponents"]) == {2}
    assert frame["target"].notna().all()
    assert "derived_capacity_utilization" in frame.columns


def test_balanced_agent_completes_a_full_season_profitably():
    result = run_match(("agents/mixed_farm_v5/main.py", "pass"), seed=17)
    assert not result.failed
    assert result.seats[0].reward is not None and result.seats[0].reward > 3000
