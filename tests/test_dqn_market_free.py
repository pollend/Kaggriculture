"""Correctness guards for the unscaffolded market agent.

These pin the invariants that would otherwise fail silently: a cold start that
cannot train, an action grid that cannot name what real agents do, train/
inference skew, and a demonstration reader that drops most of what it reads.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from agents.dqn_market_free import diagnostics, donor
from agents.dqn_market_free.diagnostics import OrderDiagnostics, diagnose_order_lists
from agents.dqn_market_free.main import MarketFreeAgent, load_network
from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import (
    ACTION_COUNT,
    ACTIONS,
    HIRE_QUANTITIES,
    NOOP_INDEX,
    QUANTITIES,
    Order,
    choose_order,
    feasible_actions,
    nearest_quantity,
    order_index,
    order_is_structurally_possible,
)
from agents.dqn_market_free.planting import market_influenced_roles
from agents.dqn_market_free.policy import OrderQNetwork
from agents.dqn_market_free.training import network as net_module
from agents.dqn_market_free.training import rare
from agents.dqn_market_free.training import updater as updater_module
from agents.dqn_market_free.training.clone import load_demonstrations, record_from_replay
from agents.dqn_market_free.training.collect import (
    EpisodeRecord,
    Recorder,
    actor_sigma_scales,
)
from agents.dqn_market_free.training.replay import ReplayBuffer
from agents.dqn_market_free.training.updater import (
    SELF_SPEC,
    SOURCE_KIND,
    _arena_order_diagnostics,
    _arena_viability,
    _behavior_clone_step,
    _behavior_clone_steps_per_update,
    _reset_learning_rates,
    _self_play_pool,
    initial_network,
    rotate_opponents,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DONOR_PATH = REPO_ROOT / "agents/dqn_market_free/donor.py"
SCORER_DONOR_PATH = REPO_ROOT / "agents/dqn_market_scorer/donor.py"
SNAPSHOT_PATH = (
    REPO_ROOT / "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"
)


def test_local_donor_matches_the_scorer_snapshot_and_ships_its_license():
    assert DONOR_PATH.read_bytes() == SCORER_DONOR_PATH.read_bytes()
    notice = (REPO_ROOT / "agents/dqn_market_free/NOTICE").read_text()
    assert "donor.py IS A MODIFIED DERIVATIVE" in notice
    assert "a8da70675b6d5f02372f24ee37f8c832fafb0ca74351add28bbcaab06f661d91" in notice
    assert (REPO_ROOT / "agents/dqn_market_free/LICENSE-pilkwang").is_file()


def test_every_declared_code_path_exists_and_excludes_training():
    """A declared-but-missing path fails `kaglab bundle`, not a test, otherwise.

    `weights.npz` is deliberately not declared: the league bundles this agent as
    an anchor long before a checkpoint has been exported. The submission
    exporter is where its absence is caught instead.
    """

    import tomllib

    with (REPO_ROOT / "agents/dqn_market_free/agent.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    paths = [Path(value) for value in manifest["code_paths"]]
    assert paths
    for relative in paths:
        assert (REPO_ROOT / relative).is_file(), f"declared but missing: {relative}"
        assert "training" not in relative.parts
        assert "competitor_" not in relative.as_posix()
    assert Path("agents/dqn_market_free/LICENSE-pilkwang") in paths
    assert Path("agents/dqn_market_free/NOTICE") in paths
    assert Path("agents/dqn_market_free/weights.npz") not in paths


def test_a_submission_without_exported_weights_is_refused(monkeypatch):
    """Bundling this agent unweighted ships a policy that never trades.

    `main.load_network` falls back to the silent network rather than to a strong
    donor, so unlike the scorer there is no acceptable weightless submission.
    The weights are absent from `code_paths` by design, which is exactly why the
    exporter has to ask for them separately.
    """

    from agents.dqn_market_free import export_submission

    manifest = export_submission._manifest()

    monkeypatch.setattr(
        export_submission, "WEIGHTS", Path("agents/dqn_market_free/no-such-weights.npz")
    )
    with pytest.raises(FileNotFoundError, match="never trades"):
        export_submission._inference_files(manifest)

    # Present, and not declared: the exporter has to add it to the archive
    # itself rather than assume the manifest carried it.
    stand_in = Path("agents/dqn_market_free/NOTICE")
    monkeypatch.setattr(export_submission, "WEIGHTS", stand_in)
    included = export_submission._inference_files(manifest)
    assert included.count(stand_in) == 1


def test_local_market_state_encoder_has_forked_from_the_scorer_schema():
    """The two schemas are independent now, and each one owns its own width.

    They used to be asserted equal, which was right while this agent's state was
    a copy of the scorer's. It no longer is: the scorer keeps a donor market
    layer that already knows the price curve, the town's demand and where supply
    is going, so its network never had to see any of it. This one deleted that
    layer, so it does. What is still shared is the board census -- the same
    tiles counted the same way -- and that part is still pinned line for line.
    """

    import inspect

    from agents.dqn_market_free import market_state as local
    from agents.dqn_market_scorer import market_policy as scorer

    assert local.MARKET_STATE_DIM == 238
    assert scorer.MARKET_STATE_DIM == 117
    for name in ("_owned_animals", "_structures", "_tile_census"):
        assert inspect.getsource(getattr(local, name)) == inspect.getsource(getattr(scorer, name))


def test_price_curve_matches_the_donor_price_model():
    """The vectorised curve must be `donor._price_at`, not an approximation of it.

    It exists only because 495 scalar calls a turn is too slow, so any numeric
    difference is a bug in the fast path -- including the integer rounding and
    the floor at 1, which is where a rewrite is most likely to drift.
    """

    from agents.dqn_market_free.market_state import price_curve

    levels = np.arange(8_000, 12_001, 13, dtype=float)
    for item in donor.PRODUCTS:
        expected = np.asarray(
            [float(donor._price_at(item, int(level))) for level in levels]
        )
        assert np.array_equal(price_curve(item, levels), expected), item


def test_price_curve_honours_market_parameter_overrides():
    """`market["params"]` moves the curve, and the features have to move with it."""

    from agents.dqn_market_free.market_state import price_curve

    obs = {"market": {"params": {"WHEAT": {"base": 200}}}}
    assert float(price_curve("WHEAT", donor.MARKET_I0, obs)) == 200.0
    assert float(price_curve("WHEAT", donor.MARKET_I0)) == float(donor.MARKET["WHEAT"][0])


def test_fill_averages_slope_with_depth():
    """A deeper fill clears worse. This is the whole reason the block exists.

    With no action features the network scores each quantity from its own
    column, so unless the state says the 45-unit fill clears below the 8-unit
    one, nothing in the input distinguishes them.
    """

    from agents.dqn_market_free.market_state import FILL_PROBES, _fill_averages

    assert FILL_PROBES == (8, 45)
    for item in donor.PRODUCTS:
        shallow, deep = _fill_averages(item, float(donor.MARKET_I0), None, side="SELL")
        assert deep <= shallow <= float(donor._price_at(item, donor.MARKET_I0))
    for item in ("WHEAT", "FERTILIZER"):
        shallow, deep = _fill_averages(item, float(donor.MARKET_I0), None, side="BUY")
        assert deep >= shallow


def test_town_demand_matches_the_installed_engine_not_the_donor():
    """The donor's town model is wrong for this engine, and by a lot.

    `donor._town_demand_per_day` ramps the town centre 2x after day 10 and 4x
    after day 20; the installed `_town_consume` removes exactly one unit of each
    non-fertilizer product per `townCenterSellInterval` and never ramps. The
    difference lands on the days the agent has to choose between holding stock
    and dumping it, so this pins the corrected version against the engine's own
    tables rather than against the donor.
    """

    from kaggle_environments.envs.kaggriculture import kaggriculture as engine

    from agents.dqn_market_free.market_state import town_demand_per_day

    shops = ["YARN_STORE", "BAKERY", "BAKERY"]
    for day in (0, 9, 10, 20, 29):
        obs = {"day": day, "town": {"unlocked_shops": shops}}
        for item in donor.PRODUCTS:
            expected = 0.0 if item == "FERTILIZER" else 1.0
            for name in shops:
                products = engine.SHOPS[name]
                if item in products:
                    # 24 steps a day / townShopSellInterval of 4, doubled for a
                    # single-product shop.
                    expected += (2 if len(products) == 1 else 1) * (24 / 4)
            assert town_demand_per_day(obs, item) == expected, (day, item)

    late = {"day": 25, "town": {"unlocked_shops": shops}}
    assert town_demand_per_day(late, "WHEAT") != donor._town_demand_per_day(late, "WHEAT")


def test_projected_supply_matches_the_donor_projection():
    """Own and opponent supply are measured by one helper, so they compare.

    The donor's version reads only the other seat, which is why it cannot be
    called for both. This pins the lifted-out copy against it on the case the
    donor does cover.
    """

    from agents.dqn_market_free.market_state import projected_supply

    tiles = [
        [
            {"kind": "PLANT", "crop": "WHEAT", "planted_day": 1, "yield_units": 3},
            {"kind": "PLANT", "crop": "MELON", "planted_day": 6, "yield_units": 0},
        ],
        [
            {"kind": "PASTURE", "animal": "COW", "yield_units": 2},
            {"kind": "PASTURE", "animal": "SHEEP", "yield_units": 0},
        ],
    ]
    obs = {"player": 0, "day": 9, "farms": [{"tiles": []}, {"tiles": tiles}]}
    for item in donor.PRODUCTS:
        for horizon in (0, 1, 3, 7):
            assert projected_supply({"tiles": tiles}, item, 9, horizon) == (
                donor._opponent_visible_supply(obs, item, horizon)
            ), (item, horizon)


def _market_obs(money: float, *, opponent: float = 0.0) -> dict:
    return {
        "player": 0,
        "farms": [{"money": money}, {"money": opponent}],
        "market": {
            "prices": {item: donor.MARKET[item][0] for item in donor.PRODUCTS},
            "inventory": {item: donor.MARKET_I0 for item in donor.PRODUCTS},
        },
    }


def test_history_separates_a_refusal_loop_from_a_fresh_turn():
    """The one thing the snapshot cannot express, and the reason the block exists.

    A policy stuck emitting an order the simulator refuses every turn sees an
    identical observation each time, so no policy defined on the observation
    alone can be the one that stops. The reward already charges for the refusal;
    this is the state that makes the charge predictable.
    """

    from agents.dqn_market_free.market_state import MarketHistory

    history = MarketHistory()
    sell = Order("SELL", "WOOL", 16)

    history.observe(_market_obs(3_000.0), {"money": 3_000.0})
    history.record_decision(sell)
    assert history.consecutive_refusals == 0

    # Money never moves: every one of these was refused.
    for expected in (1, 2, 3):
        history.observe(_market_obs(3_000.0), {"money": 3_000.0})
        history.record_decision(sell)
        assert history.consecutive_refusals == expected
        assert not history.last_filled
        assert history.turns_since_fill == expected

    # A fill resets both counters.
    history.observe(_market_obs(3_200.0), {"money": 3_200.0})
    assert history.last_filled
    assert history.consecutive_refusals == 0
    assert history.turns_since_fill == 0

    # NOOP is not a refusal, however long it runs.
    history.record_decision(Order("NOOP"))
    history.observe(_market_obs(3_200.0), {"money": 3_200.0})
    assert history.consecutive_refusals == 0


def test_history_agrees_with_the_reward_on_what_a_refusal_is():
    """One predicate, so the feature and the penalty cannot drift apart."""

    from agents.dqn_market_free.market_state import order_refused
    from agents.dqn_market_free.training import reward as reward_module

    assert reward_module.order_refused is order_refused
    assert bool(order_refused(True, 0.0))
    assert not bool(order_refused(True, -1.0))
    assert not bool(order_refused(False, 0.0))


def test_history_block_is_zero_filled_until_a_full_day_is_buffered():
    """A short history is encoded as a still market, which is what it is."""

    from agents.dqn_market_free.market_state import HISTORY_LAG, MarketHistory

    history = MarketHistory()
    obs = _market_obs(3_000.0)
    for _ in range(HISTORY_LAG):
        history.observe(obs, {"money": 3_000.0})
        assert not history.ready
        assert history.features(obs)[:20] == [0.0] * 20
    history.observe(obs, {"money": 3_000.0})
    assert history.ready


@pytest.mark.parametrize("family_id", [None, "demonstrator-family"])
def test_demonstration_query_types_nullable_family_parameter(family_id):
    """PostgreSQL must be able to prepare the optional-family query.

    A bare ``%s IS NULL`` leaves that placeholder without an inferred type and
    fails before the query can return even an empty result set.
    """

    class Reader:
        def fetchall(self, sql, parameters):
            assert "CAST(%s AS varchar) IS NULL" in sql
            assert parameters == [10_000.0, family_id, family_id, 1]
            return []

    records, stats = load_demonstrations(
        Reader(),
        limit=1,
        min_final_money=10_000.0,
        family_id=family_id,
    )

    assert records == []
    assert stats.episodes == 0


# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------


def test_action_list_starts_at_noop():
    """Index 0 must stay NOOP.

    A network that scores every action equally breaks ties to the lowest index,
    so this is what makes an untrained policy trade nothing rather than emit a
    random order on turn one. Doing nothing is survivable; the measured
    alternative ends the episode on single-digit money.
    """

    assert ACTIONS[0] == Order("NOOP")
    assert ACTIONS[0].to_wire() == []
    assert len(set(ACTIONS)) == len(ACTIONS)


def test_every_action_round_trips_to_wire_form():
    for order in ACTIONS[1:]:
        wire = order.to_wire()
        assert wire, f"{order} expands to nothing"
        assert all(row[0] == order.op for row in wire)
        if order.op == "HIRE":
            # No hand count exists on the wire: N hands is N identical orders.
            assert wire == [["HIRE"]] * order.quantity
        elif order.op == "BUY_LAND":
            assert wire == [["BUY_LAND"]]
        else:
            assert wire == [[order.op, order.item, order.quantity]]


def test_hire_carries_a_hand_count_because_the_simulator_will_not():
    """The one operation that must not be spread across turns.

    Hand cost is keyed on `hires_today`, hands last only the day they were
    hired, and a hand hired late has already lost part of its shift. Measured on
    a full donor episode, repeated HIRE was 314 of the 505 orders a strictly
    one-order-per-turn agent had to drop.
    """

    assert order_index("HIRE", None, 1) is not None
    eight = order_index("HIRE", None, 8)
    assert ACTIONS[eight].to_wire() == [["HIRE"]] * 8
    # Counts above the donor's own cap clamp rather than falling off the list.
    assert ACTIONS[order_index("HIRE", None, 99)].quantity == HIRE_QUANTITIES[-1]


def test_quantity_ladder_covers_what_real_agents_order():
    """Observed replay quantities run 1..18 then sparsely to 45.

    A ladder that stops short silently snaps a 45-unit sale down, which distorts
    exactly the behaviour the demonstration reader exists to preserve.
    """

    assert max(QUANTITIES) >= 45
    for quantity in range(1, 46):
        snapped = nearest_quantity(quantity)
        assert snapped in QUANTITIES
        assert abs(snapped - quantity) / quantity < 0.35


def test_order_index_maps_observed_orders_and_rejects_unknown_ones():
    assert ACTIONS[order_index("SELL", "WHEAT", 13)].op == "SELL"
    assert ACTIONS[order_index("BUY_LAND", None, 0)] == Order("BUY_LAND")
    # A bare `["HIRE"]` row carries no count; it means one hand.
    assert ACTIONS[order_index("HIRE", None, 0)] == Order("HIRE", None, 1)
    # Unnameable here: the caller must drop the demonstration order, not guess.
    assert order_index("SELL", "NOT_A_PRODUCT", 3) is None
    assert order_index("TELEPORT", "WHEAT", 1) is None


def test_structurally_impossible_orders_are_absent_from_the_action_list():
    """The head has one column per action, so impossible ones are simply not there.

    They used to be retained and masked to `-inf` at score time, to keep
    persisted action indices stable across a change to the eligibility rule.
    Nothing persisted uses that numbering any more, and permanently masked
    columns bought nothing.
    """

    def emittable(op: str, item: str | None = None, quantity: int = 1) -> bool:
        return order_index(op, item, quantity) is not None

    # The game only permits these two BUY_PRODUCT targets.
    assert emittable("BUY_PRODUCT", "WHEAT")
    assert emittable("BUY_PRODUCT", "FERTILIZER")
    assert not emittable("BUY_PRODUCT", "CARROT")

    # Market-directed planting makes every crop actionable. Goose remains
    # impossible because the donor scheduler never creates a goose role.
    assert emittable("BUY_SEED", "TOMATO")
    assert not emittable("BUY_ANIMAL", "GOOSE")
    assert emittable("SELL", "TOMATO")
    assert not emittable("SELL", "EGG")
    assert emittable("BUY_SEED", "MELON")
    assert emittable("BUY_ANIMAL", "COW")
    assert emittable("SELL", "MILK")

    assert len(ACTIONS) == ACTION_COUNT == 186
    assert all(order_is_structurally_possible(order) for order in ACTIONS)


def _fresh_market_free_state(seeds: dict[str, int], day: int = 0):
    def farm():
        return {
            "money": 3_000,
            "tiles": [
                [None if x < 5 and y < 5 else "LOCKED" for x in range(10)]
                for y in range(10)
            ],
            "farmer": [4, 4],
            "hands": [],
            "unlocked_quadrants": ["NW"],
            "hires_today": 0,
        }

    mine = farm()
    private = {"seeds": dict(seeds), "shed": {}, "inventories": [{}]}
    obs = {
        "player": 0,
        "step": day * 24,
        "day": day,
        "hour": 0,
        "farms": [mine, farm()],
        "private": private,
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }
    return obs, mine, private


@pytest.mark.parametrize(
    ("money", "hires_today", "affordable"),
    [
        (3_000, 0, 13),  # the whole ladder, on a full stake
        (608, 0, 12),  # one coin short of HIRE 13, which costs 609
        (609, 0, 13),
        (7, 0, 4),  # 1 + 1 + 2 + 3
        (0, 0, 0),  # broke: not even the first hand
        (7, 3, 1),  # mid-day the ladder starts at _fib(3) = 3; a second is 5 more
    ],
)
def test_hire_rungs_are_masked_on_the_cost_of_every_hand(money, hires_today, affordable):
    """`HIRE n` is feasible only when all n hands are affordable.

    Stricter than `_do_hire`, which commits what it can and stops. Deliberately:
    `HIRE 13` against 7 coins commits four hands, and `HIRE 4` is its own index
    on the same ladder, so nothing the simulator would do becomes unreachable.
    The version that opened all thirteen rungs on the first hand's cost let the
    policy spend a 3,000 stake in four turns of a 720-turn episode.
    """

    obs, farm, private = _fresh_market_free_state({})
    farm["money"] = money
    farm["hires_today"] = hires_today
    mask = feasible_actions(obs, farm, private, {})

    rows = [order_index("HIRE", None, hands) for hands in HIRE_QUANTITIES]
    assert [bool(mask[row]) for row in rows] == [
        hands <= affordable for hands in HIRE_QUANTITIES
    ]


def test_a_hire_ladder_the_cost_table_cannot_reach_is_closed_not_opened():
    """`hires_today` has no ceiling in the simulator, only affordability.

    Past the prefix table every hand costs more than the game can produce, so
    falling off the end must mask the rows rather than wave them through.
    """

    obs, farm, private = _fresh_market_free_state({})
    farm["money"] = 10**9
    farm["hires_today"] = 10_000
    mask = feasible_actions(obs, farm, private, {})

    assert not any(mask[order_index("HIRE", None, hands)] for hands in HIRE_QUANTITIES)


def test_held_tomato_seeds_replace_only_vacant_crop_roles():
    obs, farm, private = _fresh_market_free_state({"TOMATO": 3})
    base = donor._role_plan(obs, farm)
    animal_roles = {position: role for position, role in base.items() if role[0] == "ANIMAL"}

    roles = market_influenced_roles(obs, farm, private, base)
    tomato_positions = [position for position, role in roles.items() if role == ("CROP", "TOMATO")]
    assert len(tomato_positions) == 3
    assert {position: roles[position] for position in animal_roles} == animal_roles
    assert all(base[position][0] == "CROP" for position in tomato_positions)

    jobs = donor._field_jobs(obs, farm, private, roles, liquidation=False)
    assert sum(job.get("action") == ["PLANT", "TOMATO"] for job in jobs) == 3


def test_a_crop_with_no_planned_tile_cannot_take_a_melon_one():
    """The cannibalisation that cost the agent most of its income.

    `CROP_MIX` plans no TOMATO or STRAWBERRY tile anywhere and MELON is last in
    `donor.CROPS`, so the old `unassigned[0]` fallback handed every tomato seed a
    tile the donor had planned for melon -- the only crop worth 142 per tile-day
    against 24-40 for the rest. Holding melon seed has to protect those tiles.
    """

    obs, farm, private = _fresh_market_free_state({"TOMATO": 6, "MELON": 2})
    base = donor._role_plan(obs, farm)
    planned_melon = {position for position, role in base.items() if role == ("CROP", "MELON")}
    assert len(planned_melon) > 2, "the fixture must plan more melon than we hold seed for"

    roles = market_influenced_roles(obs, farm, private, base)

    # Melon keeps every tile it was planned for, not just the two it can fill.
    assert {position for position in planned_melon if roles[position] == ("CROP", "MELON")} == (
        planned_melon
    )
    tomato = {position for position, role in roles.items() if role == ("CROP", "TOMATO")}
    assert not (tomato & planned_melon), "tomato took a melon tile"


def test_the_seed_deficit_the_network_is_shown_counts_the_whole_plan():
    """Why the cannibalisation was invisible as well as expensive.

    `market_state` feeds `donor._seed_needs` per crop straight into the state
    (`market_state.py:590-591`), but it is handed the roles map *after* the
    overlay has rewritten it. While tomato could take melon tiles, those tiles
    stopped being melon, so the melon deficit the network saw was 2 when the
    plan actually wanted 8 more -- it was never told what the purchase cost it.
    """

    obs, farm, private = _fresh_market_free_state({"TOMATO": 6, "MELON": 2})
    base = donor._role_plan(obs, farm)
    planned_melon = sum(1 for role in base.values() if role == ("CROP", "MELON"))

    roles = market_influenced_roles(obs, farm, private, base)
    needs = donor._seed_needs(obs, farm, private, roles)

    assert needs.get("MELON", 0) == planned_melon - 2, "melon deficit must survive the overlay"
    # And a crop the plan never asked for does not invent a need for itself.
    assert needs.get("TOMATO", 0) == 0


def test_a_spare_tile_is_still_usable_when_its_crop_has_no_seed():
    """The protection is seed-conditional, not a blanket freeze on the plan.

    A planned tile the agent cannot plant is worth nothing standing empty, so a
    crop with no tile of its own may still use it. Only tiles whose planned crop
    is actually plantable are off limits.
    """

    obs, farm, private = _fresh_market_free_state({"TOMATO": 3})
    base = donor._role_plan(obs, farm)

    roles = market_influenced_roles(obs, farm, private, base)

    # No melon seed is held, so melon tiles are fair game and tomato still lands.
    assert sum(1 for role in roles.values() if role == ("CROP", "TOMATO")) == 3


def test_market_planting_preserves_live_plants_and_crop_deadlines():
    obs, farm, private = _fresh_market_free_state({"TOMATO": 20})
    base = donor._role_plan(obs, farm)
    live_position = next(position for position, role in base.items() if role[0] == "CROP")
    x, y = live_position
    farm["tiles"][y][x] = {
        "kind": "PLANT",
        "crop": "MELON",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 0,
    }
    influenced = market_influenced_roles(obs, farm, private, base)
    assert influenced[live_position] == base[live_position]

    late_obs, late_farm, late_private = _fresh_market_free_state({"TOMATO": 20}, day=20)
    late_base = donor._role_plan(late_obs, late_farm)
    assert market_influenced_roles(late_obs, late_farm, late_private, late_base) == late_base


# --------------------------------------------------------------------------
# Structural diagnostics
# --------------------------------------------------------------------------


def test_order_diagnostics_count_decisions_and_orders():
    """Opponents and stored replays are not held to the one-order rule.

    This agent emits at most one order a decision, but `diagnose_replay` also
    reads opponent seats and donor demonstrations, which place several. All of
    them are counted as wire orders, while `trading_decisions` counts the turns
    that emitted anything at all -- and that is the one the gate reads.
    """

    diagnostics = diagnose_order_lists(
        [
            [],
            [["SELL", "WHEAT", 4], ["SELL", "WHEAT", 4]],
            [
                ["SELL", "CARROT", 2],
                ["BUY_PRODUCT", "CARROT", 3],
                ["BUY_SEED", "CARROT", 4],
            ],
            [["BUY_SEED", "WHEAT", 2], ["BUY_SEED", "WHEAT", 3]],
        ]
    )

    assert diagnostics.decisions == 4
    assert diagnostics.orders == 7
    assert diagnostics.emitting_decisions == 3
    assert diagnostics.as_dict()["wire_orders_per_decision"] == pytest.approx(7 / 4)
    assert diagnostics.as_dict()["emitted_rate"] == pytest.approx(3 / 4)


def test_a_single_order_policy_reports_a_bounded_rate():
    """The rate the gate reads cannot exceed 1.0, and `HIRE n` is the reason why.

    `HIRE n` is one decision whose wire form is n rows, so the wire count runs
    well past `decisions` for a policy that only ever hires. The gate reads
    `trading_rate`, which counts the decision, not the rows -- a floor computed
    off the wire count would wave a hire-spam policy through while it trades on
    one turn in fifty.
    """

    hire = Order("HIRE", None, HIRE_QUANTITIES[-1])
    assert len(hire.to_wire()) > 1

    # Both the sale and the hire moved money, so both count once each.
    diagnostics = diagnose_order_lists(
        [[], [["SELL", "WHEAT", 4]], hire.to_wire()], [0.0, 48.0, -609.0]
    )
    payload = diagnostics.as_dict()

    assert payload["trading_rate"] == pytest.approx(2 / 3)
    assert payload["emitted_rate"] == pytest.approx(2 / 3)
    assert payload["wire_orders_per_decision"] > 1.0


def _diagnostic_replay(seat_zero_orders: list, seat_one_orders: list) -> dict:
    return {
        "steps": [
            [{"observation": {}}, {"observation": {}}],
            [
                {"action": {"market": seat_zero_orders}},
                {"action": {"market": seat_one_orders}},
            ],
        ]
    }


def test_arena_diagnostics_follow_the_learner_across_seat_swaps():
    """The learner is seat 0 forward and seat 1 reverse; only its orders count."""

    forward = SimpleNamespace(
        failed=False,
        replay=_diagnostic_replay([["SELL", "WHEAT", 2]], []),
    )
    reverse = SimpleNamespace(
        failed=False,
        # Seat 0 here is the opponent and must not be counted.
        replay=_diagnostic_replay(
            [["SELL", "CARROT", 2], ["BUY_PRODUCT", "CARROT", 2]],
            [["BUY_SEED", "WHEAT", 2]],
        ),
    )

    diagnostics = _arena_order_diagnostics({7: (forward, reverse)})
    assert diagnostics.decisions == 2
    assert diagnostics.orders == 2

    failed = SimpleNamespace(failed=True, replay=forward.replay)
    assert _arena_order_diagnostics({7: (forward, failed)}).decisions == 0


def _viability_result(
    *,
    mean_money: float = 900.0,
    median_money: float = 800.0,
    decisions: int = 100,
    orders: int = 20,
    trading_decisions: int | None = None,
) -> dict:
    """An arena result whose diagnostics are the real serialized payload.

    Built through `OrderDiagnostics.as_dict` deliberately. A hand-written dict
    here once carried a `trading_decisions` key the dataclass never emits, so
    every test below passed against a payload production does not produce while
    the gate read 0.0 activity for every real arena.
    """

    trading = orders if trading_decisions is None else trading_decisions
    return {
        "pairs": 50,
        "errors": {},
        "a_money_mean": mean_money,
        "a_money_median": median_money,
        "order_diagnostics": OrderDiagnostics(
            decisions=decisions,
            orders=orders,
            # One order a decision unless a case says otherwise, which is what
            # this agent's own policy produces for everything except HIRE.
            emitting_decisions=trading,
            effective_decisions=trading,
        ).as_dict(),
    }


def _benchmark_result(mean_money: float) -> dict:
    """A challenger benchmark series, carrying only what the ratchet reads."""

    return {
        "opponent": "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py",
        "pairs": 100,
        "errors": {},
        "a_money_mean": mean_money,
    }


def test_viability_gate_accepts_a_solvent_trading_policy():
    viability = _arena_viability(
        _viability_result(),
        {"a_money_mean": 1_000.0, "a_money_median": 1_000.0},
        {},
    )

    assert viability["passed"]
    assert all(viability["checks"].values())


def test_viability_gate_rejects_bankruptcy():
    """Over-trading can now only surface as lost money, so that is what is checked.

    A decision emits at most one order, so `trading_rate` is bounded by 1.0 and
    no upper limit on it could ever bind. The candidate below trades on almost
    every turn and ends with a fraction of the incumbent's money, which is the
    bankruptcy collapse.
    """

    viability = _arena_viability(
        _viability_result(mean_money=389.0, median_money=60.0, orders=97),
        {"a_money_mean": 775.0, "a_money_median": 100.0},
        {},
    )

    assert not viability["passed"]
    assert not viability["checks"]["mean_money"]
    assert not viability["checks"]["median_money"]


def test_the_ratchet_refuses_the_decay_the_head_to_head_called_an_improvement():
    """`falloff_v1`'s real series, as one test.

    Its last update won the head-to-head 0.94 with a significant +2,347 margin
    against the checkpoint before it, and scored 107 coins against the fixed
    benchmark where its own predecessor scored 1,146. Both statements are true
    because they measure against different things, and only one of them is a
    reference that stays still.
    """

    from agents.dqn_market_free.training.updater import _benchmark_ratchet

    config = {}
    decay = _benchmark_ratchet(
        _benchmark_result(107.49), 1_145.595, "2f013342ef149272d875893b", config, grace=False
    )
    assert not decay["passed"]
    assert not decay["checks"]["min_money"]
    assert not decay["checks"]["no_regression"]
    assert decay["floor_name"] == "benchmark_release_min_money"

    # The same policy would have been released under the head-to-head rule.
    from agents.dqn_market_scorer.training.updater import policy_improvement

    winning = {"errors": {}, "margin": 2_346.61, "margin_ci": [2_038.38, 2_654.84]}
    assert policy_improvement(winning, "significant")


def test_the_ratchet_releases_a_cold_start_against_the_lower_floor():
    """With nothing released there is no reference, so only the floor binds.

    `None` is not "the ratchet passes" -- the absolute floor still decides, at
    the cold-start value, which is what makes a first release reachable without
    making it cheap.
    """

    from agents.dqn_market_free.training.updater import _benchmark_ratchet

    cold = _benchmark_ratchet(_benchmark_result(2_978.68), None, None, {}, grace=False)
    assert cold["passed"]
    assert "no_regression" not in cold["checks"]
    assert cold["floor_name"] == "benchmark_cold_start_min_money"

    # 894 is `coldstart_v1@140`, the update that would have given that run its
    # first incumbent. Above the cold-start floor, below the release floor.
    foothold = _benchmark_ratchet(_benchmark_result(894.0), None, None, {}, grace=False)
    assert foothold["passed"]
    assert not _benchmark_ratchet(_benchmark_result(894.0), 1_000.0, "x", {}, grace=False)["passed"]


def test_the_grace_drops_the_floor_but_never_the_ratchet():
    """The bound on how long the gate can block, and which term pays for it.

    `market-free-rare-v20` is why this is the floor rather than the ratchet. All
    57 of its rejections failed `min_money` against a bar of 1,000 while measuring
    between 0.93 and 427, so the term grace used to stand down was the one the run
    was already clearing. Dropping the absolute bar and keeping the relative one
    leaves a gate that a genuinely improving policy can pass and a decaying one
    still cannot.
    """

    from agents.dqn_market_free.training.updater import _benchmark_ratchet

    # v20's own numbers: candidate 215.5 against a recorded reference of 145.815.
    # Under the release floor it is refused; under grace the ratchet alone decides
    # and 215.5 clears `0.9 * 145.815`.
    stalled = _benchmark_result(215.5)
    blocked = _benchmark_ratchet(stalled, 145.815, "9071d2ad", {}, grace=False)
    assert not blocked["passed"]
    assert not blocked["checks"]["min_money"]
    assert blocked["checks"]["no_regression"]

    freed = _benchmark_ratchet(
        stalled, 145.815, "9071d2ad", {"benchmark_grace_min_money": 0.0}, grace=True
    )
    assert freed["passed"]
    assert freed["floor"] == 0.0
    assert freed["floor_name"] == "benchmark_grace_min_money"

    # The ratchet is what grace does not drop, so the decay it exists to catch is
    # still refused with the floor at zero.
    decaying = _benchmark_ratchet(
        _benchmark_result(20.4), 145.815, "9071d2ad", {"benchmark_grace_min_money": 0.0}, grace=True
    )
    assert not decaying["passed"]
    assert not decaying["checks"]["no_regression"]


def test_the_ratchet_reference_is_a_median_not_a_lucky_draw():
    """`falloff_v1` failed `no_regression` 57 times without having regressed.

    Its 61 benchmark measurements ran 0.93 to 427 around a median of 36.7 while
    the snapback held the policy on one checkpoint, so the series is noise, not a
    trajectory -- `a_money_median` reads 2 against a mean of 32.6, so the policy is
    bankrupt on most seeds and the mean rides on a few survivors. `9071d2ad`
    stored 145.8, its own p75, and every later draw was ratcheted against that.

    A window median cannot be set by one lucky draw, and one lucky draw cannot
    clear it either.
    """

    from agents.dqn_market_free.training.updater import _benchmark_ratchet

    config = {"benchmark_grace_min_money": 0.0}
    # v20's ten most recent measurements, verbatim.
    window = [21.5, 15.36, 112.9, 19.59, 4.12, 1.305, 94.18, 55.47, 5.56, 46.51]

    single = _benchmark_ratchet(_benchmark_result(32.59), 145.815, "9071d2ad", config, grace=True)
    assert single["reference_kind"] == "released_measurement"
    assert not single["passed"]

    windowed = _benchmark_ratchet(
        _benchmark_result(32.59), 145.815, "9071d2ad", config, grace=True,
        reference_samples=window,
    )
    assert windowed["reference_kind"] == "window_median"
    assert windowed["reference_samples"] == len(window) + 1
    # The stored draw stays readable beside the statistic that decided.
    assert windowed["released_money"] == 145.815
    assert windowed["reference_money"] == 21.5
    assert windowed["passed"]

    # A single outlier does not move a median, so it cannot re-aim the bar.
    spiked = _benchmark_ratchet(
        _benchmark_result(32.59), 145.815, "9071d2ad", config, grace=True,
        reference_samples=[*window, 10_000.0],
    )
    assert spiked["passed"]

    # Nor does the window make the ratchet toothless: a genuine collapse still
    # fails against the same median.
    collapsed = _benchmark_ratchet(
        _benchmark_result(1.2), 145.815, "9071d2ad", config, grace=True,
        reference_samples=window,
    )
    assert not collapsed["checks"]["no_regression"]


def test_the_grace_floor_defaults_to_the_cold_start_bar():
    """An unset `benchmark_grace_min_money` must not change any existing config."""

    from agents.dqn_market_free.training.updater import _benchmark_ratchet

    graced = _benchmark_ratchet(_benchmark_result(800.0), 500.0, "x", {}, grace=True)
    assert graced["floor"] == 750.0
    assert graced["floor_name"] == "benchmark_cold_start_min_money"
    assert graced["passed"]

    # 700 is under the cold-start bar, so grace alone does not release it.
    assert not _benchmark_ratchet(_benchmark_result(700.0), 500.0, "x", {}, grace=True)["passed"]


def test_an_unmeasured_benchmark_is_not_a_pass():
    """Every way of not knowing has to fail closed."""

    from agents.dqn_market_free.training.updater import _benchmark_ratchet

    assert not _benchmark_ratchet(None, None, None, {}, grace=True)["passed"]
    errored = {**_benchmark_result(5_000.0), "errors": {"3": "crashed"}}
    assert not _benchmark_ratchet(errored, None, None, {}, grace=True)["passed"]


def test_the_ratchet_reference_is_matched_on_the_opponent_not_the_slot():
    """Reconfiguring benchmarks must reset the ratchet, not silently re-aim it.

    Comparing this update's score against a *different* opponent's recorded
    number is the one way this gate could be wrong without any of it looking
    wrong, so the lookup matches the spec string and returns `None` -- no ratchet
    term -- when it does not find it.
    """

    import json

    from agents.dqn_market_free.training.updater import _released_benchmark_money

    donor = "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"
    gate = {
        "challenger_benchmarks": [
            {"opponent": donor, "a_money_mean": 1_145.595, "errors": {}},
        ]
    }

    class _Reader:
        def __init__(self, payload):
            self.payload = payload

        def fetchone(self, sql, parameters):
            assert "model_training_checkpoints" in sql
            assert parameters == ["ckpt"]
            return (self.payload,)

    class _Active:
        checkpoint_id = "ckpt"

    money, source = _released_benchmark_money(_Reader(gate), _Active(), donor)
    assert (money, source) == (1_145.595, "ckpt")
    # Serialized the way psycopg hands back a json column that is not decoded.
    money, _ = _released_benchmark_money(_Reader(json.dumps(gate)), _Active(), donor)
    assert money == 1_145.595
    # A different opponent, an errored series, and a cold start all mean "no
    # reference", never "the reference passes".
    assert _released_benchmark_money(_Reader(gate), _Active(), "agents/other.py") == (None, None)
    assert _released_benchmark_money(_Reader(gate), None, donor) == (None, None)
    errored = {"challenger_benchmarks": [{"opponent": donor, "a_money_mean": 9e9, "errors": {"1": "x"}}]}
    assert _released_benchmark_money(_Reader(errored), _Active(), donor) == (None, None)


def test_an_unknown_promotion_rule_still_raises():
    """A typo must not fall through to a permissive default.

    `always` is still in the set and still ships every checkpoint, so a
    misspelling resolving to it would silently release everything.
    """

    from agents.dqn_market_free.training.updater import _PROMOTION_RULES

    assert _PROMOTION_RULES == {"significant", "point_estimate", "always", "benchmark_ratchet"}
    assert "alway" not in _PROMOTION_RULES
    assert "benchmark-ratchet" not in _PROMOTION_RULES
    assert "benchmark_rachet" not in _PROMOTION_RULES
    assert "" not in _PROMOTION_RULES


def test_the_ratchet_rule_does_not_fall_through_to_the_head_to_head():
    """`policy_improvement` treats any unknown rule as `significant`.

    Passing `benchmark_ratchet` straight in would therefore mean the drifting
    head-to-head test under a name that says otherwise. Asserted at the source
    because the mapping lives inside a 900-line method with a live database and
    an arena behind it.
    """

    import inspect

    from agents.dqn_market_free.training import updater as updater_module

    source = inspect.getsource(updater_module.FreeMarketUpdater.update)
    assert 'head_reference_rule = (\n            "point_estimate" if promotion_rule ==' in source
    assert "head_to_head_ok = policy_improvement(head_to_head, head_reference_rule)" in source
    # The ratchet decides, and hard viability still has to pass with it.
    assert 'promotion_rule == "benchmark_ratchet"' in source
    assert 'and benchmark_ratchet["passed"]' in source
    # The arena still runs whatever the rule says.
    assert "hard_viability_passed = head_viability" in source


def test_the_export_and_harness_guards_are_not_part_of_the_gate():
    """Two guards must keep raising under `always`, and both fail silently.

    `_reject_inert_network` catches an all-zero exported trunk -- a policy that
    can only ever emit NOOP, and which under `always` would now be *released* as
    a healthy checkpoint rather than merely rejected. It exists because a real
    `to_numpy` bug produced exactly that.

    `_reject_degenerate` catches a zero-width interval, which means a harness
    fault -- a dropped `?weights=`, a missing checkpoint -- not a weak candidate.
    Neither is a quality judgement, so neither is what was removed.
    """

    import numpy as np

    from agents.dqn_market_free.policy import OrderQNetwork
    from agents.dqn_market_free.training.updater import _reject_inert_network
    from agents.dqn_market_scorer.training.updater import _reject_degenerate

    inert = OrderQNetwork.zeros()
    with pytest.raises(RuntimeError, match="all-zero hidden weights"):
        _reject_inert_network(inert)

    degenerate = {
        "errors": {},
        "pairs": 100,
        "margin": 0.0,
        "margin_ci": [0.0, 0.0],
        "opponent": "agents/dqn_market_free/main.py",
    }
    with pytest.raises(RuntimeError):
        _reject_degenerate(degenerate, "head-to-head")

    # A real spread is not degenerate, so the guard is not simply always firing.
    healthy = {
        "errors": {},
        "pairs": 100,
        "margin": 900.0,
        "margin_ci": [120.0, 1_680.0],
        "opponent": "agents/dqn_market_free/main.py",
    }
    _reject_degenerate(healthy, "head-to-head")
    assert np.isfinite(healthy["margin"])


def test_the_viability_gate_records_which_reference_the_floors_used():
    """A grace floor and an ordinary one are the same number once written.

    19 of the 35 updates that motivated `viability_grace_after_rejections` were
    already in *ratchet* grace when they were rejected, so a payload that cannot
    say which hatch moved cannot be read to tell the two apart.
    """

    result = _viability_result()
    reference = {"a_money_mean": 1_000.0, "a_money_median": 1_000.0}

    ordinary = _arena_viability(result, reference, {})
    assert ordinary["reference_kind"] == "incumbent_seat"
    assert ordinary["grace"] is False

    graced = _arena_viability(
        result, reference, {}, reference_kind="cold_start_grace", grace=True
    )
    assert graced["reference_kind"] == "cold_start_grace"
    assert graced["grace"] is True
    # Recorded, not used: the same reference must produce the same verdict either
    # way, or the label is deciding something.
    assert graced["checks"] == ordinary["checks"]
    assert graced["median_money_floor"] == ordinary["median_money_floor"]


def test_every_viability_default_can_be_recorded_without_a_config():
    """`gate_json.hard_viability.limits` maps `_limit` over the whole defaults dict.

    Two of those defaults are `None` -- they mean "fall back to another key", not a
    number -- so `float(None)` there takes the entire update down. It reached
    production: `landrate_09` and `landrate_08` both died on
    `TypeError: float() argument must be a string or a real number, not 'NoneType'`
    at cutoff 10. `benchmark_grace_min_money` had been a `None` default for far
    longer and never hit it, only because the live `FREE_CONFIG` happens to set it.

    So the empty config is the case worth pinning: it is the one that resolves every
    default rather than overriding it.
    """

    import inspect

    from agents.dqn_market_free.training import updater as updater_module

    nullable = [
        name for name, value in updater_module.VIABILITY_DEFAULTS.items() if value is None
    ]
    assert nullable, "the guard below is only meaningful while a None default exists"

    for name, value in updater_module.VIABILITY_DEFAULTS.items():
        if value is None:
            with pytest.raises(TypeError):
                updater_module._limit({}, name)
        else:
            assert isinstance(updater_module._limit({}, name), float)

    source = inspect.getsource(updater_module.FreeMarketUpdater.update)
    limits = source.split('"limits": {', 1)[1].split("},", 1)[0]
    assert "is not None" in limits


def test_the_cold_start_money_floors_have_their_own_hatch():
    """The ratchet's grace does nothing for a slot hard viability is holding.

    Measured across every session in the lab database: 35 updates over 20 distinct
    never-released slots were rejected with `benchmark_ratchet` passing and
    `median_money` the only failing check. `rarerate_03` is the shape -- ratchet
    passing from seq 30, `a_money_median` 348 -> 910 -> 940 against a fixed 1,125,
    `mean_money` clearing 1,200 the whole way, five updates and no release.

    Asserted at the source for the reason
    `test_the_ratchet_rule_does_not_fall_through_to_the_head_to_head` gives: the
    branch lives inside a 900-line method with a live database and an arena behind
    it. The arithmetic it turns on is checked against the real defaults below.
    """

    import inspect

    from agents.dqn_market_free.training import updater as updater_module

    source = inspect.getsource(updater_module.FreeMarketUpdater.update)
    # Only on a cold start. From the first release the reference is a real
    # opponent's seat in the same paired series and never stands down.
    assert 'head_reference_kind = "incumbent_seat"' in source
    assert "viability_grace_limit > 0 and rejection_streak >= viability_grace_limit" in source
    assert 'head_reference_kind = "cold_start_grace"' in source
    assert 'head_reference_kind = "cold_start_fixed"' in source
    assert "reference_kind=head_reference_kind" in source
    assert "grace=head_viability_grace," in source
    # The grace branch is inside the `active is None` arm, so an incumbent seat
    # cannot reach it however long the streak runs.
    cold_start = source.split("head_viability_grace = False", 1)[1]
    assert cold_start.index("if active is not None:") < cold_start.index("viability_grace_limit")

    defaults = updater_module.VIABILITY_DEFAULTS
    # Off unless asked for, so every config predating the key reads as it did.
    assert defaults["viability_grace_after_rejections"] == 0
    assert defaults["viability_grace_cold_start_reference_money"] is None
    # Unset falls back to the ratchet's own first-release foothold rather than to a
    # second new number, and that foothold has to be the lower of the two or the
    # hatch would raise the bar it exists to lower.
    assert (
        defaults["benchmark_cold_start_min_money"]
        < defaults["viability_cold_start_reference_money"]
    )


def test_the_cold_start_hatch_admits_a_converging_slot_and_not_a_dead_one():
    """`rarerate_03`'s own series, run through both references.

    The point of the hatch is that 940 is not the same as 348. A floor that admits
    both is `promotion_rule = "always"` for any slot that has rejected enough times,
    which is what keeping both ratios binding is for.
    """

    graced_reference = updater_module.VIABILITY_DEFAULTS["benchmark_cold_start_min_money"]
    fixed_reference = updater_module.VIABILITY_DEFAULTS["viability_cold_start_reference_money"]

    def verdict(mean_money: float, median_money: float, reference: float) -> dict:
        return _arena_viability(
            _viability_result(mean_money=mean_money, median_money=median_money),
            {"a_money_mean": reference, "a_money_median": reference},
            {},
        )

    # Seq 50: mean 1,336 median 940. Held by the fixed reference, admitted by grace.
    assert not verdict(1_336.0, 940.0, fixed_reference)["checks"]["median_money"]
    assert verdict(1_336.0, 940.0, graced_reference)["passed"]

    # Seq 30: mean 1,633 median 348. Its mean clears the fixed floor, which is why
    # the median term is the one doing the work -- and grace still refuses it.
    assert verdict(1_633.0, 348.0, fixed_reference)["checks"]["mean_money"]
    assert not verdict(1_633.0, 348.0, graced_reference)["checks"]["median_money"]

    # Bankruptcy is still fatal under grace.
    assert not verdict(60.0, 12.0, graced_reference)["passed"]


def test_viability_gate_rejects_a_noop_collapse():
    """The other attractor. A policy that trades nothing preserves its money.

    The floor is unconditional now, cold start included. It used to bind only
    once something had been released, which left it inert exactly where a silent
    policy can be promoted unopposed: with no incumbent the head-to-head runs
    against `seeded_random`, which reliably bankrupts, so a pure-NOOP candidate
    keeps its 3,000 stake and wins 3,000-to-0 on money alone.
    """

    silent = _viability_result(orders=0)
    reference = {"a_money_mean": 900.0, "a_money_median": 800.0}

    guarded = _arena_viability(silent, reference, {}, require_activity=True)
    assert not guarded["passed"]
    assert not guarded["checks"]["min_orders_per_decision"]

    # Off only when asked for in so many words.
    disabled = _arena_viability(
        silent, reference, {"viability_min_orders_per_decision": 0.0}, require_activity=True
    )
    assert disabled["passed"]
    assert not disabled["min_orders_per_decision_required"]


def test_the_activity_floor_binds_without_a_release():
    """A silent candidate must not become the incumbent, released or not.

    Two other guards in this family were written to fire "once released" and were
    silently inert on a run that had never released -- the lineage snapback, and
    the streak that feeds it. This is the third, and the one that decides what
    becomes the incumbent in the first place, so it is asserted at the call site
    rather than only on the helper.

    The arena carrying it is the *benchmark*, not the head-to-head. On a cold
    start the head-to-head opponent is the quiet baseline, which offers no
    counterparty, so the floor measures nothing there -- see
    `test_the_cold_start_head_to_head_does_not_require_activity`. A pure-NOOP
    candidate reads 0.0 against the benchmark as well, so this call site is
    where the v6 shape (`trading_rate: 0.0` beside `hard_viability.passed:
    true`) is still caught.
    """

    import inspect

    from agents.dqn_market_free.training import updater as updater_module

    source = inspect.getsource(updater_module.FreeMarketUpdater.update)
    assert "_order_viability(result, config, require_activity=True)" in source, (
        "the activity floor must bind on the benchmark arena unconditionally; a "
        "candidate that never trades must never become the incumbent"
    )


def test_the_cold_start_head_to_head_does_not_require_activity():
    """The floor cannot be read in a match that offers no counterparty.

    `market-free-population-v25` at cutoff 143: learner_01, learner_02 and
    learner_09 traded at 0.633, 0.332 and 0.398 against the benchmark and emitted
    0, 210 and 0 orders against the quiet baseline. Five of ten slots had never
    released after 143 batches and `population_members` held one row for the
    whole run, so the mature group the population exists to build stayed empty.

    Asserted at the call site because the condition is the whole point: it must
    fall back to requiring activity when there is no benchmark arena to carry it.
    """

    import inspect

    from agents.dqn_market_free.training import updater as updater_module

    source = inspect.getsource(updater_module.FreeMarketUpdater.update)
    assert "require_head_activity = active is not None or not challenger_benchmarks" in source
    assert "require_activity=require_head_activity" in source


def test_degeneracy_reads_activity_rather_than_the_margin():
    """A zero margin is not a NOOP collapse, and v25 paid for the conflation.

    `learner_05` at cutoff 143 placed 16,686 orders over 35,950 decisions -- a
    trading rate of 0.464 -- into a head-to-head that still scored an identically
    zero margin, both seats on exactly 3,000.0, because round-trip trades are
    cash-neutral against an unchanged market. It passed the ratchet at 987
    benchmark money and passed hard viability, and the degeneracy check alone
    rejected it.
    """

    traded = _viability_result(decisions=100, orders=46, trading_decisions=46)
    assert not updater_module._collapsed_against_baseline(traded, {})

    # The collapse the check exists for is still a collapse.
    silent = _viability_result(orders=0)
    assert updater_module._collapsed_against_baseline(silent, {})

    # A hire on one turn in a hundred is not activity, however many wire rows.
    hires_once = _viability_result(
        decisions=100, orders=HIRE_QUANTITIES[-1], trading_decisions=1
    )
    assert updater_module._collapsed_against_baseline(hires_once, {})

    # Off with the rest of the activity floors, deliberately.
    assert not updater_module._collapsed_against_baseline(
        silent, {"viability_min_orders_per_decision": 0.0}
    )


def test_activity_floor_leaves_ordinary_trading_alone():
    viability = _arena_viability(
        _viability_result(),
        {"a_money_mean": 1_000.0, "a_money_median": 1_000.0},
        {},
        require_activity=True,
    )

    assert viability["passed"]
    assert viability["checks"]["min_orders_per_decision"]


def test_activity_floor_is_not_fooled_by_hire_wire_counts():
    """A hire on one turn in a hundred is not activity, however many rows it is.

    `HIRE 13` is one decision that expands to thirteen wire orders, so a floor
    read off the wire count sees 0.13 orders per decision -- comfortably over the
    0.02 minimum -- for a policy that acted exactly once in a hundred turns.
    """

    hires_once = _viability_result(
        decisions=100, orders=HIRE_QUANTITIES[-1], trading_decisions=1
    )
    reference = {"a_money_mean": 900.0, "a_money_median": 800.0}

    assert hires_once["order_diagnostics"]["orders"] / 100 > 0.02
    guarded = _arena_viability(hires_once, reference, {}, require_activity=True)
    assert not guarded["checks"]["min_orders_per_decision"]
    assert guarded["trading_rate"] == pytest.approx(0.01)


def test_activity_floor_reads_the_serialized_diagnostics_payload():
    """The floor must come off the payload `OrderDiagnostics` actually writes.

    Regression for a key mismatch: the gate recomputed the rate from a
    `trading_decisions` count that `as_dict` does not emit, so every arena in
    `market-free-noisy-v6` reported `trading_rate` 0.0 and the floor rejected an
    11k-money candidate that traded on 12% of its decisions.
    """

    payload = OrderDiagnostics(
        decisions=100, orders=50, emitting_decisions=50, effective_decisions=50
    ).as_dict()
    assert "trading_decisions" not in payload

    order = updater_module._order_viability(
        {"pairs": 100, "errors": {}, "order_diagnostics": payload},
        {},
        require_activity=True,
    )

    assert order["trading_rate"] == pytest.approx(0.5)
    assert order["checks"]["min_orders_per_decision"]
    assert order["passed"]


def _opponents_played(opponents: list[str], batch_index: int, episodes: int) -> list[str]:
    """Reproduce `collect_batch`'s per-pair draw for one batch."""

    rotated = rotate_opponents(opponents, batch_index * (episodes // 2))
    return [rotated[pair % len(rotated)] for pair in range(episodes // 2)]


def test_opponent_rotation_reaches_the_tail_of_a_long_list():
    """A list longer than one batch of pairs must not strand its tail.

    `collect_batch` draws `opponents[pair_index % len(opponents)]` with
    `pair_index` restarting at zero every batch, so without the rotation the
    head is replayed forever -- and self-play slots are appended at the end.
    """

    opponents = [f"opponent-{index}" for index in range(20)]
    episodes = 16

    played: list[str] = []
    for batch_index in range(3):
        played += _opponents_played(opponents, batch_index, episodes)

    assert len(played) == 24
    assert set(played) == set(opponents)


def test_self_play_pool_addresses_the_free_market_entry_point(tmp_path: Path):
    """The pool must not hand free-market weights to the scorer's agent.

    Both loaders validate their own shape, so a spec naming the wrong agent fails
    every episode it is drawn for.
    """

    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"weights")

    class Reader:
        def fetchall(self, sql, parameters):
            assert "model_training_checkpoints" in sql
            return [(str(checkpoint),)]

    context = SimpleNamespace(model_id="free.model")
    pool = _self_play_pool(context, Reader(), 5)

    # Percent-encoded, not interpolated: artifact roots are configurable, and a
    # path containing `&` would otherwise truncate and silently resolve to the
    # packaged weights -- which makes every mirror match a policy against itself.
    assert pool == [f"{SELF_SPEC}?{urlencode({'weights': str(checkpoint)})}"]
    assert all(spec.startswith("agents/dqn_market_free/") for spec in pool)


def test_self_play_pool_follows_the_lineage_not_the_release():
    """A run that has promoted nothing must still be able to face itself.

    `market-free-simple-v2` ran 33 updates and promoted none of them, so the
    scorer's promotion-filtered helper would have returned an empty pool on every
    batch and the mirror would never have played. Release is gated; the behaviour
    distribution is not -- the same rule the rollout policy already follows.
    """

    seen: list[str] = []

    class Reader:
        def fetchall(self, sql, parameters):
            seen.append(sql)
            return []

    _self_play_pool(SimpleNamespace(model_id="free.model"), Reader(), 5)

    assert seen and "promoted" not in seen[0]


def test_a_held_reset_keeps_the_rejection_streak_growing(tmp_path: Path):
    """The deadlock `market-free-rare-v20` sat in for 55 updates.

    The snapback holds until a promotion, and every held update writes
    `lineage_reset` -- so the ordinary streak query, which stops at the newest
    reset, stopped at the held update itself and returned 0 forever. Everything
    that scales with the streak went with it: `promotion_ratchet_grace_after_
    rejections = 4` could not fire against a streak that never left zero, and the
    run resumed the same checkpoint and trained the same 2,000 steps 55 times
    while `gradient_steps` stayed on 24,100.

    Under a hold the lineage is back on the release, so those rejections are one
    trajectory's again and the streak counts from the last promotion.
    """

    from agents.dqn_market_scorer.training.updater import lineage_checkpoint

    released_trainer = tmp_path / "released.pt"
    released_trainer.write_bytes(b"trainer")
    newer_trainer = tmp_path / "newer.pt"
    newer_trainer.write_bytes(b"trainer")

    asked: list[str] = []

    class Reader:
        def fetchone(self, sql, parameters):
            asked.append(sql)
            if "model_training_checkpoints" in sql:
                return ("newer", str(newer_trainer), str(tmp_path / "newer.npz"))
            if "status = 'REJECTED'" in sql:
                # The held query ignores resets, so it sees all 57; the ordinary
                # one stops at the most recent held update and sees none.
                return (57,) if "lineage_reset" not in sql else (0,)
            # `reset_since_promotion`: a reset is recorded with nothing promoted
            # after it.
            return (1,)

    context = SimpleNamespace(
        model_id="free.model",
        config={"resume_reset_after_rejections": 2},
        active_checkpoint=SimpleNamespace(
            checkpoint_id="9071d2ad",
            trainer_artifact_path=str(released_trainer),
            inference_artifact_path=str(tmp_path / "released.npz"),
        ),
    )

    resumed = lineage_checkpoint(context, Reader())

    assert resumed.reset
    assert resumed.released
    assert resumed.checkpoint_id == "9071d2ad"
    assert resumed.rejection_streak == 57
    # The streak that reaches the gate is the held one, so the query that produced
    # it must not have been cut off at a `lineage_reset`.
    counted = [sql for sql in asked if "status = 'REJECTED'" in sql]
    assert counted and "lineage_reset" not in counted[-1]


def test_the_streak_still_stops_at_a_reset_that_is_not_being_held():
    """The original reading, unchanged where it was right.

    A reset that fires drops the diverged trajectory's debt, and counting past it
    would charge the new trajectory for the old one's failures. That only stops
    applying once the reset is being *held*, which is a different state.
    """

    from agents.dqn_market_scorer.training.updater import rejections_since_promotion

    seen: list[str] = []

    class Reader:
        def fetchone(self, sql, parameters):
            seen.append(sql)
            return (3,)

    assert rejections_since_promotion(Reader(), "free.model") == 3
    assert "lineage_reset" in seen[-1]

    assert rejections_since_promotion(Reader(), "free.model", held=True) == 3
    assert "lineage_reset" not in seen[-1]


def test_self_play_pool_skips_checkpoints_whose_weights_are_gone(tmp_path: Path):
    """A row surviving its artifact would fail every episode it is drawn for."""

    present = tmp_path / "present.npz"
    present.write_bytes(b"weights")

    class Reader:
        def fetchall(self, sql, parameters):
            return [(str(tmp_path / "missing.npz"),), (str(present),)]

    pool = _self_play_pool(SimpleNamespace(model_id="free.model"), Reader(), 5)

    assert pool == [f"{SELF_SPEC}?{urlencode({'weights': str(present)})}"]


def test_resume_reapplies_the_configured_learning_rates():
    """`load_state_dict` restores the checkpoint's lr; config is the authority."""

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    clone_optimizer = torch.optim.Adam([parameter], lr=1.0)

    _reset_learning_rates(
        optimizer,
        clone_optimizer,
        {"learning_rate": 3e-4, "behavior_clone_learning_rate": 5e-4},
    )

    assert [group["lr"] for group in optimizer.param_groups] == [3e-4]
    assert [group["lr"] for group in clone_optimizer.param_groups] == [5e-4]

    # The clone rate defaults to the TD rate rather than to the stale value.
    _reset_learning_rates(optimizer, clone_optimizer, {"learning_rate": 2e-4})
    assert [group["lr"] for group in clone_optimizer.param_groups] == [2e-4]

    # The slot schedule overrides the peak, and the clone rate keeps its ratio to
    # it rather than staying at the configured constant -- a decaying slot has to
    # decay both phases or the clone quietly dominates the fine-tuning after it.
    _reset_learning_rates(
        optimizer,
        clone_optimizer,
        {"learning_rate": 2e-4, "behavior_clone_learning_rate": 4e-4},
        5e-5,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [5e-5]
    assert [group["lr"] for group in clone_optimizer.param_groups] == [1e-4]


# --------------------------------------------------------------------------
# Population training
# --------------------------------------------------------------------------


_SLOT_CONFIG = {
    "learning_rate": 1e-4,
    "lr_graduate_cut": 1e-5,
    "lr_warmup_updates": 3,
    "lr_plateau_patience": 2,
    "lr_plateau_min_delta": 0.05,
    "lr_decay_factor": 0.5,
    "lr_min_updates_before_graduation": 4,
}


def _run_slot(scores, config=None):
    """Drive a slot through `scores`, returning one state per update."""

    spec = updater_module._slot(config or _SLOT_CONFIG)
    previous: dict = {}
    history = []
    for score in scores:
        state = updater_module._slot_state(previous, spec)
        state = updater_module._settle_slot(state, score, spec)
        history.append((state, updater_module._graduates(state, spec)))
        previous = state.as_metrics()
    return history


def test_the_slot_warms_up_from_zero_and_then_holds_the_peak():
    """"Learning rate back to zero" is literal, and it is the first update.

    The rounds between a graduation and the update that consumes it are collected
    by the reseeded policy, so the slot's first update is fitting freshly
    perturbed weights to a buffer it has only just started refilling. Training
    that at full rate is the `landgate-v10` failure in miniature.
    """

    rates = [state.learning_rate for state, _ in _run_slot([100.0] * 6)]

    assert rates[0] == 0.0
    assert rates[:3] == pytest.approx([0.0, 1e-4 / 3, 2e-4 / 3])
    assert rates[3:] == pytest.approx([1e-4] * 3), "warmup ends at the peak"


def test_the_rate_decays_on_a_plateau_and_not_on_progress():
    """Progress is measured on benchmark money, never on `mean_loss`.

    Loss is not a progress signal on this family -- it sat at 0.0015 across 500
    batches of a policy going nowhere -- and it is not even the right sign: a
    policy that genuinely improves reaches harder states and its TD residual
    rises. `lr_plateau_min_delta` is relative for the matching reason: benchmark
    money spans three orders of magnitude across a run.
    """

    climbing = [state.learning_rate for state, _ in _run_slot([10.0 * 2**n for n in range(9)])]
    assert climbing[-1] == pytest.approx(1e-4), "a rising score must never decay the rate"

    # Three warmup updates, then the first steady one sets the baseline, then two
    # stalls at `lr_plateau_patience = 2` halve it. The counter restarts on every
    # decay, so the next halving costs another two.
    stalled = _run_slot([10.0] * 11)
    rates = [state.learning_rate for state, _ in stalled]
    assert rates[3:6] == pytest.approx([1e-4] * 3)
    assert rates[6:8] == pytest.approx([5e-5] * 2)
    assert rates[8:10] == pytest.approx([2.5e-5] * 2)
    assert rates[10] == pytest.approx(1.25e-5)

    # A score inside the delta is a stall; one outside it is progress. Read off a
    # state whose counter is at zero, so what moves is the delta and not the
    # patience budget the previous line already spent.
    spec = updater_module._slot(_SLOT_CONFIG)
    rested = replace(stalled[-1][0], stall_updates=0, best_score=10.0)
    assert updater_module._settle_slot(rested, 10.4, spec).stall_updates == 1
    settled = updater_module._settle_slot(rested, 11.0, spec)
    assert settled.stall_updates == 0 and settled.best_score == 11.0


def test_a_slot_graduates_only_after_it_has_actually_trained():
    """Warmup is below the cut too, and at update zero it is exactly zero.

    Without the floor a reseeded slot graduates on its first update, reseeds
    again, and the roster does nothing but churn -- ten learners each burning a
    generation to produce nothing.
    """

    history = _run_slot([10.0] * 14)
    graduated = [index for index, (_, ends) in enumerate(history) if ends]

    assert graduated, "a slot that never improves must eventually end"
    first = graduated[0]
    assert first >= 4, "the lifetime floor holds through warmup and past it"
    assert history[first][0].learning_rate > _SLOT_CONFIG["lr_graduate_cut"], (
        "the cut is read off the rate the *next* update would use, or it is "
        "always noticed one update late"
    )
    assert not any(ends for _, ends in history[:first])

    # A rising score never graduates, however long the slot runs.
    climbing = _run_slot([10.0 * 2**index for index in range(14)])
    assert not any(ends for _, ends in climbing)


def test_the_schedule_is_inert_without_a_graduation_cut():
    """One key turns the whole mechanism on, and every older config is unchanged.

    `lr_warmup_updates` and `lr_decay_factor` both carry non-trivial defaults, so
    a config that never asked for a population would otherwise be silently warmed
    up from zero and decayed afterwards.
    """

    rates = [
        state.learning_rate
        for state, _ in _run_slot([10.0] * 8, {"learning_rate": 1e-4})
    ]

    assert rates == pytest.approx([1e-4] * 8)
    assert not any(ends for _, ends in _run_slot([10.0] * 8, {"learning_rate": 1e-4}))


def test_a_cut_at_or_above_the_peak_is_refused():
    """Otherwise every slot in the roster graduates before it trains, forever."""

    with pytest.raises(ValueError, match="lr_graduate_cut"):
        updater_module._slot({"learning_rate": 1e-4, "lr_graduate_cut": 1e-4})


def test_the_slot_score_is_a_median_of_its_own_measurements():
    """A single arena draw is not a number on this family.

    `falloff_v1` measured its benchmark money 61 times without the policy
    meaningfully moving -- the snapback held it on one checkpoint for 55 of them
    -- and got everything from 0.93 to 427 around a median of 36.7. A plateau
    detector reading single draws finds an "improvement" by luck every few
    updates and the rate never decays, which is the stagnation this exists to end
    reproduced by the thing meant to end it.
    """

    seen: list[list] = []

    class Reader:
        def fetchall(self, sql, parameters):
            seen.append(list(parameters))
            return [
                (json.dumps({"challenger_benchmarks": [{"opponent": "pilkwang", "a_money_mean": value}]}),)
                for value in (0.93, 36.0, 427.0, 12.0)
            ]

    score = updater_module._slot_score(
        Reader(), "free.model", "pilkwang", 40.0, 5, 120, signal="benchmark"
    )

    assert score == pytest.approx(36.0), "the median, not the latest draw"
    # Bounded below by the slot's own reseed: measurements from before it describe
    # a policy this slot no longer is, and an inherited high score would pin the
    # plateau counter of a restarted slot.
    assert seen[-1][1] == 120

    # A window of one is the candidate itself, byte for byte the behaviour before
    # the window existed.
    assert (
        updater_module._slot_score(
            Reader(), "free.model", "pilkwang", 40.0, 1, 0, signal="benchmark"
        )
        == 40.0
    )
    # Nothing measured is not a stall, and `_settle_slot` must see it as such.
    spec = updater_module._slot(_SLOT_CONFIG)
    state = updater_module._slot_state({}, spec)
    assert updater_module._settle_slot(state, None, spec) == state


def test_the_slot_score_reads_the_rollout_mean_by_default():
    """The signal the plateau detector reads, and the measurement that chose it.

    Over `market-free-logmoney-v26` batches 200-360 -- ten slots, 160 consecutive
    update pairs -- the mean absolute update-to-update change was 11.7% for
    `own_mean_final_money` against 48.2% for the windowed benchmark median. The
    arena number is four times noisier than one the update already computes, and
    three irreversible decisions were riding on it.
    """

    seen: list[str] = []

    class Reader:
        def fetchall(self, sql, parameters):
            seen.append(sql)
            return [
                (json.dumps({"own_mean_final_money": value}),)
                for value in (2_000.0, 6_000.0, 3_000.0, 9_000.0)
            ]

    # The default arm reads `metrics_json`, never `gate_json`, and never needs an
    # opponent -- a rollout mean is not a comparison against anybody.
    score = updater_module._slot_score(Reader(), "free.model", None, None, 5, 0, rollout=4_000.0)
    assert score == pytest.approx(4_000.0), "the median of the window including this update"
    assert "metrics_json" in seen[-1] and "gate_json" not in seen[-1]

    # And it is the default, so a config that names nothing gets it.
    assert updater_module._slot(_SLOT_CONFIG).score_signal == "rollout"
    # A slot with no rollout measurement is "no information", not a stall.
    assert updater_module._slot_score(Reader(), "free.model", None, 40.0, 5, 0) is None


def test_an_unrepeatable_high_score_does_not_end_a_slot():
    """`best_score` leaks, because a plain maximum latches onto luck.

    Measured on `market-free-logmoney-v26`: `logmoney_07` pinned `lr_best_score`
    at 4,208.69, a draw it never repeated, spent the rest of its life scoring
    2,992-4,103, and was graduated and reseeded down to 828 -- while its rollout
    money climbed 4,673 -> 5,646 over the same updates.
    """

    config = {**_SLOT_CONFIG, "lr_plateau_patience": 4}
    # Three warmup updates, then the level, then one draw 20% above it -- the size
    # of the one that stranded `logmoney_07` -- then a genuine climb underneath.
    scores = [3_500.0] * 3 + [3_500.0, 4_200.0] + [3_500.0 * 1.08**n for n in range(4)]

    latched = _run_slot(scores, {**config, "lr_best_decay": 0.0})
    assert any(state.decay_steps for state, _ in latched), (
        "without the leak the outlier must strand the slot -- otherwise this test "
        "is not measuring what it claims"
    )
    assert not any(state.decay_steps for state, _ in _run_slot(scores, config)), (
        "with the leak the climb underneath the outlier must read as progress"
    )

    # The leak is floored at the current observation, so it can never decay into
    # declaring an improvement on a fall, and a slot that is genuinely flat still
    # stalls at every decay setting because `best` never leaks below its score.
    spec = updater_module._slot(config)
    previous: dict = {}
    for _ in range(spec.warmup_updates + 1):
        # Past warmup, where scores start counting at all.
        state = updater_module._slot_state(previous, spec)
        previous = state.as_metrics()
    state = updater_module._settle_slot(state, 5_000.0, spec)
    fallen = updater_module._settle_slot(state, 1_000.0, spec)
    assert fallen.best_score == pytest.approx(5_000.0 * (1.0 - spec.best_decay))
    assert fallen.stall_updates == 1

    flat = _run_slot([1_000.0] * 12, config)
    assert all(state.best_score == pytest.approx(1_000.0) for state, _ in flat[3:])
    assert any(state.decay_steps for state, _ in flat), "a real plateau still decays"


def test_the_reseed_perturbation_moves_a_fraction_and_keeps_the_shape():
    """Weight-space dropout, not a layer, so numpy/torch parity is untouched."""

    from agents.dqn_market_free.training import reseed

    source = OrderQNetwork.quiet_start(seed=11)
    # The output layer of a quiet start is all zeros, which makes the count of
    # changed entries exactly the count of redrawn ones.
    perturbed = reseed.perturb(source, rate=0.25, seed=3)

    assert len(perturbed.layers) == len(source.layers)
    for (weight, bias), (original, _) in zip(perturbed.layers, source.layers):
        assert weight.shape == original.shape
        assert weight.dtype == np.float32 and bias.dtype == np.float32

    head, original_head = perturbed.layers[-1][0], source.layers[-1][0]
    assert not original_head.any(), "the quiet start zeroes its head"
    moved = np.count_nonzero(head) / head.size
    assert 0.2 < moved < 0.3, "about a quarter of the head redrawn"
    # At the layer's own init scale rather than at whatever a trained layer has
    # drifted to, so a fresh entry is explorable instead of immediately dominated.
    assert np.abs(head).max() <= 1.0 / np.sqrt(original_head.shape[0])

    # The source is not mutated, and rate 0 is still a copy the caller owns.
    assert not source.layers[-1][0].any()
    untouched = reseed.perturb(source, rate=0.0, seed=3)
    assert untouched is not source
    assert_array_equal(untouched.layers[0][0], source.layers[0][0])

    # Same seed, same perturbation: the collector and the updater load the same
    # materialized artifact, but a run has to be reproducible from its config too.
    assert_array_equal(
        reseed.perturb(source, rate=0.25, seed=3).layers[0][0],
        perturbed.layers[0][0],
    )


def test_the_reseed_restores_sigma_and_survives_the_inert_guard():
    """Sigma anneals toward zero over a slot's life; a restart needs it back.

    A slot inheriting a collapsed sigma restarts with no exploration budget at
    all and can only ever rediscover what the mature member already knows -- and
    the failure is silent, because nothing else in `metrics_json` moves.
    """

    from agents.dqn_market_free.training import reseed

    module = net_module.build(noisy="head", sigma0=0.5)
    source = net_module.to_numpy(module)
    # Anneal it away, the way a trained slot does.
    source.sigma[-1] = (
        np.zeros_like(source.sigma[-1][0]),
        np.zeros_like(source.sigma[-1][1]),
    )
    assert not source.sigma[-1][0].any()

    restored = reseed.perturb(source, rate=0.1, seed=5, sigma_reset=0.5)
    assert restored.sigma[-1][0].any()
    bound = 1.0 / np.sqrt(source.layers[-1][0].shape[0])
    assert restored.sigma[-1][0].max() == pytest.approx(0.5 * bound)

    # Left alone when not asked for, and a deterministic layer stays deterministic
    # -- `from_numpy` raises if a checkpoint carries sigma the module does not.
    kept = reseed.perturb(source, rate=0.1, seed=5)
    assert not kept.sigma[-1][0].any()
    assert reseed.perturb(OrderQNetwork.quiet_start(seed=1), 0.1, sigma_reset=0.5).sigma == [
        None,
        None,
        None,
    ]

    # A perturbed network must still clear the guard the exported candidate goes
    # through. An all-zero trunk has no gradient outside its output bias, so a
    # slot restarted onto one could never leave.
    updater_module._reject_inert_network(restored)


def test_graduation_enters_the_group_and_retires_the_worst(tmp_path: Path):
    """Capacity is held by retiring, not deleting: a run stays answerable."""

    statements: list[tuple[str, list]] = []

    class Reader:
        def fetchall(self, sql, parameters):
            statements.append((sql, list(parameters)))
            if "INSERT INTO" in sql:
                return [("member",)]
            return [("worst", 1.5, None)]

    result = updater_module._graduate(
        Reader(), "decl", "decl@learner_03", "ckpt-a", 145.8, 30
    )

    insert, cull = statements
    assert "INSERT INTO population_members" in insert[0]
    assert "RETURNING" in insert[0], (
        "the trainer's only handle is the read view, whose fetchall wants rows"
    )
    assert insert[1][1:] == ["decl", "decl@learner_03", "ckpt-a", 145.8, None]

    # The cull reads the live set *after* the insert, so a graduate that is itself
    # the worst member is the one retired and the group cannot be made worse by
    # admitting to it.
    assert "retired_at = current_timestamp" in cull[0]
    # `reserved` defaults to 0, so the whole capacity goes to the score ranking
    # and the archive's cell clause can keep nothing. That is the pre-archive cull
    # expressed in the new statement's terms.
    assert cull[1] == ["decl", 30, 1, 0, 0.0, 0]
    assert result["retired"] == [{"member_id": "worst", "score": 1.5, "cell": None}]
    assert result["score"] == 145.8


def test_a_re_entering_member_is_un_retired():
    """Entry is idempotent per checkpoint, and a standing release re-enters constantly.

    `logmoney-v26` logged 87 entry events for 37 distinct members. Without this,
    a member culled while its slot still held it as the active release had every
    later update rewrite its score on a row that stayed retired -- and since the
    cull counts only live rows, at `capacity - 1` live it then retired nothing and
    the group sat under capacity permanently.
    """

    statements: list[tuple[str, list]] = []

    class Reader:
        def fetchall(self, sql, parameters):
            statements.append((sql, list(parameters)))
            return [("member",)] if "INSERT INTO" in sql else []

    updater_module._graduate(Reader(), "decl", "decl@learner_03", "ckpt-a", 145.8, 30)

    insert = statements[0][0]
    assert "ON CONFLICT (member_id) DO UPDATE" in insert
    assert "retired_at = NULL" in insert
    assert "retired_reason = NULL" in insert


def test_a_hoarder_cannot_outrank_a_trader_in_the_pool():
    """The pool is culled `ORDER BY score DESC`, so money alone selects for idling.

    The stake is 3,000 and a policy that never trades keeps it, while one that
    trades for a season carries hundreds. Ranked on raw money the hoarder enters
    at the top, is never culled, and becomes the reseed source for every slot.
    """

    config = {"population_activity_reference": 0.10}
    hoarder = updater_module._population_score(2998.8, 0.0, config)
    trader = updater_module._population_score(600.0, 0.32, config)

    assert hoarder["money"] > trader["money"], "the premise: money favours the hoarder"
    assert hoarder["score"] < trader["score"], "the ranking must not"
    assert hoarder["score"] == 0.0
    assert trader["score"] == 600.0, "at or above the reference there is no discount"


def test_barely_clearing_the_activity_floor_does_not_buy_the_top_of_the_pool():
    """`viability_min_orders_per_decision` is a binary at 0.02.

    It separates "never acts" from "acts", not "trades" from "hoards", so a policy
    that clears 0.021 while sitting on its stake passes hard viability. The pool
    score is what has to stop it dominating from there.
    """

    config = {"population_activity_reference": 0.10}
    barely = updater_module._population_score(3000.0, 0.021, config)
    trading = updater_module._population_score(600.0, 0.32, config)

    assert barely["activity_factor"] == pytest.approx(0.21)
    assert barely["score"] == pytest.approx(630.0)
    assert barely["score"] < trading["score"] * 1.1, (
        "five times the money on a tenth of the activity should rank comparably, "
        "not dominate"
    )


def test_the_discount_is_a_ramp_rather_than_a_second_threshold():
    """Repeating the floor's shape would only move the same cliff."""

    config = {"population_activity_reference": 0.10}
    scores = [
        updater_module._population_score(1000.0, rate, config)["score"]
        for rate in (0.0, 0.025, 0.05, 0.075, 0.10, 0.5)
    ]

    assert scores == sorted(scores), "monotone in activity"
    assert scores[:5] == pytest.approx([0.0, 250.0, 500.0, 750.0, 1000.0])
    # No upper penalty: over-trading can only surface as lost money, which the
    # money term already carries.
    assert scores[-1] == scores[-2]


def test_more_money_still_wins_at_equal_activity():
    """The discount must not invert the objective it is protecting."""

    config = {"population_activity_reference": 0.10}
    rich = updater_module._population_score(900.0, 0.30, config)
    poor = updater_module._population_score(300.0, 0.30, config)
    assert rich["score"] > poor["score"]


def test_an_unmeasured_activity_ranks_on_money_rather_than_at_zero():
    """A missing measurement is not evidence of idleness.

    Treating it as zero would retire every checkpoint written before the trading
    rate was recorded, the first time the pool was culled.
    """

    scored = updater_module._population_score(500.0, None, {})
    assert scored["score"] == 500.0
    assert scored["scored_by"] == "money_unmeasured_activity"


def test_the_discount_can_be_turned_off():
    scored = updater_module._population_score(2998.8, 0.0, {"population_activity_reference": 0})
    assert scored["score"] == 2998.8
    assert scored["scored_by"] == "money"


def test_a_missing_money_measurement_scores_zero_rather_than_raising():
    assert updater_module._population_score(None, 0.5, {})["score"] == 0.0


def test_the_released_trading_rate_is_read_not_re_simulated():
    """Same route as `_released_benchmark_money`: one row, no arena time."""

    gate = {
        "hard_viability": {
            "benchmarks": [{"trading_rate": 0.31, "passed": True}],
            "head_to_head": {"trading_rate": 0.29},
        }
    }

    class Reader:
        def __init__(self):
            self.queries = 0

        def fetchone(self, sql, parameters):
            self.queries += 1
            assert "model_training_checkpoints" in sql
            return (json.dumps(gate),)

    class Active:
        checkpoint_id = "ckpt-a"

    reader = Reader()
    assert updater_module._released_trading_rate(reader, Active(), "bench") == 0.31
    assert reader.queries == 1

    # No incumbent, and a payload that never recorded it, both read as "unknown".
    assert updater_module._released_trading_rate(reader, None, "bench") is None


def test_an_older_gate_payload_falls_back_through_the_benchmark_diagnostics():
    gate = {
        "challenger_benchmarks": [
            {"opponent": "other", "order_diagnostics": {"trading_rate": 0.9}},
            {"opponent": "bench", "order_diagnostics": {"trading_rate": 0.4}},
        ]
    }

    class Reader:
        def fetchone(self, sql, parameters):
            return (json.dumps(gate),)

    # Matched on the opponent spec, like the money, so a reconfigured benchmark
    # drops the term instead of reading a different opponent's activity.
    assert updater_module._released_trading_rate(Reader(), type("A", (), {"checkpoint_id": "c"})(), "bench") == 0.4
    assert updater_module._released_trading_rate(Reader(), type("A", (), {"checkpoint_id": "c"})(), "missing") is None


def test_the_mature_group_drops_members_whose_weights_are_gone(tmp_path: Path):
    """Best first, and unusable rows filtered rather than raising."""

    present = tmp_path / "present.npz"
    present.write_bytes(b"weights")

    class Reader:
        def fetchall(self, sql, parameters):
            return [
                ("m1", "decl@a", "ckpt-1", 900.0, str(tmp_path / "gone.npz"), None),
                ("m2", "decl@b", "ckpt-2", 400.0, str(present), None),
            ]

    members = updater_module.mature_members(Reader(), "decl")

    assert [member["checkpoint_id"] for member in members] == ["ckpt-2"]
    assert members[0]["score"] == 400.0
    # No fingerprint means no cell, which means no reserved berth. NULL is not
    # zero: a member graduated before `behavior_json` existed has not been
    # measured, and filing it into a cell on that basis would hand the archive a
    # berth to an unmeasured row.
    assert members[0]["cell"] is None and members[0]["reserved"] is False


class _AgeReader:
    """Answers the release-age query with a fixed count."""

    def __init__(self, count):
        self.count = count
        self.queries = 0

    def fetchone(self, sql, parameters):
        self.queries += 1
        assert "model_updates" in sql and "model_training_checkpoints" in sql
        return None if self.count is None else (self.count,)


def test_a_release_is_dated_by_the_updates_it_has_survived():
    assert updater_module._release_age(_AgeReader(7), "decl@learner_00", "ckpt-a") == 7
    assert updater_module._release_age(_AgeReader(0), "decl@learner_00", "ckpt-a") == 0


def test_an_undatable_release_reads_as_unknown_not_as_zero():
    """An unreadable history must never open the gate the lag exists to hold shut."""

    assert updater_module._release_age(_AgeReader(None), "decl@learner_00", "ckpt-a") is None
    # No checkpoint at all short-circuits without querying.
    reader = _AgeReader(5)
    assert updater_module._release_age(reader, "decl@learner_00", None) is None
    assert reader.queries == 0


def test_the_entry_lag_is_what_stops_the_roster_recycling_itself():
    """A release copyable the moment it is promoted closes the loop.

    Whichever slot released last would become the ancestor for whichever slot
    reseeds next, and ten slots converge on clones of the current best -- a greedy
    hill-climb, which is strictly worse than the single lineage the population
    replaced. The eligibility rule below is the whole defence, so it is pinned
    here rather than left to the call site.
    """

    # Read off the spec rather than restated, so the manifest and the rule cannot
    # drift apart without this failing.
    entry_after = updater_module._slot(_SLOT_CONFIG).entry_updates
    assert entry_after >= 1, "entry-on-promotion is the configuration this prevents"

    def eligible(age, graduating, warming=False):
        if warming:
            return False
        return graduating or (age is not None and age >= entry_after)

    # Fresh release, slot still training: held back. This is the 55% of releases
    # `logmoney-v26` superseded within a single update.
    assert not eligible(0, False)
    assert not eligible(entry_after - 1, False)
    # Survived the lag: admitted.
    assert eligible(entry_after, False)
    assert eligible(entry_after + 7, False)
    # Undatable: treated as not old enough.
    assert not eligible(None, False)
    # A graduating slot contributes whatever the age -- it is finishing anyway,
    # and holding its release back would lose it entirely.
    assert eligible(0, True)
    assert eligible(None, True)
    # A warm-starting slot contributes nothing whatever the age. It is declaring
    # its own initialisation wrong, and it must not write to the group while its
    # peers are still reading the ranked list to spread across.
    assert not eligible(entry_after + 7, False, warming=True)


def test_a_zero_entry_lag_is_refused():
    """0 is entry-on-promotion, which is the loop the lag exists to break.

    Refused at parse time in the same spirit as a graduate cut at or above the
    peak: a configuration whose every update is wrong is better rejected than
    obeyed.
    """

    with pytest.raises(ValueError, match="population_entry_updates"):
        updater_module._slot({**_SLOT_CONFIG, "population_entry_updates": 0})
    with pytest.raises(ValueError, match="population_warm_start_min_updates"):
        updater_module._slot(
            {**_SLOT_CONFIG, "population_warm_start_min_updates": -1}
        )


def _members(*scores):
    return [
        {
            "member_id": f"m{index}",
            "model_id": f"decl@learner_{index:02d}",
            "checkpoint_id": f"ckpt-{index}",
            "score": score,
            "inference_artifact_path": f"/tmp/ckpt-{index}.npz",
        }
        for index, score in enumerate(scores)
    ]


def test_a_reseed_draws_from_the_top_rather_than_the_single_best():
    """Argmax over noisy estimates picks the luckiest estimate, not the best policy.

    And it picks the same one every time, because the estimate is stored and
    reused -- so ten slots become ten perturbations of one ancestor.
    """

    members = _members(900.0, 850.0, 800.0, 100.0, 50.0)
    config = {"reseed_source_pool": 3}

    drawn = {
        updater_module._reseed_source(members, config, seed=7, reseed_index=index)[
            "checkpoint_id"
        ]
        for index in range(40)
    }

    assert drawn <= {"ckpt-0", "ckpt-1", "ckpt-2"}, "never below the configured top"
    assert len(drawn) > 1, "and not always the same one"


def test_the_draw_is_reproducible():
    """A restart has to be answerable afterwards, like every other seeded choice."""

    members = _members(900.0, 850.0, 800.0)
    config = {"reseed_source_pool": 3}
    first = updater_module._reseed_source(members, config, seed=3, reseed_index=2)
    again = updater_module._reseed_source(members, config, seed=3, reseed_index=2)
    assert first["checkpoint_id"] == again["checkpoint_id"]

    # Different slot seed, or a later reseed on the same slot, may draw differently.
    others = {
        updater_module._reseed_source(members, config, seed=seed, reseed_index=index)[
            "checkpoint_id"
        ]
        for seed in range(6)
        for index in range(6)
    }
    assert len(others) > 1


def test_a_pool_of_one_reproduces_taking_the_best():
    members = _members(900.0, 850.0, 800.0)
    chosen = updater_module._reseed_source(members, {"reseed_source_pool": 1}, seed=1, reseed_index=0)
    assert chosen["checkpoint_id"] == "ckpt-0"


def test_the_draw_degrades_to_what_the_group_actually_holds():
    """The pool is expected to hold very few members early on."""

    single = updater_module._reseed_source(_members(500.0), {"reseed_source_pool": 3}, seed=1, reseed_index=0)
    assert single["checkpoint_id"] == "ckpt-0"
    assert updater_module._reseed_source([], {"reseed_source_pool": 3}, seed=1, reseed_index=0) is None


def test_a_nonsense_pool_size_does_not_crash_the_restart():
    members = _members(900.0, 100.0)
    for size in (0, -5):
        chosen = updater_module._reseed_source(members, {"reseed_source_pool": size}, seed=1, reseed_index=0)
        assert chosen["checkpoint_id"] == "ckpt-0"


def _lineage_members(*pairs):
    """Members whose originating slot is given rather than one per entry.

    `_members` assigns a distinct lineage to every member, which is the shape the
    group is *assumed* to have. This builds the shape it actually had.
    """

    return [
        {
            "member_id": f"m{index}",
            "model_id": f"decl@{lineage}",
            "checkpoint_id": f"ckpt-{index}",
            "score": score,
            "inference_artifact_path": f"/tmp/ckpt-{index}.npz",
        }
        for index, (lineage, score) in enumerate(pairs)
    ]


# The live group at `market-free-streak-v27`, best first, as it was: thirty
# checkpoints from ten lineages with `logmoney_04` holding ranks 0, 1, 5, 6 and 9.
# Truncated to the head, which is the part `reseed_source_pool` ever reads.
_V27_HEAD = (
    ("logmoney_04", 6270.8),
    ("logmoney_04", 6070.3),
    ("logmoney_02", 4559.9),
    ("logmoney_07", 4380.5),
    ("logmoney_06", 3997.4),
    ("logmoney_04", 3883.0),
    ("logmoney_04", 3777.7),
    ("logmoney_02", 3452.1),
    ("logmoney_05", 3236.6),
    ("logmoney_04", 3037.7),
)


def test_the_cap_stops_one_lineage_owning_the_top_of_the_draw():
    """Ranked on score alone, a population of ten read as a population of four.

    Two of `market-free-streak-v27`'s top three were `logmoney_04`'s own history,
    so a uniform draw over the top three handed that one lineage two thirds of
    every restart -- the concentration `reseed_source_pool` exists to prevent,
    reappearing one level down because the group ranks checkpoints and the
    argument is about policies.
    """

    members = _lineage_members(*_V27_HEAD)
    config = {"reseed_source_pool": 3, "reseed_source_lineage_cap": 1}

    capped = updater_module._capped_members(members, config)
    lineages = [member["model_id"] for member in capped]
    assert len(lineages) == len(set(lineages)), "one entry per lineage"
    assert lineages[:3] == ["decl@logmoney_04", "decl@logmoney_02", "decl@logmoney_07"]

    drawn = {
        updater_module._reseed_source(members, config, seed=7, reseed_index=index)[
            "checkpoint_id"
        ]
        for index in range(40)
    }
    # ckpt-1 is `logmoney_04`'s second-best checkpoint, which the uncapped draw
    # reached on roughly a third of restarts.
    assert "ckpt-1" not in drawn
    assert drawn <= {"ckpt-0", "ckpt-2", "ckpt-3"}
    assert len(drawn) > 1


def test_the_cap_keeps_the_best_entry_of_each_lineage():
    """The cap costs rank, never quality: what it drops, the lineage already beat."""

    members = _lineage_members(
        ("logmoney_04", 6270.8), ("logmoney_04", 6070.3), ("logmoney_02", 4559.9)
    )
    capped = updater_module._capped_members(members, {"reseed_source_lineage_cap": 1})
    assert [member["checkpoint_id"] for member in capped] == ["ckpt-0", "ckpt-2"]


def test_the_cap_is_configurable_and_never_zero():
    members = _lineage_members(*_V27_HEAD)
    # `logmoney_04` holds five of these ten and `logmoney_02` two, so a cap of 2
    # admits two of each and all three singletons.
    assert len(updater_module._capped_members(members, {"reseed_source_lineage_cap": 2})) == 7
    for nonsense in (0, -3):
        capped = updater_module._capped_members(
            members, {"reseed_source_lineage_cap": nonsense}
        )
        assert [member["checkpoint_id"] for member in capped[:2]] == ["ckpt-0", "ckpt-2"]


def test_a_warm_start_gives_every_slot_a_different_ancestor():
    """Ten slots choosing independently from ten lineages would collide.

    There is no coordination point -- each slot is a separate process on its own
    round -- so the ordinal is the coordination. Distinctness has to be a property
    of the arithmetic rather than of the draw.
    """

    members = _lineage_members(*_V27_HEAD)
    config = {"reseed_source_pool": 3, "reseed_source_lineage_cap": 1}
    candidates = updater_module._capped_members(members, config)

    chosen = [
        updater_module._reseed_source(
            members, config, seed=11, reseed_index=0, ordinal=ordinal, spread=True
        )
        for ordinal in range(len(candidates))
    ]

    assert [member["checkpoint_id"] for member in chosen] == [
        member["checkpoint_id"] for member in candidates
    ], "the capped list, in order, one per slot"
    assert len({member["model_id"] for member in chosen}) == len(candidates)
    # Same slot, same answer: a warm start is as answerable afterwards as a draw.
    again = updater_module._reseed_source(
        members, config, seed=11, reseed_index=0, ordinal=3, spread=True
    )
    assert again["checkpoint_id"] == chosen[3]["checkpoint_id"]


def test_the_spread_wraps_rather_than_failing_on_a_short_list():
    """Fewer lineages than slots is unavoidable collision, not an error.

    Wrapping keeps the shortfall at the bottom, so the slots that double up double
    up on the best entries rather than on the worst.
    """

    members = _lineage_members(
        ("a", 900.0), ("b", 800.0), ("c", 700.0), ("d", 600.0)
    )
    config = {"reseed_source_lineage_cap": 1}
    chosen = [
        updater_module._reseed_source(
            members, config, seed=1, reseed_index=0, ordinal=ordinal, spread=True
        )["checkpoint_id"]
        for ordinal in range(10)
    ]
    assert chosen[0] == "ckpt-0"
    assert chosen == ["ckpt-0", "ckpt-1", "ckpt-2", "ckpt-3"] * 2 + ["ckpt-0", "ckpt-1"]


def test_the_spread_is_off_for_an_ordinary_reseed():
    """A permanent ordinal-to-rank map is the argmax failure, for a tenth of the roster.

    Ordinary reseeds happen at different times on different slots, so there is
    nothing to spread against -- and slot 0 would draw the group's single best on
    every restart forever.
    """

    members = _lineage_members(("a", 900.0), ("b", 850.0), ("c", 800.0), ("d", 100.0))
    config = {"reseed_source_pool": 3, "reseed_source_lineage_cap": 1}
    drawn = {
        updater_module._reseed_source(
            members, config, seed=5, reseed_index=index, ordinal=0, spread=False
        )["checkpoint_id"]
        for index in range(40)
    }
    assert len(drawn) > 1, "the ordinal must not pin an ordinary draw"
    assert drawn <= {"ckpt-0", "ckpt-1", "ckpt-2"}


def test_the_slot_ordinal_is_read_rather_than_parsed():
    """It is already recorded twice, so it needs deriving from neither."""

    assert updater_module._slot_ordinal({"name": "streak_07"}, "decl@streak_07") == 7
    # No stamped name: the `@` suffix is the fallback.
    assert updater_module._slot_ordinal({}, "decl@streak_09") == 9
    # A population of one draws the way it always did.
    assert updater_module._slot_ordinal({"name": "default"}, "decl") is None
    assert updater_module._slot_ordinal({}, "decl") is None


def _warm_start_state(spec, previous=None):
    return updater_module._slot_state(previous or {}, spec)


def test_a_warm_start_bypasses_the_graduation_floors():
    """The floors gate a claim about evidence. This makes no such claim.

    `lr_min_updates_before_graduation` stops warmup noise ending a slot that has
    not had a chance to train. A warm start reads no score -- it corrects the
    slot's initialisation, and every update spent waiting the floor out is an
    update spent training the initialisation being corrected.
    """

    spec = updater_module._slot(_SLOT_CONFIG)
    members = _members(900.0)
    state = _warm_start_state(spec)
    state = updater_module._settle_slot(state, 500.0, spec)

    assert state.updates_since_reseed < spec.min_updates
    assert not updater_module._graduates(state, spec)
    assert updater_module._warm_starts(state, spec, members)


def test_a_slot_warm_starts_once_and_then_never_again():
    """The marker it writes advances `lr_reseed_index`, which is the whole defence.

    No separate flag to reconcile, and a re-firing warm start would restart the
    slot on every update forever -- it would never train at all.
    """

    spec = updater_module._slot(_SLOT_CONFIG)
    members = _members(900.0)

    first = _warm_start_state(spec)
    assert first.reseed_index == 0
    assert updater_module._warm_starts(first, spec, members)

    # What the warm-starting update records: its own state plus the marker.
    recorded = {**first.as_metrics(), "reseed_pending": True, "reseed_batch_floor": 50}
    second = updater_module._slot_state(recorded, spec)

    assert second.reseed_index == 1
    assert second.batch_floor == 50, "the pre-warm-start batches are discarded"
    assert not updater_module._warm_starts(second, spec, members)


def test_a_warm_start_does_not_fire_without_usable_weights():
    """An empty group is the first run of a declared model, where a cold start is all there is."""

    spec = updater_module._slot(_SLOT_CONFIG)
    state = _warm_start_state(spec)

    assert not updater_module._warm_starts(state, spec, [])
    # Off by configuration -- the deliberate cold control arm.
    off = updater_module._slot({**_SLOT_CONFIG, "population_warm_start": False})
    assert not updater_module._warm_starts(state, off, _members(900.0))
    # And inert for a run with no population at all.
    unpopulated = updater_module._slot({**_SLOT_CONFIG, "lr_graduate_cut": 0.0})
    assert not updater_module._warm_starts(
        _warm_start_state(unpopulated), unpopulated, _members(900.0)
    )


class _WarmStartReader:
    """`_declared_model_id` and `mature_members`, and a count of how often."""

    def __init__(self, rows):
        self.rows = rows
        self.queries = 0

    def fetchone(self, sql, parameters):
        return ("decl",)

    def fetchall(self, sql, parameters):
        self.queries += 1
        return self.rows


def _warm_start_context(tmp_path: Path, config=None):
    return SimpleNamespace(
        config={**(config or _SLOT_CONFIG), "seed": 5, "population_slots": 10},
        model_id="decl@landrate_00",
        artifacts=tmp_path / "artifacts",
    )


def _ancestor_on_disk(tmp_path: Path):
    """One mature member with real weights, since `mature_members` drops missing ones."""

    path = tmp_path / "ancestor.npz"
    OrderQNetwork.random(seed=17).save(path)
    return path, [("m0", "decl@logmoney_04", "ckpt-a", 6270.75, str(path), None)]


def test_generation_zero_is_collected_from_the_group_not_from_a_cold_start(tmp_path: Path):
    """The adoption has to happen before the first round, not inside the first update.

    `_warm_starts` already decides a fresh roster should start from the group,
    but it runs inside `update`, which fires after `rounds_per_generation`
    rounds. Measured across v26 to v31: `own_mean_final_money` is exactly 3000.0
    at `cutoff_batch_seq = 10` in all six sessions -- a whole generation of NOOP
    episodes into the ring -- including the three whose first update recorded
    `population.action = "warm_start"`.
    """

    ancestor_path, rows = _ancestor_on_disk(tmp_path)
    context = _warm_start_context(tmp_path)
    reader = _WarmStartReader(rows)

    produced = updater_module._materialize_warm_start(context, reader, {})

    assert produced is not None and produced.is_file()
    network, sidecar = updater_module.warm_start_paths(context.artifacts)
    assert produced == network
    assert json.loads(sidecar.read_text())["checkpoint_id"] == "ckpt-a"

    # It is the ancestor perturbed, not the cold start -- `initial_network` has an
    # all-zero output layer, which is what makes its argmax tie to NOOP.
    seeded = OrderQNetwork.load(produced)
    cold = updater_module.initial_network(context.config)
    assert not np.allclose(seeded.layers[-1][0], cold.layers[-1][0])
    # And it really came from the ancestor rather than from noise: most of the
    # output layer survives `reseed_dropout`, but not all of it.
    ancestor = OrderQNetwork.load(ancestor_path)
    shared = float(np.isclose(seeded.layers[-1][0], ancestor.layers[-1][0]).mean())
    assert 0.5 < shared < 1.0


def test_the_warm_start_is_drawn_once_for_the_whole_first_generation(tmp_path: Path):
    """Ten rounds of generation 0, one draw, one mask, one set of bytes."""

    _, rows = _ancestor_on_disk(tmp_path)
    context = _warm_start_context(tmp_path)
    reader = _WarmStartReader(rows)

    first = updater_module._materialize_warm_start(context, reader, {})
    digest = first.read_bytes()
    after_first = reader.queries

    for _ in range(9):
        again = updater_module._materialize_warm_start(context, reader, {})
        assert again == first
        assert again.read_bytes() == digest

    assert reader.queries == after_first, "rounds 2-10 must not re-draw an ancestor"


def test_the_warm_start_leaves_the_cold_start_path_alone_when_it_does_not_apply(
    tmp_path: Path,
):
    """Every route back to `initial_network` that has to stay open."""

    _, rows = _ancestor_on_disk(tmp_path)

    # A slot that has already trained: `update` owns the decision from there.
    context = _warm_start_context(tmp_path)
    assert (
        updater_module._materialize_warm_start(
            context, _WarmStartReader(rows), {"lr_reseed_index": 0, "lr": 1e-4}
        )
        is None
    )

    # The first session of a declared model, where there is genuinely nothing to
    # start from.
    empty = _warm_start_context(tmp_path / "empty")
    assert updater_module._materialize_warm_start(empty, _WarmStartReader([]), {}) is None

    # Off by configuration -- the deliberate cold control arm.
    off = _warm_start_context(
        tmp_path / "off", {**_SLOT_CONFIG, "population_warm_start": False}
    )
    assert updater_module._materialize_warm_start(off, _WarmStartReader(rows), {}) is None

    # And inert for a run with no population at all.
    unpopulated = _warm_start_context(
        tmp_path / "unpopulated", {**_SLOT_CONFIG, "lr_graduate_cut": 0.0}
    )
    assert (
        updater_module._materialize_warm_start(unpopulated, _WarmStartReader(rows), {})
        is None
    )


def test_the_warm_start_mask_matches_the_one_the_update_would_have_drawn(tmp_path: Path):
    """Moving the materialisation must not quietly move the perturbation.

    `collect` and `update` load identical bytes by construction now -- the update
    reads this file rather than drawing again -- but the *mask* still has to be
    the one the update path produces at reseed zero, or the change silently
    altered how much of the ancestor survives.
    """

    from agents.dqn_market_free.training import reseed

    ancestor_path, rows = _ancestor_on_disk(tmp_path)
    context = _warm_start_context(tmp_path)
    spec = updater_module._slot(context.config)

    produced = updater_module._materialize_warm_start(context, _WarmStartReader(rows), {})

    expected = reseed.perturb(
        OrderQNetwork.load(ancestor_path),
        spec.dropout,
        spec.shrink,
        # The update path's `seed + reseed_index + 1`, at reseed index zero.
        seed=int(context.config["seed"]) + 0 + 1,
        sigma_reset=updater_module._noisy(context.config).sigma0 if spec.sigma_reset else None,
    )
    seeded = OrderQNetwork.load(produced)
    for mine, theirs in zip(seeded.layers, expected.layers):
        for a, b in zip(mine, theirs):
            assert np.allclose(a, b)


def test_a_warm_start_can_be_held_back_by_its_own_floor():
    spec = updater_module._slot(
        {**_SLOT_CONFIG, "population_warm_start_min_updates": 3}
    )
    members = _members(900.0)
    state = _warm_start_state(spec)
    assert not updater_module._warm_starts(state, spec, members)

    previous = state.as_metrics()
    for _ in range(3):
        state = updater_module._slot_state(previous, spec)
        previous = state.as_metrics()
    assert state.updates_since_reseed == 3
    assert updater_module._warm_starts(state, spec, members)


def test_the_shortened_ladder_keeps_the_same_terminal_rate():
    """`lr_decay_factor` 0.25 halves the ladder without moving where it ends.

    0.25**2 and 0.5**4 are both 0.0625, so a slot still finishes at 6.25e-6 --
    and `lr_plateau_patience` is untouched, so every decay is still four
    consecutive non-improving updates rather than two.
    """

    peak, cut = 1e-4, 1e-5

    def decays(factor):
        steps = 0
        while peak * factor**steps >= cut:
            steps += 1
        return steps

    assert decays(0.5) == 4
    assert decays(0.25) == 2
    assert peak * 0.25**2 == pytest.approx(peak * 0.5**4)


def test_the_self_play_pool_is_bounded_below_by_the_reseed(tmp_path: Path):
    """A restarted slot must not mirror the trajectory it was restarted to leave.

    `landgate-v10` is the record of skipping this: the reset restored an 18k
    checkpoint, the mirror kept playing the discarded 357-coin policy, and the
    next twenty batches were collected against it.
    """

    seen: list[tuple[str, list]] = []

    class Reader:
        def fetchall(self, sql, parameters):
            seen.append((sql, list(parameters)))
            return []

    context = SimpleNamespace(model_id="free.model")
    _self_play_pool(context, Reader(), 5)
    assert "created_at > (" not in seen[-1][0], "no floor without a reseed"

    _self_play_pool(context, Reader(), 5, "ckpt-reseed")
    assert "created_at > (" in seen[-1][0]
    assert seen[-1][1] == ["free.model", "ckpt-reseed", 5]


def test_the_reseed_watermarks_are_held_for_the_slot_and_not_for_one_update():
    """The other half of the reset, and the half `landgate-v10` forgot.

    `fill_buffer` rebuilds from recent batches whatever weights were loaded, and
    `_self_play_pool` takes the newest checkpoints whether promoted or not, so a
    restored checkpoint trains immediately on episodes generated by the policy
    just thrown away and mirrors it as well. Holding the two watermarks across
    only the consuming update lets both come straight back one update later,
    which is the shape that made the original reset a third of a reset.
    """

    spec = updater_module._slot(_SLOT_CONFIG)
    marker = {
        "reseed_pending": True,
        "reseed_batch_floor": 1400,
        "reseed_after_checkpoint_id": "ckpt-reseed",
        "lr_reseed_index": 2,
        # Deliberately healthy, to show the consuming update discards them.
        "lr_updates_since_reseed": 40,
        "lr_decay_steps": 3,
        "lr_best_score": 900.0,
        "lr_stall_updates": 1,
    }

    consuming = updater_module._slot_state(marker, spec)
    assert consuming.updates_since_reseed == 0
    assert consuming.decay_steps == 0 and consuming.stall_updates == 0
    assert consuming.best_score is None, "an inherited score would pin the plateau"
    assert consuming.reseed_index == 3
    assert consuming.batch_floor == 1400
    assert consuming.after_checkpoint_id == "ckpt-reseed"

    # And the update after it, which carries no marker at all, keeps both.
    following = updater_module._slot_state(consuming.as_metrics(), spec)
    assert following.updates_since_reseed == 1
    assert following.batch_floor == 1400
    assert following.after_checkpoint_id == "ckpt-reseed"


def test_cold_start_emits_nothing():
    state = np.zeros(STATE_DIM, dtype=np.float32)
    decision = choose_order(state, initial_network({}))
    assert decision.chosen == 0
    assert decision.order == Order("NOOP")
    assert decision.wire() == []


def test_cold_start_leaves_every_layer_trainable():
    """Gradients must reach the hidden layers, not just the output bias.

    An all-zero MLP zeroes both hidden activations for any input, so every
    weight gradient is zero and `relu'(0) == 0` closes the path back to the
    biases. Only the output bias moves, the symmetry never breaks, and the run
    emits NOOP forever while reporting a healthy falling loss.
    """

    module = net_module.from_numpy(
        initial_network({}), net_module.build()
    )
    linears = [layer for layer in module if isinstance(layer, torch.nn.Linear)]
    before = [layer.weight.detach().clone() for layer in linears]

    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(64, STATE_DIM, generator=generator)
    targets = torch.randn(64, ACTION_COUNT, generator=generator) * 100.0
    for _ in range(20):
        loss = torch.nn.SmoothL1Loss()(module(features), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for index, (layer, original) in enumerate(zip(linears, before)):
        assert not torch.equal(layer.weight.detach(), original), f"layer {index} never trained"


def test_an_all_zero_network_is_recognised_as_inert():
    assert net_module.has_inert_hidden_trunk(OrderQNetwork.zeros())
    assert not net_module.has_inert_hidden_trunk(
        initial_network({})
    )


def test_numpy_and_torch_networks_agree():
    module = net_module.build()
    network = net_module.to_numpy(module)
    features = np.random.default_rng(0).standard_normal((32, STATE_DIM)).astype(np.float32)
    with torch.no_grad():
        expected = module(torch.from_numpy(features)).numpy()
    observed = network(features)
    assert observed.shape == (32, ACTION_COUNT)
    assert np.allclose(observed, expected, atol=1e-5)


def _inject_matching_noise(module, network, scale=1.0):
    """Force one epsilon draw into both implementations so they are comparable."""

    generator = torch.Generator().manual_seed(4)
    net_module.reset_noise(module, generator, scale)
    network._noise = [
        (layer.epsilon_in.numpy().copy(), layer.epsilon_out.numpy().copy())
        if isinstance(layer, net_module.NoisyLinear)
        else None
        for layer in net_module._exportable(module)
    ]


@pytest.mark.parametrize("mode", ["head", "all"])
def test_noisy_numpy_and_torch_networks_agree(mode):
    """The parity gate, under noise. Same weights *and* the same epsilon draw.

    Both sides compute the factored form in the same association order on
    purpose: materializing `mu + sigma * outer(f_out, f_in)` on one side only is
    algebraically identical and rounds differently in float32, which is exactly
    the kind of drift this tolerance exists to catch.
    """

    module = net_module.build(noisy=mode, sigma0=0.5)
    network = net_module.to_numpy(module)
    features = np.random.default_rng(0).standard_normal((32, STATE_DIM)).astype(np.float32)

    _inject_matching_noise(module, network)
    with torch.no_grad():
        expected = module(torch.from_numpy(features)).numpy()
    observed = network(features)
    assert np.allclose(observed, expected, atol=1e-5)
    # And the noise has to actually be doing something, or this passes vacuously.
    assert not np.allclose(observed, OrderQNetwork(network.layers)(features), atol=1e-6)


@pytest.mark.parametrize("scale", [0.25, 3.0])
def test_a_scaled_draw_scales_the_perturbation_exactly(scale):
    """`scale` multiplies the perturbation, and nothing else.

    Both sides are bilinear in the two epsilons, so folding the factor into one
    of them has to come out as an exact multiple of the unscaled perturbation.
    Asserted against the *difference* from mu rather than against the output,
    because the mu term is not scaled and would swamp the check.
    """

    module = net_module.build(noisy="head", sigma0=0.5)
    network = net_module.to_numpy(module)
    features = np.random.default_rng(3).standard_normal((16, STATE_DIM)).astype(np.float32)
    quiet = OrderQNetwork(network.layers)(features)

    _inject_matching_noise(module, network)
    once = network(features) - quiet
    _inject_matching_noise(module, network, scale)
    scaled = network(features) - quiet

    assert np.allclose(scaled, scale * once, atol=1e-5)
    # And torch agrees with numpy under the scaled draw, not only the plain one.
    with torch.no_grad():
        assert np.allclose(module(torch.from_numpy(features)).numpy() - quiet, scaled, atol=1e-5)


def test_a_zero_scale_is_no_noise_at_all():
    """`scale = 0` has to reach the same output as never having drawn.

    The torch forward short-circuits on `epsilon_out.any()`, so this is also what
    pins that the short circuit and the general path agree.
    """

    module = net_module.build(noisy="head", sigma0=0.5)
    network = net_module.to_numpy(module)
    features = np.random.default_rng(6).standard_normal((8, STATE_DIM)).astype(np.float32)

    _inject_matching_noise(module, network, 0.0)
    with torch.no_grad():
        assert np.allclose(
            module(torch.from_numpy(features)).numpy(),
            OrderQNetwork(network.layers)(features),
            atol=1e-6,
        )
    assert np.allclose(network(features), OrderQNetwork(network.layers)(features), atol=1e-6)


def test_a_noisy_network_is_deterministic_until_noise_is_drawn():
    """Evaluation determinism is structural: nothing draws noise unless asked."""

    network = OrderQNetwork.random(seed=11, sigma0=0.5)
    state = np.zeros(STATE_DIM, dtype=np.float32)

    quiet = network(state)
    assert np.array_equal(network(state), quiet)

    rng = np.random.default_rng(0)
    network.sample_noise(rng)
    first = network(state)
    network.sample_noise(rng)
    assert not np.array_equal(first, network(state))

    network.clear_noise()
    assert np.array_equal(network(state), quiet)


def test_a_noisy_checkpoint_round_trips_and_an_old_one_still_loads(tmp_path):
    """Sigma survives save/load, and a pre-NoisyNet file reads exactly as before."""

    noisy = OrderQNetwork.random(seed=5, sigma0=0.5)
    path = tmp_path / "noisy.npz"
    noisy.save(path)
    restored = OrderQNetwork.load(path)
    assert restored.is_noisy
    for original, loaded in zip(noisy.sigma, restored.sigma):
        assert (original is None) == (loaded is None)
        if original is not None:
            assert_array_equal(original[0], loaded[0])
            assert_array_equal(original[1], loaded[1])

    plain = OrderQNetwork.quiet_start(seed=0)
    legacy = tmp_path / "legacy.npz"
    plain.save(legacy)
    with np.load(legacy) as data:
        assert not any(key.startswith("sw") for key in data.files), (
            "a deterministic network must write the pre-NoisyNet key set exactly, "
            "or every run with noise off churns its checkpoints"
        )
    back = OrderQNetwork.load(legacy)
    assert not back.is_noisy
    features = np.random.default_rng(1).standard_normal((4, STATE_DIM)).astype(np.float32)
    assert_array_equal(back(features), plain(features))


def test_a_half_written_sigma_pair_is_refused(tmp_path):
    path = tmp_path / "half.npz"
    network = OrderQNetwork.random(seed=2, sigma0=0.5)
    payload = {}
    with_sigma = tmp_path / "full.npz"
    network.save(with_sigma)
    with np.load(with_sigma) as data:
        payload = {key: data[key] for key in data.files if not key.startswith("sb")}
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="half a sigma pair"):
        OrderQNetwork.load(path)


def test_noise_explores_like_the_policy_not_like_a_coin():
    """The claim this whole change rests on, stated as a measurement.

    Epsilon draws uniformly over feasible actions with NOOP removed, so 97 of the
    186 actions -- every BUY, HIRE and BUY_LAND rung -- are spending, and the
    share of exploring turns that spend money is a property of the action space
    rather than of the policy. Weight noise perturbs Q instead, so it can only
    move the argmax between actions that were already close: a purchase the
    network scores far below its best action stays unreachable.
    """

    ruinous = order_index("BUY_LAND", None, 1)
    if ruinous is None:
        ruinous = next(
            index for index, action in enumerate(ACTIONS) if action.op == "BUY_LAND"
        )
    preferred = order_index("SELL", "WHEAT", 1)
    assert preferred is not None

    network = OrderQNetwork.random(seed=7, sigma0=0.5)
    # A head whose mu says, unambiguously, that BUY_LAND is the worst thing here.
    weight, bias = network.layers[-1]
    bias = np.zeros_like(bias)
    bias[preferred] = 10.0
    bias[ruinous] = -10.0
    network.layers[-1] = (np.zeros_like(weight), bias)

    state = np.zeros(STATE_DIM, dtype=np.float32)
    rng = np.random.default_rng(0)
    trials = 2_000
    chosen = []
    for _ in range(trials):
        network.sample_noise(rng)
        chosen.append(choose_order(state, network).chosen)

    assert ruinous not in chosen, "weight noise must never reach a clearly-bad purchase"

    # The contrast this replaced: uniform exploration over the feasible actions
    # reached the same purchase constantly, because 97 of the 186 actions spend
    # money and the draw never consulted the network. That branch was removed at
    # v13, so what is asserted here is the bound alone: the perturbation is
    # scaled by the Q gap, and a purchase mu says is ruinous stays unreachable.


def test_the_evaluation_path_never_draws_noise():
    """The arena, the viability gates and the Kaggle entry all come through here.

    They reach the network via `main.agent`, which leaves `noise_cadence` at its
    default, so promotion cannot be decided on a perturbed policy even when the
    checkpoint it loads carries sigma.
    """

    network = OrderQNetwork.random(seed=3, sigma0=0.5)
    agent = MarketFreeAgent(network, rng=np.random.default_rng(0))
    assert agent.noise_cadence == "none"
    for step in range(5):
        agent._sample_noise(step, 24)
    assert all(item is None for item in network._noise)

    exploring = MarketFreeAgent(
        network, rng=np.random.default_rng(0), noise_cadence="episode"
    )
    exploring._sample_noise(0, 24)
    assert any(item is not None for item in network._noise)


def test_day_cadence_redraws_only_on_day_boundaries():
    """One perturbed policy per in-game day, which is what `n_step` is sized to.

    Under per-decision noise a 24-step return is produced by 24 different
    perturbed policies, so the bootstrap values a return no single policy would
    generate. Holding the draw for exactly `turns_per_day` makes each n-step
    return attributable to one. It is also the granularity the economy actually
    has: hire cost is fibonacci in `hires_today` and resets daily, and hands are
    wiped at the day boundary.
    """

    turns_per_day = 24
    network = OrderQNetwork.random(seed=9, sigma0=0.5)
    agent = MarketFreeAgent(
        network, rng=np.random.default_rng(0), noise_cadence="day"
    )

    drawn_at = []
    previous = None
    for step in range(3 * turns_per_day):
        agent._sample_noise(step, turns_per_day)
        current = [None if pair is None else pair[1].copy() for pair in network._noise]
        if previous is None or any(
            (a is None) != (b is None) or (a is not None and not np.array_equal(a, b))
            for a, b in zip(current, previous)
        ):
            drawn_at.append(step)
        previous = current

    assert drawn_at == [0, turns_per_day, 2 * turns_per_day]


def test_actor_sigma_scales_span_exploring_and_greedy():
    """Same both-ends-present property `actor_epsilons` is asserted to have."""

    scales = actor_sigma_scales(8, 32.0)
    assert len(scales) == 8
    assert scales[0] == pytest.approx(1.0)
    assert scales == sorted(scales, reverse=True)
    # The last actor is effectively greedy, so every batch carries a clean read
    # of what the policy actually does alongside the exploratory episodes.
    assert scales[-1] < 0.05
    assert actor_sigma_scales(1) == [1.0]


def test_the_ladder_never_explores_above_the_setpoint():
    """Downward only, and the two mechanisms that broke that rule are retired.

    The idle burst went at v15 and `noisy_actor_hot` went with the population.
    Both were answers to the same complaint -- nothing anywhere explores harder
    than the controller resolved -- and per-episode attribution refused both. On
    `burst-v9`, 212 seats at sigma 3.159 finished 49% bankrupt against 8-26%
    across the ordinary ladder and bought zero land; on `falloff-v18` the hot
    rungs' land rate was 4.6% against the ordinary rungs' 4.9%.

    97 of 186 actions spend money, so a widened policy is broke before a rare
    action is affordable. Escaping a stuck trajectory is the population's job now.
    """

    scales = actor_sigma_scales(8, 32.0)
    assert max(scales) == pytest.approx(1.0)
    assert not [scale for scale in scales if scale > 1.0]
    # The retired keys are refused rather than ignored, because a config still
    # carrying them was asking for something the ladder no longer does.
    for key in ("noisy_actor_hot", "noisy_hot_order_rate", "noisy_hot_scale_limit"):
        with pytest.raises(ValueError, match="retired"):
            updater_module._noisy({"noisy": True, key: 2})


def test_the_flip_rate_rises_with_the_sigma_scale():
    """The property `resolve_sigma_scale` is a lookup against.

    A larger perturbation cannot flip fewer argmaxes on average, so the measured
    curve has to be non-decreasing. If it is not, the resolver is interpolating
    between readings that disagree and the whole controller is reading noise.
    """

    from agents.dqn_market_free.training.updater import SIGMA_LADDER, flip_rate_curve

    # The global seed, not just the feature generator. `build` initialises the
    # weights from the global RNG and `flip_rate_curve` draws its noise from it
    # too, so seeding only the features left this test measuring a fresh random
    # network every run -- and near the top of the ladder the rate saturates
    # around 0.8, where eight draws are not enough to keep two adjacent rungs
    # apart. It failed about one run in six that way, which is a flake reporting
    # probe noise rather than the property being asserted.
    torch.manual_seed(1)
    module = net_module.build(noisy="head", sigma0=0.5)
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(256, STATE_DIM, generator=generator)
    mask = torch.ones(256, ACTION_COUNT, dtype=torch.bool)

    curve, order_curve = flip_rate_curve(module, features, mask)

    assert list(curve) == list(SIGMA_LADDER)
    rates = list(curve.values())
    assert rates == sorted(rates)
    # Both ends have to be informative or the ladder does not bracket anything.
    assert rates[0] < rates[-1]
    # The flip-to-order curve is swept over the same rungs, because a controller
    # can only be pointed at a quantity that was measured at more than one scale.
    # It is empty only when the greedy policy never sits out on the probe, since
    # there is then nothing for the perturbation to talk out of NOOP.
    assert not order_curve or list(order_curve) == list(SIGMA_LADDER)
    assert all(0.0 <= rate <= 1.0 for rate in order_curve.values())


def test_the_probe_ladder_stays_inside_the_scale_limit():
    """A curve point the resolver would clamp away is a measurement wasted."""

    from agents.dqn_market_free.training.updater import flip_rate_curve

    module = net_module.build(noisy="head", sigma0=0.5)
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(32, STATE_DIM, generator=generator)
    mask = torch.ones(32, ACTION_COUNT, dtype=torch.bool)

    curve, _ = flip_rate_curve(module, features, mask, scale_limit=2.0, draws=1)
    assert list(curve) == [0.5, 1.0, 2.0]

    # A limit tighter than every rung still has to return a usable curve rather
    # than an empty one, or `resolve_sigma_scale` loses its only input.
    tight, _ = flip_rate_curve(module, features, mask, scale_limit=1.0, draws=1)
    assert list(tight) == [1.0]


def test_the_probe_respects_the_feasibility_mask():
    """A masked action must never be the argmax, perturbed or not.

    The probe measures the decision the policy would actually make, and the
    collector never emits an infeasible order. A flip rate counted over actions
    the simulator would refuse is a different number that happens to look like
    this one.
    """

    from agents.dqn_market_free.training.updater import flip_rate_curve

    module = net_module.build(noisy="head", sigma0=2.0)
    generator = torch.Generator().manual_seed(5)
    features = torch.randn(64, STATE_DIM, generator=generator)
    # Only NOOP is feasible, so nothing can flip anywhere at any scale.
    mask = torch.zeros(64, ACTION_COUNT, dtype=torch.bool)
    mask[:, 0] = True

    curve, order_curve = flip_rate_curve(module, features, mask, draws=2)
    assert set(curve.values()) == {0.0}
    # NOOP is the greedy choice everywhere and the only feasible one, so no
    # perturbation at any scale can talk the policy into an order.
    assert set(order_curve.values()) == {0.0}


def test_resolve_sigma_scale_reads_the_curve_rather_than_chasing_it():
    from agents.dqn_market_free.training.updater import resolve_sigma_scale

    curve = {0.25: 0.01, 0.5: 0.02, 1.0: 0.04, 2.0: 0.08, 4.0: 0.20}

    # Exactly on a ladder point.
    assert resolve_sigma_scale(curve, 0.04) == pytest.approx(1.0)
    # Between two, in log space: halfway in rate between 0.04 and 0.08 lands
    # above the geometric mean of 1.0 and 2.0 only if the interpolation is
    # linear in rate, which it is.
    resolved = resolve_sigma_scale(curve, 0.06)
    assert 1.0 < resolved < 2.0
    # Off both ends: clamp to the ladder rather than extrapolate.
    assert resolve_sigma_scale(curve, 0.005) == pytest.approx(0.25)
    assert resolve_sigma_scale(curve, 0.9) == pytest.approx(4.0)
    # And the limit binds even when the ladder reaches further.
    assert resolve_sigma_scale(curve, 0.9, limit=2.0) == pytest.approx(2.0)


def test_resolve_sigma_scale_declines_to_answer_when_it_cannot():
    """1.0 -- the learned sigma -- is the answer to every unanswerable case.

    The flat curve is the one that matters: it is what a collapsed sigma looks
    like, no multiplier can recover exploration from it, and saturating at the
    limit there would read as a controller working while exploration is zero.
    """

    from agents.dqn_market_free.training.updater import resolve_sigma_scale

    curve = {0.25: 0.01, 1.0: 0.04, 4.0: 0.20}
    assert resolve_sigma_scale(curve, 0.0) == 1.0
    assert resolve_sigma_scale(curve, -1.0) == 1.0
    assert resolve_sigma_scale({}, 0.05) == 1.0
    assert resolve_sigma_scale({0.25: 0.0, 1.0: 0.0, 4.0: 0.0}, 0.05) == 1.0


def test_the_sigma_floor_stops_the_controller_switching_exploration_off():
    """The `1 / limit` clamp is not a floor on exploration, and a cold start proved it.

    It bounds how far the resolver may depart from the learned sigma, which is a
    different thing. On an untrained network the flip-to-order rate is high
    because the *network* is random, not because sigma is large, so the controller
    reads over-exploration and turns sigma down at the one moment there is nothing
    yet to exploit. `market-free-landgate-v10` update 20 resolved exactly 0.125 --
    the `1 / 8` clamp, hit dead on -- with `mean_abs_q` at 0.046.
    """

    from agents.dqn_market_free.training.updater import _noisy, resolve_sigma_scale

    # A curve whose every rung over-explores against the target, which is what an
    # untrained network produces and what drives the resolver to its lower clamp.
    hot = {0.125: 0.31, 0.25: 0.42, 0.5: 0.55, 1.0: 0.68, 2.0: 0.80, 4.0: 0.89, 8.0: 0.95}

    assert resolve_sigma_scale(hot, 0.05, 8.0) == pytest.approx(0.125)
    assert resolve_sigma_scale(hot, 0.05, 8.0, 0.5) == pytest.approx(0.5)

    # Opt-in: 0.0 is byte-for-byte the old `1 / limit` behaviour.
    assert resolve_sigma_scale(hot, 0.05, 8.0, 0.0) == pytest.approx(0.125)
    assert resolve_sigma_scale(hot, 0.05, 4.0, 0.0) == pytest.approx(0.25)

    # A floor below the natural answer changes nothing.
    ordinary = {0.5: 0.02, 1.0: 0.05, 2.0: 0.17, 4.0: 0.44}
    assert resolve_sigma_scale(ordinary, 0.05, 8.0, 0.5) == pytest.approx(1.0)
    # ...and the floor never becomes a ceiling.
    assert resolve_sigma_scale(ordinary, 0.17, 8.0, 0.5) == pytest.approx(2.0)

    # The "cannot rescue sigma = 0" answer is unaffected: a flat curve still
    # returns the learned sigma rather than the floor.
    assert resolve_sigma_scale({0.25: 0.0, 1.0: 0.0, 4.0: 0.0}, 0.05, 8.0, 0.5) == 1.0
    assert resolve_sigma_scale({}, 0.05, 8.0, 0.5) == 1.0

    spec = _noisy({"noisy": True, "noisy_sigma_scale_floor": 0.5})
    assert spec.scale_floor == 0.5
    assert _noisy({"noisy": True}).scale_floor == 0.0


def test_the_setpoint_ceiling_still_bounds_a_flattening_curve():
    """`noisy_sigma_scale_limit` is the last stop, and it has to be.

    What a target rate costs in sigma depends on a curve that moves with the
    network, so a curve that flattens asks for far more sigma at an unchanged
    config. That is how v9 reached 3.159 and measured 49% of 212 seats bankrupt
    against 8-26% on the ordinary ladder, with zero land purchases -- the thing
    the widening existed to find. The limit is what stops a run drifting back into
    that regime without anyone changing a setting.
    """

    from agents.dqn_market_free.training.updater import _noisy, resolve_sigma_scale

    # The real curve from update 1362.
    curve = {
        0.125: 0.00301,
        0.25: 0.01054,
        0.5: 0.02711,
        1.0: 0.05572,
        2.0: 0.17018,
        4.0: 0.44277,
        8.0: 0.64006,
    }
    assert resolve_sigma_scale(curve, 0.35, 8.0) == pytest.approx(3.16, abs=0.02)
    assert resolve_sigma_scale(curve, 0.05, 8.0) < 1.5

    noisy = _noisy(
        {
            "noisy": True,
            "noisy_order_rate_target": 0.05,
            "noisy_sigma_scale_limit": 1.5,
        }
    )
    assert noisy.scale_limit == 1.5

    flat = {1.0: 0.01, 2.0: 0.02, 4.0: 0.05, 8.0: 0.30}
    assert resolve_sigma_scale(flat, 0.10, 8.0) > 4.0
    assert resolve_sigma_scale(flat, 0.10, noisy.scale_limit) == pytest.approx(1.5)


def test_the_two_controllers_disagree_on_a_collapsed_policy():
    """The v6 failure, stated as an assertion.

    `market-free-noisy-v6` promoted nothing in twelve updates with `trading_rate`
    0.0 on every gate. The cause was its own controller. A policy collapsed onto
    NOOP sits on a flat plateau where every action is near-tied, so a perturbation
    flips the argmax constantly while almost none of those flips buy anything: the
    measured argmax flip rate was 0.28-0.39 against a 0.05 setpoint, and
    `resolve_sigma_scale` duly divided sigma down, ending at
    `[0.33 .. 0.010]` across the eight actors. Not one played at the learned
    sigma. Exploration was throttled at precisely the moment it needed raising.

    The flip-to-order rate cannot be fooled the same way, because the near-ties
    that make the flip rate high are exactly what keeps this one low. Same curve
    shape, same resolver, opposite answer -- which is the whole fix.
    """

    from agents.dqn_market_free.training.updater import resolve_sigma_scale

    # Both measured on the same collapsed policy, over the same ladder.
    flip_curve = {0.25: 0.21, 0.5: 0.26, 1.0: 0.33, 2.0: 0.44, 4.0: 0.58}
    order_curve = {0.25: 0.002, 0.5: 0.006, 1.0: 0.011, 2.0: 0.031, 4.0: 0.092}

    throttled = resolve_sigma_scale(flip_curve, 0.05)
    restored = resolve_sigma_scale(order_curve, 0.05)

    assert throttled < 1.0, "the flip-rate controller is what suppressed v6"
    assert restored > 1.0, "the order-rate controller has to raise sigma here"
    # And the reading the flip-rate arm was chasing is not even in range of the
    # ladder it measured, which is why it clamps to the bottom rung.
    assert throttled == pytest.approx(0.25)


def test_every_noisy_attribute_the_updater_reads_exists():
    """A `noisy.X` that no longer exists is invisible until an update runs.

    `_noisy` and the config resolve fine without it, and every unit test that
    touches `NoisySpec` still passes -- the failure lives inside
    `FreeMarketUpdater.update`, which no test exercises end to end because it
    wants a database, an arena and a torch build.

    That is not hypothetical. Removing the idle burst left one
    `noisy.idle_trading_rate` behind at the sigma-resolution site, and the run
    collected 42 batches while every single update died with `AttributeError` --
    which looks from the outside like the mirror never playing, because with no
    checkpoint written `_self_play_pool` stays empty and only the base opponent
    is ever drawn.
    """

    import dataclasses
    import re

    from agents.dqn_market_free.training import updater as updater_module
    from agents.dqn_market_free.training.updater import NoisySpec

    source = Path(updater_module.__file__).read_text()
    referenced = set(re.findall(r"noisy\.([a-z_][a-z_0-9]*)", source))
    available = {field.name for field in dataclasses.fields(NoisySpec)}
    available |= {name for name in dir(NoisySpec) if not name.startswith("_")}

    missing = sorted(referenced - available)
    assert not missing, f"updater reads NoisySpec attributes that do not exist: {missing}"


def test_the_controller_reads_the_flip_to_order_rate():
    """One controller, and it is aimed at the NOOP attractor.

    There was a second until v13, targeting the argmax flip rate, and removing it
    is what the test above measures the cost of: on a collapsed policy every
    action is near-tied so flips are cheap, and that controller read a silent
    policy as exploring and divided sigma down. `market-free-noisy-v6` measured
    0.28-0.39 against a 0.05 setpoint and had every actor throttled to
    `[0.33 .. 0.010]` for it.
    """

    from agents.dqn_market_free.training.updater import _noisy

    on = _noisy({"noisy": True, "noisy_order_rate_target": 0.05})
    assert on.controlled

    # No target is not a controller aimed at zero, it is no controller.
    assert not _noisy({"noisy": True}).controlled
    # And NoisyNet off disables it regardless of what the setpoint says.
    assert not _noisy({"noisy": False, "noisy_order_rate_target": 0.05}).controlled


def test_resolve_sigma_scale_survives_a_saturating_order_curve():
    """Flip-to-order saturates where the flip rate does not.

    Once the perturbation always talks the policy out of NOOP the rate is 1.0 and
    cannot rise further, so the curve has a flat tail that the strictly-increasing
    flip rate never shows. The resolver must read through it rather than raise:
    a target inside the flat region resolves to the smallest scale that reaches
    it, and one above the whole ladder clamps to the top rung.
    """

    from agents.dqn_market_free.training.updater import resolve_sigma_scale

    curve = {0.25: 0.05, 0.5: 0.30, 1.0: 0.80, 2.0: 1.0, 4.0: 1.0}

    assert resolve_sigma_scale(curve, 0.80) == pytest.approx(1.0)
    assert resolve_sigma_scale(curve, 1.0) == pytest.approx(2.0)
    # Unreachable target: take the top of the ladder, not an extrapolation.
    assert resolve_sigma_scale(curve, 1.5) == pytest.approx(4.0)

    # A curve that is flat *everywhere* is still the collapsed-sigma case, and
    # still has to come back as the learned sigma rather than as the limit.
    assert resolve_sigma_scale({0.25: 0.0, 1.0: 0.0, 4.0: 0.0}, 0.05) == 1.0


def test_the_resolved_scale_crosses_from_train_to_collect_through_the_database():
    """`collect` and `train` are different rounds, so the scale goes via SQL.

    The default is the load-bearing half: a cold start, a controller that has
    never run, and a lineage whose updates predate the key all have to come back
    as 1.0 -- the learned sigma -- rather than as a missing value that later
    arithmetic turns into no exploration at all.
    """

    from agents.dqn_market_free.training.updater import collection_sigma_scale

    class Reader:
        def __init__(self, row):
            self.row = row
            self.sql = None

        def fetchone(self, sql, parameters=()):
            self.sql = sql
            assert parameters == ["model"]
            return self.row

    reader = Reader((2.5,))
    assert collection_sigma_scale(reader, "model") == pytest.approx(2.5)
    # Newest first, and rows that never ran the probe are excluded rather than
    # read as a null scale.
    assert "ORDER BY started_at DESC" in reader.sql
    assert "IS NOT NULL" in reader.sql

    assert collection_sigma_scale(Reader(None), "model") == 1.0
    assert collection_sigma_scale(Reader((None,)), "model") == 1.0


def test_the_controller_is_off_unless_a_target_is_set():
    """Off is the default, and off must leave collection byte-identical."""

    from agents.dqn_market_free.training.updater import _noisy

    assert not _noisy({"noisy": True}).controlled
    assert not _noisy({"noisy": False, "noisy_order_rate_target": 0.05}).controlled
    assert _noisy({"noisy": True, "noisy_order_rate_target": 0.05}).controlled

    # With the controller off the base is 1.0, so the ladder the actors play is
    # exactly what it was before the controller existed.
    base = 1.0
    assert [base * scale for scale in actor_sigma_scales(8, 32.0)] == actor_sigma_scales(8, 32.0)


def test_sigma_is_trainable():
    """Whatever else happens, sigma must be reachable by the TD gradient."""

    module = net_module.build(noisy="head", sigma0=0.5)
    head = [layer for layer in module if isinstance(layer, net_module.NoisyLinear)][0]
    before = head.weight_sigma.detach().clone()

    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(64, STATE_DIM, generator=generator)
    targets = torch.randn(64, ACTION_COUNT, generator=generator) * 100.0
    for _ in range(20):
        net_module.reset_noise(module)
        loss = torch.nn.SmoothL1Loss()(module(features), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert not torch.equal(head.weight_sigma.detach(), before)


def test_export_refuses_a_layer_it_cannot_write():
    """An unrecognised parameterised layer is a stale exporter, not a layer to skip.

    The old `isinstance(layer, nn.Linear)` filter dropped it in silence, so
    `to_numpy` returned a short network, `has_inert_hidden_trunk` waved it
    through on the strength of a nonzero first weight, and `save` wrote it. The
    shape error then surfaced inside an arena worker one update later.
    """

    module = torch.nn.Sequential(
        torch.nn.Linear(STATE_DIM, 8),
        torch.nn.BatchNorm1d(8),
        torch.nn.Linear(8, ACTION_COUNT),
    )
    with pytest.raises(TypeError, match="cannot export"):
        net_module.to_numpy(module)


def test_loading_a_checkpoint_of_the_wrong_depth_is_refused():
    """`zip` would have truncated, leaving the last layer at torch's init."""

    shallow = OrderQNetwork.zeros(hidden=(128,))
    with pytest.raises(ValueError, match="parameterised layers"):
        net_module.from_numpy(shallow, net_module.build())


def test_saved_weights_reject_a_stale_schema_at_both_ends(tmp_path):
    """A checkpoint from the ten-slot agent must not be read as a shape match."""

    narrow = tmp_path / "narrow.npz"
    OrderQNetwork.zeros(state_dim=STATE_DIM - 1).save(narrow)
    with pytest.raises(ValueError, match="does not match schema"):
        OrderQNetwork.load(narrow)

    # The old candidate-scoring network had a single output column.
    scalar_head = tmp_path / "scalar_head.npz"
    OrderQNetwork.zeros(actions=1).save(scalar_head)
    with pytest.raises(ValueError, match="does not match schema"):
        OrderQNetwork.load(scalar_head)


# --------------------------------------------------------------------------
# Exploration
# --------------------------------------------------------------------------


def test_greedy_choice_takes_the_argmax():
    state = np.zeros(STATE_DIM, dtype=np.float32)
    wanted = order_index("BUY_SEED", "WHEAT", 1)
    assert wanted is not None

    class PeakedNetwork:
        def __call__(self, features):
            scores = np.zeros(ACTION_COUNT, dtype=np.float32)
            scores[wanted] = 10.0
            return scores

    decision = choose_order(state, PeakedNetwork())
    assert decision.chosen == wanted
    assert decision.wire() == [["BUY_SEED", "WHEAT", 1]]


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


def _synthetic_record(
    decisions: int,
    final_money: float,
    seed: int = 0,
    chosen: int | None = None,
    forbidden: list[int] | None = None,
    episode_return: float | None = None,
) -> EpisodeRecord:
    rng = np.random.default_rng(seed)
    steps = decisions
    order_chosen = (
        rng.integers(0, ACTION_COUNT, size=decisions).astype(np.int16)
        if chosen is None
        else np.full(decisions, chosen, dtype=np.int16)
    )
    order_mask = np.ones((decisions, ACTION_COUNT), dtype=np.bool_)
    if forbidden:
        order_mask[:, forbidden] = False
    # `episode_return` pins `reward.sum()`, which is what the return-weighted
    # draw ranks on. Spread flat rather than dumped on the terminal step so an
    # n-step window anywhere in the episode still carries some of it.
    reward = (
        rng.standard_normal(steps).astype(np.float32)
        if episode_return is None
        else np.full(steps, episode_return / steps, dtype=np.float32)
    )
    return EpisodeRecord(
        reward=reward,
        raw_reward=rng.standard_normal(steps).astype(np.float32),
        money=np.asarray([*np.zeros(steps), final_money], dtype=np.float32),
        order_state=rng.standard_normal((decisions, STATE_DIM)).astype(np.float32),
        order_chosen=order_chosen,
        order_step=np.arange(decisions, dtype=np.int16),
        order_mask=order_mask,
    )


def test_demo_fraction_puts_solvent_data_in_every_batch():
    """One of the two mechanisms against a buffer that is nearly all bankruptcy.

    Solvent trajectories are rare enough that an unweighted draw over the
    agent's own episodes is almost entirely bankruptcies, and the only policy
    consistent with that data is to do nothing. Drawing a fixed share of every
    minibatch from the pinned demonstrations puts solvent transitions in by
    construction -- but only when something is pinned, which is why
    `priority_alpha` covers the case where nothing is. The demonstration share
    stays uniform inside itself either way; the ranking applies to the own share.
    """

    wheat = order_index("BUY_SEED", "WHEAT", 1)
    buffer = ReplayBuffer(capacity=64, seed=0, n_step=2)
    buffer.add([_synthetic_record(40, 10.0, seed=index, chosen=0) for index in range(20)])
    buffer.add([_synthetic_record(40, 120_000.0, seed=99, chosen=wheat)], pinned=True)

    batch = buffer.sample(64, demo_fraction=0.25)
    from_demos = int((batch.chosen == wheat).sum())
    assert from_demos == 16

    # Zero is honoured, and so is "no demonstrations pinned at all".
    assert int((buffer.sample(64, demo_fraction=0.0).chosen == wheat).sum()) == 0
    own_only = ReplayBuffer(capacity=8, seed=0, n_step=2)
    own_only.add([_synthetic_record(40, 10.0, seed=1, chosen=0)])
    assert int((own_only.sample(8, demo_fraction=0.5).chosen != 0).sum()) == 0


def test_sampling_falls_back_to_demonstrations_before_the_first_rollout():
    """The first update runs before any self-collected episode exists."""

    wheat = order_index("BUY_SEED", "WHEAT", 1)
    buffer = ReplayBuffer(capacity=8, seed=0, n_step=2)
    buffer.add([_synthetic_record(40, 120_000.0, seed=3, chosen=wheat)], pinned=True)

    batch = buffer.sample(16, demo_fraction=0.25)
    assert (batch.chosen == wheat).all()


def _priority_buffer(returns: list[float], **kwargs) -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=len(returns) * 2, seed=0, n_step=2, **kwargs)
    buffer.add(
        [
            _synthetic_record(12, 10.0, seed=index, episode_return=value)
            for index, value in enumerate(returns)
        ]
    )
    return buffer


def test_alpha_zero_draws_the_batch_the_uniform_buffer_drew():
    """The determinism guard, and the reason this landed inert first.

    Every lineage on disk was trained against the uniform draw. If adding the
    return weighting perturbs the bit stream at `priority_alpha = 0`, every
    resumed run silently starts drawing different episodes than it would have,
    and nothing anywhere reports it.

    So this asserts the *stream*, not the values: the same picks in the same
    order, and the same bit-generator state afterwards. The second half is what
    makes it a stream test -- a batch can match while the generator has been
    advanced a different number of times, and the divergence only shows up on
    the next call.
    """

    # Each own episode emits its own action, so `batch.chosen` names the episode
    # every row was drawn from and the picks are readable straight off the batch.
    wheat = order_index("BUY_SEED", "WHEAT", 1)
    buffer = ReplayBuffer(capacity=64, seed=17, n_step=2)
    buffer.add(
        [_synthetic_record(40, 10.0, seed=index, chosen=index) for index in range(20)]
    )
    buffer.add([_synthetic_record(40, 120_000.0, seed=99, chosen=wheat)], pinned=True)
    assert buffer.priority_alpha == 0.0, "the code default must be the uniform draw"

    batch = buffer.sample(64, demo_fraction=0.25)

    # The pre-change draw, by hand: 48 own picks, then 16 demo picks, then one
    # decision pick per row. Same generator, same seed, no buffer involved.
    reference = np.random.default_rng(17)
    own_rows, demo_rows = 48, 16
    expected = [int(reference.integers(0, 20)) for _ in range(own_rows)]
    for _ in range(demo_rows):
        reference.integers(0, 1)
    for _ in range(own_rows + demo_rows):
        reference.integers(0, 40)

    assert [int(value) for value in batch.chosen[:own_rows]] == expected
    assert [int(value) for value in batch.chosen[own_rows:]] == [wheat] * demo_rows
    assert buffer.rng.bit_generator.state == reference.bit_generator.state


def test_the_weight_is_all_ones_without_prioritisation():
    """An unprioritised draw has no bias, so it must carry no correction."""

    buffer = _priority_buffer([1.0, -2.0, 0.5, 3.0])
    batch = buffer.sample(16)

    assert batch.weight.shape == (16,)
    assert batch.weight.dtype == np.float32
    assert (batch.weight == 1.0).all()


def test_a_negative_return_is_a_legal_priority():
    """Why the scheme ranks rather than exponentiating the return.

    Roughly half this run's own episodes finish on a negative shaped return --
    measured, the 25th percentile is -1.02 -- so `return ** alpha` is not a real
    number on most of the buffer. A rank is defined regardless of sign, and every
    episode keeps a strictly positive probability, which is what stops the
    reservoir segment from being silently deleted.
    """

    returns = [-2.0, -0.5, 0.0, 0.25, 1.0]
    buffer = _priority_buffer(returns, priority_alpha=0.7)
    pool, pool_returns, probabilities, _weights = buffer._own_priorities()

    assert len(pool) == len(returns)
    assert np.isfinite(probabilities).all()
    assert (probabilities > 0.0).all()
    assert probabilities.sum() == pytest.approx(1.0)
    # Higher return, higher probability -- monotone, with no ties here.
    by_return = np.argsort(pool_returns)
    ordered = probabilities[by_return]
    assert (np.diff(ordered) > 0.0).all()


@pytest.mark.parametrize("alpha", [0.5, 0.7, 1.0])
@pytest.mark.parametrize("beta", [0.0, 0.4, 0.5, 1.0])
def test_the_importance_weight_never_exceeds_one(alpha, beta):
    """Max-normalised, so the correction can only ever shrink a gradient.

    `clip_grad_norm_` clips the whole batch's norm, so it cannot see a single row
    multiplying itself up. Normalising by the largest weight rather than the mean
    is what makes that unreachable: the weighted loss is bounded above by the
    unweighted one for any alpha and beta.
    """

    buffer = _priority_buffer(
        [-2.0, -0.5, 0.0, 0.25, 1.0, 2.5], priority_alpha=alpha, priority_beta=beta
    )
    _pool, _returns, probabilities, weights = buffer._own_priorities()

    assert weights.max() == pytest.approx(1.0)
    assert weights.min() > 0.0
    assert (weights <= 1.0 + 1e-6).all()
    # The rarer the draw, the larger the correction.
    order = np.argsort(probabilities)
    assert (np.diff(weights[order]) <= 1e-6).all()


def test_beta_one_is_an_unbiased_draw_and_therefore_pointless():
    """Executable form of why beta must stay below 1.

    At beta 1 the correction is exact: `P(i) * w(i)` is flat, so the weighted
    gradient equals the uniform one in expectation and prioritisation buys
    nothing but variance. The bias toward the agent's own best episodes is the
    entire mechanism, so beta is a bias dial rather than a correctness knob
    climbing toward 1 -- and nobody should later "fix" it by annealing there.
    """

    buffer = _priority_buffer(
        [-2.0, -0.5, 0.0, 0.25, 1.0], priority_alpha=0.7, priority_beta=1.0
    )
    _pool, _returns, probabilities, weights = buffer._own_priorities()
    effective = probabilities * weights
    assert effective.max() == pytest.approx(effective.min())

    # Below 1 it is not flat, and it leans toward the better episodes.
    biased = _priority_buffer(
        [-2.0, -0.5, 0.0, 0.25, 1.0], priority_alpha=0.7, priority_beta=0.5
    )
    _pool, returns, probabilities, weights = biased._own_priorities()
    effective = probabilities * weights
    assert effective[int(np.argmax(returns))] > effective[int(np.argmin(returns))]


def test_the_priority_vector_follows_the_buffer():
    """The stale-cache trap, which fails silently and forever.

    The third case is the one that matters. At steady state the FIFO length is
    constant and `_offer_to_reservoir` replaces a reservoir slot *in place*, so a
    cache keyed on `(len(recent), len(reservoir))` would never invalidate: the
    run would train on the first update's priorities for the rest of its life
    with nothing raised and no metric moving.
    """

    # (i) `add` after the cache is warm.
    buffer = _priority_buffer([0.0, 0.1, 0.2], priority_alpha=0.7)
    buffer.sample(8)
    assert buffer.priority_stats()["own_max_return"] == pytest.approx(0.2, abs=1e-5)
    buffer.add([_synthetic_record(12, 10.0, seed=91, episode_return=9.0)])
    assert buffer.priority_stats()["own_max_return"] == pytest.approx(9.0, abs=1e-4)

    # (ii) `load_planned` after the cache is warm.
    buffer.sample(8)
    buffer.load_planned([0], [], 0, {0: _synthetic_record(12, 10.0, seed=5, episode_return=-4.0)})
    assert buffer.priority_stats()["own_max_return"] == pytest.approx(-4.0, abs=1e-4)

    # (iii) in-place reservoir replacement, with both lengths unchanged.
    ring = ReplayBuffer(capacity=10, seed=3, n_step=2, reservoir_fraction=0.5)
    ring.add(
        [
            _synthetic_record(12, 10.0, seed=index, episode_return=0.0)
            for index in range(40)
        ]
    )
    ring.sample(8)
    lengths = (len(ring.recent), len(ring.reservoir))
    before = ring.priority_stats()["own_max_return"]
    for index in range(40):
        ring.add([_synthetic_record(12, 10.0, seed=200 + index, episode_return=7.0)])
    assert (len(ring.recent), len(ring.reservoir)) == lengths, "lengths must not move"
    assert ring.priority_stats()["own_max_return"] > before


def test_the_draw_prefers_the_agents_own_best_episodes():
    """End to end: the mechanism has to actually move the data it trains on.

    `sampled_mean_return` is the mean under the sampling law, so the empirical
    draw has to converge to it. Both have to sit above the buffer mean, which is
    the whole point -- and equality with the buffer mean is the inert reading
    that `demo_fraction` produced for generations without anyone noticing.
    """

    # One episode per action id, so `batch.chosen` names the episode drawn, and
    # returns rising with the id so the best episodes are identifiable.
    returns = [float(index) for index in range(40)]
    buffer = ReplayBuffer(capacity=80, seed=5, n_step=2, priority_alpha=0.7)
    buffer.add(
        [
            _synthetic_record(12, 10.0, seed=index, chosen=index, episode_return=value)
            for index, value in enumerate(returns)
        ]
    )

    stats = buffer.priority_stats()
    assert stats["own_mean_return"] == pytest.approx(np.mean(returns), abs=1e-3)
    assert stats["sampled_mean_return"] > stats["own_mean_return"]
    # The best decile takes well more than the 0.1 a uniform draw would give it.
    assert stats["priority_top_decile_share"] > 0.2

    drawn = np.concatenate([buffer.sample(256).chosen for _ in range(20)])
    empirical = float(np.mean([returns[int(index)] for index in drawn]))
    assert empirical == pytest.approx(stats["sampled_mean_return"], abs=1.0)
    assert empirical > stats["own_mean_return"]


def test_demonstration_rows_are_never_reweighted():
    """The correction undoes a bias that exists only inside the own pool.

    The pinned pool is still drawn uniformly, so there is nothing there to
    correct and a weight other than 1.0 would be correcting a bias that does not
    exist -- and would partly undo the guarantee the demonstration share exists
    to provide.
    """

    wheat = order_index("BUY_SEED", "WHEAT", 1)
    buffer = ReplayBuffer(capacity=64, seed=11, n_step=2, priority_alpha=0.7)
    buffer.add(
        [
            _synthetic_record(12, 10.0, seed=index, chosen=0, episode_return=float(index))
            for index in range(20)
        ]
    )
    buffer.add([_synthetic_record(12, 120_000.0, seed=99, chosen=wheat)], pinned=True)

    batch = buffer.sample(64, demo_fraction=0.25)
    assert (batch.weight[batch.chosen == wheat] == 1.0).all()
    # And the own rows really are corrected, or the assertion above is vacuous.
    assert (batch.weight[batch.chosen == 0] < 1.0).any()


def test_the_flip_rate_probe_sees_the_states_the_actor_visits():
    """`prioritized=False` forces the uniform draw whatever the config says.

    The probe resolves NoisyNet sigma against the states the actor really
    visits. Handed a return-weighted draw it would resolve the controller
    against a distribution no actor ever sees, and the failure is invisible --
    the probe reports a plausible flip rate either way.
    """

    def build(**kwargs):
        buffer = ReplayBuffer(capacity=64, seed=23, n_step=2, **kwargs)
        buffer.add(
            [
                _synthetic_record(
                    12, 10.0, seed=index, chosen=index, episode_return=float(index)
                )
                for index in range(20)
            ]
        )
        return buffer

    uniform = build().sample(64, demo_fraction=0.0)
    forced = build(priority_alpha=0.9).sample(64, demo_fraction=0.0, prioritized=False)
    assert (forced.chosen == uniform.chosen).all()
    assert (forced.weight == 1.0).all()

    # The same buffer left to prioritise does something else, so the kwarg is
    # load-bearing rather than a no-op on this data.
    ranked = build(priority_alpha=0.9).sample(64, demo_fraction=0.0)
    assert not (ranked.chosen == uniform.chosen).all()


def test_a_weighted_step_cannot_be_larger_than_an_unweighted_one():
    """Why the weights are normalised by their max rather than their mean.

    `clip_grad_norm_` clips the whole batch's norm, so it cannot see one row
    multiplying itself up. Max-normalisation makes that unreachable at the
    source: every weight is in (0, 1], so the weighted loss is bounded above by
    the unweighted one and the effective step can only ever shrink. A mean
    normalisation, or none at all, would put weight above 1 on exactly the rows
    that are rarest -- inflating the gradient when the batch is least
    representative.
    """

    generator = torch.Generator().manual_seed(4)
    predicted = torch.randn(64, generator=generator, requires_grad=True)
    goal = torch.randn(64, generator=generator)
    weight = torch.rand(64, generator=generator).clamp(min=1e-3)
    weight = weight / weight.max()

    per_row = torch.nn.SmoothL1Loss(reduction="none")(predicted, goal)
    assert per_row.shape == (64,)
    weighted = (per_row * weight).mean()
    unweighted = torch.nn.SmoothL1Loss()(predicted, goal)
    assert float(weighted.detach()) <= float(unweighted.detach())

    def grad_norm(loss):
        (gradient,) = torch.autograd.grad(loss, predicted, retain_graph=True)
        return float(gradient.norm())

    assert grad_norm(weighted) <= grad_norm(unweighted)

    # All-ones is exactly the unweighted objective, which is what makes
    # `priority_alpha = 0` a no-op in the loss as well as in the draw.
    assert float((per_row * torch.ones(64)).mean().detach()) == pytest.approx(
        float(unweighted.detach())
    )


def test_fill_buffer_only_opens_batches_that_still_own_an_episode(tmp_path: Path):
    """The point of the planning pass: aged-out artifacts are never read."""

    from agents.dqn_market_free.training.collect import save_episode_batch

    paths = []
    for batch in range(12):
        records = [
            _synthetic_record(12, 100.0 * (batch * 5 + index), seed=batch * 5 + index)
            for index in range(5)
        ]
        paths.append(save_episode_batch(records, tmp_path / f"batch-{batch:03d}.npz"))

    batches = [
        SimpleNamespace(
            source_kind=SOURCE_KIND,
            artifact_path=str(path),
            artifact_sha256=None,
            sample_count=5,
        )
        for path in paths
    ]
    # A non-rollout batch must be ignored without disturbing the numbering.
    batches.insert(3, SimpleNamespace(
        source_kind="model_evaluation", artifact_path=None, artifact_sha256=None, sample_count=1
    ))

    opened: list[str] = []
    real_load = updater_module.load_episode_batch

    def counting_load(path):
        opened.append(str(path))
        return real_load(path)

    buffer = ReplayBuffer(capacity=20, seed=3, n_step=2, reservoir_fraction=0.5)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(updater_module, "load_episode_batch", counting_load)
        skipped_evicted = updater_module.fill_buffer(buffer, batches)

    assert skipped_evicted > 0
    assert len(opened) + skipped_evicted == 12
    assert len(opened) < 12, "the whole history must not be read to fill a 20-episode buffer"
    assert len(buffer.recent) + len(buffer.reservoir) == 20


def test_planned_intake_reproduces_the_record_by_record_buffer():
    """Planning must be an optimisation, not a different buffer.

    `plan_intake` decides survival from `sample_count` alone so the updater can
    skip reading batches that contribute nothing. It has to take the same
    Algorithm R draws in the same order as `add`, or the reservoir stops being a
    uniform sample of history.
    """

    batches = [
        [_synthetic_record(12, 100.0 * (batch * 10 + index), seed=batch * 10 + index) for index in range(10)]
        for batch in range(12)
    ]

    naive = ReplayBuffer(capacity=40, seed=7, n_step=2, reservoir_fraction=0.5)
    for records in batches:
        naive.add(records)

    planned = ReplayBuffer(capacity=40, seed=7, n_step=2, reservoir_fraction=0.5)
    recent_ids, reservoir_ids, evicted_seen = planned.plan_intake([len(item) for item in batches])
    flat = [record for records in batches for record in records]
    wanted = set(recent_ids) | set(reservoir_ids)
    planned.load_planned(
        recent_ids, reservoir_ids, evicted_seen, {key: flat[key] for key in wanted}
    )

    assert planned.evicted_seen == naive.evicted_seen
    assert [record.money[-1] for record in planned.recent] == [
        record.money[-1] for record in naive.recent
    ]
    assert [record.money[-1] for record in planned.reservoir] == [
        record.money[-1] for record in naive.reservoir
    ]
    # The whole point: most batches never need to be opened.
    assert len(wanted) == 40
    assert len(wanted) < len(flat)


def test_pinned_demonstrations_survive_eviction():
    buffer = ReplayBuffer(capacity=4, seed=0, reservoir_fraction=0.0)
    buffer.add([_synthetic_record(8, 90_000.0, seed=7)], pinned=True)
    buffer.add([_synthetic_record(8, 1.0, seed=index) for index in range(40)])
    assert len(buffer.pinned) == 1
    assert any(record.money[-1] == 90_000.0 for record in buffer.episodes)


def test_behavior_clone_step_moves_the_head_toward_the_expert_action():
    """A non-degenerate classification target, which the ten-slot version was not.

    Cloning ten heads whose targets were 88% NOOP had silence as its
    minimum-loss answer. One head over one recorded action does not.
    """

    wheat = order_index("BUY_SEED", "WHEAT", 1)
    assert wheat is not None
    buffer = ReplayBuffer(capacity=4, seed=0)
    buffer.add([_synthetic_record(8, 90_000.0, seed=7, chosen=wheat)], pinned=True)

    module = net_module.from_numpy(
        initial_network({}), net_module.build()
    )
    before = [parameter.detach().clone() for parameter in module.parameters()]
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-2)
    losses = [
        _behavior_clone_step(module, optimizer, buffer, 8, torch.device("cpu"))
        for _ in range(30)
    ]

    assert all(np.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
    assert any(
        not torch.equal(parameter.detach(), original)
        for parameter, original in zip(module.parameters(), before)
    )
    # The point of cloning: the greedy action becomes the demonstrated one.
    state = np.zeros(STATE_DIM, dtype=np.float32)
    assert choose_order(state, net_module.to_numpy(module)).chosen == wheat


def test_demonstration_runs_enable_cloning_without_a_manifest_override():
    assert _behavior_clone_steps_per_update({"demonstration_episodes": 200}) == 100
    assert _behavior_clone_steps_per_update({"demonstration_episodes": 0}) == 0
    assert (
        _behavior_clone_steps_per_update(
            {"demonstration_episodes": 200, "behavior_clone_steps_per_update": 25}
        )
        == 25
    )


def test_batch_assembly_matches_the_declared_widths():
    """The shapes the TD step gathers against, with no candidate block at all."""

    buffer = ReplayBuffer(capacity=16, seed=0, n_step=2)
    buffer.add([_synthetic_record(30, 5_000.0, seed=index) for index in range(4)])
    batch = buffer.sample(8)

    assert batch.state.shape == (8, STATE_DIM)
    assert batch.next_state.shape == (8, STATE_DIM)
    assert batch.chosen.shape == (8,)
    assert batch.reward.shape == batch.discount.shape == (8,)
    assert batch.chosen.dtype == np.int64
    assert (batch.chosen < ACTION_COUNT).all()
    # A gather against the action head has to accept these directly.
    module = net_module.build()
    predicted = module(torch.from_numpy(batch.state)).gather(
        1, torch.from_numpy(batch.chosen).unsqueeze(1)
    )
    assert predicted.shape == (8, 1)


# --------------------------------------------------------------------------
# Demonstrations
# --------------------------------------------------------------------------


def _fake_replay(turns: int | None = 5) -> dict:
    """A replay in the environment's own step/seat shape; `None` keeps it whole."""

    from kaggle_environments import make

    env = make("kaggriculture", debug=True)
    env.run([str(SNAPSHOT_PATH), str(SNAPSHOT_PATH)])
    replay = env.toJSON()
    if turns is not None:
        replay["steps"] = replay["steps"][: turns + 1]
    return replay


@pytest.mark.slow
def test_cloned_history_block_matches_the_live_agent():
    """The history block has two constructions, and they must agree.

    Everything else in the state is a function of one observation, so a live
    rollout and a replay conversion cannot disagree about it. The history block
    is a function of the whole episode so far, built by `main.py` as the agent
    plays and rebuilt by `clone.py` walking the stored replay -- and if the two
    push at different points in their loops, every demonstration carries a state
    the live policy will never see, silently.

    The comparison starts one turn after the lag fills. `steps[0]` is the
    environment's own reset rather than a turn the agent played, so the first
    difference the clone can compute is one turn behind the live agent's.
    """

    from kaggle_environments import make

    from agents.dqn_market_free.market_state import HISTORY_LAG, MARKET_STATE_DIM

    recorder = Recorder()
    learner = MarketFreeAgent(
        # A randomly initialized head, not `quiet_start`: this needs the agent to
        # actually trade so the history block has something in it, and epsilon --
        # which used to force that -- was removed at v13. A random head's argmax
        # varies across states on its own.
        OrderQNetwork.random(seed=3),
        rng=np.random.default_rng(0),
        recorder=recorder,
    )
    env = make("kaggriculture", debug=True)
    env.run([learner, str(SNAPSHOT_PATH)])
    replay = env.toJSON()

    cloned, _stats = record_from_replay(replay, seat=0)
    assert cloned is not None
    live = np.stack([decision.state for decision in recorder.decisions])
    assert cloned.order_state.shape == live.shape

    block = MARKET_STATE_DIM - 30
    start = HISTORY_LAG + 1
    np.testing.assert_allclose(
        cloned.order_state[start:, block:], live[start:, block:], atol=1e-6
    )
    # And the block is not trivially equal because it is all zeros.
    assert np.abs(live[start:, block:]).sum() > 0.0


@pytest.mark.slow
def test_demonstrations_convert_without_losing_orders_to_the_action_list():
    """Every order a real agent placed must be nameable, or the clone learns NOOP.

    Two different losses are possible and only one is acceptable.
    `orders_unrepresentable` means the action list cannot express the order at
    all -- a schema problem, and it must be zero. `orders_over_budget` is the
    demonstrator's second and later orders on one turn, which this agent
    structurally cannot emit; that loss is expected and bounded.
    """

    # Measured whole: the opening turns are the burstiest part of an episode,
    # so a truncated replay understates coverage rather than sampling it.
    replay = _fake_replay(turns=None)
    record, stats = record_from_replay(replay, seat=0)
    assert record is not None
    assert record.decisions == len(replay["steps"]) - 1
    assert record.order_chosen.shape == (record.decisions,)

    assert stats.orders_unrepresentable == 0
    assert stats.orders_mapped + stats.orders_over_budget == stats.orders_seen

    # The standing check on the feasibility mask. The donor is solvent and the
    # simulator accepted every order it placed, so those orders are feasible by
    # demonstration. Anything the mask forbids here is the mask being wrong, and
    # over-restriction is the dangerous direction -- it removes orders the game
    # would have taken, silently.
    assert stats.orders_masked_infeasible == 0, (
        f"mask forbade {stats.orders_masked_infeasible} orders the donor actually placed"
    )
    assert record.order_mask.shape == (record.decisions, ACTION_COUNT)
    for index in range(record.decisions):
        assert record.order_mask[index][int(record.order_chosen[index])]
    # Measured on a full donor episode: 908 orders over 719 turns, of which 417
    # survive. The donor averages 1.2 orders a turn but places them in bursts --
    # 358 turns are silent and the day boundaries carry up to ten at once -- so
    # roughly half is the ceiling for one action per turn, not a conversion bug.
    # Guard the floor, so a regression that silently halves it again is visible.
    assert stats.mapped_fraction > 0.4


@pytest.mark.slow
def test_skipped_steps_keep_order_step_aligned_with_the_reward_stream():
    """`order_step` indexes this record's compacted rewards, not the replay.

    A step with no usable `farms` appends no money and therefore no reward, so
    a raw replay index would slide the n-step reward window off the decision it
    belongs to -- silently, because `replay.sample` clamps to `record.steps`.
    """

    replay = _fake_replay(turns=20)
    # Blank one mid-episode observation. The loop reads `steps[i - 1]`, so this
    # is the step the conversion has to skip.
    replay["steps"][5][0]["observation"]["farms"] = []

    record, _ = record_from_replay(replay, seat=0)
    assert record is not None
    assert record.decisions == len(replay["steps"]) - 2
    assert np.array_equal(record.order_step, np.arange(record.decisions))
    # The window the buffer will read must stay inside the reward stream.
    assert int(record.order_step[-1]) < record.steps


@pytest.mark.slow
def test_rollouts_index_order_step_the_same_way_the_clone_does():
    """The live recorder owes the buffer the same index the clone writes.

    Both record kinds go into one `ReplayBuffer` and `sample` walks
    `reward[order_step[d]:...]` without knowing which produced them, so a
    recorder that stored the simulator's turn number instead would slide the
    n-step window for every decision after a skipped turn -- and the
    `min(end_step, record.steps)` clamp would hide it.
    """

    replay = _fake_replay(turns=20)
    observations = [step[0]["observation"] for step in replay["steps"]]
    # The agent returns a pass on an observation with no usable `farms`, without
    # recording money, potential, or the decision. That is where the simulator's
    # turn number and the reward index part ways.
    observations[5] = {**observations[5], "farms": []}

    recorder = Recorder()
    agent = MarketFreeAgent(OrderQNetwork.quiet_start(seed=0), recorder=recorder)
    for observation in observations:
        agent(observation, replay.get("configuration"))

    record = recorder.finish(1_234.0, 1_000.0, 0.999)
    assert record.decisions == len(observations) - 1
    assert np.array_equal(record.order_step, np.arange(record.decisions))
    assert int(record.order_step[-1]) < record.steps


@pytest.mark.slow
def test_demonstration_reward_matches_a_live_rollout_shape():
    replay = _fake_replay(turns=60)
    record, _ = record_from_replay(replay, seat=0)
    assert record.reward.shape[0] == record.steps
    assert np.isfinite(record.reward).all()


# --------------------------------------------------------------------------
# Inference path
# --------------------------------------------------------------------------


def _called_donor_attributes(path: Path) -> set[str]:
    """Every `donor.<name>` referenced in executable code, comments excluded."""

    import ast

    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "donor":
                found.add(node.attr)
    return found


def _imported_modules(path: Path) -> set[str]:
    import ast

    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_no_donor_market_code_runs():
    """The premise of the agent: no donor market judgement anywhere.

    Parsed rather than grepped, so the prose in the module docstring -- which
    necessarily names the things being removed -- cannot pass or fail it.
    """

    import agents.dqn_market_free.main as module

    path = Path(module.__file__)
    assert not any("market_exec" in name for name in _imported_modules(path))

    called = _called_donor_attributes(path)
    # Field plan, config reads and the episode-boundary latch reset. Nothing
    # that decides a market order.
    assert called <= {
        "_role_plan",
        "_unit_actions",
        "_cfg",
        "_SIGNATURE_LAST_STEP",
        "_SIGNATURE_ACTIVE",
        "_SCHEDULE_WHEAT_REQUESTED",
    }, f"unexpected donor calls: {sorted(called)}"
    assert "_market_actions" not in called
    assert "_schedule_market_adjustment" not in called


def test_missing_weights_fall_back_to_a_silent_network(monkeypatch, tmp_path):
    monkeypatch.setattr("agents.dqn_market_free.main.WEIGHTS_PATH", tmp_path / "absent.npz")
    network = load_network(None)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    assert choose_order(state, network).wire() == []


@pytest.mark.slow
def test_agent_completes_an_episode_against_the_donor():
    from kaggle_environments import make

    env = make("kaggriculture", debug=True)
    agent = MarketFreeAgent(initial_network({}))
    env.run([agent, str(SNAPSHOT_PATH)])
    final = env.steps[-1][0]["observation"]
    farms = final["farms"]
    # The cold start trades nothing, so it ends on exactly the starting float.
    # That is the honest cold start for an unscaffolded market: no seed gets
    # bought, so the donor field policy has nothing to plant.
    assert farms[0]["money"] == pytest.approx(3000.0, abs=1.0)
    assert len(env.steps) == 720


# --------------------------------------------------------------------------
# Framework seams
#
# The three objects the lab actually reaches for. None of them was covered
# before, and each is a place where a signature change fails at run time rather
# than at import: `train.trainer` is resolved by string, `finalize_episode` is
# duck-typed by `sim.runner`, and `evaluate_episode` is resolved through
# `main.__getattr__`.
# --------------------------------------------------------------------------


def test_trainer_singleton_exposes_the_declared_hooks():
    """`agent.toml` names `agents.dqn_market_free.train:trainer` by string."""

    import tomllib

    from agents.dqn_market_free.train import FreeMarketTrainer, trainer

    manifest = tomllib.loads((REPO_ROOT / "agents/dqn_market_free/agent.toml").read_text())
    assert manifest["models"][0]["trainer"] == "agents.dqn_market_free.train:trainer"
    assert isinstance(trainer, FreeMarketTrainer)
    assert callable(trainer.on_round) and callable(trainer.on_batch)


def test_round_hook_collects_per_round_and_skips_the_generation_scope():
    """The generation-scope call must not collect a second batch for the round."""

    from agents.dqn_market_free.train import FreeMarketTrainer

    calls: list[str] = []
    hook = FreeMarketTrainer()
    hook._collector = SimpleNamespace(
        collect=lambda context, reader, writer: calls.append(context.scope)
    )

    hook.on_round(SimpleNamespace(scope="generation"), None, None)
    assert calls == []
    hook.on_round(SimpleNamespace(scope="round"), None, None)
    assert calls == ["round"]


def test_batch_hook_delegates_to_the_updater():
    from agents.dqn_market_free.train import FreeMarketTrainer

    hook = FreeMarketTrainer()
    hook._updater = SimpleNamespace(update=lambda context, reader: "result")
    assert hook.on_batch(SimpleNamespace(), None) == "result"


def test_finalize_episode_writes_the_artifact_contract(tmp_path: Path):
    """`sim/runner.py` reads this dict to register the episode's training artifact."""

    import hashlib

    from agents.dqn_market_free.match_agent import make_agent
    from agents.dqn_market_free.training.collect import load_episode_batch

    weights = tmp_path / "policy.npz"
    initial_network({}).save(weights)
    trace = tmp_path / "trace.npz"
    agent = make_agent(
        {
            "weights": str(weights),
            "trace_path": str(trace),
            "policy_revision_id": "rev-1",
            "epsilon": 0.0,
            "gamma": 0.997,
        }
    )

    obs, farm, private = _fresh_market_free_state({})
    for step in range(3):
        obs["step"] = step
        agent(obs, None)

    steps = [
        [SimpleNamespace(observation={"farms": [{"money": 4_000.0}, {"money": 1_000.0}]})] * 2
    ]
    artifact = agent.finalize_episode(steps, seat=0)

    assert set(artifact) == {
        "role",
        "policy_revision_id",
        "artifact_path",
        "artifact_sha256",
        "sample_count",
    }
    assert artifact["policy_revision_id"] == "rev-1"
    assert artifact["sample_count"] == 1
    assert artifact["artifact_path"] == str(trace)
    assert artifact["artifact_sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()

    (record,) = load_episode_batch(trace)
    assert record.decisions == 3
    assert record.money[-1] == pytest.approx(4_000.0)


def test_finalize_episode_is_a_noop_without_a_trace_path(tmp_path: Path):
    """The identity spec drops `trace_path`; it still has to resolve and play."""

    from agents.dqn_market_free.match_agent import make_agent

    weights = tmp_path / "policy.npz"
    initial_network({}).save(weights)
    agent = make_agent({"weights": str(weights)})
    assert agent.finalize_episode([], seat=0) is None


def _history(count: int, final_money: float, margin: float = 0.0) -> list[dict]:
    return [
        {"failed": False, "money_difference": margin, "win": 0.0, "final_money": final_money}
        for _ in range(count)
    ]


def _episode_reader(history: list[dict]):
    return SimpleNamespace(
        summary=lambda: {
            "final_money": 3_000.0,
            "money_difference": 0.0,
            "win": 0.0,
            "metrics": {},
        },
        replay=lambda: _diagnostic_replay([], []),
        history=lambda revision_id, limit: history,
    )


def test_episode_hook_separates_the_idle_and_insolvent_collapses():
    from agents.dqn_market_free.evaluation import evaluate_episode

    context = SimpleNamespace(revision_id="rev-1", seat=0)

    idle = evaluate_episode(context, _episode_reader(_history(40, 3_000.0)))
    assert idle["collapsed"] and idle["should_reevaluate"]
    assert idle["idle_fraction"] == 1.0
    assert "emits no orders" in idle["reevaluate_reason"]

    broke = evaluate_episode(context, _episode_reader(_history(40, 12.0)))
    assert broke["collapsed"] and broke["should_reevaluate"]
    assert broke["insolvent_fraction"] == 1.0
    assert "bankrupting itself" in broke["reevaluate_reason"]

    healthy = evaluate_episode(context, _episode_reader(_history(40, 45_000.0)))
    assert not healthy["collapsed"] and not healthy["should_reevaluate"]
    assert healthy["reevaluate_reason"] == "stable"


def test_episode_hook_withholds_a_verdict_without_enough_history():
    from agents.dqn_market_free.evaluation import evaluate_episode

    verdict = evaluate_episode(
        SimpleNamespace(revision_id="rev-1", seat=0), _episode_reader(_history(4, 3_000.0))
    )
    assert not verdict["collapsed"]
    assert verdict["reevaluate_reason"] == "insufficient history"


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------


def _bare_farm(quadrants: list[str]) -> dict:
    return {
        "money": 3_000,
        "tiles": [[None] * 10 for _ in range(10)],
        "unlocked_quadrants": quadrants,
        "farmer": [4, 4],
        "hands": [],
    }


_EMPTY_PRIVATE = {"shed": {}, "seeds": {}, "inventories": []}
_EMPTY_OBS = {"market": {"prices": {}}}


def test_the_reward_ignores_the_money_trace_entirely():
    """The stream exists to stop the market round-trip from paying anything.

    Measured on `market-free-selfplay-v3`, the last run that paid the money delta
    plus potential shaping: 219 `BUY_PRODUCT` and 204 `SELL` per episode, 94% of
    the sale volume in wheat and fertilizer -- exactly the two goods
    `BUY_PRODUCT` can buy back. A round trip is cash-neutral by the simulator's
    own quoting rule, so the only thing paying for it was `Phi` marking the whole
    shed to a quote the agent had just moved. Dropping the delta and the shaping
    together is what makes that unpayable, and v13 removed the shaped arm outright
    so there is no longer a mode that can pay it.
    """

    from agents.dqn_market_free.training.reward import episode_rewards

    quiet = [3_000.0] * 6
    churning = [3_000.0, 9_000.0, 1_000.0, 8_000.0, 2_000.0, 7_000.0]

    baseline, _, _ = episode_rewards(quiet, 3_000.0, 3_000.0, 0.999)
    churned, _, _ = episode_rewards(churning, 3_000.0, 3_000.0, 0.999)
    assert np.allclose(baseline, churned)


def test_outcome_only_still_returns_the_money_delta_as_raw():
    """`raw` drives `order_refused` and the update's money diagnostics.

    It is the stream that changes, not the trace, so both keep working and an
    update's logged money metrics stay comparable across a mode change.
    """

    from agents.dqn_market_free.training.reward import MONEY_SCALE, episode_rewards

    money = [3_000.0, 2_000.0, 2_500.0]
    _, raw, money_array = episode_rewards(
        money, 4_000.0, 3_000.0, 0.999
    )
    assert float(raw.sum()) == pytest.approx((4_000.0 - 3_000.0) / MONEY_SCALE, abs=1e-6)
    assert money_array[-1] == pytest.approx(4_000.0)


def test_outcome_only_holds_q_flat_at_the_outcome():
    """Every action in the episode is labelled with the episode's own result.

    This is the whole point of the mode: the annuity plus the terminal lump sum
    to the outcome at t = 0, so the value of the first decision already carries
    what the season ended up being worth, without waiting on a 720-link
    bootstrap chain to carry it back.
    """

    from agents.dqn_market_free.training.reward import (
        OUTCOME_LOG_SCALE,
        episode_rewards,
    )

    gamma = 0.999
    turns = 100
    # Well clear of the insolvency floor, and no orders emitted, so neither
    # penalty fires and the outcome is the only term in the stream.
    money = [3_000.0] * turns
    shaped, _, _ = episode_rewards(
        money, 5_000.0, 3_000.0, gamma
    )

    outcome = float(np.tanh(np.log(5_001.0 / 3_001.0) / OUTCOME_LOG_SCALE))
    discounted = float(sum(gamma**t * shaped[t] for t in range(turns)))
    assert discounted == pytest.approx(outcome, rel=0.01)


def test_outcome_only_keeps_both_penalties():
    """They are the only terms left that separate one action from another.

    With the delta and the shaping gone the immediate reward is identical for
    every action on a turn, so the refused-order charge and the insolvency
    charge carry the entire per-action signal. If either silently dropped out of
    this mode the network would face the flat plateau documented at
    `REJECTED_ORDER_PENALTY` -- the one that settled on `SELL WOOL 16` for 57%
    of its decisions while owning no sheep.
    """

    from agents.dqn_market_free.training.reward import (
        INSOLVENCY_GRACE_TURNS,
        INSOLVENCY_PENALTY,
        REJECTED_ORDER_PENALTY,
        episode_rewards,
    )

    # Turn 1 emits an order that moves no money: refused.
    money = [3_000.0, 3_000.0, 3_000.0]
    emitted = [False, True, False]
    charged, _, _ = episode_rewards(
        money, 3_000.0, 3_000.0, 0.999, emitted
    )
    plain, _, _ = episode_rewards(
        money, 3_000.0, 3_000.0, 0.999
    )
    assert float(plain[1] - charged[1]) == pytest.approx(REJECTED_ORDER_PENALTY, abs=1e-6)

    # And a wallet that stays under the floor past the grace is charged for every
    # turn after it, read off the *resulting* wallet. Both episodes end level
    # against the opponent, so the outcome term is identical and the gap is the
    # charge alone. `INSOLVENCY_GRACE_TURNS` is why the trace has to be a
    # sustained bust rather than the one-turn dip this used to assert on: the
    # charge no longer prices a dip, only the absorbing state.
    held = INSOLVENCY_GRACE_TURNS + 2
    broke = [3_000.0, *([10.0] * held)]
    level = [3_000.0] * (held + 1)
    starved, _, _ = episode_rewards(broke, 3_000.0, 3_000.0, 0.999)
    solvent, _, _ = episode_rewards(level, 3_000.0, 3_000.0, 0.999)
    billed = solvent - starved
    # The dip itself is free; billing starts on the grace-th consecutive turn.
    assert np.allclose(billed[: INSOLVENCY_GRACE_TURNS - 1], 0.0, atol=1e-6)
    assert billed[INSOLVENCY_GRACE_TURNS - 1 : held] == pytest.approx(
        INSOLVENCY_PENALTY, abs=1e-6
    )
    # The last turn resolves to `final_money`, which is back over the floor.
    assert float(billed[held]) == pytest.approx(0.0, abs=1e-6)


def test_reward_scale_keeps_a_full_season_in_the_huber_quadratic_zone():
    """`SmoothL1Loss` has beta 1.0: above that, gradient magnitude is constant.

    At the scorer's money scale a donor-grade season returned about 100 units
    and essentially every TD error sat in the linear regime, where a 5-unit and
    a 50-unit error push identically hard.
    """

    from agents.dqn_market_free.training.reward import MONEY_SCALE, episode_rewards

    # A donor-grade season: 3k float to ~76k, banked in even steps.
    turns = 719
    money = list(np.linspace(3_000.0, 76_000.0, turns))
    shaped, _, _ = episode_rewards(money, 76_000.0, 40_000.0, 0.999)

    # Bounded by 2 rather than by 10 now, and structurally rather than by
    # calibration: the outcome is a tanh, the annuity re-pays it once across the
    # season, and nothing else in the stream is season-scaled. The old lower
    # bound of 1.0 was read off `tanh(margin / 10_000)` saturating on a 36,000
    # margin, which is exactly the flat region `OUTCOME_LOG_SCALE` removed.
    assert 0.5 < float(shaped.sum()) < 10.0
    # Every turn, including the last, is inside the quadratic zone.
    assert float(np.abs(shaped).max()) < 1.0
    assert MONEY_SCALE >= 10_000.0


def test_outcome_discriminates_from_the_stake_to_the_benchmark():
    """One scale has to cover bankruptcy, the stake, and pilkwang.

    Three ways this term has already been informationless, all of them the same
    mistake -- a coin scale sized to whichever distribution the run was producing
    at the time. A hard clip at 20,000 returned exactly -1.0 on all 640 recorded
    episodes, whose margins ran from -21,681 to -210,076. Widening to 100,000
    only moved the saturation point: by the last recorded generation every margin
    was about -128,000 and the term was a constant -0.86. Narrowing to 10,000 for
    the self-play mirror then saturated the other end -- `tanh(m / 10_000)` is
    0.96 at 20,000 and 0.999 at 40,000, so every policy above the low twenties was
    reward-identical, and this family plateaued at 11-12k and 5-6k under it.

    A log ratio has no saturation point to place. This pins that it discriminates
    across the whole reachable range at once: broke, the 3,000 stake, pilkwang's
    ~142,000, and the 500,000 headroom `OUTCOME_LOG_SCALE` is sized on.
    """

    from agents.dqn_market_free.training.reward import episode_rewards

    stake = 3_000.0

    def terminal(final: float) -> float:
        # Well clear of the insolvency floor for the recorded turns, so this
        # reads the outcome term alone. gamma = 1.0 zeroes the annuity.
        shaped, _, _ = episode_rewards([stake, stake], final, 0.0, 1.0)
        return float(shaped[-1])

    ladder = [0.0, 600.0, stake, 12_000.0, 40_000.0, 142_000.0, 500_000.0]
    values = [terminal(final) for final in ladder]

    # Strictly ordered, and none of them crushed against the bound.
    assert values == sorted(values)
    assert max(values) < 0.99
    # Every adjacent pair genuinely separated -- the failure being pinned here is
    # a flat region, not a wrong constant. The tightest step is 40k -> 142k.
    assert min(b - a for a, b in zip(values, values[1:])) > 0.1
    # Doing nothing is exactly parity: the stake is the reference.
    assert terminal(stake) == pytest.approx(0.0)
    # Beating the benchmark is a point on the slope, not the end of it.
    assert terminal(500_000.0) - terminal(142_000.0) > 0.1

    # Bounded either side, so one blowout episode cannot dominate a batch.
    # tanh is open on (-1, 1) mathematically; float32 saturates at the bound.
    assert -1.0 <= terminal(0.0) < 0.0
    assert 0.0 < terminal(1e12) <= 1.0


def test_outcome_does_not_read_the_opponent():
    """The mirror is 65% of rollouts and its margin is exactly zero.

    `market-free-population-v25` ran `self_play_ratio = 0.65` with a margin-based
    outcome and `league_pairings.mean_margin` was 0.000 in all eleven
    generations -- both seats are the same policy off the same stake under the
    same frozen donor field. So two of every three rollouts carried no outcome
    term at all, leaving `IDLE_PENALTY` and `INSOLVENCY_PENALTY` as the whole
    objective, whose joint optimum is to emit an order every turn and stay a coin
    above the floor. The measured policy did exactly that: `trading_rate` 0.00 to
    0.78 while benchmark money fell to 98.

    An absolute outcome cannot have that failure mode, and this is what says so.

    The opponent-relative leg landed beside it does not reintroduce it, because
    `OPPONENT_WEIGHT` is 0.0: this reads the default and is therefore also the
    regression that says the leg is inert.
    """

    from agents.dqn_market_free.training.reward import OPPONENT_WEIGHT, episode_rewards

    assert OPPONENT_WEIGHT == 0.0

    money = [3_000.0] * 50
    mirror, _, _ = episode_rewards(money, 12_000.0, 12_000.0, 0.999)
    scripted, _, _ = episode_rewards(money, 12_000.0, 142_000.0, 0.999)
    bankrupt, _, _ = episode_rewards(money, 12_000.0, 0.0, 0.999)

    assert float(mirror.sum()) == pytest.approx(float(scripted.sum()))
    assert float(mirror.sum()) == pytest.approx(float(bankrupt.sum()))
    # And it is not zero, which is the whole point -- a mirror episode that grew
    # the stake fourfold is rewarded for it.
    assert float(mirror.sum()) > 0.2


def test_the_opponent_leg_is_landed_inert_and_changes_nothing_until_it_is_moved():
    """`OPPONENT_WEIGHT` is 0.0 and the stream is bit-identical without it.

    The mechanism is in the tree so that turning it on is a one-number change,
    which is only worth anything if the number being 0.0 is provably the same as
    the parameter not existing. `priority_alpha` was landed under exactly this
    rule, and it is the reason the uniform draw it replaced could be trusted as a
    baseline.

    Bit-identical rather than approximately equal: `episode_rewards` returns
    float32 and the whole point is that no arithmetic happened.
    """

    from agents.dqn_market_free.training.reward import episode_rewards

    money = [3_000.0, 2_400.0, 5_100.0, 900.0, 4_000.0] * 8
    emitted = [bool(index % 3) for index in range(len(money))]

    explicit, _, _ = episode_rewards(
        money, 12_000.0, 142_000.0, 0.999, emitted, opponent_weight=0.0
    )
    default, _, _ = episode_rewards(money, 12_000.0, 142_000.0, 0.999, emitted)

    assert np.array_equal(default, explicit)


def test_the_opponent_leg_reads_a_log_ratio_of_the_two_purses():
    """Poorer opponent scores higher, richer scores lower, parity is exactly zero.

    Parity being *exactly* zero is the property that matters and the reason this
    is a log ratio rather than a difference. It is what makes the mirror's
    degenerate case harmless on its own terms rather than merely small, so the
    caller's gate is a second line of defence and not the only one.
    """

    from agents.dqn_market_free.training.reward import episode_rewards

    money = [3_000.0] * 50
    weight = 0.1

    def terminal(opponent: float) -> float:
        # gamma = 1.0 zeroes the annuity, so this is the outcome alone -- which
        # is also what the outcome contributes to Q at every timestep.
        shaped, _, _ = episode_rewards(
            money, 12_000.0, opponent, 1.0, opponent_weight=weight
        )
        return float(shaped[-1])

    absolute, _, _ = episode_rewards(money, 12_000.0, 12_000.0, 1.0)
    baseline = float(absolute[-1])

    # Parity: the ratio is 1, its log is 0, so the leg contributes nothing at all.
    assert terminal(12_000.0) == pytest.approx(baseline)
    assert terminal(0.0) > baseline
    assert terminal(142_000.0) < baseline
    # Bounded by the weight, which is what makes it a tiebreaker rather than a
    # driver: it cannot overturn the absolute leg's ordering by more than 0.1.
    assert abs(terminal(0.0) - baseline) <= weight
    assert abs(terminal(142_000.0) - baseline) <= weight


def test_the_opponent_leg_keeps_the_annuity_flat_across_the_season():
    """Folded into `outcome` before it is paid, so Q stays flat at the total.

    The terminal lump plus the `(1 - gamma)` annuity is what makes the outcome's
    contribution to Q the same at every timestep. A second term paid any other
    way would break that, and the whole reason the leg is added to `outcome`
    rather than to `shaped` is to make breaking it impossible.
    """

    from agents.dqn_market_free.training.reward import episode_rewards

    gamma = 0.999
    money = [3_000.0] * 400
    shaped, _, _ = episode_rewards(money, 40_000.0, 4_000.0, gamma, opponent_weight=0.1)

    # Strip the charges: no turn emitted, so no refusal, and every recorded purse
    # is well clear of the insolvency floor. `emitted` is None, so no idle charge.
    turns = len(shaped)
    annuity = float(shaped[0])
    assert annuity == pytest.approx(float(shaped[turns // 2]))
    # The lump is the rest of it, and the two sum to the outcome twice over --
    # once through the annuity's whole stream and once on the terminal turn.
    outcome = annuity / (1.0 - gamma)
    assert float(shaped[-1]) == pytest.approx(annuity + outcome)


# --------------------------------------------------------------------------
# The outcome annuity
#
# The terminal margin lump lands on one turn of ~720 and the buffer samples
# 24-turn windows, so ~3% of transitions ever saw it as a reward. The rest waited
# on a bootstrap chain up to 720 links deep to learn whether their episode made
# money. The annuity pays `(1 - gamma) * outcome` on every step so the signal
# arrives by gradient instead.
# --------------------------------------------------------------------------


#: Own money, held flat, for the annuity tests below. Also the stake, since
#: `episode_rewards` reads the reference off `money[0]`, so an episode finishing
#: here is exactly parity and the comparison below is against a clean zero.
#:
#: These tests used to move the margin by varying the *opponent's* money and
#: never one's own, so `raw` stayed bit-identical between the two calls. The
#: outcome is absolute now, so the final money is what has to move -- and the
#: property still holds for a better reason: `raw` has not been part of `shaped`
#: since v13, and with `emitted` unset and both finals well clear of
#: `INSOLVENCY_FLOOR` no per-turn charge fires either. The outcome terms are the
#: only difference between the two streams.
SEASON_MONEY = 5_000.0


def _season(turns: int = 400) -> tuple[list[float], list[float]]:
    return [SEASON_MONEY] * turns


def _season_outcome_from(stake: float, final: float) -> float:
    """The outcome term an episode finishing on `final` from `stake` earns."""

    from agents.dqn_market_free.training.reward import OUTCOME_LOG_SCALE

    return float(np.tanh(np.log((final + 1.0) / (stake + 1.0)) / OUTCOME_LOG_SCALE))


def _season_outcome(final: float) -> float:
    """The same, for an episode played off `SEASON_MONEY`."""

    return _season_outcome_from(SEASON_MONEY, final)


def _outcome_gap(final: float, gamma: float, turns: int = 400) -> np.ndarray:
    """Per-step reward of an episode finishing on `final`, minus one at parity."""

    from agents.dqn_market_free.training.reward import episode_rewards

    money = _season(turns)
    played, _, _ = episode_rewards(money, final, 0.0, gamma)
    drew, _, _ = episode_rewards(money, SEASON_MONEY, 0.0, gamma)
    return played - drew


def test_the_outcome_reaches_every_decision_not_just_the_last():
    """Every step of a winning episode must score above the same step at parity."""

    gamma = 0.999
    final = 60_000.0

    outcome = _season_outcome(final)
    gap = _outcome_gap(final, gamma)

    # Not one turn in three hundred, but all of them.
    assert bool((gap > 0.0).all())
    # Every step but the last carries exactly the annuity.
    assert gap[:-1] == pytest.approx((1.0 - gamma) * outcome, rel=1e-4)
    # The last carries the annuity and the lump.
    assert float(gap[-1]) == pytest.approx((1.0 - gamma) * outcome + outcome, rel=1e-4)


def test_the_outcome_contributes_a_flat_offset_to_the_return():
    """Annuity plus lump is worth `outcome` in Q at every t, early or late.

    This is the reason the annuity is scaled by `(1 - gamma)` and not by `1 / T`.
    An even `outcome / T` spread would put ~0.71 * outcome into the return at t=0
    and ~0 at the last turn -- less late-game signal than the lump gave on its own.
    """

    gamma = 0.999
    turns = 400

    for final in (60_000.0, 500.0):
        outcome = _season_outcome(final)
        gap = _outcome_gap(final, gamma, turns)
        # Beginning, middle, and the final turn all see the same offset.
        for start in (0, turns // 2, turns - 1):
            weights = gamma ** np.arange(turns - start, dtype=np.float64)
            discounted = float((gap[start:] * weights).sum())
            assert discounted == pytest.approx(outcome, rel=0.01)


def test_the_annuity_vanishes_at_gamma_one():
    """At gamma = 1 an undiscounted lump already reaches every step at full weight.

    Every other reward test in this file calls `episode_rewards` at gamma = 1,
    including the policy-invariance one, so this is what keeps them meaningful.
    """

    from agents.dqn_market_free.training.reward import (
        OUTCOME_LOG_SCALE,
        episode_rewards,
    )

    money = [3_000.0, 2_000.0, 2_500.0, 4_000.0]

    shaped, _, _ = episode_rewards(money, 9_000.0, 3_000.0, 1.0)

    outcome = float(np.tanh(np.log(9_001.0 / 3_001.0) / OUTCOME_LOG_SCALE))
    # At gamma = 1 the annuity is zero and the lump is the whole stream.
    assert float(shaped.sum()) == pytest.approx(outcome, abs=1e-5)


def test_a_losing_episode_is_charged_on_every_turn():
    """The annuity is signed, so it is a penalty as readily as a bonus."""

    assert bool((_outcome_gap(500.0, 0.999) < 0.0).all())


def test_the_annuity_keeps_a_full_season_in_the_huber_quadratic_zone():
    """It must not cost what `MONEY_SCALE` was chosen to buy.

    Per step the annuity is at most `1 - gamma`, so it cannot move the maximum;
    across a season it adds at most `(1 - gamma) * turns`, which at gamma = 0.999
    is 0.72 of a unit on a sum the scale keeps in single digits.
    """

    from agents.dqn_market_free.training.reward import episode_rewards

    turns = 719
    money = list(np.linspace(3_000.0, 76_000.0, turns))
    shaped, _, _ = episode_rewards(money, 76_000.0, 40_000.0, 0.999)

    assert 0.5 < float(shaped.sum()) < 10.0
    assert float(np.abs(shaped).max()) < 1.0
    # The annuity's share of that is `(1 - gamma) * turns` of one outcome, which
    # is what the docstring's 0.72 bound refers to.
    outcome = _season_outcome_from(3_000.0, 76_000.0)
    assert float(shaped.sum()) == pytest.approx(outcome * (1.0 + (1.0 - 0.999) * turns), rel=1e-3)


def test_activity_is_counted_by_effect_not_by_emission():
    """An invalid order is a silent no-op, so emitting one is not activity.

    Measured on a real batch: 57% of decisions were `SELL WOOL 16` from a farm
    that owned no sheep, 98.4% of emissions were refused, and the old
    emission-based count reported 59% activity for a policy whose real activity
    was 1% -- clearing the promotion gate's activity floor, which exists
    precisely to catch a silent policy.
    """

    from agents.dqn_market_free.diagnostics import diagnose_order_lists

    orders = [[["SELL", "WOOL", 16]], [["SELL", "WOOL", 16]], [["BUY_SEED", "WHEAT", 6]], []]
    # Only the seed purchase moved money; the wool sales were refused.
    deltas = [0.0, 0.0, -120.0, 0.0]

    d = diagnose_order_lists(orders, deltas).as_dict()
    assert d["decisions"] == 4
    assert d["emitting_decisions"] == 3
    assert d["effective_decisions"] == 1
    assert d["emitted_rate"] == pytest.approx(0.75)
    assert d["trading_rate"] == pytest.approx(0.25)
    assert d["rejected_order_rate"] == pytest.approx(2 / 3)


def test_activity_fails_closed_without_money_information():
    """A caller that forgets the deltas must trip the gate, not pass it.

    The previous behaviour counted emissions as trades, which is the exact
    failure mode this replaced, so the safe default is to prove nothing.
    """

    from agents.dqn_market_free.diagnostics import diagnose_order_lists

    d = diagnose_order_lists([[["SELL", "WOOL", 16]]] * 5).as_dict()
    assert d["emitting_decisions"] == 5
    assert d["effective_decisions"] == 0
    assert d["trading_rate"] == 0.0


def test_replay_diagnostics_read_effect_from_the_replay_itself():
    from agents.dqn_market_free.diagnostics import diagnose_replay

    def seat(money, orders=None):
        state = {"observation": {"player": 0, "farms": [{"money": money}, {"money": 0}]}}
        if orders is not None:
            state["action"] = {"market": orders}
        return [state, state]

    replay = {
        "steps": [
            seat(3000),
            seat(3000, [["SELL", "WOOL", 16]]),   # refused: money unchanged
            seat(2880, [["BUY_SEED", "WHEAT", 6]]),  # accepted: money fell
            seat(2880, []),
        ]
    }
    d = diagnose_replay(replay, seat=0).as_dict()
    assert d["emitting_decisions"] == 2
    assert d["effective_decisions"] == 1


def test_rejected_orders_are_charged_and_valid_ones_are_not():
    """Break the tie the policy was exploiting, without picking a trade for it.

    Shaping leaves a real purchase at exactly zero immediate reward, so NOOP, a
    seed buy and an impossible order all scored 0 on the turn they happened. The
    charge separates "interacted with the world" from "did not" and says nothing
    about which interaction is better.
    """

    from agents.dqn_market_free.training.reward import (
        IDLE_PENALTY,
        REJECTED_ORDER_PENALTY,
        episode_rewards,
    )

    money = [3_000.0, 3_000.0, 2_880.0]
    # turn 0 emitted and moved nothing (refused); turn 1 emitted and paid;
    # turn 2 emitted nothing at all.
    emitted = [True, True, False]

    # Finishing on the stake, so the outcome term is exactly zero and the three
    # charges are the whole stream. It used to be enough to draw level with the
    # opponent; the outcome is absolute now, so it is the *stake* that has to be
    # matched. gamma = 1.0 keeps the annuity out of it either way.
    shaped, raw, _ = episode_rewards(money, 3_000.0, 2_880.0, 1.0, emitted)

    assert shaped[0] == pytest.approx(-REJECTED_ORDER_PENALTY)
    # The accepted order takes no charge: the stream pays nothing for a trade,
    # so an accepted turn sits at exactly zero.
    assert shaped[1] == pytest.approx(0.0)
    # A turn that emitted nothing pays the idle charge, scaled by how much season
    # is left: turn 2 of a 3-turn trace is a third of the way from full rate to
    # zero. The three still sit in the order the reward is meant to express.
    assert shaped[2] == pytest.approx(-IDLE_PENALTY * (1 - 2 / 3))
    assert shaped[1] > shaped[2] > shaped[0]

    # Off by default for callers that cannot say what was emitted -- neither
    # charge can be attributed without it.
    plain, _, _ = episode_rewards(money, 3_000.0, 2_880.0, 1.0)
    assert np.allclose(plain, 0.0)


def test_idling_is_charged_but_stays_cheaper_than_going_broke():
    """The two collapses, priced against each other.

    Idling has to be bad or the NOOP attractor is the objective's own optimum --
    `market-free-noisy-v6` sat at exactly 3,000 money with `trading_rate` 0.0 for
    twelve updates. It also has to stay *cheaper* than bankruptcy, or the reward
    buys the other collapse instead.
    """

    from agents.dqn_market_free.training.reward import (
        IDLE_PENALTY,
        INSOLVENCY_PENALTY,
        REJECTED_ORDER_PENALTY,
        episode_rewards,
    )

    assert 0.0 < IDLE_PENALTY < REJECTED_ORDER_PENALTY < INSOLVENCY_PENALTY

    turns = 720
    idle, _, _ = episode_rewards(
        [3_000.0] * turns, 3_000.0, 3_000.0, 1.0, [False] * turns
    )
    broke, _, _ = episode_rewards(
        [10.0] * turns, 10.0, 10.0, 1.0, [False] * turns
    )
    # The charge decays linearly to zero at the deadline, so a fully idle season
    # pays the *mean* rate -- half of `IDLE_PENALTY` -- not the full one.
    assert float(idle.sum()) == pytest.approx(-turns * IDLE_PENALTY / 2, abs=1e-3)
    assert float(broke.sum()) < float(idle.sum()), "bankruptcy must stay the worse one"

    # And the whole point of the size: it must stay under the outcome term's own
    # range, or trading yourself downward scores better than sitting still. At a
    # flat 0.0005 the season cost 0.36 against a 3,000-to-1,898 outcome gap of
    # 0.104, and `market-free-hot-v16` duly learned to trade on 73-99% of turns
    # while mean money fell 3,542 to 1,898.
    #
    # The absolute outcome tightened this rather than loosening it -- 0.091
    # against the old 0.104 -- because `OUTCOME_LOG_SCALE` is sized on 500,000 of
    # headroom instead of on the distribution of the day. That is the trade taken
    # knowingly, and this is the assertion that says how much slack is left.
    import math

    from agents.dqn_market_free.training.reward import OUTCOME_LOG_SCALE

    outcome_gap = -math.tanh(math.log(1_899.0 / 3_001.0) / OUTCOME_LOG_SCALE)
    assert turns * IDLE_PENALTY / 2 < outcome_gap


def test_an_unobserved_turn_is_not_charged_for_silence():
    """`emitted` can be shorter than the stream, and absence is not evidence.

    The agent skips a malformed observation, so the trace can end early. A turn
    the recorder never saw must not be billed for sitting out, or a truncated
    episode is charged for the tail it never played.
    """

    from agents.dqn_market_free.training.reward import IDLE_PENALTY, episode_rewards

    money = [3_000.0] * 4
    shaped, _, _ = episode_rewards(money, 3_000.0, 3_000.0, 1.0, [False, False])

    # Charged, and scaled by season position like any other idle turn.
    assert shaped[0] == pytest.approx(-IDLE_PENALTY * (1 - 0 / 4))
    assert shaped[1] == pytest.approx(-IDLE_PENALTY * (1 - 1 / 4))
    # Never observed, so never billed for silence.
    assert shaped[2] == pytest.approx(0.0)
    assert shaped[3] == pytest.approx(0.0)


def test_the_penalty_is_smaller_than_the_terminal_signal():
    """It is not policy-invariant, so it must not become the objective."""

    from agents.dqn_market_free.training.reward import REJECTED_ORDER_PENALTY

    # Worst case: every turn of a season emits a refused order.
    worst = REJECTED_ORDER_PENALTY * 719
    assert 0.1 < worst < 1.0


# --------------------------------------------------------------------------
# The insolvency floor
#
# Bankruptcy is an opportunity cost, and the money delta cannot express one: the
# wallet does not go negative, so spending the last of the stake reads as a
# one-off -0.15 and everything it forecloses shows up only as the absence of
# later fills. Measured on `market-free-simple-v2`: the policy spent the whole
# 3,000 on `HIRE 13` orders by turn four and then held NOOP for 715 turns, which
# is what the buffer was made of.
#
# It is charged on a *streak*, not on a turn. Pricing every below-floor turn also
# priced the wallet dipping under the floor between a purchase and the sale that
# funds it, which is the opposite of what the constant was bought for -- see
# `INSOLVENCY_GRACE_TURNS` for the v26 measurement that moved it.
# --------------------------------------------------------------------------


def test_every_turn_with_no_runway_past_the_grace_is_charged():
    from agents.dqn_market_free.training.reward import (
        INSOLVENCY_FLOOR,
        INSOLVENCY_GRACE_TURNS,
        INSOLVENCY_PENALTY,
        episode_rewards,
    )

    # Turn 0 spends down to the floor and the farm sits there for the rest of the
    # trace, then recovers on the last turn. Recovering all the way to the stake,
    # so the outcome term is zero and the insolvency charge is the whole stream.
    held = INSOLVENCY_GRACE_TURNS + 3
    money = [3_000.0, *([100.0] * held)]
    shaped, _, _ = episode_rewards(money, 3_000.0, 900.0, 1.0)

    charged = -shaped
    # The first `grace - 1` turns of the streak are free, and every turn from the
    # grace-th onward is billed.
    assert np.allclose(charged[: INSOLVENCY_GRACE_TURNS - 1], 0.0, atol=1e-6)
    assert charged[INSOLVENCY_GRACE_TURNS - 1 : held] == pytest.approx(
        INSOLVENCY_PENALTY, abs=1e-6
    )
    # The turn that climbs back over the floor is not charged.
    assert float(charged[held]) == pytest.approx(0.0, abs=1e-6)
    # And it is the resulting wallet that is read, not the opponent's: the same
    # trace scores identically however rich the other seat got.
    rich, _, _ = episode_rewards(money, 3_000.0, 142_000.0, 1.0)
    assert np.allclose(rich, shaped)

    # The charge reads the resulting wallet, so the turn that spends the last of
    # the stake is the first one that counts toward the streak rather than the
    # one after it.
    assert float(money[0]) > INSOLVENCY_FLOOR > float(money[1])


def test_a_dip_between_a_purchase_and_its_sale_is_never_charged():
    """The measurement that moved this constant, restated as a test.

    On `market-free-logmoney-v26` the per-turn charge accounted for -0.153 of a
    -0.193 mean return -- 78% of the objective -- against a money term worth
    +0.004, because a policy buying inputs runs its wallet under the floor between
    the purchase and the sale that funds it. `d(outcome)/d(log money)` is 1/5, so
    doubling a season is worth +0.24 while the ~120 low-cash turns it takes cost
    0.002 * 120 = -0.24. Investment netted exactly zero.
    """

    from agents.dqn_market_free.training.reward import (
        INSOLVENCY_GRACE_TURNS,
        episode_rewards,
    )

    # A whole season of buy-low-sell-high cycles, each dipping under the floor for
    # one turn short of the grace before the sale lands.
    dip = INSOLVENCY_GRACE_TURNS - 1
    money: list[float] = []
    for _ in range(20):
        money.extend([10.0] * dip)
        money.append(6_000.0)
    shaped, _, _ = episode_rewards(money, 6_000.0, 6_000.0, 1.0)

    # gamma = 1.0, so the annuity is zero and only the terminal lump is non-zero.
    # Every other turn being exactly zero is the charge being absent across all
    # twenty cycles -- the pathology this grace exists to stop pricing.
    assert np.allclose(shaped[:-1], 0.0, atol=1e-7)
    assert float(shaped[-1]) > 0.0


def test_the_grace_still_prices_the_absorbing_state():
    """What the grace must *not* buy back: the v2 bankruptcy it was bought for.

    `market-free-simple-v2` spent the stake on `HIRE 13` by turn four and then
    held 715 turns of NOOP, and that dead state was ~99% of the replay buffer. A
    grace that let a whole broke season through would delete the term.
    """

    from agents.dqn_market_free.training.reward import (
        INSOLVENCY_GRACE_TURNS,
        INSOLVENCY_PENALTY,
        episode_rewards,
    )

    season = 719
    money = [3_000.0, *([0.0] * (season - 1))]
    shaped, _, _ = episode_rewards(money, 0.0, 3_000.0, 1.0)

    billed = float(-shaped[:-1].sum())
    expected = INSOLVENCY_PENALTY * (season - 1 - (INSOLVENCY_GRACE_TURNS - 1))
    assert billed == pytest.approx(expected, abs=1e-3)
    # Still the order of magnitude the constant was sized at: a whole season broke
    # costs about what losing costs, rather than nothing.
    assert 1.0 < billed < 2.0


def test_a_solvent_season_is_never_charged():
    """A farm trading above the floor must see the shaping unchanged."""

    from agents.dqn_market_free.training.reward import (
        INSOLVENCY_PENALTY,
        episode_rewards,
    )

    money = list(np.linspace(3_000.0, 40_000.0, 400))
    shaped, _, _ = episode_rewards(money, 40_000.0, 40_000.0, 1.0)

    # gamma = 1.0, so the annuity is zero and the last step carries the outcome
    # lump alone. Every other step must be untouched -- that is the charge being
    # absent, which is what this pins.
    assert float(np.abs(shaped[:-1]).max()) < INSOLVENCY_PENALTY
    assert float(shaped[-1]) == pytest.approx(_season_outcome_from(3_000.0, 40_000.0))


def test_the_insolvency_charge_is_the_scale_of_the_outcome_not_of_a_fill():
    """Large enough to see across a season, small enough not to become the goal.

    A whole episode spent broke must cost about what losing costs -- otherwise
    the network optimises the floor instead of the margin -- while a farm that
    dips under it between a purchase and a sale pays almost nothing.
    """

    from agents.dqn_market_free.training.reward import INSOLVENCY_PENALTY

    season = INSOLVENCY_PENALTY * 719
    assert 1.0 < season < 2.0
    # A dozen turns between spending and selling: noise against a real fill,
    # which at MONEY_SCALE is 0.05 for a thousand coins.
    assert INSOLVENCY_PENALTY * 12 < 0.05


# --------------------------------------------------------------------------
# The unlocked-land potential
#
# The one policy-invariant term in `reward.py`, and the only reason it is allowed
# to exist beside four terms that are deliberately not. Buying a plot creates
# crop roles the frozen donor can work, but it pays back through harvests ten
# days later, which reaches the purchase decision only through a terminal outcome
# discounted 0.999^720 = 0.49 in 24-turn hops -- while the 1,000-coin cost lands
# in the money trace immediately. Measured on `market-free-streak-v27`
# generations 2-6: seats that bought land planted 38.0 against 36.9, lost 22.6
# plants against 15.1, and finished on a median of 2 coins against 400.
# --------------------------------------------------------------------------


def _land_season(days: int = 30, turns_per_day: int = 24):
    """A flat money trace with the day trace the simulator would produce."""

    turns = days * turns_per_day
    money = [3_000.0] * turns
    day_trace = [index // turns_per_day for index in range(turns)]
    return money, day_trace


def _land_delta(quadrants, days, money, gamma=0.999):
    """The shaping alone: the stream with the land trace minus the stream without.

    Everything else in `episode_rewards` reads only `money`, so subtracting the
    two isolates `gamma * PHI(s') - PHI(s)` exactly.

    `land_bonus` is held at zero deliberately. It is the one term in this file
    that does *not* telescope -- that is its entire purpose -- so leaving it on
    would fold a deliberate non-invariance into the helper whose whole job is to
    prove invariance, and every test below would be measuring the wrong thing.
    """

    from agents.dqn_market_free.training.reward import episode_rewards

    shaped, _, _ = episode_rewards(
        money, money[-1], money[-1], gamma, quadrants=quadrants, days=days, land_bonus=0.0
    )
    plain, _, _ = episode_rewards(money, money[-1], money[-1], gamma)
    return np.asarray(shaped, dtype=np.float64) - np.asarray(plain, dtype=np.float64)


def _discounted(delta, gamma=0.999):
    """`sum_t gamma**t * F_t` -- the quantity policy invariance is a claim about.

    Not the raw sum. Telescoping `gamma * PHI(s') - PHI(s)` under the discount
    gives `gamma**T * PHI_T - PHI_0`, which is zero when both ends are zero. The
    *undiscounted* sum is `(gamma - 1) * sum(PHI)`, a small negative number, and
    asserting on that instead would be testing the wrong theorem.
    """

    steps = np.arange(len(delta), dtype=np.float64)
    return float(np.sum(np.power(gamma, steps) * delta))


def test_the_land_potential_cannot_change_which_policy_is_optimal():
    """The property the whole term rests on: it telescopes to nothing.

    Ng, Harada & Russell -- `F(s, s') = gamma * PHI(s') - PHI(s)` leaves the
    optimal policy unchanged for any `PHI`. So this can move *when* credit for a
    plot arrives and never whether the plot was worth buying, which is what makes
    it safe where a bonus on the `BUY_LAND` turn is not: the measured land buyers
    finish on a median of 2 coins, so paying for the action pays for the
    bankruptcy. If this test is ever weakened the term must come out.
    """

    money, days = _land_season()
    # One plot taken on day 5 and held to the end.
    quadrants = [1 if day < 5 else 2 for day in days]
    delta = _land_delta(quadrants, days, money)

    # `PHI` is zero at both ends, so the discounted shaping return is zero and
    # the term adds nothing to any policy's value. Held tight deliberately: this
    # is the property the whole design rests on, not an approximation of it.
    assert _discounted(delta) == pytest.approx(0.0, abs=1e-6)

    # Two plots, taken at different times, and the same holds -- the invariance
    # is not an artifact of a single purchase.
    two = [1 if day < 5 else (2 if day < 12 else 3) for day in days]
    assert _discounted(_land_delta(two, days, money)) == pytest.approx(0.0, abs=1e-6)


def test_the_land_potential_pays_at_the_purchase_not_at_the_harvest():
    """Why it is worth adding at all: the credit arrives when the decision does."""

    from agents.dqn_market_free.training.reward import LAND_POTENTIAL_WEIGHT

    money, days = _land_season()
    quadrants = [1 if day < 5 else 2 for day in days]
    delta = _land_delta(quadrants, days, money)

    # The credit lands on the turn that *bought* the plot, one before the state
    # that first shows it: `shaped[t]` is paid with `PHI(s_{t+1})`, and the
    # quadrant appears in the state at the start of the following turn.
    purchase = days.index(5) - 1
    assert float(delta[purchase]) == pytest.approx(
        LAND_POTENTIAL_WEIGHT * (1.0 - 5.0 / 30.0), abs=5e-3
    )
    # And it is the whole point that it arrives there rather than being spread:
    # the lump dwarfs every other entry, which is what a 24-step n-step return
    # can actually see.
    assert float(delta[purchase]) > 10 * float(np.abs(np.delete(delta, purchase)).max())


def test_the_land_potential_is_proportional_to_the_remaining_season():
    """A plot with two days of harvests left is worth almost nothing."""

    def lump(bought_on: int) -> float:
        money, days = _land_season()
        quadrants = [1 if day < bought_on else 2 for day in days]
        delta = _land_delta(quadrants, days, money)
        return float(delta[days.index(bought_on) - 1])

    early, mid, late = lump(5), lump(15), lump(28)
    assert early > mid > late > 0.0
    # Two days left of thirty is worth under a tenth of twenty-five days left.
    assert late < 0.1 * early


def test_the_land_potential_is_zero_at_the_terminal_state():
    """Set rather than derived, so a season ending on day 29 still telescopes.

    `PHI` reaches zero on its own only if the last recorded day is `TOTAL_DAYS`.
    A replay that stops a day early would otherwise leave a lump on the final
    turn that nothing ever repays -- which is exactly the un-repaid credit
    policy invariance is bought to prevent.
    """

    from agents.dqn_market_free.training.reward import land_potential

    assert float(land_potential([2, 2, 2], [0, 1, 2])[-1]) == 0.0

    # A season truncated at day 29, plot taken on day 5, so `PHI` on the last
    # turn would still be `w * 1/30` if it were only derived from the clock --
    # a lump nothing repays.
    money, days = _land_season(days=29)
    quadrants = [1 if day < 5 else 2 for day in days]
    assert _discounted(_land_delta(quadrants, days, money)) == pytest.approx(0.0, abs=1e-6)


def test_the_land_trace_is_optional_and_absent_is_bit_identical():
    """The guard that keeps `clone.py` and `collect.py` from diverging.

    They are the two callers and they must produce the same kind of sample. A
    caller that cannot recover the trace has to fall back to the old stream
    exactly, not to a different quantity that happens to look similar.
    """

    from agents.dqn_market_free.training.reward import episode_rewards

    money, days = _land_season()
    quadrants = [2] * len(days)
    plain, raw, money_array = episode_rewards(money, 3_000.0, 3_000.0, 0.999)

    for kwargs in (
        {},
        {"quadrants": quadrants},
        {"days": days},
        {"quadrants": quadrants, "days": days, "land_weight": 0.0},
    ):
        shaped, other_raw, other_money = episode_rewards(
            money, 3_000.0, 3_000.0, 0.999, **kwargs
        )
        assert np.array_equal(shaped, plain)
        assert np.array_equal(other_raw, raw)
        assert np.array_equal(other_money, money_array)


def test_the_purchase_bonus_is_paid_once_on_the_turn_the_quadrant_appears():
    """The non-telescoping term, and the reason it exists.

    The potential above is policy-invariant by construction -- it can move when
    credit arrives and can never change which policy is optimal, which is why it
    has never moved the land rate in six sessions. This one is meant to change
    the optimum, so it has to land on the purchase and survive the discount.
    """

    from agents.dqn_market_free.training.reward import LAND_PURCHASE_BONUS, episode_rewards

    money, days = _land_season()
    turns_per_day = 24
    quadrants = [1 if day < 5 else 2 for day in days]

    with_bonus, _, _ = episode_rewards(
        money, money[-1], money[-1], 0.999, quadrants=quadrants, days=days
    )
    without, _, _ = episode_rewards(
        money, money[-1], money[-1], 0.999, quadrants=quadrants, days=days, land_bonus=0.0
    )
    delta = np.asarray(with_bonus, dtype=np.float64) - np.asarray(without, dtype=np.float64)

    paid = np.flatnonzero(np.abs(delta) > 1e-9)
    assert paid.tolist() == [5 * turns_per_day - 1], "one payment, on the buying turn"
    assert float(delta.sum()) == pytest.approx(LAND_PURCHASE_BONUS, abs=1e-6)
    # And unlike the potential it does *not* telescope away -- that is the point.
    assert _discounted(delta) > 1e-3


def test_the_purchase_bonus_never_repays_the_whole_outlay():
    """The bound that separates a nudge from paying the agent to go bankrupt.

    `MONEY_SCALE` is 20,000, so the first quadrant's 1,000 coins score -0.05 on
    the purchase turn. A bonus at or above that makes the land free and the
    agent buys it whether or not it can work it -- the failure that ran through
    v26 generations 2-5 at 18% of seats and a mean of 488 coins.
    """

    from agents.dqn_market_free.training.reward import (
        LAND_PURCHASE_BONUS,
        MONEY_SCALE,
        episode_rewards,
    )
    from agents.dqn_market_free import donor

    outlay = donor.LAND_PRICES[0] / MONEY_SCALE
    assert LAND_PURCHASE_BONUS < outlay, "the bonus must not cancel the purchase"

    # Priced down the ladder, so a dearer quadrant is not overpaid for its cost.
    days = [1] * 6
    flat = [3_000.0] * 5
    paid = []
    for owned in range(len(donor.LAND_PRICES)):
        quadrants = [owned + 1] * 3 + [owned + 2] * 3
        shaped, _, _ = episode_rewards(flat, flat[-1], flat[-1], 0.999, quadrants=quadrants,
                                       days=days, land_weight=0.0)
        base, _, _ = episode_rewards(flat, flat[-1], flat[-1], 0.999, quadrants=quadrants,
                                     days=days, land_weight=0.0, land_bonus=0.0)
        paid.append(float((np.asarray(shaped) - np.asarray(base)).sum()))

    assert paid == pytest.approx(
        [LAND_PURCHASE_BONUS * donor.LAND_PRICES[0] / price for price in donor.LAND_PRICES],
        abs=1e-6,
    )
    assert paid[0] > paid[1] > paid[2], "a dearer plot pays less, not more"


def test_the_land_potential_has_no_cycle_to_pump():
    """Why the v13 objection does not transfer, stated as a test.

    Potential-based shaping was retired from `reward.py` once because it marked
    the whole shed to a quote the agent moved itself by buying -- the potential
    read a quantity the policy controlled, so it could be cycled. Land is bought
    from a fixed price ladder and never sold, so the quadrant count is monotone
    and the only other input is the clock.
    """

    from agents.dqn_market_free.training.reward import land_potential

    money, days = _land_season()
    quadrants = [1 if day < 5 else 2 for day in days]
    assert all(b >= a for a, b in zip(quadrants, quadrants[1:])), "land is never sold"

    # Exactly one turn in the season raises `PHI`; every other step gives value
    # back. There is no state the policy can return to for a second payout.
    steps = np.diff(land_potential(quadrants, days))
    assert int(np.count_nonzero(steps > 0)) == 1
    assert float(np.delete(steps, int(np.argmax(steps))).max()) <= 0.0


# --------------------------------------------------------------------------
# Feasibility masking
#
# The measured failure this exists for: on a live run the policy emitted an
# order on 84% of turns and the simulator accepted 1% of them. It had settled on
# selling goods it did not own -- SELL is 88 of the 186 actions -- and could not
# leave, because nothing it could sell would exist until it first bought seed.
# --------------------------------------------------------------------------


def _mask_state(
    money: float = 3000.0,
    shed: dict | None = None,
    carried: dict | None = None,
    quadrants: tuple[str, ...] = ("NW",),
    hires_today: int = 0,
    inventory: dict | None = None,
    field: dict | None = None,
    day: int = 0,
    seeds: dict | None = None,
):
    farm = {
        "money": money,
        "unlocked_quadrants": list(quadrants),
        "hires_today": hires_today,
        # A real grid: the simulator's `_do_buy_land` walks it to unlock tiles,
        # and the ground-truth test probes that function directly.
        "tiles": [
            [None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)
        ],
        "hands": [],
        "farmer": [4, 4],
    }
    private = {
        "shed": dict(shed or {}),
        # Empty by default, which the land gate reads as "the plan still needs
        # seed", so a case has to opt in to a stocked farm to open that row.
        "seeds": dict(seeds or {}),
        "inventories": [dict(carried or {})],
    }
    # `day` defaults to 0 because that is the state the land gate exists to
    # forbid, so every case that does not name a day is asserting the closed door.
    obs = {"day": day, "market": {"prices": {}, "inventory": dict(inventory or {})}}
    return obs, farm, private, field or {"farmer": ["PASS"], "hands": []}


def _mask(**kwargs) -> np.ndarray:
    from agents.dqn_market_free.order_space import feasible_actions

    obs, farm, private, field = _mask_state(**kwargs)
    return feasible_actions(obs, farm, private, field, None)


def _rows(op: str, item: str | None = None) -> list[int]:
    return [
        index
        for index, order in enumerate(ACTIONS)
        if order.op == op and (item is None or order.item == item)
    ]


def test_the_mask_matches_the_simulator_action_by_action():
    """The definition of the function, checked against the environment itself.

    Everything downstream inherits this. If the mask and `_commit_unit` ever
    disagree the agent is either offered orders that do nothing -- the failure
    being fixed -- or denied orders the game would have taken, which is worse
    because it is invisible.

    Two definitions of "accepted", because the ladder has two kinds of rung.
    SELL and every BUY commit per unit and fill partially, so one unit committing
    is the whole test and quantity cannot matter. `HIRE n` is checked on all n
    hands committing: its cost is fibonacci in `hires_today` and rises *within*
    the order, so the rungs genuinely differ, and `HIRE k` is on the same ladder
    for a farm that can only afford k. Nothing the simulator would do is
    unreachable either way; see `feasible_actions`.
    """

    from copy import deepcopy

    from kaggle_environments.envs.kaggriculture import kaggriculture as sim

    from agents.dqn_market_free.order_space import feasible_actions

    cases = [
        dict(money=3000.0),
        dict(money=0.0),
        dict(money=9.0),
        dict(money=10.0),
        dict(money=300.0),
        dict(money=3000.0, shed={"WHEAT": 4, "MELON": 1}),
        dict(money=3000.0, shed={item: 12 for item in ("WHEAT", "CARROT", "MELON")}),
        dict(money=5000.0, quadrants=("NW", "NE")),
        dict(money=5000.0, quadrants=("NW", "NE", "SW", "SE")),
        dict(money=5000.0, hires_today=6),
    ]

    for case in cases:
        obs, farm, private, field = _mask_state(**case)
        mask = feasible_actions(obs, farm, private, field, None)
        market = {
            "inventory": {item: sim.MARKET_I0 for item in sim.PRODUCTS},
            "params": None,
        }
        for index, order in enumerate(ACTIONS):
            if order.op == "NOOP":
                continue
            probe_farm, probe_private = deepcopy(farm), deepcopy(private)
            probe_private.setdefault("shed", {})
            if order.op == "HIRE":
                hands = len(probe_farm["hands"])
                for _ in range(order.quantity):
                    sim._do_hire(probe_farm, probe_private, 10, 1)
                accepted = len(probe_farm["hands"]) == hands + order.quantity
            elif order.op == "BUY_LAND":
                # The one deliberate divergence. `_land_is_survivable` masks land
                # the simulator would happily sell, because affordability alone
                # offers 1,000 against the 3,000 stake on turn 0 and the measured
                # result is 190 coins. Asserting parity here would pin the very
                # behaviour that gate exists to remove, so the row is skipped and
                # covered by its own tests instead.
                continue
            else:
                if order.op == "SELL":
                    price = sim.market_price(order.item, market["inventory"][order.item], None)
                elif order.op == "BUY_PRODUCT":
                    price = sim.market_price(
                        order.item, market["inventory"][order.item] - 1, None
                    )
                elif order.op == "BUY_SEED":
                    price = sim.CROPS[order.item]["seed"]
                else:
                    price = sim.ANIMALS[order.item]["cost"]
                accepted = sim._commit_unit(
                    order.op, order.item, price, probe_farm, probe_private, market, 100
                )
            assert bool(mask[index]) == bool(accepted), (
                f"{case}: {order} mask={bool(mask[index])} simulator={bool(accepted)}"
            )


def test_selling_what_the_shed_does_not_hold_is_masked():
    """The exact behaviour that was measured, pinned."""

    empty = _mask()
    assert not empty[_rows("SELL")].any()
    # The three actions that were 59% of the emitted orders on the failing run.
    for op, item, quantity in (
        ("SELL", "FERTILIZER", 2),
        ("SELL", "STRAWBERRY", 24),
        ("SELL", "WHEAT", 6),
    ):
        assert not empty[order_index(op, item, quantity)]

    # One strawberry opens every strawberry quantity and nothing else: fills are
    # per-unit and partial, so quantity cannot affect feasibility.
    holding = _mask(shed={"STRAWBERRY": 1})
    assert holding[_rows("SELL", "STRAWBERRY")].all()
    assert not holding[_rows("SELL", "WHEAT")].any()


def test_the_mask_sees_the_shed_this_turn_will_fill():
    """Unit actions run before the market phase, so a pending DROP counts.

    `interpreter` applies every DROP/PLACE/PICKUP for both seats and only then
    calls `_process_market`. A hand carrying a harvested melon to the shed this
    turn makes SELL MELON commit while the observation still shows it empty --
    masking on the raw shed alone would forbid an order the game accepts.
    """

    carrying = _mask(carried={"MELON": 3}, field={"farmer": ["DROP"], "hands": []})
    assert carrying[_rows("SELL", "MELON")].all()


def test_quantity_never_changes_feasibility():
    """Per-unit commits with partial fills: one bit per (op, item), not per row.

    `HIRE` is the one exception and is checked separately below: its cost rises
    within the order, so the rungs are not interchangeable.
    """

    mask = _mask(money=137.0, shed={"WHEAT": 2})
    for op in ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"):
        for item in {order.item for order in ACTIONS if order.op == op}:
            rows = _rows(op, item)
            assert len(set(mask[rows].tolist())) == 1, f"{op} {item} splits on quantity"


def test_hire_is_the_one_rung_quantity_changes():
    """A prefix of the ladder, not all-or-nothing.

    137 coins covers 1+1+2+3+5+8+13+21+34 = 88 but not the tenth hand at 55.
    The mask must open exactly the affordable prefix: opening the whole ladder on
    the first hand's cost is what let one policy spend a 3,000 stake in four
    turns, and closing it entirely would deny hires the game would take.
    """

    rungs = _mask(money=137.0)[_rows("HIRE")].tolist()

    assert rungs == [hands <= 9 for hands in HIRE_QUANTITIES]


def test_buying_needs_money_and_shed_room():
    poor = _mask(money=5.0)
    assert not poor[_rows("BUY_SEED")].any()
    assert not poor[_rows("BUY_ANIMAL")].any()
    assert not poor[_rows("BUY_LAND")].any()
    # The first hand costs fib(0) = 1, so hiring survives on five coins -- but
    # only as far as the stake reaches: 1+1+2 buys three hands, the fourth is 3.
    assert poor[_rows("HIRE")].tolist() == [hands <= 3 for hands in HIRE_QUANTITIES]

    # Animals land in the shed and obey its capacity, exactly as products do.
    full = _mask(money=50_000.0, shed={"WHEAT": 100})
    assert not full[_rows("BUY_ANIMAL")].any()
    assert not full[_rows("BUY_PRODUCT")].any()


#: A farm that has already satisfied the land gate's readiness tests: mid-season,
#: seed on hand for every planned tile, cash to spare. Cases that want to probe
#: something *other* than readiness start from here.
_READY_FOR_LAND = {
    "day": 5,
    "seeds": {"MELON": 40, "WHEAT": 40, "CARROT": 40, "TOMATO": 40, "STRAWBERRY": 40},
}


def test_land_runs_out_and_hiring_gets_dearer():
    assert _mask(money=3000.0, quadrants=("NW",), **_READY_FOR_LAND)[_rows("BUY_LAND")].all()
    # Capped at two extra quadrants, tighter than the simulator's three: the
    # third costs 4,000 and nothing on record has come close to working two.
    assert not _mask(money=99_999.0, quadrants=("NW", "NE", "SW"), **_READY_FOR_LAND)[
        _rows("BUY_LAND")
    ].any()
    # Nothing left to buy at any price.
    assert not _mask(money=99_999.0, quadrants=("NW", "NE", "SW", "SE"), **_READY_FOR_LAND)[
        _rows("BUY_LAND")
    ].any()
    # fib(9) = 55, so nine hands deep the tenth needs 55 coins -- and an eleventh
    # in the same order needs 55 + 89 on top, which 55 does not cover.
    assert not _mask(money=54.0, hires_today=9)[_rows("HIRE")].any()
    assert _mask(money=55.0, hires_today=9)[_rows("HIRE")].tolist() == [
        hands <= 1 for hands in HIRE_QUANTITIES
    ]


def test_land_is_shut_until_the_farm_can_actually_work_it():
    """Each readiness test in `_land_is_survivable`, failed one at a time.

    Affordability alone is what produced the measured record: day-0 purchases
    finish on a mean of 190 coins across 2,124 seats, and 49.4% of all accepted
    purchases planted zero of their 25 new tiles.
    """

    rows = _rows("BUY_LAND")
    assert _mask(money=3000.0, **_READY_FOR_LAND)[rows].all(), "the ready farm must pass"

    # Day 0: affordable, and the whole of the bankruptcy tail.
    assert not _mask(money=3000.0, **{**_READY_FOR_LAND, "day": 0})[rows].any()

    # Too late to pay for itself. Land has no salvage value at the buzzer.
    assert not _mask(money=3000.0, **{**_READY_FOR_LAND, "day": 19})[rows].any()

    # Seed the plan still needs. Buying area before seed is how a quadrant ends
    # up with none of its 25 tiles planted.
    assert not _mask(money=3000.0, day=5, seeds={"MELON": 1})[rows].any()

    # Price alone is covered, a day's hiring on top is not.
    assert not _mask(money=1000.0, **_READY_FOR_LAND)[rows].any()
    assert _mask(money=1200.0, **_READY_FOR_LAND)[rows].all()


def test_the_land_gate_reads_the_plan_it_is_handed():
    """`main.py` passes the turn's effective roles; the fallback must agree.

    The seed test is against the *effective* plan, so a caller that has already
    built one has to be able to hand it over rather than have it silently
    recomputed from the donor's untouched map.
    """

    from agents.dqn_market_free.order_space import feasible_actions

    obs, farm, private, field = _mask_state(money=3000.0, **_READY_FOR_LAND)
    rows = _rows("BUY_LAND")

    passed = feasible_actions(obs, farm, private, field, None, roles=donor._role_plan(obs, farm))
    assert passed[rows].all()
    # Recomputed when absent, to the same answer.
    assert feasible_actions(obs, farm, private, field, None)[rows].all()

    # A plan wanting a crop the farm holds no seed for shuts the row, even though
    # nothing about the money changed.
    hungry = {position: ("CROP", "MELON") for position in list(donor._role_plan(obs, farm))[:5]}
    private_dry = {**private, "seeds": {}}
    assert not feasible_actions(obs, farm, private_dry, field, None, roles=hungry)[rows].any()


def test_the_mask_is_never_empty_and_survives_a_malformed_observation():
    """`main.agent` swallows four exceptions an episode; the mask must not spend one."""

    from agents.dqn_market_free.order_space import feasible_actions

    for farm, private in (
        ({}, {}),
        ({"money": 0}, {"shed": {}}),
        ({"money": None}, {"shed": None, "inventories": None}),
    ):
        mask = feasible_actions({}, farm, private, {}, None)
        assert mask[0], "NOOP must always be available"
        assert mask.any()

    broke = _mask(money=0.0)
    assert broke[0] and int(broke.sum()) == 1


def test_the_opening_turn_closes_selling_and_leaves_buying_open():
    """One number for the whole pruning, so a regression is visible."""

    mask = _mask()
    assert int(mask.sum()) == 97
    # 97 not 98: land is shut on the opening turn now. It was the 1,000-coin
    # offer against the 3,000 stake, and taking it finished on 190.
    assert not mask[_rows("BUY_LAND")].any()
    assert not mask[_rows("SELL")].any()
    assert mask[_rows("BUY_SEED")].all()
    assert mask[_rows("HIRE")].all()


def test_greedy_choice_skips_a_masked_peak():
    from agents.dqn_market_free.order_space import feasible_actions

    obs, farm, private, field = _mask_state()
    mask = feasible_actions(obs, farm, private, field, None)
    forbidden = _rows("SELL", "WHEAT")[0]
    allowed = order_index("BUY_SEED", "WHEAT", 6)
    assert not mask[forbidden] and mask[allowed]

    class PeakedOnForbidden:
        def __call__(self, state):
            scores = np.zeros(ACTION_COUNT, dtype=np.float32)
            scores[forbidden] = 100.0
            scores[allowed] = 1.0
            return scores

    decision = choose_order(np.zeros(STATE_DIM, dtype=np.float32), PeakedOnForbidden(), mask=mask)
    assert decision.chosen == allowed


def test_the_masked_choice_agrees_between_numpy_and_torch():
    """Decision-level parity, which only holds because both use MASKED_SCORE.

    An `-inf` on one side and a finite sentinel on the other would break ties
    differently and split the acting policy from the one being trained.
    """

    from agents.dqn_market_free.order_space import MASKED_SCORE

    rng = np.random.default_rng(0)
    module = net_module.build()
    network = net_module.to_numpy(module)
    for _ in range(50):
        state = rng.standard_normal(STATE_DIM).astype(np.float32)
        mask = rng.random(ACTION_COUNT) < 0.4
        mask[0] = True
        expected = int(
            torch.from_numpy(np.asarray(network(state)))
            .masked_fill(~torch.from_numpy(mask), MASKED_SCORE)
            .argmax()
        )
        assert choose_order(state, network, mask=mask).chosen == expected


def test_a_masked_bootstrap_stays_finite():
    """The nan trap: a masked score reaches a multiply by a zero discount.

    Terminal rows carry `discount == 0`. With an `-inf` sentinel the bootstrap
    is `-inf * 0.0`, which is nan rather than zero, and one nan takes the whole
    minibatch down through the backward pass.
    """

    from agents.dqn_market_free.order_space import MASKED_SCORE

    assert np.isfinite(MASKED_SCORE)

    scores = torch.zeros(4, ACTION_COUNT)
    empty = torch.zeros(4, ACTION_COUNT, dtype=torch.bool)
    bootstrap = scores.masked_fill(~empty, MASKED_SCORE).max(dim=1).values
    discount = torch.zeros(4)
    goal = torch.zeros(4) + discount * bootstrap
    assert torch.isfinite(goal).all()


def test_the_batch_carries_the_next_state_mask():
    buffer = ReplayBuffer(capacity=16, seed=0, n_step=2)
    buffer.add([_synthetic_record(30, 5_000.0, seed=index) for index in range(4)])
    batch = buffer.sample(8)

    assert batch.mask.shape == (8, ACTION_COUNT)
    assert batch.next_mask.shape == (8, ACTION_COUNT)
    assert batch.mask.dtype == np.bool_
    # Rows whose episode ended have no next state; their mask must stay all-True
    # so the masked max cannot manufacture a nan against a zero discount.
    terminal = batch.discount == 0.0
    if terminal.any():
        assert batch.next_mask[terminal].all()


def test_the_recorded_mask_is_the_one_the_policy_acted_on():
    """Store-don't-recompute, paid off.

    The chosen action being feasible under its own stored mask is what keeps the
    clone's cross-entropy target off a masked class and the TD gather on an
    action the argmax can actually reach.
    """

    from agents.dqn_market_free.training.collect import Recorder

    recorder = Recorder()
    agent = MarketFreeAgent(
        initial_network({}), rng=np.random.default_rng(0), recorder=recorder
    )
    obs, farm, private = _fresh_market_free_state({"WHEAT": 4})
    for step in range(5):
        obs["step"] = step
        agent(obs, None)

    record = recorder.finish(3_000.0, 3_000.0, 0.999)
    assert record.order_mask.shape == (record.decisions, ACTION_COUNT)
    for index in range(record.decisions):
        assert record.order_mask[index][int(record.order_chosen[index])], (
            "the policy acted on an action its own mask forbade"
        )
        assert record.order_mask[index][0], "NOOP must survive in every stored mask"


def test_the_episode_artifact_round_trips_the_mask(tmp_path: Path):
    from agents.dqn_market_free.training.collect import load_episode_batch, save_episode_batch

    record = _synthetic_record(12, 5_000.0, seed=1, forbidden=[3, 9, 40])
    path = save_episode_batch([record], tmp_path / "batch.npz")
    (restored,) = load_episode_batch(path)

    assert np.array_equal(restored.order_mask, record.order_mask)
    assert not restored.order_mask[:, 3].any()


def test_a_per_action_scale_widens_only_that_action():
    """The whole mechanism in one assertion.

    Uniform sigma cannot express "try this row more often"; a per-output vector
    on `epsilon_out` can, because that vector is already the per-action half of
    the factored perturbation.
    """

    from agents.dqn_market_free.order_space import NOOP_INDEX

    land = order_index("BUY_LAND", None, 0)
    module = net_module.build(noisy="head", sigma0=0.5)
    network = net_module.to_numpy(module)
    scale = np.ones(ACTION_COUNT, dtype=np.float32)
    scale[land] = 8.0

    rng = np.random.default_rng(0)
    state = rng.standard_normal(STATE_DIM).astype(np.float32)
    quiet = np.asarray(network(state)).copy()

    spread = np.zeros(ACTION_COUNT)
    for _ in range(200):
        network.sample_noise(rng, scale)
        spread += np.abs(np.asarray(network(state)) - quiet)
    spread /= 200

    others = np.delete(spread, [land, NOOP_INDEX])
    assert spread[land] > 4.0 * others.mean(), "the widened row is not actually wider"
    assert others.std() / max(1e-9, others.mean()) < 0.5, "the other 185 rows moved too"


def test_the_per_action_scale_agrees_between_numpy_and_torch():
    """Parity, on the vector path as well as the scalar one.

    The collector perturbs through `policy.py` and the flip-rate probe through
    `training/network.py`. If the two read a per-action scale differently, the
    controller measures an exploration rate no actor ever played.
    """

    land = order_index("BUY_LAND", None, 0)
    module = net_module.build(noisy="head", sigma0=0.5)
    network = net_module.to_numpy(module)
    scale = np.ones(ACTION_COUNT, dtype=np.float32)
    scale[land] = 6.0

    state = np.random.default_rng(3).standard_normal(STATE_DIM).astype(np.float32)
    epsilon_in = np.random.default_rng(4).standard_normal(128).astype(np.float32)
    epsilon_out = np.random.default_rng(5).standard_normal(ACTION_COUNT).astype(np.float32)

    head = [layer for layer in module.modules() if isinstance(layer, net_module.NoisyLinear)][-1]
    head.epsilon_in = torch.from_numpy(epsilon_in)
    head.epsilon_out = torch.from_numpy(epsilon_out * scale)
    expected = module(torch.from_numpy(state)).detach().numpy()

    network._noise = [None] * len(network.layers)
    network._noise[-1] = (epsilon_in, epsilon_out * scale)
    assert np.allclose(np.asarray(network(state)), expected, atol=1e-5)


def test_a_scale_vector_skips_layers_it_does_not_fit():
    """`noisy_layers = "all"` must not receive the 186-wide action vector."""

    module = net_module.build(noisy="all", sigma0=0.5)
    scale = np.full(ACTION_COUNT, 5.0, dtype=np.float32)
    net_module.reset_noise(module, scale=scale)

    layers = [layer for layer in module.modules() if isinstance(layer, net_module.NoisyLinear)]
    hidden, head = layers[0], layers[-1]
    assert hidden.out_features != ACTION_COUNT
    assert float(hidden.epsilon_out.abs().max()) < 4.0, "the hidden layer took the action vector"
    assert head.out_features == ACTION_COUNT


def test_the_sigma_scale_is_never_below_one():
    """It may only add variance to a row, never remove it."""

    from agents.dqn_market_free.training import rare

    chosen, opportunities = rare.visit_counts([_synthetic_record(200, 10.0, seed=6)])
    scale = rare.sigma_scale_vector(chosen, opportunities, gain=1.0, cap=3.0)
    assert scale.min() >= 1.0
    assert scale.max() <= 3.0


def test_the_widening_targets_the_rows_the_run_has_not_sampled():
    from agents.dqn_market_free.order_space import NOOP_INDEX
    from agents.dqn_market_free.training import rare

    land = order_index("BUY_LAND", None, 0)
    chosen = np.full(ACTION_COUNT, 500.0)
    chosen[land] = 0.0
    opportunities = np.ones(ACTION_COUNT)
    never_feasible = order_index("SELL", "WHEAT", 45)
    opportunities[never_feasible] = 0.0

    scale = rare.sigma_scale_vector(chosen, opportunities, gain=1.0, cap=3.0)
    assert scale[land] > scale[order_index("HIRE", None, 1)]
    assert scale[NOOP_INDEX] == 1.0, "extra mass on the collapse attractor"
    assert scale[never_feasible] == 1.0, (
        "a row the mask always refuses would arrive at full width the moment it opens"
    )


def test_the_widening_decays_as_a_row_accumulates_visits():
    """Self-annealing, which is why there is no schedule to tune."""

    from agents.dqn_market_free.training import rare

    land = order_index("BUY_LAND", None, 0)
    opportunities = np.ones(ACTION_COUNT)
    unsampled = np.full(ACTION_COUNT, 500.0)
    unsampled[land] = 0.0
    sampled = np.full(ACTION_COUNT, 500.0)

    wide = rare.sigma_scale_vector(unsampled, opportunities, 1.0, 3.0)[land]
    narrow = rare.sigma_scale_vector(sampled, opportunities, 1.0, 3.0)[land]
    assert wide > narrow >= 1.0


def test_the_visit_counts_age_toward_the_recent_window():
    from agents.dqn_market_free.training import rare

    early = _synthetic_record(40, 10.0, seed=1, chosen=order_index("HIRE", None, 1))
    late = _synthetic_record(40, 10.0, seed=2, chosen=order_index("BUY_SEED", "WHEAT", 6))
    counts = rare.fold(None, rare.visit_counts([early]), 0.5)
    for _ in range(5):
        counts = rare.fold(counts, rare.visit_counts([late]), 0.5)

    hire = counts[0][order_index("HIRE", None, 1)]
    seed = counts[0][order_index("BUY_SEED", "WHEAT", 6)]
    assert seed > hire, "a stale round still dominates the counts"
    assert hire > 0.0, "decay is not deletion"


def test_the_rollout_spec_carries_the_gain_only_when_it_is_on(tmp_path: Path):
    """Spec bytes are the policy identity; an always-on key churns revisions."""

    from agents.dqn_market_free.training.collect import learner_spec

    off = learner_spec(tmp_path / "w.npz", gamma=0.999, exploration_seed=3)
    assert "rare_sigma" not in off

    on = learner_spec(
        tmp_path / "w.npz",
        gamma=0.999,
        exploration_seed=3,
        rare_sigma_gain=1.0,
        rare_sigma_cap=3.0,
        rare_sigma_path=tmp_path / "visits.npz",
    )
    assert "rare_sigma_gain=1.000000" in on and "rare_sigma_cap=3.000000" in on


def test_the_rollout_spec_carries_the_opponent_weight_only_when_it_is_on(tmp_path: Path):
    """Same rule as the gain, and it is what keeps the leg free while it is inert.

    At `OPPONENT_WEIGHT = 0.0` every episode passes 0.0, so every spec string is
    byte-identical to what this agent emitted before the parameter existed and no
    revision churns on landing the mechanism.
    """

    from agents.dqn_market_free.training.collect import learner_spec

    off = learner_spec(tmp_path / "w.npz", gamma=0.999, exploration_seed=3)
    assert "opponent_weight" not in off

    on = learner_spec(
        tmp_path / "w.npz", gamma=0.999, exploration_seed=3, opponent_weight=0.1
    )
    assert "opponent_weight=0.100000" in on


def test_the_opponent_weight_stays_out_of_the_policy_identity(tmp_path: Path):
    """It is per-opponent, and the identity spec must be per-policy.

    Folding it in would mint two revisions for one set of weights -- one for the
    mirror and one for everything else -- splitting `rating_snapshots` and the
    self-play pool down the middle. `_learner_policy` is what the identity is
    actually built from, so this pins that it never grew the key.
    """

    from urllib.parse import parse_qs, urlsplit

    from agents.dqn_market_free.training.updater import _learner_policy

    weights = tmp_path / "w.npz"
    policy = _learner_policy(weights, 0.999)

    # Parsed rather than substring-matched: the weights path is in the spec and
    # can contain anything, including this key's own name.
    query = parse_qs(urlsplit(policy.agent_spec).query)
    assert "opponent_weight" not in query


def test_the_mirror_is_gated_out_of_the_opponent_leg(monkeypatch, tmp_path: Path):
    """A self-play opponent scores absolutely; a scripted one carries the weight.

    Against a recent checkpoint of the same slot the relative leg is noise by
    construction -- the opponent's purse carries this policy's own variance and
    nothing else. `market-free-population-v25` is the run that measured what
    happens without this: at `self_play_ratio = 0.65`, two of every three
    rollouts would carry a zero-mean term worth only what it adds in noise.
    """

    from agents.dqn_market_free.training import collect as collect_module

    monkeypatch.setattr(collect_module, "OPPONENT_WEIGHT", 0.1)

    seen: list[str] = []

    def fake_learner_spec(_weights, **kwargs):
        seen.append(f"{kwargs['opponent_weight']:.6f}")
        # Stop before `run_match`: the gate is decided by the time the spec is
        # built, which is the whole of what this reads.
        raise RuntimeError("stop")

    monkeypatch.setattr(collect_module, "learner_spec", fake_learner_spec)

    mirror = f"{collect_module.SELF_SPEC}?weights={tmp_path / 'ckpt.npz'}"
    for opponent in (mirror, "agents/competitor_pilkwang/main.py"):
        with pytest.raises(RuntimeError):
            collect_module.collect_episode(
                tmp_path / "w.npz", opponent, 7, 0, 0.999, tmp_path, 0
            )

    assert seen == ["0.000000", "0.100000"]


def test_the_widening_rides_the_widest_end_of_the_ladder(monkeypatch, tmp_path: Path):
    """The widest rungs already exist to find things; the greedy tail stays clean.

    The money series is only comparable across this change while some pairs
    collect what the run before it collected.
    """

    from agents.dqn_market_free.training import collect as collect_module

    seen: list[tuple[float, float]] = []

    def fake_job(payload):
        seen.append((payload[11], payload[12]))
        return None

    monkeypatch.setattr(collect_module, "_job", fake_job)
    collect_module.collect_batch(
        tmp_path / "w.npz",
        ["opponent"],
        list(range(16)),
        0.999,
        tmp_path,
        jobs=1,
        sigma_scales=[1.0, 0.5, 0.25, 0.125, 0.06, 0.03, 0.015, 0.008],
        rare_sigma_gain=1.0,
        rare_sigma_cap=3.0,
        rare_sigma_path=tmp_path / "visits.npz",
        rare_sigma_pairs=3,
    )

    carried = {sigma: gain for sigma, gain in seen}
    assert carried[1.0] == 1.0, "the widest rung is the arm whose job is finding things"
    assert carried[0.008] == 0.0, "the greedy tail must stay a clean read"
    assert sum(1 for value in carried.values() if value > 0.0) == 3
    assert all(value in (0.0, 1.0) for value in carried.values()), (
        "the gain must not be scaled by the rung's own sigma"
    )


def test_the_counts_round_trip_and_a_stale_width_is_dropped(tmp_path: Path):
    from agents.dqn_market_free.training import rare

    chosen, opportunities = rare.visit_counts([_synthetic_record(40, 10.0, seed=4)])
    path = tmp_path / "visits.npz"
    rare.save(path, chosen, opportunities)
    restored = rare.load(path)
    assert restored is not None
    assert_array_equal(restored[0], chosen)

    assert rare.load(tmp_path / "absent.npz") is None
    np.savez(tmp_path / "stale.npz", chosen=np.zeros(3), opportunities=np.zeros(3))
    assert rare.load(tmp_path / "stale.npz") is None, (
        "counts from a different action space would misattribute every widening"
    )


# --- The behaviour archive -------------------------------------------------
#
# The mature group used to be a ranking, and a ranking cannot hold a population
# apart. `logmoney-v26` collapsed from 3.66 distinct crops per seat at generation
# 4 to 1.29 by generation 15 and locked onto wheat, on *rising* money, so neither
# the promotion gate nor `_population_score`'s activity discount had anything to
# object to. These cover the reserved berths that are supposed to survive it.


def _archive_members(*rows):
    """`(lineage, score, distinct_crops, land_per_episode, money)` per member."""

    return [
        {
            "member_id": f"m{index}",
            "model_id": f"decl@{lineage}",
            "checkpoint_id": f"ckpt-{index}",
            "score": score,
            "inference_artifact_path": f"/tmp/ckpt-{index}.npz",
            "behavior": {
                "episodes": 50,
                "distinct_crops": crops,
                "land_per_episode": land,
                "money": money,
            },
        }
        for index, (lineage, score, crops, land, money) in enumerate(rows)
    ]


def test_the_archive_bins_a_fingerprint_onto_two_axes():
    cell = updater_module._behavior_cell

    assert cell({"episodes": 4, "distinct_crops": 3, "land_per_episode": 0.0}) == (3, 0)
    # Both axes are capped at their top edge, so the highest bin is open-ended and
    # a policy that plants five crops does not fall off the grid.
    assert cell({"episodes": 4, "distinct_crops": 5, "land_per_episode": 4.0}) == (3, 2)
    # Land rounds rather than truncates: a policy buying a quadrant in most of its
    # episodes is running the expansion strategy, and truncation would file it with
    # the policies that never buy one.
    assert cell({"episodes": 4, "distinct_crops": 2, "land_per_episode": 0.6}) == (2, 1)
    assert cell({"episodes": 4, "distinct_crops": 2, "land_per_episode": 0.4}) == (2, 0)
    # Unmeasured is not zero. A member with no fingerprint has to stay out of the
    # grid entirely rather than landing in the no-crop, no-land corner.
    assert cell(None) is None
    assert cell({"distinct_crops": 3}) is None, "no episodes means nothing was measured"


def test_the_monoculture_cell_is_never_reserved():
    """The attractor already owns the score bucket; a berth there buys nothing.

    `(1, 0)` is what `logmoney-v26` converged on and held for twenty-six
    generations. Reserving for it would spend one of six berths on the thing the
    other twenty-four already hold.
    """

    mono = {"episodes": 9, "distinct_crops": 1, "land_per_episode": 0.0}
    assert updater_module._behavior_cell(mono) == (1, 0)
    assert updater_module._reserved_cell(mono) is None
    # One step off it in either direction is reservable.
    assert updater_module._reserved_cell({**mono, "distinct_crops": 2}) == (2, 0)
    assert updater_module._reserved_cell({**mono, "land_per_episode": 1.0}) == (1, 1)


def test_the_archive_holds_one_berth_per_cell_best_first():
    """Occupancy is decided once, off the score-ordered list the cull reads."""

    members = _archive_members(
        ("a", 900.0, 1, 0.0, 3000.0),   # the attractor, never reserved
        ("b", 800.0, 3, 1.0, 2400.0),   # best in (3, 1)
        ("c", 700.0, 3, 1.0, 2200.0),   # same cell, beaten by b
        ("d", 600.0, 2, 0.0, 1900.0),   # best in (2, 0)
    )
    marked = updater_module._mark_reserved(members)

    assert [member["cell"] for member in marked] == [None, (3, 1), (3, 1), (2, 0)]
    assert [member["reserved"] for member in marked] == [False, True, False, True]


def test_a_reserved_draw_reaches_what_the_ranking_would_have_culled():
    """The archive is only worth filling if something reads it.

    Ranked on score the diversified members here are last, so `reseed_source_pool`
    never sees them. The explorer share is the path that does.
    """

    members = updater_module._mark_reserved(
        _archive_members(
            ("a", 900.0, 1, 0.0, 3000.0),
            ("b", 880.0, 1, 0.0, 2900.0),
            ("c", 860.0, 1, 0.0, 2800.0),
            ("d", 200.0, 3, 1.0, 1800.0),
        )
    )
    config = {"reseed_source_pool": 3, "reseed_source_lineage_cap": 1}

    ranked = {
        updater_module._reseed_source(
            members, config, seed=5, reseed_index=index
        )["checkpoint_id"]
        for index in range(40)
    }
    assert "ckpt-3" not in ranked, "the score bucket cannot reach it, which is the problem"

    mixed = {
        updater_module._reseed_source(
            members,
            {**config, "reseed_source_explorer_share": 0.25},
            seed=5,
            reseed_index=index,
        )["checkpoint_id"]
        for index in range(40)
    }
    assert "ckpt-3" in mixed, "the reserve is reachable"
    assert mixed - {"ckpt-3"}, "and the ranking still supplies most restarts"


def test_the_explorer_share_defaults_to_the_pre_archive_draw():
    """0.0 is byte-identical to the draw before the archive existed.

    The migration safety property, and the retirement path: one key returns the
    reseed to exactly what every run through v27 measured.
    """

    members = updater_module._mark_reserved(
        _archive_members(
            ("a", 900.0, 1, 0.0, 3000.0),
            ("b", 880.0, 3, 2.0, 2900.0),
            ("c", 860.0, 2, 1.0, 2800.0),
        )
    )
    config = {"reseed_source_pool": 3, "reseed_source_lineage_cap": 1}
    bare = [
        updater_module._reseed_source(members, config, seed=4, reseed_index=index)
        for index in range(25)
    ]
    zeroed = [
        updater_module._reseed_source(
            members, {**config, "reseed_source_explorer_share": 0.0}, seed=4, reseed_index=index
        )
        for index in range(25)
    ]
    assert [m["checkpoint_id"] for m in bare] == [m["checkpoint_id"] for m in zeroed]


def test_the_lineage_cap_applies_inside_the_reserve_too():
    """A reserve of one lineage's own history is the same failure, one level down."""

    members = updater_module._mark_reserved(
        _archive_members(
            ("hot", 900.0, 3, 1.0, 2500.0),   # (3, 1)
            ("hot", 880.0, 2, 1.0, 2400.0),   # (2, 1), same lineage
            ("cold", 300.0, 2, 0.0, 2000.0),  # (2, 0)
        )
    )
    config = {
        "reseed_source_pool": 3,
        "reseed_source_lineage_cap": 1,
        "reseed_source_explorer_share": 1.0,
    }
    drawn = {
        updater_module._reseed_source(members, config, seed=2, reseed_index=index)["model_id"]
        for index in range(30)
    }
    assert drawn == {"decl@hot", "decl@cold"}, "one entry per lineage, inside the reserve"


def test_a_whole_roster_warm_start_hands_the_reserve_to_its_tail():
    """Deterministic, because ten slots choose at once with nothing to coordinate on.

    The tail takes the archive so the slots drawing archived ancestors are the same
    ones `train_market_free.py` gives `rare_sigma_gain` to generate them.
    """

    members = updater_module._mark_reserved(
        _archive_members(
            ("a", 900.0, 1, 0.0, 3000.0),
            ("b", 800.0, 1, 0.0, 2900.0),
            ("c", 700.0, 1, 0.0, 2800.0),
            ("d", 600.0, 3, 1.0, 2000.0),
            ("e", 500.0, 2, 1.0, 1900.0),
        )
    )
    config = {"reseed_source_lineage_cap": 1, "reseed_source_explorer_share": 0.4}
    chosen = [
        updater_module._reseed_source(
            members, config, seed=8, reseed_index=0, ordinal=ordinal, roster=5, spread=True
        )
        for ordinal in range(5)
    ]

    assert [member["reserved"] for member in chosen] == [False, False, False, True, True]
    assert [member["checkpoint_id"] for member in chosen[3:]] == ["ckpt-3", "ckpt-4"]
    # Same slot, same answer -- a warm start stays as answerable as a draw.
    again = updater_module._reseed_source(
        members, config, seed=8, reseed_index=0, ordinal=4, roster=5, spread=True
    )
    assert again["checkpoint_id"] == "ckpt-4"


def test_the_cull_splits_capacity_and_floors_the_reserve():
    """The floor is the whole safety argument for reserving anything.

    Land buyers finish on a median of 7 coins against 3,000 for the rest, so an
    unfloored cell would hold day-0 bankrupts permanently and reseed the roster
    from them -- worse than the collapse it is preventing.
    """

    statements: list[tuple[str, list]] = []

    class Reader:
        def fetchall(self, sql, parameters):
            statements.append((sql, list(parameters)))
            return [("member",)] if "INSERT INTO" in sql else []

    behavior = {
        "episodes": 50,
        "distinct_crops": 3,
        "land_per_episode": 1.0,
        "money": 2400.0,
        "crop_bin": 3,
        "land_bin": 1,
    }
    result = updater_module._graduate(
        Reader(),
        "decl",
        "decl@streak_08",
        "ckpt-a",
        145.8,
        30,
        behavior=behavior,
        reserved=6,
        reserved_min_money=1500.0,
    )

    insert, cull = statements
    assert json.loads(insert[1][-1])["crop_bin"] == 3
    # Split, not extended: 24 berths to the ranking and 6 to the cells, so the
    # group's size and its mirror-opponent cost are unchanged.
    assert cull[1] == ["decl", 24, 1, 0, 1500.0, 6]
    assert "DISTINCT ON (crop_bin, land_bin)" in cull[0]
    assert "money >= " in cull[0]
    assert result["cell"] == [3, 1]
    assert result["reserved_capacity"] == 6


def test_an_unmeasured_member_keeps_the_fingerprint_it_already_had():
    """Re-entry happens on every update of a standing release.

    `logmoney-v26` logged 87 entry events for 37 members. An update whose arena
    payload lacks the fingerprint must not blank a column the archive is keying
    berths on.
    """

    statements: list[tuple[str, list]] = []

    class Reader:
        def fetchall(self, sql, parameters):
            statements.append((sql, list(parameters)))
            return [("member",)] if "INSERT INTO" in sql else []

    updater_module._graduate(
        Reader(), "decl", "decl@streak_08", "ckpt-a", 145.8, 30, behavior=None
    )

    insert = statements[0][0]
    assert insert.count("coalesce(") == 1
    assert "population_members.behavior_json" in insert
    assert statements[0][1][-1] is None


def test_the_reserved_capacity_cannot_take_the_whole_group():
    """An archive holding every berth means nothing enters on merit at all."""

    base = {"lr_graduate_cut": 0.0, "mature_capacity": 30}
    assert updater_module._slot({**base, "mature_reserved_capacity": 6}).reserved_capacity == 6
    for nonsense in (30, 31, -1):
        with pytest.raises(ValueError, match="mature_reserved_capacity"):
            updater_module._slot({**base, "mature_reserved_capacity": nonsense})
    # Defaulted off, and defaulted to the floor this agent already uses for
    # "traded a season and did not die" rather than to a new number.
    spec = updater_module._slot(base)
    assert spec.reserved_capacity == 0
    assert spec.reserved_min_money == pytest.approx(
        updater_module.VIABILITY_DEFAULTS["viability_cold_start_reference_money"]
    )


def test_the_fingerprint_counts_choices_not_effects():
    """The opposite rule to `diagnose_replay`, and deliberately so.

    That function measures whether the policy is doing anything, so a refused order
    has to count as nothing. This one measures *which* strategy it is running, and
    a refused `BUY_LAND` is still the policy asking for land -- which is exactly the
    signal the archive must not lose.
    """

    replay = {
        "steps": [
            [{"observation": {"player": 0, "farms": [{"money": 3000}]}}],
            [
                {
                    "observation": {"player": 0, "farms": [{"money": 3000}]},
                    "action": {"market": [["BUY_LAND"]]},
                }
            ],
            [
                {
                    "observation": {"player": 0, "farms": [{"money": 2900}]},
                    "action": {"market": [["BUY_SEED", "WHEAT", 4]]},
                }
            ],
        ]
    }

    counts = diagnostics.behavior_from_replay(replay, 0)
    assert counts.land == 1, "money never moved, so the purchase was refused"
    assert counts.seeds == {"WHEAT": 4}
    assert counts.distinct_crops == 1
    assert counts.top_crop_share == 1.0
    # And the activity reading of the same replay still counts by effect.
    assert diagnostics.diagnose_replay(replay, 0).effective_decisions == 1
    # The land-*use* columns take the opposite rule again: they exist to price a
    # purchase, and a refused one bought no tiles, so none of them may fire here.
    assert counts.land_accepted == 0
    assert counts.land_episodes == 0
    assert counts.land_tiles_planted == 0


def _land_replay(*, planted: bool) -> dict:
    """A 4x4 board that unlocks NE on step 1, so a quadrant is a 2x2 corner.

    Two workers throughout, so the idle split has something to divide. The
    farmer stands on NE's weed at (2, 0); the hand stays home in NW, which is
    what makes `land_weed_digs` provably quadrant-scoped rather than a global
    DIG count.
    """

    weed = {"kind": "WEED"}
    crop = {"kind": "PLANT", "crop": "WHEAT"}

    def tiles(ne_corner):
        return [
            [None, None, ne_corner, None],
            [None, None, None, None],
            ["LOCKED"] * 4,
            ["LOCKED"] * 4,
        ]

    def locked_tiles():
        rows = [[None, None, "LOCKED", "LOCKED"] for _ in range(2)]
        return rows + [["LOCKED"] * 4 for _ in range(2)]

    def step(day, money, quadrants, board, action):
        return [
            {
                "observation": {
                    "player": 0,
                    "day": day,
                    "farms": [
                        {
                            "money": money,
                            "unlocked_quadrants": list(quadrants),
                            "tiles": board,
                            "farmer": [2, 0],
                            "hands": [[0, 0]],
                        }
                    ],
                },
                "action": action,
            }
        ]

    def act(farmer, hand, market=()):
        return {"farmer": [farmer], "hands": [[hand]], "market": [list(o) for o in market]}

    return {
        "steps": [
            step(3, 3000, ["NW"], locked_tiles(), None),
            # The purchase itself. Day and cash must be read from the step above.
            step(3, 2000, ["NW", "NE"], tiles(weed), act("PASS", "PASS", [["BUY_LAND"]])),
            # Farmer digs NE's weed; the hand passes in NW and must not count.
            step(4, 2000, ["NW", "NE"], tiles(crop if planted else None), act("DIG", "PASS")),
            step(4, 2000, ["NW", "NE"], tiles(crop if planted else weed), act("WATER", "PASS")),
        ]
    }


def test_land_use_prices_an_accepted_purchase():
    """`land` says the policy asked; these say what the ask bought.

    The distinction is the whole point: a money curve that falls when land is
    bought has two incompatible explanations -- expansion is negative inside 30
    days, or the field scheduler never works the tiles -- and only the second is
    fixable in the scheduler.
    """

    counts = diagnostics.behavior_from_replay(_land_replay(planted=True), 0)

    assert counts.land == 1, "one BUY_LAND was emitted"
    assert counts.land_accepted == 1, "and NE actually appeared, so it was accepted"
    assert counts.land_episodes == 1
    # Read from the step *before* the quadrant appears -- the state the policy
    # decided from, not the one its own purchase already drained.
    assert counts.land_day_sum == 3
    assert counts.land_money_sum == pytest.approx(3000.0)
    assert counts.land_tiles_planted == 1
    assert counts.land_weed_digs == 1, "the hand's PASS in NW is not a dig in NE"

    # Idle is the PASS share either side of the purchase, matching
    # `episode_metrics`. Before: both workers pass. After: the farmer works both
    # turns while the hand passes both.
    assert counts.land_idle_delta_sum == pytest.approx(0.5 - 1.0)


def test_land_bought_and_never_worked_reads_as_unworked():
    """The failure this column exists to name, and it must not read as absent.

    Every high-money land buyer in the league database plants 21 of NW's 25
    tiles and 0 of the quadrant it paid for. That has to be distinguishable from
    never having bought land at all, which `buy_land_rate` alone cannot do.
    """

    counts = diagnostics.behavior_from_replay(_land_replay(planted=False), 0)

    assert counts.land_accepted == 1, "the purchase still happened"
    assert counts.land_tiles_planted == 0, "and bought exactly nothing that got planted"
    assert counts.land_weed_digs == 1, "while still costing labour to keep clear"


def test_land_use_folds_across_a_paired_arena_series():
    """Sums, not means, so the fold is associative over seats and episodes."""

    bought = diagnostics.BehaviorCounts(
        episodes=1,
        land=1,
        land_accepted=2,
        land_day_sum=14,
        land_money_sum=4000.0,
        land_tiles_planted=30,
        land_weed_digs=6,
        land_idle_delta_sum=0.2,
        land_episodes=1,
    )
    total = bought + diagnostics.BehaviorCounts(episodes=1)
    payload = total.as_dict()

    # Per accepted purchase: a seat that bought twice made two timing decisions.
    assert payload["land_day"] == pytest.approx(7.0)
    assert payload["land_money"] == pytest.approx(2000.0)
    assert payload["land_tiles_planted_per_purchase"] == pytest.approx(15.0)
    # Per *buying* episode, so the seat that never expanded cannot dilute it.
    assert payload["land_weed_digs_per_episode"] == pytest.approx(6.0)
    assert payload["post_land_idle_delta"] == pytest.approx(0.2)
    # And a series with no purchases at all divides by zero nowhere.
    assert diagnostics.BehaviorCounts(episodes=4).as_dict()["land_day"] == 0.0


def test_fingerprints_sum_across_a_paired_arena_series():
    total = diagnostics.BehaviorCounts(episodes=1, land=1, seeds={"WHEAT": 3})
    total += diagnostics.BehaviorCounts(episodes=1, land=0, seeds={"WHEAT": 1, "MELON": 6})

    assert total.as_dict()["land_per_episode"] == 0.5
    assert total.distinct_crops == 2
    assert total.top_crop_share == pytest.approx(6 / 10)
    # Nothing planted reads as unconcentrated, so a policy that never buys a seed
    # cannot occupy the monoculture cell the archive refuses to reserve.
    assert diagnostics.BehaviorCounts(episodes=1).top_crop_share == 0.0


def test_the_count_novelty_cannot_see_an_abandoned_row():
    """The blind spot that motivated the rate arm, reproduced on v26's real shape.

    `logmoney-v26` folded 1,150,400 selections across ten slots. `log1p` of that is
    13.96, so the count statistic's numerator is effectively constant and its whole
    spread lives in `1 + n_a` -- a row feasible on 883,000 turns and chosen on 1,917
    of them earns a multiplier indistinguishable from no widening.
    """

    opportunities = np.full(ACTION_COUNT, 900_000.0)
    wheat, melon = 1, 2
    land = updater_module.LAND_INDEX
    # Bulk first, so `total` matches the run rather than just the three rows the
    # assertions read, then the three rows on top of it.
    chosen = np.full(ACTION_COUNT, (1_150_400 - 46_588.0) / (ACTION_COUNT - 3))
    chosen[wheat], chosen[melon], chosen[land] = 44_610.0, 1_917.0, 61.0

    counted = rare.sigma_scale_vector(chosen, opportunities, gain=1.0, cap=3.0)
    assert counted[melon] < 1.1, "the abandoned crop earns essentially nothing"
    assert counted[updater_module.LAND_INDEX] < 1.6, "and the never-chosen row misses the cap"

    rated = rare.sigma_scale_vector(
        chosen, opportunities, gain=0.5, cap=3.0, novelty="rate"
    )
    # The ordering the mechanism was always supposed to produce.
    assert rated[wheat] == 1.0, "the overfitted row is not widened at all"
    assert rated[melon] > 1.5
    assert rated[updater_module.LAND_INDEX] > rated[melon]


def test_the_rate_novelty_reads_selection_against_opportunity():
    """Two rows on equal counts and unequal opportunity are not equally novel.

    Which is the whole difference: `chosen` alone cannot tell a row the mask rarely
    offered from a row the policy rarely took.
    """

    chosen = np.full(ACTION_COUNT, 500.0)
    opportunities = np.full(ACTION_COUNT, 100_000.0)
    often, seldom = 1, 2
    opportunities[often] = 900_000.0   # offered constantly, taken 500 times
    opportunities[seldom] = 1_000.0    # barely offered, taken 500 times

    scale = rare.sigma_scale_vector(chosen, opportunities, gain=0.5, cap=3.0, novelty="rate")
    assert scale[often] > scale[seldom], "declining a standing offer is the novel one"
    # The count arm is blind to it, which is the point.
    counted = rare.sigma_scale_vector(chosen, opportunities, gain=0.5, cap=3.0)
    assert counted[often] == pytest.approx(counted[seldom])


def test_the_rate_novelty_floors_a_never_chosen_row_at_one_selection():
    """No epsilon to tune: an unchosen row is treated as chosen exactly once.

    That is the largest claim the data supports, and it keeps the log finite
    without a constant that would have to be re-sized whenever the run length or
    the action space moved.
    """

    chosen = np.full(ACTION_COUNT, 400.0)
    opportunities = np.full(ACTION_COUNT, 80_000.0)
    unchosen = 3
    chosen[unchosen] = 0.0

    scale = rare.sigma_scale_vector(chosen, opportunities, gain=0.5, cap=3.0, novelty="rate")
    assert np.isfinite(scale).all()
    assert scale[unchosen] == max(scale)

    once = chosen.copy()
    once[unchosen] = 1.0
    assert rare.sigma_scale_vector(
        once, opportunities, gain=0.5, cap=3.0, novelty="rate"
    )[unchosen] == pytest.approx(scale[unchosen])


def test_both_novelty_arms_keep_the_invariants_that_make_widening_safe():
    """Bounded below at 1.0, NOOP pinned, never-feasible rows left alone.

    The family history here is the epsilon purchase pump, so a new statistic has to
    inherit every guard the old one carried: this can only ever add variance to a
    row, and a row the policy learned is ruinous still has to win a plain argmax.
    """

    rng = np.random.default_rng(11)
    chosen = rng.integers(0, 5_000, ACTION_COUNT).astype(float)
    opportunities = rng.integers(1_000, 500_000, ACTION_COUNT).astype(float)
    never_feasible = 7
    opportunities[never_feasible] = 0.0
    chosen[never_feasible] = 0.0

    for novelty in rare.NOVELTY_KINDS:
        scale = rare.sigma_scale_vector(
            chosen, opportunities, gain=0.5, cap=2.0, novelty=novelty
        )
        assert scale.dtype == np.float32
        assert scale.min() >= 1.0, "only ever adds variance"
        assert scale.max() <= 2.0, "and the cap is a cap"
        assert scale[NOOP_INDEX] == 1.0, "the collapse attractor is never widened"
        assert scale[never_feasible] == 1.0, (
            "a row the mask refuses keeps maximum novelty forever and would arrive "
            "at full width the moment the mask opened"
        )


def test_the_count_arm_stays_byte_identical_and_a_bad_name_is_refused():
    """`count` is the default, so every run before this key reproduces exactly."""

    rng = np.random.default_rng(3)
    chosen = rng.integers(0, 9_000, ACTION_COUNT).astype(float)
    opportunities = rng.integers(1, 400_000, ACTION_COUNT).astype(float)

    assert_array_equal(
        rare.sigma_scale_vector(chosen, opportunities, gain=1.0, cap=3.0),
        rare.sigma_scale_vector(chosen, opportunities, gain=1.0, cap=3.0, novelty="count"),
    )
    with pytest.raises(ValueError, match="rare_sigma_novelty"):
        rare.sigma_scale_vector(chosen, opportunities, gain=1.0, cap=3.0, novelty="ucb")


def test_the_rate_novelty_has_nothing_to_widen_on_the_first_round():
    """No selections yet means nothing has been avoided yet, and it says so."""

    opportunities = np.full(ACTION_COUNT, 1_000.0)
    scale = rare.sigma_scale_vector(
        np.zeros(ACTION_COUNT), opportunities, gain=0.5, cap=3.0, novelty="rate"
    )
    assert_array_equal(scale, np.ones(ACTION_COUNT, dtype=np.float32))


def _lot_trace(events: list[tuple[int, str, int, float]], turns: int = 60):
    """Build `(seeds, standing, stock, money, ops)` from `(turn, kind, count, coins)`.

    `kind` is one of `seed`, `plant`, `harvest`, `sell`, `hire`. Every event is
    applied *from* the given turn onward, because the traces are levels rather
    than deltas -- the same shape `Recorder.record_turn` appends.

    `harvest` clears the tile as well as filling the shed, which is what the
    simulator does to a non-ongoing crop (`kaggriculture.py:432-455`), so the
    standing count a caller gets back is the one a real trace would carry.
    """

    from agents.dqn_market_free import lots

    melon_seed = lots.CROP_ITEMS.index("MELON")
    melon_stock = lots.STOCK_ITEMS.index("MELON")
    seeds = [[0] * len(lots.CROP_ITEMS) for _ in range(turns)]
    standing = [[0] * len(lots.CROP_ITEMS) for _ in range(turns)]
    stock = [[0] * len(lots.STOCK_ITEMS) for _ in range(turns)]
    money = [3_000.0] * turns
    ops: list[str | None] = [None] * turns
    for turn, kind, count, coins in events:
        if kind == "seed":
            ops[turn] = "BUY_SEED"
        elif kind == "sell":
            ops[turn] = "SELL"
        elif kind == "hire":
            ops[turn] = "HIRE"
        for later in range(turn + 1, turns):
            if kind == "seed":
                seeds[later][melon_seed] += count
                money[later] -= coins
            elif kind == "plant":
                seeds[later][melon_seed] -= count
                standing[later][melon_seed] += count
            elif kind == "harvest":
                # One tile cleared, `count` units into the shed -- a melon plant
                # carries up to six of them.
                standing[later][melon_seed] -= 1
                stock[later][melon_stock] += count
            elif kind == "sell":
                stock[later][melon_stock] -= count
                money[later] += coins
            elif kind == "hire":
                money[later] -= coins
    # A trace that harvests without planting first is shorthand, not a farm that
    # owes a tile.
    for row in standing:
        row[melon_seed] = max(0, row[melon_seed])
    return seeds, standing, stock, money, ops


def _lot_reward(seeds, standing, stock, money, **kwargs):
    from agents.dqn_market_free.training.reward import episode_rewards

    return episode_rewards(
        money,
        money[-1],
        money[-1],
        0.999,
        seeds=seeds,
        standing=standing,
        stock=stock,
        **kwargs,
    )[0]


def test_the_lot_potential_pays_on_the_purchase_turn():
    """The whole point: a BUY_SEED stops scoring the same as a NOOP.

    `LOT_CREDIT_WEIGHT` paid this same credit but took it back at the sale turn,
    which cost the sale state the full amount of its return-to-go while the buy
    state gained only `1 - gamma**D` of it. The potential form pays the credit and
    charges nothing anywhere, because `PHI` is a function of the state.
    """

    seeds, standing, stock, money, _ = _lot_trace(
        [
            (5, "seed", 1, 80.0),
            (10, "plant", 1, 0.0),
            (20, "harvest", 4, 0.0),
            (30, "sell", 4, 1_000.0),
        ]
    )
    shaped = _lot_reward(seeds, standing, stock, money)
    base = _lot_reward(seeds, standing, stock, money, lot_weight=0.0)
    moved = np.asarray(shaped) - np.asarray(base)

    assert moved[5] > 0.0
    # Planting is free: the cost basis moves from the seed bag to the tile and
    # `PHI` does not notice. This is what `standing` is in the trace for, and
    # without it the term would charge for the action it exists to encourage.
    assert moved[10] == pytest.approx(moved[11], abs=1e-9)


def test_the_lot_potential_is_sized_by_the_seed_price():
    """One melon seed at 80 coins, against `MONEY_SCALE` and the other charges."""

    from agents.dqn_market_free.training.reward import (
        IDLE_PENALTY,
        INSOLVENCY_PENALTY,
        LOT_POTENTIAL_WEIGHT,
        MONEY_SCALE,
    )

    seeds, standing, stock, money, _ = _lot_trace([(5, "seed", 1, 80.0)])
    shaped = _lot_reward(seeds, standing, stock, money)
    base = _lot_reward(seeds, standing, stock, money, lot_weight=0.0)

    expected = LOT_POTENTIAL_WEIGHT * 80.0 / MONEY_SCALE
    assert float(shaped[5] - base[5]) == pytest.approx(expected, abs=1e-5)
    # Visible against what it competes with on the purchase turn, and still well
    # under the one term in the file that can pay for a purchase outright.
    assert expected > INSOLVENCY_PENALTY > IDLE_PENALTY


def test_the_lot_potential_telescopes_to_nothing_over_an_episode():
    """Policy invariance, in the only form a test can state it.

    A potential contributes `-PHI(s_0)` to the episode's undiscounted total and
    nothing else, because the terminal entry is pinned to zero. So two seats that
    start empty and differ only in what they bought and sold in between carry the
    identical total -- the term cannot pay for a purchase, only move when the
    credit arrives.
    """

    empty = _lot_trace([])
    traded = _lot_trace(
        [
            (5, "seed", 1, 80.0),
            (10, "plant", 1, 0.0),
            (20, "harvest", 6, 0.0),
            (30, "sell", 6, 1_500.0),
        ]
    )
    hoarded = _lot_trace(
        [(5, "seed", 1, 80.0), (10, "plant", 1, 0.0), (20, "harvest", 6, 0.0)]
    )

    totals = []
    for seeds, standing, stock, money, _ in (empty, traded, hoarded):
        shaped = _lot_reward(seeds, standing, stock, money)
        base = _lot_reward(seeds, standing, stock, money, lot_weight=0.0)
        totals.append(float(np.sum(np.asarray(shaped) - np.asarray(base))))

    # Every seat starts with nothing held, so `PHI(s_0)` is zero for all three
    # and the shaping sums to (gamma - 1) times the basis they carried, never to
    # a payout. What matters is that hoarding is not rewarded relative to selling.
    assert totals[0] == pytest.approx(0.0, abs=1e-6)
    assert totals[1] <= 0.0
    assert totals[2] < totals[1]


def test_a_lot_that_never_sold_is_still_recorded_as_unrecovered():
    """Unsold stock keeps its cost where it was spent, which is the punishment."""

    from agents.dqn_market_free import lots

    seeds, standing, stock, money, ops = _lot_trace(
        [(5, "seed", 1, 80.0), (10, "plant", 1, 0.0), (20, "harvest", 4, 0.0)]
    )
    ledger = lots.match(seeds, stock, money, ops, standing)
    assert ledger.contributions == []
    assert ledger.summary()["seed_lots"] == 1.0
    assert ledger.summary()["seed_lots_matched"] == 0.0
    assert ledger.summary()["unrecovered_seed_share"] == pytest.approx(1.0)


def test_one_seed_lot_absorbs_a_whole_ongoing_harvest():
    """A melon seed yields up to six, so six sold units still trace to one buy."""

    from agents.dqn_market_free import lots

    seeds, standing, stock, money, ops = _lot_trace(
        [
            (2, "seed", 1, 80.0),
            (5, "plant", 1, 0.0),
            (20, "harvest", 6, 0.0),
            (25, "sell", 3, 750.0),
            (35, "sell", 3, 600.0),
        ]
    )
    ledger = lots.match(seeds, stock, money, ops, standing)

    assert {buy for buy, _, _ in ledger.contributions} == {2}
    assert len(ledger.contributions) == 6
    assert ledger.unattributed_proceeds == pytest.approx(0.0)
    summary = ledger.summary()
    assert summary["seed_proceeds"] == pytest.approx(1_350.0)
    # First matched sale, not the last: the holding period is how long the
    # purchase waited to be told anything at all.
    assert summary["seed_holding_turns"] == pytest.approx(23.0)


def test_a_harvest_landing_on_a_hire_turn_opens_no_lot():
    """Field actions run before the market, so the two collide on one diff.

    `main.py` emits HARVEST every turn regardless of what the market decision is,
    so a harvest arriving on a HIRE turn reads as "stock up, money down". The
    sign heuristic booked that as a purchase -- dated to the harvest and priced
    at the hire -- and it landed straight in `seed_spend`.
    """

    from agents.dqn_market_free import lots

    seeds, standing, stock, money, ops = _lot_trace(
        [
            (5, "seed", 1, 80.0),
            (10, "plant", 1, 0.0),
            (20, "harvest", 4, 0.0),
            (20, "hire", 0, 55.0),
        ]
    )
    ledger = lots.match(seeds, stock, money, ops, standing)

    assert [lot.buy_turn for lot in ledger.lots] == [5]
    assert ledger.summary()["seed_spend"] == pytest.approx(80.0)

    # Without the op the old sign reading is preserved, which is the fallback the
    # signature promises -- and the phantom it produces is the bug itself: a lot
    # dated to the harvest, priced at the hire, sitting in the melon queue ready
    # to absorb a sale the seed at turn 5 actually paid for.
    blind = lots.match(seeds, stock, money)
    assert [lot.buy_turn for lot in blind.lots] == [5, 20]
    phantom = blind.lots[1]
    assert phantom.unit_cost == pytest.approx(55.0 / 4)
    # The `bought` tag keeps it out of `seed_spend` even here, which is the second
    # half of the fix and independent of the op.
    assert phantom.bought is True
    assert blind.summary()["seed_spend"] == pytest.approx(80.0)


def test_a_bought_product_lot_is_not_counted_as_seed_spend():
    """Wheat bought to resell is not coins committed to growing anything."""

    from agents.dqn_market_free import lots

    turns = 40
    seeds = [[0] * len(lots.CROP_ITEMS) for _ in range(turns)]
    standing = [[0] * len(lots.CROP_ITEMS) for _ in range(turns)]
    stock = [[0] * len(lots.STOCK_ITEMS) for _ in range(turns)]
    money = [3_000.0] * turns
    ops: list[str | None] = [None] * turns
    ops[5] = "BUY_PRODUCT"
    for later in range(6, turns):
        stock[later][lots.STOCK_ITEMS.index("WHEAT")] += 4
        money[later] -= 200.0

    ledger = lots.match(seeds, stock, money, ops, standing)
    assert len(ledger.lots) == 1
    assert ledger.lots[0].bought is True
    assert ledger.summary()["seed_lots"] == 0.0
    assert ledger.summary()["seed_spend"] == pytest.approx(0.0)


def test_a_finished_crop_stops_absorbing_sales_it_could_not_have_grown():
    """Room left open for units that rotted would starve every later purchase.

    `_YIELD_CAP` gives a melon seed six units of capacity. If the plant only ever
    yields two and the tile is then gone, the other four are a claim on units
    that were never grown -- and FIFO would hand a later lot's sales to the first
    purchase forever. `_reconcile` closes the gap against the trace.
    """

    from agents.dqn_market_free import lots

    seeds, standing, stock, money, ops = _lot_trace(
        [
            (2, "seed", 1, 80.0),
            (4, "plant", 1, 0.0),
            (10, "harvest", 2, 0.0),
            (12, "sell", 2, 500.0),
            (14, "seed", 1, 80.0),
            (16, "plant", 1, 0.0),
            (30, "harvest", 2, 0.0),
            (34, "sell", 2, 500.0),
        ]
    )
    ledger = lots.match(seeds, stock, money, ops, standing)

    # The second sale belongs to the second purchase. Without reconciliation the
    # first lot still had four units of room and would have taken it.
    assert sorted({buy for buy, _, _ in ledger.contributions}) == [2, 14]
    summary = ledger.summary()
    assert summary["seed_lots_matched"] == 2.0
    # 10 turns for the first lot and 20 for the second. Both are real holding
    # periods only because the second lot got its own sale; before reconciliation
    # the first lot took both and the second read as never matched at all.
    assert summary["seed_holding_turns"] == pytest.approx(15.0)


def test_the_lot_holdings_read_the_shed_and_every_hand():
    """A unit in a hand's inventory is held, and the ledger has to see it."""

    from agents.dqn_market_free import lots

    seeds, stock = lots.holdings(
        {
            "seeds": {"MELON": 2},
            "shed": {"MELON": 3},
            "inventories": [{"MELON": 1}, {"MELON": 4, "WHEAT": 2}],
        }
    )
    assert seeds[lots.CROP_ITEMS.index("MELON")] == 2
    assert stock[lots.STOCK_ITEMS.index("MELON")] == 8
    assert stock[lots.STOCK_ITEMS.index("WHEAT")] == 2
