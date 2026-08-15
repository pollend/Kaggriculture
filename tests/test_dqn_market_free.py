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

from agents.dqn_market_free import donor
from agents.dqn_market_free.diagnostics import OrderDiagnostics, diagnose_order_lists
from agents.dqn_market_free.main import MarketFreeAgent, load_network
from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import (
    ACTION_COUNT,
    ACTIONS,
    HIRE_QUANTITIES,
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

    score = updater_module._slot_score(Reader(), "free.model", "pilkwang", 40.0, 5, 120)

    assert score == pytest.approx(36.0), "the median, not the latest draw"
    # Bounded below by the slot's own reseed: measurements from before it describe
    # a policy this slot no longer is, and an inherited high score would pin the
    # plateau counter of a restarted slot.
    assert seen[-1][1] == 120

    # A window of one is the candidate itself, byte for byte the behaviour before
    # the window existed.
    assert updater_module._slot_score(Reader(), "free.model", "pilkwang", 40.0, 1, 0) == 40.0
    # Nothing measured is not a stall, and `_settle_slot` must see it as such.
    spec = updater_module._slot(_SLOT_CONFIG)
    state = updater_module._slot_state({}, spec)
    assert updater_module._settle_slot(state, None, spec) == state


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
            if sql.strip().startswith("UPDATE"):
                return [("worst", 1.5)]
            return [("member",)]

    result = updater_module._graduate(
        Reader(), "decl", "decl@learner_03", "ckpt-a", 145.8, 30
    )

    insert, cull = statements
    assert "INSERT INTO population_members" in insert[0]
    assert "RETURNING" in insert[0], (
        "the trainer's only handle is the read view, whose fetchall wants rows"
    )
    assert insert[1][1:] == ["decl", "decl@learner_03", "ckpt-a", 145.8]

    # The cull reads the live set *after* the insert, so a graduate that is itself
    # the worst member is the one retired and the group cannot be made worse by
    # admitting to it.
    assert "retired_at = current_timestamp" in cull[0]
    assert "OFFSET" in cull[0] and cull[1] == ["decl", 30]
    assert result["retired"] == [{"member_id": "worst", "score": 1.5}]
    assert result["score"] == 145.8


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
                ("m1", "decl@a", "ckpt-1", 900.0, str(tmp_path / "gone.npz")),
                ("m2", "decl@b", "ckpt-2", 400.0, str(present)),
            ]

    members = updater_module.mature_members(Reader(), "decl")

    assert [member["checkpoint_id"] for member in members] == ["ckpt-2"]
    assert members[0]["score"] == 400.0


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

    entry_after = 4

    def eligible(age, graduating):
        return graduating or (age is not None and age >= entry_after)

    # Fresh release, slot still training: held back.
    assert not eligible(0, False)
    assert not eligible(3, False)
    # Survived the lag: admitted.
    assert eligible(4, False)
    assert eligible(11, False)
    # Undatable: treated as not old enough.
    assert not eligible(None, False)
    # A graduating slot contributes whatever the age -- it is finishing anyway,
    # and holding its release back would lose it entirely.
    assert eligible(0, True)
    assert eligible(None, True)


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
        TERMINAL_MARGIN_SCALE,
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

    outcome = float(np.tanh(2_000.0 / TERMINAL_MARGIN_SCALE))
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

    # And a wallet that drops under the floor is charged for the turn it spends
    # there, read off the *resulting* wallet. Both episodes end level against the
    # opponent, so the outcome term is identical and the gap is the charge alone.
    broke = [3_000.0, 10.0, 3_000.0]
    starved, _, _ = episode_rewards(
        broke, 3_000.0, 3_000.0, 0.999
    )
    solvent, _, _ = episode_rewards(
        money, 3_000.0, 3_000.0, 0.999
    )
    assert float(solvent[0] - starved[0]) == pytest.approx(INSOLVENCY_PENALTY, abs=1e-6)
    assert np.allclose(solvent[1:], starved[1:])


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

    assert 1.0 < float(shaped.sum()) < 10.0
    # Every turn, including the last, is inside the quadratic zone.
    assert float(np.abs(shaped).max()) < 1.0
    assert MONEY_SCALE >= 10_000.0


def test_terminal_margin_discriminates_across_the_self_play_range():
    """The scale has to fit the margins the mix actually produces.

    Two ways this term has already been informationless. A hard clip at 20,000
    returned exactly -1.0 on all 640 recorded episodes, whose margins ran from
    -21,681 to -210,076. Widening to 100,000 only moved the saturation point: by
    the last recorded generation every margin was about -128,000 and the term was
    a constant -0.86. A constant on the terminal reward shifts every Q value
    uniformly and can never change an argmax.

    The mix is now mostly self-play, where both seats hold a 3,000 stake and the
    margins are hundreds to low thousands. That is the range the scale is sized
    for, and the range this test pins.
    """

    from agents.dqn_market_free.training.reward import (
        TERMINAL_MARGIN_SCALE,
        episode_rewards,
    )

    def terminal(margin: float) -> float:
        # Both seats well clear of the insolvency floor, so this reads the
        # outcome term alone.
        own = 20_000.0
        shaped, _, _ = episode_rewards([own, own], own, own - margin, 1.0)
        return float(shaped[-1])

    mirror = [-4_000.0, -1_500.0, 500.0, 2_000.0, 6_000.0]
    values = [terminal(margin) for margin in mirror]

    # Strictly ordered, and none of them crushed against the bound.
    assert values == sorted(values)
    assert min(values) > -0.99 and max(values) < 0.99
    assert max(values) - min(values) > 0.9
    # A win and a loss must land on opposite sides of parity, not merely differ.
    assert terminal(-1_500.0) < 0.0 < terminal(500.0)

    # Bounded either side, so one blowout episode cannot dominate a batch.
    # tanh is open on (-1, 1) mathematically; float32 saturates at the bound.
    assert -1.0 <= terminal(-10_000_000.0) <= -0.99
    assert 0.99 <= terminal(10_000_000.0) <= 1.0
    # Symmetric about parity.
    assert terminal(0.0) == pytest.approx(0.0)
    assert terminal(TERMINAL_MARGIN_SCALE) == pytest.approx(-terminal(-TERMINAL_MARGIN_SCALE))


# --------------------------------------------------------------------------
# The outcome annuity
#
# The terminal margin lump lands on one turn of ~720 and the buffer samples
# 24-turn windows, so ~3% of transitions ever saw it as a reward. The rest waited
# on a bootstrap chain up to 720 links deep to learn whether their episode made
# money. The annuity pays `(1 - gamma) * outcome` on every step so the signal
# arrives by gradient instead.
# --------------------------------------------------------------------------


#: Own money, held flat, for the annuity tests below.
#:
#: The margin in these tests is always moved by varying the *opponent's* money and
#: never one's own, so `raw` is bit-identical between the two calls being compared
#: and the difference is the outcome terms alone.
SEASON_MONEY = 5_000.0


def _season(turns: int = 400) -> tuple[list[float], list[float]]:
    return [SEASON_MONEY] * turns


def _outcome_gap(margin: float, gamma: float, turns: int = 400) -> np.ndarray:
    """Per-step reward of an episode won by `margin`, minus the same one drawn."""

    from agents.dqn_market_free.training.reward import episode_rewards

    money = _season(turns)
    played, _, _ = episode_rewards(money, SEASON_MONEY, SEASON_MONEY - margin, gamma
    )
    drew, _, _ = episode_rewards(money, SEASON_MONEY, SEASON_MONEY, gamma)
    return played - drew


def test_the_outcome_reaches_every_decision_not_just_the_last():
    """Every step of a winning episode must score above the same step at parity."""

    from agents.dqn_market_free.training.reward import TERMINAL_MARGIN_SCALE

    gamma = 0.999
    margin = 60_000.0

    outcome = float(np.tanh(margin / TERMINAL_MARGIN_SCALE))
    gap = _outcome_gap(margin, gamma)

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

    from agents.dqn_market_free.training.reward import TERMINAL_MARGIN_SCALE

    gamma = 0.999
    turns = 400

    for margin in (60_000.0, -120_000.0):
        outcome = float(np.tanh(margin / TERMINAL_MARGIN_SCALE))
        gap = _outcome_gap(margin, gamma, turns)
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
        TERMINAL_MARGIN_SCALE,
        episode_rewards,
    )

    money = [3_000.0, 2_000.0, 2_500.0, 4_000.0]

    shaped, _, _ = episode_rewards(money, 9_000.0, 3_000.0, 1.0)

    outcome = float(np.tanh(6_000.0 / TERMINAL_MARGIN_SCALE))
    # At gamma = 1 the annuity is zero and the lump is the whole stream.
    assert float(shaped.sum()) == pytest.approx(outcome, abs=1e-5)


def test_a_losing_episode_is_charged_on_every_turn():
    """The annuity is signed, so it is a penalty as readily as a bonus."""

    assert bool((_outcome_gap(-125_000.0, 0.999) < 0.0).all())


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

    assert 1.0 < float(shaped.sum()) < 10.0
    assert float(np.abs(shaped).max()) < 1.0


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

    shaped, raw, _ = episode_rewards(money, 2_880.0, 2_880.0, 1.0, emitted)

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
    plain, _, _ = episode_rewards(money, 2_880.0, 2_880.0, 1.0)
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
    import math

    from agents.dqn_market_free.training.reward import TERMINAL_MARGIN_SCALE

    outcome_gap = math.tanh(3_000 / TERMINAL_MARGIN_SCALE) - math.tanh(
        1_898 / TERMINAL_MARGIN_SCALE
    )
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
# --------------------------------------------------------------------------


def test_every_turn_with_no_runway_is_charged():
    from agents.dqn_market_free.training.reward import (
        INSOLVENCY_FLOOR,
        INSOLVENCY_PENALTY,
        episode_rewards,
    )

    # Turn 0 spends down to the floor, turns 1 and 2 sit there, turn 3 recovers.
    money = [3_000.0, 100.0, 100.0, 100.0]
    shaped, _, _ = episode_rewards(money, 900.0, 900.0, 1.0)

    charged = -shaped
    assert charged[:3] == pytest.approx(INSOLVENCY_PENALTY, abs=1e-6)
    # The turn that climbs back over the floor is not charged.
    assert float(charged[3]) == pytest.approx(0.0, abs=1e-6)

    # The charge reads the resulting wallet, so the turn that spends the last of
    # the stake is the first one billed rather than the one after it.
    assert float(money[0]) > INSOLVENCY_FLOOR > float(money[1])


def test_a_solvent_season_is_never_charged():
    """A farm trading above the floor must see the shaping unchanged."""

    from agents.dqn_market_free.training.reward import (
        INSOLVENCY_PENALTY,
        episode_rewards,
    )

    money = list(np.linspace(3_000.0, 40_000.0, 400))
    shaped, _, _ = episode_rewards(money, 40_000.0, 40_000.0, 1.0)

    assert float(np.abs(shaped).max()) < INSOLVENCY_PENALTY


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
        "seeds": {},
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
                before = probe_farm["money"]
                sim._do_buy_land(probe_farm, 10)
                accepted = probe_farm["money"] != before
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


def test_land_runs_out_and_hiring_gets_dearer():
    assert _mask(money=3000.0, quadrants=("NW",))[_rows("BUY_LAND")].all()
    # Third extra quadrant costs 4000.
    assert not _mask(money=3000.0, quadrants=("NW", "NE", "SW"))[_rows("BUY_LAND")].any()
    # Nothing left to buy at any price.
    assert not _mask(money=99_999.0, quadrants=("NW", "NE", "SW", "SE"))[
        _rows("BUY_LAND")
    ].any()
    # fib(9) = 55, so nine hands deep the tenth needs 55 coins -- and an eleventh
    # in the same order needs 55 + 89 on top, which 55 does not cover.
    assert not _mask(money=54.0, hires_today=9)[_rows("HIRE")].any()
    assert _mask(money=55.0, hires_today=9)[_rows("HIRE")].tolist() == [
        hands <= 1 for hands in HIRE_QUANTITIES
    ]


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
    assert int(mask.sum()) == 98
    assert mask[_rows("BUY_LAND")].any()
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
