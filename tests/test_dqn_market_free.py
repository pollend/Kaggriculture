"""Correctness guards for the unscaffolded market agent.

These pin the invariants that would otherwise fail silently: a cold start that
cannot train, an action grid that cannot name what real agents do, train/
inference skew, and a demonstration reader that drops most of what it reads.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.dqn_market_free import donor
from agents.dqn_market_free.diagnostics import diagnose_order_lists
from agents.dqn_market_free.main import MarketFreeAgent, load_network
from agents.dqn_market_free.market_state import MARKET_STATE_DIM as STATE_DIM
from agents.dqn_market_free.order_space import (
    ACTION_COUNT,
    ACTIONS,
    HIRE_QUANTITIES,
    QUANTITIES,
    Order,
    choose_order,
    nearest_quantity,
    order_index,
    order_is_structurally_possible,
)
from agents.dqn_market_free.planting import market_influenced_roles
from agents.dqn_market_free.policy import OrderQNetwork
from agents.dqn_market_free.training import network as net_module
from agents.dqn_market_free.training import updater as updater_module
from agents.dqn_market_free.training.clone import load_demonstrations, record_from_replay
from agents.dqn_market_free.training.collect import EpisodeRecord, Recorder
from agents.dqn_market_free.training.replay import ReplayBuffer
from agents.dqn_market_free.training.updater import (
    SELF_SPEC,
    SOURCE_KIND,
    _arena_order_diagnostics,
    _arena_viability,
    _behavior_clone_step,
    _behavior_clone_steps_per_update,
    _reset_learning_rates,
    initial_network,
    rotate_opponents,
    self_play_pool,
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


def test_local_market_state_encoder_matches_the_scorer_schema():
    import inspect

    from agents.dqn_market_free import market_state as local
    from agents.dqn_market_scorer import market_policy as scorer

    assert local.MARKET_STATE_DIM == scorer.MARKET_STATE_DIM
    for name in ("_owned_animals", "_structures", "_tile_census", "market_state_features"):
        assert inspect.getsource(getattr(local, name)) == inspect.getsource(getattr(scorer, name))


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
    return {
        "pairs": 50,
        "errors": {},
        "a_money_mean": mean_money,
        "a_money_median": median_money,
        "order_diagnostics": {
            "decisions": decisions,
            "orders": orders,
            # One order a decision unless a case says otherwise, which is what
            # this agent's own policy produces for everything except HIRE.
            "trading_decisions": orders if trading_decisions is None else trading_decisions,
        },
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


def test_viability_gate_rejects_a_noop_collapse_once_a_release_exists():
    """The other attractor. A policy that trades nothing preserves its money."""

    silent = _viability_result(orders=0)
    reference = {"a_money_mean": 900.0, "a_money_median": 800.0}

    # A cold start is legitimately silent and must stay promotable.
    assert _arena_viability(silent, reference, {})["passed"]

    guarded = _arena_viability(silent, reference, {}, require_activity=True)
    assert not guarded["passed"]
    assert not guarded["checks"]["min_orders_per_decision"]


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
    """A shared helper must not hand free-market weights to the scorer's agent.

    Both loaders validate their own shape, so a spec naming the wrong agent fails
    every episode it is drawn for.
    """

    checkpoint = tmp_path / "promoted.npz"
    checkpoint.write_bytes(b"weights")

    class Reader:
        def fetchall(self, sql, parameters):
            assert "model_training_checkpoints" in sql
            return [(str(checkpoint),)]

    context = SimpleNamespace(model_id="free.model")
    pool = self_play_pool(context, Reader(), 5, SELF_SPEC)

    assert pool == [f"{SELF_SPEC}?weights={checkpoint}"]
    assert all(spec.startswith("agents/dqn_market_free/") for spec in pool)


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


# --------------------------------------------------------------------------
# Cold start
# --------------------------------------------------------------------------


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


def test_epsilon_is_the_whole_exploration_rate():
    """One knob, and its value is the measured trade rate.

    It absorbed a separate per-slot `act_rate`: only the product of the two ever
    mattered. Exploring on every turn is measurably fatal -- at per-turn rates of
    1.0 down to 0.005 the episode ends on single- to triple-digit money against a
    3,000 float -- so what this pins is that the configured number *is* the rate,
    with no second factor hiding between it and the behaviour.
    """

    state = np.zeros(STATE_DIM, dtype=np.float32)
    network = initial_network({})
    rng = np.random.default_rng(0)
    epsilon = 0.02
    trials = 20_000
    placed = sum(
        choose_order(state, network, epsilon=epsilon, rng=rng).chosen != 0
        for _ in range(trials)
    )
    observed = placed / trials
    assert 0.5 * epsilon < observed < 2.0 * epsilon


def test_exploration_emits_at_most_one_order_and_never_draws_noop():
    state = np.zeros(STATE_DIM, dtype=np.float32)
    rng = np.random.default_rng(1)
    network = initial_network({})
    for _ in range(200):
        decision = choose_order(state, network, epsilon=1.0, rng=rng)
        # An exploring turn always trades: NOOP is drawn by not exploring.
        assert decision.chosen != 0
        wire = decision.wire()
        assert wire
        # One action, but HIRE expands to one wire order per hand.
        assert len({tuple(row) for row in wire}) == 1
        assert len(wire) == 1 or decision.order.op == "HIRE"


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
    decisions: int, final_money: float, seed: int = 0, chosen: int | None = None
) -> EpisodeRecord:
    rng = np.random.default_rng(seed)
    steps = decisions
    order_chosen = (
        rng.integers(0, ACTION_COUNT, size=decisions).astype(np.int16)
        if chosen is None
        else np.full(decisions, chosen, dtype=np.int16)
    )
    return EpisodeRecord(
        reward=rng.standard_normal(steps).astype(np.float32),
        raw_reward=rng.standard_normal(steps).astype(np.float32),
        money=np.asarray([*np.zeros(steps), final_money], dtype=np.float32),
        order_state=rng.standard_normal((decisions, STATE_DIM)).astype(np.float32),
        order_chosen=order_chosen,
        order_step=np.arange(decisions, dtype=np.int16),
    )


def test_demo_fraction_puts_solvent_data_in_every_batch():
    """What replaced rank-based prioritisation.

    Solvent trajectories are rare enough that an unweighted draw over the
    agent's own episodes is almost entirely bankruptcies, and the only policy
    consistent with that data is to do nothing. Drawing a fixed share of every
    minibatch from the pinned demonstrations puts the same signal in by
    construction, with no importance-sampling bias to correct.
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


def test_buy_land_is_shaped_to_roughly_neutral():
    """The one capital purchase whose cost the shaping used to never return.

    Seeds and animals leave cash and enter `Phi` at the price paid, so they cost
    about nothing on the turn they happen. Land did not, so 1,000-4,000 coins
    read as permanent loss with the payoff dozens of turns away -- and the
    cheapest response is to never leave the starting quadrant.
    """

    from agents.dqn_market_free.training.reward import MONEY_SCALE, potential

    gamma = 0.999
    before = potential(_bare_farm(["NW"]), _EMPTY_PRIVATE, _EMPTY_OBS)
    after = potential(_bare_farm(["NW", "NE"]), _EMPTY_PRIVATE, _EMPTY_OBS)

    unshaped = -donor.LAND_PRICES[0] / MONEY_SCALE
    shaped = unshaped + gamma * after - before

    assert unshaped < -0.04
    # All that survives is the (1 - gamma) leak on the potential itself.
    assert abs(shaped) < 0.001


def test_potential_prices_land_at_what_was_paid_for_it():
    from agents.dqn_market_free.training.reward import MONEY_SCALE, potential

    for extra in range(4):
        quadrants = ["NW", "NE", "SW", "SE"][: extra + 1]
        value = potential(_bare_farm(quadrants), _EMPTY_PRIVATE, _EMPTY_OBS)
        assert value == pytest.approx(sum(donor.LAND_PRICES[:extra]) / MONEY_SCALE)


def test_shaping_telescopes_so_the_return_is_policy_invariant():
    """Sum of the shaping terms is `-Phi(s_0)`, whatever `Phi` prices.

    That identity is why `Phi` is free to carry land: shaping of this form
    cannot change which policy is optimal, only how fast it is found. It holds
    only because the terminal potential is forced to zero.
    """

    from agents.dqn_market_free.training.reward import MONEY_SCALE, episode_rewards

    money = [3_000.0, 2_000.0, 2_500.0, 4_000.0]
    potentials = [0.5, 1.4, 0.9, 0.2]

    # Exact at gamma = 1.
    shaped, raw, _ = episode_rewards(money, potentials, 9_000.0, 9_000.0, 1.0)
    # The margin term is zero here, so shaped - raw is the shaping alone.
    assert float((shaped - raw).sum()) == pytest.approx(-potentials[0], abs=1e-5)

    # Below 1 it leaks (1 - gamma) * sum(Phi), which is the only reason the
    # identity is approximate rather than exact.
    shaped, raw, _ = episode_rewards(money, potentials, 9_000.0, 9_000.0, 0.999)
    leak = 0.001 * sum(potentials[1:])
    assert float((shaped - raw).sum()) == pytest.approx(-potentials[0] - leak, abs=1e-5)

    assert float(raw.sum()) == pytest.approx((9_000.0 - 3_000.0) / MONEY_SCALE, abs=1e-6)


def test_terminal_potential_is_forced_to_zero():
    """Inventory held past the last turn scores nothing, so it must be worth nothing."""

    from agents.dqn_market_free.training.reward import episode_rewards

    hoarding = [1.0, 5.0]
    liquidating = [1.0, 0.0]
    money = [3_000.0, 3_000.0]

    hoard, _, _ = episode_rewards(hoarding, [0.0, 0.0], 3_000.0, 3_000.0, 0.999)
    # A big terminal stockpile must not be rewarded for existing.
    stocked, _, _ = episode_rewards(money, hoarding, 3_000.0, 3_000.0, 0.999)
    sold, _, _ = episode_rewards(money, liquidating, 3_000.0, 3_000.0, 0.999)
    assert stocked.sum() < sold.sum()
    assert np.isfinite(hoard).all()


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
    shaped, _, _ = episode_rewards(money, [0.0] * turns, 76_000.0, 40_000.0, 0.999)

    assert 1.0 < float(shaped.sum()) < 10.0
    # Every turn, including the last, is inside the quadratic zone.
    assert float(np.abs(shaped).max()) < 1.0
    assert MONEY_SCALE >= 10_000.0


def test_actor_epsilons_spread_a_batch_across_temperatures():
    """One rate per batch means one policy at one temperature in the buffer.

    Measured at a base of 0.014 with a single rate: 81% mean pairwise action
    agreement between the sixteen episodes of a batch, and identical actions on
    38% of turns. Spreading the rate is what puts exploratory and near-greedy
    trajectories in the same buffer instead of in different generations.
    """

    from agents.dqn_market_free.training.collect import actor_epsilons

    spread = actor_epsilons(0.18, 8, 4.0)
    assert len(spread) == 8
    # The first actor runs at the schedule's own rate; the rest are quieter.
    assert spread[0] == pytest.approx(0.18)
    assert spread == sorted(spread, reverse=True)
    assert spread[-1] < 0.001
    # Both ends present: something exploring, something effectively greedy.
    assert sum(1 for value in spread if value > 0.01) >= 3
    assert sum(1 for value in spread if value < 0.01) >= 3

    # Degenerate widths must not divide by zero.
    assert actor_epsilons(0.18, 1, 4.0) == [0.18]


def test_paired_seats_explore_at_the_same_temperature():
    """Both orientations of a seed must be one policy, or the margin is a fiction.

    `collect_batch` indexes `epsilons` by pair for the same reason it indexes
    the opponent by pair: the league settles a seat-balanced `paired_margin`
    across the two orientations, and that number only means something if the
    same agent played both.
    """

    from agents.dqn_market_free.training.collect import collect_batch

    captured: list[tuple] = []

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "agents.dqn_market_free.training.collect._job",
            lambda payload: captured.append(payload) or None,
        )
        collect_batch(
            Path("weights.npz"),
            ["opponent-a", "opponent-b"],
            [11, 11, 22, 22, 33, 33],
            [0.5, 0.05, 0.005],
            0.999,
            Path("traces"),
            jobs=1,
        )

    # payload layout: (weights, opponent, seed, seat, epsilon, gamma, ...)
    by_seed: dict[int, set] = {}
    for payload in captured:
        by_seed.setdefault(payload[2], set()).add(payload[4])
    assert by_seed == {11: {0.5}, 22: {0.05}, 33: {0.005}}



def test_terminal_margin_discriminates_across_the_observed_range():
    """A hard clip at 20,000 made this term a constant, not a signal.

    Measured across 640 recorded episodes the final margin ran from -21,681 to
    -210,076 -- every one outside the clip -- so the term returned exactly -1.0
    every time. A constant on the terminal reward shifts every Q value uniformly
    and can never change an argmax, so the one term tying training to what the
    promotion gate measures was contributing nothing at all.
    """

    from agents.dqn_market_free.training.reward import (
        TERMINAL_MARGIN_SCALE,
        episode_rewards,
    )

    def terminal(margin: float) -> float:
        shaped, _, _ = episode_rewards([0.0, 0.0], [0.0, 0.0], 0.0, -margin, 1.0)
        return float(shaped[-1])

    observed = [-210_076.0, -126_975.0, -75_000.0, -21_681.0]
    values = [terminal(m) for m in observed]

    # Strictly ordered: a less bad loss must score better than a worse one.
    assert values == sorted(values)
    assert min(values) > -1.0 and max(values) < 0.0
    # And the spread is usable, not four values crowded onto the bound.
    assert max(values) - min(values) > 0.5

    # Bounded either side, so one blowout episode cannot dominate a batch.
    # tanh is open on (-1, 1) mathematically; float32 saturates at the bound.
    assert -1.0 <= terminal(-10_000_000.0) <= -0.99
    assert 0.99 <= terminal(10_000_000.0) <= 1.0
    # Symmetric about parity.
    assert terminal(0.0) == pytest.approx(0.0)
    assert terminal(TERMINAL_MARGIN_SCALE) == pytest.approx(-terminal(-TERMINAL_MARGIN_SCALE))


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
        REJECTED_ORDER_PENALTY,
        episode_rewards,
    )

    money = [3_000.0, 3_000.0, 2_880.0]
    potentials = [0.0, 0.0, 0.0]
    # turn 0 emitted and moved nothing (refused); turn 1 emitted and paid;
    # turn 2 emitted nothing at all.
    emitted = [True, True, False]

    shaped, raw, _ = episode_rewards(money, potentials, 2_880.0, 2_880.0, 1.0, emitted)

    assert shaped[0] == pytest.approx(-REJECTED_ORDER_PENALTY)
    # The accepted order keeps its own cost and takes no charge on top.
    assert shaped[1] == pytest.approx(raw[1])
    # A turn that emitted nothing is never charged, so NOOP stays above a
    # refused order rather than below it.
    assert shaped[2] > shaped[0]

    # Off by default for callers that cannot say what was emitted.
    plain, _, _ = episode_rewards(money, potentials, 2_880.0, 2_880.0, 1.0)
    assert plain[0] == pytest.approx(0.0)


def test_the_penalty_is_smaller_than_the_terminal_signal():
    """It is not policy-invariant, so it must not become the objective."""

    from agents.dqn_market_free.training.reward import REJECTED_ORDER_PENALTY

    # Worst case: every turn of a season emits a refused order.
    worst = REJECTED_ORDER_PENALTY * 719
    assert 0.1 < worst < 1.0
