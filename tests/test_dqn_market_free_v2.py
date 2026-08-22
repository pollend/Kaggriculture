"""What multi-order emission changed, and what it must not have.

`tests/test_dqn_market_free.py` covers the forked half. This module covers the
delta: the pending-turn walk, the mask that reads it, the intra-turn state block,
and the two places the training path had to stop assuming one order per turn.

The first test is the fork's safety net and the most important one here. v2 is a
copy of v1 with an inner loop bolted through it, so every inherited test is only
meaningful while the un-pended path is still the mask those tests were written
against.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from agents.dqn_market_free.order_space import feasible_actions as v1_feasible_actions
from agents.dqn_market_free_v2 import donor, herd, lots
from agents.dqn_market_free_v2.market_state import (
    HERD_VIABILITY_DIM,
    INTRA_TURN_DIM,
    MARKET_STATE_DIM,
    ORDER_LOOP_CAP,
    order_refused,
)
from agents.dqn_market_free_v2.order_space import (
    ACTIONS,
    sell_depth_cap,
    MAX_DECISIONS_PER_TURN,
    NOOP_INDEX,
    Order,
    PendingTurn,
    RESERVE_ABS,
    RESERVE_FRAC,
    feasible_actions,
    order_index,
)
from tests.test_dqn_market_free import _mask_state


def _pending(**kwargs) -> tuple[PendingTurn, tuple]:
    state = _mask_state(**kwargs)
    obs, farm, private, field = state
    return PendingTurn(obs, farm, private, field, None), state


def test_unpended_mask_is_v1s_mask():
    """The fork's safety net: `pending=None` must be v1, element for element.

    Every feasibility test v2 inherited was written against v1's mask. The moment
    this drifts, those tests are asserting something nobody wrote them to assert --
    and they will keep passing while they do it.
    """

    mismatches = []
    for money, shed, seeds, hires, day, plants, inventory in itertools.product(
        (0, 80, 700, 3000, 40000),
        (None, {"WHEAT": 30}, {"MELON": 5, "EGG": 2}),
        (None, {"MELON": 3}),
        (0, 3, 9),
        (0, 12, 29),
        (0, 6),
        (None, {"WHEAT": 9000}),
    ):
        obs, farm, private, field = _mask_state(
            money=money,
            shed=shed,
            seeds=seeds,
            hires_today=hires,
            day=day,
            plants=plants,
            inventory=inventory,
        )
        if not np.array_equal(
            v1_feasible_actions(obs, farm, private, field, None),
            feasible_actions(obs, farm, private, field, None),
        ):
            mismatches.append((money, shed, seeds, hires, day, plants, inventory))
    assert not mismatches


def test_state_width_names_its_own_arithmetic():
    assert MARKET_STATE_DIM == 238 + INTRA_TURN_DIM + HERD_VIABILITY_DIM
    # One affordability column per animal, plus the three species triples. Pinned
    # against the roster rather than the literal 10 so a fourth animal cannot widen
    # the vector without this failing.
    assert HERD_VIABILITY_DIM == 1 + 3 * len(donor.ANIMALS)
    # The two constants are deliberately duplicated across the state and action
    # schemas -- see their comments -- so the pin belongs here.
    assert ORDER_LOOP_CAP == MAX_DECISIONS_PER_TURN


def test_pending_sale_shuts_the_stock_it_promised_away():
    pending, (obs, farm, private, field) = _pending(money=3000, shed={"WHEAT": 6})
    rows = [index for index, order in enumerate(ACTIONS) if order.op == "SELL" and order.item == "WHEAT"]
    assert feasible_actions(obs, farm, private, field, None, pending=pending)[rows].any()

    pending.apply(Order("SELL", "WHEAT", 6))
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert not mask[rows].any(), "wheat already queued for sale is not still in the shed"
    assert mask[NOOP_INDEX], "the terminator is never shut"


def test_pending_purchase_spends_the_wallet_for_the_next_decision():
    """The failure this whole design exists to stop.

    v1 could not queue two orders, so a buy it could afford was a buy it could
    afford. Ten of them against one wallet is how a policy part-fills itself to
    zero -- 31.5% of v1's ladder buys, per the episode review.
    """

    pending, (obs, farm, private, field) = _pending(money=3000, inventory={"WHEAT": 9000})
    rows = [
        index
        for index, order in enumerate(ACTIONS)
        if order.op == "BUY_PRODUCT" and order.item == "WHEAT"
    ]
    assert feasible_actions(obs, farm, private, field, None, pending=pending)[rows].any()

    before = pending.money
    pending.apply(Order("BUY_PRODUCT", "WHEAT", 45))
    assert pending.money < before
    pending.money = 1.0  # whatever the quote was, the wallet is empty now
    assert not feasible_actions(obs, farm, private, field, None, pending=pending)[rows].any()


def test_reserve_floor_shuts_buys_but_never_sales_or_the_terminator():
    pending, (obs, farm, private, field) = _pending(
        money=int(RESERVE_ABS), shed={"WHEAT": 4}, inventory={"WHEAT": 9000}
    )
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    buys = [index for index, order in enumerate(ACTIONS) if order.op.startswith("BUY")]
    sells = [index for index, order in enumerate(ACTIONS) if order.op == "SELL"]
    assert not mask[buys].any(), "the reserve floor holds the whole buy side"
    assert mask[sells].any(), "selling is how the floor is climbed back over"
    assert mask[NOOP_INDEX]


def test_reserve_is_measured_against_the_money_the_turn_opened_with():
    pending, _ = _pending(money=10_000)
    assert pending.reserve == pytest.approx(max(RESERVE_ABS, RESERVE_FRAC * 10_000))
    pending.apply(Order("SELL", "WHEAT", 4))
    assert pending.reserve == pytest.approx(max(RESERVE_ABS, RESERVE_FRAC * 10_000)), (
        "a turn that sells raises its own ceiling rather than its floor"
    )


def test_liquidation_shuts_every_buy_and_leaves_the_sale():
    obs, farm, private, field = _mask_state(money=40_000, shed={"WHEAT": 20}, day=29)
    field = {**field, "liquidation": True}
    pending = PendingTurn(obs, farm, private, field, None)
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    buys = [index for index, order in enumerate(ACTIONS) if order.op.startswith("BUY")]
    assert not mask[buys].any()
    assert mask[[index for index, order in enumerate(ACTIONS) if order.op == "SELL"]].any()


def test_wire_slots_bound_the_turn():
    pending, (obs, farm, private, field) = _pending(money=40_000, shed={"WHEAT": 45})
    assert pending.slots_left == 10
    for _ in range(10):
        pending.apply(Order("SELL", "WHEAT", 1))
    assert pending.slots_left == 0
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert mask[NOOP_INDEX]
    assert not mask[1:].any(), "a full wire leaves only the terminator"


def test_hire_claims_one_slot_per_hand():
    """`HIRE n` is n wire entries, which is why the wire cap is not a decision cap.

    v1 emitted `HIRE 11` into a ten-order cap and the simulator silently dropped
    the eleventh -- visible in the ladder replays as five turns of eleven orders.
    """

    pending, (obs, farm, private, field) = _pending(money=40_000)
    for _ in range(8):
        pending.apply(Order("SELL", "WHEAT", 1))
    assert pending.slots == 8
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    open_hires = [
        ACTIONS[index].quantity
        for index in np.flatnonzero(mask)
        if ACTIONS[index].op == "HIRE"
    ]
    assert open_hires and max(open_hires) <= 2, "only what the remaining slots can carry"


def test_one_hire_action_per_turn():
    """`HIRE 7` then `HIRE 3` is ten `["HIRE"]` entries, which is `HIRE 10`'s wire.

    The ladder exists so that each reachable hand count has exactly one row. A
    second hire action in the same turn rebuilds the alias one sub-decision later,
    and -- worse -- makes the turn unrecoverable from its own replay, because
    `clone._chosen_sequence` can only map that wire back to a single row.
    """

    pending, (obs, farm, private, field) = _pending(money=40_000)
    pending.apply(Order("HIRE", None, 3))
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert not any(ACTIONS[index].op == "HIRE" for index in np.flatnonzero(mask))


def test_view_moves_the_world_the_turn_has_already_committed():
    """Block E says *that* something was queued; the view says what of.

    Without it every feature block but E reads the turn's opening observation, so a
    policy that has just sold its wheat is still shown holding it.
    """

    obs, farm, private, field = _mask_state(money=3000, shed={"WHEAT": 30})
    # The mask fixture carries no `farms`, and this is the one behaviour that
    # needs it: the opponent-facing columns index `farms` by player, so the view
    # has to keep that entry and the farm dict the same object.
    obs = {**obs, "player": 0, "farms": [farm, {"money": 3000}]}
    pending = PendingTurn(obs, farm, private, field, None)
    pending.apply(Order("SELL", "WHEAT", 24))
    obs_view, farm_view, private_view = pending.view(obs, farm, private)

    assert private_view["shed"]["WHEAT"] == 6
    assert farm_view["money"] > 3000
    assert obs_view["market"]["inventory"]["WHEAT"] > int(
        (obs.get("market", {}).get("inventory", {}) or {}).get("WHEAT", 10_000)
    )
    assert private["shed"]["WHEAT"] == 30, "the caller's own dicts are not mutated"
    player = int(obs_view.get("player", 0) or 0)
    assert obs_view["farms"][player] is farm_view, "farms and farm cannot disagree"


def test_expected_delta_signs_match_the_wallet():
    pending, _ = _pending(money=3000, shed={"WHEAT": 10}, inventory={"WHEAT": 9000})
    opening = pending.money
    pending.apply(Order("SELL", "WHEAT", 4))
    pending.apply(Order("BUY_SEED", "MELON", 2))
    assert pending.expected_delta == pytest.approx(pending.money - opening)


def test_refusal_reads_the_quote_when_a_turn_holds_both_sides():
    """A filled buy and a filled sale can net to zero. v1's test calls that refused."""

    emitted = np.asarray([True, True])
    delta = np.asarray([0.0, 0.0])
    # Quoted at a wash -- a buy against a sale of the same size. Undecidable from
    # the wallet, and it fails toward filled on purpose: see `order_refused`.
    assert not order_refused(emitted, delta, np.asarray([0.0, 0.0]))[0]
    # Quoted at a real outlay that never left the wallet: refused.
    assert order_refused(emitted, delta, np.asarray([0.0, -1200.0]))[1]
    # A quoted outlay that mostly landed is filled, whatever the interleave cost.
    assert not order_refused(
        np.asarray([True]), np.asarray([-1100.0]), np.asarray([-1200.0])
    )[0]
    # Without a quote this is v1's rule, unchanged.
    assert order_refused(emitted, delta).all()


def test_lot_events_price_a_mixed_turn_per_order():
    """`lots.match` split one net wallet move between a buy and a sale.

    Bought stock took `-money_delta` as its basis, so a purchase on a turn that also
    sold read as cheaper than it was -- or, past the crossover, as a negative basis.
    """

    seeds = [[0] * len(lots.CROP_ITEMS)] * 3
    wheat = lots.STOCK_ITEMS.index("WHEAT")
    stock = [[0] * len(lots.STOCK_ITEMS) for _ in range(3)]
    stock[1][wheat] = 10  # bought 10 wheat on turn 0
    stock[2][wheat] = 10
    # Turn 0 bought 10 wheat for 800 and sold something else for 750: a net of -50.
    money = [3000.0, 2950.0, 2950.0]
    events = [
        [("BUY_PRODUCT", "WHEAT", 10, -800.0), ("SELL", "MELON", 3, 750.0)],
        [],
        [],
    ]
    ledger = lots.match(seeds, stock, money, ["BUY_PRODUCT", None, None], events=events)
    bought = [lot for lot in ledger.lots if lot.bought]
    assert bought and bought[0].unit_cost == pytest.approx(80.0)

    naive = lots.match(seeds, stock, money, ["BUY_PRODUCT", None, None])
    assert naive.lots and naive.lots[0].unit_cost == pytest.approx(5.0), (
        "the per-turn net is what this test exists to show is wrong"
    )


def test_clone_maps_a_multi_order_turn_to_a_sequence():
    from agents.dqn_market_free_v2.training.clone import _chosen_sequence

    orders = [
        ["SELL", "WHEAT", 6],
        ["BUY_SEED", "MELON", 2],
        ["HIRE"],
        ["HIRE"],
        ["BUY_LAND"],
    ]
    indices, unrepresentable, land = _chosen_sequence(orders)
    assert unrepresentable == 0
    assert land == 1
    assert [ACTIONS[index].op for index in indices] == ["SELL", "BUY_SEED", "HIRE"]
    assert ACTIONS[indices[-1]].quantity == 2, "repeated HIRE rows fold into one action"
    assert indices[0] == order_index("SELL", "WHEAT", 6)


# --------------------------------------------------------------------------
# Live / replay lockstep
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_cloned_history_block_matches_the_live_agent():
    """The history block has two constructions, and they must agree.

    Ported from v1's test of the same name -- which is the whole point of adding
    it here. `market_state.py` and `training/clone.py` both carry comments
    promising this is pinned by a `test_clone_matches_live_features`, and no test
    by that or any other name existed for v2: the fork inherited the comment and
    not the guard.

    Everything else in the state is a function of one observation, so a live
    rollout and a replay conversion cannot disagree about it. The history block is
    a function of the whole episode so far, built by `main.py` as the agent plays
    and rebuilt by `clone.py` walking the stored replay -- and if the two push at
    different points in their loops, every demonstration carries a state the live
    policy will never see, silently.

    v2 adds a second way for the two to disagree that v1 did not have. A turn is
    now a *sequence* of sub-decisions, so the clone has to reproduce not just the
    history block but how many rows the turn produced and in which order, and
    `_chosen_sequence` has to map the demonstrator's wire orders back onto the
    action space that emitted them. Both are checked here by the shape assertion
    before the block comparison ever runs.

    The comparison starts one turn after the lag fills. `steps[0]` is the
    environment's own reset rather than a turn the agent played, so the first
    difference the clone can compute is one turn behind the live agent's.
    """

    from kaggle_environments import make

    from agents.dqn_market_free_v2.main import MarketFreeAgent
    from agents.dqn_market_free_v2.market_state import HISTORY_LAG
    from agents.dqn_market_free_v2.policy import OrderQNetwork
    from agents.dqn_market_free_v2.training.clone import record_from_replay
    from agents.dqn_market_free_v2.training.collect import Recorder
    from tests.test_dqn_market_free import SNAPSHOT_PATH

    recorder = Recorder()
    learner = MarketFreeAgent(
        # A randomly initialized head, not `quiet_start`: this needs the agent to
        # actually trade so the history block has something in it, and a quiet
        # start emits nothing at all. A random head's argmax varies across states
        # on its own.
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

    # Block D sits between the snapshot columns and block E, so it is addressed
    # from the end rather than from `MARKET_STATE_DIM - 30` as it was in v1.
    end = MARKET_STATE_DIM - INTRA_TURN_DIM
    start_row = HISTORY_LAG + 1
    np.testing.assert_allclose(
        cloned.order_state[start_row:, end - 30 : end],
        live[start_row:, end - 30 : end],
        atol=1e-6,
    )
    # And the block is not trivially equal because it is all zeros.
    assert np.abs(live[start_row:, end - 30 : end]).sum() > 0.0


def _herd_state(shed: dict | None = None, day: int = 0, tiles: dict | None = None):
    """A board with one unlocked quadrant, and whatever the case puts on it."""

    obs, farm, private, _field = _mask_state(shed=shed, day=day)
    for position, tile in (tiles or {}).items():
        x, y = position
        farm["tiles"][y][x] = tile
    obs = {**obs, "player": 0, "farms": [farm]}
    return obs, farm, private


def _animal_roles(roles: dict) -> dict:
    return {
        position: item
        for position, (kind, item) in roles.items()
        if kind == "ANIMAL"
    }


def test_no_animal_buys_no_herd_but_one_warm_slot():
    """The 52%-of-episodes waste, and the reason it was a waste.

    `donor._herd_targets` floors its targets at `CORE_HERD_SEQUENCE.count()`
    whatever the farm owns, so v1 built 14.6 pastures an episode in the 52% of
    episodes that bought no animal at all, and those episodes ended 16% poorer.
    One warm structure is deliberate and is the other half of the fix -- without it
    the first purchase of any species waits on a build and a place, which is exactly
    the delay that made the goose column unlearnable.
    """

    obs, farm, private = _herd_state()
    base = _animal_roles(donor._role_plan(obs, farm))
    assert len(base) >= donor.CORE_HERD, "the donor still plans a speculative herd"

    roles = _animal_roles(herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm)))
    assert len(roles) == herd.WARM_SLOTS


def test_a_bought_animal_gets_a_structure_whatever_the_species():
    """Every species, one turn, symmetrically. The goose is the case that regressed.

    v1 housed a shed-held goose and nothing else: `donor._role_plan` emits COW and
    SHEEP only, so a bought bird's coop came from `herd.py` while a bought cow landed
    on a pasture the donor had already built speculatively. The learner priced that
    asymmetry correctly and stopped buying geese.
    """

    for animal in herd.ANIMALS:
        obs, farm, private = _herd_state(shed={animal: 2})
        roles = _animal_roles(
            herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm))
        )
        housed = [item for item in roles.values() if item == animal]
        assert len(housed) >= 2, f"{animal} bought twice, housed {len(housed)}"


def test_a_placed_animal_keeps_its_own_species_role():
    """The role repair, from the direction that destroys an animal.

    A COW role standing on a COOP is the priority-3 `replace_incompatible_structure`
    dig. The dig spares a tile that already carries an animal, but the role is what
    `_seed_needs`, the state's `role_targets` block and the liquidation path read.
    """

    obs, farm, private = _herd_state(
        tiles={(1, 1): {"kind": "COOP", "animal": "GOOSE", "yield_units": 0}}
    )
    roles = _animal_roles(
        herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm))
    )
    assert roles[(1, 1)] == "GOOSE"


def test_a_bare_structure_is_not_torn_down_or_double_counted():
    """A built coop keeps a goose role and absorbs the warm slot.

    Every animal passes through a bare structure between `BUILD_` and `PLACE`, and
    the empty-structure state is also what the simulator's escape rule leaves behind.
    Neither may read as "build another one".
    """

    obs, farm, private = _herd_state(tiles={(1, 1): {"kind": "COOP"}})
    roles = _animal_roles(
        herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm))
    )
    assert roles == {(1, 1): "GOOSE"}


def test_the_warm_slot_closes_when_nothing_can_still_produce():
    """Past every deadline a structure is a hand-turn spent on an asset that cannot pay."""

    obs, farm, private = _herd_state(day=donor.TOTAL_DAYS - 4)
    assert herd.warm_slot_species(obs, int(obs["day"])) is None
    roles = _animal_roles(
        herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm))
    )
    assert roles == {}


def test_the_warm_slot_follows_the_market_the_herd_is_selling_into():
    """One cow is worth more than one goose; the third cow is not.

    The scorer prices the *herd's* remaining output, so MILK walks down its own
    `linear 1.60` curve while EGG's `log 0.20` barely moves. This is the fleet effect
    the Kaggle loss was made of and it cannot be seen one animal at a time.
    """

    assert herd.warm_slot_species({}, 0, {}) == "COW"
    assert herd.warm_slot_species({}, 0, {"COW": 1}) == "GOOSE"


def test_remaining_yield_events_agrees_with_the_ledger():
    """`herd` ships in the Kaggle bundle and `lots` does not, so the arithmetic is
    duplicated. It must not drift: the state column and the waste charge are the
    same number."""

    for animal in herd.ANIMALS:
        for day in range(donor.TOTAL_DAYS):
            assert herd.remaining_yield_events(animal, day) == lots.unplaced_events(
                animal, day
            )


def test_thin_markets_lose_the_rungs_that_crater_them():
    """WOOL and MILK are the thinnest markets in the game; WHEAT and EGG are not."""

    pending, (obs, farm, private, field) = _pending(
        money=3000, shed={"WHEAT": 45, "EGG": 45, "WOOL": 45, "MILK": 45}
    )
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)

    def top(item: str) -> int:
        return max(
            ACTIONS[index].quantity
            for index in np.flatnonzero(mask)
            if ACTIONS[index].op == "SELL" and ACTIONS[index].item == item
        )

    assert top("WHEAT") == 45 and top("EGG") == 45
    assert top("WOOL") < 45 and top("MILK") < top("WOOL")


def test_the_cap_tightens_as_the_turn_sells_into_it():
    """The cap is per order and reprices itself off the inventory its own sales made,
    which is what makes a second budget unnecessary."""

    pending, (obs, farm, private, field) = _pending(money=3000, shed={"MILK": 90})

    def cap() -> int:
        return sell_depth_cap(
            "MILK", int(pending.market_inventory.get("MILK", donor.MARKET_I0)), obs
        )

    before = cap()
    # Rung-quantised, so one order at the cap need not move it a whole rung. Four
    # do, and four is what a turn with slots to spare would emit.
    for _ in range(4):
        pending.apply(Order("SELL", "MILK", before))
    assert cap() < before


def test_liquidation_lifts_the_cap():
    """Stock the season ends holding is worth zero however cheaply it went, and the
    endgame is short of wire slots rather than of price."""

    pending, (obs, farm, private, field) = _pending(money=3000, shed={"WOOL": 45})
    field = {**field, "liquidation": True}
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert max(
        ACTIONS[index].quantity
        for index in np.flatnonzero(mask)
        if ACTIONS[index].op == "SELL" and ACTIONS[index].item == "WOOL"
    ) == 45
