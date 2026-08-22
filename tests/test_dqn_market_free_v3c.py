"""What the recurrence changed, and what it must not have.

`tests/test_dqn_market_free_v2.py` covers the multi-order delta this agent
inherits. This module covers v3's: a GRU that ticks once per turn, a head that
scores `concat(hidden, state)`, and the numpy mirror of both that ships to Kaggle.

The parity tests are the centre of gravity. v2 had two implementations of an MLP
and a disagreement cost one wrong decision; v3 has two implementations of a
*recurrence*, and a disagreement compounds over 720 ticks into a submission that
plays a different policy than the one the gate measured.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from agents.dqn_market_free_v3c import donor, herd, lots
from agents.dqn_market_free_v3c.market_state import (
    HERD_VIABILITY_DIM,
    HISTORY_BLOCK_DIM,
    INTRA_TURN_DIM,
    INTRA_TURN_SLICE,
    MARKET_FLOW_DIM,
    MARKET_STATE_DIM,
    PRODUCTS,
    REFUSAL_BLOCK_DIM,
)
from agents.dqn_market_free_v3c.order_space import (
    _BUY_ANIMAL_ROWS,
    _ROW_QUANTITY,
    animal_room,
    ACTION_COUNT,
    ACTIONS,
    MASKED_SCORE,
    Order,
    PendingTurn,
    feasible_actions,
    sell_depth_cap,
)
from agents.dqn_market_free_v3c.policy import (
    _LAYER_KEY,
    HEAD_IN,
    HIDDEN_SIZE,
    OrderQNetwork,
    _sigmoid,
)
from agents.dqn_market_free_v3c.training import network as net_module
from tests.test_dqn_market_free import _mask_state

STATE_DIM = MARKET_STATE_DIM


def _module(seed: int = 0, noisy: str = "none", **fill) -> net_module.OrderQModule:
    """A built module with non-trivial GRU biases.

    Torch initialises `bias_hh` from `U(-1/sqrt(H), 1/sqrt(H))`, which at H = 128
    is +-0.088 -- small enough that a wrong reset-gate association is inside a
    1e-5 tolerance on a short sequence. Every parity test here therefore widens
    the biases first, so the terms that distinguish the right form from the wrong
    one are actually present in the arithmetic being compared.
    """

    torch.manual_seed(seed)
    module = net_module.build(noisy=noisy)
    with torch.no_grad():
        module.cell.bias_hh.uniform_(-1.5, 1.5)
        module.cell.bias_ih.uniform_(-1.5, 1.5)
        for name, value in fill.items():
            getattr(module.cell, name).fill_(value)
    return module


# --------------------------------------------------------------------------
# Widths
# --------------------------------------------------------------------------


def test_state_width_names_its_own_arithmetic():
    assert MARKET_STATE_DIM == (
        238
        - HISTORY_BLOCK_DIM
        + REFUSAL_BLOCK_DIM
        + MARKET_FLOW_DIM
        + INTRA_TURN_DIM
        + HERD_VIABILITY_DIM
    )
    # Three of block D's thirty came back and twenty-seven did not.
    assert REFUSAL_BLOCK_DIM == 3
    # Block F: flow, own emitted and the day window per product, three tick-phase
    # scalars, and two headroom bands per product.
    assert MARKET_FLOW_DIM == 3 * len(PRODUCTS) + 3 + 2 * len(PRODUCTS)
    assert MARKET_STATE_DIM == 280
    assert HISTORY_BLOCK_DIM == 30, "the block the GRU replaced was 30 columns wide"
    # One affordability column per animal, plus the three species triples. Pinned
    # against the roster rather than the literal 10 so a fourth animal cannot widen
    # the vector without this failing.
    assert HERD_VIABILITY_DIM == 1 + 3 * len(donor.ANIMALS)
    assert HEAD_IN == MARKET_STATE_DIM + HIDDEN_SIZE
    assert INTRA_TURN_SLICE == slice(MARKET_STATE_DIM - INTRA_TURN_DIM, MARKET_STATE_DIM)


def test_no_gru_key_can_inflate_the_head_layer_count():
    """`load` counts head layers by regex, and the GRU keys must not match it.

    Asserted rather than left to inspection because the consequence is silent:
    a `gru_*` key that matched would make `data[f"w{count - 1}"]` raise KeyError,
    which `main.load_network` swallows into a fallback to `zeros()`. Challenger
    and incumbent would then play as one policy and every paired margin would
    come out exactly 0.0.
    """

    for key in ("gru_wx", "gru_wh", "gru_bx", "gru_bh", "hidden_size"):
        assert not _LAYER_KEY.fullmatch(key)


# --------------------------------------------------------------------------
# numpy / torch parity
# --------------------------------------------------------------------------


def test_numpy_and_torch_recurrences_agree_over_a_sequence():
    module = _module()
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(0)
    batch, length, decisions = 4, 64, 3
    turns = rng.standard_normal((batch, length, STATE_DIM)).astype(np.float32)
    states = rng.standard_normal((batch, length, decisions, STATE_DIM)).astype(np.float32)
    # Not zeros. At `h0 = 0` the hidden projection and `b_hh` are both invisible on
    # the first step and enter only faintly afterwards, so a transposed `wh` shows
    # up as slow drift rather than as a failure.
    hidden = rng.standard_normal((batch, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        rolled = module.recur(torch.from_numpy(turns), torch.from_numpy(hidden))
        expected = module.head_forward(torch.from_numpy(states), rolled[:, :, None, :])
    observed, final = network.forward_sequence(turns, states, hidden)

    assert observed.shape == (batch, length, decisions, ACTION_COUNT)
    assert np.allclose(final, rolled[:, -1].numpy(), atol=1e-5)
    assert np.allclose(observed, expected.numpy(), atol=1e-4)


def _reset_before_bias(gru, x, h):
    """The plausible wrong GRU: the reset gate applied before `b_hn` is added."""

    size = gru.hidden_size
    gx = x @ gru.wx + gru.bx
    hh = h @ gru.wh
    r = _sigmoid(gx[:, :size] + hh[:, :size] + gru.bh[:size])
    z = _sigmoid(gx[:, size : 2 * size] + hh[:, size : 2 * size] + gru.bh[size : 2 * size])
    n = np.tanh(gx[:, 2 * size :] + r * hh[:, 2 * size :] + gru.bh[2 * size :])
    return (1.0 - z) * n + z * h


def test_the_reset_gate_is_applied_after_the_hidden_bias():
    """The one parity break that is invisible at default biases.

    torch computes `n = tanh(W_in x + b_in + r * (W_hn h + b_hn))`. The plausible
    wrong form -- reset before the bias -- differs by exactly `(1 - r) * b_hn`,
    which is zero only while `b_hn` is zero, that is, never after the first
    optimizer step. So `b_hn` is driven large here and `b_in` to zero, and the
    test asserts both that numpy matches torch *and* that the wrong reference does
    not. The second half is what stops this passing vacuously.
    """

    module = _module(seed=1, bias_hh=3.0, bias_ih=0.0)
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((16, STATE_DIM)).astype(np.float32)
    h = rng.standard_normal((16, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        expected = module.cell(torch.from_numpy(x), torch.from_numpy(h)).numpy()
    assert np.allclose(network.gru.step(x, h), expected, atol=1e-5)
    assert not np.allclose(_reset_before_bias(network.gru, x, h), expected, atol=1e-3)


def _swapped_legs(gru, x, h):
    """The plausible wrong GRU: `z` treated as the write gate instead of the keep
    gate, i.e. `z * n + (1 - z) * h`."""

    size = gru.hidden_size
    gx = x @ gru.wx + gru.bx
    gh = h @ gru.wh + gru.bh
    r = _sigmoid(gx[:, :size] + gh[:, :size])
    z = _sigmoid(gx[:, size : 2 * size] + gh[:, size : 2 * size])
    n = np.tanh(gx[:, 2 * size :] + r * gh[:, 2 * size :])
    return z * n + (1.0 - z) * h


def test_the_update_gate_keeps_rather_than_writes():
    """`h' = (1 - z) * n + z * h`. Swapping the legs is a different network.

    Held to the same standard as the reset-gate test: match torch, and *fail* to
    match the wrong reference. A large positive update-gate bias -- `bias_iz`, the
    middle third of `bias_ih` -- pins `z` near 1, so the correct form carries `h`
    almost unchanged while the swapped one almost discards it, which is what makes
    the two separable at all.
    """

    module = _module(seed=2, bias_ih=0.0, bias_hh=0.0)
    with torch.no_grad():
        module.cell.bias_ih[HIDDEN_SIZE : 2 * HIDDEN_SIZE].fill_(8.0)
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((8, STATE_DIM)).astype(np.float32)
    h = rng.standard_normal((8, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        expected = module.cell(torch.from_numpy(x), torch.from_numpy(h)).numpy()
    assert np.allclose(network.gru.step(x, h), expected, atol=1e-5)
    assert not np.allclose(_swapped_legs(network.gru, x, h), expected, atol=1e-3)


def test_the_recurrence_does_not_drift_over_a_full_episode():
    """720 ticks. A per-step bias a short test cannot see compounds here."""

    module = _module(seed=4)
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(11)
    turns = rng.standard_normal((2, 720, STATE_DIM)).astype(np.float32)
    hidden = rng.standard_normal((2, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        expected = module.recur(torch.from_numpy(turns), torch.from_numpy(hidden))
        expected = expected[:, -1].numpy()
    observed = hidden.copy()
    for step in range(turns.shape[1]):
        observed = network.gru.step(turns[:, step], observed)

    error = float(np.abs(observed - expected).max())
    assert error < 1e-4, f"hidden drifted {error:.3e} over 720 ticks"


def test_the_stateful_path_is_the_same_arithmetic_as_the_batched_one():
    """The submission's path and the training path must not diverge from each other.

    `forward_sequence` loops over the identical `GruCell.step` and `_head` the
    stateful inference path calls, so the two are the same *expressions* -- but not
    bit-identical results, and the reason is worth writing down rather than
    absorbing into a loose tolerance. Inference scores a rank-1 state against a
    rank-2 weight and the batched path scores a rank-3 one, so BLAS dispatches a
    dot kernel in one case and a gemm in the other, and the two accumulate in
    different orders. The gap is pure float32 rounding, ~2e-7 here.

    The tolerance is therefore set just above that and nowhere near the 1e-4 the
    numpy-versus-torch tests use: anything larger than rounding means one of the
    two paths has grown arithmetic the other does not have, which is exactly how a
    submission and the training runs that produced it come to disagree while every
    cross-framework parity test still passes.
    """

    network = net_module.to_numpy(_module(seed=6))
    rng = np.random.default_rng(13)
    length, decisions = 32, 3
    turns = rng.standard_normal((1, length, STATE_DIM)).astype(np.float32)
    states = rng.standard_normal((1, length, decisions, STATE_DIM)).astype(np.float32)

    network.reset_hidden()
    rows = []
    for step in range(length):
        network.tick(turns[0, step])
        rows.append(np.stack([network(states[0, step, k]) for k in range(decisions)]))
    batched, _final = network.forward_sequence(turns, states, network.gru.zeros(1))

    error = float(np.abs(np.stack(rows) - batched[0]).max())
    assert error < 1e-6, f"the two paths differ by {error:.3e}, beyond float32 rounding"


def test_the_masked_choice_agrees_between_numpy_and_torch_over_a_sequence():
    """Integer equality on the argmax, not a tolerance on Q.

    A 1e-4 gap that flips one argmax is a real behavioural divergence that every
    float comparison above passes on.
    """

    module = _module(seed=8)
    network = net_module.to_numpy(module)
    rng = np.random.default_rng(17)
    batch, length, decisions = 3, 48, 2
    turns = rng.standard_normal((batch, length, STATE_DIM)).astype(np.float32)
    states = rng.standard_normal((batch, length, decisions, STATE_DIM)).astype(np.float32)
    hidden = rng.standard_normal((batch, HIDDEN_SIZE)).astype(np.float32)
    mask = rng.random((batch, length, decisions, ACTION_COUNT)) > 0.5
    mask[..., 0] = True  # NOOP is always available, as `feasible_actions` guarantees

    with torch.no_grad():
        rolled = module.recur(torch.from_numpy(turns), torch.from_numpy(hidden))
        expected = module.head_forward(torch.from_numpy(states), rolled[:, :, None, :]).numpy()
    observed, _final = network.forward_sequence(turns, states, hidden)

    torch_choice = np.where(mask, expected, MASKED_SCORE).argmax(axis=-1)
    numpy_choice = np.where(mask, observed, MASKED_SCORE).argmax(axis=-1)
    disagreed = int((torch_choice != numpy_choice).sum())
    assert disagreed == 0, f"{disagreed} of {torch_choice.size} decisions disagreed"


@pytest.mark.parametrize("noisy", ["head", "all"])
def test_noisy_numpy_and_torch_agree_over_a_sequence(noisy):
    """One epsilon draw, held across the whole turn.

    `main.py` redraws once per turn rather than once per sub-decision, so the
    parity test has to hold the draw the same way or it is comparing two different
    policies.
    """

    module = _module(seed=9, noisy=noisy)
    network = net_module.to_numpy(module)
    generator = torch.Generator().manual_seed(4)
    net_module.reset_noise(module, generator)
    network._noise = [
        (layer.epsilon_in.numpy().copy(), layer.epsilon_out.numpy().copy())
        if isinstance(layer, net_module.NoisyLinear)
        else None
        for layer in net_module._decompose(module)[1]
    ]

    rng = np.random.default_rng(19)
    turns = rng.standard_normal((2, 24, STATE_DIM)).astype(np.float32)
    states = rng.standard_normal((2, 24, 2, STATE_DIM)).astype(np.float32)
    hidden = rng.standard_normal((2, HIDDEN_SIZE)).astype(np.float32)

    with torch.no_grad():
        rolled = module.recur(torch.from_numpy(turns), torch.from_numpy(hidden))
        expected = module.head_forward(torch.from_numpy(states), rolled[:, :, None, :]).numpy()
    observed, _final = network.forward_sequence(turns, states, hidden)
    assert np.allclose(observed, expected, atol=1e-4)
    # And the noise is doing something, so this is not a comparison of two
    # unperturbed networks.
    network.clear_noise()
    quiet, _ = network.forward_sequence(turns, states, hidden)
    assert not np.allclose(quiet, observed, atol=1e-6)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_the_recurrence_is_never_noisy():
    """The invariant the TD step's `reset_noise` ordering rests on.

    v3 runs the *graded* GRU rollout before two `reset_noise` calls, which v2's
    ordering comment forbids -- legally, but only because nothing in the recurrent
    graph references an epsilon buffer. Put a `NoisyLinear` in the cell and the
    ordering silently becomes wrong.
    """

    for noisy in ("none", "head", "all"):
        module = net_module.build(noisy=noisy)
        assert not any(
            isinstance(layer, net_module.NoisyLinear) for layer in module.cell.modules()
        )


def test_an_export_raises_on_a_parameter_it_does_not_know_how_to_write():
    """`_decompose` accounts for parameters by identity, not by isinstance.

    The module is no longer a flat `Sequential`, so walking the children and
    recognising each one is no longer a proof of completeness -- a sub-module hung
    off `OrderQModule` and forgotten in the exporter would never be visited at all.
    """

    module = net_module.build()
    module.stray = torch.nn.Linear(4, 4)
    with pytest.raises(TypeError, match="cannot export parameters"):
        net_module.to_numpy(module)


def test_a_quiet_start_is_silent_but_not_inert():
    network = OrderQNetwork.quiet_start(seed=5)
    network.reset_hidden()
    rng = np.random.default_rng(23)
    network.tick(rng.standard_normal(STATE_DIM).astype(np.float32))
    scores = network(rng.standard_normal(STATE_DIM).astype(np.float32))
    assert np.allclose(scores, 0.0), "every action must tie, so the argmax falls to NOOP"
    # The head trunk is live, which is what `zeros()` cannot say...
    assert not net_module.has_inert_hidden_trunk(network)
    # ...and so is the recurrence.
    assert not net_module.has_inert_recurrence(network)
    assert net_module.has_inert_recurrence(OrderQNetwork.zeros())


def test_an_all_zero_network_stays_silent_for_the_whole_episode():
    """The inference fallback ties on turn 720, not only on turn 1.

    At zero weights `h' = (1 - 0.5) * tanh(0) + 0.5 * h`, so a hidden state that
    starts at zero stays there and the head keeps seeing the same dead columns.
    """

    network = OrderQNetwork.zeros()
    network.reset_hidden()
    rng = np.random.default_rng(29)
    for _ in range(50):
        network.tick(rng.standard_normal(STATE_DIM).astype(np.float32))
    assert np.allclose(network.hidden, 0.0)
    assert np.allclose(network(rng.standard_normal(STATE_DIM).astype(np.float32)), 0.0)


def test_scoring_before_a_reset_raises_rather_than_scoring_against_zeros():
    """A network that silently scored against zeros would be indistinguishable
    from a trained policy that had learned to ignore its memory."""

    network = OrderQNetwork.quiet_start(seed=2)
    with pytest.raises(RuntimeError, match="reset_hidden"):
        network(np.zeros(STATE_DIM, dtype=np.float32))
    with pytest.raises(RuntimeError, match="reset_hidden"):
        network.tick(np.zeros(STATE_DIM, dtype=np.float32))


def test_the_stateful_and_batched_paths_reject_each_others_input():
    network = OrderQNetwork.quiet_start(seed=3)
    network.reset_hidden()
    with pytest.raises(ValueError, match="single-decision path"):
        network(np.zeros((4, STATE_DIM), dtype=np.float32))
    with pytest.raises(ValueError, match="single-sample path"):
        network.tick(np.zeros((4, STATE_DIM), dtype=np.float32))


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def test_a_checkpoint_round_trips_and_a_pre_v3_one_is_refused_by_name(tmp_path):
    network = net_module.to_numpy(_module(seed=12, noisy="head"))
    path = tmp_path / "weights.npz"
    network.save(path)
    loaded = OrderQNetwork.load(path)

    for name in ("wx", "wh", "bx", "bh"):
        assert_array_equal(getattr(loaded.gru, name), getattr(network.gru, name))
    assert len(loaded.layers) == len(network.layers)
    assert loaded.is_noisy

    # A file with the head but no recurrence is a pre-v3 checkpoint, and must be
    # refused by name rather than by width -- `STATE_DIM + 30` is v2's head width
    # exactly, so a future `HIDDEN_SIZE` of 30 would let one load as a v3 network
    # with no memory at all.
    payload = dict(np.load(path))
    for name in ("gru_wx", "gru_wh", "gru_bx", "gru_bh"):
        payload.pop(name)
    stale = tmp_path / "v2.npz"
    np.savez_compressed(stale, **payload)
    with pytest.raises(ValueError, match="carries no GRU"):
        OrderQNetwork.load(stale)


def test_half_a_gru_is_refused(tmp_path):
    network = net_module.to_numpy(_module(seed=14))
    path = tmp_path / "weights.npz"
    network.save(path)
    payload = dict(np.load(path))
    payload.pop("gru_wh")
    partial = tmp_path / "partial.npz"
    np.savez_compressed(partial, **payload)
    with pytest.raises(ValueError, match="partial GRU"):
        OrderQNetwork.load(partial)


# The herd, and the ladder that made half of it unsellable
# --------------------------------------------------------------------------


def _pending(**kwargs) -> tuple[PendingTurn, tuple]:
    state = _mask_state(**kwargs)
    obs, farm, private, field = state
    return PendingTurn(obs, farm, private, field, None), state


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


def test_the_donors_speculative_herd_survives_the_overlay():
    """The strip is gone, and this is the test that used to pin it.

    It asserted `len(roles) == WARM_SLOTS` -- one animal role on a zero-animal
    board, the donor's four thrown away -- on the grounds that 52% of episodes never
    bought the animal and those episodes ended 16% poorer.

    Both halves of that failed. The waste it named was already refused one layer
    down: `donor._field_jobs` emits `BUILD_` only at `left >= animal_rule["first"]
    + 2` and never during liquidation, so a role the season cannot pay back yields
    no job. And the 52% was measured on a learner that was not buying animals, which
    is circular -- the strip deletes the housing, the housing is what makes a
    purchase land. `market-free-splitladder-v1` averages 9.7 cows, 7.4 sheep and 8.9
    geese an episode.

    Measured over 24 seeds in both orientations with the donor's market rules on
    both sides: 50,240 with the strip against 83,525 without, and pilkwang held to
    86,111 instead of running to 122,515.
    """

    obs, farm, private = _herd_state()
    base = _animal_roles(donor._role_plan(obs, farm))
    assert len(base) >= donor.CORE_HERD, "the donor still plans a speculative herd"

    roles = _animal_roles(
        herd.herd_influenced_roles(obs, farm, private, donor._role_plan(obs, farm))
    )
    assert len(roles) >= len(base), "the overlay adds housing, it does not take it away"


def test_the_late_season_build_is_refused_by_the_donor_not_by_the_overlay():
    """Where the protection the strip claimed actually lives.

    This is the guard that makes deleting the strip safe, so it is pinned here
    rather than left to `donor.py` -- a speculative role late in the season is free
    precisely because no `BUILD_` job comes out of it.
    """

    obs, farm, private = _herd_state()
    roles = herd.field_roles(obs, farm, private)
    animal_tiles = [position for position, role in roles.items() if role[0] == "ANIMAL"]
    assert animal_tiles, "the fixture has animal roles to build on"

    def _builds(day):
        jobs = donor._field_jobs(
            {**obs, "day": day}, farm, private, roles, False
        )
        return [job for job in jobs if str(job["action"][0]).startswith("BUILD_")]

    assert _builds(int(obs.get("day", 0) or 0)), "early, the build is planned"
    assert not _builds(donor.TOTAL_DAYS - 1), "late, the donor refuses it itself"


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
    planned = donor._role_plan(obs, farm)
    roles = _animal_roles(herd.herd_influenced_roles(obs, farm, private, planned))

    assert roles[(1, 1)] == "GOOSE", "the built coop keeps its species"
    # The claim is absorption, not emptiness: the coop satisfies `WARM_SLOTS`, so the
    # overlay adds nothing beyond it. Asserted against the donor's plan rather than
    # against `{}` -- the plan used to be stripped here and is not any more.
    assert set(roles) == set(_animal_roles(planned)) | {(1, 1)}


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


def test_a_buy_animal_beyond_free_housing_is_shut():
    """The leak this gate exists to close.

    `donor.ANIMAL_SLOTS` is 15 tiles board-wide and 4 in the opening quadrant, but
    `shedCapacity` is 100, so affordability alone let a turn buy ~100 head that
    `herd.herd_influenced_roles` can never find ground for. Measured at generation
    17 of `market-free-refusal-v1`, the five worst learner slots were each spending
    ~37,000 coins an episode on animals and recovering ~4,000, at an
    `animal_lots_matched_share` of 0.11.
    """

    obs, farm, private, field = _mask_state(money=100_000)
    assert len(donor._reserved_animal_slots(farm)[0]) == 4, "the opening quadrant"
    assert animal_room(obs, farm, private) == {a: 4 for a in herd.ANIMALS}

    # More head waiting in the shed than the board can ever house.
    crowded = {**private, "shed": {**(private.get("shed") or {}), "COW": 50}}
    assert animal_room(obs, farm, crowded) == {}, "no proof of a slot, no open row"
    pending = PendingTurn(obs, farm, crowded, field, None)
    mask = feasible_actions(obs, farm, crowded, field, None, pending=pending)
    for animal in herd.ANIMALS:
        assert not mask[_BUY_ANIMAL_ROWS[animal]].any(), animal


def test_the_rung_is_capped_at_free_housing_not_shut_wholesale():
    """A cap, not a ban. Two free tiles still buy two head."""

    obs, farm, private, field = _mask_state(money=100_000)
    crowded = {**private, "shed": {**(private.get("shed") or {}), "COW": 2}}
    assert animal_room(obs, farm, crowded) == {a: 2 for a in herd.ANIMALS}

    pending = PendingTurn(obs, farm, crowded, field, None)
    mask = feasible_actions(obs, farm, crowded, field, None, pending=pending)
    rows = _BUY_ANIMAL_ROWS["COW"]
    for index, quantity in enumerate(_ROW_QUANTITY[rows]):
        assert bool(mask[rows][index]) == (int(quantity) <= 2), quantity


def test_a_turn_cannot_stuff_more_animals_than_it_has_tiles():
    """The `PendingTurn` half. Without it the mask is right once and wrong four
    times, because the wire holds ten orders and the board holds four tiles."""

    obs, farm, private, field = _mask_state(money=100_000)
    pending = PendingTurn(obs, farm, private, field, None)
    rows = _BUY_ANIMAL_ROWS["COW"]
    assert feasible_actions(obs, farm, private, field, None, pending=pending)[rows].all()

    pending.apply(Order(op="BUY_ANIMAL", item="COW", quantity=4))
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    # Every species, not just the one bought: COW and SHEEP share the pasture and a
    # bare COOP is re-roled by `herd._species_for`, so the ground is gone for all.
    for animal in herd.ANIMALS:
        assert not mask[_BUY_ANIMAL_ROWS[animal]].any(), animal


@pytest.mark.parametrize("animal", sorted(herd.ANIMALS))
def test_an_animal_that_cannot_produce_before_the_season_ends_is_unbuyable(animal):
    """The calendar leg. `market_state.py:812-815` names this exact hole: past day
    21 a cow cannot milk once, the row stayed open, and the waste term charged the
    full 400 coins. Tied to `remaining_yield_events` rather than to a literal, so it
    cannot drift from the charge `lots.unplaced_events` prices."""

    dead = next(
        day
        for day in range(1, donor.TOTAL_DAYS + 1)
        if herd.remaining_yield_events(animal, day) == 0
    )
    obs, farm, private, field = _mask_state(money=100_000)

    alive = {**obs, "day": dead - 1}
    assert animal_room(alive, farm, private).get(animal, 0) > 0, "open the day before"

    expired = {**obs, "day": dead}
    assert animal in animal_room(alive, farm, private)
    assert animal not in animal_room(expired, farm, private)
    pending = PendingTurn(expired, farm, private, field, None)
    mask = feasible_actions(expired, farm, private, field, None, pending=pending)
    assert not mask[_BUY_ANIMAL_ROWS[animal]].any()


def test_buying_land_widens_the_animal_rows():
    """The gate is recomputed per turn off `_reserved_animal_slots`, so unlocking a
    quadrant reopens rows rather than freezing them at the NW count."""

    obs, farm, private, field = _mask_state(money=100_000)
    # The flag alone is not the purchase: `_reserved_animal_slots` skips a "LOCKED"
    # tile whatever the quadrant list says, so the ground has to open too.
    tiles = [list(row) for row in farm["tiles"]]
    for y in range(5):
        for x in range(5, 10):
            tiles[y][x] = None
    wide = {**farm, "tiles": tiles, "unlocked_quadrants": ["NW", "NE"]}

    assert len(donor._reserved_animal_slots(farm)[0]) == 4
    assert len(donor._reserved_animal_slots(wide)[0]) == 4 + donor.ANIMAL_SLOTS["NE"]
    assert animal_room(obs, wide, private)["COW"] > animal_room(obs, farm, private)["COW"]


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


# --------------------------------------------------------------------------
# Sequence replay and the TD step
# --------------------------------------------------------------------------


def _episode(seed: int = 0, turns: int = 200, rate: int = 3):
    """A synthetic episode with a variable number of decisions per turn."""

    from agents.dqn_market_free_v3c.training.collect import HIDDEN_STRIDE, EpisodeRecord

    rng = np.random.default_rng(seed)
    counts = rng.integers(1, rate + 1, size=turns)
    steps = np.repeat(np.arange(turns), counts).astype(np.int16)
    decisions = int(steps.shape[0])
    stored = int(np.ceil(turns / HIDDEN_STRIDE))
    return EpisodeRecord(
        reward=rng.standard_normal(turns).astype(np.float32),
        raw_reward=rng.standard_normal(turns).astype(np.float32),
        money=rng.standard_normal(turns + 1).astype(np.float32),
        order_state=rng.standard_normal((decisions, STATE_DIM)).astype(np.float32),
        order_chosen=rng.integers(0, ACTION_COUNT, size=decisions).astype(np.int16),
        order_step=steps,
        order_mask=np.ones((decisions, ACTION_COUNT), dtype=np.bool_),
        turn_state=rng.standard_normal((turns, STATE_DIM)).astype(np.float32),
        turn_hidden=rng.standard_normal((stored, HIDDEN_SIZE)).astype(np.float16),
        hidden_stride=np.asarray([HIDDEN_STRIDE], dtype=np.int16),
    )


def _buffer(count: int = 8, **kwargs):
    from agents.dqn_market_free_v3c.training.replay import ReplayBuffer

    buffer = ReplayBuffer(capacity=64, n_step=24, seed=1, **kwargs)
    buffer.add([_episode(seed=index) for index in range(count)])
    return buffer


def test_a_chunk_bootstraps_the_turn_n_step_ahead_and_never_clamps():
    """Every turn records at least a NOOP terminator, so `searchsorted` cannot
    skip the horizon. `clamped` is the counter that would say otherwise, and it
    is the difference between an n-step return and a silently shortened one."""

    batch = _buffer().sample_sequences(6, burn_in=20, train=40)
    assert batch.clamped == 0
    assert batch.turn_input.shape == (6, 20 + 40 + 24, STATE_DIM)
    rows = batch.state.shape[0]
    # Every target array is per decision now, not per turn.
    for name in ("reward", "discount", "next_turn", "chosen", "turn_share"):
        assert getattr(batch, name).shape == (rows,), name
    assert batch.next_state.shape == (rows, STATE_DIM)
    assert batch.next_mask.shape == (rows, ACTION_COUNT)
    # Each row's owning turn is inside its own chunk's window, and every
    # bootstrap lands inside the rolled post-burn-in grid.
    assert int(batch.decision_turn.max()) < 6 * 40
    assert int(batch.next_turn.max()) < 6 * (40 + 24)


def test_a_turns_sub_decisions_get_individuated_targets():
    """The correction this fork exists for.

    v2 gave every sub-decision of a turn the same reward, discount and bootstrap,
    so ten transitions were regressed onto one scalar and nothing in the target
    could say the seventh order was worse than the first. Its policy converged on
    a ten-slot zero-P&L wash for exactly that reason.

    The MDP a turn describes: sub-decisions are simultaneous in game time, so an
    intra-turn transition pays no reward and does not discount -- it asks "having
    queued that, what is the next order worth?". Time passes only at the boundary,
    where the turn's whole n-step return goes to the decision that closed it.
    """

    batch = _buffer().sample_sequences(6, burn_in=20, train=40)
    intra = batch.discount == 1.0
    boundary = ~intra

    assert intra.any() and boundary.any(), "need both kinds of row to test either"
    # Intra-turn rows: no reward, no discounting, successor scored against the
    # same latent -- the GRU does not advance inside a turn.
    assert np.all(batch.reward[intra] == 0.0)
    assert np.array_equal(
        batch.next_turn[intra] // (40 + 24), batch.decision_turn[intra] // 40
    )
    # Exactly one boundary row per turn, and it is the turn's last decision.
    turns, counts = np.unique(batch.decision_turn[boundary], return_counts=True)
    assert np.all(counts == 1)
    assert turns.size == np.unique(batch.decision_turn).size
    # Boundary discounts are gamma**n_step, or 0 where the episode ended inside
    # the horizon. Never 1.0 -- that would mean a turn boundary cost no time.
    live = batch.discount[boundary][batch.discount[boundary] > 0.0]
    assert np.allclose(live, 0.999**24, atol=1e-6)


def test_two_sub_decisions_of_a_turn_can_disagree():
    """The property v2 structurally could not have.

    Same turn, different successor states, therefore different targets. Asserted
    on the successors rather than on Q, because the sampler is what was wrong --
    v2's rows were identical before the network ever saw them.
    """

    batch = _buffer().sample_sequences(8, burn_in=20, train=40)
    by_turn: dict[int, list[int]] = {}
    for index, turn in enumerate(batch.decision_turn.tolist()):
        by_turn.setdefault(turn, []).append(index)
    multi = [rows for rows in by_turn.values() if len(rows) > 1]
    assert multi, "need a turn holding several sub-decisions"

    differing = 0
    for rows in multi:
        first, second = rows[0], rows[1]
        if not np.array_equal(batch.next_state[first], batch.next_state[second]):
            differing += 1
    assert differing == len(multi), (
        "every multi-decision turn must give its rows distinct successors; "
        "identical successors is exactly v2's defect"
    )


def test_a_turns_decisions_share_exactly_one_unit_of_loss_weight():
    """`turn_share` is the inner half of the two-level mean.

    Without it a turn holding eleven sub-decisions would contribute eleven times
    the gradient of one holding a single terminator -- which is a bias toward
    exactly the behaviour `decisions_per_turn` exists to watch for, and it arrived
    with the chunk draw rather than existing in v2.
    """

    batch = _buffer().sample_sequences(4, burn_in=20, train=40)
    totals = np.bincount(
        batch.decision_turn, weights=batch.turn_share, minlength=4 * 40
    )
    populated = totals[totals > 0]
    assert populated.size
    np.testing.assert_allclose(populated, 1.0, atol=1e-6)


def test_a_burn_in_off_the_stored_grid_is_refused():
    """`hidden0` has to be a state somebody stored, and `HIDDEN_STRIDE` decides
    which turns those are."""

    from agents.dqn_market_free_v3c.training.collect import HIDDEN_STRIDE

    with pytest.raises(ValueError, match="multiple of the stored-hidden stride"):
        _buffer().sample_sequences(2, burn_in=HIDDEN_STRIDE + 1, train=40)


def test_an_episode_too_short_for_a_training_window_is_not_drawn():
    """`record.decisions` -- the flat sampler's predicate -- would pass an episode
    with plenty of decisions packed into too few turns, and then index off the end
    of `turn_state`."""

    from agents.dqn_market_free_v3c.training.replay import ReplayBuffer

    buffer = ReplayBuffer(capacity=8, n_step=24, seed=0)
    buffer.add([_episode(seed=1, turns=30)])
    with pytest.raises(ValueError, match="long enough"):
        buffer.sample_sequences(2, burn_in=20, train=40)


def test_the_left_pad_of_a_chunk_cannot_move_the_hidden_state():
    """A padded step must be a no-op, not a zero input.

    A GRU fed zeros from a zero state settles on `(1 - z) * tanh(b_in + b_hn)`,
    not on zero, so a chunk that begins before its episode does would start its
    burn-in from a bias artefact that grows with every pad step. The gate in
    `recur` is what prevents that, and this pins it by filling the pad with
    garbage and asserting the answer does not move.
    """

    module = _module(seed=21)
    turn_input = torch.zeros(2, 12, STATE_DIM)
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[:, :5] = False
    torch.manual_seed(0)
    turn_input[:, 5:] = torch.randn(2, 7, STATE_DIM)
    hidden = torch.zeros(2, HIDDEN_SIZE)

    with torch.no_grad():
        padded = module.recur(turn_input, hidden, valid)[:, -1]
        # The same real turns with no pad at all.
        bare = module.recur(turn_input[:, 5:], hidden)[:, -1]
        # And the pad filled with something else entirely.
        noisy_pad = turn_input.clone()
        noisy_pad[:, :5] = torch.randn(2, 5, STATE_DIM) * 10.0
        polluted = module.recur(noisy_pad, hidden, valid)[:, -1]

    assert torch.allclose(padded, bare, atol=1e-6)
    assert torch.allclose(padded, polluted, atol=1e-6)


def test_the_gradient_reaches_the_recurrence_over_the_training_window():
    """The whole point of the fork, asserted rather than assumed.

    Also checks the burn-in is genuinely excluded: `_roll_chunk` runs it under
    `no_grad`, so the graph the loss backprops through is `train` steps deep and
    not `burn_in + train`.
    """

    from agents.dqn_market_free_v3c.training.updater import _roll_chunk

    module = _module(seed=22)
    batch = _buffer().sample_sequences(3, burn_in=20, train=40)
    device = torch.device("cpu")
    train_hidden, post = _roll_chunk(module, batch, device, 20, 40, grad=True)
    assert train_hidden.requires_grad
    assert not post.requires_grad, "the bootstrap latents must not carry gradient"

    rows = torch.from_numpy(batch.decision_turn)
    states = torch.from_numpy(batch.state)
    hidden = train_hidden.reshape(-1, module.hidden_size)[rows]
    module(torch.cat((states, hidden), dim=1)).sum().backward()

    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        grad = getattr(module.cell, name).grad
        assert grad is not None, f"{name} received no gradient"
        assert float(grad.abs().sum()) > 0.0, f"{name} gradient is identically zero"


def test_a_full_td_step_runs_and_moves_every_parameter():
    """The whole loop body, not just its pieces.

    Shape errors in the chunk machinery are concentrated in the gather indices --
    `boot_turn` into the flattened post-burn-in grid, `decision_turn` into the
    training grid -- and every one of them is a plausible off-by-`n_step` that
    still broadcasts. Running the real arithmetic once is the only thing that
    catches those.

    Asserts every parameter moved, the GRU's four tensors included -- which is the
    fork's premise stated as a test rather than as a comment.

    Two optimizer steps rather than one, because this module is built at torch's
    own initialisation and a real run is not: `initial_network` cold-starts from
    `quiet_start`, whose output layer is zero, so on its first step `dL/da` is
    identically zero and nothing upstream of it moves at all. Running two here
    keeps this test honest against the trajectory a real run takes.
    """

    from agents.dqn_market_free_v3c.order_space import MASKED_SCORE
    from agents.dqn_market_free_v3c.training.updater import _roll_chunk

    burn_in, train, chunks = 20, 40, 4
    online = _module(seed=31)
    target = _module(seed=31)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    buffer = _buffer()
    device = torch.device("cpu")
    before = {name: p.detach().clone() for name, p in online.named_parameters()}

    for _ in range(2):
        batch = buffer.sample_sequences(chunks, burn_in=burn_in, train=train)
        train_hidden, post_online = _roll_chunk(
            online, batch, device, burn_in, train, grad=True
        )
        with torch.no_grad():
            _t, post_target = _roll_chunk(
                target, batch, device, burn_in, train, grad=False
            )
            boot = torch.from_numpy(batch.next_turn)
            boot_state = torch.from_numpy(batch.next_state)
            boot_mask = torch.from_numpy(batch.next_mask)
            best = online(torch.cat((boot_state, post_online[boot]), dim=1))
            best = best.masked_fill(~boot_mask, MASKED_SCORE).argmax(dim=1, keepdim=True)
            bootstrap = (
                target(torch.cat((boot_state, post_target[boot]), dim=1))
                .gather(1, best)
                .squeeze(1)
            )
            goal = torch.from_numpy(batch.reward) + (
                torch.from_numpy(batch.discount) * bootstrap
            )

        rows = torch.from_numpy(batch.decision_turn)
        hidden = train_hidden.reshape(-1, online.hidden_size)[rows]
        predicted = (
            online(torch.cat((torch.from_numpy(batch.state), hidden), dim=1))
            .gather(1, torch.from_numpy(batch.chosen).unsqueeze(1))
            .squeeze(1)
        )
        per_row = loss_fn(predicted, goal)
        share = torch.from_numpy(batch.turn_share)
        per_turn = torch.zeros(chunks * train, dtype=per_row.dtype).index_add_(
            0, rows, per_row * share
        )
        per_chunk = per_turn.view(chunks, train).mean(dim=1)
        loss = (per_chunk * torch.from_numpy(batch.weight)).mean()

        assert torch.isfinite(loss), "a nan reached the loss"
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
        optimizer.step()

    unmoved = [
        name
        for name, p in online.named_parameters()
        if torch.equal(p.detach(), before[name])
    ]
    assert not unmoved, f"parameters never moved: {unmoved}"


# --------------------------------------------------------------------------
# Wallet-sized product buys
# --------------------------------------------------------------------------


def test_a_product_buy_rung_the_wallet_cannot_cover_is_shut():
    """One affordable unit must not unlock the whole quantity ladder.

    The simulator does not refuse an order it cannot fill -- it part-fills it and
    drains the wallet to zero -- so an open rung the wallet cannot cover is a real
    loss, not a rejected order. On the Kaggle ladder 31.5% of `BUY_PRODUCT` orders
    were sized past the wallet against opponents' 3.9%, with 0.0% refused.

    `BUY_SEED` has always masked per rung; this pins the two symmetric.
    """

    from agents.dqn_market_free_v3c import donor
    from agents.dqn_market_free_v3c.order_space import (
        _BUY_PRODUCT_ROWS,
        _ROW_QUANTITY,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    item = "WHEAT"
    rows = _BUY_PRODUCT_ROWS[item]
    quantities = _ROW_QUANTITY[rows]

    # A wallet that covers a handful of units and nowhere near the top rung.
    obs, farm, private, field = _mask_state(money=60.0)
    mask = feasible_actions(obs, farm, private, field, None)

    stock = int(
        ((obs.get("market") or {}).get("inventory") or {}).get(item, donor.MARKET_I0)
        or donor.MARKET_I0
    )
    cost = np.cumsum(
        [
            float(donor._price_at(item, stock - offset, obs))
            for offset in range(1, int(quantities.max()) + 1)
        ]
    )

    open_rows = mask[rows]
    assert open_rows.any(), "a wallet covering several units must leave some rung open"
    assert not open_rows.all(), "a 60-coin wallet cannot cover a 45-unit buy"
    # Every open rung is affordable and every shut one is not: the mask is exactly
    # the affordability predicate, not an approximation of it.
    for index, quantity in enumerate(quantities):
        affordable = cost[int(quantity) - 1] <= 60.0
        assert bool(open_rows[index]) == bool(affordable), (
            f"rung {quantity} open={bool(open_rows[index])} affordable={affordable}"
        )


def test_the_mask_prices_a_product_buy_the_way_pending_turn_does():
    """The mask and the wallet accounting must not disagree about one order's price.

    `PendingTurn.purchase_cost` is what actually debits the wallet when the order
    is queued. If the mask priced the same rung differently the policy would be
    offered orders the turn then cannot pay for -- the part-fill again, arriving by
    a different route.
    """

    from agents.dqn_market_free_v3c import donor
    from agents.dqn_market_free_v3c.order_space import PendingTurn
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0)
    pending = PendingTurn(obs, farm, private, field, None)
    item = "WHEAT"
    stock = int(
        ((obs.get("market") or {}).get("inventory") or {}).get(item, donor.MARKET_I0)
        or donor.MARKET_I0
    )
    for quantity in (1, 4, 12, 45):
        mask_cost = float(
            np.cumsum(
                [
                    float(donor._price_at(item, stock - offset, obs))
                    for offset in range(1, quantity + 1)
                ]
            )[-1]
        )
        assert mask_cost == pytest.approx(pending.purchase_cost(item, quantity))


# --------------------------------------------------------------------------
# The turn-opening encode, reused
# --------------------------------------------------------------------------


def _encode_case(field: dict, shed: dict, carried: dict):
    """`(reused, fresh, reusable)` for one turn, at its first sub-decision.

    `reused` is what `main.py` builds when it stands the turn-opening encode in
    for the first sub-decision; `fresh` is the full encode off `pending.view` that
    v2 did unconditionally. `reusable` is the gate `main.py` actually consults.
    """

    from agents.dqn_market_free_v3c.main import _shed_unmoved
    from agents.dqn_market_free_v3c.market_state import (
        intra_turn_block,
        market_state_features,
    )

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "tiles": tiles,
        "hands": [],
        "farmer": [4, 4],
    }
    private = {"shed": dict(shed), "seeds": {}, "inventories": [dict(carried)]}
    obs = {
        "player": 0,
        "step": 30,
        "day": 1,
        "hour": 6,
        "farms": [farm, farm],
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
        "private": private,
    }
    roles = herd.field_roles(obs, farm, private)
    pending = PendingTurn(obs, farm, private, field, None, plantable={}, roles=roles)

    turn_state = market_state_features(
        obs, farm, private, roles, field, turn=None
    )
    reused = turn_state.copy()
    reused[INTRA_TURN_SLICE] = intra_turn_block(pending)

    obs_view, farm_view, private_view = pending.view(obs, farm, private)
    fresh = market_state_features(
        obs_view, farm_view, private_view, roles, field, turn=pending
    )
    return reused, fresh, _shed_unmoved(pending, private)


def test_the_reused_turn_state_equals_a_fresh_encode():
    """The gate on `main.py`'s encode-reuse, in both directions.

    Reusing the turn-opening vector for a turn's first sub-decision is what keeps
    v3 paying v2's encoder bill on the common single-decision turn -- a second full
    encode costs a `projected_supply_map`. It is only sound while `pending.view` is
    the identity, and `PendingTurn.shed` is the permissive
    `max(raw, _post_field_storage)` view, so a turn whose field plan DROPs carried
    produce breaks that before any order has been applied.

    Both halves are asserted. Where the gate opens the two encodes must agree
    *exactly* -- not within a tolerance, since one is a copy of the other. Where it
    closes they must genuinely differ, or the gate would be free to be wrong in the
    quiet direction and nothing would notice.
    """

    idle, idle_fresh, idle_reusable = _encode_case(
        {"farmer": ["PASS"], "hands": []}, {"WHEAT": 3}, {"WHEAT": 12}
    )
    assert idle_reusable
    assert_array_equal(idle, idle_fresh)

    dropped, dropped_fresh, dropped_reusable = _encode_case(
        {"farmer": ["DROP"], "hands": []}, {"WHEAT": 3}, {"WHEAT": 12}
    )
    assert not dropped_reusable, (
        "a DROP turn moves the shed into `pending.view` before the first decision"
    )
    assert not np.array_equal(dropped, dropped_fresh), (
        "the reused vector must be refused precisely because it differs"
    )


def test_the_reuse_gate_names_every_column_that_moves():
    """What the gate is protecting, stated as a count rather than a claim.

    The divergence is not confined to block E -- if it were, the block-E overwrite
    would already cover it. It reaches the per-product shed column, the shed-load
    pair, and the block-C constraint columns downstream of `shed_total`, none of
    which the overwrite touches.
    """

    dropped, dropped_fresh, _ = _encode_case(
        {"farmer": ["DROP"], "hands": []}, {"WHEAT": 3}, {"WHEAT": 12}
    )
    moved = np.flatnonzero(dropped != dropped_fresh)
    assert moved.size, "the DROP case must actually move columns"
    outside = [index for index in moved if not INTRA_TURN_SLICE.start <= index < INTRA_TURN_SLICE.stop]
    assert outside, (
        "the whole point is that columns outside block E move; the overwrite "
        "already handles block E"
    )


def test_cloned_state_matches_the_live_agent():
    """`clone.py` reproduces `main.py`'s walk, vector for vector and tick for tick.

    The three properties `main.py`'s tick comment claims, asserted rather than
    described: the tick lands after `history.observe` so the turn's money has
    settled last turn's order, before the decision loop so every sub-decision of a
    turn scores against one latent, and on the untouched observation with block E
    carrying the *previous* turn's finished `PendingTurn`.

    Compared on the whole vector and the whole hidden trajectory, not one block.
    A recurrence that drifts does not degrade a feature -- it poisons every state
    after the drift, and the two walks are written out by hand in two files that
    build `PendingTurn` in a different order relative to the encode.
    """

    from kaggle_environments import make

    from agents.dqn_market_free_v3c.main import MarketFreeAgent
    from agents.dqn_market_free_v3c.training.clone import record_from_replay
    from agents.dqn_market_free_v3c.training.collect import HIDDEN_STRIDE, Recorder
    from tests.test_dqn_market_free import SNAPSHOT_PATH

    recorder = Recorder()
    # A random head rather than `quiet_start`: this needs the agent to trade, so
    # that block E carries something on the recurrent input and the two walks are
    # actually being asked to agree about a moving state.
    network = OrderQNetwork.random(seed=3)
    learner = MarketFreeAgent(
        network, rng=np.random.default_rng(0), recorder=recorder
    )
    env = make("kaggriculture", debug=True)
    env.run([learner, str(SNAPSHOT_PATH)])
    replay = env.toJSON()

    live = recorder.finish(
        final_money=float(env.state[0]["observation"]["farms"][0]["money"]),
        gamma=0.999,
    )
    cloned, _stats = record_from_replay(replay, seat=0, gru=network.gru)
    assert cloned is not None

    # The recurrent input: one row per turn, encoded off the untouched observation
    # with the previous turn's `PendingTurn` in block E.
    assert cloned.turn_state.shape == live.turn_state.shape
    np.testing.assert_allclose(cloned.turn_state, live.turn_state, atol=1e-6)

    # The hidden trajectory the chunks start from. Stored strided, pre-tick, and
    # row 0 is the zero state the episode opened on.
    assert cloned.turn_hidden.shape == live.turn_hidden.shape
    assert not np.any(live.turn_hidden[0]), "row 0 is the state the episode opened on"
    np.testing.assert_allclose(
        cloned.turn_hidden.astype(np.float32),
        live.turn_hidden.astype(np.float32),
        atol=1e-3,
    )
    assert np.abs(live.turn_hidden[1:]).sum() > 0.0, "the recurrence has to move"
    assert live.turn_hidden.shape[0] == -(-live.turn_state.shape[0] // HIDDEN_STRIDE)

    # And the per-decision states, which is where the encode-reuse gate lives:
    # `clone.py` takes `pending.view` for every sub-decision, so any turn where
    # `main.py` reused the turn-opening vector has to have been a turn where the
    # two are the same thing.
    assert cloned.order_state.shape == live.order_state.shape
    np.testing.assert_allclose(cloned.order_state, live.order_state, atol=1e-6)


def test_block_e_lands_where_the_slice_says_it_does():
    """The slice against a *built* vector, not against its own definition.

    `test_state_width_names_its_own_arithmetic` restates `INTRA_TURN_SLICE`'s
    definition and so cannot fail for the reason that matters: block E is appended
    last, and `main.py` overwrites exactly that span when it reuses the
    turn-opening encode. Append anything after it and the overwrite starts writing
    into someone else's columns while every width assertion still passes.
    """

    from agents.dqn_market_free_v3c.market_state import (
        intra_turn_block,
        market_state_features,
    )

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "tiles": tiles,
        "hands": [],
        "farmer": [4, 4],
    }
    private = {"shed": {"WHEAT": 5}, "seeds": {}, "inventories": [{}]}
    obs = {
        "player": 0,
        "step": 30,
        "day": 1,
        "hour": 6,
        "farms": [farm, farm],
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
        "private": private,
    }
    field = {"farmer": ["PASS"], "hands": []}
    roles = herd.field_roles(obs, farm, private)
    pending = PendingTurn(obs, farm, private, field, None, plantable={}, roles=roles)
    pending.apply(Order("SELL", "WHEAT", 2))

    built = market_state_features(obs, farm, private, roles, field, turn=pending)
    block = np.asarray(intra_turn_block(pending), dtype=np.float32)
    assert_array_equal(built[INTRA_TURN_SLICE], block)
    assert np.abs(block).sum() > 0.0, "a committed sale has to show in block E"


def test_the_agent_resets_its_recurrence_between_episodes():
    """`step == 0` clears the memory, so a reused agent object cannot carry it over.

    `main.agent` keeps a module-level `_AGENT` that survives across episodes inside
    one interpreter, so the constructor's reset is not enough on its own. A network
    that resumed the previous episode's hidden state would read on every metric
    exactly like a policy that had learned to ignore its memory, and there is no
    way to separate those after the fact.

    Asserted as an equality against a network that never played the first episode,
    which is the only form that can fail: "the state changed" is true whether or
    not the reset ran.
    """

    from kaggle_environments import make

    from agents.dqn_market_free_v3c.main import MarketFreeAgent
    from tests.test_dqn_market_free import SNAPSHOT_PATH

    def opening_hidden(warm: bool) -> np.ndarray:
        network = OrderQNetwork.random(seed=5)
        assert not network.is_noisy, "a noisy network would advance `rng` across episodes"
        learner = MarketFreeAgent(network, rng=np.random.default_rng(0))
        if warm:
            # A full episode first, so the recurrence is carrying 720 turns of
            # memory when the next episode's opening turn arrives.
            env = make("kaggriculture", debug=True)
            env.run([learner, str(SNAPSHOT_PATH)])
            assert np.abs(network.hidden).sum() > 0.0, "the recurrence has to have moved"
            assert learner.last_turn is not None
        env = make("kaggriculture", debug=True)
        state = env.reset()
        learner(state[0]["observation"], env.configuration)
        return network.hidden.copy()

    assert_array_equal(opening_hidden(warm=True), opening_hidden(warm=False))

def test_the_burn_in_probe_compares_two_burn_ins_of_the_same_window():
    """`q_discrepancy` reads downward, and it has to.

    A chunk holds exactly `burn_in` turns of history, so a *longer* refresh cannot
    be built from one. An earlier version clamped `4 * burn_in` to the chunk width
    and so rolled straight into the graded window, reporting how far Q moves across
    `n_step` turns of play -- large for a healthy time-sensitive GRU and constant
    in `burn_in_turns`, the dial it was documented to steer.

    Pinned two ways: the probe must not touch turns past the burn-in boundary, and
    at a burn-in the recurrence has fully forgotten, the two legs must agree.
    """

    from agents.dqn_market_free_v3c.training.updater import _recurrence_metrics

    module = _module(seed=7)
    buffer = _buffer()
    metrics = _recurrence_metrics(module, buffer, torch.device("cpu"), 20, 40)

    assert set(metrics) >= {
        "hidden_recovery_l2",
        "hidden_recovery_cosine",
        "q_discrepancy",
        "hidden_norm",
    }
    assert np.isfinite(metrics["q_discrepancy"])
    assert metrics["q_discrepancy"] >= 0.0

    # And the discriminating case. A GRU with its reset and update gates pinned
    # shut is memoryless -- `h_t` is a function of `x_t` alone -- so two burn-ins
    # of *different lengths* leave the window's latents bit-identical and the
    # discrepancy is exactly zero. Two latents from different *turns* are not, so
    # this is the case that separates a burn-in comparison from a comparison
    # across the window, which is what the clamp used to compute.
    memoryless = _module(seed=8)
    with torch.no_grad():
        memoryless.cell.weight_hh.zero_()
        memoryless.cell.bias_hh.zero_()
        memoryless.cell.bias_ih[:HIDDEN_SIZE] = -30.0
        memoryless.cell.bias_ih[HIDDEN_SIZE : 2 * HIDDEN_SIZE] = -30.0
    forgetful = _recurrence_metrics(memoryless, buffer, torch.device("cpu"), 20, 40)
    assert forgetful["q_discrepancy"] == pytest.approx(0.0, abs=1e-6)
    # The same network still moves Q across turns, so the zero above is a property
    # of the comparison rather than of a network that scores everything alike.
    assert forgetful["hidden_norm"] > 0.0


def test_the_clone_step_clips_only_what_it_will_step():
    """Sigma is excluded from `clone_optimizer`, so its gradient is never cleared.

    A clip over the whole module would keep counting those stale gradients, the
    total norm would grow across the phase, and the clip factor would shrink the
    head's and the GRU's real gradients a little further every step -- an effective
    learning rate that decays for a reason nothing reports.
    """

    from agents.dqn_market_free_v3c.training import network as net
    from agents.dqn_market_free_v3c.training.updater import _behavior_clone_step

    module = _module(seed=11, noisy="head")
    buffer = _buffer(count=4)
    buffer.add([_episode(seed=90 + index) for index in range(2)], pinned=True)
    optimizer = torch.optim.Adam(
        [
            parameter
            for name, parameter in module.named_parameters()
            if "sigma" not in name
        ],
        lr=1e-3,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }
    for _ in range(3):
        loss = _behavior_clone_step(
            module, optimizer, buffer, 2, torch.device("cpu"), 20, 40, 5.0
        )
        assert np.isfinite(loss)

    moved = {
        name
        for name, parameter in module.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    assert any(name.startswith("cell.") for name in moved), "the GRU must train"
    assert not any("sigma" in name for name in moved), (
        "sigma is excluded from the clone optimizer and must not move"
    )
    assert isinstance(net.NOISY_LAYERS, tuple)


# --------------------------------------------------------------------------
# The wash
# --------------------------------------------------------------------------


def test_a_same_item_round_trip_is_worth_exactly_nothing():
    """The premise of the mask, priced rather than asserted.

    Buying walks the market inventory down and selling it back walks the same
    price levels up, so the pair nets exactly zero at every quantity. That is why
    v2 could spend 47% of its turns on it for free.
    """

    from agents.dqn_market_free_v3c import donor

    start = donor.MARKET_I0
    obs = {"market": {"inventory": {"WHEAT": start}, "prices": {}}}
    for quantity in (8, 24, 32, 45):
        cost = sum(
            float(donor._price_at("WHEAT", start - offset, obs))
            for offset in range(1, quantity + 1)
        )
        # The sell starts from the post-buy inventory, which is the whole point.
        proceeds = sum(
            float(donor._price_at("WHEAT", (start - quantity) + offset, obs))
            for offset in range(quantity)
        )
        assert proceeds - cost == pytest.approx(0.0), quantity


def test_a_turn_that_sold_an_item_cannot_buy_it_back():
    """Shut structurally, because a price cannot be tuned against a free action.

    A penalty has to be calibrated against the behaviour it suppresses and can be
    learned around at the margin. v1 could not wash at all -- one order per turn --
    and v1 is the lineage this fork has to beat.
    """

    from agents.dqn_market_free_v3c.order_space import (
        _BUY_PRODUCT_ROWS,
        _SELL_ROWS,
        Order,
        PendingTurn,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0, shed={"WHEAT": 60})
    pending = PendingTurn(obs, farm, private, field, None)

    opening = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert opening[_SELL_ROWS["WHEAT"]].any(), "the turn must be able to sell at all"
    assert opening[_BUY_PRODUCT_ROWS["WHEAT"]].any(), "and to buy at all"

    pending.apply(Order("SELL", "WHEAT", 6))
    after = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert not after[_BUY_PRODUCT_ROWS["WHEAT"]].any(), "buying back is the wash"
    # The other item is untouched: this shuts a round trip, not the market.
    assert after[_BUY_PRODUCT_ROWS["FERTILIZER"]].any()


def test_a_turn_that_bought_an_item_cannot_sell_it_back():
    from agents.dqn_market_free_v3c.order_space import (
        _SELL_ROWS,
        Order,
        PendingTurn,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0, shed={"WHEAT": 60})
    pending = PendingTurn(obs, farm, private, field, None)
    pending.apply(Order("BUY_PRODUCT", "WHEAT", 4))
    after = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert not after[_SELL_ROWS["WHEAT"]].any()


def test_a_mixed_turn_is_still_legal():
    """Selling one item and buying another is ordinary trading, not a wash."""

    from agents.dqn_market_free_v3c.order_space import (
        _BUY_PRODUCT_ROWS,
        _SELL_ROWS,
        Order,
        PendingTurn,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(
        money=5000.0, shed={"WHEAT": 60, "MELON": 20}
    )
    pending = PendingTurn(obs, farm, private, field, None)
    pending.apply(Order("SELL", "MELON", 3))
    after = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert after[_BUY_PRODUCT_ROWS["WHEAT"]].any(), "a different item stays buyable"
    assert after[_SELL_ROWS["WHEAT"]].any(), "and a different item stays sellable"


def test_the_unpended_mask_is_unaffected_by_the_wash_rule():
    """No `PendingTurn` means no turn in flight, so nothing has been queued."""

    from agents.dqn_market_free_v3c.order_space import _BUY_PRODUCT_ROWS, feasible_actions
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0, shed={"WHEAT": 60})
    mask = feasible_actions(obs, farm, private, field, None)
    assert mask[_BUY_PRODUCT_ROWS["WHEAT"]].any()


def test_a_zero_quote_turn_is_charged_again_when_nothing_moved():
    """v2 waved this through and it was one of the three things that made the
    wash reward-optimal. With same-item round trips unreachable, a zero quote and
    a zero delta is a refusal again -- v1's reading."""

    from agents.dqn_market_free_v3c.market_state import order_refused

    assert bool(order_refused(True, 0.0, 0.0)), "refused: nothing quoted, nothing moved"
    assert not bool(order_refused(True, 42.0, 0.0)), "filled: the wallet moved"
    assert bool(order_refused(True, 0.0, 500.0)), "refused: quoted 500, moved nothing"
    assert not bool(order_refused(True, 500.0, 500.0))
    assert not bool(order_refused(False, 0.0, 0.0)), "nothing emitted, nothing to refuse"


def test_a_buy_animal_rung_the_wallet_cannot_cover_is_shut():
    """The last place a single affordable unit opened the whole ladder."""

    from agents.dqn_market_free_v3c import donor
    from agents.dqn_market_free_v3c.order_space import (
        _BUY_ANIMAL_ROWS,
        _ROW_QUANTITY,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    animal = "COW"
    unit = float(donor.ANIMALS[animal]["cost"])
    # Enough for one head and not two.
    obs, farm, private, field = _mask_state(money=unit * 1.5)
    mask = feasible_actions(obs, farm, private, field, None)
    rows = _BUY_ANIMAL_ROWS[animal]
    for index, quantity in enumerate(_ROW_QUANTITY[rows]):
        affordable = int(quantity) * unit <= unit * 1.5
        assert bool(mask[rows][index]) == bool(affordable), quantity


def _lot_batch(**metadata):
    """A stand-in for `TrainingBatch` carrying only what the aggregator reads."""

    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class _Batch:
        metadata: dict[str, Any] = field(default_factory=dict)

    return _Batch(metadata=dict(metadata))


def test_batch_lot_metrics_average_the_batches_the_update_consumed():
    """`reward.py` names these columns as its instrument; the update now carries them."""

    from agents.dqn_market_free_v3c.training.updater import _batch_lot_metrics

    metrics = _batch_lot_metrics(
        (
            _lot_batch(animal_waste_cost=1000.0, animal_gain_credit=100.0),
            _lot_batch(animal_waste_cost=3000.0, animal_gain_credit=300.0),
        )
    )
    assert metrics["animal_waste_cost"] == pytest.approx(2000.0)
    assert metrics["animal_gain_credit"] == pytest.approx(200.0)


def test_batch_lot_metrics_skip_batches_that_carry_no_lot_columns():
    """An evaluation-sourced batch has no rollouts behind it.

    Folding it in as a zero would report the herd term decaying every time one
    landed, which is the failure this skip exists to prevent.
    """

    from agents.dqn_market_free_v3c.training.updater import _batch_lot_metrics

    metrics = _batch_lot_metrics(
        (
            _lot_batch(animal_waste_cost=1000.0),
            _lot_batch(transitions=1234),  # evaluation batch: no lot columns
            _lot_batch(),
            object(),  # no `metadata` attribute at all
        )
    )
    assert metrics["animal_waste_cost"] == pytest.approx(1000.0)


def test_batch_lot_metrics_omit_a_column_no_batch_measured():
    """Absent is not zero. 'Not measured' and 'measured as nothing' differ."""

    from agents.dqn_market_free_v3c.training.updater import _batch_lot_metrics

    metrics = _batch_lot_metrics((_lot_batch(animal_waste_cost=500.0),))
    assert "animal_waste_cost" in metrics
    assert "seed_waste_cost" not in metrics
    assert _batch_lot_metrics(()) == {}


def test_batch_lot_metrics_reject_values_that_are_not_finite_numbers():
    """A None or a NaN must not poison the mean for the whole update."""

    from agents.dqn_market_free_v3c.training.updater import _batch_lot_metrics

    metrics = _batch_lot_metrics(
        (
            _lot_batch(animal_waste_cost=1000.0, unrecovered_animal_share=None),
            _lot_batch(animal_waste_cost=float("nan"), unrecovered_animal_share="0.5"),
            _lot_batch(animal_waste_cost=2000.0, unrecovered_animal_share=float("inf")),
        )
    )
    assert metrics["animal_waste_cost"] == pytest.approx(1500.0)
    # "0.5" is a JSON-legal number that arrived as a string; it still counts.
    assert metrics["unrecovered_animal_share"] == pytest.approx(0.5)


def test_batch_lot_keys_are_exactly_what_the_collector_writes():
    """Every lifted key has to exist in batch metadata or it silently never appears.

    The aggregator omits a column no batch carried, so a typo here is invisible at
    runtime -- the metric would simply never show up. This pins the names against
    the `BehaviorCounts` export the collector builds its metadata from.
    """

    from agents.dqn_market_free_v3c.diagnostics import BehaviorCounts
    from agents.dqn_market_free_v3c.training.updater import _BATCH_LOT_KEYS

    exported = BehaviorCounts().as_dict()
    # The collector writes the `_per_episode` variants under the bare name.
    available = set(exported) | {
        key[: -len("_per_episode")] for key in exported if key.endswith("_per_episode")
    }
    missing = sorted(key for key in _BATCH_LOT_KEYS if key not in available)
    assert not missing, f"lifted keys the collector never writes: {missing}"


# --------------------------------------------------------------------------
# Block D': the restored refusal latches
# --------------------------------------------------------------------------


def _refusal_state(history):
    """A built vector for a fixed board, so only the latches vary."""

    from agents.dqn_market_free_v3c.market_state import market_state_features

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "tiles": tiles,
        "hands": [],
        "farmer": [4, 4],
    }
    private = {"shed": {"WHEAT": 5}, "seeds": {}, "inventories": [{}]}
    obs = {
        "player": 0,
        "step": 30,
        "day": 1,
        "hour": 6,
        "farms": [farm, farm],
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
        "private": private,
    }
    field = {"farmer": ["PASS"], "hands": []}
    roles = herd.field_roles(obs, farm, private)
    return market_state_features(obs, farm, private, roles, field, history=history)


def test_the_refusal_block_lands_between_the_herd_block_and_block_e():
    """Three columns, and they must not disturb block E's span.

    `main.py` overwrites exactly `INTRA_TURN_SLICE` when it reuses the
    turn-opening encode, so anything appended after block E would be silently
    rewritten. This asserts the restored block sits *before* it and that block E
    is still the tail.
    """

    from agents.dqn_market_free_v3c.market_state import MarketHistory, refusal_block

    assert INTRA_TURN_SLICE.stop == MARKET_STATE_DIM
    assert INTRA_TURN_SLICE.start == MARKET_STATE_DIM - INTRA_TURN_DIM

    history = MarketHistory()
    state = _refusal_state(history)
    assert state.shape == (MARKET_STATE_DIM,)
    # Block F sits between the two now, so the latches are that much further back.
    flow_start = INTRA_TURN_SLICE.start - MARKET_FLOW_DIM
    block = slice(flow_start - REFUSAL_BLOCK_DIM, flow_start)
    assert_array_equal(
        state[block], np.asarray(refusal_block(history), dtype=np.float32)
    )


def test_a_refusal_run_moves_the_block_and_a_fill_clears_it():
    """The latches are integrators, which is the whole argument for the columns.

    A GRU can be asked to learn a 24-turn price difference; asking it to hold a
    running count of consecutive refusals across 720 ticks is the case
    recurrences are worst at, and `training/reward.py` charges
    `REJECTED_ORDER_PENALTY` against exactly this condition.
    """

    from agents.dqn_market_free_v3c.market_state import MarketHistory

    farm = {"money": 1000.0}
    obs = {"player": 0, "farms": [farm, farm], "market": {"prices": {}, "inventory": {}}}
    history = MarketHistory()
    history.observe(obs, farm)

    idle = _refusal_state(history)
    flow_start = INTRA_TURN_SLICE.start - MARKET_FLOW_DIM
    block = slice(flow_start - REFUSAL_BLOCK_DIM, flow_start)
    assert_array_equal(idle[block], np.zeros(REFUSAL_BLOCK_DIM, dtype=np.float32))

    # Three turns that each emit an order and move no money: refused, by the same
    # `order_refused` rule the reward charges on.
    for _ in range(3):
        history.record_decision(Order("SELL", "WHEAT", 1))
        history.observe(obs, farm)
    assert history.consecutive_refusals == 3
    refused = _refusal_state(history)
    assert refused[block][0] == 0.0, "nothing filled"
    assert refused[block][1] > 0.0, "turns_since_fill has run up"
    assert refused[block][2] == pytest.approx(3 / 8)

    # A turn whose money moves is a fill, and it resets both counters.
    history.record_decision(Order("SELL", "WHEAT", 1))
    history.observe(obs, {"money": 1400.0})
    assert history.consecutive_refusals == 0
    filled = _refusal_state(history)
    assert filled[block][0] == 1.0
    assert_array_equal(filled[block][1:], np.zeros(2, dtype=np.float32))


def test_a_v3b_checkpoint_cannot_load_into_v3c():
    """232 against 280. The fork is a cold start and `load` has to say so."""

    from agents.dqn_market_free_v3c.policy import STATE_DIM

    assert STATE_DIM == MARKET_STATE_DIM == 280
    import agents.dqn_market_free_v3b.market_state as v3b_state

    assert v3b_state.MARKET_STATE_DIM == 232, "the arm being compared against"


def _market_obs(step: int, inventory: dict, shops=()):
    """The same fixed board as `_refusal_state`, with a market that moves."""

    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000.0,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
        "tiles": tiles,
        "hands": [],
        "farmer": [4, 4],
    }
    private = {"shed": {"WHEAT": 5}, "seeds": {}, "inventories": [{}]}
    stock = {item: float(donor.MARKET_I0) for item in PRODUCTS}
    stock.update(inventory)
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [farm, farm],
        "market": {"prices": {}, "inventory": stock},
        "town": {"unlocked_shops": list(shops)},
        "private": private,
    }, farm


def test_the_flow_block_lands_between_the_refusal_block_and_block_e():
    """Block F must not disturb block E's span.

    `main.py` overwrites exactly `INTRA_TURN_SLICE` when it reuses the turn-opening
    encode, so a block appended after E would be silently rewritten.
    """

    from agents.dqn_market_free_v3c.market_state import MarketHistory, market_flow_block

    history = MarketHistory()
    state = _refusal_state(history)
    obs, farm = _market_obs(30, {})
    history.observe(obs, farm)
    block = slice(INTRA_TURN_SLICE.start - MARKET_FLOW_DIM, INTRA_TURN_SLICE.start)
    state = _refusal_state(history)
    assert state.shape == (MARKET_STATE_DIM,)
    bases, curves = _curves_at(obs, {item: float(donor.MARKET_I0) for item in PRODUCTS})
    assert_array_equal(
        state[block],
        np.asarray(market_flow_block(history, obs, bases, curves), dtype=np.float32),
    )


def test_the_flow_column_is_what_the_agent_did_not_do():
    """The whole premise: inventory moved, minus the town, minus us, is them.

    Two turns four apart so the shop tick lands, with one shop buying wheat. The
    market ends 6 wheat heavier; the town took 12 of them; this agent queued 4. The
    remaining 14 are the opponent's, and no part of the observation says so.
    """

    from agents.dqn_market_free_v3c.market_state import MarketHistory
    from agents.dqn_market_free_v3c.order_space import Order

    wheat = PRODUCTS.index("WHEAT")
    history = MarketHistory()
    obs, farm = _market_obs(4, {}, shops=("YARN_STORE", "BAKERY"))
    history.observe(obs, farm)
    history.record_decision(Order("SELL", "WHEAT", 4))

    # BAKERY buys wheat once every four turns; step 4 is a tick, so 1 unit of wheat
    # leaves for the shop. The town centre only fires on multiples of 24.
    later, farm = _market_obs(5, {"WHEAT": donor.MARKET_I0 + 17.0}, shops=("YARN_STORE", "BAKERY"))
    history.observe(later, farm)

    assert history._flow[-1]["WHEAT"] == 18.0, "17 on the books plus the 1 the shop took"
    assert history.last_emitted["WHEAT"] == 4.0
    # 18 - 4 = 14 units this agent did not sell, which is the signal the block exists
    # to carry. The two halves are separate columns because the engine never reports
    # a fill, so the subtraction is the network's to make.
    assert history._flow[-1]["WHEAT"] - history.last_emitted["WHEAT"] == 14.0
    assert history._flow[-1]["MELON"] == 0.0, "a product nobody touched does not move"
    del wheat


def test_the_flow_block_does_not_vary_within_a_turn():
    """Own emitted is read off the history, never off the `PendingTurn` in hand.

    The turn encode passes the *previous* turn and each sub-decision passes the
    *current* one. If block F read `turn` it would disagree between the two, and
    `main.py`'s reuse of the turn-opening encode -- which rewrites block E's span and
    nothing else -- would silently ship a stale column.
    """

    from agents.dqn_market_free_v3c.market_state import (
        MarketHistory,
        market_state_features,
    )
    from agents.dqn_market_free_v3c.order_space import Order, PendingTurn

    history = MarketHistory()
    obs, farm = _market_obs(8, {"WHEAT": donor.MARKET_I0 + 3.0})
    private = obs["private"]
    history.observe(obs, farm)
    history.record_decision(Order("SELL", "WHEAT", 2))
    later, farm = _market_obs(9, {"WHEAT": donor.MARKET_I0 + 9.0})
    private = later["private"]
    history.observe(later, farm)

    roles = herd.field_roles(later, farm, private)
    field = {"farmer": ["PASS"], "hands": []}
    pending = PendingTurn(later, farm, private, field, None)
    block = slice(INTRA_TURN_SLICE.start - MARKET_FLOW_DIM, INTRA_TURN_SLICE.start)
    opening = market_state_features(
        later, farm, private, roles, field, history=history, turn=None
    )
    decision = market_state_features(
        later, farm, private, roles, field, history=history, turn=pending
    )
    assert_array_equal(opening[block], decision[block])
    assert not np.array_equal(opening[INTRA_TURN_SLICE], decision[INTRA_TURN_SLICE])


def test_headroom_counts_units_to_the_band_and_a_glut_closes_it():
    """F3 is the impact curve inverted, which is the question a sell order asks.

    WOOL is an `above_func = "sq"` product: at equilibrium it takes tens of units to
    push the quote through 0.9 of base, and once the market is already 60 deep the
    quote has cratered and there is no room left at all.
    """

    from agents.dqn_market_free_v3c.market_state import (
        HEADROOM_BANDS,
        MarketHistory,
        market_flow_block,
    )

    history = MarketHistory()
    obs, _ = _market_obs(12, {})
    quiet = {item: float(donor.MARKET_I0) for item in PRODUCTS}
    glutted = dict(quiet, WOOL=float(donor.MARKET_I0) + 60.0)
    wool = PRODUCTS.index("WOOL")
    head = 3 * len(PRODUCTS) + 3

    assert HEADROOM_BANDS == (0.9, 0.75)
    loose = head + len(PRODUCTS)
    tight = market_flow_block(history, obs, *_curves_at(obs, quiet))
    after = market_flow_block(history, obs, *_curves_at(obs, glutted))

    # 19 units before wool's quote falls through 0.9 of base, 30 before 0.75. The
    # two bands are a slope, not a duplicate column.
    assert round(tight[head + wool] * 45) == 19
    assert round(tight[loose + wool] * 45) == 30

    # 60 units in and wool is already under both bands, so there is no room left at
    # all -- which is the state a policy needs to see *before* it queues the sale,
    # and the exact cliff that costs this family money.
    assert after[head + wool] == 0.0
    assert after[loose + wool] == 0.0

    # The glut is one product's, and the columns say so.
    melon = PRODUCTS.index("MELON")
    assert after[head + melon] == tight[head + melon] == 1.0


def _curves_at(obs, stock):
    """`market_flow_block`'s two price arguments, as `market_state_features` builds them."""

    from agents.dqn_market_free_v3c.market_state import _FILL_OFFSETS, price_curve

    bases = {item: float(donor._market_parameters(obs, item)[0]) for item in PRODUCTS}
    curves = {item: price_curve(item, stock[item] + _FILL_OFFSETS, obs) for item in PRODUCTS}
    return bases, curves


def test_a_dump_cannot_push_the_flow_column_out_of_distribution():
    """Every other column in this vector sits in [0, 2]; F1 has to be bounded too.

    The failure this pre-empts is `SEED_FEATURE_CAP`'s, which was found the other way
    round -- after an unbounded column had already degraded the Q estimate for the
    states that produced it.
    """

    from agents.dqn_market_free_v3c.market_state import (
        FLOW_COLUMN_CLIP,
        MarketHistory,
        market_flow_block,
    )
    from agents.dqn_market_free_v3c.order_space import Order

    history = MarketHistory()
    obs, farm = _market_obs(6, {})
    history.observe(obs, farm)
    # Both seats empty a full shed into the same product on one turn.
    for _ in range(4):
        history.record_decision(Order("SELL", "MELON", 45))
    dumped, farm = _market_obs(7, {"MELON": donor.MARKET_I0 + 400.0})
    history.observe(dumped, farm)

    melon = PRODUCTS.index("MELON")
    stock = {item: float(donor.MARKET_I0) for item in PRODUCTS}
    stock["MELON"] += 400.0
    block = market_flow_block(history, dumped, *_curves_at(dumped, stock))
    assert history._flow[-1]["MELON"] == 400.0, "the raw quantity is not clipped"
    assert history.last_emitted["MELON"] == 180.0
    assert block[melon] == FLOW_COLUMN_CLIP
    assert block[len(PRODUCTS) + melon] == FLOW_COLUMN_CLIP
    assert block[2 * len(PRODUCTS) + melon] == FLOW_COLUMN_CLIP
    assert max(abs(value) for value in block) == FLOW_COLUMN_CLIP


def test_a_lot_resolves_on_its_last_sale_not_its_first():
    """`resolution_turn` is where the split credit's second half lands.

    A lot sold across several turns is only settled by the last of them, and a lot
    that never sold stayed sellable until the buzzer -- so it resolves at the end.
    """

    from agents.dqn_market_free_v3c.lots import _Lot, resolution_turn

    sold = _Lot(item="WHEAT", buy_turn=5, units=3, unit_cost=10.0, capacity=6)
    sold.first_sale_turn = 40
    sold.last_sale_turn = 96
    assert resolution_turn(sold, 719) == 96

    never = _Lot(item="MELON", buy_turn=5, units=1, unit_cost=80.0, capacity=6)
    assert never.last_sale_turn is None
    assert resolution_turn(never, 719) == 719


def test_split_credit_moves_half_the_charge_without_changing_its_size():
    """The pair stays bounded by the lot's own cost; only the indices change.

    At `LOT_CREDIT_BUY_SHARE = 1.0` the walk has to reproduce the old single-index
    behaviour exactly, which is what makes the constant a dial rather than a rewrite.
    """

    from agents.dqn_market_free_v3c.lots import _Lot, _split, resolution_turn

    wasted = _Lot(item="MELON", buy_turn=12, units=1, unit_cost=80.0, capacity=6)
    paid = _Lot(item="WHEAT", buy_turn=20, units=2, unit_cost=10.0, capacity=12)
    paid.proceeds = 50.0
    paid.first_sale_turn = 30
    paid.last_sale_turn = 64

    entries = [(lot.buy_turn, max(0.0, lot.cost - lot.proceeds)) for lot in (wasted, paid)]
    split = _split(entries, (wasted, paid), 719)
    assert split[0] == (12, 719, 80.0), "a never-sold lot resolves at the buzzer"
    assert split[1] == (20, 64, 0.0), "a lot paid back in full owes nothing"

    # The coins are untouched by the split -- the whole change is which index pays.
    assert [coins for _, _, coins in split] == [coins for _, coins in entries]
    for buy_turn, resolved, _ in split:
        assert resolved >= buy_turn, "a lot cannot resolve before it was bought"


# ---------------------------------------------------------------------------
# The Kaggle ladder mirror as demonstrations
# ---------------------------------------------------------------------------


def _ladder_replay(steps: int = 96):
    """A short episode in the ladder replay's own shape.

    `env.toJSON()` is the same `kaggle-environments` document the Kaggle episode
    mirror stores, which is the whole reason `record_from_replay` reads both
    without knowing which it got. Generated rather than committed as a fixture:
    a real mirrored replay is ~30 MB and `data/` is gitignored.
    """

    from kaggle_environments import make

    from agents.dqn_market_free_v3c.main import MarketFreeAgent
    from tests.test_dqn_market_free import SNAPSHOT_PATH

    network = OrderQNetwork.random(seed=5)
    learner = MarketFreeAgent(network, rng=np.random.default_rng(0))
    env = make("kaggriculture", configuration={"episodeSteps": steps}, debug=True)
    env.run([learner, str(SNAPSHOT_PATH)])
    return env.toJSON(), network


def test_rehidden_reproduces_the_walk_it_replaced():
    """The invariant the demonstration cache rests on.

    `training/ladder.py` converts a replay once and re-rolls the hidden states per
    update, because the policy that has to have produced them moves every update
    and inflating 30 MB of JSON to redo the walk does not. That is only sound if
    rolling the GRU over the *stored* `turn_state` lands exactly where the walk
    did -- same zero start, same pre-tick store, same stride, same fp16. If it
    drifts, every mid-episode chunk resumes from a state the policy never
    occupied, and nothing about the record's widths or stride would say so.
    """

    from agents.dqn_market_free_v3c.training.clone import record_from_replay, rehidden

    replay, network = _ladder_replay()
    record, _stats = record_from_replay(replay, seat=0, gru=network.gru)
    assert record is not None
    assert np.abs(record.turn_hidden[1:].astype(np.float32)).sum() > 0.0

    # Under the same GRU: bit-identical, because it is the same arithmetic.
    same = rehidden(record, network.gru)
    np.testing.assert_array_equal(same.turn_hidden, record.turn_hidden)
    assert int(same.hidden_stride[0]) == int(record.hidden_stride[0])

    # And under a different one it actually moves, so the assertion above is
    # testing the roll rather than a column that was copied through.
    other = rehidden(record, OrderQNetwork.random(seed=11).gru)
    assert other.turn_hidden.shape == record.turn_hidden.shape
    assert not np.array_equal(other.turn_hidden, record.turn_hidden)
    assert not np.any(other.turn_hidden[0]), "row 0 is the state the episode opened on"


def test_rehidden_refuses_a_pack_from_another_schema():
    from agents.dqn_market_free_v3c.training.clone import record_from_replay, rehidden

    replay, network = _ladder_replay()
    record, _stats = record_from_replay(replay, seat=0, gru=network.gru)
    record.turn_state = record.turn_state[:, :-1]
    with pytest.raises(ValueError, match="another schema"):
        rehidden(record, network.gru)


def test_ladder_pack_round_trips_and_reports_only_what_it_loaded(tmp_path):
    """A pack is a prefix, and its metrics have to describe the prefix.

    Shards are written in descending ladder score and `demonstration_episodes`
    reads the head, so a pack built at 320 and loaded at 96 must not report the
    320-episode totals -- a `demonstrations.episodes` that disagreed with the
    number of pinned records is a metrics row nobody can reconcile.
    """

    from agents.dqn_market_free_v3c.training import ladder
    from agents.dqn_market_free_v3c.training.clone import record_from_replay

    replay, network = _ladder_replay()
    records = []
    totals = ladder.CloneStats()
    provenance = []
    for seat in (0, 1):
        record, stats = record_from_replay(replay, seat, 0.999, gru=network.gru)
        assert record is not None
        records.append(record)
        totals = ladder._merge(totals, stats)
        provenance.append({"seat": seat, "stats": ladder._as_dict(stats)})
    ladder.save_episode_batch(records[:1], tmp_path / "part-0000.npz")
    ladder.save_episode_batch(records[1:], tmp_path / "part-0001.npz")
    (tmp_path / ladder.SIDECAR).write_text(
        json.dumps({"stats": ladder._as_dict(totals), "provenance": provenance})
    )

    loaded, stats = ladder.load_ladder_demonstrations(tmp_path, gru=network.gru, limit=1)
    assert len(loaded) == 1
    assert stats.episodes == 1
    assert stats.decisions == provenance[0]["stats"]["decisions"]

    whole, stats = ladder.load_ladder_demonstrations(tmp_path, gru=network.gru, limit=8)
    assert len(whole) == 2
    assert stats.episodes == 2
    assert stats.decisions == totals.decisions
    # Re-rolled on the way out rather than trusted from disk.
    np.testing.assert_array_equal(whole[0].turn_hidden, records[0].turn_hidden)


def test_ladder_seat_finals_read_the_observation_player():
    """Through `observation["player"]`, never the seat index.

    They agree in every replay seen so far. Disagreeing would rank a seat by the
    other seat's wallet, which silently selects the demonstrations backwards.
    """

    from agents.dqn_market_free_v3c.training import ladder

    replay = {
        "steps": [
            [
                {"observation": {"player": 1, "farms": [{"money": 10}, {"money": 900}]}},
                {"observation": {"player": 0, "farms": [{"money": 10}, {"money": 900}]}},
            ]
        ]
    }
    assert ladder.seat_finals(replay) == [900.0, 10.0]


def test_ladder_mirror_walk_is_ranked_and_skips_absent_replays(tmp_path):
    """The manifests list every episode a day published; the mirror holds a slice."""

    from agents.dqn_market_free_v3c.training import ladder

    for date, rows in (
        ("2026-08-01", [("1", 3000.0), ("2", 2000.0)]),
        ("2026-08-02", [("3", 2500.0)]),
    ):
        day = tmp_path / date
        day.mkdir()
        lines = ["episode_id,avg_score"] + [f"{episode},{score}" for episode, score in rows]
        (day / "manifest.csv").write_text("\n".join(lines) + "\n")
        for episode, _score in rows:
            # Episode "2" is listed and not mirrored, which is the ordinary case.
            if episode != "2":
                (day / f"{episode}.json.gz").write_bytes(b"")

    walked = list(ladder.iter_mirror_episodes(tmp_path))
    assert [item.episode_id for item in walked] == ["1", "3"]
    assert [item.episode_id for item in ladder.iter_mirror_episodes(tmp_path, days=1)] == ["3"]


# --------------------------------------------------------------------------
# The outcome term's scale -- see `reward.OUTCOME_LOG_SCALE`.
#
# Every reward-constant test in this repo lives in `test_dqn_market_free.py` and
# imports the **v1** package, so until these landed nothing guarded this fork's
# copy. What they pin is the argument the constant moved on, not the number: the
# return has to track money, and it has to keep slope where the policy is trying
# to get to.
# --------------------------------------------------------------------------


def _flat_season(final_money: float, turns: int = 720, stake: float = 3_000.0):
    """A money trace that holds the stake and lands on `final_money`."""

    return [stake] * turns, final_money


def _lot_traces(turns: int, seed_counts=None, stock_counts=None, held_from: int = 0):
    """Lot traces for a season, sized as `episode_rewards` expects them.

    `held_from` is the turn the holding appears on. It defaults to 0 -- a farm that
    owned the stock all along -- which is what a flat comparison wants, but any test
    that reads the sum needs the purchase inside the season: the stream telescopes
    over the basis, so stock present on turn 0 is stock the season is charged for
    losing without ever being credited for buying.
    """

    from agents.dqn_market_free_v3c import lots

    crops = len(lots.CROP_ITEMS)
    items = len(lots.STOCK_ITEMS)
    seeds, standing, stock = [], [], []
    for turn in range(turns):
        owns = turn >= held_from
        seeds.append(list(seed_counts) if owns and seed_counts else [0] * crops)
        standing.append([0] * crops)
        stock.append(list(stock_counts) if owns and stock_counts else [0] * items)
    return seeds, standing, stock


def test_the_stream_telescopes_to_the_money_the_season_ends_with():
    """The whole objective, and the reason there is no terminal term.

    The league ranks on final money, so the stream has to sum to it. The cash
    delta telescopes to `final - stake` on its own; the held basis is what makes a
    purchase score about nothing on the buy turn, and it is written off at the last
    step because the simulator pays for `farm["money"]` and nothing else. Holding
    stock at the end must not inflate the season.
    """

    from agents.dqn_market_free_v3c.training.reward import MONEY_SCALE, episode_rewards

    money, final = _flat_season(9_000.0)
    seeds, standing, stock = _lot_traces(
        len(money), seed_counts=[40, 0, 0, 0, 0], held_from=10
    )
    shaped, _, _ = episode_rewards(
        money,
        final,
        1.0,
        seeds=seeds,
        standing=standing,
        stock=stock,
    )
    assert float(shaped.sum()) == pytest.approx((final - 3_000.0) / MONEY_SCALE, abs=1e-6)


def test_a_purchase_is_not_a_loss():
    """What the six deleted lot legs were reconstructing, in one line.

    Buying seed is cash out and basis in, in equal coins. Under a bare cash delta
    that turn is charged the full price and buying is locally the worst thing
    available -- the NOOP attractor `SEED_WASTE_WEIGHT` was introduced to patch.
    Under net worth it scores about nothing, and the loss lands where the lot
    actually fails to sell.
    """

    from agents.dqn_market_free_v3c.training import reward as reward_module
    from agents.dqn_market_free_v3c.training.reward import episode_rewards

    # Priced off the same table the basis reads, so the trace cannot drift from it.
    counts = [0] * len(reward_module._SEED_BASIS)
    counts[0] = 10
    spend = sum(n * price for n, price in zip(counts, reward_module._SEED_BASIS))

    turns = 100
    final = 3_000.0 - spend
    money = [3_000.0] * 10 + [final] * (turns - 10)
    seeds, standing, stock = _lot_traces(turns, seed_counts=counts, held_from=10)

    held, _, _ = episode_rewards(
        money, final, 1.0, seeds=seeds, standing=standing, stock=stock
    )
    cash, _, _ = episode_rewards(money, final, 1.0)
    buy = 9
    assert float(cash[buy]) == pytest.approx(
        -spend / reward_module.MONEY_SCALE, abs=1e-9
    ), "the cash delta alone charges the full price"
    assert float(held[buy]) == pytest.approx(0.0, abs=1e-6)
    # The coins are not forgiven, only deferred: the write-off lands at the end.
    assert float(held.sum()) == pytest.approx(float(cash.sum()), abs=1e-6)


def test_an_idle_season_is_worth_only_the_idle_charge():
    """The NOOP guard, restated against a linear objective.

    A season that does nothing moves no coins and opens no lots, so the objective
    scores it at exactly zero -- neither better nor worse than a season that traded
    to a standstill. `IDLE_PENALTY` is the only thing separating them, which is why
    it is the constant to re-read if `own_orders_per_decision` collapses.
    """

    from agents.dqn_market_free_v3c.training.reward import IDLE_PENALTY, episode_rewards

    money, final = _flat_season(3_000.0)
    seeds, standing, stock = _lot_traces(len(money))
    kwargs = dict(seeds=seeds, standing=standing, stock=stock)

    quiet, _, _ = episode_rewards(money, final, 0.999, **kwargs)
    assert float(quiet.sum()) == pytest.approx(0.0, abs=1e-6)

    charged, _, _ = episode_rewards(money, final, 0.999, [False] * len(money), **kwargs)
    # Linear decay to the deadline, so a fully idle season pays about half the
    # per-turn charge on every turn.
    assert float(charged.sum()) == pytest.approx(
        -IDLE_PENALTY * len(money) / 2.0, rel=0.01
    )
    assert float(charged.sum()) > -0.1, "and stays well inside a season's range"


def test_a_filled_turn_quoted_in_coins_is_not_charged_as_refused():
    """The units on `order_refused`'s two arguments, which drifted apart once.

    `expected` is `PendingTurn.expected_delta` in coins; `raw` inside
    `episode_rewards` is already divided by `MONEY_SCALE`. Compared as they arrive,
    `|delta| < FILL_FRACTION * |quoted|` holds for every quote above a hundredth of a
    coin, so a turn whose orders all filled is charged as refused. Measured on
    `market-free-networth-v1` at generation 19 that was 50.1% of turns against a
    simulator refusal rate of 0.3%, and -0.36 an episode against a season's cash
    return of +0.16.

    The state side cannot catch it: `market_state` calls `order_refused` with no
    quote and a delta in coins, so the `refusal` feature stayed on the true 0.3%
    while the reward charged half the season -- exactly the drift the function's own
    docstring exists to prevent.
    """

    from agents.dqn_market_free_v3c.training.reward import (
        MONEY_SCALE,
        REJECTED_ORDER_PENALTY,
        episode_rewards,
    )

    turns = 100
    # A season that buys 80 coins of seed on every turn and ends holding the cash it
    # would have kept: every turn emits, every turn fills, nothing is refused.
    money = [3_000.0 - 80.0 * turn for turn in range(turns)]
    final = 3_000.0 - 80.0 * turns
    seeds, standing, stock = _lot_traces(turns)
    kwargs = dict(seeds=seeds, standing=standing, stock=stock)
    emitted = [True] * turns
    quoted = [-80.0] * turns

    filled, _, _ = episode_rewards(
        money, final, 1.0, emitted, expected=quoted, idle_penalty=0.0, **kwargs
    )
    assert float(filled.sum()) == pytest.approx(
        (final - 3_000.0) / MONEY_SCALE, abs=1e-6
    ), "a turn that moved the coins it quoted is not a refusal"

    # And the charge still lands where it belongs: a turn that quoted 80 coins and
    # moved none of them.
    stalled = [3_000.0] * turns
    refused, _, _ = episode_rewards(
        stalled, 3_000.0, 1.0, emitted, expected=quoted, idle_penalty=0.0, **kwargs
    )
    assert float(refused.sum()) == pytest.approx(
        -REJECTED_ORDER_PENALTY * turns, abs=1e-6
    )


def test_the_shaped_stream_ranks_a_winning_season_over_a_losing_one():
    """End to end through `episode_rewards`, which is what the network fits."""

    from agents.dqn_market_free_v3c.training.reward import episode_rewards

    kept, final_kept = _flat_season(124_000.0)
    lost, final_lost = _flat_season(1_000.0)
    won, _, _ = episode_rewards(kept, final_kept, 0.999)
    lose, _, _ = episode_rewards(lost, final_lost, 0.999)
    assert float(won.sum()) > 1.0
    assert float(lose.sum()) < 0.0
    assert float(won.sum()) - float(lose.sum()) > 1.2


def test_holding_the_basis_costs_a_little_and_selling_early_is_not_free():
    """The one thing `gamma` on the basis half buys.

    Discounting a cash objective leans toward early liquidation -- at 0.999 a sale
    deferred 240 turns is worth 0.79 of one made now. The basis is discounted with
    it, so sitting on stock costs `(1 - gamma) * basis` a turn: small, the correct
    sign, and absent at `gamma = 1.0` where nothing needs fixing.
    """

    from agents.dqn_market_free_v3c.training.reward import episode_rewards

    money, final = _flat_season(3_000.0)
    seeds, standing, stock = _lot_traces(
        len(money), seed_counts=[40, 0, 0, 0, 0], held_from=10
    )
    kwargs = dict(seeds=seeds, standing=standing, stock=stock)

    discounted, _, _ = episode_rewards(money, final, 0.999, **kwargs)
    undiscounted, _, _ = episode_rewards(money, final, 1.0, **kwargs)
    assert float(discounted.sum()) < float(undiscounted.sum())
    assert float(undiscounted.sum()) == pytest.approx(0.0, abs=1e-6)
    assert float(discounted.sum()) > -0.01, "a holding cost, not a charge"


def test_without_the_lot_traces_the_stream_falls_back_to_cash():
    """The all-or-nothing contract, and a warning rather than a reassurance.

    A demonstration read out of a replay may not recover the traces. The fallback
    is a bare cash delta, which scores every purchase as a pure loss -- so this
    pins that the group really is what supplies the basis, not that the fallback
    is safe.
    """

    from agents.dqn_market_free_v3c.training.reward import episode_rewards

    money, final = _flat_season(9_000.0)
    seeds, standing, stock = _lot_traces(len(money), seed_counts=[40, 0, 0, 0, 0])
    cash, raw, _ = episode_rewards(money, final, 0.999)
    assert cash.tolist() == raw.tolist()
    held, _, _ = episode_rewards(
        money, final, 0.999, seeds=seeds, standing=standing, stock=stock
    )
    assert held.tolist() != cash.tolist()

# --------------------------------------------------------------------------
# The behaviour-clone loss -- see `updater._large_margin_loss`.
#
# `market-free-splitladder-v1` ran a whole session at `max_abs_q` 20-27 because
# the clone was a cross-entropy over Q-values. These pin the property that fixed
# it: the loss has a fixed point, and it is inside the reward's range.
# --------------------------------------------------------------------------


def test_the_clone_loss_stops_once_the_demonstrated_action_leads():
    """Zero loss *and* zero gradient, which cross-entropy never reaches.

    This is the whole difference. Cross-entropy's optimum is the chosen logit at
    infinity, so it keeps pushing a Q head that is already ranking correctly, and
    `grad_clip_norm` bounds the step rather than the destination.
    """

    from agents.dqn_market_free_v3c.training.updater import (
        BEHAVIOR_CLONE_MARGIN,
        _large_margin_loss,
    )

    logits = torch.tensor([[5.0, 1.0, 2.0]], requires_grad=True)
    loss = _large_margin_loss(logits, torch.tensor([0]), BEHAVIOR_CLONE_MARGIN)
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert float(logits.grad.abs().sum()) == 0.0


def test_the_clone_loss_is_the_ranking_gap_plus_the_margin_in_q_units():
    """It states a preference ordering, not a target distribution."""

    from agents.dqn_market_free_v3c.training.updater import _large_margin_loss

    behind = torch.tensor([[1.0, 2.0, 0.0]])
    assert float(_large_margin_loss(behind, torch.tensor([0]), 0.3)) == pytest.approx(1.3)
    # Exactly at the margin is the boundary, and it is closed.
    edge = torch.tensor([[2.3, 2.0, 0.0]])
    assert float(_large_margin_loss(edge, torch.tensor([0]), 0.3)) == pytest.approx(0.0)


def test_the_clone_loss_cannot_be_dragged_up_by_a_masked_rival():
    """Infeasible rows arrive at `MASKED_SCORE`; the margin must not resurrect one."""

    from agents.dqn_market_free_v3c.order_space import MASKED_SCORE
    from agents.dqn_market_free_v3c.training.updater import _large_margin_loss

    masked = torch.tensor([[1.0, float(MASKED_SCORE), 0.0]])
    assert float(_large_margin_loss(masked, torch.tensor([0]), 0.3)) == 0.0


def test_the_clone_margin_is_sized_inside_the_reward_range():
    """A margin above the return's own range would set the Q scale itself.

    `outcome` is a tanh and every other term in `reward.py` is bounded, so the
    whole return lives inside roughly +/-2.
    """

    from agents.dqn_market_free_v3c.training.updater import BEHAVIOR_CLONE_MARGIN

    assert 0.0 < BEHAVIOR_CLONE_MARGIN < 1.0


# --------------------------------------------------------------------------
# Sell rungs against held stock -- see `order_space.stock_ceiling`.
# --------------------------------------------------------------------------


def test_the_stock_ceiling_is_the_smallest_rung_that_clears_the_shed():
    """Everything above it is the same fill under a different name."""

    from agents.dqn_market_free_v3c.order_space import QUANTITIES, stock_ceiling

    for held in range(0, 50):
        ceiling = stock_ceiling(held)
        assert ceiling in QUANTITIES
        assert ceiling >= min(held, QUANTITIES[-1])
        below = [level for level in QUANTITIES if level < ceiling]
        assert all(level < held for level in below), "only rungs under the shed survive below it"
    # Past the ladder it saturates rather than falling off.
    assert stock_ceiling(10_000) == QUANTITIES[-1]


def test_the_stock_clip_never_shuts_selling_outright():
    """A seat holding anything must keep a way to sell it.

    The smallest rung is 1 and the caller has already skipped items with nothing
    held, so the clip can narrow the ladder but never empty it -- the same
    guarantee `sell_depth_cap` makes with its `QUANTITIES[0]` floor.
    """

    from agents.dqn_market_free_v3c.order_space import QUANTITIES, stock_ceiling

    assert all(stock_ceiling(held) >= QUANTITIES[0] for held in range(1, 200))


def test_a_turn_sells_each_item_at_most_once():
    """One SELL per item per turn.

    A *preference*, not a proof like the wash rule above -- 45 + 45 units is a
    reachable outcome no single rung reproduces, because `QUANTITIES` stops at 45 --
    so it is carried on a measured ceiling instead. Over 300 ladder seats a second
    same-item SELL is 1.9% of sell orders and 1.4% of units; over
    `market-free-splitladder-v1` it was 72.7% of them.
    """

    from agents.dqn_market_free_v3c.order_space import (
        NOOP_INDEX,
        _SELL_ROWS,
        Order,
        PendingTurn,
        feasible_actions,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(
        money=5000.0, shed={"WHEAT": 60, "MELON": 20}
    )
    pending = PendingTurn(obs, farm, private, field, None)

    opening = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert opening[_SELL_ROWS["WHEAT"]].any(), "the turn must be able to sell at all"

    pending.apply(Order("SELL", "WHEAT", 6))
    after = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert not after[_SELL_ROWS["WHEAT"]].any(), "wheat is done for the turn"
    # A cap on repetition, not on trading: a different item and the terminator both
    # survive, so nothing the seat holds becomes unsellable.
    assert after[_SELL_ROWS["MELON"]].any(), "a different item stays sellable"
    assert after[NOOP_INDEX], "the terminator is never shut"


def test_sell_rungs_are_clipped_to_the_shed():
    """`stock_ceiling` in the mask: rungs past the holding are the same fill.

    The rung that clears the shed stays open -- that is the one that sells it all --
    and everything above it goes, because the shed bounds the fill and they would
    put the identical units on the wire.
    """

    from agents.dqn_market_free_v3c.order_space import (
        _ROW_QUANTITY,
        _SELL_ROWS,
        PendingTurn,
        feasible_actions,
        stock_ceiling,
    )
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0, shed={"WHEAT": 5})
    pending = PendingTurn(obs, farm, private, field, None)
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)

    rows = _SELL_ROWS["WHEAT"]
    ceiling = stock_ceiling(5)
    assert ceiling == 6, "the smallest rung that clears five units"
    assert mask[rows].any(), "five units stay sellable"
    for open_row, quantity in zip(mask[rows], _ROW_QUANTITY[rows]):
        if open_row:
            assert int(quantity) <= ceiling, quantity


def test_flip_rate_curve_measures_down_to_the_configured_sigma_floor():
    """The probe ladder spans exactly what `resolve_sigma_scale` may return.

    `noisy_sigma_scale_floor` *replaces* the implicit `1 / scale_limit` lower clamp
    rather than tightening it, and the resolver only ever answers with a scale the
    curve mentions. Clipping the ladder on the limit alone therefore made the floor
    dead config in one direction and let the probe measure unreachable rungs in the
    other: `market-free-cashherd-v1` ran `limit = 2` with `floor = 0.25`, measured
    {0.5, 1, 2}, and five of ten slots sat above the 0.05 flip-to-order setpoint at
    the bottom rung with nowhere to descend to.
    """

    import torch
    import torch.nn as nn

    from agents.dqn_market_free_v3c.training import updater

    class _Stub(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.linear = nn.Linear(4, width)

        def forward(self, features):  # noqa: D102 - test stub
            return self.linear(features)

    torch.manual_seed(0)
    module = _Stub(updater.ACTION_COUNT)
    features = torch.randn(16, 4)
    mask = torch.ones(16, updater.ACTION_COUNT, dtype=torch.bool)

    def rungs(limit: float, floor: float) -> list[float]:
        curve, _ = updater.flip_rate_curve(
            module, features, mask, limit, draws=1, scale_floor=floor
        )
        return sorted(curve)

    # The floor reaches below the implicit clamp, which is the case that was broken.
    assert min(rungs(2.0, 0.25)) == pytest.approx(0.25)
    # And above it, where the resolver may not return the lower rungs, they are not
    # measured either -- the ladder tracks the resolver in both directions.
    assert min(rungs(8.0, 0.5)) == pytest.approx(0.5)
    # Unset is byte-for-byte the old behaviour.
    assert rungs(2.0, 0.0) == [0.5, 1.0, 2.0]
    assert min(rungs(8.0, 0.0)) == pytest.approx(0.125)

    # Every rung the widened ladder adds is one the resolver can actually answer
    # with, which is the property that makes the floor mean anything.
    curve = {scale: 0.4 for scale in rungs(2.0, 0.25)}
    curve[0.25] = 0.01
    assert updater.resolve_sigma_scale(curve, 0.01, 2.0, 0.25) == pytest.approx(0.25)


def test_benchmark_reference_window_is_bounded_at_the_reseed_not_the_promotion():
    """The ratchet's anti-bias median has to accumulate on a *promoting* slot.

    Scoping the window to the incumbent's own tenure resets it on every promotion,
    which starves it exactly where the ratchet is load-bearing. `market-free-
    cashherd-v1` was configured for eleven samples and ran `cashherd_03` and
    `cashherd_09` -- the two slots that promoted on nearly every update -- at two
    throughout, while its stalled slots reached five.

    The bound is the slot's reseed, matching `_slot_score`, because that is the
    real discontinuity: a promotion one update back is the same near-neighbour the
    window already medians over inside a tenure.
    """

    from agents.dqn_market_free_v3c.training.updater import _benchmark_reference_window

    opponent = "agents/competitor_pilkwang/main.py"

    class _Reader:
        def __init__(self):
            self.sql = None
            self.parameters = None

        def fetchall(self, sql, parameters):
            self.sql = " ".join(sql.split())
            self.parameters = list(parameters)
            return [
                ({"challenger_benchmarks": [{"opponent": opponent, "a_money_mean": money}]},)
                for money in (3000.0, 1000.0, 2000.0)
            ]

    reader = _Reader()
    active = SimpleNamespace(checkpoint_id="ckpt")
    samples = _benchmark_reference_window(reader, "model", active, opponent, 11, 20)

    assert samples == [3000.0, 1000.0, 2000.0], "every measurement, unordered by value"
    # The bound is the reseed's batch floor, carried as a parameter...
    assert "cutoff_batch_seq > %s" in reader.sql
    assert reader.parameters == ["model", 20, 11]
    # ...and the promotion scope that used to starve the window is gone.
    assert "PROMOTED" not in reader.sql

    # A window of one is still the single stored draw, and no incumbent is no window.
    assert _benchmark_reference_window(reader, "model", active, opponent, 1, 20) == []
    assert _benchmark_reference_window(reader, "model", None, opponent, 11, 20) == []


# --------------------------------------------------------------------------
# The money delta and the bought-stock pair -- see `reward.MONEY_DELTA_WEIGHT`,
# `reward.STOCK_LOSS_WEIGHT` and `reward.STOCK_GAIN_WEIGHT`.
#
# What these guard is the hole those constants closed. `market-free-splitladder-v1`
# moved 213,280 coins of gross revenue an episode-seat and kept 1,843 of them, a
# 0.9% net margin against a stored ladder champion's 67.3%, because 89% of that
# turnover was WHEAT and FERTILIZER round-tripped through the market and a
# `BUY_PRODUCT` lot was read by no reward leg at all.
# --------------------------------------------------------------------------


def _bought_lot(
    item: str = "WHEAT",
    units: int = 45,
    unit_cost: float = 40.0,
    absorbed: int = 0,
    proceeds: float = 0.0,
    buy_turn: int = 10,
):
    """One `BUY_PRODUCT` lot, posed directly rather than matched out of a trace.

    Bought stock resells one for one, so `capacity == units` is what `lots.match`
    opens it with and `absorbed` is the unit count that later left through a SELL.
    """

    from agents.dqn_market_free_v3c.lots import _Lot

    lot = _Lot(
        item=item,
        buy_turn=buy_turn,
        units=units,
        unit_cost=unit_cost,
        capacity=units,
        bought=True,
    )
    lot.absorbed = absorbed
    lot.proceeds = proceeds
    if absorbed:
        lot.first_sale_turn = buy_turn + 1
        lot.last_sale_turn = buy_turn + 1
    return lot


def _ledger_of(*lots_in):
    from agents.dqn_market_free_v3c.lots import LotLedger

    return LotLedger(lots=list(lots_in))


def test_a_bought_lot_that_lost_on_the_round_trip_charges_its_buy_turn():
    """The behaviour the pair exists to rank: buy 45 wheat, sell it back for less.

    Both legs read the *realised* trip -- proceeds against the cost of the units
    that actually left -- so a losing trip charges and a winning one credits, and
    the two are clamped so a policy cannot collect on both.
    """

    lost = _bought_lot(absorbed=45, proceeds=1_500.0)  # cost of sold units 1,800
    won = _bought_lot(absorbed=45, proceeds=2_400.0)

    assert _ledger_of(lost).stock_loss() == [(10, 300.0)]
    assert _ledger_of(lost).stock_gain() == [(10, 0.0)]
    assert _ledger_of(won).stock_loss() == [(10, 0.0)]
    assert _ledger_of(won).stock_gain() == [(10, 600.0)]


def test_bought_stock_consumed_rather_than_resold_is_charged_nothing():
    """The reason this is a realised-margin leg and not `unrecovered()`.

    Wheat bought to feed the herd and fertilizer bought to spread on the tiles
    both end with `proceeds == 0`, so the seed and animal definition -- `cost -
    proceeds` over the whole lot -- would charge the farm for using the input it
    bought. Counting only `unit_cost * absorbed` prices the round trip and nothing
    else.
    """

    from agents.dqn_market_free_v3c.lots import _unrecovered

    fed = _bought_lot(item="WHEAT", units=45, unit_cost=40.0, absorbed=0)
    spread = _bought_lot(item="FERTILIZER", units=20, unit_cost=85.0, absorbed=0)

    for lot in (fed, spread):
        ledger = _ledger_of(lot)
        assert ledger.stock_loss() == [(10, 0.0)]
        assert ledger.stock_gain() == [(10, 0.0)]
    # And the definition that would have got this wrong, stated so the difference
    # is not re-derived by hand the next time someone reaches for the seed helper.
    assert _unrecovered([fed]) == [(10, 1_800.0)]


def test_the_bought_stock_legs_are_clamped_per_lot_and_never_netted():
    """Each lot moves at most the cost of what it sold, in either direction.

    The cap is what bounds the pair by resale turnover, and is why a linear-coin
    term cannot quietly displace `outcome` as the objective.
    """

    windfall = _bought_lot(absorbed=45, proceeds=20_000.0)
    ledger = _ledger_of(windfall)
    sold_cost = 45 * 40.0
    assert ledger.stock_gain() == [(10, sold_cost)], "capped at the cost of what it sold"
    assert ledger.stock_loss() == [(10, 0.0)]

    partial = _bought_lot(units=45, absorbed=10, proceeds=100.0)
    # Only the ten units that left are priced; the 35 still in the shed are not.
    assert _ledger_of(partial).stock_loss() == [(10, 300.0)]


def test_the_bought_stock_legs_are_only_the_lots_no_other_leg_reads():
    """`bought_lots()` and the two existing accessors partition the ledger.

    WHEAT is both a crop and a `BUY_PRODUCT` item, so a lot landing in two legs
    would be billed twice for the same coins.
    """

    from agents.dqn_market_free_v3c.lots import _Lot

    seed = _Lot(item="WHEAT", buy_turn=3, units=2, unit_cost=10.0, capacity=12)
    cow = _Lot(item="MILK", buy_turn=4, units=1, unit_cost=400.0, capacity=8, animal="COW")
    ledger = _ledger_of(seed, cow, _bought_lot())

    assert ledger.seed_lots() == [seed]
    assert ledger.animal_lots() == [cow]
    assert [lot.item for lot in ledger.bought_lots()] == ["WHEAT"]
    assert len(ledger.seed_lots()) + len(ledger.animal_lots()) + len(
        ledger.bought_lots()
    ) == len(ledger.lots)


def test_split_legs_carries_the_bought_stock_pair():
    """Six keys, and the bought pair resolves like every other lot."""

    ledger = _ledger_of(_bought_lot(absorbed=45, proceeds=1_500.0))
    legs = ledger.split_legs(719)
    assert set(legs) == {
        "unrecovered",
        "surplus",
        "animal_unrecovered",
        "animal_surplus",
        "stock_loss",
        "stock_gain",
    }
    assert legs["stock_loss"] == [(10, 11, 300.0)], "resolved on its last sale"
    assert legs["stock_gain"] == [(10, 11, 0.0)]

    never = _ledger_of(_bought_lot())
    assert never.split_legs(719)["stock_loss"] == [(10, 719, 0.0)]


def test_a_fertilizer_lot_is_reconciled_against_the_shed():
    """The gap the third `_reconcile` pass closed.

    FERTILIZER is in `STOCK_ITEMS` but is neither a crop nor an animal's product, so
    neither existing ceiling iterated it: a lot whose units were spread on the tiles
    kept its full room forever, stayed at the head of the FIFO and absorbed a later
    sale a later purchase had paid for.
    """

    from agents.dqn_market_free_v3c import lots

    fert = lots.STOCK_ITEMS.index("FERTILIZER")
    assert lots._TRADED_ONLY == ("FERTILIZER",)

    crops = len(lots.CROP_ITEMS)
    turns = 5
    seeds = [[0] * crops for _ in range(turns + 1)]
    standing = [[0] * crops for _ in range(turns + 1)]
    stock = [[0] * len(lots.STOCK_ITEMS) for _ in range(turns + 1)]
    ops: list[str | None] = [None] * turns
    events: list[list] = [[] for _ in range(turns)]
    money = [3_000.0] * (turns + 1)

    # Turn 0 buys 10 fertilizer for 850; turn 1 spreads all ten on the tiles --
    # stock down, money flat, matching no rule in `match`.
    stock[1][fert] = 10
    ops[0] = "BUY_PRODUCT"
    events[0] = [("BUY_PRODUCT", "FERTILIZER", 10, -850.0)]
    money[1] = 2_150.0
    for turn in range(2, turns + 1):
        stock[turn][fert] = 0

    # Turn 3 buys 4 more and turn 4 sells them for 400.
    stock[4][fert] = 4
    ops[3] = "BUY_PRODUCT"
    events[3] = [("BUY_PRODUCT", "FERTILIZER", 4, -340.0)]
    money[4] = 1_810.0
    stock[5][fert] = 0
    ops[4] = "SELL"
    events[4] = [("SELL", "FERTILIZER", 4, 400.0)]
    money[5] = 2_210.0

    ledger = lots.match(seeds, stock, money, ops, standing, events=events)
    bought = ledger.bought_lots()
    assert [lot.buy_turn for lot in bought] == [0, 3]
    spread, resold = bought
    assert spread.absorbed == 0, "the spread lot must not absorb the later sale"
    assert resold.absorbed == 4 and resold.proceeds == 400.0
    # And therefore the reward charges the trip that happened, not the one that did not.
    assert ledger.stock_loss() == [(0, 0.0), (3, 0.0)]
    assert ledger.stock_gain() == [(0, 0.0), (3, 60.0)]


def test_a_full_season_stays_in_the_huber_quadratic_zone():
    """`SmoothL1Loss` has a beta of 1.0, so a per-step reward past it flattens the
    gradient: a 5-unit and a 50-unit TD error would give the same step.

    This is the test `MONEY_SCALE` was chosen against. The objective is now a paid
    per-turn delta, so what it bounds is the largest single wallet move -- and the
    whole-season total too, since the stream sums to `final - stake` and a
    ladder-grade season is the largest that total gets. A trace that sits at the
    stake and then teleports to 124,000 on the terminal step is not a season, it is
    one 121,000-coin transaction, so the money climbs turn by turn.
    """

    from agents.dqn_market_free_v3c.training.reward import episode_rewards

    crops = 5
    turns = 720
    money = [3_000.0 + 168.0 * turn - 40.0 * (turn % 25) for turn in range(turns)]
    shaped, _, _ = episode_rewards(
        money,
        124_000.0,
        0.999,
        emitted=[True] * turns,
        seeds=[[0] * crops] * turns,
        standing=[[0] * crops] * turns,
        stock=[[0] * 9] * turns,
    )
    assert float(np.abs(shaped).max()) < 1.0
    assert float(np.abs(shaped.sum())) < 2.0, "and so does a whole ladder-grade season"


def test_the_lagged_wash_counter_sees_what_the_mask_does_not():
    """`wash_turns` reads zero all session; the round trip moved one turn out.

    Measured over 20 stored `market-free-splitladder-v1` seats: 0.0% of resold
    units at lag 0, 17.3% of wheat and 49.8% of fertilizer at lag 1.
    """

    from agents.dqn_market_free_v3c.order_space import ACTIONS
    from agents.dqn_market_free_v3c.training.collect import (
        LAGGED_WASH_WINDOW,
        EpisodeRecord,
    )

    def index_of(op: str, item: str) -> int:
        for index, action in enumerate(ACTIONS):
            if action.op == op and action.item == item:
                return index
        raise AssertionError(f"no {op} {item} in the action space")

    buy = index_of("BUY_PRODUCT", "WHEAT")
    sell = index_of("SELL", "WHEAT")

    def record(pairs):
        chosen = np.asarray([choice for _, choice in pairs], dtype=np.int16)
        steps = np.asarray([turn for turn, _ in pairs], dtype=np.int16)
        empty = np.zeros(0, dtype=np.float32)
        return EpisodeRecord(
            reward=np.zeros(800, dtype=np.float32),
            raw_reward=np.zeros(800, dtype=np.float32),
            money=np.zeros(801, dtype=np.float32),
            order_state=np.zeros((len(pairs), 1), dtype=np.float32),
            order_chosen=chosen,
            order_step=steps,
            order_mask=np.ones((len(pairs), len(ACTIONS)), dtype=bool),
            turn_state=np.zeros((800, 1), dtype=np.float32),
            turn_hidden=np.zeros((1, 1), dtype=np.float32),
            hidden_stride=np.asarray([20], dtype=np.int16),
        )

    next_turn = record([(0, buy), (1, sell)])
    assert next_turn.wash_turns == 0, "the mask holds: the legs are not on one turn"
    assert next_turn.lagged_wash_turns == 1

    # Beyond the window it is inventory management, not a round trip.
    far = record([(0, buy), (LAGGED_WASH_WINDOW + 1, sell)])
    assert far.lagged_wash_turns == 0
    edge = record([(0, buy), (LAGGED_WASH_WINDOW, sell)])
    assert edge.lagged_wash_turns == 1

    # A sale with no purchase behind it is the farm selling what it grew.
    assert record([(0, sell), (1, sell)]).lagged_wash_turns == 0
