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

import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from agents.dqn_market_free_v3 import donor, herd, lots
from agents.dqn_market_free_v3.market_state import (
    HERD_VIABILITY_DIM,
    HISTORY_BLOCK_DIM,
    INTRA_TURN_DIM,
    INTRA_TURN_SLICE,
    MARKET_STATE_DIM,
)
from agents.dqn_market_free_v3.order_space import (
    ACTION_COUNT,
    ACTIONS,
    MASKED_SCORE,
    Order,
    PendingTurn,
    feasible_actions,
    sell_depth_cap,
)
from agents.dqn_market_free_v3.policy import (
    _LAYER_KEY,
    HEAD_IN,
    HIDDEN_SIZE,
    OrderQNetwork,
    _sigmoid,
)
from agents.dqn_market_free_v3.training import network as net_module
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
    assert (
        MARKET_STATE_DIM == 238 - HISTORY_BLOCK_DIM + INTRA_TURN_DIM + HERD_VIABILITY_DIM
    )
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


def test_a_noisy_initial_network_keeps_sigma_after_export(tmp_path):
    from agents.dqn_market_free_v3.training.updater import initial_network

    network = initial_network(
        {"seed": 3, "noisy": True, "noisy_layers": "head", "noisy_sigma0": 0.5}
    )
    assert network.is_noisy
    assert not any(pair is not None for pair in network.sigma[:-1])
    assert network.sigma[-1] is not None

    path = tmp_path / "initial.npz"
    network.save(path)
    loaded = OrderQNetwork.load(path)
    assert loaded.is_noisy


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


def test_a_buy_animal_beyond_free_housing_is_shut():
    from agents.dqn_market_free_v3.order_space import _BUY_ANIMAL_ROWS, animal_room

    obs, farm, private, field = _mask_state(money=100_000)
    assert animal_room(obs, farm, private) == {animal: 4 for animal in herd.ANIMALS}

    crowded = {**private, "shed": {**(private.get("shed") or {}), "COW": 50}}
    assert animal_room(obs, farm, crowded) == {}
    pending = PendingTurn(obs, farm, crowded, field, None)
    mask = feasible_actions(obs, farm, crowded, field, None, pending=pending)
    for animal in herd.ANIMALS:
        assert not mask[_BUY_ANIMAL_ROWS[animal]].any(), animal


def test_a_turn_cannot_buy_more_animals_than_it_can_house():
    from agents.dqn_market_free_v3.order_space import _BUY_ANIMAL_ROWS

    obs, farm, private, field = _mask_state(money=100_000)
    pending = PendingTurn(obs, farm, private, field, None)
    pending.apply(Order("BUY_ANIMAL", "COW", 4))
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    for animal in herd.ANIMALS:
        assert not mask[_BUY_ANIMAL_ROWS[animal]].any(), animal


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


def test_sell_accounting_commits_only_stock_the_shed_holds():
    from agents.dqn_market_free_v3.order_space import _ROW_QUANTITY, _SELL_ROWS

    pending, (obs, farm, private, field) = _pending(shed={"WHEAT": 1})
    rows = _SELL_ROWS["WHEAT"]
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert mask[rows].tolist() == (_ROW_QUANTITY[rows] <= 1).tolist()

    pending.apply(Order("SELL", "WHEAT", 45))
    expected = float(donor._price_at("WHEAT", donor.MARKET_I0, obs))
    assert pending.shed["WHEAT"] == 0
    assert pending.occupancy == 0
    assert pending.expected_delta == pytest.approx(expected)
    assert pending.market_inventory["WHEAT"] == donor.MARKET_I0 + 1
    assert pending.events == [("SELL", "WHEAT", 1, expected)]


def test_product_buys_are_capped_by_remaining_shed_capacity():
    from agents.dqn_market_free_v3.order_space import _BUY_PRODUCT_ROWS, _ROW_QUANTITY

    shed = {f"ITEM_{index}": 1 for index in range(99)}
    pending, (obs, farm, private, field) = _pending(money=100_000, shed=shed)
    rows = _BUY_PRODUCT_ROWS["FERTILIZER"]
    mask = feasible_actions(obs, farm, private, field, None, pending=pending)
    assert mask[rows].tolist() == (_ROW_QUANTITY[rows] <= 1).tolist()

    pending.apply(Order("BUY_PRODUCT", "FERTILIZER", 45))
    assert pending.occupancy == 100
    assert pending.shed["FERTILIZER"] == 1
    assert pending.events[-1][2] == 1


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

    from agents.dqn_market_free_v3.training.collect import HIDDEN_STRIDE, EpisodeRecord

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
    from agents.dqn_market_free_v3.training.replay import ReplayBuffer

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

    from agents.dqn_market_free_v3.training.collect import HIDDEN_STRIDE

    with pytest.raises(ValueError, match="multiple of the stored-hidden stride"):
        _buffer().sample_sequences(2, burn_in=HIDDEN_STRIDE + 1, train=40)


def test_an_episode_too_short_for_a_training_window_is_not_drawn():
    """`record.decisions` -- the flat sampler's predicate -- would pass an episode
    with plenty of decisions packed into too few turns, and then index off the end
    of `turn_state`."""

    from agents.dqn_market_free_v3.training.replay import ReplayBuffer

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

    from agents.dqn_market_free_v3.training.updater import _roll_chunk

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

    from agents.dqn_market_free_v3.order_space import MASKED_SCORE
    from agents.dqn_market_free_v3.training.updater import _roll_chunk

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

    from agents.dqn_market_free_v3 import donor
    from agents.dqn_market_free_v3.order_space import (
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

    from agents.dqn_market_free_v3 import donor
    from agents.dqn_market_free_v3.order_space import PendingTurn
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

    from agents.dqn_market_free_v3.main import _shed_unmoved
    from agents.dqn_market_free_v3.market_state import (
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

    from agents.dqn_market_free_v3.main import MarketFreeAgent
    from agents.dqn_market_free_v3.training.clone import record_from_replay
    from agents.dqn_market_free_v3.training.collect import HIDDEN_STRIDE, Recorder
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
        opponent_money=float(env.state[0]["observation"]["farms"][1]["money"]),
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

    from agents.dqn_market_free_v3.market_state import (
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

    from agents.dqn_market_free_v3.main import MarketFreeAgent
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

    from agents.dqn_market_free_v3.training.updater import _recurrence_metrics

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

    from agents.dqn_market_free_v3.training import network as net
    from agents.dqn_market_free_v3.training.updater import _behavior_clone_step

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


def test_behavior_clone_loss_uses_turn_shares():
    import inspect

    from agents.dqn_market_free_v3.training.updater import _behavior_clone_step

    source = inspect.getsource(_behavior_clone_step)
    assert 'reduction="none"' in source
    assert "batch.turn_share" in source
    assert ".index_add_" in source


def test_benchmark_ratchet_rejects_catastrophic_benchmark_losses():
    from agents.dqn_market_free_v3.training.updater import _benchmark_ratchet

    result = {
        "a_money_mean": 9_000.0,
        "b_money_mean": 133_000.0,
        "errors": {},
    }
    gate = _benchmark_ratchet(result, None, None, {}, grace=False)
    assert gate["checks"]["min_money"]
    assert gate["checks"]["opponent_ratio"]
    assert not gate["checks"]["max_deficit"]
    assert not gate["passed"]


def test_order_viability_requires_productive_behavior_when_recorded():
    from agents.dqn_market_free_v3.training.updater import _order_viability

    result = {
        "pairs": 30,
        "errors": {},
        "order_diagnostics": {
            "decisions": 720,
            "orders": 30,
            "trading_rate": 0.05,
        },
        "behavior": {
            "seed_lots": 0,
            "animal_lots": 0,
        },
    }
    gate = _order_viability(result, {}, require_activity=True)
    assert gate["checks"]["min_orders_per_decision"]
    assert not gate["checks"]["productive_lots"]
    assert not gate["passed"]


# --------------------------------------------------------------------------
# The wash
# --------------------------------------------------------------------------


def test_a_same_item_round_trip_is_worth_exactly_nothing():
    """The premise of the mask, priced rather than asserted.

    Buying walks the market inventory down and selling it back walks the same
    price levels up, so the pair nets exactly zero at every quantity. That is why
    v2 could spend 47% of its turns on it for free.
    """

    from agents.dqn_market_free_v3 import donor

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

    from agents.dqn_market_free_v3.order_space import (
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
    from agents.dqn_market_free_v3.order_space import (
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

    from agents.dqn_market_free_v3.order_space import (
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

    from agents.dqn_market_free_v3.order_space import _BUY_PRODUCT_ROWS, feasible_actions
    from tests.test_dqn_market_free import _mask_state

    obs, farm, private, field = _mask_state(money=5000.0, shed={"WHEAT": 60})
    mask = feasible_actions(obs, farm, private, field, None)
    assert mask[_BUY_PRODUCT_ROWS["WHEAT"]].any()


def test_a_zero_quote_turn_is_charged_again_when_nothing_moved():
    """v2 waved this through and it was one of the three things that made the
    wash reward-optimal. With same-item round trips unreachable, a zero quote and
    a zero delta is a refusal again -- v1's reading."""

    from agents.dqn_market_free_v3.market_state import order_refused

    assert bool(order_refused(True, 0.0, 0.0)), "refused: nothing quoted, nothing moved"
    assert not bool(order_refused(True, 42.0, 0.0)), "filled: the wallet moved"
    assert bool(order_refused(True, 0.0, 500.0)), "refused: quoted 500, moved nothing"
    assert not bool(order_refused(True, 500.0, 500.0))
    assert not bool(order_refused(False, 0.0, 0.0)), "nothing emitted, nothing to refuse"


def test_a_buy_animal_rung_the_wallet_cannot_cover_is_shut():
    """The last place a single affordable unit opened the whole ladder."""

    from agents.dqn_market_free_v3 import donor
    from agents.dqn_market_free_v3.order_space import (
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

    from agents.dqn_market_free_v3.training.updater import _batch_lot_metrics

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

    from agents.dqn_market_free_v3.training.updater import _batch_lot_metrics

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

    from agents.dqn_market_free_v3.training.updater import _batch_lot_metrics

    metrics = _batch_lot_metrics((_lot_batch(animal_waste_cost=500.0),))
    assert "animal_waste_cost" in metrics
    assert "seed_waste_cost" not in metrics
    assert _batch_lot_metrics(()) == {}


def test_batch_lot_metrics_reject_values_that_are_not_finite_numbers():
    """A None or a NaN must not poison the mean for the whole update."""

    from agents.dqn_market_free_v3.training.updater import _batch_lot_metrics

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

    from agents.dqn_market_free_v3.diagnostics import BehaviorCounts
    from agents.dqn_market_free_v3.training.updater import _BATCH_LOT_KEYS

    exported = BehaviorCounts().as_dict()
    # The collector writes the `_per_episode` variants under the bare name.
    available = set(exported) | {
        key[: -len("_per_episode")] for key in exported if key.endswith("_per_episode")
    }
    missing = sorted(key for key in _BATCH_LOT_KEYS if key not in available)
    assert not missing, f"lifted keys the collector never writes: {missing}"


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

    from agents.dqn_market_free_v3.main import MarketFreeAgent
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

    from agents.dqn_market_free_v3.training.clone import record_from_replay, rehidden

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
    from agents.dqn_market_free_v3.training.clone import record_from_replay, rehidden

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

    from agents.dqn_market_free_v3.training import ladder
    from agents.dqn_market_free_v3.training.clone import record_from_replay

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

    from agents.dqn_market_free_v3.training import ladder

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

    from agents.dqn_market_free_v3.training import ladder

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
