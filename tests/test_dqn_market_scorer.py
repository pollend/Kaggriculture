"""Correctness guards for the market-scoring agent.

These pin the invariants that would otherwise fail silently: a donor copy that
drifted from the licensed snapshot, train/inference feature skew, a learned
reserve that leaks into terminal liquidation, and a cold start that does not
actually reproduce the donor.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from agents.dqn_market_scorer import donor, market_exec
from agents.dqn_market_scorer.main import MarketScorerAgent
from agents.dqn_market_scorer.market_policy import (
    DEFAULT_MODIFIER,
    MARKET_ACTION_DIM,
    MARKET_FEATURE_DIM,
    MARKET_STATE_DIM,
    MODIFIERS,
    PRODUCTS,
    MarketModifier,
    choose_market,
    enumerate_market_candidates,
    market_state_features,
    modifier_action_features,
    reserve_action_features,
)
from agents.dqn_market_scorer.policy import MarketQNetwork
from agents.dqn_market_scorer.training import network as net_module
from agents.dqn_market_scorer.training.collect import (
    EPISODE_ARRAY_FIELDS,
    MONEY_SCALE,
    TERMINAL_MARGIN_SCALE,
    Recorder,
    collect_episode,
    load_episode_batch,
    save_episode_batch,
)
from agents.dqn_market_scorer.training.replay import ReplayBuffer

REPO_ROOT = Path(__file__).resolve().parents[1]
DONOR_PATH = REPO_ROOT / "agents/dqn_market_scorer/donor.py"
SNAPSHOT_PATH = (
    REPO_ROOT
    / "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"
)
GAMMA = 0.997


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_donor_is_a_verbatim_copy_of_the_licensed_snapshot():
    """The NOTICE claims donor.py is unmodified. Keep that claim true.

    Everything the licence obligation rests on -- and the argument that only
    `market_exec.py` is a derivative work -- is that this file was not touched.
    Edit the donor and this test is what tells you to edit `market_exec.py`
    instead.
    """

    assert _digest(DONOR_PATH) == _digest(SNAPSHOT_PATH)


def test_donor_digest_matches_the_recorded_provenance():
    notice = (REPO_ROOT / "agents/dqn_market_scorer/NOTICE").read_text()
    assert _digest(DONOR_PATH) in notice
    assert (REPO_ROOT / "agents/dqn_market_scorer/LICENSE-pilkwang").is_file()


def test_submission_manifest_ships_the_licence_with_the_copy():
    import tomllib

    with (REPO_ROOT / "agents/dqn_market_scorer/agent.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    paths = set(manifest["code_paths"])
    assert "agents/dqn_market_scorer/donor.py" in paths
    assert "agents/dqn_market_scorer/LICENSE-pilkwang" in paths
    assert "agents/dqn_market_scorer/NOTICE" in paths
    assert not any("competitor_" in path for path in paths)


# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------


def test_candidate_grid_starts_at_the_donor_default():
    # A zero network scores every row equally and argmax breaks to index 0, so
    # the cold-start policy is whatever sits in this slot.
    assert MODIFIERS[0] == DEFAULT_MODIFIER
    assert len(set(MODIFIERS)) == len(MODIFIERS)
    assert DEFAULT_MODIFIER.feed_days == donor.FEED_STOCK_DAYS


def test_action_encoders_match_the_declared_width():
    assert modifier_action_features(DEFAULT_MODIFIER).shape == (MARKET_ACTION_DIM,)
    assert reserve_action_features("WHEAT", 4, 10).shape == (MARKET_ACTION_DIM,)
    # Purchase and reserve rows must be separable by the network, or the shared
    # trunk cannot tell a modifier row from a reserve row for the same state.
    purchase = modifier_action_features(DEFAULT_MODIFIER)
    reserve = reserve_action_features("WHEAT", 4, 10)
    assert (purchase[0], purchase[1]) == (1.0, 0.0)
    assert (reserve[0], reserve[1]) == (0.0, 1.0)


# --------------------------------------------------------------------------
# Live episode invariants
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_states():
    """Real observations, roles and field plans sampled across one episode."""

    from kaggle_environments import make

    captured = []
    agent = MarketScorerAgent(MarketQNetwork.zeros())

    def probe(obs, configuration=None):
        action = agent(obs, configuration)
        step = int(obs.get("step", 0) or 0)
        if step % 180 == 0 or step == 719:
            farm = obs["farms"][int(obs.get("player", 0) or 0)]
            private = obs.get("private", {}) or {}
            roles = donor._role_plan(obs, farm)
            field = donor._unit_actions(obs, configuration, farm, private, roles)
            captured.append(
                (copy.deepcopy(obs), configuration, copy.deepcopy(farm), copy.deepcopy(private), roles, field)
            )
        return action

    env = make("kaggriculture", debug=True)
    env.run([probe, str(SNAPSHOT_PATH)])
    assert captured
    return captured


def test_state_features_match_the_declared_width(live_states):
    for obs, config, farm, private, roles, field in live_states:
        row = market_state_features(obs, farm, private, roles, field, config=config)
        assert row.shape == (MARKET_STATE_DIM,)
        assert np.isfinite(row).all()


def test_assembled_candidates_match_the_network_input_width(live_states):
    obs, config, farm, private, roles, field = live_states[1]
    candidates = enumerate_market_candidates(obs, farm, private, roles, field, config=config)
    assert candidates.purchase_static.shape == (len(MODIFIERS), MARKET_ACTION_DIM)
    assert candidates.reserve_static.shape[0] == len(PRODUCTS)
    assert candidates.factor_count == 1 + len(PRODUCTS)
    network = MarketQNetwork.zeros()
    decision = choose_market(candidates, network)
    # Zero Q everywhere: every factor takes index 0, which is the donor default
    # and a zero reserve.
    assert decision.modifier == DEFAULT_MODIFIER
    assert set(decision.reserves.values()) == {0}
    assert len(decision.chosen) == candidates.factor_count


def test_reserve_levels_never_exceed_what_is_in_the_shed(live_states):
    for obs, config, farm, private, roles, field in live_states:
        candidates = enumerate_market_candidates(obs, farm, private, roles, field, config=config)
        shed, _ = donor._post_field_storage(private, field, donor.SHED_CAPACITY)
        for index, item in enumerate(PRODUCTS):
            available = max(0, int(shed.get(item, 0) or 0))
            assert max(candidates.reserve_values[index]) <= available


# --------------------------------------------------------------------------
# The derivative executor
# --------------------------------------------------------------------------


def test_cold_start_reproduces_the_donor_exactly():
    """A zero network must play unmodified pilkwang, action for action.

    This is what makes an untrained checkpoint a strong agent instead of a
    random one, and it is the only check that the knobs in `market_exec` are
    genuinely no-ops at their defaults.
    """

    from kaggle_environments import make

    import agents.competitor_pilkwang_kaggriculture_structured_economic_policy.main as snapshot

    agent = MarketScorerAgent(MarketQNetwork.zeros())
    differences = []

    def probe(obs, configuration=None):
        ours = agent(obs, configuration)
        theirs = snapshot.agent(copy.deepcopy(obs), configuration)
        if ours != theirs:
            differences.append(int(obs.get("step", 0) or 0))
        # Play the donor so both trajectories stay on the same states.
        return theirs

    env = make("kaggriculture", debug=True)
    env.run([probe, str(SNAPSHOT_PATH)])
    assert differences == []


def test_learned_reserve_withholds_stock_from_the_sale(live_states):
    obs, config, farm, private, roles, field = live_states[2]
    shed, _ = donor._post_field_storage(private, field, donor.SHED_CAPACITY)
    stocked = [item for item in PRODUCTS if int(shed.get(item, 0) or 0) >= 2]
    if not stocked:
        pytest.skip("no product in the shed at the sampled step")
    item = stocked[0]
    available = int(shed[item])

    baseline = market_exec.market_actions(obs, config, farm, private, roles, field)
    held = market_exec.market_actions(
        obs, config, farm, private, roles, field, reserves={item: available}
    )

    def sold(orders):
        return sum(int(order[2]) for order in orders if order[0] == "SELL" and order[1] == item)

    assert sold(held) == 0
    assert sold(held) <= sold(baseline)


def test_reserves_are_ignored_once_the_donor_is_liquidating(live_states):
    obs, config, farm, private, roles, field = live_states[2]
    shed, _ = donor._post_field_storage(private, field, donor.SHED_CAPACITY)
    stocked = [item for item in PRODUCTS if int(shed.get(item, 0) or 0) >= 2]
    if not stocked:
        pytest.skip("no product in the shed at the sampled step")
    reserves = {item: int(shed[item]) for item in stocked}

    liquidating = dict(field)
    liquidating["liquidation"] = True
    withheld = market_exec.market_actions(
        obs, config, farm, private, roles, liquidating, reserves=reserves
    )
    free = market_exec.market_actions(obs, config, farm, private, roles, liquidating)
    # Holding stock into the terminal liquidation strands it worthless, so the
    # reserve must not survive the switch.
    assert withheld == free


def test_seed_scale_zero_blocks_seed_purchases(live_states):
    for obs, config, farm, private, roles, field in live_states:
        blocked = market_exec.market_actions(
            obs,
            config,
            farm,
            private,
            roles,
            field,
            modifier=MarketModifier(seed_scale=0.0),
        )
        assert not [order for order in blocked if order[0] == "BUY_SEED"]


def test_animal_gate_only_ever_removes_permission(live_states):
    for obs, config, farm, private, roles, field in live_states:
        vetoed = market_exec.market_actions(
            obs, config, farm, private, roles, field, modifier=MarketModifier(animal_gate=0)
        )
        assert not [order for order in vetoed if order[0] == "BUY_ANIMAL"]


def test_hand_delta_respects_the_donor_labour_ceiling(live_states):
    for obs, config, farm, private, roles, field in live_states:
        greedy = market_exec.market_actions(
            obs, config, farm, private, roles, field, modifier=MarketModifier(hand_delta=2)
        )
        hires = sum(1 for order in greedy if order[0] == "HIRE")
        already = int(farm.get("hires_today", 0) or 0)
        assert already + hires <= market_exec._hand_cap(obs, farm)


# --------------------------------------------------------------------------
# Train / inference parity
# --------------------------------------------------------------------------


def test_systematic_failure_is_raised_rather_than_passed_over(monkeypatch):
    """A broken agent must not evaluate as a merely bad one.

    `_arena_evaluate` counts seat errors, so an agent that silently returns PASS
    forever would be scored as a legitimate loss and the promotion gate would
    read a harness fault as evidence about the candidate.
    """

    import agents.dqn_market_scorer.main as module

    class Exploding:
        recorder = None

        def __call__(self, obs, configuration=None):
            raise RuntimeError("feature schema drifted")

    monkeypatch.setattr(module, "_AGENT", Exploding())
    monkeypatch.setattr(module, "_ERRORS", 0)
    obs = {"step": 0, "player": 0, "farms": [{"hands": [[0, 0]]}, {}]}
    for step in range(module.MAX_SWALLOWED_ERRORS):
        fallback = module.agent({**obs, "step": step + 1})
        assert fallback["farmer"] == ["PASS"]
        assert fallback["hands"] == [["PASS"]]
    with pytest.raises(RuntimeError, match="feature schema drifted"):
        module.agent({**obs, "step": 99})


def test_numpy_and_torch_networks_agree():
    module = net_module.build()
    network = net_module.to_numpy(module)
    features = np.random.default_rng(0).standard_normal((32, MARKET_FEATURE_DIM)).astype(np.float32)
    with torch.no_grad():
        expected = module(torch.from_numpy(features)).reshape(-1).numpy()
    assert np.allclose(network(features), expected, atol=1e-5)


def test_saved_weights_reject_a_stale_feature_schema(tmp_path):
    path = tmp_path / "stale.npz"
    MarketQNetwork.zeros(feature_dim=MARKET_FEATURE_DIM - 1).save(path)
    with pytest.raises(ValueError, match="feature dimension"):
        MarketQNetwork.load(path)


# --------------------------------------------------------------------------
# Rollout, storage, replay
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rollout(tmp_path_factory):
    root = tmp_path_factory.mktemp("market-rollout")
    weights = root / "behavior.npz"
    MarketQNetwork.random(seed=1).save(weights)
    result = collect_episode(
        weights,
        opponent_spec="starter",
        seed=3,
        epsilon=0.3,
        gamma=GAMMA,
        trace_root=root,
    )
    assert result is not None
    return result


def test_rollout_records_two_decisions_per_day(rollout):
    record = rollout.record
    assert record.steps == 719
    # Hour 0 and hour 12 of each of 30 days, minus the trailing partial day the
    # episode ends inside.
    assert 55 <= record.decisions <= 60
    assert np.all(np.diff(record.market_step) > 0)
    assert record.market_chosen.shape == (record.decisions, 1 + len(PRODUCTS))


def test_shaping_is_potential_based_and_terminal_potential_is_zero(rollout):
    record = rollout.record
    assert record.potential[-1] == 0.0
    expected = record.raw_reward + GAMMA * record.potential[1:] - record.potential[:-1]
    terminal = record.reward[-1] - expected[-1]
    assert np.allclose(record.reward[:-1], expected[:-1], atol=1e-4)
    assert abs(terminal) <= 1.0 + 1e-6
    assert np.allclose(record.raw_reward, np.diff(record.money) / MONEY_SCALE, atol=1e-4)
    assert TERMINAL_MARGIN_SCALE > 0


def test_episode_batches_round_trip(tmp_path, rollout):
    path = tmp_path / "batch.npz"
    save_episode_batch([rollout.record], path)
    restored = load_episode_batch(path)[0]
    for name in EPISODE_ARRAY_FIELDS:
        assert np.array_equal(getattr(restored, name), getattr(rollout.record, name))


def test_replay_samples_the_recorded_factor_decision(rollout):
    buffer = ReplayBuffer(capacity=8, gamma=GAMMA, n_step=3, seed=0)
    buffer.add([rollout.record])
    batch = buffer.sample(16)
    factors = 1 + len(PRODUCTS)
    assert batch.features.shape == (16, factors, MARKET_FEATURE_DIM)
    assert batch.next_features.shape[:2] == (16, factors)
    assert batch.next_features.shape[-1] == MARKET_FEATURE_DIM
    assert batch.next_mask.shape == batch.next_features.shape[:3]
    # A bootstrap that has no candidates must carry no discount, or the target
    # picks up value from a state the episode never reached.
    terminal = ~batch.next_mask.any(axis=2).any(axis=1)
    assert np.all(batch.discount[terminal] == 0.0)
    assert np.all(batch.discount[~terminal] > 0.0)


def _synthetic_record(rollout, *, steps: int, decision_steps: list[int]):
    """A record with a known reward stream, borrowing real candidate arrays."""

    from dataclasses import replace as dataclass_replace

    template = rollout.record
    count = len(decision_steps)
    return dataclass_replace(
        template,
        reward=np.ones(steps, dtype=np.float32),
        raw_reward=np.ones(steps, dtype=np.float32),
        potential=np.zeros(steps + 1, dtype=np.float32),
        money=np.zeros(steps + 1, dtype=np.float32),
        market_state=template.market_state[:count].copy(),
        market_purchase_static=template.market_purchase_static[:count].copy(),
        market_purchase_mask=template.market_purchase_mask[:count].copy(),
        market_reserve_static=template.market_reserve_static[:count].copy(),
        market_reserve_mask=template.market_reserve_mask[:count].copy(),
        market_chosen=template.market_chosen[:count].copy(),
        market_step=np.asarray(decision_steps, dtype=np.int16),
    )


def test_replay_reward_window_spans_the_turns_between_decisions(rollout):
    """The market MDP is day-scale.

    A decision is credited with every per-turn reward until the next decision
    (or the end of the episode), discounted by elapsed *turns* -- not by
    decisions. Getting this wrong is silent: the loss still falls, the agent
    just learns the wrong horizon.
    """

    steps, decision_steps = 40, [0, 12, 24, 36]
    record = _synthetic_record(rollout, steps=steps, decision_steps=decision_steps)
    buffer = ReplayBuffer(capacity=8, gamma=GAMMA, n_step=3, seed=0)
    buffer.add([record])
    batch = buffer.sample(256)

    def window(start: int, end: int) -> float:
        return float(sum(GAMMA**offset for offset in range(end - start)))

    # n_step = 3 decisions, so each decision bootstraps three decisions later --
    # which for the last two runs off the end and terminates at `steps`.
    expected = np.array(
        [
            window(0, 36),  # decision 0 -> decision 3
            window(12, 40),  # decision 1 -> end of episode
            window(24, 40),
            window(36, 40),
        ]
    )
    observed = np.unique(batch.reward)
    assert observed.size == expected.size
    assert np.allclose(np.sort(observed), np.sort(expected), atol=1e-3)

    # Only the decision that bootstraps within the episode carries a discount,
    # and it is gamma raised to the elapsed turns.
    non_terminal = batch.discount > 0.0
    assert np.allclose(batch.discount[non_terminal], GAMMA**36, atol=1e-6)
    assert np.allclose(batch.reward[non_terminal], window(0, 36), atol=1e-3)
