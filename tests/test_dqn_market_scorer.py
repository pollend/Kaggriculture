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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.dqn_market_scorer import donor, market_exec
from agents.dqn_market_scorer.evaluation import evaluate_episode
from agents.dqn_market_scorer.main import MarketScorerAgent
from agents.dqn_market_scorer.market_policy import (
    DEFAULT_MODIFIER,
    MARKET_ACTION_DIM,
    MARKET_FEATURE_DIM,
    MARKET_STATE_DIM,
    MODIFIERS,
    POLICY_KINDS,
    PRODUCTS,
    MarketModifier,
    choose_market,
    enumerate_market_candidates,
    market_state_features,
    modifier_action_features,
    reserve_action_features,
)
from agents.dqn_market_scorer.policy import WEIGHTS_PATH, MarketQNetwork
from agents.dqn_market_scorer.training import network as net_module
from agents.dqn_market_scorer.training.collect import (
    EPISODE_ARRAY_FIELDS,
    MONEY_SCALE,
    TERMINAL_MARGIN_SCALE,
    collect_episode,
    load_episode_batch,
    save_episode_batch,
)
from agents.dqn_market_scorer.training.replay import ReplayBuffer
from agents.dqn_market_scorer.training.updater import (
    _benchmark_kind,
    _has_inert_hidden_trunk,
    _materialize_behavior_policy,
    _reject_degenerate,
    _reject_inert_network,
    initial_network,
    policy_improvement,
)

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


def test_donor_is_a_declared_derivative_of_the_licensed_snapshot():
    assert _digest(DONOR_PATH) != _digest(SNAPSHOT_PATH)


def test_donor_digest_matches_the_recorded_provenance():
    notice = (REPO_ROOT / "agents/dqn_market_scorer/NOTICE").read_text()
    assert _digest(SNAPSHOT_PATH) in notice
    assert _digest(DONOR_PATH) in notice
    assert "donor.py IS A MODIFIED DERIVATIVE" in notice
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


def test_q_guidance_gates_only_new_field_expansion(live_states):
    obs, _config, farm, _private, _roles, _field = live_states[0]
    default_roles = donor._role_plan(obs, farm, DEFAULT_MODIFIER)
    blocked = donor._role_plan(
        obs,
        farm,
        MarketModifier(seed_scale=0.0, animal_gate=0),
    )
    half_crops = donor._role_plan(
        obs,
        farm,
        MarketModifier(seed_scale=0.5, animal_gate=1),
    )

    assert len(blocked) < len(default_roles)
    default_crops = sum(kind == "CROP" for kind, _item in default_roles.values())
    half_crop_count = sum(kind == "CROP" for kind, _item in half_crops.values())
    assert 0 < half_crop_count < default_crops

    for (x, y), (kind, _item) in blocked.items():
        tile = farm["tiles"][y][x]
        if kind == "CROP":
            assert isinstance(tile, dict) and tile.get("kind") == "PLANT"
        elif kind == "ANIMAL":
            assert isinstance(tile, dict) and tile.get("animal") in donor.ANIMALS


def test_q_modifier_replans_the_donor_field_on_the_same_turn(live_states):
    class BlockExpansionNetwork:
        def __call__(self, features):
            rows = np.asarray(features, dtype=np.float32)
            actions = rows[..., MARKET_STATE_DIM:]
            modifier_rows = actions[..., 0] > 0.5
            # Among modifier rows, prefer seed_scale=0 and animal_gate=0.
            scores = np.where(
                modifier_rows,
                10.0 - actions[..., 2] - actions[..., 4],
                0.0,
            )
            return scores.reshape(-1)

    obs, config, farm, private, _roles, _field = live_states[0]
    agent = MarketScorerAgent(BlockExpansionNetwork())
    action = agent(copy.deepcopy(obs), config)

    assert agent._modifier.seed_scale == 0.0
    assert agent._modifier.animal_gate == 0
    guided_roles = donor._role_plan(obs, farm, agent._modifier)
    expected = donor._unit_actions(obs, config, farm, private, guided_roles)
    assert action["farmer"] == expected["farmer"]
    assert action["hands"] == expected["hands"]


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


@pytest.mark.slow
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


def test_training_initialization_starts_at_the_donor_default(live_states):
    """The trainable cold start must still score every candidate identically.

    That tie is what makes round one roll out unmodified pilkwang instead of a
    random market policy, and it is the property the all-zero network was
    originally chosen for.
    """

    obs, config, farm, private, roles, field = live_states[1]
    candidates = enumerate_market_candidates(obs, farm, private, roles, field, config=config)
    decision = choose_market(candidates, initial_network({"initialization": "donor_default"}))
    assert decision.modifier == DEFAULT_MODIFIER
    assert set(decision.reserves.values()) == {0}


def test_the_legacy_zeros_spelling_resolves_to_the_trainable_cold_start():
    """`zeros` is a deprecated alias, never a literal all-zero MLP.

    Sessions and configs written before the rename still carry it, and resolving
    it to `MarketQNetwork.zeros()` would silently restore the untrainable run.
    """

    alias = initial_network({"initialization": "zeros", "seed": 5})
    renamed = initial_network({"initialization": "donor_default", "seed": 5})
    assert [
        (weight.tolist(), bias.tolist()) for weight, bias in alias.layers
    ] == [(weight.tolist(), bias.tolist()) for weight, bias in renamed.layers]
    assert not _has_inert_hidden_trunk(alias)


def test_training_initialization_leaves_every_layer_trainable():
    """Gradients must reach the hidden layers, not just the output bias.

    An all-zero MLP zeroes both hidden activations for any input, so
    `dL/dw` is zero for every layer and `relu'(0) == 0` closes the path back to
    the biases. Only the output bias moves, the symmetry never breaks, and the
    network stays byte-identical to the donor for the life of the run while the
    promotion gate reports a healthy checkpoint. This is the check that catches
    that.
    """

    module = net_module.from_numpy(
        initial_network({"initialization": "donor_default"}), net_module.build()
    )
    linears = [layer for layer in module if isinstance(layer, torch.nn.Linear)]
    before = [layer.weight.detach().clone() for layer in linears]

    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(64, MARKET_FEATURE_DIM, generator=generator)
    targets = torch.randn(64, 1, generator=generator) * 100.0
    for _ in range(20):
        loss = torch.nn.SmoothL1Loss()(module(features), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    for index, (layer, original) in enumerate(zip(linears, before)):
        assert not torch.equal(layer.weight.detach(), original), f"layer {index} never trained"


def test_an_inert_network_is_rejected_before_it_can_be_promoted():
    _reject_inert_network(initial_network({"initialization": "donor_default"}))
    assert not _has_inert_hidden_trunk(initial_network({"initialization": "donor_default"}))
    assert _has_inert_hidden_trunk(MarketQNetwork.zeros())
    with pytest.raises(RuntimeError, match="all-zero hidden weights"):
        _reject_inert_network(MarketQNetwork.zeros())


def test_a_fresh_run_replaces_a_cached_legacy_initial_policy(tmp_path):
    path = tmp_path / "rollout_policies" / "initial.npz"
    _materialize_behavior_policy(MarketQNetwork.zeros(), path)
    assert _has_inert_hidden_trunk(MarketQNetwork.load(path))

    replacement = initial_network({"initialization": "zeros", "seed": 73})
    _materialize_behavior_policy(replacement, path, refresh=True)

    restored = MarketQNetwork.load(path)
    assert not _has_inert_hidden_trunk(restored)
    assert np.all(restored(np.zeros((1, MARKET_FEATURE_DIM), dtype=np.float32)) == 0.0)


def test_a_zero_width_evaluation_interval_is_rejected():
    """Exact cancellation over many paired seeds is a fault, not a tie."""

    degenerate = {
        "opponent": "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py",
        "pairs": 100,
        "margin_ci": [0.0, 0.0],
        "errors": {},
    }
    with pytest.raises(RuntimeError, match="identically zero"):
        _reject_degenerate(degenerate, "benchmark")

    # A real interval that merely straddles zero is a legitimate "no better than
    # the donor yet" and must pass through.
    _reject_degenerate({**degenerate, "margin_ci": [-120.0, 340.0]}, "benchmark")
    # So must a single pair, where exact cancellation is not evidence of
    # anything, and an errored evaluation, which is reported by its own path.
    _reject_degenerate({**degenerate, "pairs": 1}, "benchmark")
    _reject_degenerate({**degenerate, "errors": {"seat0": "boom"}}, "benchmark")


def test_initial_release_requires_primary_benchmark_improvement():
    tied = {"errors": {}, "margin": 0.0, "margin_ci": [0.0, 0.0]}
    worse = {"errors": {}, "margin": -10.0, "margin_ci": [-20.0, -1.0]}
    noisy_gain = {"errors": {}, "margin": 10.0, "margin_ci": [-2.0, 22.0]}
    resolved_gain = {"errors": {}, "margin": 10.0, "margin_ci": [1.0, 19.0]}

    assert not policy_improvement(tied, "significant")
    assert not policy_improvement(worse, "significant")
    assert not policy_improvement(noisy_gain, "significant")
    assert policy_improvement(resolved_gain, "significant")
    assert policy_improvement(noisy_gain, "point_estimate")


def test_primary_evaluation_kind_is_not_a_generic_fixed_benchmark():
    assert _benchmark_kind(0) == POLICY_KINDS["promotion_baseline"]
    assert _benchmark_kind(1) == "fixed_benchmark:1"


def test_rollout_results_never_request_checkpoint_reevaluation():
    """A changing exploration opponent is not a checkpoint trend."""

    class Reader:
        def summary(self):
            return {
                "final_money": 1.0,
                "money_difference": -2_000.0,
                "win": 0.0,
                "metrics": {},
            }

        def fetchone(self, sql, parameters=()):
            assert "pairing_kind" in sql
            return ("market_rollout",)

        def history(self, revision_id, limit=50):
            # Newest first: a large apparent regression that would normally
            # exceed the evaluator threshold.
            margins = [-2_000.0] * 10 + [0.0] * 10
            return [
                {"failed": False, "money_difference": margin, "win": 0.0}
                for margin in margins
            ]

    result = evaluate_episode(
        SimpleNamespace(match_id="match", revision_id="checkpoint"), Reader()
    )
    assert result["margin_regression"] == 2_000.0
    assert result["pairing_kind"] == "market_rollout"
    assert not result["should_reevaluate"]
    assert result["reevaluate_reason"] == "training rollout"

    class ComparableReader(Reader):
        def fetchone(self, sql, parameters=()):
            assert "pairing_kind" in sql
            return ("fixed_benchmark",)

    comparable = evaluate_episode(
        SimpleNamespace(match_id="match", revision_id="checkpoint"), ComparableReader()
    )
    assert comparable["should_reevaluate"]
    assert comparable["reevaluate_reason"].startswith("mean margin fell 2,000")


def test_saved_weights_reject_a_stale_feature_schema(tmp_path):
    path = tmp_path / "stale.npz"
    MarketQNetwork.zeros(feature_dim=MARKET_FEATURE_DIM - 1).save(path)
    with pytest.raises(ValueError, match="does not match schema"):
        MarketQNetwork.load(path)


def test_saved_weights_reject_a_stale_output_schema(tmp_path):
    path = tmp_path / "stale-output.npz"
    MarketQNetwork.zeros().save(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["w2"] = np.zeros((128, 2), dtype=np.float32)
    payload["b2"] = np.zeros(2, dtype=np.float32)
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="does not match schema"):
        MarketQNetwork.load(path)


def test_packaged_weights_match_the_current_scorer_schema():
    network = MarketQNetwork.load(WEIGHTS_PATH)
    assert network.layers[0][0].shape == (MARKET_FEATURE_DIM, 128)
    assert network.layers[-1][0].shape == (128, 1)


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
