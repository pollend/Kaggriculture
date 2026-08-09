from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from agents.dqn_job_scorer.market_policy import MARKET_FEATURE_DIM
from agents.dqn_job_scorer.policy import PolicyBundle, QNetwork
from agents.sp_psro_job_scorer.policy import PopulationArtifact
from agents.sp_psro_job_scorer.training import collect
from agents.sp_psro_job_scorer.training.updater import (
    _update_restricted_distribution,
    initial_state,
    load_state,
    mixture_probabilities,
    passes_release_gate,
    population_networks,
)
from kaggriculture_lab.identity import AgentManifest
from kaggriculture_lab.league import LeagueConfig, create_session
from kaggriculture_lab.model_trainers import RoundContext, TrainingBatchWriter
from kaggriculture_lab.model_updates import ActiveCheckpoint, ModelUpdateContext, TrainingBatch
from kaggriculture_lab.storage import ExperimentStore


def test_manifest_declares_new_training_family():
    manifest = AgentManifest.load("agents/sp_psro_job_scorer/agent.toml")
    assert manifest.family_id == "sp_psro_job_scorer"
    assert manifest.entry_point == "agents.sp_psro_job_scorer.main:agent"
    assert manifest.models[0].model_id == "sp_psro_job_scorer.joint_market_population_v3"
    assert manifest.models[0].config["population_size"] == 10


def test_cli_model_default_tracks_manifest():
    from agents.sp_psro_job_scorer.cli import _default_model_id

    assert _default_model_id() == "sp_psro_job_scorer.joint_market_population_v3"


def test_sp_psro_league_is_trainer_only_and_has_no_surrogate_population(tmp_path):
    config = LeagueConfig.from_toml("configs/sp_psro_joint_league.toml")
    assert config.pairing_mode == "trainer_only"
    assert config.manifest == "agents/sp_psro_job_scorer/agent.toml"
    create_session(config, tmp_path / "sp.duckdb", session_id="sp")
    with ExperimentStore(tmp_path / "sp.duckdb") as store:
        assert store.connection.execute(
            "SELECT count(*) FROM generation_members"
        ).fetchone()[0] == 0
        assert store.connection.execute(
            "SELECT count(*) FROM agent_families WHERE family_id = ?",
            ["configurable_crop_surrogate"],
        ).fetchone()[0] == 0


def test_initial_state_has_one_virtual_member_and_two_live_learners():
    state = initial_state(17)
    members = population_networks(state)
    assert [member_id for member_id, _ in members] == ["member-000000-initial"]
    assert set(state["live"]) == {"nu", "beta"}
    nu = state["live"]["nu"]["online"]
    beta = state["live"]["beta"]["online"]
    assert any(not np.array_equal(nu[key].numpy(), beta[key].numpy()) for key in nu)
    assert "market_online" in state["live"]["nu"]
    assert members[0][1].market is not None


def test_initial_state_can_bootstrap_from_joint_weights(tmp_path: Path):
    anchor = tmp_path / "anchor.npz"
    expected = PolicyBundle.random(seed=41)
    expected.save(anchor)
    state = initial_state(17, anchor)
    loaded = population_networks(state)[0][1]
    assert np.array_equal(loaded.job.layers[0][0], expected.job.layers[0][0])
    assert np.array_equal(loaded.market.layers[0][0], expected.market.layers[0][0])
    assert state["initial_source"] == str(anchor)


def test_release_gate_requires_profit_win_rate_and_margin():
    config = {
        "release_min_money": 3000,
        "release_min_win_rate": 0.5,
        "release_min_margin": 0,
    }
    good = {"mean_money": 3200, "mean_win_rate": 0.6, "mean_margin": 50, "errors": 0}
    assert passes_release_gate(good, config)
    assert not passes_release_gate({**good, "mean_money": 2999}, config)
    assert not passes_release_gate({**good, "mean_win_rate": 0.49}, config)
    assert not passes_release_gate({**good, "mean_margin": -1}, config)
    assert not passes_release_gate({**good, "errors": 1}, config)


def test_v3_state_migrates_without_reinitializing_live_learners(tmp_path: Path):
    state = initial_state(9)
    expected = state["live"]["nu"]["online"]["0.weight"].clone()
    state["version"] = 3
    for key in (
        "restricted",
        "nu_snapshots",
        "nu_snapshots_seen",
        "nu_snapshot_rng_state",
        "validation_suite",
    ):
        state.pop(key)
    path = tmp_path / "v3.pt"
    torch.save(state, path)
    migrated = load_state(path, 123)
    assert migrated["version"] == 4
    assert torch.equal(migrated["live"]["nu"]["online"]["0.weight"], expected)
    assert migrated["nu_snapshots"] == []


def test_multiplicative_weights_favors_winning_sampled_arm():
    state = initial_state(3)
    config = {"meta_exploration": 0.1, "meta_learning_rate": 0.2}
    arms, before = mixture_probabilities(state, config)
    records = [
        SimpleNamespace(
            behavior_member_id="live-nu",
            seed=11,
            population_payoff=1.0,
            sampling_probability=before[arms.index("live-nu")],
        )
        for _ in range(2)
    ]
    _update_restricted_distribution(state, records, config)
    _, after = mixture_probabilities(state, config)
    assert after[arms.index("live-nu")] > before[arms.index("live-nu")]


def test_population_artifact_round_trips_seeded_snapshot_mixture(tmp_path: Path):
    artifact = PopulationArtifact(
        (PolicyBundle.random(seed=1), PolicyBundle.random(seed=2)),
        np.asarray([0.25, 0.75]),
    )
    path = tmp_path / "mixture.npz"
    artifact.save(path)
    loaded = PopulationArtifact.load(path)
    assert len(loaded.components) == 2
    assert np.allclose(loaded.weights, [0.25, 0.75])
    first = loaded.choose(np.random.default_rng(77)).job.layers[0][0]
    second = loaded.choose(np.random.default_rng(77)).job.layers[0][0]
    assert np.array_equal(first, second)


def test_collection_schedule_rotates_population_members_and_beta_seats(monkeypatch):
    calls = []

    def fake_episode(*args):
        calls.append(args)
        return object()

    monkeypatch.setattr(collect, "collect_episode", fake_episode)
    layers = QNetwork.random(seed=1).layers
    records = collect.collect_batch(
        [("a", layers), ("b", layers)],
        layers,
        layers,
        list(range(8)),
        0.2,
        0.99,
        jobs=1,
    )
    assert len(records) == 8
    assert [call[2] for call in calls] == ["population"] * 8
    assert [call[3] for call in calls] == ["a", "a", "b", "b", "a", "a", "b", "b"]
    assert [call[7] for call in calls] == [0, 1, 0, 1, 0, 1, 0, 1]


def test_collection_requires_even_batch():
    layers = QNetwork.random(seed=1).layers
    try:
        collect.collect_batch([("a", layers)], layers, layers, [1, 2, 3], 0.0, 0.99)
    except ValueError as error:
        assert "even" in str(error)
    else:
        raise AssertionError("odd batch was accepted")


def test_collection_can_force_live_nu_from_weighted_distribution(monkeypatch):
    calls = []

    def fake_episode(*args):
        calls.append(args)
        return object()

    monkeypatch.setattr(collect, "collect_episode", fake_episode)
    layers = QNetwork.random(seed=1).layers
    records = collect.collect_batch(
        [("member", layers)],
        layers,
        layers,
        [5, 5],
        0.2,
        0.99,
        mixture_weights=[0.0, 1.0],
    )
    assert len(records) == 2
    assert [call[2] for call in calls] == ["new_policy", "new_policy"]
    assert [call[3] for call in calls] == ["live-nu", "live-nu"]


def test_updater_advances_both_live_learners_and_writes_resumable_state(
    tmp_path: Path, monkeypatch
):
    from agents.sp_psro_job_scorer.training import updater

    population_path = tmp_path / "population.npz"
    nu_path = tmp_path / "nu.npz"
    beta_path = tmp_path / "beta.npz"
    PolicyBundle.random(seed=1).save(population_path)
    PolicyBundle.random(seed=2).save(nu_path)
    PolicyBundle.random(seed=3).save(beta_path)
    records = collect.collect_batch(
        [("member-a", population_path)],
        nu_path,
        beta_path,
        [19, 19],
        0.0,
        0.997,
        config={"episodeSteps": 2, "trace_root": str(tmp_path / "traces")},
        mixture_weights=[0.5, 0.5],
        sampling_seed=4,
    )
    batch_path = collect.save_batch(records, tmp_path / "batch.npz")
    batch = TrainingBatch(
        "model",
        1,
        "batch-1",
        "sp_psro_rollout",
        "source-1",
        str(batch_path),
        collect.digest(batch_path),
        len(records),
        {},
    )

    calls = []

    def fake_train(role_state, buffer, config, steps, device):
        calls.append((buffer.transitions, buffer.market_transitions, steps))
        copied = dict(role_state)
        copied["gradient_steps"] = int(role_state["gradient_steps"]) + steps
        return copied, [float(len(calls))], [float(len(calls))]

    monkeypatch.setattr(updater, "_train_role", fake_train)
    monkeypatch.setattr(
        updater,
        "evaluate_admission",
        lambda *args, **kwargs: {
            "pairs": 2,
            "errors": 0,
            "wins": 1,
            "losses": 1,
            "margin": 0.0,
            "margin_ci": [-1.0, 1.0],
        },
    )
    monkeypatch.setattr(
        updater,
        "evaluate_member",
        lambda path, *args, **kwargs: {
            "opponents": [],
            "mean_win_rate": 0.6 if "candidate" in Path(path).name else 0.1,
            "mean_margin": 100.0 if "candidate" in Path(path).name else -100.0,
            "mean_money": 3400.0 if "candidate" in Path(path).name else 1000.0,
            "errors": 0,
        },
    )
    context = ModelUpdateContext(
        "model",
        "sp_psro_job_scorer",
        "update-1",
        1,
        (batch,),
        (batch,),
        None,
        {
            "warmup_episodes": 1,
            "gradient_steps_per_batch": 3,
            "buffer_episodes": 8,
            "admission_seeds": 2,
            "nu_average_snapshots": 2,
            "population_size": 4,
            "validation_opponents": ["starter"],
        },
        tmp_path / "artifacts",
        "dataset",
    )
    result = updater.PopulationUpdater().update(context, object())
    saved = torch.load(result.trainer_artifact_path, map_location="cpu", weights_only=False)
    assert len(calls) == 2
    assert saved["live"]["nu"]["gradient_steps"] == 3
    assert saved["live"]["beta"]["gradient_steps"] == 3
    assert len(saved["nu_snapshots"]) == 1
    assert result.metrics["nu_loss"] == 1.0
    assert result.metrics["beta_loss"] == 2.0
    assert result.gate["passed"] is True
    assert saved["inference_champion"]["candidate_id"] == "beta-00000001"

    monkeypatch.setattr(
        updater,
        "evaluate_admission",
        lambda *args, **kwargs: {
            "pairs": 2,
            "errors": 0,
            "wins": 2,
            "losses": 0,
            "margin": 2.0,
            "margin_ci": [1.0, 3.0],
        },
    )
    monkeypatch.setattr(
        updater,
        "evaluate_member",
        lambda *args, **kwargs: {
            "opponents": [],
            "mean_win_rate": 0.5,
            "mean_margin": 0.0,
            "mean_money": 3000.0,
            "errors": 0,
        },
    )
    batch2 = TrainingBatch(
        "model",
        2,
        "batch-2",
        "sp_psro_rollout",
        "source-2",
        str(batch_path),
        collect.digest(batch_path),
        len(records),
        {},
    )
    accepted_context = ModelUpdateContext(
        "model",
        "sp_psro_job_scorer",
        "update-2",
        2,
        (batch2,),
        (batch, batch2),
        ActiveCheckpoint(
            result.checkpoint_id,
            result.trainer_artifact_path,
            result.inference_artifact_path,
            result.artifact_sha256,
            result.metrics,
        ),
        context.config,
        context.artifacts,
        "dataset-2",
    )
    accepted = updater.PopulationUpdater().update(accepted_context, object())
    accepted_state = torch.load(
        accepted.trainer_artifact_path, map_location="cpu", weights_only=False
    )
    assert accepted_state["iteration"] == 1
    assert [member["role"] for member in accepted_state["population"][-2:]] == [
        "beta",
        "nu-average",
    ]
    assert accepted_state["nu_snapshots"] == []
    assert accepted_state["live"]["nu"]["gradient_steps"] == 0
    assert accepted_state["live"]["beta"]["gradient_steps"] == 0
    assert accepted.gate["passed"] is True
    assert accepted_state["inference_champion"] is not None


def test_inference_agent_plays_a_legal_action():
    from kaggle_environments import make

    from sim.registry import resolve

    env = make("kaggriculture", configuration={"episodeSteps": 2}, debug=True)
    env.run([resolve("agents/sp_psro_job_scorer/main.py"), env.agents["pass"]])
    assert env.steps[-1][0].status == "DONE"


def test_validation_pool_combines_frozen_and_active_dqns(monkeypatch):
    from agents.sp_psro_job_scorer.training import updater

    monkeypatch.setattr(
        updater,
        "active_checkpoint_opponents",
        lambda context, reader, types: ["active-a", "frozen", "active-b"],
    )
    opponents = updater.validation_opponent_specs(
        object(),
        object(),
        {
            "validation_opponents": ["frozen"],
            "validation_peer_types": ["dqn_job_scorer"],
        },
    )
    assert opponents == ["frozen", "active-a", "active-b"]


def test_manifest_rollouts_are_persisted_as_paired_episode_artifacts(tmp_path: Path):
    population_path = tmp_path / "population.npz"
    beta_path = tmp_path / "beta.npz"
    PolicyBundle.random(seed=1).save(population_path)
    PolicyBundle.random(seed=2).save(beta_path)
    rollouts = collect.collect_batch(
        [("member-a", population_path)],
        beta_path,
        beta_path,
        [11, 11],
        0.0,
        0.997,
        jobs=1,
        config={"episodeSteps": 2, "trace_root": str(tmp_path / "traces")},
    )

    with ExperimentStore(tmp_path / "ledger.duckdb") as store:
        store.connection.execute(
            "INSERT INTO training_sessions VALUES (?, ?, ?, ?, ?, current_timestamp)",
            ["session", "session", "RUNNING", 1, "{}"],
        )
        store.connection.execute(
            "INSERT INTO generations (generation_id, session_id, generation_index, status) VALUES (?, ?, ?, ?)",
            ["session:g0000", "session", 0, "RUNNING"],
        )
        writer = TrainingBatchWriter(store, "model")
        context = RoundContext(
            "model", "sp_psro_match_policy_v3", "round", "session",
            "session:g0000", 0, 0, None, {}, tmp_path,
        )
        weighted_id = writer.record_policy_mixture(
            context, ["member-a", "live-nu"], max_members=2, weights=[0.8, 0.2]
        )
        assert store.connection.execute(
            "SELECT revision_id, weight FROM policy_mixture_members "
            "WHERE mixture_id = ? ORDER BY ordinal",
            [weighted_id],
        ).fetchall() == [("member-a", 0.8), ("live-nu", 0.2)]
        mixture_id = writer.record_policy_mixture(context, ["member-a"], max_members=10)
        for rollout in rollouts:
            writer.record_policy_episode(
                context=context,
                manifest_path="agents/sp_psro_job_scorer/match_agent.toml",
                pairing_index=0,
                beta_seat=rollout.beta_seat,
                population_revision_id="member-a",
                beta_revision_id="beta-a",
                mixture_id=mixture_id,
                result=rollout.match,
            )
        assert store.connection.execute(
            "SELECT status FROM league_pairings WHERE pairing_kind = 'sp_psro_training'"
        ).fetchone()[0] == "COMPLETE"
        assert store.connection.execute(
            "SELECT count(*) FROM episode_training_artifacts"
        ).fetchone()[0] == 4
        assert store.connection.execute(
            "SELECT count(*) FROM episode_payloads"
        ).fetchone()[0] == 2
        assert store.connection.execute(
            "SELECT count(*) FROM pairing_policy_assignments"
        ).fetchone()[0] == 1
    assert rollouts[0].best_response.market_state is not None
    assert rollouts[0].best_response.market_state.shape[1] + 30 == MARKET_FEATURE_DIM
