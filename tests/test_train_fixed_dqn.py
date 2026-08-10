import json
from pathlib import Path

import numpy as np
import pytest

import train_fixed_dqn
import train_sp_psro
from agents.dqn_job_scorer.policy import PolicyBundle
from agents.dqn_job_scorer.training.updater import initial_bundle
from kaggriculture_lab.league import create_session
from kaggriculture_lab.storage import ExperimentStore
from tests.postgres_support import postgres_database


def test_fixed_dqn_config_has_one_fixed_curriculum_learner():
    config = train_fixed_dqn._effective_config()
    config.validate()

    assert config.pairing_mode == "trainer_only"
    assert config.population_size == 0
    assert config.training_manifests == ("agents/dqn_job_scorer/agent.toml",)
    assert len(config.training_agents) == 1
    learner = config.training_agents[0]
    assert (learner.type, learner.name, learner.train, learner.primary) == (
        "dqn_job_scorer",
        "fixed",
        True,
        True,
    )
    model = learner.model_config or {}
    assert model["opponents"] == list(train_fixed_dqn.FIXED_OPPONENTS)
    assert model["league_opponents"] is False
    assert model["self_play_ratio"] == 0.0
    assert model["initialization"] == "incumbent"
    assert model["incumbent_weights"] == str(train_fixed_dqn.BOOTSTRAP_WEIGHTS)


def test_fixed_dqn_session_registers_the_fixed_model_without_population(tmp_path: Path):
    database = postgres_database(tmp_path, "fixed")
    create_session(
        train_fixed_dqn._effective_config(),
        database,
        workspace=train_fixed_dqn.REPO_ROOT,
        session_id="fixed-test",
    )

    with ExperimentStore(database) as store:
        row = store.connection.execute(
            "SELECT model_id, config_json FROM training_models WHERE agent_name = 'fixed'"
        ).fetchone()
        members = store.connection.execute(
            "SELECT count(*) FROM generation_members"
        ).fetchone()[0]

    assert row is not None
    assert row[0] == "dqn_job_scorer.adaptive_hybrid_q_v2@fixed"
    registered = json.loads(row[1])
    assert registered["opponents"] == list(train_fixed_dqn.FIXED_OPPONENTS)
    assert registered["self_play_ratio"] == 0.0
    assert members == 0


def test_incumbent_initialization_loads_joint_policy(tmp_path: Path):
    expected = PolicyBundle.random(seed=71)
    path = tmp_path / "incumbent.npz"
    expected.save(path)

    loaded = initial_bundle(
        {"initialization": "incumbent", "incumbent_weights": str(path), "seed": 9}
    )

    assert loaded.market is not None
    assert np.array_equal(loaded.job.layers[0][0], expected.job.layers[0][0])
    assert np.array_equal(loaded.market.layers[0][0], expected.market.layers[0][0])


def test_incumbent_initialization_rejects_missing_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="incumbent checkpoint"):
        initial_bundle(
            {
                "initialization": "incumbent",
                "incumbent_weights": str(tmp_path / "missing.npz"),
            }
        )


def test_sp_psro_command_temporarily_routes_to_fixed_dqn():
    assert train_sp_psro.main is train_fixed_dqn.main
