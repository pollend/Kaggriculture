from pathlib import Path

import train_dqn


def test_effective_dqn_config_builds_joint_self_play_roster(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(train_dqn, "ARTIFACTS", tmp_path)
    config = train_dqn._effective_config()
    config.validate()

    assert config.anchor_manifests == ()
    assert config.training_manifests == ("agents/dqn_job_scorer/agent.toml",)
    assert [(item.type, item.name, item.train) for item in config.training_agents] == [
        ("dqn_job_scorer", "default", True),
        ("dqn_job_scorer", "peer_a", True),
        ("dqn_job_scorer", "peer_b", True),
    ]
    for item in config.training_agents:
        assert item.model_config == train_dqn.DQN_SELF_PLAY_CONFIG
    assert config.population_size == 2
    assert config.rounds_per_generation == 20
    assert config.episode_steps == 1


def test_dqn_runner_uses_an_isolated_store_and_global_device():
    assert train_dqn.DATABASE == Path("artifacts/dqn_self_play_experiments.duckdb")
    assert train_dqn.ARTIFACTS == Path("artifacts/dqn_self_play")
    assert train_dqn.FROZEN_DQN.endswith("?network_seed=0")
    assert train_dqn._device() == "auto"
