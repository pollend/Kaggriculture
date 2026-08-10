import subprocess
from pathlib import Path

import pytest

import train_mixed_farm


def test_effective_mixed_farm_config_uses_named_training_instance():
    config = train_mixed_farm._effective_config()
    config.validate()

    assert config.name == "configurable-mixed-farm-v3-training"
    assert config.manifest == "agents/configurable_mixed_farm_surrogate/agent.toml"
    assert config.training_manifests == (
        "agents/configurable_mixed_farm_surrogate/agent.toml",
    )
    assert [
        (item.type, item.name, item.primary, item.train)
        for item in config.training_agents
    ] == [("configurable_mixed_farm_surrogate", "v3", True, True)]
    assert config.population_size == train_mixed_farm.POPULATION_SIZE
    assert config.rounds_per_generation == train_mixed_farm.ROUNDS_PER_GENERATION
    assert config.episode_steps == train_mixed_farm.EPISODE_STEPS


def test_mixed_farm_runner_uses_shared_artifact_root():
    assert train_mixed_farm.LeagueConfig.from_toml(train_mixed_farm.LEAGUE_CONFIG).league == "mixed_farm_v3"
    assert train_mixed_farm.ARTIFACTS == Path("artifacts")


def test_mixed_farm_acceptance_failure_propagates(monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(train_mixed_farm, "_run", fail)

    with pytest.raises(subprocess.CalledProcessError):
        train_mixed_farm._run_acceptance()
