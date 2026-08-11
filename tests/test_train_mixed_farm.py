import train_mixed_farm
from kaggriculture_lab.constants import ARTIFACTS_ROOT


def test_effective_mixed_farm_config_uses_named_training_instance():
    config = train_mixed_farm._effective_config()
    config.validate()

    assert config.name == "configurable-mixed-farm-v5-training"
    assert config.manifest == "agents/mixed_farm_v5/agent.toml"
    assert config.training_manifests == (
        "agents/mixed_farm_v5/agent.toml",
    )
    assert [
        (item.type, item.name, item.primary, item.train)
        for item in config.training_agents
    ] == [("mixed_farm_v5", "v5", True, True)]
    assert config.population_size == train_mixed_farm.POPULATION_SIZE
    assert config.rounds_per_generation == train_mixed_farm.ROUNDS_PER_GENERATION
    assert config.anchor_rounds == train_mixed_farm.ANCHOR_ROUNDS
    assert len(config.anchor_manifests) == 12
    # V5 restarts from the bounded prior; inheriting v3/v4 champions is gone.
    assert config.seed_session_ids == ()
    assert config.episode_steps == train_mixed_farm.EPISODE_STEPS


def test_mixed_farm_runner_uses_shared_artifact_root():
    assert train_mixed_farm.LeagueConfig.from_toml(train_mixed_farm.LEAGUE_CONFIG).league == "mixed_farm_v5"
    assert train_mixed_farm.ARTIFACTS == ARTIFACTS_ROOT
