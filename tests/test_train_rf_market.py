from pathlib import Path

import train_rf_market


def test_rf_market_runner_is_trainer_only_and_uses_its_own_family():
    config = train_rf_market._effective_config()
    assert config.pairing_mode == "trainer_only"
    assert config.population_size == 0
    assert config.training_agents[0].type == "mixed_farm_rf_market"
    assert config.training_agents[0].primary
    assert Path(config.manifest) == Path("agents/mixed_farm_rf_market/agent.toml")
