import inspect
from pathlib import Path

import train_rf_market
from agents.mixed_farm_rf_market import training
from agents.mixed_farm_rf_market.strategy import Strategy
from kaggriculture_lab.identity import AgentManifest


def test_rf_market_bootstraps_itself_without_another_family():
    """The experiment used to read a champion out of a v3/v4 mixed-farm session.

    Those sessions and that agent family were retired, which left the runner dead on
    a bootstrap guard. Nothing here may depend on another family existing.
    """

    assert training.FROZEN_STRATEGY == Strategy()
    assert training.HEURISTIC_SPEC == training.SELF_SPEC
    assert not hasattr(training, "BASE_FAMILY_ID")
    assert not hasattr(training, "_champion")

    # Warm-start collection plays the game; it does not read the database.
    assert "reader" not in inspect.signature(training.bootstrap_dataset).parameters

    manifest = AgentManifest.load("agents/mixed_farm_rf_market/agent.toml")
    assert "bootstrap_sessions" not in manifest.models[0].config

    source = Path("train_rf_market.py").read_text(encoding="utf-8")
    assert "BOOTSTRAP_SESSION" not in source


def test_rf_market_runner_is_trainer_only_and_uses_its_own_family():
    config = train_rf_market._effective_config()
    assert config.pairing_mode == "trainer_only"
    assert config.population_size == 0
    assert config.training_agents[0].type == "mixed_farm_rf_market"
    assert config.training_agents[0].primary
    assert Path(config.manifest) == Path("agents/mixed_farm_rf_market/agent.toml")
