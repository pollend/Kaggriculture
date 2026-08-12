from pathlib import Path

import train_dqn
from kaggriculture_lab.constants import ARTIFACTS_ROOT


def test_effective_dqn_config_builds_a_single_learner_roster(tmp_path: Path, monkeypatch):
    """One learner, trainer-only. The league schedules the clock, not a population."""
    monkeypatch.setattr(train_dqn, "ARTIFACTS", tmp_path)
    config = train_dqn._effective_config()
    config.validate()

    assert config.anchor_manifests == ()
    assert config.training_manifests == ("agents/dqn_job_scorer/agent.toml",)
    assert [(item.type, item.name, item.train) for item in config.training_agents] == [
        ("dqn_job_scorer", "default", True),
    ]
    assert config.pairing_mode == "trainer_only"
    assert config.population_size == 0
    assert config.rounds_per_generation == 100
    assert config.episode_steps == 720


def test_the_runner_overrides_measurement_and_leaves_the_curriculum_to_the_manifest():
    """The opponent mix belongs to `agent.toml`.

    Overriding it here is what put dqn-self-play-v1 on 12 of every 16 episodes
    against frozen snapshots and 4 against `starter`, while promotion was decided
    against `configurable_crop_surrogate` -- an opponent the learner never played
    -- for seventeen consecutive rejected updates.
    """
    config = train_dqn.SELF_PLAY_CONFIG

    assert config["benchmarks"] == list(train_dqn.BENCHMARKS)
    assert config["benchmark"] == "agents/configurable_crop_surrogate/main.py"
    for curriculum_key in (
        "opponents",
        "league_opponents",
        "self_play_after",
        "self_play_ratio",
    ):
        assert curriculum_key not in config, curriculum_key


def test_dqn_runner_uses_an_isolated_artifact_root_and_global_device():
    assert train_dqn.LeagueConfig.from_toml(train_dqn.LEAGUE_CONFIG).league == "dqn_self_play"
    assert train_dqn.ARTIFACTS == ARTIFACTS_ROOT / "dqn_self_play"
    assert train_dqn._device() == "auto"
