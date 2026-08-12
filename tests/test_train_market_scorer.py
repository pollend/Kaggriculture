from pathlib import Path

import train_market_scorer
from kaggriculture_lab.constants import ARTIFACTS_ROOT


def test_effective_market_config_builds_a_single_learner_roster(tmp_path: Path, monkeypatch):
    """One learner, trainer-only. The league schedules the clock, not a population."""
    monkeypatch.setattr(train_market_scorer, "ARTIFACTS", tmp_path)
    config = train_market_scorer._effective_config()
    config.validate()

    assert config.anchor_manifests == ()
    assert config.training_manifests == ("agents/dqn_market_scorer/agent.toml",)
    assert [(item.type, item.name, item.train) for item in config.training_agents] == [
        ("dqn_market_scorer", "default", True),
    ]
    assert config.pairing_mode == "trainer_only"
    assert config.population_size == 0
    # One update per generation: `rounds_per_generation` x `batches_per_round`
    # must match the manifest's `min_new_batches`.
    assert config.rounds_per_generation == 20
    assert config.episode_steps == 720


def test_promotion_is_measured_against_the_donor():
    """The benchmark is the frozen field policy this agent is built on.

    With the same field policy on both sides, the paired margin isolates the
    learned market layer. Any weaker benchmark lets the donor's field play carry
    the result, so a passing gate would stop being evidence about the thing
    being trained.
    """
    config = train_market_scorer.MARKET_CONFIG

    assert config["benchmark"] == train_market_scorer.DONOR
    assert config["benchmarks"][0] == train_market_scorer.DONOR
    assert "competitor_pilkwang" in train_market_scorer.DONOR


def test_the_runner_leaves_the_opponent_curriculum_to_the_manifest():
    config = train_market_scorer.MARKET_CONFIG
    for curriculum_key in (
        "opponents",
        "league_opponents",
        "self_play_after",
        "self_play_ratio",
    ):
        assert curriculum_key not in config, curriculum_key


def test_cold_start_is_the_donor_rather_than_a_random_market_policy():
    # A zero network scores every candidate equally, argmax breaks to index 0,
    # and index 0 is the donor-default modifier with a zero reserve.
    assert train_market_scorer.MARKET_CONFIG["initialization"] == "zeros"


def test_market_runner_uses_an_isolated_artifact_root_and_global_device():
    league = train_market_scorer.LeagueConfig.from_toml(train_market_scorer.LEAGUE_CONFIG)
    assert league.league == "dqn_market_scorer"
    assert train_market_scorer.ARTIFACTS == ARTIFACTS_ROOT / "dqn_market_scorer"
    assert train_market_scorer._device() == "auto"


def test_progress_seed_base_does_not_collide_with_the_other_runners():
    import train_dqn

    assert (
        train_market_scorer.MARKET_CONFIG["progress_seed_base"]
        != train_dqn.SELF_PLAY_CONFIG["progress_seed_base"]
    )
