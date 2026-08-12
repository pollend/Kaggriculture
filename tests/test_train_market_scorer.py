from pathlib import Path

import train_market_scorer
from agents.dqn_market_scorer.market_policy import POLICY_KINDS
from kaggriculture_lab.constants import ARTIFACTS_ROOT
from kaggriculture_lab.identity import (
    effective_model_id,
    training_artifact_root,
    training_instance_seed,
)


def test_effective_market_config_builds_a_single_learner_roster(tmp_path: Path, monkeypatch):
    """One learner, trainer-only. The league schedules the clock, not a population."""
    monkeypatch.setattr(train_market_scorer, "ARTIFACTS", tmp_path)
    config = train_market_scorer._effective_config()
    config.validate()

    assert config.anchor_manifests == ()
    assert config.training_manifests == ("agents/dqn_market_scorer/agent.toml",)
    assert [(item.type, item.name, item.train) for item in config.training_agents] == [
        ("dqn_market_scorer", "joint03", True),
    ]
    assert config.pairing_mode == "trainer_only"
    assert config.population_size == 0
    # One update per generation: `rounds_per_generation` x `batches_per_round`
    # must match the manifest's `min_new_batches`.
    assert config.rounds_per_generation == 20
    assert config.episode_steps == 720


def test_joint03_has_an_independent_model_identity_and_artifact_root(tmp_path: Path):
    assert train_market_scorer.MODEL_NAME == "joint03"
    assert train_market_scorer.SESSION_ID == "market-scorer-joint03"
    assert effective_model_id(
        "dqn_market_scorer.donor_market_q_v1", train_market_scorer.MODEL_NAME
    ) == "dqn_market_scorer.donor_market_q_v1@joint03"
    assert training_instance_seed(
        "dqn_market_scorer", "joint03"
    ) != training_instance_seed("dqn_market_scorer", "default")
    assert training_artifact_root(
        tmp_path,
        {"agent_type": "dqn_market_scorer", "name": train_market_scorer.MODEL_NAME},
    ) == tmp_path / "dqn_market_scorer" / "joint03"


def test_promotion_is_measured_against_the_donor():
    """The benchmark is the unmodified policy this agent derives from.

    The paired margin measures the complete Q-guided residual against its donor
    baseline. Any weaker primary benchmark would let inherited donor behavior
    carry the result.
    """
    config = train_market_scorer.MARKET_CONFIG

    assert config["benchmark"] == train_market_scorer.DONOR
    assert config["benchmarks"][0] == train_market_scorer.DONOR
    assert "competitor_pilkwang" in train_market_scorer.DONOR


def test_policy_kinds_keep_the_planner_learner_and_baseline_distinct():
    assert train_market_scorer.MARKET_CONFIG["policy_kinds"] == POLICY_KINDS
    assert len(POLICY_KINDS) == len(set(POLICY_KINDS.values()))
    assert POLICY_KINDS["market_learner"] == "joint_residual_q"
    assert POLICY_KINDS["field_policy"] == "q_guided_field_policy"
    assert POLICY_KINDS["promotion_baseline"] == "promotion_baseline"


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
    # A trainable zero-Q network scores every candidate equally, argmax breaks
    # to index 0, and index 0 is the donor-default modifier with a zero reserve.
    # Spelled `donor_default`, not `zeros`: the literal reading of `zeros` is an
    # untrainable network, and the config should not say a thing it must not mean.
    assert train_market_scorer.MARKET_CONFIG["initialization"] == "donor_default"


def test_market_runner_uses_an_isolated_artifact_root_and_global_device():
    league = train_market_scorer.LeagueConfig.from_toml(train_market_scorer.LEAGUE_CONFIG)
    assert league.league == "dqn_market_scorer"
    assert league.name == "dqn-market-scorer-joint03"
    assert train_market_scorer.ARTIFACTS == ARTIFACTS_ROOT / "dqn_market_scorer"
    assert train_market_scorer._device() == "auto"


def test_progress_seed_base_does_not_collide_with_the_other_runners():
    import train_dqn

    assert (
        train_market_scorer.MARKET_CONFIG["progress_seed_base"]
        != train_dqn.SELF_PLAY_CONFIG["progress_seed_base"]
    )
