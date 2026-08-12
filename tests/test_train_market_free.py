"""Guards on the free-market runner's constants.

These catch drift between `train_market_free.py`, its league config, and the
manifest -- the kind that only shows up as a session that trains the wrong thing
for a generation before anyone notices.
"""

import tomllib
from pathlib import Path

import pytest

import train_market_free
from agents.dqn_market_free.training.updater import POLICY_KINDS
from agents.dqn_market_scorer.training.benchmarks import promotion_benchmark
from kaggriculture_lab.constants import ARTIFACTS_ROOT, REPO_ROOT


def _manifest() -> dict:
    with (REPO_ROOT / train_market_free.MANIFEST).open("rb") as stream:
        return tomllib.load(stream)


def test_effective_config_builds_a_single_learner_roster(tmp_path: Path, monkeypatch):
    """One learner, trainer-only. The league schedules the clock, not a population."""

    monkeypatch.setattr(train_market_free, "ARTIFACTS", tmp_path)
    config = train_market_free._effective_config()
    config.validate()

    assert config.anchor_manifests == ()
    assert config.training_manifests == ("agents/dqn_market_free/agent.toml",)
    assert [(item.type, item.name, item.train) for item in config.training_agents] == [
        ("dqn_market_free", "simple_v2", True),
    ]
    assert config.pairing_mode == "trainer_only"
    assert config.population_size == 0
    assert config.episode_steps == 720


def test_one_update_per_generation():
    """`rounds_per_generation` x `batches_per_round` must equal `min_new_batches`.

    Off by any factor and the update either never fires or fires mid-generation
    on a partial round, which makes the generation boundary meaningless as a
    comparison point.
    """

    league = train_market_free.LeagueConfig.from_toml(
        REPO_ROOT / train_market_free.LEAGUE_CONFIG
    )
    model = _manifest()["models"][0]
    per_round = int(model["config"]["batches_per_round"])
    assert league.rounds_per_generation * per_round == int(model["min_new_batches"])


def test_promotion_is_measured_against_the_donor():
    """The benchmark is the field policy this agent is still built on.

    The market layer is gone but the field policy is not, so pilkwang on both
    sides still isolates exactly what is being learned -- and it is the number
    that says whether an unscaffolded market can reach a hand-written one.
    """

    config = train_market_free.FREE_CONFIG
    # Index 0 is the release baseline; there is no separate singular key that
    # could name a different opponent than the list's first entry.
    assert promotion_benchmark(config) == train_market_free.DONOR
    assert "benchmark" not in config
    assert "competitor_pilkwang" in train_market_free.DONOR


def test_policy_kinds_do_not_collide_with_the_scaffolded_learner():
    """A query grouping by policy kind must not merge the two agents.

    They are the experiment and its control; sharing a kind label would make
    "how is the market learner doing" silently average them together.
    """

    from agents.dqn_market_scorer.market_policy import POLICY_KINDS as SCORER_KINDS

    assert len(POLICY_KINDS) == len(set(POLICY_KINDS.values()))
    assert POLICY_KINDS["market_learner"] == "raw_order_q"
    assert POLICY_KINDS["market_learner"] != SCORER_KINDS["market_learner"]


def test_demo_run_is_isolated_from_the_reward_only_control():
    """The manifest remains the control; this runner names a seeded lineage."""

    model = _manifest()["models"][0]
    assert model["config"]["demonstration_episodes"] == 0
    assert train_market_free.SESSION_ID == "market-free-simple-v2"
    assert train_market_free.MODEL_NAME == "simple_v2"
    assert train_market_free.FREE_CONFIG["demonstration_episodes"] == 200
    assert train_market_free.FREE_CONFIG["demonstration_min_money"] == 10_000.0
    assert train_market_free.FREE_CONFIG["behavior_clone_steps_per_update"] == 100
    assert (
        train_market_free.FREE_CONFIG["demonstration_family"]
        == "competitor_pilkwang_structured_economic_policy"
    )


def test_manifest_hard_gate_catches_both_collapses():
    """Money floors catch bankruptcy; the activity floor catches silence.

    There is no upper limit on orders per decision, and there cannot usefully be
    one: a decision emits at most one order, so the ratio is bounded by 1.0.
    """

    config = _manifest()["models"][0]["config"]

    assert config["viability_min_mean_money_ratio"] == 0.80
    assert config["viability_min_median_money_ratio"] == 0.75
    assert config["viability_min_orders_per_decision"] == 0.02
    assert not any(key.startswith("viability_max_") for key in config)


def test_manifest_declares_no_retired_knobs():
    """Every key the ten-slot action space needed is gone from the manifest.

    A stale key here is silent: `config.get` ignores it, so the run keeps its
    default while the file says otherwise.
    """

    config = _manifest()["models"][0]["config"]
    retired = {
        "act_rate",
        "candidate_chunk",
        "priority_alpha",
        "priority_beta",
        "priority_beta_end",
        "priority_beta_anneal_steps",
        "shadow_reset_after_hard_rejections",
    }
    assert retired.isdisjoint(config)


def test_exploration_rate_is_a_single_small_knob():
    """`epsilon` is now the whole trade rate; it used to be scaled by `act_rate`.

    Exploring on every turn is measurably fatal, so the start value has to be the
    small number rather than 1.0 with a 0.02 multiplier hidden behind it.
    """

    config = _manifest()["models"][0]["config"]
    assert config["epsilon_start"] == 0.18
    assert config["epsilon_end"] == 0.01
    assert config["epsilon_start"] > config["epsilon_end"] > 0.0
    # The ten-slot scheme's per-turn probability of acting randomly was
    # `epsilon * (1 - (1 - act_rate)^10)` = 0.183 at its start, not the 0.02 that
    # reading `act_rate` alone suggests. Setting the single knob to 0.02 explored
    # an order of magnitude less than the scheme it replaced.
    assert config["epsilon_start"] == pytest.approx(1.0 * (1 - (1 - 0.02) ** 10), abs=0.01)
    assert config["epsilon_actor_alpha"] > 0.0


def test_demonstration_share_replaces_prioritised_sampling():
    config = _manifest()["models"][0]["config"]
    assert 0.0 < config["demo_fraction"] < 1.0
    # The wide batch is only affordable because nothing is chunked any more.
    assert config["batch_size"] == 256


def test_promotion_gate_stays_on_for_this_run():
    """The runner must not quietly relax the manifest's release bar."""

    assert _manifest()["models"][0]["config"]["promotion_rule"] == "significant"
    assert "promotion_rule" not in train_market_free.FREE_CONFIG


def test_runner_overrides_carry_no_inert_keys():
    """Every override must be read by *this* family.

    `active_peer_types` and `history_peer_types` were copied from
    `train_market_scorer.py` and are only read by the scorer and job-scorer
    updaters; here they persisted into `config_json` as rows nothing consumes.
    """

    assert "active_peer_types" not in train_market_free.FREE_CONFIG
    assert "history_peer_types" not in train_market_free.FREE_CONFIG
    assert train_market_free.FREE_CONFIG["learning_rate"] == 1e-4


def test_manifest_never_bundles_a_competitor_or_the_trainer():
    manifest = _manifest()
    paths = set(manifest["code_paths"])
    assert not any("competitor_" in path for path in paths)
    assert not any("/training/" in path for path in paths)
    # The local donor is a licensed snapshot; no scorer package path belongs in
    # the free agent's submission bundle.
    assert "agents/dqn_market_free/donor.py" in paths
    assert "agents/dqn_market_free/LICENSE-pilkwang" in paths
    assert "agents/dqn_market_free/NOTICE" in paths
    assert "agents/dqn_market_free/weights.npz" not in paths
    assert not any("agents/dqn_market_scorer/" in path for path in paths)
    # There is no market executor here. That is the whole point of the agent.
    assert not any("market_exec" in path for path in paths)


def test_declared_code_paths_all_exist():
    for path in _manifest()["code_paths"]:
        assert (REPO_ROOT / path).exists(), path


def test_runner_uses_an_isolated_artifact_root_and_global_device():
    league = train_market_free.LeagueConfig.from_toml(
        REPO_ROOT / train_market_free.LEAGUE_CONFIG
    )
    assert league.league == "dqn_market_free"
    assert train_market_free.ARTIFACTS == ARTIFACTS_ROOT / "dqn_market_free"
    assert train_market_free._device() == "auto"
