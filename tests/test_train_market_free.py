"""Guards on the free-market runner's constants.

These catch drift between `train_market_free.py`, its league config, and the
manifest -- the kind that only shows up as a session that trains the wrong thing
for a generation before anyone notices.
"""

import tomllib
from pathlib import Path

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
        ("dqn_market_free", train_market_free.MODEL_NAME, True),
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


def test_the_session_moves_and_the_model_lineage_does_not():
    """Only the session id moves at v20, and the pairing is the whole point.

    `ensure_session` digests the experiment manifest, so adding `rare_sigma_gain`
    makes the previous session unresumable and a new id is forced.
    `MODEL_NAME` stays because `all_batches` and `_self_play_pool` are scoped by
    `model_id` and the reward function did not move: v18's batches are the same
    kind of sample, so inheriting them is the buffer working as intended rather
    than a leak between regimes. Only the behaviour distribution changed.

    The reward is still the thing that would *force* a new name, so it stays
    asserted here: reusing one across a reward change fits one Q function to two
    reward streams.
    """

    assert train_market_free.SESSION_ID == "market-free-rare-v20"
    assert train_market_free.MODEL_NAME == "falloff_v1"
    assert train_market_free.MODEL_NAME not in train_market_free.SESSION_ID


def test_the_runner_clones_nothing():
    """Demonstrations are gone, and the manifest was always the control.

    v4 pinned 200 `configurable_crop_surrogate` seats at 25% of every minibatch.
    The clone itself worked -- behaviour-clone loss fell 0.42 to 0.06 over five
    updates -- but under `outcome_only` those transitions also carried a saturated
    +1.0 outcome, opposite the anchor's -1.0, on states the network cannot tell
    apart. Half of every batch was a constant no action explains.
    """

    assert _manifest()["models"][0]["config"]["demonstration_episodes"] == 0
    assert train_market_free.FREE_CONFIG["demonstration_episodes"] == 0
    # Dead keys are how v3 ran with `demo_fraction` silently zero while its config
    # said 200. If nothing is pinned, nothing may claim to be.
    assert "demonstration_family" not in train_market_free.FREE_CONFIG
    assert "demonstration_min_money" not in train_market_free.FREE_CONFIG
    assert "behavior_clone_steps_per_update" not in train_market_free.FREE_CONFIG


def test_the_reward_pays_only_the_outcome():
    """There is one reward stream and no mode to choose, which is the point.

    v3 paid the money delta plus potential shaping, and the shaping marked the
    whole shed to a quote the agent moved itself by buying. Measured over 300 of
    its episodes: 219 `BUY_PRODUCT` and 204 `SELL` per episode, wheat and
    fertilizer 94% of the sale volume -- the only two goods `BUY_PRODUCT` can buy
    back -- against 13 wheat actually planted. `outcome_only` switched it off at
    v4 and v13 removed the shaped arm outright, so neither config can select it.
    """

    import inspect

    from agents.dqn_market_free.training import reward as reward_module

    assert not hasattr(reward_module, "REWARD_MODES")
    assert not hasattr(reward_module, "potential")
    assert "mode" not in inspect.signature(reward_module.episode_rewards).parameters
    assert "reward_mode" not in train_market_free.FREE_CONFIG
    assert "reward_mode" not in _manifest()["models"][0]["config"]


def test_the_manifest_stays_the_no_self_play_control():
    """Self-play is a runner curriculum, not the family's default.

    `self_play_after = 0` disables it. `simple_v2` never set the key anywhere, so
    the mirror never played across all 666 of its batches -- the default has to
    be visible in the manifest rather than implied by a missing key.
    """

    config = _manifest()["models"][0]["config"]

    assert config["self_play_after"] == 0
    assert config["self_play_pool"] >= 1
    assert 0.0 < config["self_play_ratio"] < 1.0


def test_the_mirror_cold_starts_behind_a_non_saturating_anchor():
    """The mix is never empty, and the one static opponent is not a wall.

    Two failures are pinned here at once.

    An empty `opponents` is pure mirror play, which only cold-starts when the
    inherited model already has checkpoints. When it does not, `_self_play_pool`
    returns nothing, the base is empty, and collection raises "free-market
    collection needs at least one opponent" on batch 0 -- which is exactly how
    `market-free-noisy-v6` died on its first round.

    The other failure is what the empty base was avoiding. Under `outcome_only`
    the whole reward is `tanh(margin / TERMINAL_MARGIN_SCALE)` plus two ~0.002
    charges, and every scripted *competitor* finishes far enough above a learner
    at ~0 to saturate it: pilkwang 139k, lucifer19 157k, tetsutani 156k,
    kaitofukami 159k, `configurable_crop_surrogate` 32k, against a scale of
    10,000. A constant on every terminal reward cannot change an argmax. Seeded
    noise is the one static opponent that finishes in the learner's own range, so
    the margin against it stays inside the linear part of the tanh.

    `train_dqn.py` records the cost of having no static opponent at all: 17
    consecutive rejections, because promotion is gated on a benchmark the policy
    never plays. The noise slot is what keeps collection and the gate in contact.
    """

    from agents.dqn_market_scorer.training.updater import compose_opponents

    config = train_market_free.FREE_CONFIG
    base = config["opponents"]

    assert len(base) == 1
    assert base[0].startswith("kaggriculture_lab.opponents:make_seeded_random")
    # Reproducible: the built-in `random` agent builds a fresh unseeded RNG per
    # call, so two runs of the same seed diverge.
    assert "seed=" in base[0]
    # No competitor may be the static opponent -- that is the saturating case.
    assert not any("competitor_" in spec or "surrogate" in spec for spec in base)

    assert config["league_opponents"] is False
    # Must be truthy. The updater reads `if self_play_after and batch_index >=
    # self_play_after`, so the manifest's 0 disables the mirror entirely.
    assert config["self_play_after"] >= 1
    assert config["self_play_pool"] >= 1
    assert train_market_free.BENCHMARKS == (train_market_free.DONOR,)
    assert not any("surrogate" in spec for spec in config["benchmarks"])

    # Reproduce the updater's own branch. The mix must be non-empty at every
    # batch index, before and after the mirror joins, and once it joins the ratio
    # must resolve to two mirror slots against the single noise slot rather than
    # rounding one of them away.
    pool = [f"self-{index}" for index in range(config["self_play_pool"])]
    for batch_index, has_pool in ((0, False), (1, False), (20, True), (120, True)):
        opponents = list(base)
        if config["self_play_after"] and batch_index >= config["self_play_after"] and has_pool:
            opponents = compose_opponents(opponents, pool, config["self_play_ratio"])
        assert opponents, f"empty opponent mix at batch {batch_index}"
        if has_pool:
            assert opponents[0] == base[0]
            assert sum(spec.startswith("self-") for spec in opponents) == 2


def test_the_runner_declares_the_noise_knobs_it_means_to_change():
    """A misspelled key is silent: `config.get` keeps the default either way."""

    manifest = _manifest()["models"][0]["config"]
    config = {**manifest, **train_market_free.FREE_CONFIG}

    for key in (
        "noisy",
        "noisy_layers",
        "noisy_cadence",
        "noisy_sigma0",
        "noisy_actor_alpha",
        # Both controller setpoints are declared in the manifest, so that a
        # misspelling in either is a failure here rather than a silently
        # uncontrolled run. v7 overrides both from `FREE_CONFIG`; that is safe
        # only because it also takes a new session id, since `ensure_session`
        # digests `experiment.training_agents`.
        "noisy_order_rate_target",
        "noisy_sigma_scale_limit",
        "noisy_hot_order_rate",
        "noisy_hot_scale_limit",
        "noisy_sigma_scale_floor",
    ):
        assert key in manifest, f"{key} must be declared in the manifest to be readable"

    # "head" leaves `has_inert_hidden_trunk` -- the only structural guard against
    # the NOOP collapse -- correct as written, and keeps the trunk shared with
    # the behaviour-clone cross-entropy head deterministic.
    assert config["noisy_layers"] == "head"
    assert config["behavior_clone_noise"] is False

    # One perturbed policy per *episode*, which satisfies the `n_step` pairing
    # strictly harder than "day" did: a 24-step return must come from one policy,
    # and one policy per 720-turn season gives that trivially. What it adds is
    # commitment across the 48-96 turn BUY_SEED-to-harvest cycle, which day
    # cadence cannot span. Pinned to the value rather than the set, because
    # drifting back breaks the commitment silently -- every setting is valid and
    # none of them errors.
    assert config["noisy_cadence"] == "episode"
    assert config["n_step"] == 24, "cadence must still cover the credit horizon"


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

    `priority_alpha` and `priority_beta` came back, so they are no longer listed.
    The *schedules* did not, and stay listed: the objection prioritisation was
    retired over was an anneal that has to key off `gradient_steps`, which a
    lineage reset snaps backwards while the buffer does not move. A constant
    exponent has no such state. Anything here ending `_anneal_steps` or `_end`
    would reintroduce exactly the drift that was the reason.
    """

    config = _manifest()["models"][0]["config"]
    retired = {
        "act_rate",
        "candidate_chunk",
        "priority_alpha_anneal_steps",
        "priority_beta_end",
        "priority_beta_anneal_steps",
        "shadow_reset_after_hard_rejections",
    }
    assert retired.isdisjoint(config)


def test_both_defences_against_an_all_bankruptcy_buffer_are_declared():
    """The demonstration share and the return weighting cover different cases.

    `demo_fraction` only bites when something is pinned, and on this run nothing
    is -- `demonstration_episodes` is 0, so the share resolves to zero and the
    mechanism goes inert without moving a number anyone was watching. That is the
    failure this test exists to make loud: both knobs have to be declared, and
    `priority_alpha` is the one that still works with no demonstrator.
    """

    config = _manifest()["models"][0]["config"]
    assert 0.0 < config["demo_fraction"] < 1.0
    # The wide batch is only affordable because nothing is chunked any more.
    assert config["batch_size"] == 256

    # Declared, so `config.get` reads the file rather than the code default.
    assert "priority_alpha" in config
    assert "priority_beta" in config
    # Below 1.0: at 1.0 the correction is exact, so the weighted gradient equals
    # the uniform one in expectation and the reweighting buys nothing.
    assert 0.0 < config["priority_beta"] < 1.0


def test_every_update_is_the_release():
    """The gate is off, and the knob it makes unreachable is set to say so.

    Nothing here ever gated training -- `lineage_checkpoint` takes the newest
    checkpoint promoted or not and `_self_play_pool` does not filter on
    `promoted`, so a rejected candidate always went straight into the next
    generation. What a rejection withheld was the release.

    `resume_reset_after_rejections` has to be 0 as a consequence, not as a
    preference: with every update promoting, `rejections_since_promotion` is
    permanently 0 and the streak can never trip. Left at 2 it would read as a
    live safety net that cannot fire.
    """

    config = _manifest()["models"][0]["config"]

    assert config["promotion_rule"] == "always"
    assert config["resume_reset_after_rejections"] == 0
    # The rule lives in the manifest, so the runner cannot disagree with it.
    assert "promotion_rule" not in train_market_free.FREE_CONFIG

    # The arena is still measured -- only its effect on the boolean is gone.
    assert config["viability_min_orders_per_decision"] > 0.0
    assert config["promotion_seeds"] > 0


def test_runner_overrides_carry_no_inert_keys():
    """Every override must be read by *this* family.

    `active_peer_types` and `history_peer_types` were copied from
    `train_market_scorer.py` and are only read by the scorer and job-scorer
    updaters; here they persisted into `config_json` as rows nothing consumes.
    """

    assert "active_peer_types" not in train_market_free.FREE_CONFIG
    assert "history_peer_types" not in train_market_free.FREE_CONFIG
    assert train_market_free.FREE_CONFIG["learning_rate"] == 1e-4


def test_the_priority_knobs_stay_out_of_the_runner_override():
    """They belong in the manifest, and putting them here would break the resume.

    `ensure_session` digests `experiment.training_agents`, which embeds
    `FREE_CONFIG` verbatim, and raises `run manifest mismatch` when the digest
    moves. A key added here therefore costs a new session ID -- a restarted
    rating series and a cold buffer -- for a knob the manifest can carry for
    free, since `register_training_agent` upserts `config_json` on every start.
    """

    assert "priority_alpha" not in train_market_free.FREE_CONFIG
    assert "priority_beta" not in train_market_free.FREE_CONFIG


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
