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


def test_the_session_starts_a_fresh_model_lineage():
    """Both names move together again, after six sessions where only one did.

    v6 through v10 kept `outcome_v4` deliberately: `reward_mode` was unchanged so
    the old batches were the same kind of sample, and `_self_play_pool` is scoped
    by `model_id`, so a fresh name has an empty pool on batch 0 -- which, with no
    base opponent to fall back on, is the "needs at least one opponent" error
    rather than a slower start.

    Neither holds now. The database was reset, so there is no buffer, lineage or
    checkpoint to inherit and the name was pointing at nothing; and `opponents`
    carries a seeded-noise base, so `compose_opponents` has something to mix
    against and the cold start is survivable.

    The reward is still the thing that would *force* a new name, so it stays
    asserted here: reusing one across a reward change fits one Q function to two
    reward streams.
    """

    assert train_market_free.SESSION_ID == "market-free-idleburst-v12"
    assert train_market_free.MODEL_NAME == "coldstart_v1"
    assert train_market_free.MODEL_NAME not in train_market_free.SESSION_ID
    # Inheriting the lineage is only safe while the reward is untouched. If this
    # moves, the session needs a new model name as well.
    assert train_market_free.FREE_CONFIG["reward_mode"] == "outcome_only"


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


def test_the_runner_pays_only_the_outcome():
    """The reason this instance exists, asserted rather than left in a comment.

    v3 paid the money delta plus potential shaping, and the shaping marked the
    whole shed to a quote the agent moved itself by buying. Measured over 300 of
    its episodes: 219 `BUY_PRODUCT` and 204 `SELL` per episode, wheat and
    fertilizer 94% of the sale volume -- the only two goods `BUY_PRODUCT` can buy
    back -- against 13 wheat actually planted. The manifest stays the shaped
    control, so the departure has to live here.
    """

    from agents.dqn_market_free.training.reward import REWARD_MODES

    config = train_market_free.FREE_CONFIG
    assert config["reward_mode"] == "outcome_only"
    assert config["reward_mode"] in REWARD_MODES
    # The manifest is the control and must not have drifted into the mode.
    assert "reward_mode" not in _manifest()["models"][0]["config"]


def test_the_outcome_baseline_key_is_an_inert_digest_relic():
    """It is still in the config and nothing reads it -- both halves, or neither.

    `ensure_session` digests `experiment.training_agents`, and the stored
    manifest for `market-free-noisy-v6` carries this key, so deleting it makes
    that session unresumable. It stays for that alone: the
    reward no longer takes a baseline at all. Left unasserted this is exactly the
    silent dead key `test_the_runner_clones_nothing` exists to forbid, so the
    pairing is pinned -- if the parameter ever comes back, this fails and the
    config has to say so deliberately.
    """

    import inspect

    from agents.dqn_market_free.training.reward import episode_rewards

    assert train_market_free.FREE_CONFIG["outcome_baseline"] == 0.0
    assert "outcome_baseline" not in inspect.signature(episode_rewards).parameters


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


def test_exactly_one_exploration_mechanism_is_live():
    """Epsilon and NoisyNet must never both be putting states in the buffer.

    They are not additive. Epsilon's explore branch bypasses the network
    entirely -- uniform over feasible actions with NOOP removed, so an exploring
    turn is forced to place an order and 97 of the 186 actions spend money. That
    is the overbuying the low-scoring half of a batch shows, and running it
    alongside weight noise would keep manufacturing the exact failure the noise
    was turned on to remove, while making the two contributions unattributable.

    Zeroed rather than deleted: `epsilon` is part of the policy identity string,
    so removing the key would rewrite every revision id in the family.
    """

    from agents.dqn_market_free.training.updater import _epsilon

    manifest = _manifest()["models"][0]["config"]
    config = {**manifest, **train_market_free.FREE_CONFIG}

    assert config["noisy"] is True
    assert config["epsilon_start"] == 0.0
    assert config["epsilon_end"] == 0.0
    assert _epsilon(config, 0) == 0.0
    assert _epsilon(config, 3_000_000) == 0.0
    assert "epsilon_start" in config, "the key stays, so revision ids do not churn"

    # The manifest still carries a live epsilon schedule, which is only safe
    # while it is *not* also noisy. Whichever way that key is set there, the two
    # must not both be on: assert the pairing rather than either value, so
    # flipping one without the other fails here instead of quietly doubling the
    # exploration budget in whatever run picks the manifest up without overrides.
    manifest_epsilon = _epsilon(manifest, 1_380_480)
    assert not (manifest.get("noisy", False) and manifest_epsilon > 0.0), (
        "the manifest has both exploration mechanisms live: `noisy` is on while "
        f"epsilon still anneals to {manifest_epsilon:.3f}. Zero `epsilon_start`/"
        "`epsilon_end` there, or turn `noisy` back off and keep it as the control."
    )


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
        "noisy_flip_rate_target",
        "noisy_order_rate_target",
        "noisy_sigma_scale_limit",
        "noisy_burst_every",
        "noisy_burst_order_rate",
        "noisy_burst_scale_limit",
        "noisy_sigma_scale_floor",
        "noisy_idle_trading_rate",
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


def test_exactly_one_sigma_controller_is_live():
    """Two setpoints, one curve allowed to move sigma.

    They disagree by construction on a collapsed policy -- that disagreement is
    the entire reason `noisy_order_rate_target` exists -- so a run with both set
    could not say afterwards which number moved sigma.

    v6 controlled on the argmax flip rate and that is what broke it: the policy
    collapsed onto NOOP, where every action is near-tied and flips are therefore
    cheap and plentiful, so the rate read 0.28-0.39 against a 0.05 setpoint and
    the controller divided sigma down to `[0.33 .. 0.010]` across the eight
    actors. Exploration was throttled at the one moment it needed raising.
    """

    from agents.dqn_market_free.training.updater import _noisy

    manifest = _manifest()["models"][0]["config"]
    config = {**manifest, **train_market_free.FREE_CONFIG}

    assert config["noisy_order_rate_target"] > 0.0
    assert config["noisy_flip_rate_target"] == 0.0

    noisy = _noisy(config)
    assert noisy.controlled
    assert noisy.controls_on_orders

    # The manifest stays the uncontrolled epsilon-greedy arm, so whichever way
    # this run sets the pair, picking the manifest up without overrides never
    # silently inherits a controller aimed at the wrong quantity.
    assert manifest["noisy_order_rate_target"] == 0.0
    assert not _noisy(manifest).controls_on_orders


def test_the_burst_answers_idleness_rather_than_a_schedule():
    """The periodic schedule is retired; the trigger is silence.

    It was measured twice and never came out positive. At `noisy_burst_order_rate
    = 0.20` it changed nothing over 560 seats -- 4.6% of burst-round seats bought
    land against 4.9% of ordinary ones. Flat at 0.35 (sigma 3.159) it put 49% of
    212 seats bankrupt against 8-26% across the ordinary ladder, with zero land
    purchases. On a fixed cadence the burst fires when the policy is *healthy*,
    which is exactly when high sigma is pure damage.

    Idleness inverts that. A NOOP policy produces no learning signal at all, so
    bankruptcy is strictly the better failure -- it at least generates transitions
    with a gradient. The episodes a burst replaces there were worth nothing, and
    it stops by itself as soon as trading resumes, so the 49%-bankruptcy objection
    is the point rather than the cost.
    """

    from agents.dqn_market_free.training.updater import _noisy

    manifest = _manifest()["models"][0]["config"]
    config = {**manifest, **train_market_free.FREE_CONFIG}
    noisy = _noisy(config)

    assert noisy.burst_every == 0, "the periodic schedule is retired, not retuned"
    rounds = train_market_free.LeagueConfig.from_toml(
        REPO_ROOT / train_market_free.LEAGUE_CONFIG
    ).rounds_per_generation
    assert not any(noisy.burst_round(index) for index in range(rounds))

    # Silence arms it at any point in the generation.
    assert noisy.idle_trading_rate > 0.0
    assert all(noisy.burst_round(index, idle=True) for index in range(rounds))

    # Below the gate's own activity floor, so the escape answers genuine silence
    # rather than merely cautious trading.
    assert noisy.idle_trading_rate < config["viability_min_orders_per_decision"]

    # Sized as a rate against the ordinary setpoint, not as a sigma multiplier:
    # the base a multiplier would scale moves every update as the controller
    # tracks the setpoint, and what a given scale buys moves with the network.
    assert noisy.burst_order_rate > noisy.order_rate_target
    # Hard on purpose now. The cap still binds -- an unbounded burst is how 0.35
    # reached 3.159 in the first place -- but it is set to permit that sigma,
    # because on an idle policy that is the intent and not the accident.
    assert noisy.burst_scale_limit <= noisy.scale_limit

    # The manifest stays the no-burst control arm on both triggers.
    assert manifest["noisy_burst_every"] == 0
    assert manifest["noisy_idle_trading_rate"] == 0.0
    assert _noisy(manifest).burst_every == 0
    assert not _noisy(manifest).burst_round(0, idle=True)


def test_the_lineage_snaps_back_before_it_drifts_far():
    """Two rejections, not five.

    Both training and collection follow the newest checkpoint on the lineage,
    promoted or not, so a rejected update's policy is what collects the next
    generation. Measured across v7, v8 and v9 the run oscillates rather than
    diverging monotonically: it promotes, then produces candidates at 1,335 / 975
    / 3,877 against a released incumbent scoring ~17,000, and sometimes recovers
    on its own before a limit of 5 is reached. That recovery is what made 5 look
    adequate -- but every rejected update in between is a generation collected
    from a policy the gate has already judged far worse than one sitting released
    and unused.

    Asserted rather than left in a comment because the knob is only readable in
    the manifest: `lineage_checkpoint` reads it through `config.get`, so a
    misspelling silently restores the default.
    """

    from agents.dqn_market_free.training.updater import lineage_checkpoint  # noqa: F401

    manifest = _manifest()["models"][0]["config"]
    config = {**manifest, **train_market_free.FREE_CONFIG}

    assert config["resume_reset_after_rejections"] == 2
    assert config["resume_reset_after_rejections"] > 0, "0 disables the reset entirely"
    # It lives in the manifest, not the runner, so every session inherits it.
    assert "resume_reset_after_rejections" not in train_market_free.FREE_CONFIG


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
