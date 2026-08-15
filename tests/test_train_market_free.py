"""Guards on the free-market runner's constants.

These catch drift between `train_market_free.py`, its league config, and the
manifest -- the kind that only shows up as a session that trains the wrong thing
for a generation before anyone notices.
"""

import os
import tomllib
from pathlib import Path

import train_market_free
from agents.dqn_market_free.training.updater import POLICY_KINDS
from agents.dqn_market_scorer.training.benchmarks import promotion_benchmark
from kaggriculture_lab.constants import ARTIFACTS_ROOT, REPO_ROOT


def _manifest() -> dict:
    with (REPO_ROOT / train_market_free.MANIFEST).open("rb") as stream:
        return tomllib.load(stream)


def test_effective_config_builds_the_learner_roster(tmp_path: Path, monkeypatch):
    """One trainable slot per learner, all trainer-only.

    `population_size` stays 0 and `pairing_mode` stays `trainer_only`: the league
    population is a *breeding* population settled by Glicko off pairings, and this
    is not that. Each slot collects its own full-length episodes and the roster is
    a roster of trainers, which the harness expresses as named instances of one
    agent type rather than as league members.
    """

    monkeypatch.setattr(train_market_free, "ARTIFACTS", tmp_path)
    config = train_market_free._effective_config()
    config.validate()

    assert config.anchor_manifests == ()
    assert config.training_manifests == ("agents/dqn_market_free/agent.toml",)
    assert [(item.type, item.name, item.train) for item in config.training_agents] == [
        ("dqn_market_free", f"learner_{index:02d}", True)
        for index in range(train_market_free.LEARNERS)
    ]
    # Exactly one primary, which `LeagueConfig.validate` requires and
    # `_primary_training_model_id` resolves. Pinned to slot 0 rather than left to
    # whichever slot is currently best: a primary that moved as the population
    # turned over would make every query resolving it non-reproducible.
    assert sum(item.primary for item in config.training_agents) == 1
    assert config.training_agents[0].primary

    # Distinct seeds are the only thing separating the slots at the start. Sharing
    # one would make ten identical searches of the same space and the population
    # would have nothing to select between.
    seeds = [item.model_config["seed"] for item in config.training_agents]
    assert len(set(seeds)) == train_market_free.LEARNERS

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


def test_the_session_and_every_model_name_move_together():
    """Both move at v23, which is the first time the names have moved since v10.

    They have to. `all_batches`, `_self_play_pool`, `lineage_checkpoint`,
    `training_models.active_checkpoint_id` and the slot state are every one of
    them scoped by `model_id`, so ten named slots are ten independent lineages --
    and any of them inheriting v22's single lineage would start the population
    with nine cold slots and one carrying the stalled trajectory the population
    exists to leave. The mature group would then be seeded from exactly that.

    The earlier rule this replaces still holds for the ordinary case: a name is
    kept across a change that only moves the behaviour distribution, because v20's
    batches are the same kind of sample and inheriting them is the buffer working
    as intended. A reward change forces a new name; so does this.
    """

    assert train_market_free.SESSION_ID == "market-free-population-v25"
    names = [item["name"] for item in train_market_free.TRAINING_AGENTS]
    assert names == [f"learner_{index:02d}" for index in range(train_market_free.LEARNERS)]
    assert train_market_free.MODEL_NAME == names[0]
    assert all(name not in train_market_free.SESSION_ID for name in names)


def test_the_core_budget_is_concurrency_times_jobs():
    """v23's overhead was idle cores, not simulation cost.

    Ten learners on a 32-core box with `training_concurrency` at its default of 1:
    max concurrent model updates 1, load average 9, 77 minutes a generation of
    which 96% was arena nothing was waiting on. The two knobs multiply, because
    each learner's rollouts and arena series spawn their own `jobs`-wide pool, so
    they have to be read together or the budget is invisible.
    """

    config = train_market_free._effective_config()
    assert config.training_concurrency == train_market_free.CONCURRENCY
    assert train_market_free.FREE_CONFIG["jobs"] == train_market_free.JOBS_PER_LEARNER
    # Not oversubscribed. The pools are `spawn` and unpinned BLAS workers that
    # outnumber the cores are what `_single_threaded_workers` exists to prevent.
    assert train_market_free.CONCURRENCY * train_market_free.JOBS_PER_LEARNER <= os.cpu_count()
    # No point running more slots at once than there are slots.
    assert train_market_free.CONCURRENCY <= train_market_free.LEARNERS


def test_a_benchmark_the_run_never_plays_has_a_way_out_of_the_gate():
    """The invariant `market-free-rare-v20` broke, stated so both fixes satisfy it.

    v20 gated on pilkwang while `opponents` held one seeded-noise agent and
    nothing else. The learner solved that opponent at generation 6 -- paired margin
    flat near 5,900, win rate exactly 1.000 for the 55 generations after -- and
    accumulated 57 consecutive rejections against a benchmark it had never played.
    `train_dqn.py` recorded the same shape at 17 rejections before it.

    Two ways to not be in that state, and either is enough. Play the benchmark, so
    the rollouts and the gate measure the same distribution; or leave the gate an
    escape, so a stalled ratchet cannot become a fixed point. v21 tried the first
    and its own 120 pairings argued against it -- pilkwang at -134,558 and the
    surrogate at -29,376 are both a saturated -1.0 through the tanh, so they cost
    22% of collection and returned a constant. v22 takes the second.

    What is not optional is that *one* of them holds. v20 had neither.
    """

    config = train_market_free.FREE_CONFIG
    opponents = config["opponents"]

    plays_the_benchmark = promotion_benchmark(config) in opponents
    manifest = _manifest()["models"][0]["config"]
    gate_has_an_exit = (
        int(manifest["promotion_ratchet_grace_after_rejections"]) > 0
        and config.get("benchmark_grace_min_money") is not None
    )
    assert plays_the_benchmark or gate_has_an_exit

    # Either way, something in the mix has to carry terminal gradient, and seeded
    # noise is the only static opponent that does at this strength.
    assert any("make_seeded_random" in opponent for opponent in opponents)


def test_the_stalled_lineage_is_given_a_reachable_grace_floor():
    """0.0 rather than the 750 default, and the ratchet is what still binds.

    v20's best benchmark measurement across 57 rejections was 427, so any positive
    floor leaves the deadlock exactly where it was. What decides instead is
    non-regression against the incumbent's recorded 145.815, which grace does not
    drop -- and hard viability, which never stands down at all.
    """

    assert train_market_free.FREE_CONFIG["benchmark_grace_min_money"] == 0.0
    manifest = _manifest()["models"][0]["config"]
    assert manifest.get("benchmark_grace_min_money") is None
    assert int(manifest["promotion_ratchet_grace_after_rejections"]) > 0


def test_the_buffer_outlives_a_generation_of_ingest():
    """v20 aged out 1,212 of 1,361 batches before a gradient step opened them.

    20 rounds x 1 batch x 16 episodes is 320 episodes per generation into a
    400-episode ring, so a batch was evicted about as fast as it arrived and 89% of
    everything collection paid for was never sampled.
    """

    league = train_market_free.LeagueConfig.from_toml(
        REPO_ROOT / train_market_free.LEAGUE_CONFIG
    )
    model = _manifest()["models"][0]["config"]
    per_generation = (
        league.rounds_per_generation
        * int(model["batches_per_round"])
        * int(model["episodes_per_batch"])
    )
    assert train_market_free.FREE_CONFIG["buffer_episodes"] >= 2 * per_generation


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

    v21 widens the base past noise alone, and what is pinned here is the part that
    did not change: the *first* slot -- the one the mirror is composed around and
    the only one guaranteed present at every batch index -- still has to be the
    non-saturating opponent. v20 is why the wider base exists at all; it ran the
    single-noise version to a 1.000 win rate by generation 6 and then learned
    nothing for 55 generations. See `test_the_gate_benchmark_is_an_opponent_the_
    run_actually_plays`.
    """

    from agents.dqn_market_scorer.training.updater import compose_opponents

    config = train_market_free.FREE_CONFIG
    base = config["opponents"]

    assert base, "an empty base cannot cold-start"
    assert base[0].startswith("kaggriculture_lab.opponents:make_seeded_random")
    # Reproducible: the built-in `random` agent builds a fresh unseeded RNG per
    # call, so two runs of the same seed diverge.
    assert "seed=" in base[0]
    # The saturating opponents earn their slots on state coverage, so they must
    # stay a minority of the static base rather than becoming the anchor.
    walls = [spec for spec in base if "competitor_" in spec or "surrogate" in spec]
    assert len(walls) < len(base)

    assert config["league_opponents"] is False
    # Must be truthy. The updater reads `if self_play_after and batch_index >=
    # self_play_after`, so the manifest's 0 disables the mirror entirely.
    assert config["self_play_after"] >= 1
    assert config["self_play_pool"] >= 1
    assert train_market_free.BENCHMARKS == (train_market_free.DONOR,)
    assert not any("surrogate" in spec for spec in config["benchmarks"])

    # Reproduce the updater's own branch. The mix must be non-empty at every
    # batch index, before and after the mirror joins, and once it joins every
    # static entry must survive rather than being rounded away -- a base slot that
    # disappears is an opponent the run believes it is training against and is not.
    pool = [f"self-{index}" for index in range(config["self_play_pool"])]
    for batch_index, has_pool in ((0, False), (1, False), (20, True), (120, True)):
        opponents = list(base)
        if config["self_play_after"] and batch_index >= config["self_play_after"] and has_pool:
            opponents = compose_opponents(opponents, pool, config["self_play_ratio"])
        assert opponents, f"empty opponent mix at batch {batch_index}"
        if has_pool:
            assert opponents[0] == base[0]
            for spec in base:
                assert spec in opponents, f"{spec} rounded out of the mix"
            mirrored = sum(spec.startswith("self-") for spec in opponents)
            # `self_play_ratio` is what this has to land on, and whole slots only
            # approximate it -- three static entries at 0.65 give 6 of 9.
            assert abs(mirrored / len(opponents) - config["self_play_ratio"]) < 0.05


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
    # The money floors are ratios, so before a release they need something to be
    # a ratio *of*. The quiet baseline is not usable there: it never trades, so
    # it keeps the full stake and the floor becomes "lose nothing".
    assert 0.0 < config["viability_cold_start_reference_money"] < 3_000.0


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


def test_the_release_is_gated_on_a_fixed_benchmark():
    """The release decision reads something that does not move with the candidate.

    Every other comparison in this gate is against the previous update, so a run
    can walk itself down while each individual step reads as an improvement --
    `falloff_v1` released nine of those in a row. The ratchet reads the
    candidate's own money against the fixed benchmark instead.

    `resume_reset_after_rejections` is off under the population, and that is not
    caution. The snapback and the population answer the same question -- this
    trajectory has gone bad, restart it from something better -- and the snapback
    answers it by restoring the slot's own incumbent, whose score is exactly what
    the plateau detector reads. Leaving both on pins `lr_best_score` with the
    mechanism meant to rescue the slot, and the slot never reaches its cut.
    """

    config = _manifest()["models"][0]["config"]

    assert config["promotion_rule"] == "benchmark_ratchet"
    assert config["resume_reset_after_rejections"] == 0
    assert train_market_free.FREE_CONFIG["resume_reset_after_rejections"] == 0
    assert config["lr_graduate_cut"] > 0.0, "the population must be the live escape route"
    # The rule lives in the manifest, so the runner cannot disagree with it.
    assert "promotion_rule" not in train_market_free.FREE_CONFIG

    # Declared rather than left to `config.get`, which would keep the code
    # default while the file says nothing -- the silent-key failure this file
    # tests for elsewhere.
    assert config["benchmark_release_min_money"] > config["benchmark_cold_start_min_money"] > 0.0
    assert 0.0 < config["benchmark_money_ratchet"] < 1.0
    # A relative bar that can never be released is how a gated run freezes.
    assert config["promotion_ratchet_grace_after_rejections"] > 0
    # Pinned, so this update's benchmark score and the incumbent's recorded one
    # are the same hundred seeds.
    assert config["benchmark_seed_stride"] == 0

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
