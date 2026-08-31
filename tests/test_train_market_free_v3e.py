"""Pins on `train_market_free_v3e.py`'s constants.

`train_market_free.py` has had these since `landheuristic-v43`; this driver had
none, and it is the file another session is most likely to edit under a live run.
What is pinned is the *relationship* between the keys, not a snapshot of every
value: the session and prefix that name the run, and the overrides that the
`rewardtogo-v1` review settled on and that would silently reintroduce a measured
failure if flipped back.
"""

from __future__ import annotations

import train_market_free_v3e as driver
from agents.dqn_market_free_v3e.training.reward import MONEY_SCALE


def test_the_session_and_prefix_name_the_n_step_run():
    # The prefix names the run: a reward change is a prefix sever (see
    # `SLOT_PREFIX` in the driver), so the two must move together.
    # crop-v2 folded the version into the prefix (`crop2`) so the session key could
    # keep its `-v2` suffix; the stem still has to agree.
    stem = driver.SLOT_PREFIX.rstrip("0123456789")
    assert driver.SESSION_ID.startswith(f"market-free-{stem}-v")
    names = [item["name"] for item in driver.TRAINING_AGENTS]
    assert names == [f"{driver.SLOT_PREFIX}_{index:02d}" for index in range(driver.LEARNERS)]
    assert driver.MODEL_NAME == names[0]
    assert sum(item["primary"] for item in driver.TRAINING_AGENTS) == 1
    assert MONEY_SCALE == 25_000.0


def test_the_overrides_the_review_settled_on_are_still_set():
    config = driver.FREE_CONFIG
    # The rollback pair remains the one-member compatibility behavior; the active
    # five-member run recovers through its current-versus-previous pool instead.
    assert config["lineage_source"] == "promoted"
    assert config["resume_reset_after_rejections"] == 2
    # Sized per *member* and in units of that member's own generations, so it tracks
    # `member_batches`: the target is two generations of its own history. At 20
    # batches a member collects 320 episodes a generation, so 640 keeps two.
    import tomllib
    from pathlib import Path

    manifest = tomllib.loads(Path(driver.MANIFEST).read_text())["models"][0]["config"]
    episodes_per_batch = int(manifest["episodes_per_batch"])
    episodes_per_generation = config["member_batches"] * episodes_per_batch
    assert config["buffer_episodes"] >= 2 * episodes_per_generation
    assert config["buffer_episodes"] <= 2 * episodes_per_generation
    # mark-v14: no benchmark arena; the margin over the mirror is the release read.
    assert config["benchmarks"] == list(driver.BENCHMARKS) == []
    assert config["promotion_score_signal"] == "margin"
    assert config["promotion_reference_window"] > 0
    assert config["promotion_tolerance_ci"] is True
    # Insolvent seasons are the negative examples.
    assert config["terminal_quality_weighting"] is False
    # A fresh prefix inherits no Adam moments; True here would reset them every update.
    assert config["reset_optimizer_moments"] is False
    # The learner pool is gone; nothing may reintroduce its keys by override.
    for retired in (
        "lr_graduate_cut",
        "population_slots",
        "population_warm_start",
        "mature_reserved_min_money",
        "self_play_mature_backfill",
    ):
        assert retired not in config, retired
    assert config["rare_sigma_pairs"] <= 2 and config["rare_sigma_cap"] <= 3.0


def test_the_mirror_pool_has_an_outside_anchor_and_a_money_floor():
    """The `anchor-v1` changes, which is the whole point of the run.

    Every arm from `members-v1` onward finished within 600 coins of the 3,000 stake
    while every arm before it floored at 6,113, and the difference was not the reward
    -- `herd` and `weighted3` also ran an empty `opponents`. What they had was a
    shared mature archive across ten slots with a money floor. It was deleted, and
    `_self_play_pool` reads a single `model_id`, so a slot that stops trading becomes
    its own only teacher. These four keys are what break that loop; flipping any one
    back reintroduces a measured failure.
    """

    config = driver.FREE_CONFIG
    # mark-v14 runs the mirror-only path on purpose -- pilkwang out of both roles --
    # and the history above is the risk it takes: every earlier empty-`opponents`
    # arm floored at the stake. Against that: the money floor on the pool, the newest
    # checkpoints as teachers, and a margin gate a dead mirror cannot inflate.
    # mark-v15: pilkwang in half the rollouts (the margin needs a market with
    # someone in it -- mark-v14's pure mirror read +-0), no benchmark arena.
    assert config["opponents"] == [driver.DONOR] and config["benchmarks"] == []
    assert config["self_play_ratio"] == config["self_play_ratio_final"] == 0.5
    assert config["self_play_latest_backfill"] is False
    # No teacher below the stake. 3,000 is what a policy that never trades scores.
    assert config["self_play_min_score"] >= 3000.0


def test_the_activity_floor_sits_in_the_regime_that_makes_money():
    # Contested money is monotone in this quantity over all recorded gates, and every
    # arm that cleared 10k ran 0.40-0.86. crop-v2 cleared the old 0.02 at 0.051 while
    # scoring 616 coins. It gates promotion, `member_rank_active` and the prune tier.
    assert driver.FREE_CONFIG["viability_min_orders_per_decision"] >= 0.17


def test_the_roster_arithmetic_keeps_the_prune_window_two_generations():
    """The three numbers that must move together, and did not on `anchor-v1`.

    `members.roster_candidates` asks `_CANDIDATE_SQL` for `2 * members_per_slot`
    distinct cutoffs, and that is "this generation and the previous one" only while
    each generation contains exactly `members_per_slot` cutoffs -- i.e. while
    `members_per_slot * member_batches == rounds_per_generation` and
    `member_batches == min_new_batches`. anchor-v1 ran 1 x 10 against 20 rounds, so
    the window silently covered two cutoffs, half of one generation, and the
    previous-generation half of the (mu+lambda) comparison was gone. A lineage that
    got worse could no longer lose to its own parent, which is the whole point of the
    roster.
    """

    import tomllib
    from pathlib import Path

    config = driver.FREE_CONFIG
    model = tomllib.loads(Path(driver.MANIFEST).read_text())["models"][0]
    league = tomllib.loads(Path(driver.LEAGUE_CONFIG).read_text())["league"]
    rounds = int(league["rounds_per_generation"])

    assert config["member_batches"] == model["min_new_batches"], (
        "min_new_batches is what fires the update; a drift trains one member on "
        "another's batches"
    )
    assert config["members_per_slot"] * config["member_batches"] == rounds
    assert driver._effective_config().rounds_per_generation == rounds
    assert rounds == driver.ROUNDS_PER_GENERATION == 100
    assert config["members_per_slot"] == 5
    assert config["member_batches"] * config["gradient_steps_per_batch"] == 960


def test_one_slot_five_members():
    """mark-v18: one slot, five members, the full 20-batch budget each."""

    config = driver.FREE_CONFIG
    assert driver.LEARNERS == driver.CONCURRENCY == 1
    assert len(driver.TRAINING_AGENTS) == 1
    assert config["members_per_slot"] == 5
    assert config["cross_slot_reseed"] is False
    # The retained checkpoint history, used as both the mirror pool and the bar.
    assert config["self_play_pool"] == 25
    assert config["promotion_reference_window"] == 25


def test_mark21_widens_the_herd_scaffolding():
    assert driver.SESSION_ID == "market-free-mark-v21"
    assert driver.SLOT_PREFIX == "mark21"
    from agents.dqn_market_free_v3e import donor, herd
    assert sum(donor.ANIMAL_SLOTS.values()) == 24 and donor.ANIMAL_SLOTS["NW"] == 6
    assert herd.WARM_SLOTS == 3
    assert driver.FREE_CONFIG["priority_signal"] == "margin"
    assert driver.FREE_CONFIG["member_exploit_count"] == 2
    # The other session's hire prior and sigma exclusion are off: neither was in
    # a run that climbed, and the cap is back on the mask.
    assert driver.FREE_CONFIG["hire_prior_weight"] == 0.0
    assert driver.FREE_CONFIG["hire_prior_bootstrap_steps"] == 0
    assert driver.FREE_CONFIG["rare_sigma_exclude_hire"] is False
    assert driver.FREE_CONFIG["promotion_score_signal"] == "margin"
    # The mark10 rosters were trained under the hire cap; warm-starting them onto
    # the open ladder is the mark-v6 collapse. Cold, both slots.
    for agent in driver.TRAINING_AGENTS:
        assert "member_warm_start_dir" not in agent["model_config"]


def test_the_five_member_path_queries_current_and_previous_five():
    """The member implementation asks storage for twice the configured pool."""

    from pathlib import Path

    members = Path("agents/dqn_market_free_v3e/training/members.py").read_text()
    assert "[model_id, 2 * members, model_id]" in members


def test_the_manifest_agrees_with_the_driver_on_the_anchor_keys():
    """The driver wins at runtime (`storage.py`), so a manifest left behind is a trap."""

    import tomllib
    from pathlib import Path

    manifest = tomllib.loads(Path(driver.MANIFEST).read_text())["models"][0]["config"]
    for key in (
        "members_per_slot",
        "member_batches",
        "buffer_episodes",
        "viability_min_orders_per_decision",
        "self_play_min_score",
    ):
        assert manifest[key] == driver.FREE_CONFIG[key], key


def test_every_override_key_is_one_the_manifest_or_updater_knows():
    """A typo in `FREE_CONFIG` is silently ignored by `config.get`; this catches it."""

    import tomllib
    from pathlib import Path

    manifest = tomllib.loads(Path(driver.MANIFEST).read_text())["models"][0]["config"]
    updater = Path("agents/dqn_market_free_v3e/training/updater.py").read_text()
    collect = Path("agents/dqn_market_free_v3e/training/collect.py").read_text()
    match_agent = Path("agents/dqn_market_free_v3e/match_agent.py").read_text()
    unknown = [
        key
        for key in driver.FREE_CONFIG
        if key not in manifest
        and f'"{key}"' not in updater
        and f'"{key}"' not in collect
        and f'"{key}"' not in match_agent
    ]
    assert unknown == [], unknown
