"""Pins on `train_market_free_v3d.py`'s constants.

`train_market_free.py` has had these since `landheuristic-v43`; this driver had
none, and it is the file another session is most likely to edit under a live run.
What is pinned is the *relationship* between the keys, not a snapshot of every
value: the session and prefix that name the run, and the overrides that the
`rewardtogo-v1` review settled on and that would silently reintroduce a measured
failure if flipped back.
"""

from __future__ import annotations

import train_market_free_v3d as driver
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
    # The rollback pair. Live only because a one-member roster routes the resume
    # through `lineage_checkpoint`; under member pools both of these are dead config.
    assert config["lineage_source"] == "promoted"
    assert config["resume_reset_after_rejections"] == 3
    # Sized per *member* and in units of that member's own generations, so it tracks
    # `member_batches`: the target is ~2.5 generations of its own history. At 10
    # batches a member collects 160 episodes a generation, so 400; at 4 it collects
    # 64, so 160. A flat 400 at 4 batches would be 6 generations and 2.5x the
    # resident memory. The one-generation regime is what failed, and both settings
    # here are clear of it.
    import tomllib
    from pathlib import Path

    manifest = tomllib.loads(Path(driver.MANIFEST).read_text())["models"][0]["config"]
    episodes_per_batch = int(manifest["episodes_per_batch"])
    episodes_per_generation = config["member_batches"] * episodes_per_batch
    assert config["buffer_episodes"] >= 2 * episodes_per_generation
    assert config["buffer_episodes"] <= 3 * episodes_per_generation
    # The candidate itself is measured, never the incumbent's collection window.
    assert config["promotion_score_signal"] == "benchmark"
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
    # A counterparty that actually trades. Empty here is the mirror-only path:
    # `compose_opponents` short-circuits to `list(base) or list(pool)`.
    assert config["opponents"], "an empty base forces pure self-play"
    assert driver.DONOR in config["opponents"]
    # Dead while the base was empty; live now, and 0.65 is what every arm above 16k ran.
    assert 0.0 < config["self_play_ratio"] <= 0.9
    # A rejected child is the behaviour the next update should not imitate.
    assert config["self_play_latest_backfill"] is False
    # No teacher below the stake. 3,000 is what a policy that never trades scores.
    assert config["self_play_min_score"] >= 3000.0


def test_the_activity_floor_sits_in_the_regime_that_makes_money():
    # Contested money is monotone in this quantity over all recorded gates, and every
    # arm that cleared 10k ran 0.40-0.86. crop-v2 cleared the old 0.02 at 0.051 while
    # scoring 616 coins. It gates promotion, `member_rank_active` and the prune tier.
    assert driver.FREE_CONFIG["viability_min_orders_per_decision"] >= 0.20


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


def test_a_single_agent_has_no_cross_slot_reseed_and_keeps_its_snapback():
    """The two halves of "one agent with rollback".

    `admit_foreign` has no lineage dedup and its `max_adopt` default is
    `max(1, members - 1)`, whose floor takes the only seat at one member -- anchor-v1
    forked all five slots from a single checkpoint at the g0 prune that way, off a
    source scoring 8,016 on one 100-seed draw and 1,260 on the next. With one learner
    there is nothing to reach across anyway, so it is off, and the reject-streak
    snapback is the recovery path instead.
    """

    config = driver.FREE_CONFIG
    assert driver.LEARNERS == 1
    assert config["cross_slot_reseed"] is False
    assert config["resume_reset_after_rejections"] >= 2, "2 or more, so one noisy arena draw cannot snap back"
    # The retained checkpoint history, used as both the mirror pool and the bar.
    assert config["self_play_pool"] == 25
    assert config["promotion_reference_window"] == 25


def test_the_rollback_path_is_reachable_at_one_member():
    """`lineage_checkpoint` was defined and called from nowhere in this fork.

    `_member_checkpoint` took the resume unconditionally, and its docstring justifies
    dropping the streak fields by saying pruning supersedes them -- which is false at
    a roster of one, where the prune has a single seat and drops nobody.
    """

    from pathlib import Path

    updater = Path("agents/dqn_market_free_v3e/training/updater.py").read_text()
    assert "lineage_checkpoint(context, reader)\n            if members <= 1" in updater


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
