"""Pins on `train_market_free_v3c.py`'s constants.

`train_market_free.py` has had these since `landheuristic-v43`; this driver had
none, and it is the file another session is most likely to edit under a live run.
What is pinned is the *relationship* between the keys, not a snapshot of every
value: the session and prefix that name the run, and the overrides that the
`rewardtogo-v1` review settled on and that would silently reintroduce a measured
failure if flipped back.
"""

from __future__ import annotations

import train_market_free_v3c as driver
from agents.dqn_market_free_v3c.training.reward import MONEY_SCALE


def test_the_session_and_prefix_name_the_n_step_run():
    # The prefix names the run: a reward change is a prefix sever (see
    # `SLOT_PREFIX` in the driver), so the two must move together.
    assert driver.SESSION_ID.startswith(f"market-free-{driver.SLOT_PREFIX}-v")
    names = [item["name"] for item in driver.TRAINING_AGENTS]
    assert names == [f"{driver.SLOT_PREFIX}_{index:02d}" for index in range(driver.LEARNERS)]
    assert driver.MODEL_NAME == names[0]
    assert sum(item["primary"] for item in driver.TRAINING_AGENTS) == 1
    assert MONEY_SCALE == 25_000.0


def test_the_overrides_the_review_settled_on_are_still_set():
    config = driver.FREE_CONFIG
    # Promotion decides what is released; the lineage trains on regardless.
    assert config["lineage_source"] == "latest"
    # Inert under member pools (`updater._member_checkpoint`); the manifest's 0
    # is the honest value and the driver must not claim otherwise.
    assert "resume_reset_after_rejections" not in config
    # `agent.toml`'s 400 is 2.5 generations of a member's own history at
    # `member_batches = 10`; 160 was the one-generation regime that failed.
    assert config["buffer_episodes"] == 400
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
    # The mirror pool must track the trained lineage, not the frozen release.
    assert config["self_play_latest_backfill"] is True
    assert config["rare_sigma_pairs"] <= 2 and config["rare_sigma_cap"] <= 3.0


def test_every_override_key_is_one_the_manifest_or_updater_knows():
    """A typo in `FREE_CONFIG` is silently ignored by `config.get`; this catches it."""

    import tomllib
    from pathlib import Path

    manifest = tomllib.loads(Path(driver.MANIFEST).read_text())["models"][0]["config"]
    updater = Path("agents/dqn_market_free_v3c/training/updater.py").read_text()
    collect = Path("agents/dqn_market_free_v3c/training/collect.py").read_text()
    match_agent = Path("agents/dqn_market_free_v3c/match_agent.py").read_text()
    unknown = [
        key
        for key in driver.FREE_CONFIG
        if key not in manifest
        and f'"{key}"' not in updater
        and f'"{key}"' not in collect
        and f'"{key}"' not in match_agent
    ]
    assert unknown == [], unknown
