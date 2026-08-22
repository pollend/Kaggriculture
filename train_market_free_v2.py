#!/usr/bin/env python3
"""Run the resumable free-market workflow, multi-order variant.

Same shape as `train_market_scorer.py` and the same benchmark, so the two are
directly comparable. The difference is that this agent has no `market_exec`, no
donor market rules and no `_schedule_market_adjustment`. Where v1 picked one order
per turn out of the flat list, this one picks up to ten -- each priced against the
wallet, shed and quote its own earlier orders already moved, with NOOP as the
terminator. Why: `docs/dqn_market_free_kaggle_episode_review.md`.

Four failure modes, recorded per batch in `metadata_json`. The first two are v1's,
the last two are what a turn-sized action budget adds:

    orders_per_decision ~ 0        the NOOP collapse: it stopped trading
    mean_final_money    ~ 0        the bankruptcy collapse: it traded itself broke
    decisions_per_turn  ~ 1        the STOP collapse: it never opens a turn
    decisions_per_turn  ~ 11       slot stuffing: it fills the wire with junk

A healthy run sits between them with `solvent_episodes` climbing off zero.

    SELECT batch_seq,
           metadata_json ->> 'orders_per_decision' AS orders,
           metadata_json ->> 'mean_final_money'    AS money,
           metadata_json ->> 'solvent_episodes'    AS solvent,
           metadata_json ->> 'decisions_per_turn'   AS per_turn
    FROM training_batches
    WHERE model_id = 'dqn_market_free_v2.raw_order_q_v1@herd_00'
    ORDER BY batch_seq;

What each session changed and measured: `docs/dqn_market_free_v2_sessions.md`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

from kaggriculture_lab.constants import ARTIFACTS_ROOT, REPO_ROOT
from kaggriculture_lab.identity import TrainingAgentSpec
from kaggriculture_lab.league import LeagueConfig
from kaggriculture_lab.model_admin import model_status
from kaggriculture_lab.runner import ensure_session, run_generations
from sim.workers import install_signal_handlers

# Edit these experiment constants. The runner intentionally accepts no flags.
# The run is unbounded: there is no annealing schedule to wait out, so a stopping
# point is a judgement about the metrics above. Ctrl-C after a generation closes.
SYNC_ENVIRONMENT = True

# `ensure_session` digests the league config and `experiment.training_agents`, so
# any change below makes the previous session unresumable and this has to move.
# Reward constants are *not* in that digest -- when one of those changes, move
# `SLOT_PREFIX` too or one session's history holds two reward regimes.
#
# Live question: through v39 the lot ledger was seed-only, so a 400-coin cow was
# unopposed on the turn it was bought -- `shaped` carries no money delta -- and the
# milk it sold for the rest of the season was invisible to every shaping term. At
# `TARGET_HERD = 15` that is up to 6,500 coins the stream had no opinion about, in
# the two highest-priced non-melon products. `lots.py` now matches animals on the
# same two legs, with a lot's capacity set by the season's remaining production
# events rather than a `max_yield`, and `ANIMAL_WASTE_WEIGHT` / `ANIMAL_GAIN_WEIGHT`
# pay the result onto the buy turn. Watch `animal_gain_credit` against
# `animal_waste_cost`: the charge climbing with the credit flat is the herd being
# bought and never placed, or bought past the day it could still yield.
# v41 drops the static base entirely: the mix is now this family's own checkpoints
# and nothing else. `seeded_random` held a third of every batch (`compose_opponents`
# solves the one-entry base to `[noise, self, self]`) and it is not a competitor --
# it never emits `SELL`, the only op that credits money, so its wallet walks the
# seed-price grid down from 3,000 and lands on exactly 0 in every episode it has
# ever played. v40:g0002 measured the cost: identical behaviour from the learner
# either way (18.4% vs 19.9% productive, 47.7% vs 43.3% idle) for 3,312 money
# against the noise and 2,432 against a real policy. A third of the fitted return
# was a 36% premium for an opponent that does not exist in the arena.
#
# v43 takes `BUY_LAND` off the network entirely. v42 had already pulled it out of
# the argmax and hung it on `Q(BUY_LAND) - Q(NOOP)`, and that did not take, because
# the row it read is dead: `buy_land_rate` ran 0.000-0.050 across all ten slots over
# batches 351-390, against v33's 0.37. A gated row is starved of gradient, so its Q
# is arbitrary in exactly the states where the gate opens, and no set point placed on
# an arbitrary number is a decision rule.
#
# `order_space.land_score` decides it now, as a product of three factors the network
# never saw priced together -- how late the purchase is, whether the land already
# owned is being worked, and whether the wage bill for the tiles afterwards is
# covered. The reserve is the tile-count term: hands are re-hired every day on a
# fibonacci ladder, so 25 more tiles means a larger daily bill for every remaining
# day. `LAND_BUY_SET_POINT = None` reverts to the plain argmax.
#
# The prefix moves with it, because `LAND_PURCHASE_BONUS` goes to 0.0 and the rule
# is that the prefix moves when the reward moves. That was blocked until now: a
# fresh prefix restarts at batch 0, where `self_play_after` is read as
# `batch_index >= 1` and the empty `opponents` base therefore reached the updater
# with no mix at all and raised. It seeds the quiet network there instead -- one
# batch against an opponent that emits nothing, which is the same baseline the
# promotion gate already treats as a real incumbent. A rule that could not be
# followed was not protecting anything.
# A new session, not a resume. The state vector widened from 249 to 259 (the herd
# viability block) and the sell ladder is depth-capped, so every stored checkpoint is
# a different function on a different action distribution: `policy.load` refuses the
# width outright, and the replay buffer's transitions were collected under a mask
# that no longer exists.
SESSION_ID = "market-free-herd-v1"
ARTIFACTS = ARTIFACTS_ROOT / "dqn_market_free_v2"
LEAGUE_CONFIG = Path("configs/dqn_market_free_v2_league.toml")
MANIFEST = Path("agents/dqn_market_free_v2/agent.toml")

# Ten concurrent learner slots under one declared model, which the harness
# namespaces as `dqn_market_free_v2.raw_order_q_v1@herd_00` and so on.
LEARNERS = 10
# Named for the reward it was fitted to. `_self_play_pool`, `lineage_checkpoint`,
# `all_batches` and the slot LR state are all scoped by `model_id`, so renaming
# severs all four at once -- which is required whenever the reward changes,
# because a stored shaped return is then a different quantity.
SLOT_PREFIX = "herd"
MODEL_NAME = f"{SLOT_PREFIX}_00"

# How many learners run their round hooks and their update at once, and how wide
# each one's rollout pool is. The product is the core budget, so these two belong
# together. `training_concurrency` is a `ThreadPoolExecutor` width over *models*
# consumed by `list(pool.map(...))`, so it is a hard barrier: ten models over
# five threads is 2 x 5 with no half-empty tail.
CONCURRENCY = 5
JOBS_PER_LEARNER = 8

DONOR = "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"

# The release objective, and still the only benchmark. Index 0 is what the ratchet
# reads; a second entry would only add diagnostics the gate does not consult.
BENCHMARKS = (DONOR,)

# Overrides only. `register_training_agent` materializes `agent.toml`'s
# `[models.config]` and then applies this on top, so every key here is a
# deliberate departure from the manifest.
FREE_CONFIG = {
    "benchmarks": list(BENCHMARKS),
    "episode_steps": 720,
    "jobs": JOBS_PER_LEARNER,
    # Read only by `_reseed_source`'s whole-roster spread, which needs to know
    # which ordinals are the tail. Declared rather than derived because a slot is
    # a separate process that never sees the others.
    "population_slots": LEARNERS,
    # The floor grace falls back to. Zero does not stand the gate down:
    # `no_regression` still measures against the incumbent's recorded reference,
    # and hard viability never stands down at all.
    "benchmark_grace_min_money": 0.0,
    # Three generations of ingest, so a batch survives long enough to be sampled
    # more than once. Costs memory linearly -- roughly 1 GB at this size.
    "buffer_episodes": 400,
    # No static base. `compose_opponents` returns the mirror pool unmixed when the
    # base is empty, which is how the updater spells pure self-play, and the check
    # for an empty mix sits *after* the pool has been consulted so this is a
    # supported configuration rather than a hole. Safe on a fresh prefix too since
    # v43: batch 0 has no pool to consult and the updater now seeds the quiet
    # network there instead of raising; see `SLOT_PREFIX`.
    #
    # What this gives up, and why it is accepted: the run no longer plays the
    # donor it is gated on. `train_dqn.py` recorded 17 consecutive rejections in
    # that state. The exits are `benchmark_grace_min_money = 0.0` below plus the
    # manifest's `promotion_ratchet_grace_after_rejections`, and the mature pool is
    # 24 members spanning 4,140-19,347 rather than the saturating wall the old
    # comment was guarding against. If the gate does stall, the fix is a donor
    # entry here, not the noise agent back.
    "opponents": [],
    # The league population is this same family, and `league_opponents` excludes
    # it by construction, so leaving this on would only re-add anchors.
    "league_opponents": False,
    # 1, not 0: read as `if self_play_after and batch_index >= self_play_after`,
    # so 0 is the manifest's *disable* value. The real gate is `_self_play_pool`
    # being non-empty, and a checkpoint is only written by a model update, so
    # generation 0 is noise-only warmup whatever this says.
    "self_play_after": 1,
    "self_play_pool": 5,
    # `compose_opponents` solves `slots / (len(base) + slots)` in whole slots, so
    # against this one-entry base 0.65 yields [noise, self, self]. Moves with the
    # base size or the mix is neither of the things it was designed to be.
    "self_play_ratio": 0.65,
    # No demonstrations. The pilkwang donor is the family worth cloning and it has
    # zero rows in `episode_players`: it only appears as a promotion-gate
    # benchmark, and those matches are not persisted as league episodes. To clone
    # it, they have to be persisted first.
    "demonstration_episodes": 0,
    # NoisyNet takes over exploration. Epsilon stays declared at 0.0 rather than
    # being removed, because it is part of the policy identity string.
    "noisy": True,
    "noisy_layers": "head",
    # Per episode, for temporally extended commitment: a perturbation redrawn
    # every 24 turns cannot hold a plan across the 48-96 turn BUY_SEED-to-harvest
    # cycle. Costs 8 distinct policies per batch against day cadence's 480, and
    # one bad draw wastes a whole episode. The per-actor sigma ladder is what
    # spreads behaviour across the eight pairs instead.
    "noisy_cadence": "episode",
    # Control sigma on the flip-to-order rate, not the argmax flip rate: on a
    # NOOP-collapsed policy every action is near-tied, so the argmax rate reads as
    # exploring and throttles every actor. Setting this ignores
    # `noisy_flip_rate_target`, which is zeroed below so the config says so.
    "noisy_order_rate_target": 0.05,
    # No floor. It was a binding constraint rather than a floor -- slots sat *on*
    # it exploring two to four times harder than asked and unable to come down.
    # The cold-start case it was bought for is covered by `lr_warmup_updates` and
    # `viability_cold_start_reference_money`.
    "noisy_sigma_scale_floor": 0.0,
    # Sized to what a converged slot resolves, not to a clamp that never binds. A
    # runaway controller reaches high sigma at a *low* flip-to-order rate, which
    # nothing else catches.
    "noisy_sigma_scale_limit": 2.0,
    # Directed exploration: the head's sigma row for an action this run has barely
    # sampled is drawn wider than the rest. `epsilon_out` is per-output, so one
    # entry moves one action. The manifest defaults the gain to 0.0 and stays the
    # undirected control arm.
    #
    # 0.5 is where the rate arm's ordering is widest without saturating; at 0.75 a
    # third of the space hits the cap and the ordering between rare rows is
    # destroyed, which is the whole thing being bought.
    "rare_sigma_gain": 0.5,
    # Selection rate given feasibility, against the rate under evenly spread
    # attention. A raw count cannot tell a rare action from a declined one.
    "rare_sigma_novelty": "rate",
    # High enough that nothing saturates, so the ordering across rare rows is what
    # the novelty statistic says it is. Not a lever on the crop distribution --
    # none of the crop rows are clamped at either setting.
    "rare_sigma_cap": 6.0,
    # 4 of 8. Raising it is nearly worthless: `actor_sigma_scales` runs
    # [1.0, 0.61, 0.37, 0.23, 0.138, 0.084, 0.051, 0.031], so rungs 5-8 carry a
    # sigma scale under 0.14. The ladder is the attenuator, not the pair count.
    "rare_sigma_pairs": 4,
    # Re-applied from config each update, and under the population it is the
    # schedule's *peak*: a slot warms up to it from zero after a reseed and halves
    # away from it on every plateau.
    "learning_rate": 1e-4,
    # Declared here as well as in the manifest because this file decides what the
    # run is. 1e-5 against a peak of 1e-4 needs two decays at the manifest's 0.25,
    # which at `lr_plateau_patience = 4` is twelve updates and so twelve
    # generations here.
    "lr_graduate_cut": 1e-5,
    # Safe now that `lr_score_signal` is "rollout": the detector reads the ring
    # the slot just trained on, so a snapback moves it rather than pinning it.
    # Buys the case with no other answer -- a diverged slot has nothing to climb
    # back to, and `lr_best_decay` only unsticks a detector still improving.
    "resume_reset_after_rejections": 2,
}

# The roster is uniform: the archive handles retention, and `rare_sigma_gain`
# already handles generation on every slot. `LeagueConfig.validate` requires
# exactly one `primary` per agent type, so slot 0 carries it -- deliberately not
# re-derived from scores, which would make every query resolving it
# non-reproducible. Distinct seeds are the only thing separating the slots at the
# start; what diverges them afterwards is which mature member each is reseeded
# from and which weights the perturbation redrew.
TRAINING_AGENTS = [
    {
        "type": "dqn_market_free_v2",
        "name": f"{SLOT_PREFIX}_{index:02d}",
        "train": True,
        "primary": index == 0,
        "model_config": {**FREE_CONFIG, "seed": 2718 + 1009 * index},
    }
    for index in range(LEARNERS)
]

MAIN_CONFIG = REPO_ROOT / "main.toml"


def _device() -> str:
    with MAIN_CONFIG.open("rb") as stream:
        return str(tomllib.load(stream).get("compute", {}).get("device", "auto"))


def _prepare_environment() -> None:
    if SYNC_ENVIRONMENT:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to synchronize the environment")
        subprocess.run(
            [uv, "sync", "--locked", "--python", "3.12"],
            cwd=REPO_ROOT,
            check=True,
        )


def _effective_config() -> LeagueConfig:
    base = LeagueConfig.from_toml(REPO_ROOT / LEAGUE_CONFIG)
    return replace(
        base,
        training_agents=tuple(TrainingAgentSpec.from_dict(item) for item in TRAINING_AGENTS),
        training_concurrency=CONCURRENCY,
    )


def _ensure_session() -> None:
    config = _effective_config()
    created, _ = ensure_session(
        config,
        SESSION_ID,
        workspace=REPO_ROOT,
        artifacts=ARTIFACTS,
        runner="train_market_free",
        experiment={"training_agents": TRAINING_AGENTS},
    )
    print(
        f"{'Creating' if created else 'Resuming'} market-free session: {SESSION_ID}",
        flush=True,
    )


def _model_status() -> None:
    print(model_status(REPO_ROOT / MANIFEST, name=MODEL_NAME))


def main() -> int:
    if len(sys.argv) != 1:
        raise SystemExit("train_market_free.py accepts no arguments; edit its constants")

    # Before anything spawns: SIGTERM has to reap the worker pools rather than
    # orphan them, and a wedged run has to be dumpable with `kill -USR1 <pid>`.
    install_signal_handlers()

    os.environ["KAGGRICULTURE_TORCH_DEVICE"] = _device()
    _prepare_environment()
    _ensure_session()

    def report(generation: int, _: dict) -> None:
        print(f"\n=== market-free generation {generation} complete ===", flush=True)
        _model_status()

    try:
        run_generations(
            session_id=SESSION_ID,
            artifacts=ARTIFACTS,
            # Continuous mode never queues: the next generation is created only
            # after the current one has closed, so `generations` is only the
            # positional floor the runner validates.
            generations=1,
            continuous=True,
            workspace=REPO_ROOT,
            on_generation=report,
        )
    except KeyboardInterrupt:
        print(f"\n[{SESSION_ID}] stopped; committed work remains resumable", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
