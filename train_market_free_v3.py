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
    WHERE model_id = 'dqn_market_free_v3.raw_order_q_v1@herd_00'
    ORDER BY batch_seq;

What each session changed and measured: `docs/dqn_market_free_sessions.md`.
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
# A new session, not a resume, and the third time on this fork.
#
# This one moves for a different reason than the last two, and the distinction
# matters because it is the reason a resume was even considered. The reward, the
# TD target, the action mask and `order_refused` are all *unchanged* from
# `market-free-credit-v1`, so its 10,880 episodes are the right kind of sample and
# its shaped returns are comparable with these. Nothing about the buffer forces a
# fork.
#
# `noisy_sigma0` does. Sigma is a trained parameter that lives in the checkpoint,
# initialised once at construction (`training/network.py:70`) and otherwise only
# ever written by a reseed (`training/updater.py:1671, 4438`, `sigma_reset=`). So
# raising it on a resumed session reaches exactly nothing: all ten slots keep the
# sigma they were minted with, and `sigma_mean_0` is measured not to move under
# training -- v2 went 0.0426 -> 0.0354 across 330 batches, and `credit_*` sat at
# 0.021 for its whole life. A resume would silently apply the fix to no one.
#
# What credit-v1 measured, and why sigma is what moved:
#
#   * `noisy_flip_to_order_rate` ran 0.028-0.083 against a 0.05 target, while v2 on
#     the same target ran 0.19-0.33 and was the run that learned.
#   * `noisy_sigma_scale_target` climbed to 1.0-1.46 with four to five of the ten
#     slots pinned at the 2.0 ceiling. The controller was pushing *up* and still
#     could not reach the target.
#
# That is the upper-saturation signature the `noisy_sigma0` comment below predicted
# for itself, and its own stated remedy is to raise it back. Halving sigma at birth
# halved the absolute perturbation the head can ever carry, and the scale
# multiplier could not buy it back inside its own limit.
#
# `ensure_session` would refuse the resume anyway once `batch_size` moves, but the
# checkpoint-resident sigma is the reason this had to move regardless.
# ---------------------------------------------------------------------------
# A fourth session on this fork, and the first one that shows this policy an agent
# it is not.
#
# Collection is 100% mirror (`opponents = []`), so every opponent this lineage has
# ever met is a copy of itself or the one pilkwang benchmark. v2's
# `market-free-herd-v1` measured what that buys: over batches 190-330 rollout money
# rose 16,899 -> 19,373 while the benchmark median the gate reads went
# 12,105 -> 11,651. Skill against copies of yourself is not skill, and the
# `opponents` comment below has named a donor entry as the fix for three sessions
# without one being available -- the donor is a promotion-gate benchmark whose
# matches are never persisted, so it has no rows to clone and no revision to pair
# against.
#
# `data/kaggle_episodes` does. `scripts/pull_kaggle_episodes.py` mirrors the
# competition's public episode datasets -- top 50 a day by `avg_score`, 22 days,
# 1,050 replays -- and they are unrelated ladder agents playing the same 720-step,
# 3,000-coin simulator. `training/ladder.py` converts them into ordinary
# `EpisodeRecord`s through the same `clone.record_from_replay` the league source
# uses, and half the roster pins 96 of them.
#
# What the conversion measures on this fork, because it decides how to read the
# metrics:
#
#   * `mapped_fraction` 0.997 and `orders_unrepresentable` **0**. The ladder's
#     orders are in this action space; nothing is being silently dropped toward
#     NOOP, which is the failure `CloneStats` exists to catch.
#   * `orders_masked_infeasible` ~25%. Not missing ops -- missing *rungs*: ladder
#     agents buy quantities past what `pending.seed_room` allows. `clone` forces
#     the bit so the label survives, so that share of the imitation gradient points
#     at actions the argmax cannot reach in play. A known cost, and the first thing
#     to revisit if the imitation loss falls without the money following.
#
# Five slots seeded and five held back; see `SEEDED_SLOTS` for why the control half
# is only a control until the first graduation crosses between them.
#
# `SLOT_PREFIX` does **not** move. The reward, the mask and the network are all
# unchanged from `market-free-sigma-v1`, so `sigma_*`'s batches, lineage and mature
# members are the right kind of sample and are inherited; the demonstrations are
# then the only variable that moved. What forces a new `SESSION_ID` is not the data
# but `ensure_session`, which digests `experiment.training_agents` -- a per-slot
# config key is a manifest change and a resume is refused.
#
# `market-free-sigma-v1` must stay stopped. Both sessions write the same
# `model_id`s, and two writers interleave `training_batches.batch_seq` and mix one
# lineage into the other.
# ---------------------------------------------------------------------------
SESSION_ID = "market-free-kaggle-v1"
ARTIFACTS = ARTIFACTS_ROOT / "dqn_market_free_v3"
LEAGUE_CONFIG = Path("configs/dqn_market_free_v3_league.toml")
MANIFEST = Path("agents/dqn_market_free_v3/agent.toml")

# Ten concurrent learner slots under one declared model, which the harness
# namespaces as `dqn_market_free_v3.raw_order_q_v1@herd_00` and so on.
LEARNERS = 10
# Named for the reward it was fitted to. `_self_play_pool`, `lineage_checkpoint`,
# `all_batches` and the slot LR state are all scoped by `model_id`, so renaming
# severs all four at once -- which is required whenever the reward changes,
# because a stored shaped return is then a different quantity.
#
# Moved off "credit" with `SESSION_ID`. The shaped returns *are* comparable this
# time -- the reward did not change -- but the severance is still required, and for
# the sharper of the two reasons: `credit_*`'s mature checkpoints all carry the
# halved sigma this fork exists to undo. Left reachable, they would re-enter as
# reseed ancestors and mirror opponents and quietly reintroduce the exact parameter
# being changed.
SLOT_PREFIX = "sigma"
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
    # Doubled from v2's 400, back toward v1's 1200.
    #
    # Prioritisation is *episode*-level -- `_own_priorities` ranks whole episodes by
    # shaped return -- so this is the size of the ranked pool, not just a memory
    # dial. v1 ranked over 1200 trajectories and v2 over 400, and the comment that
    # sat here was copied from v1 without being updated. In rows v2 was not short
    # (its episodes held ~5.7x more decisions), but almost all of that excess was
    # the ten-slot wash, so it bought diversity of *rows* and none of trajectories.
    #
    # Affordable now because the wash mask cut decisions per turn from 5.71 to 1.77
    # and the record from ~5 MB to 2.28 MB: 800 episodes is ~1.8 GB measured on an
    # untrained policy. A trained one trades more, so watch `buffer_megabytes` in
    # `metrics_json` over the first few generations and come back down if it is
    # heading past ~4 GB. This is the one value here set from a measurement that
    # will drift as the policy learns.
    "buffer_episodes": 800,
    # No static base. `compose_opponents` returns the mirror pool unmixed when the
    # base is empty, which is how the updater spells pure self-play, and the check
    # for an empty mix sits *after* the pool has been consulted so this is a
    # supported configuration rather than a hole. Safe on a fresh prefix too since
    # v43: batch 0 has no pool to consult and the updater now seeds the quiet
    # network there instead of raising; see `SLOT_PREFIX`.
    #
    # Kept empty deliberately, and the cost is known rather than assumed. v2's
    # `market-free-herd-v1` ran 34 generations this way and the two money series
    # came apart over batches 190-330: rollout money rose 16,899 -> 19,373 while
    # the benchmark median the gate reads went 12,105 -> 11,651. A mirror-only
    # curriculum buys skill against copies of yourself. v3 keeps it anyway -- the
    # recurrence is the variable under test, and adding a donor to collection at
    # the same time would leave any change unattributable between the two.
    #
    # So watch for the same divergence rather than assuming it is gone: benchmark
    # median flat while `own_mean_final_money` climbs is the signature, and the
    # fix when it appears is a donor entry here.
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
    # against a one-entry base 0.65 yields [noise, self, self]. Moves with the base
    # size or the mix is neither of the things it was designed to be.
    #
    # Inert as configured: `opponents` above is empty, so there is nothing to mix
    # against and collection is 100% mirror rather than 65%. Left at 0.65 so the
    # value means something the moment a base entry is added.
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
    # 0.25, not 0.0. The old comment here said "no floor" because a floor had been
    # a binding constraint -- slots sat on it exploring two to four times harder
    # than asked and unable to come down. Setting it to 0.0 did not remove the
    # bind, it moved it: 0.0 means "use `1 / limit`", so with the limit at 2.0 the
    # lower clamp was 0.5, and v2's run then sat on *that* with exactly the symptom
    # the old comment describes. Measured over `market-free-herd-v1` batches
    # 180-330: `noisy_sigma_scale_target` pinned at 0.5 in 137 of 150 updates with
    # `noisy_flip_to_order_rate` at 0.219 against a 0.05 target -- 4.4x over, and
    # the controller with no authority left to cut.
    #
    # 0.25 doubles the room to come down while staying twice as high as the 0.125
    # that took `market-free-landgate-v10` quiet at update 20 -- the cold-start
    # failure `resolve_sigma_scale`'s own comment documents, and one this fork is
    # freshly exposed to because v3 is a cold start.
    #
    # Note the probe ladder is clipped by `scale_limit`, not by this, so it still
    # bottoms out at 0.5 and the resolver extrapolates to the floor when the whole
    # ladder over-explores. That is the saturated case, which is the case v2 was in.
    "noisy_sigma_scale_floor": 0.25,
    # Unchanged at 2.0. The problem measured on v2 was entirely on the *lower*
    # bound, and raising this to widen the downward range would also hand back the
    # upward runaway it is here to prevent -- a controller reaching high sigma at a
    # low flip-to-order rate, which nothing else catches. `floor` above is
    # independent (`low = floor if floor > 0 else 1 / limit`), so the two ends move
    # separately and only the end that needed moving has moved.
    "noisy_sigma_scale_limit": 2.0,
    # Back to the code default of 0.5, undoing credit-v1's halving. Note the
    # manifest *does* declare this key -- `agents/dqn_market_free_v3/agent.toml:361`
    # sets 0.5 -- so the previous comment's "the manifest declares no
    # `noisy_sigma0`" was wrong on the fact even though the override won either way.
    #
    # The standing observation is unchanged and is what makes this knob load-bearing:
    # sigma is a trained parameter that does not train. v2's `sigma_mean_0` went
    # 0.0426 -> 0.0354 across 330 batches, still sitting on its `sigma0/sqrt(fan_in)`
    # initialisation, and the falling `sigma_ratio_0` (7.0 -> 1.9) was mu growing
    # threefold rather than sigma shrinking. The absolute perturbation the head
    # carries is set at birth and stays there.
    #
    # Which is why halving it could not be bought back. credit-v1 ran
    # `noisy_flip_to_order_rate` at 0.028-0.083 against a 0.05 target while
    # `noisy_sigma_scale_target` climbed to 1.0-1.46 with four to five of ten slots
    # pinned at the 2.0 ceiling -- the controller pushing up against its own limit
    # and still short. v2, at 0.5, held 0.19-0.33 on the same target. The 0.25 arm
    # is answered: it went too far, exactly as the arm's own comment predicted it
    # would look if it had.
    #
    # `noisy_sigma_scale_floor` stays at 0.25 rather than reverting with this. The
    # two knobs treat opposite ends and only one of them was wrong: v2's complaint
    # was that it could not come *down* (floor 0.0 -> `1/limit` = 0.5), credit-v1's
    # is that it could not go *up*. Keeping both gives a minimum effective
    # perturbation of 0.5 * 0.25 = 0.125 against v2's 0.5 * 0.5 = 0.25, so the
    # downward room credit-v1 bought survives at the restored magnitude.
    "noisy_sigma0": 0.5,
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
    #
    # Back to v2's 1e-4. The doubling was compensation for a step count that has
    # since been restored: it was sized against "48 steps a batch against 400", and
    # `gradient_steps_per_batch` below is now 192, so the deficit it was paying for
    # is 2.1x rather than 8.3x. Paying for it twice -- 4x the steps *and* 2x the
    # rate -- would be 8x the parameter movement per update against a run that has
    # never been observed stable at any setting, and it would confound the sigma
    # arm this fork exists to measure. 1e-4 is the one value on this axis a run in
    # this family has actually learned at.
    #
    # The sweep the previous comment asked for is still open, and `mean_grad_norm`
    # is still the instrument -- it read 0.010-0.034 against a 5.0 clip on
    # credit-v1, so the clip never bound and there is headroom above this. Sweep it
    # after the geometry change has a baseline, not at the same time.
    "learning_rate": 1e-4,
    # v3's chunk geometry, restated here for the same reason the learning rate is:
    # this file decides what the run is. `burn_in_turns` must stay a multiple of
    # `collect.HIDDEN_STRIDE` or `sample_sequences` refuses the draw outright.
    # Chunks per gradient step, quartered, and `gradient_steps_per_batch` raised by
    # the same factor. This holds rows-per-update and wall clock roughly constant
    # while giving Adam 4x the steps, because the deficit credit-v1 measured is in
    # *steps*, not in data:
    #
    #   credit-v1   480 steps/update x ~2,750 rows = ~1.32M rows, ~2.3 replay epochs
    #   here      1,920 steps/update x   ~688 rows = ~1.32M rows, ~2.3 replay epochs
    #   v2        4,000 steps/update x    256 rows = ~1.02M rows, ~1.5 replay epochs
    #
    # Raising the step count alone would have been 4x the rows too -- ~9.4 replay
    # epochs an update against a buffer of 800 episodes, which is a reuse rate
    # nothing in this family has run at, and ~17% of wall clock against ~4%.
    # Shrinking the chunk count pays for the steps out of the batch instead of out
    # of the buffer.
    #
    # ~688 rows a step is still 2.7x v2's 256, so this is not a small-batch regime;
    # it is v2's step count within 2.1x at a larger batch than v2 ever used.
    # `burn_in_turns` and `train_turns` are untouched -- the chunk *shape* is
    # unchanged, only how many of them ride in a step.
    "batch_size": 8,
    "burn_in_turns": 20,
    "train_turns": 40,
    "gradient_steps_per_batch": 192,
    # Scaled with `gradient_steps_per_batch` to hold two syncs an update, which is
    # the invariant v2 and credit-v1 both ran at -- v2 at 2000 against ~4,000 steps
    # an update, credit-v1 at 250 against ~480. An update here is ~1,920 steps
    # (192 x `min_new_batches` 10), so 960.
    #
    # This is the coupling the manifest comment at `agent.toml:198-201` calls "the
    # single easiest value to leave stale across the fork, and nothing would report
    # it: a target that lags four updates just looks like a harder target". Left at
    # 250 alongside the 4x step raise it would have gone the other way -- ~7.7 syncs
    # an update, a target net moving nearly four times faster than any run in this
    # family has bootstrapped against, and equally unreported.
    "target_sync": 960,
    "grad_clip_norm": 5.0,
    # Moves back down with the peak. 1e-5 against a peak of 1e-4 needs two decays at
    # the manifest's 0.25, which at `lr_plateau_patience = 4` is twelve updates and
    # so twelve generations here -- the same ladder v2 walked. Held at 2e-5 against
    # the restored 1e-4 peak this would be a 1:5 ratio and graduate a slot one decay
    # early.
    "lr_graduate_cut": 1e-5,
    # Safe now that `lr_score_signal` is "rollout": the detector reads the ring
    # the slot just trained on, so a snapback moves it rather than pinning it.
    # Buys the case with no other answer -- a diverged slot has nothing to climb
    # back to, and `lr_best_decay` only unsticks a detector still improving.
    "resume_reset_after_rejections": 2,
}

# Half the roster's departure from the other half, and the whole experiment. The
# pack is built offline, once:
#
#   uv run python -m agents.dqn_market_free_v3.training.ladder \
#       --out artifacts/dqn_market_free_v3/demonstrations/ladder-001 \
#       --limit 320 --min-money 50000
#
# It holds more than the 96 pinned here and is ordered by ladder score, so
# `demonstration_episodes` is a dial over the head of it rather than a rebuild.
# Pinned episodes are never evicted, so they are a permanent addition to the
# buffer that `buffer_episodes = 800` already sizes; a v3 record is ~2.28 MB, so
# 96 of them is roughly 220 MB against that ring's ~1.8 GB. `demonstrations.
# megabytes` in `metrics_json` is what to read before moving this.
#
# `demo_fraction` is left at the manifest's 0.25, so a seeded slot draws a quarter
# of every TD minibatch from the ladder and runs a behavior-clone phase that is
# 100% ladder on top of it.
LADDER_DEMOS = {
    "demonstration_source": "ladder",
    "demonstration_pack": str(ARTIFACTS / "demonstrations" / "ladder-001"),
    "demonstration_episodes": 96,
}

# Slots 00-04 take them; 05-09 stay pure self-play. Five and five rather than v3b's
# one and nine because the read has to survive the reseed pool: `_graduates` and
# `_warm_starts` both draw from `population_members` keyed on `declared_model_id`,
# which the whole roster shares, so a single seeded slot's evidence is one slot
# wide until it graduates and then everyone's at once.
#
# Which is also the caveat on this experiment, stated rather than discovered later:
# the halves are independent only until the first graduation crosses between them.
# Read the seeded-vs-control gap over the early generations; after that this
# measures whether the demonstrations helped the *roster*, which is a different
# question and the one v3b's design was actually built to answer.
SEEDED_SLOTS = frozenset(range(LEARNERS // 2))

# The roster is uniform except for the demonstrations: the archive handles
# retention, and `rare_sigma_gain` already handles generation on every slot.
# `LeagueConfig.validate` requires exactly one `primary` per agent type, so slot 0
# carries it -- deliberately not re-derived from scores, which would make every
# query resolving it non-reproducible. Distinct seeds are the only other thing
# separating the slots at the start; what diverges them afterwards is which mature
# member each is reseeded from and which weights the perturbation redrew.
TRAINING_AGENTS = [
    {
        "type": "dqn_market_free_v3",
        "name": f"{SLOT_PREFIX}_{index:02d}",
        "train": True,
        "primary": index == 0,
        "model_config": {
            **FREE_CONFIG,
            "seed": 2718 + 1009 * index,
            **(LADDER_DEMOS if index in SEEDED_SLOTS else {}),
        },
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
