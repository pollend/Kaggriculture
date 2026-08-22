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
    WHERE model_id = 'dqn_market_free_v3c.raw_order_q_v1@herd_00'
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
#
# ---------------------------------------------------------------------------
# A fourth session, and the first one on this fork that moves the *optimizer*
# rather than the exploration arm.
#
# `market-free-refusal-v1` ran 17 generations and lost to v2 at every matched
# update index -- 1,254 against 5,954 at update 3, 3,205 against 12,145 at 17 --
# with `own_mean_return` pinned at -1.3 to -1.6 while v2 crossed zero by
# generation 12. So the recurrence was never the thing being measured; the
# gradient was.
#
# Three findings from it, each with its knob above:
#
#   * The chunk loss reduces to one scalar per chunk before the outer mean
#     (`updater.py:3633-3643`), so `batch_size = 8` meant eight independent
#     gradient directions a step against v2's 256 -- not the ~688 rows the old
#     comment counted. `batch_size` is 32 and `gradient_steps_per_batch` 48.
#   * `mean_grad_norm` read 0.017 against a 5.0 clip, and `max_abs_q` 0.6 against
#     v2's 1.0-2.3. The head's weights were not growing, which is also why
#     `sigma_ratio_0` sat at 12 where v2 annealed to 1.9 -- that ratio falls
#     because mu grows, not because sigma shrinks. `learning_rate` is 3e-4.
#   * `buffer_megabytes` reached 4.47 GB against the ~4 GB watch-condition the
#     `buffer_episodes` comment set for itself. It is 400.
#
# Deliberately *not* touched, because credit-v1 already answered them: sigma0
# stays 0.5 (halving it pinned four to five slots at the scale ceiling), and
# `n_step` stays 24 (one in-game day, with the credit gap closed by
# `LOT_POTENTIAL_WEIGHT` rather than by the horizon).
#
# `order_space.py` also gains a housing and calendar gate on `BUY_ANIMAL`, which
# mints a new revision through `code_paths`. That is a mask *narrowing* -- every
# action still open was open before -- so the previous run's checkpoints load and
# play under it unchanged.
# ---------------------------------------------------------------------------
# Split credit: half the lot charge at the buy, half where the lot resolved.
#
# `market-free-flow-v1` answered the state question and closed it. Block F -- own
# flow, opponent flow, town drain, the impact curve inverted -- removed the trough
# outright: the control fell to 1,093 at batch 60 and spent ~70 batches under its
# 3,000 stake, while flow-v1 never dipped below stake and led 3.6x at the floor.
# Then it flattened. Its last six updates were +311, +95, +111, +84, -26, -24,
# parked at ~5,450, while the control climbed past it to 9,127 by batch 240. A
# ~90-batch sample-efficiency win and no better ceiling.
#
# So the binding constraint is not what the policy can see. `flow_04`, the first
# member to graduate, says what it is: 5,303 coins an episode of animal waste
# against 2,032 of credit, `animal_margin_goose` -66,132, 17.6 cows and 17.1 geese
# bought per episode. It keeps buying stock it cannot recover.
#
# `reward.LOT_CREDIT_BUY_SHARE` is the variable. The four lot legs were paid whole
# onto the buy turn, which `SEED_WASTE_WEIGHT` documents as deliberate and expensive:
# "a fact about the rest of the episode paid at a turn whose state cannot see it, so
# two identical buy states score differently and the network can only fit the mean
# ... at the cost of target variance on buy turns." At 0.5 the deterrent stays at the
# decision -- moving it out entirely would make buying-and-never-selling locally
# cheapest again, which is the failure that leg exists to prevent -- and the other
# half lands on `lots.resolution_turn`, the lot's last attributed sale, or the final
# turn for a lot that never sold. Each lot still contributes at most +/- its own
# cost; only the index changes.
#
# Strong move by the rule at the top of this file: a stored shaped return is now a
# different quantity, so `SLOT_PREFIX` moves with `SESSION_ID` and flow-v1's
# buffers, lineage, self-play pool and LR state are all severed. The state schema is
# unchanged at 280, so this is a like-for-like arm against flow-v1 -- read it against
# that curve, not against the v3b control.
#
# Pass condition, and it is not "money goes up": waste per episode falling relative
# to credit on both ledgers while `lr_score` holds flow-v1's early curve. Waste
# falling because the policy stopped trading is the failure mode, and
# `orders_per_decision` near zero is how it shows.
# ---------------------------------------------------------------------------
# Continued as `market-free-splitladder-v1`, which adds the one thing this fork has
# never had: an opponent it is not.
#
# Collection is 100% mirror (`opponents = []`), so every opponent this lineage has
# met is a copy of itself or the one pilkwang benchmark. v2's `market-free-herd-v1`
# measured what that buys -- over batches 190-330 rollout money rose 16,899 ->
# 19,373 while the benchmark median the gate reads went 12,105 -> 11,651.
#
# `data/kaggle_episodes` is the fix that was actually available. The mirror holds
# 1,050 public competition replays (top 50 a day by `avg_score`, 22 days, 31
# distinct ladder agents) and `training/ladder.py` converts them through the same
# `clone.record_from_replay` the league source uses -- so they arrive as ordinary
# `EpisodeRecord`s under *this* fork's encoder and this fork's reward, which is why
# the pack is rebuilt here rather than borrowed from v3b.
#
# Half the roster pins 96 of them; see `SEEDED_SLOTS`. `demo_fraction` stays at the
# manifest's 0.25, so a seeded slot draws a quarter of every TD minibatch from the
# ladder and runs a 100%-ladder behavior-clone phase on top of it.
#
# `SLOT_PREFIX` does **not** move. The split-credit reward, the mask and the network
# are all unchanged, so `split_*`'s batches, lineage and mature members are the
# right kind of sample and are inherited; the demonstrations are then the only
# variable that moved. What forces a new `SESSION_ID` is `ensure_session`, which
# digests `experiment.training_agents` -- a per-slot config key is a manifest change
# and a resume is refused.
#
# Block F's pass condition above still stands and is now read on the control half.
# ---------------------------------------------------------------------------
# Continued as `market-free-cashweight-v1`, which moves the reward for the first
# time on this fork since split-credit, and moves exactly one constant:
# `reward.OUTCOME_LOG_SCALE`, 5.0 -> 2.5.
#
# Block G's premise is that the previous blocks kept solving the wrong quantity.
# `splitladder-v1` at cutoff 120 reads `unrecovered_seed_share` and
# `unrecovered_animal_share` at 0.0 on nine of ten slots, seed margin +36,934 an
# episode and animal margin +5,272 -- the split-credit objective is *met* -- with a
# best slot on 10,868 money. Decomposing that slot's `own_mean_return` of -0.261:
# outcome +0.252, the four lot legs +0.155 net, and therefore about -0.67 of
# penalty. On `split_02` the penalty side is about -1.11. This is v26's finding
# ("the objective was 98% penalty") unfixed, and the part of it that matters here
# is the ratio inside the positive half: buying and selling stock pays +0.155 while
# growing the purse from the 3,000 stake to 10,868 pays +0.252.
#
# Against the ladder mirror the behaviour that ratio buys is unambiguous. Per
# episode-seat we move 218,563 coins of gross revenue against a ladder champion's
# 136,737, and give back 214,104 against its 40,588 -- a 2% net margin against 70%,
# with 34.7% of episodes finishing under the stake the ladder never once finishes
# under. The policy is not failing to trade. It reinvests everything it earns.
#
# The value was not argued, it was measured. `episode_rewards` is a pure function of
# a trace, so 300 real episodes of this session were re-scored under candidate
# constants before anything was edited; the sweep and the reason 2.5 beats the 1.5
# optimum are in `OUTCOME_LOG_SCALE`'s own docstring. Measured on the same 300 and
# rejected: scaling the lot weights down, which costs win/lose separation because
# those legs are the only action-differentiable money term in the stream.
#
# `SLOT_PREFIX` moves, by the rule at the top of this file: a stored shaped return
# is now a different quantity, so `split_*`'s buffers, lineage, self-play pool and
# mature members are all severed. That is the expensive half -- ten cold slots at
# batch 0 -- and it is not optional, because a mature member fitted to the old
# objective would re-enter as a reseed ancestor and a mirror opponent and carry the
# old ratio back in.
#
# The demonstration arm does **not** come along, and not only to keep this to one
# variable. `splitladder-v1` measured what pinning a pack does here: `max_abs_q`
# went 0.49 -> 20.82 and `mean_loss` 0.0013 -> 1.194 on the *first* update carrying
# demonstrations, and stayed there, against control slots flat at ~0.5 / 0.0005 all
# session. Every reward term is bounded, so a correct Q is single digits.
# `updater.py:2195` runs `cross_entropy` straight on the Q head's logits, whose
# optimum is the chosen logit at infinity -- `grad_clip_norm` bounds the step, not
# the fixed point -- and then the target net is synced to the inflated head. Any
# demonstration arm reproduces that until the clone loss is bounded (DQfD's
# large-margin hinge is the fix, and it is zero-gradient once the demonstrated
# action leads). So the seeded/control comparison in this session is void, and the
# only clean curve to read `cashweight` against is `splitladder-v1` slots 05-09.
#
# Landing with it, and recorded here because it is a second change even though it
# is not a second *variable*: `order_space.stock_ceiling` shuts every SELL rung
# above the smallest one that clears the shed. Deduplication rather than
# preference -- the shed bounds the fill, so those rungs are one action spelled
# several ways -- which is the proof `feasible_actions` demands rather than the
# measurement it refuses. It is a narrowing, so `split_*` checkpoints still load
# and play unchanged. What it buys is on the reward: `PendingTurn.apply` quotes an
# order at its full quantity, so an oversized rung inflates `expected_delta` and
# `order_refused` reads the whole turn as refused, charging
# `REJECTED_ORDER_PENALTY` on a turn whose legs filled. 2,113 of 7,910 sell units
# an episode had no stock behind them.
#
# Also landing: `updater._large_margin_loss` replaces the clone phase's
# cross-entropy. Inert this session -- `demonstration_episodes` is 0 roster-wide,
# so the phase never runs -- and landed now so the next demonstration arm is
# readable rather than reproducing the divergence table above.
#
# `v2`, and the `cash_*` prefix with it: `cashweight-v1` had collected 30 batches of
# generation-zero warmup before the two narrowings below landed, and warmup or not
# those episodes were played under a wider action space than the one the run
# measures. Nothing was lost -- zero model updates and zero checkpoints, so no
# gradient step ever read them -- and they stay in PostgreSQL under their own
# session id rather than being deleted, which keeps the record honest about what
# was abandoned and why.
#
# `cashherd`, because the field layer moved too and the prefix has to move with a
# changed state distribution the same way it moves with a changed reward.
# `herd.herd_influenced_roles` no longer strips the donor's speculative animal
# roles. Over 24 seeds in both orientations, everything else equal and the donor's
# market rules on both sides: 50,240 with the strip against 83,525 without, with
# pilkwang held to 86,111 instead of running to 122,515. The waste the strip was
# bought against is already refused by `donor._field_jobs`, which emits `BUILD_`
# only at `left >= first + 2` and never while liquidating -- a payback gate on the
# strip measured bit-identical to deleting it -- and its 52%-never-buy premise was
# measured on a learner that was not buying animals.
#
# **So this session moves two things against `splitladder-v1` and the cross-session
# read is confounded. Stated rather than discovered later.** What makes it
# tolerable is that both changes are evidenced *outside* the league: the reward
# scale by re-scoring 300 real episodes offline, the field layer by 48 arena
# episodes on matched seeds. Neither needs this run to establish it. What this run
# is for is whether the combination trains, which is the one thing neither offline
# measurement can answer.
#
# `cashweight-v2` is abandoned at 80 batches, again with zero model updates and
# zero checkpoints, so again nothing had read them.
#
# Pass condition, and it is not "money goes up": `own_solvent_fraction` mean over
# 0.50 against the splitladder control half's 0.276, and `own_mean_return` crossing
# zero on a majority of slots, **with `order_diagnostics.trading_rate` held within
# ~25% of 0.392**. The third clause is the whole guard. A slot that stops trading
# converges on the 3,000 stake, every trading slot in `splitladder-v1` finishes
# *below* the stake, so going quiet raises `lr_score` mechanically -- `split_03` is
# doing exactly that at cutoff 120-130 on a `trading_rate` of 0.017. Money up with
# the wire going silent is the NOOP collapse at the top of this file, not a result.
# ---------------------------------------------------------------------------
# Continued as `market-free-quote-v1`, which moves no constant. It fixes the units
# on the one test the reward runs every turn.
#
# `networth-v1` is stopped at generation 20 on a failed pass condition:
# `own_solvent_fraction` 0.258 against the >0.50 the block above asked for, and
# `own_mean_return` negative on ten slots of ten. The second clause turned out to be
# unreachable by construction, which is what led here.
#
# The stream telescopes to `(final_money - stake) / MONEY_SCALE` and nothing else,
# so a season ending on 18,546 coins is worth +0.155. Over 48 stored rollout
# episodes from `networth_01` batch 209, `e*_reward` summed to -0.343 against
# `e*_raw_reward`'s +0.155 -- a drain of -0.498 an episode, 3.2x the whole objective,
# and `max(shaped)` of -0.016 over the set, meaning no episode this fork ever played
# scored above zero. `IDLE_PENALTY` caps at 0.072 and the simulator refused 0.3% of
# orders, so neither charge accounts for it.
#
# Reconstructing the per-turn difference over 60 generation-19 traces: 50.1% of turns
# carry -0.001, which is `REJECTED_ORDER_PENALTY` charged on half the season. The
# cause is `reward.py`'s call into `order_refused`, which tests
# `|delta| < FILL_FRACTION * |quoted|` with `delta` already divided by `MONEY_SCALE`
# and `quoted` in coins straight off `PendingTurn.expected_delta`. At a 100,000x
# mismatch the test is true for every quote above a hundredth of a coin, so every
# turn that emitted a priced order was charged for it. -0.36 an episode, 74% of the
# drain.
#
# `market_state.py:598` calls the same function with no quote and a delta in coins,
# so it takes v1's `delta == 0` branch and the `refusal` feature stayed on the true
# 0.3%. The state said "filled" on half the turns the reward charged as refused --
# the drift `order_refused`'s docstring exists to prevent, and half a season of
# per-turn charge the network had no feature to predict.
#
# The bug is `expected`-path-wide and reaches v2, v3 and v3b unchanged. It is fixed
# here only: `market-free-herd-v1` is still collecting, `reward.py` is not in
# `ensure_session`'s digest, and editing it under a running session would silently
# rebase the one lineage whose benchmark money is climbing (6,199 -> 14,158 over
# batches 40-280) onto a different objective mid-run. Why v2 tolerated it: its
# `tanh` outcome term paid on the order of 1.0, so -0.36 was a tax. v3c deleted that
# term and put `MONEY_SCALE` up 5x, which left a linear +0.155 with the same tax
# against it.
#
# `SLOT_PREFIX` moves. The charge landed on half of every stored episode's turns, so
# every `networth_*` buffer, lineage checkpoint, self-play pool entry and mature
# member holds a shaped return that is not this quantity.
#
# Pass condition, and the first clause is now the one that was unreachable:
# `own_mean_return` positive on a majority of slots, and the gate's own
# `benchmark_ratchet.candidate_money` -- money against pilkwang, not the mirror --
# above `networth-v1`'s at matched batch (2,021 at 40, 2,423 at 80, 3,920 at 120,
# 2,185 at 200) rather than merely above its own start. `lr_score` reads the rollout
# mirror and rose 5,711 -> 8,405 across `networth-v1` while the benchmark it is
# supposed to stand in for fell 4,053 -> 2,185; the mirror is not the read. Guard
# clause unchanged from block G: `order_diagnostics.trading_rate` held near 0.6, so
# that the drop in charges is not collected by going quiet.
# ---------------------------------------------------------------------------
SESSION_ID = "market-free-quote-v1"
ARTIFACTS = ARTIFACTS_ROOT / "dqn_market_free_v3c"
LEAGUE_CONFIG = Path("configs/dqn_market_free_v3c_league.toml")
MANIFEST = Path("agents/dqn_market_free_v3c/agent.toml")

# Ten concurrent learner slots under one declared model, which the harness
# namespaces as `dqn_market_free_v3c.raw_order_q_v1@herd_00` and so on.
LEARNERS = 10
# Named for the reward it was fitted to. `_self_play_pool`, `lineage_checkpoint`,
# `all_batches` and the slot LR state are all scoped by `model_id`, so renaming
# severs all four at once -- which is required whenever the reward changes,
# because a stored shaped return is then a different quantity.
#
# Moved off "networth" with `SESSION_ID`. No constant moved this time, but the
# quantity did: `REJECTED_ORDER_PENALTY` was charged on 50.1% of turns against a
# simulator refusal rate of 0.3%, so every stored `networth_*` return is the shaped
# stream minus about 0.36 of tax that this session does not levy. The sharp reason
# applies as it did off "cashherd" -- a mature member fitted while half its turns
# were taxed re-enters as a reseed ancestor and a mirror opponent carrying the
# behaviour that tax bought, which was to stop emitting priced orders.
SLOT_PREFIX = "quote"
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
    # Back to 400. The watch-condition the previous comment set was hit: it asked to
    # "come back down if it is heading past ~4 GB", and `buffer_megabytes` across
    # generations 13-18 of `market-free-refusal-v1` read ~3.0 GB mean against a
    # **4.47 GB max**. The prediction was right -- a trained policy trades more and
    # the record grew -- so this is that comment being obeyed, not overruled. At ten
    # slots it is also ~15-22 GB resident across a concurrency of five.
    #
    # The trajectory-diversity argument above is not lost to the cut, because
    # `batch_size` is now 32 rather than 8: the sampler draws four times as many
    # distinct episodes per gradient step, so the ranked pool is consumed far more
    # thoroughly at 400 than it ever was at 800. Staleness improves too -- at 160
    # new episodes a generation, 800 held five generations of off-policy data.
    "buffer_episodes": 400,
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
    # No demonstrations on any slot. The pilkwang donor was the family worth
    # cloning and it has zero rows in `episode_players` -- it only appears as a
    # promotion-gate benchmark, and those matches are not persisted as league
    # episodes -- so the league source has nothing to offer here. v3b's
    # `market-free-ladder-v1` is testing whether the Kaggle ladder mirror does;
    # that is a separate variable and this arm does not carry it.
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
    # manifest *does* declare this key -- `agents/dqn_market_free_v3c/agent.toml:361`
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
    # 3e-4. The sweep the previous comment left open, run on the baseline it asked
    # to wait for: `market-free-refusal-v1` gave the geometry 17 generations, and
    # `mean_grad_norm` across them read **0.017** with `max_grad_norm` at 0.16
    # against a `grad_clip_norm` of 5.0. The clip is not merely un-binding, it is
    # inert by two orders of magnitude, which is the headroom that comment
    # predicted.
    #
    # What the starved run showed downstream of that: `max_abs_q` 0.6 and
    # `mean_abs_q` 0.05 against v2's 1.0-2.3 and 0.11, and `sigma_mean_0` drifting
    # only 0.0442 -> 0.0412 across the whole run. Sigma is a trained parameter, so
    # a head whose weights do not grow keeps `sigma_ratio_0` at 12 where v2 sat at
    # 1.9, and `noisy_argmax_flip_rate` at 0.42 where v2 sat at 0.30. The rate is
    # what binds, and it binds on the exploration arm as much as on the head.
    #
    # The "do not pay twice" rule the previous comment set is respected, not
    # broken: steps go *down* 4x (192 -> 48) as the batch goes up 4x, so parameter
    # movement per update is ~0.75x what it was even at 3x the rate. This buys
    # better-estimated directions, not more of them.
    "learning_rate": 3e-4,
    # v3's chunk geometry, restated here for the same reason the learning rate is:
    # this file decides what the run is. `burn_in_turns` must stay a multiple of
    # `collect.HIDDEN_STRIDE` or `sample_sequences` refuses the draw outright.
    # Reversed, because the accounting above measured a quantity the loss throws
    # away. `updater.py:3633-3643` reduces in three levels:
    #
    #   per_turn  = zeros(B*train).index_add_(0, rows, per_row * turn_share)
    #   per_chunk = per_turn.view(B, train).mean(dim=1)
    #   loss      = (per_chunk * batch.weight).mean()
    #
    # The rows are averaged into turn-scalars, those into **one scalar per chunk**,
    # and only then meaned over `B`. So a step's gradient is the mean of
    # `batch_size` numbers drawn from `batch_size` episodes -- 8, not 688. (The
    # real row count was ~1,360 measured, higher than the estimate above, and
    # equally collapsed.) v2's 256 rows fed a *flat row mean* and were genuinely 256
    # independent directions, so "~688 rows a step is still 2.7x v2's 256" compared
    # a post-collapse number against a pre-collapse one. On the quantity that
    # survives the reduction the run was at 8 against v2's 256.
    #
    # That is the deficit `market-free-refusal-v1` actually measured: worse than v2
    # at every matched update index (1,254 vs 5,954 at update 3; 3,205 vs 12,145 at
    # 17) with `own_mean_return` pinned at -1.3 to -1.6 while v2 crossed zero by
    # generation 12.
    #
    # So: 32 chunks, 48 steps. Wall-clock-neutral -- 4x the cost per step against
    # a quarter of the steps, so ~230s an update stands -- and 4x the independent
    # gradient directions. Rows per update roughly double, to ~4.6 replay epochs
    # against the 400-episode buffer restored below.
    #
    # `burn_in_turns` and `train_turns` stay untouched here. The chunk *shape* is
    # gated on the inert-GRU test, not on this change.
    "batch_size": 32,
    "burn_in_turns": 20,
    "train_turns": 40,
    "gradient_steps_per_batch": 48,
    # Scaled with `gradient_steps_per_batch` to hold two syncs an update, which is
    # the invariant v2 and credit-v1 both ran at -- v2 at 2000 against ~4,000 steps
    # an update, credit-v1 at 250 against ~480. An update here is now 480 steps
    # (48 x `min_new_batches` 10), so 240.
    #
    # This is the coupling the manifest comment at `agent.toml:198-201` calls "the
    # single easiest value to leave stale across the fork, and nothing would report
    # it: a target that lags four updates just looks like a harder target". Left at
    # 960 alongside the step cut it would have been exactly that -- one sync every
    # two updates, a target net lagging half the data it is bootstrapping against.
    # It moves with the step count or not at all.
    "target_sync": 240,
    "grad_clip_norm": 5.0,
    # Moves back down with the peak. 1e-5 against a peak of 1e-4 needs two decays at
    # the manifest's 0.25, which at `lr_plateau_patience = 4` is twelve updates and
    # so twelve generations here -- the same ladder v2 walked. Moves with the peak
    # to hold that 1:10 ratio: left at 1e-5 against the 3e-4 peak above it would be
    # 1:30, three decays rather than two, and a slot would run a third of a ladder
    # further before graduating.
    "lr_graduate_cut": 3e-5,
    # Two, not three. A slot gets ~17 updates in a session and the first update of
    # every warmup trains at a rate of exactly zero -- generation 1 of
    # `market-free-refusal-v1` records `lr` = 0 -- so at three the warmup was
    # spending a sixth of a slot's life, and repeating it after every reseed.
    "lr_warmup_updates": 2,
    # Six, for the same arithmetic. At ten, a slot could not graduate until update
    # ten of about seventeen, so the back half of the ladder never ran: eight of ten
    # slots finished the session at `lr_reseed_index` 1 despite rejection streaks of
    # 12-15.
    "lr_min_updates_before_graduation": 6,
    # Five, up from the manifest's two. Two updates is not enough evidence to
    # graduate a checkpoint into an archive that `self_play_ratio = 0.65` makes the
    # *opponent pool*: `market-free-refusal-v1` admitted members scoring 811 and 451
    # that buy 546 and 237 cows an episode, and then trained against them. Archive
    # mean was 2,346 against a best of 9,166.
    "population_entry_updates": 5,
    # 4,000, up from the 1,500 default. `reseed_source_explorer_share = 0.25` draws a
    # quarter of reseeds from the reserved berths, so the reserve is an ancestor
    # pool and not just a display shelf. Only 4 of the previous run's 21 members
    # clear 4,000, which is the point -- the two that work score 9,166 and 8,357.
    "mature_reserved_min_money": 4000.0,
    # Safe now that `lr_score_signal` is "rollout": the detector reads the ring
    # the slot just trained on, so a snapback moves it rather than pinning it.
    # Buys the case with no other answer -- a diverged slot has nothing to climb
    # back to, and `lr_best_decay` only unsticks a detector still improving.
    "resume_reset_after_rejections": 2,
}

# No demonstration arm this session. `splitladder-v1` ran a five-and-five ladder
# split and the read came back void rather than negative: the seeded half's value
# function diverged on contact with the pack. `max_abs_q` by cutoff, seeded against
# control, from `model_updates.metrics_json`:
#
#     cutoff   20     30     60    100    130
#     seeded  0.49  20.82  26.98  21.33  18.35
#     control 0.57   0.50   0.45   0.49   0.60
#
# Cutoff 30 is the first update carrying `demonstrations`. `mean_loss` steps
# 0.0013 -> 1.194 on the same row and never comes back, against control slots at
# 0.0005 for the whole session. Every term in `reward.py` is bounded -- `outcome` is
# a tanh -- so a correct Q sits in single digits and 20 to 27 is a diverged head,
# not a scale.
#
# The cause is the loss, not the data, so it would have reproduced on any pack:
# `updater._behavior_clone_step` runs `cross_entropy` directly on the Q head's
# logits, and cross-entropy's optimum is the chosen logit at infinity. There is no
# fixed point in Q units for it to settle on; `grad_clip_norm` bounds the step and
# not the destination, the phase runs 100 steps a batch against the TD loop's 48,
# and `updater.py:3556` then syncs the target net to whatever the clone inflated.
# The bounded form is DQfD's large-margin hinge --
# `relu(max_a[Q(s,a) + margin * (a != a_E)] - Q(s,a_E))`, exactly zero once the
# demonstrated action leads -- and the roadmap already carries it as unbuilt work
# under "Out of scope, worth recording", where it is filed as a way to make the
# clone *stronger*. The measurement above says the opposite: it is already strong
# enough to set the head's scale and the TD loop spends the rest of the session
# pulling back. Same prescription either way; build it before any pack is pinned
# again.
#
# So `LADDER_DEMOS` and `SEEDED_SLOTS` are gone rather than swapped for a donor
# pack. Two reasons and either would do: this session moves the reward and a
# demonstration arm would be a second variable, and there is no readable
# demonstration arm to run until the clone loss is bounded.

# The roster is uniform -- one reward, no demonstrations -- and the archive handles
# retention, with `rare_sigma_gain` handling generation on every slot.
# `LeagueConfig.validate` requires exactly one `primary` per agent type, so slot 0
# carries it -- deliberately not re-derived from scores, which would make every
# query resolving it non-reproducible. Distinct seeds are the only other thing
# separating the slots at the start; what diverges them afterwards is which mature
# member each is reseeded from and which weights the perturbation redrew.
TRAINING_AGENTS = [
    {
        "type": "dqn_market_free_v3c",
        "name": f"{SLOT_PREFIX}_{index:02d}",
        "train": True,
        "primary": index == 0,
        "model_config": {
            **FREE_CONFIG,
            "seed": 2718 + 1009 * index,
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
