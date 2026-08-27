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
    WHERE model_id = 'dqn_market_free_v3c.raw_order_q_v1@rewardtogo_00'
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
# Half the roster pinned 96 of them under a `SEEDED_SLOTS` split that no longer
# exists; today every slot pins 128 (`demonstration_episodes` in `FREE_CONFIG`).
# `demo_fraction` stays at the manifest's 0.25, so a slot draws a quarter of every
# TD minibatch from the ladder and runs a ladder behavior-clone phase on top.
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
# cross-entropy. Inert in that session -- `demonstration_episodes` was 0
# roster-wide, so the phase never ran -- and landed then so the next demonstration
# arm would be readable rather than reproducing the divergence table above. It is
# the loss the ladder pack trains under today.
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
# The bug is `expected`-path-wide and reaches v2, v3 and v3b unchanged. It was fixed
# here only: `market-free-herd-v1` was still collecting, `reward.py` is not in
# `ensure_session`'s digest (it *is* in `match_agent.toml`'s `code_paths`, which
# mints a new revision per episode -- a label, not a guard; nothing halts), and
# editing it under a running session would silently rebase the one lineage whose
# benchmark money was climbing (6,199 -> 14,158 over batches 40-280) onto a
# different objective mid-run. Why v2 tolerated it: its `tanh` outcome term paid
# on the order of 1.0, so -0.36 was a tax. v3c deleted that term and put
# `MONEY_SCALE` up 5x, which left a linear +0.155 with the same tax against it.
# (Historical: none of `order_refused`, `REJECTED_ORDER_PENALTY`, `FILL_FRACTION`
# or `IDLE_PENALTY` reach the reward any more, and `MONEY_SCALE` is 25,000 now.)
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
# Continued as `market-free-scale-v1`, which moves two things and stops the fork's
# one-change-a-session rule only because both are the same measurement.
#
# `MONEY_SCALE` 100,000 -> 250,000. The old value was fitted to a single ladder
# number, `99,038 / 100,000 = 0.99`, which is the median of the demonstration set
# sitting exactly on `SmoothL1`'s beta rather than under it. Over the 321 stored
# `ladder-001` seasons the final-money distribution is median 85,867, p99 149,431,
# max 155,637 -- so at 100,000 the top decile of *demonstrated* play trains in the
# linear regime where a 5-unit and a 50-unit TD error give the same gradient, and
# pilkwang's 131,000 was outside the quadratic zone before this fork could reach
# it. At 250,000 the whole distribution fits: 0.34, 0.60, 0.62. The per-step bound
# moves with it -- largest single wallet move 14,513 coins over the demonstrations
# and 27,670 over 1,920 stored `networth_*` rollout episodes, so 0.06 and 0.11 here
# against 0.15 and 0.28 before.
#
# `IDLE_PENALTY` and `REJECTED_ORDER_PENALTY` are divided by the same 2.5. They are
# absolute reward units and do not scale with the delta, so leaving them would make
# them 2.5x heavier against money -- and against a policy finishing on 2,185 coins
# (0.022 here) the anti-idle term already outweighs the profit signal, which is the
# shape `wire_orders_per_decision` 2.6 with 33% of stock spend rotting looks like.
# Held at 20 and 100 coins, every ratio their docstrings argue for unchanged.
#
# `batch_size` 32 -> 128, `gradient_steps_per_batch` left at 48. `updater.py`'s
# reduction means a step's gradient is the mean of `batch_size` per-chunk scalars,
# so the batch is the count of independent directions and the step count is not.
# The previous block fixed 8 -> 32; 32 is still an eighth of v2's 256, and v2 is
# the only lineage in this family whose benchmark money ever climbed. `learning_rate`
# 3e-4 -> 7.5e-4 is not a third hike: the 2.5x smaller reward is a 2.5x smaller
# gradient at every step, and this restores the parameter movement per update the
# rate comment already sized.
#
# Cost, from 970 completed updates: an update is ~375s of which `train_seconds` is
# ~124, so 4x the training puts an update near 750s. Half the updates an hour.
#
# `SLOT_PREFIX` moves. Both the reward scale and the two charges changed, so every
# stored `quote_*` shaped return, lineage checkpoint, self-play pool entry and
# mature member holds a quantity this session does not pay.
#
# Pass condition, unchanged in kind from `quote-v1` because it is the read that was
# wrong for twenty updates: the gate's own `benchmark_ratchet.candidate_money` --
# money against pilkwang, not the mirror -- above `quote-v1`'s at matched batch,
# and above `networth-v1`'s (2,021 at 40, 2,423 at 80, 3,920 at 120, 2,185 at 200).
# Second clause on the batch change specifically: `mean_grad_norm` off the 0.017-0.06
# floor the last two forks read against an inert `grad_clip_norm` of 5.0, and
# `max_abs_q` off 0.6 toward v2's 1.0-2.3. Guard clause unchanged:
# `order_diagnostics.trading_rate` held near 0.6, so that a lighter idle charge is
# not collected by going quiet.
#
# ---------------------------------------------------------------------------
# idle-v1: the idle charge was the objective, so it was cut 10x. The refusal charge
# goes with it for a different reason -- it is measurably nothing.
#
# `scale-v1` answered its own pass condition and a larger question with it. Across
# 25 updates none of the ten slots promoted once, and every rejection was
# `benchmark_ratchet` rather than `hard_viability`: candidate money 177 to 4,316
# against pilkwang's 115,000 to 141,000, so `opponent_money_ratio` (0.05, needing
# ~6,300) and `max_money_deficit` (100,000) failed on every row while `min_money`
# passed. The gate was unreachable, and the reason is upstream of the gate.
#
# **`scale-v1` was rebased mid-run and its generations are not one experiment.** The
# uncommitted edit to `reward.py` -- `IDLE_PENALTY` 8e-5 -> 8e-6 and
# `REJECTED_ORDER_PENALTY` deleted -- was on disk while `scale-v1` was collecting, and
# its workers picked it up between g0002 (trace dir written 11:33) and g0003 (12:18).
# Recovered from the traces, not from any log: the median shaping delta on turns with
# no non-NOOP row tracks `-IDLE_PENALTY * falloff * MONEY_SCALE`, and it reads 8.5e-5
# through g0002 and ~1.2e-5 from g0003 on, on both `scale_00` and `scale_05`, with the
# falloff structure flattening at the same generation. `quote-v1` and `networth-v1`
# probe clean at 2.1e-4 across every generation they ran.
#
# So g0000-g0002 are `scale-v1` as declared and g0003+ are `idle-v1`'s objective on
# `scale-v1`'s checkpoints, buffers and self-play pool. Anything read across the
# boundary -- `own_mean_return`, the reservoir half of `own_mean_final_money`, the
# plateau median, every `scale_*` mature member from g0003 on -- mixes two objectives.
#
# `own_mean_return` -- the quantity the network actually fits -- is negative in every
# update of both halves, while the money term it telescopes to spans -0.007 to +0.018.
# Where the difference goes, decomposed per episode over every stored `episode_traces`
# npz. `drain` is `sum(raw_reward) - sum(reward)`, everything the shaping takes off
# the money term; `idle` recomputes `IDLE_PENALTY * falloff` from `order_chosen` /
# `order_step` on turns carrying no non-NOOP row:
#
#     slot / regime                 eps   final$   money   shaped   drain    idle  share  resid
#     networth_00  2e-4, bug      3,360    9,688  +0.0669  -0.4413  0.5082  0.0371   7.3% 0.4711
#     networth_05                 3,360    4,690  +0.0169  -0.4509  0.4678  0.0433   9.3% 0.4245
#     quote_00     2e-4, fixed      512    5,496  +0.0250  -0.0322  0.0572  0.0411  71.9% 0.0161
#     quote_05                       496    7,685  +0.0469  -0.0392  0.0861  0.0458  53.2% 0.0403
#     scale_00     8e-5, g0-g2      480    4,317  +0.0053  -0.0174  0.0227  0.0160  70.5% 0.0067
#     scale_05                       480    7,262  +0.0171  -0.0091  0.0261  0.0193  73.8% 0.0069
#     scale_00     8e-6, g3+        912    2,206  -0.0032  -0.0116  0.0084  0.0019  22.5% 0.0065
#     scale_05                       912    3,552  +0.0022  -0.0045  0.0067  0.0025  37.5% 0.0042
#
# `shaped` is negative on every slot of every run and every regime while `money` is
# positive on all but one. Under `scale-v1`'s declared constants the idle charge is
# 70-74% of the whole drain: the objective was anti-idle with money as noise on top,
# which is what a policy that grosses 80,900 coins of seed proceeds an episode and
# finishes on 1,400 looks like from the inside. `networth-v1` is the contrast -- 7-28x
# its money term with idle under a tenth of it, which is `REJECTED_ORDER_PENALTY`
# under the unit bug `quote-v1` fixed, charged on ~61% of turns. Two runs, two
# different terms, one shape.
#
# `idle` is an upper bound and `resid` a lower one: the reconstruction reads
# `chosen != 0` and misses the `buy_land` turns `collect.finish` also marks emitted,
# so it scores some trading turns as idle.
#
# `IDLE_PENALTY` 0.00008 -> 0.000008. The previous move held every ratio fixed by
# construction and so changed nothing; the ratio was the bug. It was sized as 7% of a
# ladder-grade season and this policy plays a season 30x smaller, where it is 240%.
# Two orderings it inverted: a fully idle season cost 0.0288 against 0.012 for losing
# the entire stake, so bankruptcy was cheaper than sitting still, and
# `own_solvent_fraction` measured 0.013-0.29, so most idling was an empty wallet
# rather than a choice worth pricing.
#
# `REJECTED_ORDER_PENALTY` is deleted, not re-scaled, and **not for the reason the
# first draft of this block gave.** That draft said the charge fired on ~3.4% of
# turns, nine times the 0.37% the simulator actually refuses (`scale_00` batch 20:
# 138,470 emitting, 137,619 effective, of 230,080). That 3.4% was a subtraction:
# penalty mass 0.021-0.028 minus an idle mass "of about 0.011 at the measured 38% idle
# rate", with `own_order_diagnostics.emitted_rate` read as the emitting fraction of
# turns. Per turn off the traces the emitting fraction is 0.484-0.592 under the
# declared constants, so idle is 41-52% and the idle mass ~1.5x what the subtraction
# allowed, which caps the charge at 2.3-2.4% of turns rather than 3.4%.
#
# The rebase settles it properly. Deleting the charge at g0003 is a controlled removal
# of that term, and `resid` -- the quantity the 3.4% was inferred from -- does not
# move across it: 0.0067 -> 0.0065 on `scale_00`, 0.0069 -> 0.0042 on `scale_05`. A
# term that survives its own deletion was never that term. The residual is the
# `(1 - gamma)` basis holding drag in `reward.net_worth_delta`. So the charge is
# deleted because it is ~0, which is also why deleting it moved no number this run
# reads. The `expected` argument goes with it, along with the recorder and clone
# plumbing that fed it. Whether `sell_rung` really does inflate `expected_delta` is an
# `order_space` question this measurement does not settle.
#
# The refusal state block stays. Nothing about it depended on the charge, and
# dropping it would move `MARKET_STATE_DIM` and invalidate every checkpoint.
#
# `SLOT_PREFIX` moves. Both per-turn charges changed, so every stored `scale_*`
# shaped return, lineage checkpoint, self-play pool entry and mature member holds a
# quantity this session does not pay -- and the sharp version of that rule applies
# here more than anywhere: a mature member fitted while the objective was anti-idle
# re-enters as a reseed ancestor and a mirror opponent carrying exactly the behaviour
# that objective bought.
#
# Pass condition, same read as `scale-v1` because it is still the right one: the
# gate's own `benchmark_ratchet.candidate_money` above `scale-v1`'s at matched batch
# (1,980 / 3,236 / 4,316 at 10, then 995-3,541 at 20). The specific claim this makes
# is that `own_mean_return` turns positive for a slot that is making money, so read
# it beside `own_mean_final_money` and reject the change if the two still disagree in
# sign. **That clause is withdrawn: it asks a question the quantity cannot answer.**
# `reward.net_worth_delta`'s basis half is potential-based shaping, so the undiscounted
# sum `own_mean_return` reports carries `basis[0] + (1 - gamma) * sum(basis[1:-1])`
# whatever the policy does. Recovered by inverting the stream over 960 episodes a side,
# that is 0.0033-0.0058 an episode, exactly proportional to held basis (`scale_00`
# holds 1,999 coins and pays 0.00575; `scale_05` holds 1,132 and pays 0.00325, both
# 1.77x). A slot making money while holding stock reads negative there by construction,
# and the g0003+ rebase shows precisely that: idle falls to 0.0019-0.0025 as intended
# and the sign does not move.
#
# So the pass condition is `benchmark_ratchet.candidate_money` and nothing else, with
# `own_mean_final_money` -- or equivalently the sum of `raw_reward`, which telescopes
# to `(final_money - stake) / MONEY_SCALE` exactly -- as the supporting read.
# `own_mean_return` is reportable but not gradeable until it is decomposed.
#
# Two follow-ups this measurement opens, neither blocking this session:
#
#   * `own_mean_raw_return` is worth adding beside `own_mean_return` in
#     `updater._records_diagnostics`. `EpisodeRecord.raw_reward` is already stored and
#     already summed for `own_mean_final_money`; naming the reward-unit version means
#     the shaped and unshaped returns can be diffed in SQL, which is how the g0003
#     rebase would have been caught the same day.
#   * `basis[0]` averages 199 coins with a standard deviation of 1,204, because a
#     minority of episodes open with thousands of coins of stock already held while
#     `money[0]` is 3,000 on every one. That is the recorder's first turn not always
#     being the season's first, and it injects a per-episode nuisance the size of the
#     money term into every shaped return. Worth a look in `collect.Recorder` before
#     anyone grades a slot on shaped returns again.
#
# Guard clause, tightened because a 10x lighter idle charge is exactly what would buy
# silence: `order_diagnostics.trading_rate` held near 0.6 and
# `own_orders_per_decision` not below `scale-v1`'s 0.44 floor.
# ---------------------------------------------------------------------------
# v2 rather than a resume of `market-free-weighted-v1`. That run promoted nothing
# in twelve generations: it started after the uncommitted
# `benchmark_min_opponent_money_ratio` / `benchmark_max_money_deficit` hunk, whose
# two checks armed against pilkwang's *uncontested* 131k and so could not be
# cleared by a family that earns ~2-3k contested. With the gate never releasing a
# checkpoint there was also nothing to restore when `max_abs_q` inflated from 0.3
# to 60 on six of ten slots and took slot 02 from 32,353 to 3,507 at a fixed
# learning rate. The impossible benchmark checks are not part of this config, and
# the optimizer falls back to the manifest's conservative values until the new
# namespace proves it can promote. The batches are not comparable and both the
# session id and slot prefix have to move.
#
# v3 moves again because the promotion gate changed after `weighted2` had already
# collected and partly evaluated g0002. The reward and replay data are compatible,
# but the release decision is not: `weighted2_*` carries active checkpoints and
# rejection state produced under the stricter lot-quality gate. A fresh key gives
# the simplified old-style gate a clean read.
#
# v4 moved off `weighted3` because that session had stored the strict online gate
# in its manifest and had already rejected checkpoints under it.
#
# terminal-v1 moves again because the reward target changed from a shaped per-turn
# stream to final-profit Monte Carlo labels backfilled across each episode. Stored
# weighted5 returns, replay priorities, Q scales, and active checkpoints are fitted
# to a different target and must not be mixed into this run.
#
# terminal-mix-v1 rerolls from terminal-v1 because the collection curriculum changes:
# terminal-v1's empty base made the self-play pool the whole opponent distribution
# once it existed. That can overfit the family to its own checkpoints, so this run
# restores the fixed donor as a base opponent and lowers the requested self-play
# share.
#
# terminal-selfplay-ladder-v1 rerolls from terminal-mix-v1 to separate the two
# anti-overfit signals: rollout collection is pure family self-play, while the
# outside-policy anchor enters only through pinned Kaggle ladder recordings.
#
# terminal-selfplay-ladder-clean-v1 rerolls because terminal-selfplay-ladder-v1
# was flat for three generations (see `model_updates` for the run):
#
#   * 2 of 26 post-warm-start updates promoted; the rest rejected
#     `below_best_promoted_benchmark_money`, and lineage resumed from the best
#     *promoted* checkpoint -- so every update retrained the warm-start weights
#     and threw the result away. -> `lineage_source = "latest"`.
#   * The label was `(final - money[0])` copied to every turn: one number per
#     season, no credit inside it, a 0.02-wide band at `MONEY_SCALE` 250k.
#     -> reward-to-go `(final - money[t]) / 25_000` in `reward.py`.
#   * `_terminal_quality_weight` put 0.10 on the 81% of own episodes that ended
#     insolvent -- the negative examples. -> `terminal_quality_weighting = False`.
#   * `ladder-001` stored the old shaped stream at scale 20k; 25% of every batch.
#     -> relabelled on load from the money trace.
#   * `max_abs_q` 22-51 on three slots from a noisy target max over eleven
#     intra-turn rows. -> target noise off for the bootstrap, value clamp.
#   * Rollouts were 100% greedy-mirror; the gate was 100% pilkwang, win rate 0.00.
#     -> pilkwang back in the mix at half the pairs (since reverted: the
#        shipped config is `opponents = []`, pure mirror self-play).
#
# rewardtogo-v1 rerolls the key off terminal-selfplay-ladder-clean-v1 rather than
# changing anything: `clean-v1` was already registered in `training_sessions` and
# had collected a generation under a roster of `terminalselfclean_*` slots before
# this driver was settled. Resuming it would put this run's generations beside
# batches whose provenance nobody here can vouch for, and `ensure_session` cannot
# tell the two apart -- its digest covers `LeagueConfig` and `TRAINING_AGENTS`, not
# who launched them or what was on disk at the time.
#
# Named for the change that defines the run rather than for the curriculum, which
# is the honest label: every fix above is downstream of the label moving from one
# season-constant number to a per-turn reward-to-go.
# rewardtogo-v2 is the same experiment as v1 under a fresh key. `ensure_session`
# digests `LeagueConfig` plus `TRAINING_AGENTS` and refuses a resume whose digest
# moved ("run manifest mismatch"), and v1 was registered at 04:45 UTC with
# `reset_optimizer_moments: true` on all ten slots. The review pass that followed
# set it False -- the prefix is fresh, so there are no inherited Adam moments to
# drop, and True re-read every update would reset the moments on every update
# rather than once. That one key is the entire digest difference; every other
# value in the stored manifest matches what is below.
#
# So: a new key rather than a resume. Reverting the key to match the stored
# manifest would be resuming into the bug the review pass fixed.
# members-v1 replaces the learner pool with per-slot member pools. Each slot now
# carries 5 independent lineages, round-robins them through the generation's 20
# rounds (4 batches each), and prunes to the best 5 of its own 10 checkpoints from
# this generation and the previous one.
#
# What it replaces and why: `rewardtogo-v2` gave every slot one lineage, so a
# lineage that broke took the slot with it. Slot 09 diverged (`max_abs_q` 81.8) and
# sat at 3,000 benchmark money for sixteen generations while slot 07 reached 13,661.
# Reseeding existed but fired only when the LR ladder decayed past `lr_graduate_cut`,
# which a slot that never improves never trips. Both the ladder and the shared
# mature archive are deleted; the prune replaces them.
# Moved off "members-v1" with the cut from ten slots to five. Forced, not chosen:
# the run manifest `ensure_session` digests is the `training_agents` list, one
# entry per slot, so dropping five entries changes the digest and v1 will not
# resume. Entries 00-04 are byte-identical across the change -- same seeds, same
# `model_config` -- so the only thing v2 records differently is that the other
# five are gone. The alternative was rewriting v1's stored manifest in place,
# which would have left the session row claiming settings its first two hours did
# not run under.
# Moved off "members-v2" with the member budget fix. Forced again -- `members_per_slot`
# and `buffer_episodes` are both `[models.config]`, so the run manifest digest moves
# and v2 will not resume -- but this one would be right regardless of the digest,
# for the reason below `SLOT_PREFIX`.
#
# Moved off "top25b" to isolate what actually stopped the roster collapse.
#
# `market-free-top25-v1` lost every slot at its first prune: at `cutoff_batch_seq = 30`
# all five forked from the *same* lineage `dae925496076fa89`, and
# `own_mean_final_money` fell from 3,248-3,947 to 31-88. That checkpoint is still in
# the pool -- `members_08`, `member_rank_score` exactly 3000.0, `member_rank_active`
# null -- and it was admitted because `cross_slot_reseed_min_score` was 3,000, which is
# the stake, compared with `>=`.
#
# top25b changed *two* things at once and observed one outcome. It raised the floor to
# 6,000, and `admit_foreign` separately gained a strict comparison and an
# `active is True` requirement. No forks happened, and that was read as the code fix
# working -- but the raised floor alone would also have produced it, so neither claim
# is supported.
#
# This run puts the floor back to 3,000 and keeps today's `admit_foreign`. The whole
# head of the foreign pool is inert policies scoring exactly 3,000 with `active` null,
# so the outcome is categorical rather than a noisy scalar and the reseeded per-slot
# seeds do not confound it: if no slot adopts one, the code gates are sufficient and
# the floor is belt-and-braces. If a slot does, the floor was carrying it.
SESSION_ID = "market-free-top25d-v1"
ARTIFACTS = ARTIFACTS_ROOT / "dqn_market_free_v3c"
LEAGUE_CONFIG = Path("configs/dqn_market_free_v3c_league.toml")
MANIFEST = Path("agents/dqn_market_free_v3c/agent.toml")

# Five concurrent learner slots under one declared model, which the harness
# namespaces as `dqn_market_free_v3c.raw_order_q_v1@members_00` and so on.
#
# Ten until members-v1. The cut is the member pool paying for itself: slot count
# bought exactly one thing, an independent replicate to hedge a lineage that
# breaks, and `members_per_slot = 5` now buys that *inside* a slot at a fifth of
# the cost -- a broken lineage loses one member for one generation instead of a
# whole slot for the whole run. `rewardtogo-v2` is the case for it: slots 01 and
# 09 diverged (`max_abs_q` 56.0 and 81.8) and finished at 2,230 and 2,176 recent
# mean money against a ~4,900 median, so a fifth of that run's budget produced
# nothing the prune would not have caught by generation two.
#
# What the cut does *not* buy is memory. Peak resident is set by `CONCURRENCY`,
# which is unchanged -- five slots are live either way, each holding
# `members_per_slot` x `buffer_episodes` -- so this is a wall-clock change only.
# What it buys is the second wave: at `CONCURRENCY = 5` ten slots is 2 x 5 and
# five is one full wave with no tail, which halves the generation. Measured
# warmup rounds, which is the clean comparison because generation 0 runs no gate
# arena either way: members-v1 at ten slots took ~50s a round, members-v2 at five
# takes a steady 23-24s, so a 20-round generation goes ~17min -> ~8min. Once the
# mirror pool exists the rounds roughly double, to 42-43s at five slots, and a
# generation settles at ~17min: ~14min of rounds against ~3min of gate arena.
# The rounds dominate, not the gate -- so the halving is a rounds effect and it
# holds past warmup rather than being washed out by a fixed arena cost.
#
# Slot diversity is not a reason to keep ten, because there is no cross-slot
# channel to lose. `_self_play_pool` is "promoted checkpoints of *this* model"
# and the mature backfill reads the slot's own members table -- the shared
# archive went away with reseeding -- so a slot's mirror pool is unchanged by how
# many siblings it has. The slots are independent replicates, and ten of them
# only ran the same experiment twice per wall-clock hour.
LEARNERS = 5
# Named for the reward it was fitted to. `_self_play_pool`, `lineage_checkpoint`,
# `all_batches` and the slot LR state are all scoped by `model_id`, so renaming
# severs all four at once -- which is required whenever the reward changes,
# because a stored shaped return is then a different quantity.
#
# Moved off "networth" with `SESSION_ID`. No constant moved that time, but the
# quantity did: `REJECTED_ORDER_PENALTY` was charged on 50.1% of turns against a
# simulator refusal rate of 0.3%, so every stored `networth_*` return is the shaped
# stream minus about 0.36 of tax that session did not levy. The sharp reason
# applies as it did off "cashherd" -- a mature member fitted while half its turns
# were taxed re-enters as a reseed ancestor and a mirror opponent carrying the
# behaviour that tax bought, which was to stop emitting priced orders.
#
# Moved off "quote" for the plain version of the same rule: `MONEY_SCALE` and both
# per-turn charges moved, so a stored `quote_*` shaped return is 2.5x a quantity
# this session pays and its Q head is fitted to that scale.
#
# Moved off "scale" for the sharp version. `REJECTED_ORDER_PENALTY` is gone and
# `IDLE_PENALTY` is 10x lighter, so a stored `scale_*` return carries 0.021-0.028 of
# tax this session does not levy -- which was most of what those Q heads were fitted
# to. Reseeding from them would import the behaviour that tax bought.
#
# Moved off "weighted" because `market-free-weighted-v1` never released a checkpoint.
# Its `weighted_*` model ids now hold rejected update state, self-play pools and
# buffers collected under a gate/optimizer regime this run is explicitly leaving.
#
# Moved off "weighted2" because its promotion state straddles the old strict gate
# and the simplified old-style gate. Keep the diagnostic comparison clean.
#
# Moved off "weighted3" with `SESSION_ID`; active checkpoints from that run encode
# a reject/ratchet policy this run no longer uses.
#
# Moved off "weighted5" with terminal-v1; reward records now carry final-profit
# Monte Carlo labels rather than a shaped net-worth stream.
#
# Moved off "terminal" with terminal-mix-v1; the self-play pool and lineage state
# from a pure mirror curriculum should not become this mixed curriculum's starting
# point.
#
# Moved off "terminalmix" with terminal-selfplay-ladder-v1; the rollout mix and
# pinned replay source both changed, so this run should not inherit terminalmix's
# lineage, self-play pool, or LR state.
#
# Moved off "terminalselfladder" with terminal-selfplay-ladder-clean-v1:
# `MONEY_SCALE` moved 250k -> 25k and the label moved from a flat terminal profit
# to reward-to-go, so a stored `terminalselfladder_*` return is a different
# quantity at a tenth of the scale and its Q head is fitted to that.
#
# Moved off "terminalselfclean" with rewardtogo-v1. No constant moved this time:
# ten `terminalselfclean_*` models already exist with a batch history this driver
# did not write, and `_self_play_pool`, `lineage_checkpoint`, `all_batches` and the
# slot LR state are all scoped by `model_id`. Reusing the prefix would hand this
# run four inherited states at once, which is the same failure a reward change
# causes and is severed the same way.
# Deliberately *not* moved with the session key. The convention here is that
# `SLOT_PREFIX` moves when a reward constant moves, because a stored return is
# then a different quantity -- and nothing about the label changed between v1 and
# v2. Keeping it inherits v1's 105 collected batches (0 updates, 0 checkpoints, so
# no lineage, LR or mirror-pool state comes with them), and `fill_buffer`
# relabels every record from its money trace on load, so they carry this session's
# objective whatever they were written with.
# Moved with the session: a slot's artifacts now hold a `members/` table, and a
# `rewardtogo_*` root has none. Reusing the prefix would hand the first prune a
# roster of checkpoints carrying no `member_lineage_id` at all.
#
# Deliberately *not* moved with the session key for v2, by the convention above:
# the slot count is not a reward constant, so a stored return is the same
# quantity it was. Keeping it inherits v1's 12 batches on each of slots 00-04,
# and inherits nothing else -- v1 was stopped in generation 0 with 0 updates and
# 0 checkpoints, so there is no lineage, LR or mirror-pool state to come with
# them. Slots 05-09 keep their 12 batches too; nothing drives or reads them now,
# and no query is scoped by `declared_model_id`, so they are inert rather than a
# pool the surviving five can draw from.
# Moved to "fed" with members-v3, and this one is a reward-grade sever even though
# no reward constant moved. What a slot inherits through the prefix is its member
# roster and its mirror pool, and members-v2's are inert: `trading_rate` 0.016
# against the gate's 0.02 floor, 8 of 105 updates clearing it. Keeping the prefix
# would hand the first prune a roster of policies the new activity tier is built to
# rank last, and `self_play_latest_backfill` would seed the mirror pool with the
# same ones -- the learner would train against opponents that do nothing, which is
# the loop being fixed. The 12-batch inheritance argument that kept the prefix from
# rewardtogo does not apply: `fill_buffer` relabels a stored record from its money
# trace, but nothing relabels a checkpoint's behaviour.
# The prefix moves with `SESSION_ID` as always, and note what that costs: the stored
# per-slot seed is `sha256(agent_type \0 slot_name)` (`identity.training_instance_seed`,
# applied at `storage.py:142`), so renaming the prefix reseeds every slot. That is
# forced by starting a new session, not an independent variable -- but it does mean no
# two sessions on this fork ever share a seed, and a single-generation difference
# between two of them is never attributable to config alone.
#
# It also means `TRAINING_AGENTS`' `"seed": 2718 + 1009 * index` below is dead: it
# lands in `instance.model_config`, and `register_training_agent` overwrites it with
# the derived seed whenever the instance name is not the default.
SLOT_PREFIX = "top25d"
MODEL_NAME = f"{SLOT_PREFIX}_00"

# How many learners run their round hooks and their update at once, and how wide
# each one's rollout pool is. The product is the core budget, so these two belong
# together. `training_concurrency` is a `ThreadPoolExecutor` width over *models*
# consumed by `list(pool.map(...))`, so it is a hard barrier and a half-empty
# tail is paid for in full. It equals `LEARNERS` from members-v1 on: five models
# over five threads is one wave with no tail. It was ten over five before, which
# is 2 x 5 -- also tail-free, but two waves to a round.
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
    # 400. Went to 800 once and came back; the history is below.
    #
    # Prioritisation is *episode*-level -- `_own_priorities` ranks whole episodes by
    # terminal money -- so this is the size of the ranked pool, not just a memory
    # dial. v1 ranked over 1200 trajectories and v2 over 400, and the comment that
    # sat here was copied from v1 without being updated. In rows v2 was not short
    # (its episodes held ~5.7x more decisions), but almost all of that excess was
    # the ten-slot wash, so it bought diversity of *rows* and none of trajectories.
    #
    # Back to 400. The watch-condition the previous comment set was hit: it asked to
    # "come back down if it is heading past ~4 GB", and `buffer_megabytes` across
    # generations 13-18 of `market-free-refusal-v1` read ~3.0 GB mean against a
    # **4.47 GB max**. The prediction was right -- a trained policy trades more and
    # the record grew -- so this is that comment being obeyed, not overruled. It is
    # also ~15-22 GB resident, which is a `CONCURRENCY` cost and not a `LEARNERS`
    # one: only the live wave's slots hold buffers, so cutting to five slots left
    # this figure where it was.
    #
    # The trajectory-diversity argument above is not lost to the cut, because
    # `batch_size` is now 128 rather than 8: the sampler draws sixteen times as
    # many distinct episodes per gradient step, so the ranked pool is consumed far
    # more thoroughly at 400 than it ever was at 800. Staleness improves too -- at 160
    # new episodes a generation, 800 held five generations of off-policy data.
    # Per *member*, not per slot. Each member collects `member_batches` x
    # `episodes_per_batch` = 10 x 16 = 160 episodes a generation, so 400 is 2.5
    # generations of its own history -- the same staleness ratio the slot-wide
    # 400 gave when one slot was one lineage. members-v2/v3 overrode this to 160,
    # which is exactly one generation of history, the regime `agent.toml` records
    # as the failure; the manifest's 400 is the value, restated here only because
    # this dict is what `ensure_session` digests.
    "buffer_episodes": 400,
    # Pure self-play rollout collection. The outside-policy anchor stays in the
    # pinned Kaggle ladder demonstrations below; the live rollout opponent pool is
    # this family only.
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
    # With `lineage_source = "latest"` the trained lineage moves past the release
    # every update; a mirror pool of promoted checkpoints alone would be a frozen
    # copy the learner has already left. Newest checkpoints top the pool up.
    "self_play_latest_backfill": True,
    "self_play_ratio": 1.0,
    # Pinned Kaggle ladder recordings are the outside-policy anchor for this run.
    #
    # `ladder-top25-001`, not `ladder-001`. The old pack's `min_final_money = 50_000`
    # was not a filter: a survey of all 2,100 mirrored seats puts the 25th percentile
    # at 75,335 and the median at 97,728, so 50k admitted 95% of the mirror and the
    # quality bar did no work. Its 128 pinned seats then came off the ladder-rank head
    # and reached 12 players, 34 of them one competitor -- the clone was fitting a
    # blend weighted by who ranked highest.
    #
    # The new pack is the top quarter by seat money (>= 118,418): 525 seats, 98
    # distinct players, median 134,255, ~0.998 mapped order coverage.
    # `ladder.stratified_indices` then round-robins the pinned 128 across players, so
    # they reach all 98 at no more than two seats each, at a 144,246 median -- against
    # the old head's 27 players, 42 from one, at 89,168.
    #
    # The clone loss is the bounded large-margin hinge, not the old cross-entropy that
    # inflated Q values. It still fits a blend: where two competitors disagree at a
    # state no single action clears the margin, and even sampling makes that blend
    # unweighted rather than making it a single policy.
    "demonstration_source": "ladder",
    "demonstration_pack": str(ARTIFACTS / "demonstrations" / "ladder-top25-001"),
    "demonstration_episodes": 128,
    "demo_fraction": 0.25,
    "behavior_clone_steps_per_update": 12,
    # Benchmark, not "rollout": the rollout arm reads `recent_mean_final_money`,
    # the *incumbent's* collection money while it filled this candidate's buffer.
    # It grades the parent, not the candidate.
    "promotion_score_signal": "benchmark",
    # Cross-slot rescue, which the manifest leaves off. `market-free-rewardtogo-v2`
    # is the case: at generation 16 its ten independent slots ran from 2,086 to 18,298
    # rollout money with no channel between them, and `nstep-v1` reproduced the same
    # shape at five. A slot whose own members are all more than 25% behind the roster's
    # best may import that checkpoint. `max_adopt` defaults to `members_per_slot - 1`,
    # leaving each slot one lineage of its own.
    #
    # The floor is 6,000, not the 3,000 `top25-v1` ran. 3,000 is the *stake*, which is
    # exactly what a policy that never trades scores -- and with the comparison at
    # `>=` that run imported such a checkpoint into position 0 of all five slots, each
    # of which then collected at 0.11 orders per decision. The floor is now twice the
    # stake, the comparison is strict, and `admit_foreign` additionally requires the
    # checkpoint to have measured *active* in its rank arena.
    "cross_slot_reseed": True,
    # 3,000 deliberately, against the manifest's 6,000: this run exists to find out
    # whether `admit_foreign`'s strict comparison and `active is True` gate refuse the
    # stake-scoring checkpoints on their own. Do not copy this value forward.
    "cross_slot_reseed_min_score": 3000.0,
    "cross_slot_reseed_margin": 0.25,
    # Promotion decides what is released; the lineage trains on regardless.
    # Root cause one of the flat run. No reject-streak snapback: under member
    # pools a lineage that stops earning its place is pruned, not walked back
    # (`updater._member_checkpoint`), so `resume_reset_after_rejections` is inert
    # and stays at the manifest's 0.
    "lineage_source": "latest",
    # The bar is the best of the last six releases, and a candidate inside the
    # arena's own confidence interval of it promotes. A strict all-time ratchet
    # against a 100-seed arena rejected the same policy measured twice.
    "promotion_reference_window": 6,
    "promotion_tolerance_ci": True,
    # Insolvent seasons train at full weight. See `agent.toml`.
    "terminal_quality_weighting": False,
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
    # exploring and throttles every actor. Setting this makes the argmax-flip
    # target (`noisy_flip_rate_target`, unset here and in the manifest) ignored;
    # the manifest zeroes *this* key, so the override is what turns the controller
    # on.
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
    # 3.0 from 6.0, two pairs from four. The learner seat made 4.5k a season on
    # weights the greedy mirror made 11.2k on, with the sigma controller already
    # pinned at its floor on half the slots -- the widened rare rows on half the
    # batch were what bankrupted the collection policy.
    "rare_sigma_cap": 3.0,
    # 2 of 8, down from 4 (see `rare_sigma_cap` above). Raising it is nearly
    # worthless anyway: `actor_sigma_scales` runs
    # [1.0, 0.61, 0.37, 0.23, 0.138, 0.084, 0.051, 0.031], so rungs 5-8 carry a
    # sigma scale under 0.14. The ladder is the attenuator, not the pair count.
    "rare_sigma_pairs": 2,
    # `learning_rate` is deliberately *not* overridden: the manifest's 2e-4 stands.
    # `market-free-weighted-v1` promoted nothing and showed Q-scale inflation at a
    # higher peak, and nothing since has earned it back.
    #
    # v3's chunk geometry, restated here because this file decides what the run is.
    # `burn_in_turns` must stay a multiple of
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
    #
    # **128, up from 32, and the step count does not move with it.** The reduction
    # above is the whole argument and it has not changed: a step's gradient is the
    # mean of `batch_size` per-chunk scalars, so `batch_size` *is* the count of
    # independent directions and `gradient_steps_per_batch` is not. The previous
    # comment measured the run at 8 against v2's 256 and fixed it to 32. That is
    # still an eighth of v2, on the one quantity that survives the reduction, and
    # v2 is the only lineage in this family whose benchmark money has ever climbed
    # (3,000 -> 13,229 over 33 updates against this fork's flat 4,053 -> 2,185 over
    # 20). 128 is half of v2's 256 rather than all of it because the cost is paid
    # in wall clock: over 970 completed updates an update is ~375s of which
    # `train_seconds` is ~124, so 4x the training is ~500s and an update lands near
    # 750s. Half the updates an hour, for 4x the directions in each.
    #
    # The step count is deliberately left at 48. Raising both is the "do not pay
    # twice" violation the paragraph above names, and steps were already traded
    # away 192 -> 48 to buy this exact quantity once.
    "batch_size": 128,
    "burn_in_turns": 20,
    "train_turns": 40,
    "gradient_steps_per_batch": 48,
    # `target_sync` is deliberately *not* overridden: the manifest's 1920 stands.
    # The faster 480-step sync `market-free-weighted-v1` used let the bootstrap
    # target chase the online net while `max_abs_q` inflated.
    "grad_clip_norm": 5.0,
    # False. The prefix is fresh, so there are no inherited Adam moments to drop;
    # True here would reset them on *every* update, since the key is re-read each
    # time. Set True for exactly one generation if a slot ever resumes trainer
    # state fitted to a different label scale.
    "reset_optimizer_moments": False,
}

# Demonstrations: every slot pins the `ladder-001` pack (128 episodes, a quarter
# of each TD batch, twelve large-margin clone steps an update), relabelled from
# its money trace on load so it carries the current reward. The history of how
# that became safe to do:
#
# `splitladder-v1` ran a five-and-five ladder split and the read came back void
# rather than negative: the seeded half's value function diverged on contact with
# the pack. `max_abs_q` by cutoff, seeded against
# control, from `model_updates.metrics_json`:
#
#     cutoff   20     30     60    100    130
#     seeded  0.49  20.82  26.98  21.33  18.35
#     control 0.57   0.50   0.45   0.49   0.60
#
# Cutoff 30 is the first update carrying `demonstrations`. `mean_loss` steps
# 0.0013 -> 1.194 on the same row and never comes back, against control slots at
# 0.0005 for the whole session. Every term in that era's `reward.py` was bounded --
# `outcome` was a tanh -- so a correct Q sat in single digits and 20 to 27 was a
# diverged head, not a scale.
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
# clone *stronger*. The measurement above said the opposite: it was already strong
# enough to set the head's scale and the TD loop spent the rest of the session
# pulling back. The hinge is built (`updater._large_margin_loss`), and the pack is
# pinned again under it.

# The roster is uniform -- one reward, one pack on every slot -- and the archive
# handles retention, with `rare_sigma_gain` handling generation on every slot.
# `LeagueConfig.validate` requires exactly one `primary` per agent type, so slot 0
# carries it -- deliberately not re-derived from scores, which would make every
# query resolving it non-reproducible. Distinct seeds are the only other thing
# separating the slots at the start; what diverges them afterwards is the member
# prune inside each slot (`training/members.py`), which keeps the top-ranked
# lineages and forks duplicates.
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
