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
# v5 moves again because the policy changed one more time: online promotion is now
# an activity gate, and lineage resumes from the best promoted benchmark-money
# checkpoint instead of following rejected checkpoints forward. The stored v4 model
# ids encode the old always-promote/export-best experiment.
SESSION_ID = "market-free-weighted-v5"
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
# Moved off "weighted4" with v5; the online gate and lineage parent policy both
# changed, so rejected idle checkpoints from v4 must not become ancestors here.
SLOT_PREFIX = "weighted5"
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
    # Deliberately inherit the manifest learning rate. `market-free-weighted-v1`
    # promoted nothing and showed Q-scale inflation, so this recovery run should
    # prove it can release before reintroducing a higher peak rate.
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
    # Deliberately inherit the manifest target sync. The faster 480-step sync used
    # by `market-free-weighted-v1` let the bootstrap target chase the online net
    # while `max_abs_q` inflated, so the recovery run starts with the conservative
    # lag documented in `agent.toml`.
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
    # Activity-gated lineage ignores rejected checkpoints, so there is no separate
    # rejection-streak snapback to apply.
    "resume_reset_after_rejections": 0,
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
