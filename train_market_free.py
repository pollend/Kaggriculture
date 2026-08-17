#!/usr/bin/env python3
"""Run the resumable free-market workflow.

Same shape as `train_market_scorer.py` -- one learner, one session, one
Q-network on top of a frozen pilkwang field policy -- and the same benchmark, so
the two are directly comparable. The difference is everything below the network:
this agent has no `market_exec`, no donor market rules, and no
`_schedule_market_adjustment`. Every turn it picks one order out of a flat list
of 186, or nothing, and has to learn affordability, the feed floor and terminal
liquidation from reward.

Expect this to be hard, and expect it to look like nothing is happening for a
long time. The two failure modes to watch, both recorded per batch in
`metadata_json` and per update in `metrics_json`:

    orders_per_decision ~ 0        the NOOP collapse: it stopped trading
    mean_final_money    ~ 0        the bankruptcy collapse: it traded itself broke

A healthy run has `orders_per_decision` somewhere in between and
`solvent_episodes` climbing off zero. Query them rather than guessing:

    SELECT batch_seq,
           metadata_json ->> 'orders_per_decision' AS orders,
           metadata_json ->> 'mean_final_money'    AS money,
           metadata_json ->> 'solvent_episodes'    AS solvent
    FROM training_batches
    WHERE model_id = 'dqn_market_free.raw_order_q_v1@simple_v2'
    ORDER BY batch_seq;
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
#
# The run is unbounded: generations are created and played one at a time until
# the process is interrupted. There is no annealing schedule to wait out --
# exploration is NoisyNet, and `noisy_order_rate_target` holds it at a measured
# rate rather than decaying toward one -- so a useful stopping point is a
# judgement about the metrics in the docstring above, not a number decided in
# advance. Ctrl-C after a generation closes; committed work stays resumable.
SYNC_ENVIRONMENT = True

# A fresh session and instance name.
#
# `market-free-demo-v1` predates the action space changing from ten order slots
# to one, which changed both the rollout record format and the network shape, so
# none of its checkpoints or batches can be read.
#
# `market-free-simple-v1` is readable but not resumable: `ensure_session` hashes
# the experiment manifest, and this config no longer carries the dead singular
# `benchmark` key, so its digest no longer matches the stored one. Its 60 batches
# and 2 rejected checkpoints stay queryable under `@simple_v1`; nothing was
# promoted, so nothing trained is being discarded.
#
# A new instance also means a clean artifact root rather than one where new
# `rollout-000000NN.npz` files land on top of the old run's.
#
# `market-free-simple-v2` ran 666 batches and promoted none of its 33 updates.
# Its measured failure was not slow learning: it spent the whole 3,000 stake on
# `HIRE 13` orders inside the first four turns of a 720-turn episode and then
# held NOOP for the rest, finishing at 0.7 coins against a never-trade baseline's
# 3,000. Generation 0 -- untrained -- was the best that policy ever was. Its
# batches stay queryable under `@simple_v2`; nothing promoted is being discarded,
# because nothing was.
#
# `market-free-selfplay-v3` changed three things that all bore on that: the hire
# mask priced the whole order, the reward charged for turns spent with no runway,
# and the opponent mix was mostly this model's own recent checkpoints. It ran 214
# batches over 10 generations and promoted none of its 10 updates. Its measured
# failure was neither of the two before it: the policy traded constantly and lost
# anyway, finishing at a median of 22 coins from a 3,000 stake, below the
# insolvency floor in 85% of episodes.
#
# What it was doing, from `market_orders` averaged over 300 episodes: 219
# `BUY_PRODUCT` and 204 `SELL` per episode, with wheat and fertilizer 94% of the
# sale volume -- exactly the two goods `BUY_PRODUCT` can buy back. It planted 13
# wheat an episode and offered 2,306 for sale. That is a round-trip loop, not a
# farm, and the simulator quotes it cash-neutral by construction (README: buy then
# sell against an unchanged market nets exactly zero). What paid for it was `Phi`,
# which marked the whole shed to a quote the agent had just moved by buying.
#
# So `market-free-outcome-v4` stopped paying for trades at all: `reward_mode =
# "outcome_only"` drops the money delta and the potential shaping together and
# leaves the margin against the opponent, spread over every turn by the annuity
# that was already there. See `training/reward.REWARD_MODES` for what that costs.
#
# A new model name as well as a new session id, and that pairing was
# load-bearing: `all_batches` is scoped by `model_id`, so reusing `selfplay_v3`
# would have reloaded its 214 shaped-scored artifacts into this run's buffer and
# fit one Q function across two reward functions.
#
# v4 is the first instance of this family that ever released anything: update
# `cutoff_batch_seq = 40` promoted at 5,984 money against the quiet baseline's
# 3,000, win rate 0.94, hard viability passed. It then diverged. The next three
# updates produced candidates finishing at 1.8, 2.5 and 0.8 coins while
# `mean_loss` climbed 4.48 -> 5.94 -> 9.63, against a reward stream bounded in
# roughly [-3, +2]. That is not slow learning.
#
# The mechanism is `outcome_only` plus a saturating `tanh`. With the money delta
# and the shaping gone, the whole stream is `tanh(margin / TERMINAL_MARGIN_SCALE)`
# plus two ~0.002 charges -- and against anything scripted that tanh is a
# *constant*, not a function of the action:
#
#   pinned demonstrations   surrogate seats at >=10k vs a learner at ~0   -> +1.0
#   anchor rollouts         learner at ~0 vs the surrogate's ~26k         -> -1.0
#   mirror rollouts         margins in the hundreds                       -> ~0.05
#
# Demonstrations were 25% of every minibatch and anchor games 35% of collection,
# so about half of every batch carried a flat +-1.0 label on states the network
# cannot tell apart -- both seats open at 3,000 coins under the same frozen donor
# field policy. No action explains that spread, it is irreducible target variance,
# and the Double-DQN max-bootstrap compounds it. `selfplay_v3` never showed it
# because the money delta anchored the regression there: its `mean_loss` is 0.002.
#
# So `market-free-mirror-v5` removes both saturated sources. No scripted anchor in
# `opponents`, no scripted family in the demonstrations, and the surrogate is gone
# from `BENCHMARKS` as well -- what is left is the mirror, whose margins are the
# only ones `TERMINAL_MARGIN_SCALE` was ever sized to discriminate, and pilkwang
# as the release objective the gate measures.
#
# What that costs, recorded rather than argued away: with the demonstrations gone
# this is the reward-only control, which is what `selfplay_v3` ran and which
# promoted nothing in 12 updates; and `train_dqn.py` records 17 consecutive
# rejections from a curriculum with no static opponent at all, because promotion
# is gated on a benchmark the policy never plays. Both are live risks here. The
# reason to take them is that neither of the things being removed could ever have
# changed an argmax.
#
# A new session id, because `ensure_session` digests `experiment.training_agents`
# and any change below makes v4's session unresumable. But the *same model name*,
# which is the opposite of the v3 -> v4 decision above and equally load-bearing:
# `_self_play_pool` reads `model_training_checkpoints` scoped by `model_id`, so a
# fresh name has an empty pool on batch 0, and with no base opponent to fall back
# on collection raises "free-market collection needs at least one opponent". A
# pure-mirror run cannot cold-start. Keeping `outcome_v4` was meant to give a
# live five-checkpoint pool from the first batch, keep the promoted seq-40
# checkpoint as the incumbent to beat, and reload v4's batches under an unchanged
# reward function -- which is the criterion, not the model name itself.
#
# That inheritance does not exist in this database. `training_batches`,
# `model_training_checkpoints` and `training_sessions` all return zero rows for
# `dqn_market_free`, so v6 raised on its first batch. Everything above about v1
# through v5 is the measured history of the run and stays as the record; what it
# can no longer be relied on for is a warm pool. The mirror now cold-starts
# behind a seeded-noise base opponent -- see `opponents` below.
#
# `market-free-noisy-v6` changes the exploration mechanism and nothing else.
# v5's low-scoring episodes overbuy, and the scheme rather than the policy is
# what produces them: the explore branch draws uniformly over feasible actions
# with NOOP removed, so an exploring turn *must* place an order, and 97 of the
# 186 actions spend money against 88 that sell. Epsilon is therefore a purchase
# pump whose rate is set by a schedule, not by anything the network believes.
# NoisyNet perturbs the weights instead, so the argmax moves only among actions
# already near-tied in Q and a ruinous purchase stays unreachable. Epsilon is
# zeroed rather than removed -- it is part of the policy identity string.
#
# Same model name again, for the reason given directly above, plus one more: the
# torch state dict from v5 stores `4.weight` where a noisy module wants
# `4.weight_mu`, so the resume goes through `_load_online_state`, which remaps mu
# and leaves sigma at its initialization. Sigma cannot be inherited -- there is
# none to inherit -- and it is deliberately not invented from the checkpoint.
# That migration also resets the Adam moments, because sigma adds tensors to the
# parameter group and `optimizer.load_state_dict` cannot span the change; it is
# the same event `reset_optimizer_moments` exists to express, once per lineage.
#
# `market-free-explore-v7` fixes what v6 measured. v6 promoted nothing in twelve
# updates with `trading_rate` 0.0 on every gate, and the cause was its own
# controller: the policy collapsed onto NOOP, where every action is near-tied and
# the argmax flip rate is therefore *high* -- 0.28-0.39 against a 0.05 setpoint --
# so `resolve_sigma_scale` divided sigma down to [0.33 .. 0.010] across the eight
# actors. Not one played at the learned sigma. `agent.toml` had warned in writing
# that targeting the flip rate on a collapsed policy would do exactly this.
#
# So the controller moves onto `noisy_order_rate_target`, which reads the NOOP
# attractor directly and cannot be fooled by cheap near-ties, and the cadence goes
# to "episode". Same model name once more: `_self_play_pool` is scoped by
# `model_id`, the reward is untouched, and neither change alters the checkpoint
# format, so the lineage carries straight over.
#
# `market-free-burst-v8` added the exploration bursts. v7 had worked: it promoted
# on its second update, `mean_abs_q` went 0.04-0.09 to 0.56-0.60, mean money left
# the 3,000 NOOP fixed point for 11-12k, and BUY_LAND came back from extinct at
# 7-10% of seats. What it did not do is make those purchases any good --
# land-buying seats finished at a 304 median against 13,844 for the rest, because
# the ladder only spreads *downward* from the setpoint and the day-0 purchase is
# the only one it can reach.
#
# `market-free-burst-v9` was v8's own measurement applied: burst rounds went flat
# and the rate went to 0.35. That resolved to sigma 3.159, and per-episode
# attribution -- joining `metadata_json -> 'seeds'` to `actor_sigma_scales` by
# pair -- settled the whole exploration question against it. At 3.159, across 212
# seats: 49% bankrupt against 8-26% on the ordinary ladder, and *zero* land
# purchases. Every land purchase in the run happened at sigma <= 0.871, most often
# at the *lowest* rungs. Land is chosen by the greedy policy, not found by noise.
#
# `market-free-landgate-v10` acts on that. The binding constraint was never the
# perturbation, it was the action space: `LAND_PRICES` opens at 1,000 against a
# 3,000 stake, so an ungated mask offers the purchase on turn 0 and that is the
# only version the policy ever saw -- a 304 median against 13,844 for seats that
# bought none. `feasible_actions` now mirrors the donor's day-and-reserve rule.
# The burst survives at a tenth of its v9 strength, and
# `resume_reset_after_rejections` drops to 2 so the run stops carrying diverged
# trajectories through three rejections at a time.
#
# `market-free-coldstart-v11` is the first run to start from nothing, and the
# first to move the model name since v4. Two things v10 measured forced it.
#
# `resume_reset_after_rejections = 2` was set from v10's *released* behaviour, and
# it is a wipe rather than a snapback on a run with nothing released: sixty
# batches of learning discarded and the network reinitialized to `quiet_start`,
# repeating every two rejections, which no cold start survives long enough to
# promote through. The reset now requires a release.
#
# And the controller floored collection sigma at 0.125 by update 20 -- the
# `1 / noisy_sigma_scale_limit` clamp, hit exactly -- because an untrained
# network's flip-to-order rate is high from randomness rather than from sigma.
# `noisy_sigma_scale_floor` stops it switching exploration off.
#
# The model name moves because the reasons that pinned it no longer hold: the
# database was reset, so there is no buffer or lineage to inherit, and
# `_self_play_pool` being empty on batch 0 is survivable now that `opponents`
# carries a seeded-noise base for `compose_opponents` to fall back on.
#
# `market-free-idleburst-v12` reverts and redirects. Scored against measurement,
# only the v7 change ever helped -- the controller retarget onto flip-to-order and
# `noisy_cadence = "episode"`, which took money from the 3,000 NOOP fixed point to
# 11-12k with two promotions in three updates. Everything after it was neutral or
# worse: periodic bursts twice (nothing, then 49% bankruptcy), and the land gate,
# under which the policy stopped buying land entirely rather than buying it later.
#
# So `feasible_actions` is back to pure feasibility, the periodic burst is off,
# and the burst machinery is repointed at the case it is actually good for. A NOOP
# policy produces no learning signal at all, so bankruptcy is the better failure;
# `noisy_actor_hot` forces it directly, on every batch.
# `market-free-rare-v20` makes exploration *directed* for the first time.
# Everything from v6 onward has been a way of asking NoisyNet for more or less of
# the same undirected thing, and the probe that motivated this says the quantity
# was never the problem. Over 23,008 states from
# v18's own rollouts, against its live checkpoint: `BUY_LAND` sits at median rank
# 33 of 104 feasible actions on a median Q gap of 0.031 -- inside what the
# perturbation crosses routinely -- and the greedy policy picks it on 0.0000% of
# states, while 55% of the argmax flips go to HIRE and BUY_SEED rows. That is the
# head-only factorized noise picking near state-independently, exactly as the
# `noisy_hot_order_rate` comment above predicts it would.
#
# The league agrees. Across v15-v18, 7,266 seats bought land 89 times, no seat
# ever bought a *second* quadrant, and `hot_v1` had it affordable on 88% of turns
# and bought none.
#
# So the lever is the correlation, not the magnitude. `rare_sigma_gain` widens
# the head's own sigma row for an action this run has barely sampled --
# `epsilon_out` is already per-output, so one entry moves one action and the other
# 185 stay where the controller put them. Zero-mean, never in the TD target, and
# bounded below at 1.0, so it can only add variance to a row.
#
# Not a bonus added to the Q scores, which was the first shape and the wrong one:
# a deterministic tilt makes a rare row win *every* turn its gap is under the
# coefficient rather than surfacing occasionally, which is a policy change rather
# than exploration and is what the epsilon runs bankrupted on.
#
# On the four hot pairs, overlapping the two `noisy_actor_hot` rungs. v9's
# finding that land is only ever bought at sigma <= 0.871 is about *global*
# sigma -- every row widened into indiscriminate trading -- and does not transfer
# to widening one row while the rest hold still.
#
# What it does not answer, and this is v15's finding rather than a caveat added
# here: seats that bought land finished at a mean of 355 coins against 2,753 for
# those that did not. Land is bought and then not farmed. Sampling the row more
# often is how the Q function gets the evidence to price it at all -- the bonus is
# generic over rare rows, not an opinion that land is good -- but if the mix
# improves and money does not, the comment on `noisy_hot_scale_limit` above named
# the next place to look before this run existed: the reward, or the mask.
#
# Same `MODEL_NAME` deliberately. The reward function is untouched, so v18's
# batches are the same quantity this run's buffer wants, and `all_batches` is
# scoped by `model_id`. Only the behaviour distribution moves.
#
# v21: stop tuning exploration. It was never the binding constraint, and v20
# proved it twice over.
#
# v20 ran 61 generations and learned nothing after generation 11, for a reason
# that had nothing to do with sigma. `resume_reset_after_rejections` fired at
# batch 281 and `lineage_checkpoint` held the snapback -- correctly, that is what
# it is for -- but every held update writes `lineage_reset`, and
# `rejections_since_promotion` used to stop counting at the newest one. So the
# streak read 0 on all 55 held updates, `promotion_ratchet_grace_after_rejections`
# could never fire, and the run resumed `9071d2ad` and trained exactly 2,000
# steps 55 times in a row: `gradient_steps` pinned at 24,100 for seven and a half
# hours, 57 consecutive rejections, nothing cumulative. Both halves are fixed --
# the streak counts through a held reset, and grace now drops the *floor* rather
# than the ratchet, because the floor was the unreachable term (candidate
# benchmark money 0.93-427 against a bar of 1,000, on a reference of 145.8 that
# could not have cleared its own floor either).
#
# The exploration numbers over those same 55 updates say the rest of it:
# `noisy_argmax_flip_rate` 0.24-0.46, `sigma_ratio_0` pinned around 3.5 at the
# `rare_sigma_cap`, `noisy_flip_to_order_rate` 0.07-0.14 against a 0.05 setpoint.
# The controller was live and above target throughout. There is nothing to turn
# up.
#
# What was actually missing is an opponent worth beating. `opponents` was a
# single seeded-noise agent, and the learner solved it at generation 6 -- paired
# margin flat at ~5,900 and a win rate of exactly 1.000 for the following 55
# generations -- while the gate measured it against pilkwang, which it had never
# once played. The comment on `self_play_ratio` below already named this failure
# from `train_dqn.py`; v20 reproduced it one level up. See `opponents` for what
# replaces it and why it is three entries rather than the manifest's five.
#
# `market-free-population-v23` is where the roster arrives, and the model names
# move with it for the first time since v10. They have to: `_self_play_pool`,
# `lineage_checkpoint` and the slot state are all scoped by `model_id`, so ten
# named slots are ten independent lineages and none of them may inherit v22's.
#
# What the stall looked like, since it is the whole reason for the change.
# `falloff_v1` over batches 1100-1688, by 50-batch bucket:
#
#     money  4997 5179 5745 6227 6872 5983 5074 4861 6527 3906 5559 4498
#
# No trend in either direction across 590 batches. `mean_loss` 0.0014-0.0017 and
# `mean_abs_q` 0.19-0.33 over the same span, both flat, and benchmark money
# against pilkwang between 0.9 and 1,208 from a 3,000 stake. Two promotions in the
# last twelve updates, each immediately followed by rejections. Every exploration
# lever tried from v7 through v20 aimed at the behaviour distribution; the thing
# that had stopped moving was the trajectory.
# v25 rather than resuming v24 because `rounds_per_generation` changed, and that
# is in the run manifest: `ensure_session` hashes the league config and refuses a
# resume whose digest moved, which is the right guard -- a generation would mean
# two different things inside one session's history.
#
# The learner slots do not restart. `training_models` is keyed on model id rather
# than on session, so the LR clocks, lineage checkpoints and the batch watermark
# all carry across, and the graduation clock keeps the updates it has already
# banked. What resets is the generation counter and the session's own rollups.
#
# v26: the reward changed, so this time the slots *do* restart, and the model
# names move with them for the first time since v23. That is the whole reason
# they move -- `training_models` being keyed on model id is exactly what would
# otherwise carry v25's checkpoints, its batches and its `_self_play_pool` into a
# run whose returns mean something different. A shaped return is not comparable
# across `OUTCOME_LOG_SCALE`, so every stored episode is the wrong quantity and
# every checkpoint was fitted to it.
#
# What v25 measured, and it is unambiguous. Eleven generations, 110 batches,
# 17,600 episodes, zero failures, and a policy worse than never trading: win rate
# 0.000 against pilkwang at every one of fourteen measured batches, benchmark
# money 98-2,717 from a 3,000 stake against pilkwang's ~142,000, `trading_rate`
# climbing 0.00 to 0.78 while money fell, and `own_solvent_fraction` between
# 0.003 and 0.028 throughout. It learned to act and every action lost.
#
# Three causes, landed together because none is observable without the others:
# the outcome term was a margin while the gate promotes on absolute money and was
# structurally *zero* in the 65% of rollouts played against the mirror
# (`training/reward.py`, `OUTCOME_LOG_SCALE`); `priority_alpha` was 0.0, so a
# uniform draw over a buffer that is 99.7% insolvent taught the only policy
# consistent with it (`agent.toml`); and the two opponent-wealth features were
# raw coin scales, so the promotion benchmark was judged fifty times outside the
# training distribution (`market_state.py`).
#
# This is the branch `LEARNERS` below already named: "if all ten slots decay with
# benchmark money still under 100, the binding constraint is the reward". They
# did not get that far -- `population.action` was `train` on all 109 updates and
# no slot has reached `lr_graduate_cut` -- but the benchmark money was there from
# the first generation, so the population question is deferred rather than
# answered. The roster stays; it is now ten independent reads on a reward change.
#
# **v26's own verdict, and what v27 changes.** v26 ran 37 generations, 366
# batches and 58,528 episodes with zero failures, and it did not meet its own
# stated pass condition -- benchmark money over the 3,000 idle baseline on a
# *majority* of slots, and `own_solvent_fraction` over 0.2. It reached 5 of 10 and
# one slot. On `episode_players` the median seat finished generation 35 on 3,112
# against generation 0's cold-start 3,000: +3.7% on doing nothing, over 33
# generations spent climbing back out of the hole dug in generations 1 to 5.
#
# Two causes, measured, and neither is the one the roadmap deferred.
#
# **The objective was 98% penalty.** Decomposing `own_mean_return` = -0.193 on
# `logmoney_00` at batch 360 against a mean final money of 3,038 on a 3,000 stake:
# the outcome term is worth +0.004 and `INSOLVENCY_PENALTY` -0.153. The term the
# reward exists to express was 45x smaller than a charge bought to price v2's
# permanent bankruptcy, which was instead pricing the wallet dipping under the
# floor between a purchase and the sale that funds it. `INSOLVENCY_GRACE_TURNS`
# is the fix and is why the slot prefix moves: a shaped return is not comparable
# across it.
#
# **And the run was reseeding its best slots.** `_slot_score` fed the plateau
# detector the windowed benchmark median, whose mean absolute update-to-update
# change over batches 200-360 is 48.2% against 11.7% for the rollout mean the same
# updates already computed. `_settle_slot` then latched `best_score` onto the
# maximum of it. `logmoney_07` was graduated on a draw it never repeated while its
# rollout money climbed 4,673 to 5,646, and reseeded down to 828; `logmoney_02`
# 3,497 to 332. The population mean turned over at batch 350, 3,518 to 2,930.
# `lr_score_signal` and `lr_best_decay` are the fix.
#
# `market-free-archive-v28` stops the pool being a ranking. Everything through v27
# selected the mature group on one scalar, and one scalar cannot hold a population
# apart. Per generation on `logmoney-v26`, distinct crops planted per seat against
# the seat rate of `BUY_LAND`:
#
#     gen        0     4     9    12    15    20    30    38
#     crops   3.31  3.66  2.32  1.37  1.29  1.85  1.82  1.89
#     land     1.4% 18.5%  3.1%  0.3%  0.3%  0.4%  1.6%  1.7%
#     money  8,726   454 1,669 2,460 3,365 3,054 4,061 3,916
#
# The roster converges onto a wheat monoculture by generation 15 and holds it, and
# money *rises* while it does -- so this is not something the promotion gate or
# `_population_score`'s activity discount can object to. A monoculture genuinely is
# the best thing the ranking has found. Ranking is the wrong instrument for keeping
# an alternative alive long enough to beat it. v25 and v27 reproduce the same curve
# on the same schedule.
#
# So `mature_reserved_capacity` holds 6 of the 30 berths for the best-scoring member
# of each behaviour cell rather than for score, `reseed_source_explorer_share` reads
# them on one restart in four, and `mature_reserved_min_money` floors what may
# occupy a cell -- without which the archive would preserve the day-0 land bankrupts
# that finish on a median of 7 coins.
#
# Retention is the whole change, because generation is already handled. v27 runs
# `rare_sigma_gain = 1.0` on all ten slots, so rare rows *are* being surfaced --
# every run spikes to 15-22% land purchase around generation 4. They are then
# selected away, because the perturbation diversifies where a slot starts and not
# what it does: a jittered monoculturist re-converges within a few updates, and the
# cull it re-converges into ranks on one number. That is why the collapse curve is
# identical across v25, v26 and v27 with `reseed_dropout` and, latterly, the
# directed widening on throughout.
#
# So the control arm is v27 itself -- same reward, same generator, same ten slots,
# no reserved berths -- and the money series stays directly comparable across the
# prefix because `reward.py` is untouched. `population_members` is keyed on
# `declared_model_id` and survives the rename, so the archive starts populated from
# v26 and v27's thirty live members and the first warm start has something to adopt.
# `market-free-rarerate-v29` folds the archive together with a second change, and
# the confound is stated rather than hidden: v28 ran for five minutes and produced
# no graduations, so nothing separable was lost by restarting on both.
#
# The second change is that the directed widening was never doing its job.
# `rare_sigma_gain` has been on for every batch since v25, and the folded counts say
# what it bought: at gain 1.0 the count statistic gave `BUY_LAND` 1.47x against a
# 3.0 cap and `BUY_SEED MELON` 1.085x, with one row of 186 ever reaching the cap.
# It also asks the wrong question -- MELON was feasible on 883,000 turns and chosen
# on 1,917, which is declined rather than rare, and a raw selection count cannot
# tell those apart. `rare_sigma_novelty = "rate"` reads the selection rate given
# feasibility against the rate under evenly spread attention, which can.
#
# So the two halves now match: the widening surfaces the crop the policy abandoned,
# and the archive keeps whatever comes back. v27 is still the control for both --
# count arm, no reserved berths.
#
# **And it carries a second, independent change: `LAND_POTENTIAL_WEIGHT`.** The
# two are folded into one run deliberately, because they act on different
# mechanisms and read apart cleanly in the data -- the rare-rate arm in the sigma
# metrics and `rare_sigma_novelty`, the land potential in the `BUY_LAND` seat
# share and seed conversion. What they share is a diagnosis: the policy finds a
# rare capital action, tries it, executes it badly and abandons it.
#
# The land half, measured on v27. The agent asks for ~311 seeds an episode and
# plants ~35, and the thing that would make the rest usable is land: a plot
# unlocks a quadrant, `donor._plan_roles` gives it crop roles, `donor._seed_needs`
# rises and `planting.py` converts the held seed. It tried exactly that -- land
# buying ran 3.6 -> 14 -> 21.5 -> 22.9 -> 12.3 -> 0 -> 1 percent of seats across
# generations 2 to 8, peaking at the money trough -- and over generations 2-6 the
# buyers planted 38.0 against 36.9, lost 22.6 plants against 15.1 and finished on
# a median of 2 coins against 400. The return arrives ten days later through a
# terminal outcome discounted `0.999^720 = 0.49` in 24-turn hops; the 1,000-coin
# cost lands immediately.
#
# `gamma * PHI(s') - PHI(s)` over unlocked quadrants scaled by the remaining
# season. Ng, Harada & Russell: it cannot change which policy is optimal, only
# when the credit arrives. So if land really is bad under the frozen donor labour
# this run declines it exactly as v27 did, and that is the answer rather than a
# confound -- which is what makes it safe to fold beside the rare-rate arm.
# `market-free-widecap-v31` raises `rare_sigma_cap` 3.0 -> 6.0 and changes nothing
# else. The clamp was binding on exactly two rows of 186 -- `BUY_LAND` and
# `BUY_ANIMAL SHEEP 4` -- so it was a ceiling on the rarest actions rather than the
# safety bound it reads as, and the safety it was doing nothing for is provided by
# the floor at 1.0 and the plain argmax instead.
#
# Weak reason for a session move, and it moves only because it has to:
# `ensure_session` digests `FREE_CONFIG`, so any key here breaks a resume. The
# reward is unchanged from v30, so v30's money series stays directly comparable and
# is the control -- same reward, same rate arm, same gain, cap 3.0.
#
# Expect a small effect. `BUY_LAND`'s unclamped multiplier at gain 0.5 is 3.53, so
# this is a 3.00 -> 3.53 move, which the episode-cadence sweep puts at roughly 6%
# -> 7-8% of episodes buying a quadrant. It is deliberately not paired with a gain
# rise: land buyers in v30 finish on a median of 48 coins against 3,000 for
# non-buyers, and amplifying the rate before `LAND_POTENTIAL_WEIGHT` demonstrates
# that land can pay would buy proportionally more bankruptcies.
# `market-free-melon-v32` is not a knob change. v31 and every session before it
# plateaued against the same wall -- benchmark money flat in a 3,000-4,000 band,
# win rate against the pilkwang gate exactly 0.000 across ~1,500 arenas in six
# sessions -- and the cause was a bug, not a hyperparameter.
#
# `planting.market_influenced_roles` overlaid held seed onto the donor's vacant
# crop roles and, when a crop had no planned tile of its own, fell through to
# `unassigned[0]`. `MELON` is last in `donor.CROPS` and `CROP_MIX` plans no TOMATO
# or STRAWBERRY tile anywhere, so every tomato and strawberry seed took a tile
# planned for melon -- the one crop worth 142 per tile-day against 24-40 for the
# rest. Measured on v31 generations 5+, 16,000 seats, by melon share of seed spend:
#
#     melon share    seats   mean money   median
#     80%+             270        4,371    2,353
#     40-80%         1,158        5,642    4,109
#     5-40%          6,445        2,870       64
#     under 5%       7,723        3,321    2,568
#
# A barrier, not a gradient: both pure strategies are viable and the path between
# them collapses to a median of 64, because a mixed purchase pays for melon tiles
# and then plants carrots on them. 14,168 of 16,000 seats sat in the two bad
# buckets. This is also why widening sigma made things worse rather than better --
# it sampled the valley more often, which is how v31 reached the same ceiling
# roughly 4x faster than v26 rather than passing it.
#
# Worse, the deficit was invisible: `market_state` feeds `donor._seed_needs` into
# the state but is handed the roles map *after* the overlay, so a plan wanting 10
# melon with 2 held reported a deficit of 2 rather than 8. Fixing the overlay
# repairs that feature rather than needing a new one, so `MARKET_STATE_DIM` is
# unchanged at 238.
#
# Land moves with it, and in the opposite direction to previous sessions. The
# mask now gates `BUY_LAND` on readiness rather than affordability alone -- see
# `order_space._land_is_survivable` -- because 49.4% of the 7,342 accepted
# purchases on record planted zero of their 25 new tiles and day-0 purchases
# finish on a mean of 190. `LAND_PURCHASE_BONUS` then pays 0.02 on the accepted
# purchase, which repays 40% of the 1,000-coin outlay at `MONEY_SCALE`: enough to
# exercise a row the policy has learned to avoid, not enough to make the land free.
# `LAND_POTENTIAL_WEIGHT` stays as it was; being policy-invariant it cannot fight
# the new term, and it still carries credit back toward the purchase turn.
#
# v31 is the control: same rate arm, same gain, same cap, and its money series
# stays directly comparable. The decisive read is the melon-share table above
# re-run on this session -- the 5-40% bucket's median rising off 64 is the barrier
# flattening, and seat share migrating into 40-80% is the win.
# `market-free-landbonus-v33` moves for one constant: `LAND_PURCHASE_BONUS` goes
# 0.02 -> 0.035. Nothing `ensure_session` digests has changed, so v32 would have
# resumed cleanly and silently -- which is the reason to move rather than a reason
# to stay. The reward constants are not in the digest, so a resume would leave one
# session's batch history holding two reward regimes and no column saying where the
# boundary is.
#
# What v32 showed. The 0.02 bonus did the thing it was bought for: land stopped
# being unlearned. v31 held a 0.13-0.16 buy rate for 70 batches after the term went
# in, where v26, v27 and v25 all spiked to 0.24-0.34 and decayed back to 0.005-0.01.
# Bought land is also worked -- 14.2 of 25 tiles planted per purchase on v32, 8.7 on
# v31 -- so `land_tiles_planted_per_purchase` is not the failing column.
#
# What it did not show is a rate on a run that started with the term. v32 bought a
# quadrant in 3.2% of episodes against v31's 14.3%, and 257 of its 295 purchases
# were a seat's first and only one (1.15 per buying episode against 1.62). The gate
# is not the binding constraint: purchases land on day 10.3 holding 1,965 coins,
# roughly twice `LAND_PRICES[0]`, so `_land_is_survivable` is clearing states the
# policy then declines. Declining on value is what this constant prices.
#
# v32 is the control and stays directly comparable -- same mask, same planting fix,
# same rate arm -- so the read is its own land columns re-run here: `buy_land_rate`
# rising off 0.032 while `land_tiles_planted_per_purchase` holds near 14. The known
# failure is the pair moving apart, and `mean_final_money` is what says whether the
# extra purchases paid; v32 sits at 3,300-3,400 by batch 40-50.
#
# `market-free-donormix-v34` moves because `ensure_session` digests `FREE_CONFIG`
# and `opponents` is in it, so v33 would raise "run manifest mismatch" on a resume.
# The reward is untouched, which is why `SLOT_PREFIX` stays where it is: v33's
# episodes are the same quantity under the same shaped return, so the buffer, the
# self-play pool, the lineage checkpoints and the slot LR state all carry forward
# and only the opponent distribution changes. v33 is the control -- same reward,
# same mask, same land constants -- and its benchmark series is the flat one this
# is trying to move.
#
# `market-free-throughput-v35` carries three throughput changes on top of v34's
# opponent mix. It is a separate session only because v34 had already collected a
# generation before they landed -- `CONCURRENCY` is in `LeagueConfig`, which
# `ensure_session` digests, so v34 raises "run manifest mismatch" on a resume.
# `SLOT_PREFIX` stays: none of these touch the reward, so v34's episodes are the
# same quantity and the buffer, self-play pool, lineage checkpoints and slot LR
# state all carry forward.
#
# They are measurements, not tuning -- taken over 40 seconds mid-generation on
# v33, the box ran **12.7 busy cores of 32** and `train_seconds` was 18-32 s of a
# ~1,900 s generation:
#
#   * `gradient_steps_per_batch` 100 -> 400. The replay ratio was 0.32 against an
#     ordinary DQN range of 1-8, and raising it costs ~5% wall clock because
#     learning is 1.6% of it. The largest single change here by some distance.
#   * `CONCURRENCY` 4 -> 5. Ten models over four threads is a 4/4/2 wave with a
#     half-empty tail; five is 2 x 5. The old "4 x 8 = 32 exactly" budget was
#     sizing against a saturation that never happened.
#   * `promotion_seeds` 50 -> 30. Not `eval_seeds`, deliberately: the comment on
#     that constant records the reversal that put it back to 100 and the arena
#     noise measurement behind it. `promotion_seeds` buys a paired significance
#     test `promote` does not consult.
#
# These move wall clock, not direction, so the benchmark series and the land
# columns mean the same thing here as on v33 and v34 and stay directly comparable.
# If generation time does not fall below ~1,900 s, `CONCURRENCY` is the one to
# revert first -- and v34's single generation is the like-for-like baseline for
# that, since it is the only other session carrying the donor in `opponents`.
#
# `market-free-lotbasis-v37` is the strong reason again, and the discovery behind
# it is worth more than the constant. `episode_rewards` builds `shaped` from
# `np.zeros_like(raw)` and never adds `raw` to it -- `collect.py` keeps `raw` as
# `raw_reward` for diagnostics and `replay.py` trains on `reward` -- so **the money
# delta has never been in the training reward**. A BUY_SEED and a NOOP score the
# same on the turn they happen, which is the ordering table `IDLE_PENALTY` already
# states, and the only term that ever tells them apart is the terminal outcome.
#
# The distance to it is longer than v36 believed. `CROPS[crop]["first_yield_day"]`
# at `turnsPerDay = 24` is 48 turns for WHEAT and CARROT, 192 for TOMATO and 240
# for MELON and STRAWBERRY -- and `planting.py` optimises for melon. Against
# `n_step = 24` that is ten bootstraps, not the two to four this comment used to
# claim, and `0.999 ** 240 = 0.786` on top.
#
# `LOT_POTENTIAL_WEIGHT` closes that distance with a potential over held cost
# basis: seeds at their price, standing plants at the price of the seed that
# became them, harvested units at that price over `max_yield`. `gamma * PHI(s') -
# PHI(s)` pays the credit on the purchase turn and telescopes, so the optimal
# policy provably cannot move -- the same guarantee `LAND_POTENTIAL_WEIGHT` buys.
#
# It replaces v36's `LOT_CREDIT_WEIGHT`, which moved a lot's realised proceeds
# from the sale turn back to the buy turn. That preserved the episode *total*,
# which is what made it look safe, but a preserved total is not a preserved
# ordering: at the sale state the debit was entirely in the future, so
# return-to-go fell by the full amount, while at the buy state the credit was
# immediate and the debit 240 turns away, so return-to-go rose by only 21% of it.
# It deterred selling roughly five times harder than it encouraged buying, and a
# 24-unit melon dump charged -0.03 on a single turn against an outcome annuity
# capped at 0.001 per step. It moved the credit assignment problem from buying to
# selling rather than removing it.
#
# `lots.py` survives the change as measurement, not as reward input, and two
# ledger bugs went with it: a harvest landing on a HIRE, BUY_LAND or BUY_ANIMAL
# turn opened a phantom lot dated to the harvest and priced at the outlay (field
# actions run before `_process_market`, and `main.py` emits HARVEST every turn),
# and `_YIELD_CAP` left room open for units that rotted or overflowed the shed.
# Both fed `seed_spend` and `unrecovered_seed_share` -- the columns this run is
# read on.
#
# The same discovery invalidates `LAND_PURCHASE_BONUS`'s stated justification: it
# priced itself as repaying a fraction of a -0.05 purchase-turn cost that does not
# exist, so 0.035 is unopposed and worth about half a fully idle season. Held
# anyway with the comment corrected in place -- v33 took the land rate 0.043 to
# 0.35 with tiles worked *rising* 12.3 to 16.3, and a constant that is working is
# not cut on the strength of a corrected comment.
#
# v35 is the control for everything except the reward: same mix, same throughput
# settings, same land constants.
#
# `market-free-mirroronly-v38` takes pilkwang back out of `opponents` and returns
# `self_play_ratio` to 0.65, so the rollout mix is [noise, self, self] again --
# what every session up to v33 played. `opponents` is a `FREE_CONFIG` key, so
# `ensure_session` refuses a resume and the id has to move.
#
# `SLOT_PREFIX` stays `lotbasis`: the reward is untouched, so v37's episodes are
# the same quantity under the same shaped return and the buffer, self-play pool,
# lineage checkpoints and slot LR state all carry forward. That inheritance is
# also what makes the comparison work -- v37's lineages continue into a different
# opponent distribution and nothing else changes.
#
# The evidence this revert runs against is recorded at `opponents` and is not
# small: greedy benchmark money left the 2,500-3,900 band it had held for six
# sessions within thirty batches of the donor joining the mix, reaching 9,096.
# What that series cannot do is attribute, because v35's throughput changes and
# `LOT_POTENTIAL_WEIGHT` landed inside the same window. This session is the way to
# find out: if greedy money falls back into the band on a mirror-only mix, the
# donor was carrying it.
SESSION_ID = "market-free-mirroronly-v38"
ARTIFACTS = ARTIFACTS_ROOT / "dqn_market_free"
LEAGUE_CONFIG = Path("configs/dqn_market_free_league.toml")
MANIFEST = Path("agents/dqn_market_free/agent.toml")

# Ten concurrent learner slots under one declared model, which the harness
# namespaces as `dqn_market_free.raw_order_q_v1@logmoney_00` and so on.
#
# The cost was wall clock and it was linear, because `run_round_hooks` and
# `_update_session_models` both walked the roster one slot at a time in one
# process. v23 measured what that costs: 77 minutes a generation, load average 9
# on 32 cores, max concurrent model updates 1. `training_concurrency` below is the
# fix -- the slots are independent lineages and there was never a reason to walk
# them -- and `SESSION_ID` moves to v24 because `ensure_session` digests both the
# league config and this file's `model_config`, so neither could change under a
# resume. The ten `MODEL_NAME`s stay, so all ten lineages, their checkpoints and
# v23's batches carry forward; only the session row is new.
#
# Ten rather than more for a second reason worth being honest about: `mean_abs_q`
# at 0.2 with `mean_loss` at 0.0015 is a Q head carrying almost no TD signal, and
# a population fixes a stuck trajectory rather than a flat one. If all ten slots
# decay to `lr_graduate_cut` with benchmark money still under 100, the binding
# constraint is the reward and the answer is Phase 3 of the roadmap, not more
# slots.
LEARNERS = 10
# `streak_` rather than `logmoney_`, for exactly the reason `logmoney_` replaced
# `learner_`. `_self_play_pool`, `lineage_checkpoint`, `all_batches` and the slot
# LR state are all scoped by `model_id`, and every one of them would otherwise hand
# v26's checkpoints and episodes to a run whose shaped return is a different
# quantity -- `INSOLVENCY_GRACE_TURNS` changes the labels on every transition in
# the ring, not just the ones collected after it. Renaming severs all four at once.
#
# Named for the change rather than numbered, so the prefix says which reward a
# `model_id` was fitted to without a lookup.
#
# `landrate_` rather than `streak_`, and this time for the strongest reason the
# rule has: `LAND_POTENTIAL_WEIGHT` changes the reward, so a v27 shaped return is
# not the same quantity. Every v27 episode in the ring would be mislabelled and
# every v27 checkpoint was fitted to the old labels, and `_self_play_pool`,
# `lineage_checkpoint`, `all_batches` and the slot LR state are all scoped by
# `model_id` -- renaming severs all four at once.
#
# The rare-rate half needed no severing on its own. It gets one anyway by being in
# the same run, which is the honest cost of folding two changes together and is
# recorded here rather than left for someone to rediscover from the prefix.
#
# `landbonus_` rather than `melon_`, and it is the strong reason again:
# `LAND_PURCHASE_BONUS` moves 0.02 -> 0.035, so a v32 shaped return is not the same
# quantity. Every v32 episode in the ring is mislabelled against it and every v32
# checkpoint was fitted to the old labels.
#
# `lotcredit_` rather than `landbonus_`, and the two names come back together for
# the strongest reason the rule has: `LOT_CREDIT_WEIGHT` changes what a shaped
# return *is*. Every v33-v35 episode in the ring is mislabelled against it and
# every checkpoint from those sessions was fitted to the old labels, so the
# buffer, `_self_play_pool`, `lineage_checkpoint` and the slot LR state all have
# to be severed -- which renaming does in one move.
#
# `lotbasis_` rather than `lotcredit_`, same rule, same strength.
# `LOT_POTENTIAL_WEIGHT` replaces `LOT_CREDIT_WEIGHT` outright rather than
# retuning it, so a v36 shaped return is a different quantity again -- and this
# one is not even the same *kind* of term, a state potential where v36 had a
# trajectory reallocation. v36's episodes and checkpoints sever with the rename.
SLOT_PREFIX = "lotbasis"
MODEL_NAME = f"{SLOT_PREFIX}_00"

# How many learners run their round hooks and their update at once, and how wide
# each one's rollout pool is. The product is the core budget, so these two belong
# together.
#
# v23 ran the population at the default concurrency of 1, which is the right
# default for one trainable model and badly wrong for ten. Measured over its first
# three generations on a 32-core box: max concurrent model updates 1, load average
# 9, and a generation taking 77 minutes made of 37 minutes of collection and 37 of
# the promotion gate, both walked one learner at a time behind `jobs = 8`. Twenty
# four cores idle throughout.
#
# The gate is the surprising half. An update averaged 222.6 seconds of which
# `train_seconds` was 8.2 -- 96% of it is the arena, because `eval_seeds = 100`
# against the benchmark plus `promotion_seeds = 100` against the incumbent, both
# seat-swapped, is 400 episodes per learner per generation against collection's
# 320. More than half the machine's episodes are evaluation, and none of them land
# in the `episodes` table, so the dashboard reports 3,200 a generation while the
# box actually ran 7,200.
#
# 4 -> 5 at v34, and the reason the old "4 x 8 = 32 exactly" arithmetic was wrong
# is that it was sizing against a saturation that never happened. Measured over 40
# seconds mid-generation on v33: 37.6% user, 35.2% iowait, 24.9% idle -- about
# **12.7 busy cores of 32**, not 32. The budget was reserving a machine the run was
# never using.
#
# What actually caps it is the wave structure. `training_concurrency` is a
# `ThreadPoolExecutor` width over *models*, capped `min(concurrency, len(models))`
# and consumed by `list(pool.map(...))` in both `model_trainers.py` and
# `league.py`, so it is a hard barrier. Ten models over four threads runs 4 / 4 / 2
# -- three waves, the last at half occupancy, every round. Five runs 5 / 5: two
# full waves, and the half-empty tail disappears.
#
# The oversubscription the old comment worried about is already handled elsewhere:
# `sim/workers.py` pins `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 on every
# worker, so 5 x 8 = 40 single-threaded processes against 32 cores is a scheduler
# queue and not the BLAS thrash that comment describes. Verified: 150 threads
# across all python processes, not the ~1,300 an unpinned cohort would carry.
#
# If the occupancy sample does not move after this, the bottleneck is not the wave
# structure and the answer is to revert rather than push on to
# `CONCURRENCY = 10, JOBS_PER_LEARNER = 3` -- which is the same idea at one wave.
CONCURRENCY = 5
JOBS_PER_LEARNER = 8

DONOR = "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"

# The release objective, and still the only benchmark. Index 0 is what the ratchet
# reads; adding a second here would only add non-regression diagnostics the gate
# does not consult.
#
# The rule this list used to be justified by -- "nothing this run measures is a
# policy it never plays" -- was the right rule and v20 broke it in the other
# direction: the benchmark stayed correct and `opponents` stopped containing it.
# That claim was written as already true and was not: `opponents` carried seeded
# noise alone from v21's revert through v33. v34 makes it true -- see `opponents`.
BENCHMARKS = (DONOR,)

# Overrides only. `register_training_agent` materializes `agent.toml`'s
# `[models.config]` and then applies this on top, so every key here is a
# deliberate departure from the manifest.
FREE_CONFIG = {
    # Promotion baseline first: index 0 is what the release gate measures, the
    # rest are non-regression diagnostics.
    "benchmarks": list(BENCHMARKS),
    "episode_steps": 720,
    # The floor grace falls back to, and 0.0 rather than the 750 default because
    # this lineage is being deliberately unstuck. v20's 57 rejections all failed
    # the absolute floor, and the best benchmark money it ever measured was 427,
    # so any positive floor keeps the deadlock intact and the fix does nothing.
    #
    # Zero does not mean the gate stands down. `no_regression` stays on under
    # grace and is what actually decides: the reference is the incumbent's own
    # recorded 145.8, so a release still has to measure at or above 131. Eight of
    # v20's 57 rejected updates already did. Hard viability -- the money ratios
    # against the incumbent's seat and the `min_orders_per_decision` floor -- never
    # stands down at all, and it passed on every one of those 57, so it is a real
    # bar rather than a formality.
    #
    # Expect this to hold only until the first promotion. Once something releases,
    # the ratchet reference moves up with it and the run is back on a rising bar.
    # Restated here rather than left to the manifest because it is now half of a
    # budget: `CONCURRENCY * JOBS_PER_LEARNER` is what lands on the cores, and the
    # two numbers have to be read together to see that.
    "jobs": JOBS_PER_LEARNER,
    # How many slots the roster has. Read only by `_reseed_source`'s whole-roster
    # spread, which needs to know which ordinals are the tail before it can hand
    # them the archive's reserved cells. Declared rather than derived because a
    # slot is a separate process that never sees the others.
    "population_slots": LEARNERS,
    "benchmark_grace_min_money": 0.0,
    # 400 was throwing away most of what collection paid for. One generation is
    # 20 rounds x 1 batch x 16 episodes = 320 episodes into a 400-episode ring
    # against 2,000 gradient steps, and v20 finished with
    # `skipped_evicted_batches` at 1,212 of 1,361 -- 89% of every episode ever
    # collected aged out of both segments before a gradient step opened it.
    #
    # 1200 is three generations of ingest, so a batch survives long enough to be
    # sampled more than once and the n-step returns get drawn from something wider
    # than the last twenty rounds. It costs memory linearly -- v20 reported
    # `buffer_megabytes` at 332 for 400 episodes, so expect roughly 1 GB.
    "buffer_episodes": 1200,
    # One base opponent: seeded noise.
    #
    # This was `[]` -- pure mirror play, which `compose_opponents` expresses by
    # returning the self-play pool unmixed when there is no base. That is only
    # reachable when the inherited `MODEL_NAME` already has checkpoints, and in
    # this database it does not: `training_batches`, `model_training_checkpoints`
    # and `training_sessions` are all empty for `dqn_market_free`, so batch 0
    # found an empty pool, fell back to an empty base, and collection raised
    #
    #     free-market collection needs at least one opponent
    #
    # A base opponent is therefore required to cold-start, and it has to be one
    # this reward can actually read. `outcome_only` pays
    # `tanh(margin / TERMINAL_MARGIN_SCALE)`, and every scripted competitor
    # saturates it: measured over 2,134 seats each, pilkwang finishes at 139k,
    # lucifer19 at 157k, tetsutani at 156k, kaitofukami at 159k, and
    # `configurable_crop_surrogate` -- the weakest wall available -- at 32k.
    # Against a learner at ~0 all of those are a constant -1.0, and a constant
    # added to every terminal reward can never change an argmax.
    #
    # Seeded noise is the one opponent that is not a wall. It plants at random,
    # trades on a 10% coin, and finishes in the same range a barely-trained
    # learner does, so the margin against it stays inside the linear part of the
    # tanh and carries gradient. It is also reproducible -- the built-in `random`
    # agent builds a fresh unseeded RNG per call, which is why this goes through
    # `kaggriculture_lab.opponents:make_seeded_random` instead.
    #
    # v21 tried widening this to three -- noise, `configurable_crop_surrogate` and
    # pilkwang -- on the argument that the gate measures an opponent the run never
    # plays, which is the shape `train_dqn.py` recorded at 17 rejections and v20
    # reproduced at 57. It ran 15 rounds and 120 pairings before being reverted,
    # and those 120 are why this is one entry again:
    #
    #     seeded noise    +6,241   win rate 1.000
    #     mirror          -5,015   win rate 0.333
    #     surrogate      -29,376   win rate 0.000
    #     pilkwang      -134,558   win rate 0.000
    #
    # The surrogate was supposed to be the middle rung, on the reading that its 32k
    # is close enough to the learner's 9-13k mirror scores for the tanh to still
    # have slope. It is not: against the surrogate the learner scores 2-3k rather
    # than 9-13k, so -29,376 through `tanh(margin / 10000)` is -0.995 with a slope
    # near 0.01. Both new slots were walls, and together they took 22% of
    # collection to deliver a constant.
    #
    # There is no mid-strength opponent in this repo. The paragraph above is right
    # as written: noise is the only static entry carrying terminal gradient at this
    # strength, and adding saturated ones does not change that.
    #
    # The deadlock those slots were meant to break is handled at the gate instead,
    # by `benchmark_grace_min_money` above, which is the half of the fix v20's data
    # actually pointed at. If the run stalls again with that hatch open, the
    # binding constraint is `TERMINAL_MARGIN_SCALE` rather than the opponent list:
    # no opponent finishing 30-140k above the learner can teach it anything through
    # a tanh sized at 10,000.
    #
    # ---- v34 ----
    #
    # It stalled again with that hatch open, and the sentence above named the right
    # constraint. What it did not anticipate is that `TERMINAL_MARGIN_SCALE` would
    # be gone by the time the prediction came due. `OPPONENT_WEIGHT` is 0.0 and the
    # outcome term is `tanh(log(final / stake) / OUTCOME_LOG_SCALE)`, read off the
    # learner's own purse. **The whole v21 table above is a margin table for a
    # reward that no longer reads margins.** Pilkwang finishing 134,558 ahead is not
    # a constant -1.0 any more; it is not in the reward at all. What the learner
    # scores in the market pilkwang leaves behind is, at full slope.
    #
    # So pilkwang joins the base, and the argument for it is v33's own numbers.
    # Greedy benchmark money sat in the 2,500-3,900 band across all 15 measured
    # cutoffs -- the same wall v26 through v32 hit -- while head-to-head money
    # climbed to 5,891 against an incumbent at 7,676. The mirror was improving
    # against itself and nothing was improving against the objective.
    #
    # One slot's gate record says why, at cutoff 150, one checkpoint, two markets:
    #
    #     vs its own incumbent   0.56 quadrants/episode   money 3,215
    #     vs pilkwang            0 quadrants, 26,455 of ~31k seeds WHEAT, money 1,849
    #
    # The land behaviour `LAND_PURCHASE_BONUS` bought -- 0.35 of episodes, 16 of 25
    # tiles worked -- exists only in markets pilkwang is not in. The policy has
    # never once played the market it is graded in, which is verbatim the failure
    # `self_play_ratio` below cites `train_dqn.py` for.
    #
    # Noise stays, and not out of caution: it is still the only opponent whose own
    # money lands in a cold-started learner's range, and a reseeded slot restarts
    # from there. This is a two-entry base, not a replacement.
    #
    # ---- v38: the donor comes back out ----
    #
    # Reverted by decision, and the measurement is recorded here against it rather
    # than argued away, because it is the strongest result this family has and the
    # revert is being taken anyway.
    #
    # Greedy money against pilkwang, `landbonus` prefix, donor entering the base at
    # batch 161:
    #
    #     cutoff   greedy money   promoted   own_return
    #     150         3,164         3/10       -0.611     <- v33, no donor
    #     170         3,659         1/8        -0.497
    #     190         6,556         3/8        -0.555
    #     210         8,044         3/8        -0.417
    #     220         9,096         4/8        -0.353
    #     230         7,951         3/7        -0.318
    #
    # v26 through v33 sat in a 2,500-3,900 band across every measured cutoff. That
    # band is the wall this file has documented six times, and it broke within
    # thirty batches of the donor joining the rollout mix. Rollout money over the
    # same span went 6,515 to 10,912 and solvent episodes 4.1 to 7.1.
    #
    # What that does not settle is attribution: v35 folded three throughput changes
    # in at batch ~170 and `LOT_POTENTIAL_WEIGHT` landed later still, so the series
    # above is not a clean single-variable read. Taking the donor out is one way to
    # find out which of them carried it -- if greedy money falls back into the band
    # on a mirror-only mix, the donor was the cause and this comment is the argument
    # for putting it back.
    #
    # Cost of the revert, stated so it is not rediscovered: a quarter of collection
    # returns to the mirror, and the policy stops playing the market it is graded
    # in -- which is the failure `self_play_ratio` cites `train_dqn.py` for, and
    # which the v34 block above was written to fix.
    "opponents": [
        "kaggriculture_lab.opponents:make_seeded_random?seed=7",
    ],
    # The league population is this same family, and `league_opponents` excludes
    # it by construction, so leaving this on would only re-add anchors.
    "league_opponents": False,
    # 1, not 0: `self_play_after` is read as `if self_play_after and batch_index
    # >= self_play_after`, so 0 is the manifest's *disable* value rather than
    # "from the start". 1 keeps the mirror on from the earliest batch that can
    # have anything behind it.
    #
    # It gates on a batch index, but the real gate is `_self_play_pool` being
    # non-empty, and a checkpoint is only written by a model update -- one per
    # generation, at `min_new_batches = 20`. So generation 0 is noise-only warmup
    # whatever this value is, and the mirror joins at g1 against the first
    # checkpoint. The noise opponent is what makes that warmup legal instead of
    # fatal.
    "self_play_after": 1,
    "self_play_pool": 5,
    # Live again now that the base is non-empty, so it is set rather than left to
    # the 0.3 default. `compose_opponents` solves `slots / (len(base) + slots)`
    # in whole slots against a single-entry base: 0.3 and 0.6 both round down to
    # one slot and land on 50%, 0.65 yields [noise, self, self] -- two of every
    # three rollouts against the mirror, one against noise. The anchor slot is
    # not decoration: `train_dqn.py` records 17 consecutive rejections from a
    # curriculum with no static opponent, because promotion is gated on a
    # benchmark the policy never plays.
    #
    # v34 took this to 0.5 because the base was two entries: the same solve gave
    # `repeats = 1`, `slots = 2`, so the list was [noise, pilkwang, self, self] --
    # half mirror, a quarter noise, a quarter the policy the gate measures.
    #
    # v38 puts it back to 0.65 with the donor out of the base, which restores the
    # single-entry solve described above exactly: [noise, self, self]. The two
    # constants move together or the mix is neither of the things it was designed
    # to be -- 0.5 on a one-entry base is a 50/50 mirror-noise split, which is
    # twice the noise share either version wanted.
    "self_play_ratio": 0.65,
    # No demonstrations. v4 pinned 200 `configurable_crop_surrogate` seats and
    # cloned their order choices, and the clone itself worked -- behaviour-clone
    # loss fell 0.42 to 0.06 over five updates. What it also did was put a
    # saturated +1.0 outcome on 25% of every minibatch, opposite the anchor's
    # -1.0, which is half the divergence described in the session comment.
    #
    # Nothing replaces it. The pilkwang donor is the family worth cloning and it
    # has zero rows in `episode_players` -- `league_opponents` is off and it only
    # appears as a promotion-gate benchmark, whose matches are not persisted as
    # league episodes. That is the same reason v3 asked for demonstrations and
    # silently got none. Recorded here so the next attempt starts from the real
    # blocker: to clone the donor, its gate matches have to be persisted first.
    "demonstration_episodes": 0,
    # NoisyNet takes over exploration, and epsilon is zeroed rather than deleted.
    #
    # Epsilon's explore branch draws uniformly over feasible actions with NOOP
    # removed, so every exploring turn is forced to place an order, and 97 of the
    # 186 actions spend money against 88 that sell. At 720 decisions and a 0.121
    # resume rate that is ~87 forced random orders an episode, most of them
    # purchases -- which is the overbuying the low-scoring half of every batch
    # shows, and it is manufactured by the exploration scheme rather than learned.
    # Perturbing the weights moves the argmax only among actions already near-tied
    # in Q, so a purchase the policy has learned is ruinous stays unreachable.
    #
    # The key stays at 0.0 rather than being removed because it is part of the
    # policy identity string: dropping it would rewrite every revision id in the
    # family, and the manifest remains the epsilon-greedy control arm.
    "noisy": True,
    "noisy_layers": "head",
    # "episode", up from v6's "day". The n-step argument the manifest makes for
    # "day" is satisfied strictly harder here: it wants each 24-step return to
    # come from one perturbed policy, and one policy per 720-turn episode gives
    # that trivially. What the change buys is temporally extended commitment --
    # the property that separates deep exploration from dithering. A perturbation
    # redrawn every 24 turns cannot hold a plan across the 48-96 turn
    # BUY_SEED-to-harvest cycle, let alone the day-5-buy-land to day-15-harvest
    # chain, so any action whose payoff needs a consistent hypothesis over that
    # span is unreachable no matter how large sigma is.
    #
    # The manifest records the cost and it is real: 8 distinct policies per batch
    # against day cadence's 480, and one bad draw wastes a whole 720-turn episode.
    # Taken knowingly. The per-actor sigma ladder still spreads behaviour *across*
    # the eight pairs, which is what the 81%-pairwise-agreement measurement that
    # motivated the ladder was actually about.
    "noisy_cadence": "episode",
    # Control sigma on the flip-to-order rate, not the argmax flip rate. See the
    # session comment: on v6 the argmax rate read a NOOP-collapsed policy as
    # exploring and throttled all eight actors for it. Setting this ignores
    # `noisy_flip_rate_target`, which is zeroed below so the config says so
    # outright rather than relying on the precedence rule.
    #
    # 0.05 carried over from the flip-rate setpoint deliberately, so the first
    # comparison between the two controllers is at the same number. It is a
    # harder ask on this quantity than on the flip rate, which is the point.
    "noisy_order_rate_target": 0.05,
    # A floor under the resolved scale, because the implicit `1 / limit` clamp is
    # not one. On `landgate-v10`'s cold start the controller resolved exactly
    # 0.125 by update 20 -- the clamp, hit dead on -- with `mean_abs_q` at 0.046:
    # an untrained network's flip-to-order rate is high because the network is
    # random, not because sigma is large, so the controller read over-exploration
    # and throttled it before there was anything to exploit.
    #
    # 0.5 rather than 1.0 so the controller can still turn sigma down when the
    # policy genuinely over-trades; it just cannot switch exploration off. The
    # 32x actor ladder still supplies the near-greedy read either way.
    #
    # **Down to 0.0, because it stopped being a floor and became a binding
    # constraint.** Averaged over `logmoney-v26` batches 320+, the resolved scale
    # per slot was 0.51 / 0.52 / 0.63 / 0.63 / 0.70 / 0.71 / 0.94 / 1.16 / 2.00 /
    # 5.78. Three slots sat *on* 0.5 with a measured `noisy_flip_to_order_rate` of
    # 0.13 to 0.20 against the 0.05 target -- exploring two to four times harder
    # than asked and unable to come down. The cold-start case the floor was bought
    # for is upstream of that: it fires on an untrained network, and
    # `lr_warmup_updates` plus `viability_cold_start_reference_money` both cover it.
    "noisy_sigma_scale_floor": 0.0,
    # And a ceiling to match, because the divergence ran both ways: `logmoney_01`
    # resolved 5.78 and still only reached a flip-to-order rate of 0.022, which is
    # a controller that has run away rather than one that is working. It promoted 6
    # of 36 updates on 2,204 money, the worst pair in the roster. 8.0 was sized as
    # a clamp that should never bind; 2.0 is sized to what a converged slot
    # resolves, which the same table puts under 1.2 on eight of ten.
    "noisy_sigma_scale_limit": 2.0,
    # Directed exploration, on for the first time here: the head's sigma row for
    # an action this run has barely sampled is drawn up to 3x wider than the rest.
    # The manifest defaults the gain to 0.0 and stays the undirected control.
    #
    # 1.0 and a 3.0 cap, against `sigma_ratio_0` around 6 on this family: an
    # unvisited row is perturbed hard enough to surface in an argmax it currently
    # loses by a median 0.031, and a row with a few hundred visits is back to
    # roughly the sigma the controller resolved.
    #
    # Four of the eight pairs, from the widest end of the ladder. The greedy half
    # carries none of it, which is what keeps `mean_final_money` comparable
    # across the change and keeps a clean read of what the policy actually does.
    # **On the rate arm from v29.** The widening was on for every batch of v25
    # through v28 and was doing almost nothing, which the folded counts say
    # directly: at gain 1.0 the count statistic gave `BUY_LAND` a 1.47x multiplier
    # against a 3.0 cap and `BUY_SEED MELON` 1.085x, and only one row of 186 ever
    # reached the cap. `log1p(1,150,400)` is 13.96, so the numerator is effectively
    # constant and the spread is compressed into `1 + n_a`.
    #
    # Worse, it is the wrong question for the failure in front of it. MELON was
    # feasible on 883,000 turns and chosen on 1,917 of them -- not rare, declined --
    # and a raw count cannot tell those apart. The crop collapse is invisible to it.
    #
    # 0.5 rather than 1.0 because the two statistics are not calibrated alike, and
    # 0.5 is where the rate arm's ordering is widest without saturating. Measured on
    # the same counts: WHEAT q1 1.000, CARROT q4 1.400, MELON q1 1.846, TOMATO q1
    # 2.310, BUY_LAND 3.000, with 10 of 186 rows at the cap. At 0.75 a third of the
    # space saturates and the ordering between the rare rows is destroyed, which is
    # the whole thing being bought here.
    "rare_sigma_gain": 0.5,
    "rare_sigma_novelty": "rate",
    # 6.0 rather than 3.0, because at 3.0 the clamp was binding on exactly the row
    # the mechanism exists for. Measured on v30's folded counts, 1,133,396
    # selections: two rows of 186 sat at the cap -- `BUY_LAND` and `BUY_ANIMAL
    # SHEEP 4` -- and every other row, including all five crops, was strictly
    # inside it. So the cap was not a safety bound, it was a ceiling on the two
    # rarest actions and nothing else. At 6.0 nothing saturates and the ordering
    # across the rare rows is what the novelty statistic says it is.
    #
    # What this is *not*: a lever on the crop distribution. `BUY_SEED WHEAT` and
    # `CARROT` sit at exactly 1.000 and `MELON`/`TOMATO`/`STRAWBERRY` at 1.74/1.63/
    # 1.87, identical at cap 3.0 and cap 10.0, because none of them is clamped.
    # `rare_sigma_gain` is the knob that moves those.
    #
    # And it is a smaller change than it reads. `BUY_LAND`'s unclamped value at
    # gain 0.5 is 3.53, so this takes it 3.00 -> 3.53, which the episode-cadence
    # sweep puts at roughly 6% -> 7-8% of episodes buying at least one quadrant.
    # Reaching a land multiplier of 6 needs gain ~1.0; 10 needs gain ~1.8. Left at
    # 0.5 deliberately: land buyers in v30 finish on a median of 48 coins against
    # 3,000 for non-buyers, so amplifying the rate before `LAND_POTENTIAL_WEIGHT`
    # shows it can pay would buy proportionally more bankruptcies.
    "rare_sigma_cap": 6.0,
    # 4 of 8, and raising it is nearly worthless: `actor_sigma_scales` runs
    # [1.0, 0.61, 0.37, 0.23, 0.138, 0.084, 0.051, 0.031], so rungs 5-8 carry a
    # sigma scale under 0.14 and widening them adds almost no reachable swing. The
    # ladder is the attenuator, not the pair count.
    "rare_sigma_pairs": 4,
    # A resume restores the optimizer's learning rate from the checkpoint, so
    # this key was previously inert on every generation after the first. It is
    # now re-applied from config each update, and under the population it is the
    # schedule's *peak* rather than a constant: a slot warms up to it from zero
    # after a reseed and halves away from it on every plateau.
    "learning_rate": 1e-4,
    # The slot schedule, declared here as well as in the manifest because this is
    # the file that decides what the run is. Everything else about the population
    # -- `mature_capacity`, the reseed perturbation, the plateau terms -- takes
    # the manifest's value, so a change to any of those is a manifest change and
    # shows up in `ensure_session`'s digest the same way.
    #
    # 1e-5 against a peak of 1e-4 needs `lr_decay_factor**d < 0.1`, which at the
    # manifest's 0.25 is two decays. At `lr_plateau_patience = 4` that is eight
    # stalled updates, plus three of warmup and one spent setting the baseline:
    # twelve updates, and so twelve generations here, since `min_new_batches` and
    # `rounds_per_generation` are both 10 and a generation is exactly one update.
    # Against the measured stall that is still generous rather than trigger-happy:
    # `falloff_v1` went 57 consecutive rejections without anything ending.
    "lr_graduate_cut": 1e-5,
    # Back on, because the interaction that switched it off no longer exists. The
    # argument was that a stalling slot is repeatedly restored to the incumbent
    # whose score is *exactly what the plateau detector reads*, so `lr_best_score`
    # is pinned by the mechanism meant to rescue it and the slot never reaches its
    # cut. That coupling ran through the benchmark arena number. `lr_score_signal`
    # is now "rollout", so the detector reads the ring the slot just trained on and
    # the snapback moves it rather than pinning it.
    #
    # 2 is the manifest's own value, and what it buys is the case with no other
    # answer: a slot that has diverged has nothing to climb back to, since
    # `lr_best_decay` only unsticks a detector reading a policy that is still
    # improving.
    "resume_reset_after_rejections": 2,
}

# The roster is uniform, and the archive is deliberately not paired with a
# per-slot exploration split.
#
# The generation half of the problem is already solved run-wide: `rare_sigma_gain`
# is 1.0 in `FREE_CONFIG` above on all ten slots, and has been since v27. What v27
# lacks is *retention* -- the widening surfaces a rare row, the policy tries it,
# and the score-ranked cull drops the result within a few generations. So the
# archive is the only new mechanism here and the control arm is v27 itself: same
# generator, same reward, same ten slots, no reserved berths.
#
# Restricting the widening to a subset of slots was considered and rejected. It
# would make the two halves of the roster differ, but only by *reducing* v27's
# exploration on eight of them, so the comparison it buys is against a run that is
# worse than the one already measured.
#
# One entry per learner slot. `LeagueConfig.validate` requires exactly one
# `primary` per agent type -- it is the name `_primary_training_model_id` resolves
# for the surrogate-propose path -- so slot 0 carries it and the rest do not. The
# choice is arbitrary between identical slots and deliberately not re-derived from
# scores: a primary that moved as the population turned over would make every
# query that resolves it non-reproducible.
#
# Distinct seeds, because that is the only thing separating the slots at the
# start. `register_training_agent` overlays `model_config` onto the manifest and
# then stamps `agent_type`/`name`/`seed`, so identical configs with one seed apart
# is exactly the intended shape: ten independent searches of the same space, and
# what diverges them afterwards is which mature member each one is reseeded from
# and which weights the perturbation redrew.
TRAINING_AGENTS = [
    {
        "type": "dqn_market_free",
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
    # orphan them, and a wedged run has to be dumpable with `kill -USR1 <pid>`
    # instead of only killable. See `sim.workers.install_signal_handlers`.
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
