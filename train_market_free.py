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
SESSION_ID = "market-free-logmoney-v26"
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
# `logmoney_` rather than `learner_` so the ten slots are ten fresh lineages.
# `_self_play_pool`, `lineage_checkpoint`, `all_batches` and the slot LR state are
# all scoped by `model_id`, and every one of them would otherwise hand v25's
# old-objective checkpoints and episodes to a run whose shaped return is a
# different quantity. Renaming is the only thing that severs all four at once.
SLOT_PREFIX = "logmoney"
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
# 4 x 8 = 32 exactly. Deliberately not oversubscribed: the pools are `spawn`, and
# the comment on `_single_threaded_workers` records what unpinned BLAS workers
# cost when they outnumber the cores. If a generation is still gate-bound after
# this, the next lever is `eval_seeds`/`promotion_seeds` rather than more threads.
CONCURRENCY = 4
JOBS_PER_LEARNER = 8

DONOR = "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"

# The release objective, and still the only benchmark. Index 0 is what the ratchet
# reads; adding a second here would only add non-regression diagnostics the gate
# does not consult.
#
# The rule this list used to be justified by -- "nothing this run measures is a
# policy it never plays" -- was the right rule and v20 broke it in the other
# direction: the benchmark stayed correct and `opponents` stopped containing it.
# Both sides agree again now that pilkwang is in the rollout mix below.
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
    "opponents": [
        "kaggriculture_lab.opponents:make_seeded_random?seed=7"
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
    "noisy_sigma_scale_floor": 0.5,
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
    "rare_sigma_gain": 1.0,
    "rare_sigma_cap": 3.0,
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
    # Off, and this is the one interaction worth stating outright. The snapback
    # and the population answer the same question differently, and leaving both on
    # means a stalling slot is repeatedly restored to the incumbent whose score is
    # exactly what the plateau detector reads -- so `lr_best_score` is pinned by
    # the mechanism meant to rescue it and the slot never reaches its cut.
    "resume_reset_after_rejections": 0,
}

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
