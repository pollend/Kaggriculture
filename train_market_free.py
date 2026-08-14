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
# the process is interrupted. Exploration anneals over
# `epsilon_decay_transitions` (2,000,000) at roughly 11,500 transitions a batch
# and 20 batches a generation, so epsilon does not reach its floor until
# generation ~9. Any fixed stop below that answers only that the policy had not
# finished annealing, not whether it learns -- and a useful stopping point is a
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
# `noisy_idle_trading_rate` fires the burst on silence and stops as soon as
# trading resumes.
SESSION_ID = "market-free-idleburst-v12"
ARTIFACTS = ARTIFACTS_ROOT / "dqn_market_free"
LEAGUE_CONFIG = Path("configs/dqn_market_free_league.toml")
MANIFEST = Path("agents/dqn_market_free/agent.toml")
MODEL_NAME = "coldstart_v1"

DONOR = "agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py"

# The release objective, and now the only benchmark. `configurable_crop_surrogate`
# was here as a non-regression diagnostic; it is dropped with the rest of it, so
# that nothing this run measures is a policy it never plays.
BENCHMARKS = (DONOR,)

# Overrides only. `register_training_agent` materializes `agent.toml`'s
# `[models.config]` and then applies this on top, so every key here is a
# deliberate departure from the manifest.
FREE_CONFIG = {
    # Promotion baseline first: index 0 is what the release gate measures, the
    # rest are non-regression diagnostics.
    "benchmarks": list(BENCHMARKS),
    "episode_steps": 720,
    # Only the margin against the opponent is paid, spread over every turn of the
    # episode by the annuity. The money delta and the potential shaping are both
    # out; see the session comment for the round-trip loop that made them
    # unusable, and `reward.REWARD_MODES` for what dropping them costs.
    #
    "reward_mode": "outcome_only",
    # Inert, and kept only to hold this session's identity. `outcome_baseline` was
    # the documented lever for a saturated `tanh`; it never ran at anything but
    # 0.0, and a single scalar could not have centred v4's *bimodal* outcome
    # distribution anyway (+1.0 demonstrations, -1.0 anchor). Removing the two
    # saturated sources was the fix, so the knob is gone from `reward.py` and no
    # code reads this key.
    #
    # It stays because `ensure_session` digests `experiment.training_agents` and
    # `market-free-noisy-v6` already has a stored manifest carrying this key:
    # deleting it changes the digest and v6 stops resuming, which is what
    # stranded `market-free-simple-v1` above. Delete it with the next session id,
    # not before. `test_the_outcome_baseline_key_is_an_inert_digest_relic` pins
    # both halves so this cannot quietly become a live setting again.
    "outcome_baseline": 0.0,
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
    "opponents": ["kaggriculture_lab.opponents:make_seeded_random?seed=7"],
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
    # Exploration is now the only thing putting unfamiliar states in the buffer,
    # with neither demonstrations nor a scripted opponent left to do it.
    #
    # `_epsilon` anneals linearly in `total_transitions`, which is scoped by
    # `model_id` and therefore inherited: v4 stands at 1,380,480, so on the old
    # 2,000,000 schedule this run would resume at 0.063 and reach the 0.01 floor
    # about three generations in. Doubling the horizon puts it back at 0.121 and
    # keeps it above 0.05 for the rest of the run.
    "epsilon_decay_transitions": 4_000_000,
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
    "noisy_flip_rate_target": 0.0,
    # One burst round in five, collecting at four times the ordinary order rate.
    #
    # Five because `rounds_per_generation` is 20 and `min_new_batches` is 20, so a
    # generation is exactly one model update: at every-fifth, each update trains
    # on four consolidation rounds per burst -- 20% of the batches collected, and
    # rather less than 20% of the ring once the FIFO has churned. Every-other
    # would put a burst in half the buffer permanently, which is the noise floor
    # this schedule exists to avoid.
    #
    # 0.10 against the 0.05 setpoint, expressed as a rate rather than a sigma
    # multiplier because a multiplier is the wrong unit: the base it multiplies
    # moves every update as the controller tracks the setpoint, and what a given
    # scale buys moves with the network. `resolve_sigma_scale` reads it off the
    # same `noisy_flip_to_order_curve` the setpoint uses, so the burst re-sizes
    # itself as both drift.
    #
    # Down hard from v9's 0.35, which resolved to sigma 3.159 and was measured a
    # failure on its own terms. Per-episode over 212 seats at that sigma: 49%
    # bankrupt against 8-26% across the ordinary ladder, mean money 4,892 against
    # 11-13k, and *zero* land purchases -- the thing the burst was built to find.
    # The mechanism is the epsilon purchase pump restated: 97 of 186 actions spend
    # money, so at that perturbation the wallet is gone in the first turns and
    # `feasible_actions` offers nothing but NOOP for the remaining ~700.
    #
    # 0.10 log-interpolates to about 1.31 on the curve update 1362 recorded
    # ({1.0: 0.0557, 2.0: 0.1702, 4.0: 0.4428, 8.0: 0.6401}) -- just above the
    # ordinary ladder's top rung of 0.871, and below the 2.129 rung that already
    # measured 29% bankrupt. A burst has to search, not detonate.
    #
    # Kept rather than removed because the hypothesis is not fully dead: what v9
    # showed is that *this* much noise destroys episodes, not that a burst just
    # above the setpoint buys nothing. That is now the experiment.
    # The periodic schedule is retired. Measured twice and never positive: at
    # rate 0.20 it changed nothing over 560 seats, and flat at 0.35 (sigma 3.159)
    # it put 49% of 212 seats bankrupt and found no new behaviour. The mechanism
    # and its tests stay -- the idle trigger below reuses the same path, and a
    # tested negative result is worth keeping.
    "noisy_burst_every": 0,
    # Burst when the policy has gone silent, and burst *hard*. A NOOP policy
    # produces no learning signal at all, so bankruptcy is the better failure --
    # it at least generates transitions with a gradient. The 49%-bankruptcy
    # objection to sigma 3.159 was right on a healthy policy and is the point on
    # an idle one, and it is self-limiting: one batch back above the floor ends
    # it.
    #
    # 0.005 is well under `viability_min_orders_per_decision` (0.02), so this
    # answers genuine silence rather than merely cautious trading.
    "noisy_idle_trading_rate": 0.005,
    "noisy_burst_order_rate": 0.35,
    # The hard stop the rate cannot express, because what a rate costs in sigma
    # depends on a curve that moves with the network. Without this, a steeper
    # curve reproduces the 3.159 event silently.
    "noisy_burst_scale_limit": 4.0,
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
    "epsilon_start": 0.0,
    "epsilon_end": 0.0,
    # A resume restores the optimizer's learning rate from the checkpoint, so
    # this key was previously inert on every generation after the first. It is
    # now re-applied from config each update.
    "learning_rate": 1e-4,
}

TRAINING_AGENTS = [
    {
        "type": "dqn_market_free",
        "name": MODEL_NAME,
        "train": True,
        "primary": True,
        "model_config": FREE_CONFIG,
    }
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
