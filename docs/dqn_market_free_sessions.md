# dqn_market_free session log

One entry per league session. The code no longer carries this history. A comment
that needs a number cites the session id and stops there.

A session id moves for one of two reasons:

- **Strong** -- the reward changed, so a stored return is a different quantity.
  `SLOT_PREFIX` moves too, which severs the buffer, `_self_play_pool`,
  `lineage_checkpoint` and the slot LR state in one move (all four are scoped by
  `model_id`).
- **Weak** -- `ensure_session` digests `FREE_CONFIG` and the league config, so
  any key here breaks a resume whether or not the reward moved. The prefix stays
  and the lineages carry forward.

Standings against the public ladder: [dqn_market_free_ladder_standing.md](dqn_market_free_ladder_standing.md)

## The rule the code comments follow

1. Restates the next line of code -&gt; delete.
2. Documents a key, branch or constant that no longer exists, or is pinned inert
   (`0.0` / `false`) with no plan to move it -&gt; delete. The code guard and its
   test are the record.
3. Narrates a past run -- a session id, a "used to be", a measurement table, a
   post-mortem -&gt; it belongs in this file, not in the code.
4. Would let a future edit reintroduce a bug or silently no-op something if it
   were gone -&gt; keep, at most four lines, present tense.
5. The same fact in two files -&gt; keep it once, closest to where it is enforced
   (test &gt; code &gt; config).
6. Docstrings are one line. A body only when rule 4 applies.
7. Torn between 3 and 4 -&gt; 3. This file is one link away.

## Summary

| id | session | headline |
| --- | --- | --- |
| v1 | `market-free-demo-v1`, `market-free-simple-v1` | Predate the single-action space; unreadable or unresumable |
| v2 | `market-free-simple-v2` | 666 batches, 0 promotions. Spent the stake on `HIRE 13` in four turns |
| v3 | `market-free-selfplay-v3` | Traded constantly, lost anyway. Median 22 coins, 85% insolvent |
| v4 | `market-free-outcome-v4` | First release. Then diverged on saturated `tanh` targets |
| v5 | `market-free-mirror-v5` | Removed both saturated sources. Could not cold-start |
| v6 | `market-free-noisy-v6` | NoisyNet replaces epsilon. Controller throttled all eight actors |
| v7 | `market-free-explore-v7` | Retarget onto flip-to-order. Worked -- money 3,000 -&gt; 11-12k |
| v8 | `market-free-burst-v8` | Exploration bursts. Land bought but never farmed |
| v9 | `market-free-burst-v9` | Flat bursts at rate 0.35. 49% bankrupt, zero land purchases |
| v10 | `market-free-landgate-v10` | Gated `BUY_LAND` on day and reserve |
| v11 | `market-free-coldstart-v11` | First run from nothing. Reset now requires a release |
| v12 | `market-free-idleburst-v12` | Reverted the gate and the periodic burst |
| v13 | -- | Epsilon removed. Money delta and potential shaping removed from the reward |
| v14 | -- | No record in the tree |
| v15 | -- | Periodic idle burst retired. Land bought and not farmed: 355 vs 2,753 |
| v16 | `market-free-hot-v16` | Flat `IDLE_PENALTY` mis-sized. Money 3,542 -&gt; 1,898 |
| v17 | -- | No record in the tree |
| v18 | `market-free-falloff-v18` | The flat stall. 590 batches, no trend in either direction |
| v19 | -- | Last of the undirected-exploration levers |
| v20 | `market-free-rare-v20` | Directed exploration. 61 generations, learned nothing after 11 |
| v21 | -- | Widened `opponents` to three. Reverted after 120 pairings |
| v22 | -- | Carried v21's revert; superseded by the roster |
| v23 | `market-free-population-v23` | Ten learner slots arrive. 77 minutes a generation |
| v24 | -- | `training_concurrency` added; folded into v25 |
| v25 | `market-free-population-v25` | Win rate 0.000 at every measured batch. It learned to act and every action lost |
| v26 | `market-free-logmoney-v26` | Log-growth outcome term. Reached 5 of 10 slots on its own pass condition |
| v27 | `market-free-streak-v27` | Insolvency grace, rollout LR signal, decaying best score |
| v28 | `market-free-archive-v28` | Behaviour archive. Ran five minutes, no graduations |
| v29 | `market-free-rarerate-v29` | Rate novelty + `LAND_POTENTIAL_WEIGHT`, folded deliberately |
| v30 | `market-free-landrate-v30` | Control for the cap raise |
| v31 | `market-free-widecap-v31` | `rare_sigma_cap` 3.0 -&gt; 6.0 |
| v32 | `market-free-melon-v32` | The planting overlay bug. A barrier, not a gradient |
| v33 | `market-free-landbonus-v33` | `LAND_PURCHASE_BONUS` 0.02 -&gt; 0.035 |
| v34 | `market-free-donormix-v34` | Pilkwang joins the rollout base |
| v35 | `market-free-throughput-v35` | Three throughput changes. 12.7 busy cores of 32 |
| v36 | -- | `LOT_CREDIT_WEIGHT`: moved credit from the sale turn to the buy turn |
| v37 | `market-free-lotbasis-v37` | The money delta had never been in the training reward |
| v38 | `market-free-mirroronly-v38` | Donor back out. Live |
| v39 | `market-free-seedbias-v39` | De-biased the seed P&L pair: `_reconcile` every turn, `SEED_WASTE_WEIGHT` 1.0 -> 0.35 |
| v40 | `market-free-animallots-v40` | The ledger covers animals. Live |

---

## v1 -- `market-free-demo-v1`, `market-free-simple-v1`

- Changed: nothing to carry forward. Both predate the action space moving from
  ten order slots to one.
- Measured: `demo-v1`'s checkpoints and batches cannot be read at all -- the
  rollout record format and the network shape both changed. `simple-v1` is
  readable but not resumable; it carried a dead singular `benchmark` key, so its
  digest no longer matches. 60 batches, 2 rejected checkpoints, nothing promoted.

## v2 -- `market-free-simple-v2`

- Measured: 666 batches, 33 updates, 0 promotions. Spent the whole 3,000 stake on
  `HIRE 13` orders inside the first four turns of a 720-turn episode, then held
  NOOP. Finished at 0.7 coins against a never-trade baseline of 3,000. Generation
  0 -- untrained -- was the best that policy ever was.
- Hire cost is fibonacci in `hires_today` and resets daily; one `HIRE 13` is 609
  coins and eighteen of them is 4,026.

## v3 -- `market-free-selfplay-v3`

- Changed: the hire mask priced the whole order, the reward charged for turns
  with no runway, and the opponent mix became mostly this model's own checkpoints.
- Measured: 214 batches, 10 updates, 0 promotions. Median 22 coins from a 3,000
  stake, below the insolvency floor in 85% of episodes. Over 300 episodes: 219
  `BUY_PRODUCT` and 204 `SELL` per episode, wheat and fertilizer 94% of sale
  volume -- the two goods `BUY_PRODUCT` can buy back. Planted 13 wheat, offered
  2,306 for sale.
- Why it paid: a round-trip loop is cash-neutral by construction, but `Phi`
  marked the whole shed to a quote the agent had just moved by buying.

## v4 -- `market-free-outcome-v4`

- Changed: `reward_mode = "outcome_only"` -- drop the money delta and the
  potential shaping, keep the margin against the opponent.
- Measured: first release in the family. `cutoff_batch_seq = 40` promoted at
  5,984 money against a 3,000 baseline, win rate 0.94. Then diverged: the next
  three updates finished at 1.8, 2.5 and 0.8 coins while `mean_loss` climbed
  4.48 -&gt; 5.94 -&gt; 9.63 against a reward bounded in roughly [-3, +2].
- Mechanism: with the money delta gone the stream is
  `tanh(margin / TERMINAL_MARGIN_SCALE)`, which against anything scripted is a
  *constant*, not a function of the action.

    | source | spread | label |
    | --- | --- | --- |
    | pinned demonstrations | surrogate ≥10k vs learner ~0 | +1.0 |
    | anchor rollouts | learner ~0 vs surrogate ~26k | -1.0 |
    | mirror rollouts | margins in the hundreds | ~0.05 |

  Demonstrations were 25% of every minibatch and anchor games 35% of collection,
  so half of every batch carried a flat ±1.0 label on states the network cannot
  tell apart -- both seats open at 3,000 under the same frozen donor policy.
  Irreducible target variance, compounded by the Double-DQN max-bootstrap.
  `selfplay_v3` never showed it because the money delta anchored the regression
  there: its `mean_loss` is 0.002.
- New model name as well as a new session id: `all_batches` is scoped by
  `model_id`, so reusing `selfplay_v3` would have fitted one Q function across two
  reward functions.

## v5 -- `market-free-mirror-v5`

- Changed: removed both saturated sources. No scripted anchor in `opponents`, no
  scripted family in the demonstrations, surrogate out of `BENCHMARKS`. What is
  left is the mirror, whose margins are the only ones `TERMINAL_MARGIN_SCALE` was
  sized for, plus pilkwang as the release objective.
- Cost, recorded rather than argued away: this is the reward-only control, which
  is what `selfplay_v3` ran and which promoted nothing in 12 updates.
  `train_dqn.py` separately records 17 consecutive rejections from a curriculum
  with no static opponent.
- Kept the model name, meaning to inherit v4's checkpoint pool. That inheritance
  did not exist -- `training_batches`, `model_training_checkpoints` and
  `training_sessions` all returned zero rows -- so a pure-mirror run cannot
  cold-start, and v6 raised on its first batch.

## v6 -- `market-free-noisy-v6`

- Changed: NoisyNet replaces epsilon-greedy. Epsilon is zeroed rather than
  removed, because it is part of the policy identity string.
- Why: epsilon's explore branch draws uniformly over feasible actions with NOOP
  removed, so an exploring turn *must* place an order, and 97 of the 186 actions
  spend money against 88 that sell. At 720 decisions and a 0.121 rate that is ~87
  forced random orders an episode, mostly purchases. Epsilon was a purchase pump.
- Measured: 0 promotions in twelve updates, `trading_rate` 0.0 on every gate. The
  cause was its own controller -- the policy collapsed onto NOOP, where every
  action is near-tied and the argmax flip rate is therefore *high* (0.28-0.39
  against a 0.05 setpoint), so `resolve_sigma_scale` divided sigma down to
  [0.33 .. 0.010] across eight actors. Not one played at the learned sigma.
- The v5 -&gt; v6 resume goes through `_load_online_state`: a v5 state dict stores
  `4.weight` where a noisy module wants `4.weight_mu`. Sigma is left at its
  initialization, never invented from the checkpoint, and the Adam moments reset
  because sigma adds tensors to the parameter group.

## v7 -- `market-free-explore-v7`

- Changed: the controller moves onto `noisy_order_rate_target`, which reads the
  NOOP attractor directly and cannot be fooled by cheap near-ties. Cadence to
  `"episode"`.
- Measured: promoted on its second update. `mean_abs_q` 0.04-0.09 -&gt; 0.56-0.60,
  mean money left the 3,000 NOOP fixed point for 11-12k, `BUY_LAND` back from
  extinct at 7-10% of seats. Scored against everything after it, this is the only
  change from v6 to v19 that ever helped.

## v8 -- `market-free-burst-v8`

- Changed: periodic exploration bursts.
- Measured: land-buying seats finished at a 304 median against 13,844 for the
  rest. The ladder only spreads *downward* from the setpoint, so the day-0
  purchase is the only one it can reach.

## v9 -- `market-free-burst-v9`

- Changed: burst rounds flat, rate to 0.35, which resolved to sigma 3.159.
- Measured: across 212 seats at 3.159 -- 49% bankrupt against 8-26% on the
  ordinary ladder, and *zero* land purchases. Every land purchase in the run
  happened at sigma ≤ 0.871, most often at the lowest rungs. Land is chosen by
  the greedy policy, not found by noise. This settled the exploration-magnitude
  question against itself.

## v10 -- `market-free-landgate-v10`

- Changed: `feasible_actions` mirrors the donor's day-and-reserve rule for
  `BUY_LAND`. Burst survives at a tenth of v9's strength.
  `resume_reset_after_rejections` drops to 2.
- Why: `LAND_PRICES` opens at 1,000 against a 3,000 stake, so an ungated mask
  offers the purchase on turn 0 and that is the only version the policy ever saw.
- Also measured here: the controller floored collection sigma at 0.125 by update
  20 -- the `1 / noisy_sigma_scale_limit` clamp, hit exactly -- with `mean_abs_q`
  at 0.046. An untrained network's flip-to-order rate is high from randomness,
  not from sigma.

## v11 -- `market-free-coldstart-v11`

- Changed: the reset now requires a release, and `noisy_sigma_scale_floor` stops
  the controller switching exploration off. First run to start from nothing, and
  the first to move the model name since v4.
- Why: `resume_reset_after_rejections = 2` was set from v10's *released*
  behaviour. On a run with nothing released it is a wipe, not a snapback -- sixty
  batches discarded and the network reinitialized to `quiet_start`, every two
  rejections. No cold start survives that long enough to promote.

## v12 -- `market-free-idleburst-v12`

- Changed: `feasible_actions` back to pure feasibility, periodic burst off, burst
  machinery repointed at forcing bankruptcy rather than idleness.
- Why: scored against measurement, only v7 ever helped. The bursts were neutral
  then harmful, and under the land gate the policy stopped buying land entirely
  rather than buying it later. A NOOP policy produces no learning signal at all,
  so bankruptcy is the better failure.

## v13

- Changed: epsilon removed outright, and the money delta and potential-based
  shaping removed from the reward.
- Why the shaping went: it had been marking the shed to a quote the agent had
  just moved. This is the "v13 objection" the potential terms are measured
  against when they came back at v29 and v37 -- cost basis, never market price.

## v14

- No record in the tree.

## v15

- Changed: the periodic idle burst retired.
- Measured: seats that bought land finished at a mean of 355 coins against 2,753
  for those that did not. Land is bought and then not farmed. This is the finding
  the directed-exploration work at v20 could not answer.

## v16 -- `market-free-hot-v16`

- Measured: mean money fell 3,542 -&gt; 1,898 while `mean_abs_q` rose. The flat
  `IDLE_PENALTY` at 0.0005 was mis-sized -- it made acting worth more than
  waiting on states where waiting was correct.
- Also: `hot_v1` had land affordable on 88% of turns and bought none.

## v17

- No record in the tree.

## v18 -- `market-free-falloff-v18`

- Measured: the flat stall, and the reason the roster exists. `falloff_v1` over
  batches 1100-1688, by 50-batch bucket:

    ```
    money  4997 5179 5745 6227 6872 5983 5074 4861 6527 3906 5559 4498
    ```

  No trend in either direction across 590 batches. `mean_loss` 0.0014-0.0017 and
  `mean_abs_q` 0.19-0.33 over the same span, both flat. Benchmark money against
  pilkwang between 0.9 and 1,208 from a 3,000 stake. Two promotions in the last
  twelve updates, each immediately followed by rejections.
- Every exploration lever from v7 through v19 aimed at the behaviour
  distribution. What had stopped moving was the trajectory.
- Across v15-v18: 7,266 league seats bought land 89 times, and no seat ever
  bought a *second* quadrant.

## v19

- Last of the undirected-exploration levers. Nothing separable recorded.

## v20 -- `market-free-rare-v20`

- Changed: `rare_sigma_gain` -- directed exploration. Widen the head's own sigma
  row for an action the run has barely sampled. `epsilon_out` is already
  per-output, so one entry moves one action and the other 185 stay put.
- Why: over 23,008 states from v18's rollouts against its live checkpoint,
  `BUY_LAND` sits at median rank 33 of 104 feasible actions on a median Q gap of
  0.031 -- inside what the perturbation crosses routinely -- and the greedy policy
  picks it on 0.0000% of states, while 55% of argmax flips go to HIRE and
  BUY_SEED rows. The lever is the correlation, not the magnitude.
- Rejected shape: a bonus added to the Q scores. A deterministic tilt makes a
  rare row win *every* turn its gap is under the coefficient, which is a policy
  change rather than exploration, and is what the epsilon runs bankrupted on.
- Measured: 61 generations, learned nothing after generation 11, for a reason
  unrelated to sigma. `resume_reset_after_rejections` fired at batch 281 and
  `lineage_checkpoint` held the snapback correctly, but every held update writes
  `lineage_reset` and `rejections_since_promotion` stopped counting at the newest
  one. The streak read 0 on all 55 held updates,
  `promotion_ratchet_grace_after_rejections` could never fire, and the run
  resumed `9071d2ad` and trained exactly 2,000 steps 55 times in a row --
  `gradient_steps` pinned at 24,100 for seven and a half hours, 57 consecutive
  rejections.
- Both halves fixed: the streak counts through a held reset, and grace drops the
  *floor* rather than the ratchet. The floor was the unreachable term -- candidate
  benchmark money 0.93-427 against a bar of 1,000, on a reference of 145.8 that
  could not have cleared its own floor either.
- The exploration numbers over those 55 updates say the rest:
  `noisy_argmax_flip_rate` 0.24-0.46, `sigma_ratio_0` pinned near 3.5 at the cap,
  `noisy_flip_to_order_rate` 0.07-0.14 against a 0.05 setpoint. The controller was
  live and above target throughout. There was nothing to turn up.
- What was missing was an opponent worth beating. `opponents` was a single
  seeded-noise agent; the learner solved it at generation 6 -- paired margin flat
  at ~5,900, win rate exactly 1.000 for the following 55 generations -- while the
  gate measured it against pilkwang, which it had never once played.

## v21

- Changed: `opponents` widened to three -- noise, `configurable_crop_surrogate`
  and pilkwang -- on the argument that the gate measures an opponent the run never
  plays.
- Measured: reverted after 15 rounds and 120 pairings.

    | opponent | margin | win rate |
    | --- | --- | --- |
    | seeded noise | +6,241 | 1.000 |
    | mirror | -5,015 | 0.333 |
    | surrogate | -29,376 | 0.000 |
    | pilkwang | -134,558 | 0.000 |

  The surrogate was meant to be a middle rung on the reading that its 32k is
  close to the learner's 9-13k mirror scores. It is not: against the surrogate
  the learner scores 2-3k, so -29,376 through `tanh(margin / 10000)` is -0.995 on
  a slope near 0.01. Both new slots were walls, and together they took 22% of
  collection to deliver a constant.
- There is no mid-strength opponent in this repo. Noise is the only static entry
  carrying terminal gradient at this strength.

## v22

- Carried v21's revert. Superseded by the roster at v23; its lineages were not
  inherited.

## v23 -- `market-free-population-v23`

- Changed: ten concurrent learner slots under one declared model, namespaced
  `dqn_market_free.raw_order_q_v1@<prefix>_00` and so on. Model names move --
  `_self_play_pool`, `lineage_checkpoint` and the slot state are all scoped by
  `model_id`, so ten named slots are ten independent lineages.
- Why: v18's stall is a flat trajectory, and a population fixes a stuck
  trajectory rather than a flat one. Ten rather than more, with a stated exit: if
  all ten decay to `lr_graduate_cut` with benchmark money still under 100, the
  binding constraint is the reward.
- Measured: 77 minutes a generation, load average 9 on 32 cores, max concurrent
  model updates 1. `run_round_hooks` and `_update_session_models` both walked the
  roster one slot at a time in one process.
- The gate is the surprising half: an update averaged 222.6 seconds of which
  `train_seconds` was 8.2. 96% is the arena -- `eval_seeds = 100` plus
  `promotion_seeds = 100`, both seat-swapped, is 400 episodes per learner per
  generation against collection's 320. Evaluation episodes do not land in the
  `episodes` table, so the dashboard reported 3,200 a generation while the box ran
  7,200.

## v24

- Changed: `training_concurrency` added, so the slots stop being walked serially.
- New session id because `ensure_session` digests both the league config and this
  file's `model_config`. The ten model names stay, so all ten lineages carry
  forward.

## v25 -- `market-free-population-v25`

- New session id because `rounds_per_generation` changed, which is in the run
  manifest. The learner slots do *not* restart -- `training_models` is keyed on
  model id rather than session, so LR clocks, lineage checkpoints and the batch
  watermark carry across. Only the generation counter and session rollups reset.
- Measured: 11 generations, 110 batches, 17,600 episodes, zero failures, and a
  policy worse than never trading. Win rate 0.000 against pilkwang at every one of
  fourteen measured batches; benchmark money 98-2,717 from a 3,000 stake against
  pilkwang's ~142,000; `trading_rate` climbing 0.00 -&gt; 0.78 *while* money fell;
  `own_solvent_fraction` 0.003-0.028 throughout. It learned to act and every
  action lost.
- Three causes, landed together at v26 because none is observable without the
  others: the outcome term was a margin while the gate promotes on absolute money,
  and was structurally *zero* in the 65% of rollouts played against the mirror;
  `priority_alpha` was 0.0, so a uniform draw over a 99.7%-insolvent buffer taught
  the only policy consistent with it; and the two opponent-wealth features were
  raw coin scales, so the promotion benchmark was judged fifty times outside the
  training distribution.
- `population.action` was `train` on all 109 updates and no slot reached
  `lr_graduate_cut`, so the population question is deferred rather than answered.

## v26 -- `market-free-logmoney-v26`

- Changed (strong): `TERMINAL_MARGIN_SCALE` replaced by `OUTCOME_LOG_SCALE = 5.0`
  -- the outcome term becomes `tanh(log(final / stake) / OUTCOME_LOG_SCALE)`, read
  off the learner's own purse. `priority_alpha` on. Opponent-wealth features
  rescaled.
- Measured: 37 generations, 366 batches, 58,528 episodes, zero failures, and it
  did not meet its own pass condition (benchmark money over the 3,000 idle
  baseline on a majority of slots, `own_solvent_fraction` over 0.2). Reached 5 of
  10 and one slot. On `episode_players` the median seat finished generation 35 on
  3,112 against generation 0's cold-start 3,000 -- +3.7% on doing nothing, over 33
  generations spent climbing out of the hole dug in generations 1-5.
- Cause one, the objective was 98% penalty. Decomposing `own_mean_return` =
  -0.193 on `logmoney_00` at batch 360 against a mean final money of 3,038: the
  outcome term is worth +0.004 and `INSOLVENCY_PENALTY` -0.153. The term the
  reward exists to express was 45x smaller than a charge bought to price v2's
  permanent bankruptcy -- which was instead pricing the wallet dipping under the
  floor between a purchase and the sale that funds it.
- Cause two, the run was reseeding its best slots. `_slot_score` fed the plateau
  detector the windowed benchmark median, whose mean absolute update-to-update
  change over batches 200-360 is 48.2% against 11.7% for the rollout mean the same
  updates already computed. `_settle_slot` then latched `best_score` onto the
  maximum of it. `logmoney_07` was graduated on a draw it never repeated while its
  rollout money climbed 4,673 -&gt; 5,646, and was reseeded down to 828;
  `logmoney_02` went 3,497 -&gt; 332. The population mean turned over at batch 350,
  3,518 -&gt; 2,930.

## v27 -- `market-free-streak-v27`

- Changed (strong): `INSOLVENCY_GRACE_TURNS` -- the insolvency charge is priced on
  a streak, not a turn. Plus `lr_score_signal = "rollout"` and `lr_best_decay`.
- Measured: `rare_sigma_gain = 1.0` on all ten slots, so rare rows *were* being
  surfaced -- every run spikes to 15-22% land purchase around generation 4. They
  are then selected away. The perturbation diversifies where a slot starts, not
  what it does: a jittered monoculturist re-converges within a few updates into a
  cull that ranks on one number.
- Per generation on `logmoney-v26`, distinct crops planted per seat against the
  seat rate of `BUY_LAND`:

    ```
    gen        0     4     9    12    15    20    30    38
    crops   3.31  3.66  2.32  1.37  1.29  1.85  1.82  1.89
    land     1.4% 18.5%  3.1%  0.3%  0.3%  0.4%  1.6%  1.7%
    money  8,726   454 1,669 2,460 3,365 3,054 4,061 3,916
    ```

  The roster converges onto a wheat monoculture by generation 15 and money
  *rises* while it does, so neither the promotion gate nor `_population_score`'s
  activity discount can object. A monoculture genuinely is the best thing the
  ranking has found. v25 and v27 reproduce the same curve on the same schedule.

## v28 -- `market-free-archive-v28`

- Changed: `mature_reserved_capacity` holds 6 of 30 berths for the best-scoring
  member of each behaviour cell rather than for score;
  `reseed_source_explorer_share` reads them on one restart in four; and
  `mature_reserved_min_money` floors what may occupy a cell -- without which the
  archive preserves day-0 land bankrupts finishing on a median of 7 coins.
- Why: ranking is the wrong instrument for keeping an alternative alive long
  enough to beat the incumbent. Retention is the whole change; generation was
  already handled by v27's widening.
- `population_members` is keyed on `declared_model_id` and survives the rename,
  so the archive starts populated from v26 and v27's thirty live members.
- Measured: ran five minutes and produced no graduations. Nothing separable was
  lost by restarting on top of it at v29.

## v29 -- `market-free-rarerate-v29`

- Changed (strong): two things, folded deliberately.
  `rare_sigma_novelty = "rate"`, and `LAND_POTENTIAL_WEIGHT`. They act on
  different mechanisms and read apart cleanly -- the rate arm in the sigma
  metrics, the land potential in `BUY_LAND` seat share and seed conversion.
- Why the rate arm: at gain 1.0 the count statistic gave `BUY_LAND` 1.47x against
  a 3.0 cap and `BUY_SEED MELON` 1.085x, with one row of 186 ever reaching the
  cap. `log1p(1,150,400)` is 13.96, so the numerator is effectively constant and
  the spread compresses into `1 + n_a`. It also asks the wrong question: MELON was
  feasible on 883,000 turns and chosen on 1,917 -- declined, not rare -- and a raw
  count cannot tell those apart.
- Why the land potential, measured on v27: the agent asks for ~311 seeds an
  episode and plants ~35, and land is what makes the rest usable. It tried exactly
  that -- land buying ran 3.6 -&gt; 14 -&gt; 21.5 -&gt; 22.9 -&gt; 12.3 -&gt; 0 -&gt; 1 percent of
  seats across generations 2-8, peaking at the money trough -- and over
  generations 2-6 the buyers planted 38.0 against 36.9, lost 22.6 plants against
  15.1, and finished on a median of 2 coins against 400. The return arrives ten
  days later through a terminal outcome discounted `0.999^720 = 0.49` in 24-turn
  hops; the 1,000-coin cost lands immediately.
- Shared diagnosis: the policy finds a rare capital action, tries it, executes it
  badly, and abandons it.

## v30 -- `market-free-landrate-v30`

- Control for the cap raise: same reward, same rate arm, same gain, cap 3.0.
- Measured: on 1,133,396 folded selections, two rows of 186 sat at the cap --
  `BUY_LAND` and `BUY_ANIMAL SHEEP 4`. Every other row, including all five crops,
  was strictly inside it. Land buyers finished on a median of 48 coins against
  3,000 for non-buyers.

## v31 -- `market-free-widecap-v31`

- Changed (weak): `rare_sigma_cap` 3.0 -&gt; 6.0 and nothing else.
- Why: at 3.0 the clamp bound on exactly the rows the mechanism exists for, so it
  was a ceiling on the rarest actions rather than a safety bound. The safety it
  was not providing comes from the floor at 1.0 and the plain argmax.
- Deliberately not paired with a gain rise -- amplifying the rate before
  `LAND_POTENTIAL_WEIGHT` demonstrates land can pay would buy proportionally more
  bankruptcies. `BUY_LAND`'s unclamped multiplier at gain 0.5 is 3.53, so this is
  a 3.00 -&gt; 3.53 move: roughly 6% -&gt; 7-8% of episodes buying a quadrant.

## v32 -- `market-free-melon-v32`

- Changed: a bug fix, not a knob. `planting.market_influenced_roles` overlaid held
  seed onto the donor's vacant crop roles and, when a crop had no planned tile of
  its own, fell through to `unassigned[0]`. `MELON` is last in `donor.CROPS` and
  `CROP_MIX` plans no TOMATO or STRAWBERRY tile anywhere, so every tomato and
  strawberry seed took a tile planned for melon -- the one crop worth 142 per
  tile-day against 24-40 for the rest.
- Measured on v31 generations 5+, 16,000 seats, by melon share of seed spend:

    | melon share | seats | mean money | median |
    | --- | --- | --- | --- |
    | 80%+ | 270 | 4,371 | 2,353 |
    | 40-80% | 1,158 | 5,642 | 4,109 |
    | 5-40% | 6,445 | 2,870 | 64 |
    | under 5% | 7,723 | 3,321 | 2,568 |

  A barrier, not a gradient: both pure strategies are viable and the path between
  them collapses to a median of 64, because a mixed purchase pays for melon tiles
  and then plants carrots on them. 14,168 of 16,000 seats sat in the two bad
  buckets. This is also why widening sigma made things worse -- it sampled the
  valley more often, which is how v31 reached the same ceiling ~4x faster than v26
  rather than passing it.
- The deficit was invisible: `market_state` feeds `donor._seed_needs` into the
  state but is handed the roles map *after* the overlay, so a plan wanting 10
  melon with 2 held reported a deficit of 2 rather than 8. Fixing the overlay
  repairs the feature, so `MARKET_STATE_DIM` is unchanged at 238.
- Land moved too: the mask now gates `BUY_LAND` on readiness rather than
  affordability alone, because 49.4% of the 7,342 accepted purchases on record
  planted zero of their 25 new tiles, and day-0 purchases finish on a mean of 190.
  `LAND_PURCHASE_BONUS` pays 0.02 on the accepted purchase.

## v33 -- `market-free-landbonus-v33`

- Changed (strong): `LAND_PURCHASE_BONUS` 0.02 -&gt; 0.035. Nothing `ensure_session`
  digests moved, so v32 would have resumed cleanly and silently -- which is the
  reason to move rather than to stay. Reward constants are not in the digest, so a
  resume would leave one session's batch history holding two reward regimes with
  no column saying where the boundary is.
- Measured on v32: the 0.02 bonus stopped land being unlearned. v31 held a
  0.13-0.16 buy rate for 70 batches after the term went in, where v25, v26 and v27
  all spiked to 0.24-0.34 and decayed back to 0.005-0.01. Bought land is also
  worked -- 14.2 of 25 tiles planted per purchase on v32 against 8.7 on v31.
- What v32 did not show is a rate on a run that *started* with the term: it bought
  a quadrant in 3.2% of episodes against v31's 14.3%, and 257 of its 295 purchases
  were a seat's first and only one. The gate is not binding -- purchases land on
  day 10.3 holding 1,965 coins, roughly twice `LAND_PRICES[0]` -- so
  `_land_is_survivable` is clearing states the policy then declines on value.

## v34 -- `market-free-donormix-v34`

- Changed (weak): pilkwang joins the rollout base, and `self_play_ratio` drops to
  0.5 to match a two-entry base. Noise stays -- it is still the only opponent
  whose own money lands in a cold-started learner's range, and a reseeded slot
  restarts from there.
- Why the v21 result no longer applies: `TERMINAL_MARGIN_SCALE` is gone.
  `OPPONENT_WEIGHT` is 0.0 and the outcome term reads the learner's own purse, so
  pilkwang finishing 134,558 ahead is not a constant -1.0 -- it is not in the
  reward at all. What the learner scores in the market pilkwang leaves behind is,
  at full slope.
- Measured on v33: greedy benchmark money sat in the 2,500-3,900 band across all
  15 measured cutoffs -- the same wall as v26 through v32 -- while head-to-head
  money climbed to 5,891 against an incumbent at 7,676. The mirror was improving
  against itself and nothing was improving against the objective. One slot's gate
  record at cutoff 150, one checkpoint, two markets:

    | market | behaviour | money |
    | --- | --- | --- |
    | vs its own incumbent | 0.56 quadrants/episode | 3,215 |
    | vs pilkwang | 0 quadrants, 26,455 of ~31k seeds WHEAT | 1,849 |

  The land behaviour `LAND_PURCHASE_BONUS` bought exists only in markets pilkwang
  is not in.

## v35 -- `market-free-throughput-v35`

- Changed (weak): three throughput changes on top of v34's mix. A separate session
  only because v34 had already collected a generation and `CONCURRENCY` is in
  `LeagueConfig`, which `ensure_session` digests.
- Measured over 40 seconds mid-generation on v33: 37.6% user, 35.2% iowait, 24.9%
  idle -- about **12.7 busy cores of 32**, not 32. `train_seconds` was 18-32 s of a
  ~1,900 s generation.
  - `gradient_steps_per_batch` 100 -&gt; 400. The replay ratio was 0.32 against an
    ordinary DQN range of 1-8, and raising it costs ~5% wall clock because
    learning is 1.6% of it. The largest single change by some distance.
  - `CONCURRENCY` 4 -&gt; 5. `training_concurrency` is a `ThreadPoolExecutor` width
    over *models*, consumed by `list(pool.map(...))`, so it is a hard barrier. Ten
    models over four threads is a 4/4/2 wave with a half-empty tail; five is 2 x 5.
    The old "4 x 8 = 32 exactly" budget sized against a saturation that never
    happened, and `sim/workers.py` already pins `OMP_NUM_THREADS` and
    `MKL_NUM_THREADS` to 1, so 5 x 8 = 40 single-threaded processes on 32 cores is
    a scheduler queue, not BLAS thrash. Verified at 150 threads, not ~1,300.
  - `promotion_seeds` 50 -&gt; 30. Not `eval_seeds`, deliberately -- that one buys
    arena precision the gate reads. `promotion_seeds` buys a paired significance
    test `promote` does not consult.
- If generation time does not fall below ~1,900 s, `CONCURRENCY` is the one to
  revert first, and v34's single generation is the like-for-like baseline.

## v36

- Changed (strong): `LOT_CREDIT_WEIGHT` -- move a lot's realised proceeds from the
  sale turn back to the buy turn.
- Why it was wrong: it preserved the episode *total*, which is what made it look
  safe, but a preserved total is not a preserved ordering. At the sale state the
  debit was entirely in the future, so return-to-go fell by the full amount; at
  the buy state the credit was immediate and the debit 240 turns away, so
  return-to-go rose by only 21% of it. It deterred selling roughly five times
  harder than it encouraged buying, and a 24-unit melon dump charged -0.03 on a
  single turn against an outcome annuity capped at 0.001 per step. It moved the
  credit-assignment problem from buying to selling rather than removing it.

## v37 -- `market-free-lotbasis-v37`

- Discovery, and it is worth more than the constant: `episode_rewards` builds
  `shaped` from `np.zeros_like(raw)` and never adds `raw` to it -- `collect.py`
  keeps `raw` as `raw_reward` for diagnostics and `replay.py` trains on `reward`
  -- so **the money delta has never been in the training reward**. A BUY_SEED and
  a NOOP score the same on the turn they happen, and the only term that ever tells
  them apart is the terminal outcome.
- The distance to that terminal outcome is longer than v36 believed.
  `CROPS[crop]["first_yield_day"]` at `turnsPerDay = 24` is 48 turns for WHEAT and
  CARROT, 192 for TOMATO, 240 for MELON and STRAWBERRY -- and `planting.py`
  optimises for melon. Against `n_step = 24` that is ten bootstraps, not the two
  to four previously claimed, with `0.999 ** 240 = 0.786` on top.
- Changed (strong): `LOT_POTENTIAL_WEIGHT` replaces `LOT_CREDIT_WEIGHT` outright.
  A potential over held cost basis -- seeds at their price, standing plants at the
  price of the seed that became them, harvested units at that price over
  `max_yield`. `gamma * PHI(s') - PHI(s)` pays on the purchase turn and telescopes,
  so the optimal policy provably cannot move.
- `lots.py` survives as measurement, not as reward input. Two ledger bugs went
  with it: a harvest landing on a HIRE, BUY_LAND or BUY_ANIMAL turn opened a
  phantom lot dated to the harvest and priced at the outlay (field actions run
  before `_process_market`, and `main.py` emits HARVEST every turn), and
  `_YIELD_CAP` left room open for units that rotted or overflowed the shed. Both
  fed `seed_spend` and `unrecovered_seed_share`.
- The same discovery invalidates `LAND_PURCHASE_BONUS`'s stated justification: it
  priced itself as repaying a fraction of a -0.05 purchase-turn cost that does not
  exist. Held anyway -- v33 took the land rate 0.043 -&gt; 0.35 with tiles worked
  *rising* 12.3 -&gt; 16.3, and a constant that is working is not cut on the strength
  of a corrected comment.

## v38 -- `market-free-mirroronly-v38`

- Changed (weak): pilkwang back out of `opponents`, `self_play_ratio` back to
  0.65, so the rollout mix is `[noise, self, self]` again -- what every session up
  to v33 played. The reward is untouched, so `SLOT_PREFIX` stays `lotbasis` and
  v37's lineages continue into a different opponent distribution.
- The evidence this revert runs against, and it is the strongest result the family
  has. Greedy money against pilkwang, `landbonus` prefix, donor entering the base
  at batch 161:

    | cutoff | greedy money | promoted | own_return |
    | --- | --- | --- | --- |
    | 150 | 3,164 | 3/10 | -0.611 |
    | 170 | 3,659 | 1/8 | -0.497 |
    | 190 | 6,556 | 3/8 | -0.555 |
    | 210 | 8,044 | 3/8 | -0.417 |
    | 220 | 9,096 | 4/8 | -0.353 |
    | 230 | 7,951 | 3/7 | -0.318 |

  v26 through v33 sat in a 2,500-3,900 band across every measured cutoff. That
  band broke within thirty batches of the donor joining the mix. Rollout money over
  the same span went 6,515 -&gt; 10,912 and solvent episodes 4.1 -&gt; 7.1.
- What it cannot do is attribute: v35 folded three throughput changes in at batch
  ~170 and `LOT_POTENTIAL_WEIGHT` landed later still. Taking the donor out is how
  to find out which of them carried it.
- Cost of the revert: a quarter of collection returns to the mirror, and the
  policy stops playing the market it is graded in -- the failure `train_dqn.py`
  records at 17 consecutive rejections.
- **The open question.** If greedy money falls back into the 2,500-3,900 band on a
  mirror-only mix, the donor was carrying it and this table is the argument for
  putting it back.

## v39 -- `market-free-seedbias-v39`

- Changed (strong): the seed P&L pair had been measured a net tax -- over 20 stored
  episodes a mean of -0.086 per episode against an `outcome` whose whole range on
  that sample is [-0.92, +0.42], on seasons whose seeds had in aggregate returned
  2.4x. The cause is that `lots.match` is FIFO over fungible units: `surplus()`
  clips a winning lot at its own cost while `unrecovered()` charges every lot behind
  it in full, so the sign of a season's total follows an accounting convention.
- Two answers shipped together. `_reconcile` now caps a crop's outstanding room
  every turn against `stock + max_yield * (seeds + standing)` rather than only once
  the crop is exhausted -- measured, 60.7% of seed-lot yield capacity was never
  absorbed, so the season's first lots held room for units nobody grew and FIFO
  handed them every later sale. Matched share 0.17 -> 0.47. And
  `SEED_WASTE_WEIGHT` moved 1.0 -> 0.35, where the per-episode mean is neutral, the
  tail sits inside the outcome's range instead of past it, and `corr(outcome)` is at
  its maximum. `SEED_GAIN_WEIGHT` held at 1.0: the credit leg is the one being
  clipped, so the charge is the one that moves.
- Ran 2 generations, 3,200 episodes. Superseded before the pair could be re-measured
  on a full run, so whether the de-biasing held in training is still open -- the
  `seed_gain_credit` / `seed_waste_cost` pair per batch is where it would show.

## v40 -- `market-free-animallots-v40`

- Changed (strong): `lots.py` matches animals. Through v39 the ledger was seed-only
  and a bought animal was invisible to every shaping term -- `shaped` carries no
  money delta, no lot was opened for a cow, and `_UNIT_BASIS` is zero for MILK and
  WOOL -- so a 400-coin cow was free on the turn it was bought and the milk it sold
  for the rest of the season paid nothing back. At `donor.TARGET_HERD = 15` that is
  up to 6,500 coins the stream had no opinion about, in the two highest-priced
  non-melon products. Milk and wool sales landed in `unattributed_proceeds` and were
  dropped.
- The structural difference from a seed, and why this is not a rename. A seed is
  consumed and a plant has a `max_yield`, so a seed lot's capacity is a constant. An
  animal is durable with no lifetime cap -- `max_held` is what accumulates on the
  tile awaiting harvest -- so what closes an animal lot is the season ending.
  `remaining_events` is that count, and `LAST_YIELD_DAY` is 29 rather than 30
  because DONE fires at step 718 and the last end-of-day boundary at or below it is
  step 695. From placement on day 0 a cow yields 11 times and a sheep 8. The buy
  deadlines are day 21 and day 23, and `order_space` gates BUY_ANIMAL on shed room
  and cash only, so buying a dead-weight animal was a reachable and unpriced action.
- The ceiling counts unplaced stock, which is load-bearing rather than a nicety.
  BUY_ANIMAL lands in the shed and PICKUP then PLACE are later turns, and
  `_reconcile` only ever decrements, so a placed-only ceiling would trim every
  animal lot to zero on its own purchase turn and never restore it -- the whole
  feature would be a pure tax on owning a herd.
  `test_an_animal_lot_is_not_trimmed_on_the_turn_it_was_bought` is the guard.
- `ANIMAL_WASTE_WEIGHT = 0.25`, `ANIMAL_GAIN_WEIGHT = 0.70`. A port of the seed
  pair's prior rather than a new tuning: break-even probability of waste against
  NOOP is 0.25 / 0.95 = 0.263 where the seed pair's is 0.35 / 1.35 = 0.259. In
  coins a cow's charge is 0.0050, one melon seed's, and eight times below the 0.04
  the seed docstring identifies as a NOOP attractor. **These have not been through
  the 20-episode mean/worst/`corr(outcome)` sweep the seed pair was chosen by.**
- No animal potential, deliberately. A `PHI(owned, day)` would be action-independent
  on every turn except a filled BUY_ANIMAL -- `day` moves on the clock and FEED is
  emitted by the frozen donor planner -- so it would be a per-state constant
  contributing nothing to `argmax_a Q(s, a)`, and on the buy turn still an invariant
  term that cancels out of every difference. The conserving alternative, putting
  `cost / remaining_events` on each MILK unit the way a harvested crop carries
  `seed / max_yield`, is the exact analogue and is unusable: it needs the count of
  *placed* animals, and an animal in the shed is not a feature, so `PHI` would be a
  function of something the network cannot see. The ledger may use the exact
  per-tile trace because it is offline accounting; a potential may use only what the
  state vector carries.
- Known and left: the wheat a herd eats is charged to the wheat's own lot. FEED
  consumes 1 WHEAT and an animal escapes at two consecutive unfed days, so a season
  is 11+ feeds per animal, roughly 275 coins against a 400-coin cow. Wheat leaving
  stock with money flat matches no rule in `match`, so `_reconcile` writes it off
  and `SEED_WASTE_WEIGHT` bills the wheat buy turn -- right episode, wrong lot.
  Folding expected feed into the animal's `unit_cost` would be the invented
  allocation the "No equivalent for HIRE" note bans. Watch `unrecovered_seed_cost`
  against herd size.
- What to watch: `animal_gain_credit` against `animal_waste_cost` per batch.
  `animal_lots` at zero while the seat is buying animals means the trace never
  reached the ledger. The charge climbing with the credit flat means the herd is
  being bought and not placed, or bought past the day it could still yield. A MILK
  or WOOL entry in `unattributed_by_item` now means the ceiling is too tight, which
  is what the care bonus can cause -- a production event pops the whole accumulated
  bonus, so an event can deliver `1 + interval` units and only the bonus in hand is
  counted.

## v41 -- `market-free-mirrorpure-v41`

- Changed (weak): `opponents = []`. No static base at all, so `compose_opponents`
  returns the mirror pool unmixed and every rollout episode is this family against
  its own checkpoints. The reward is untouched, so `SLOT_PREFIX` stays `animallots`
  and v40's ten lineages carry forward on their batch-30 watermark.
- What came out. `seeded_random` was not a weak competitor, it was a free-money
  episode: its `market` list can only ever hold `BUY_SEED`, and `SELL` is the only
  op that credits money (`kaggriculture.py:644`), so its wallet is monotonically
  non-increasing, walks the 10-coin seed-price grid down from the 3,000 stake, and
  terminates on exactly 0. Across v26-v40 it has never once recorded a non-zero
  `final_money` -- ~15,000 seats, `max(final_money) = 0`.
- What it was costing, measured on `market-free-animallots-v40:g0002`. The learner
  plays the same game either way, so the bias is in the return and not the
  behaviour:

    | | vs seeded_random | vs a real policy |
    | --- | --- | --- |
    | episodes | 520 | 1,080 |
    | total actions | 2,361 | 2,353 |
    | productive % | 18.4 | 19.9 |
    | idle % | 47.7 | 43.3 |
    | mean money | 3,312 | 2,432 |
    | median money | 932 | 520 |
    | win rate | 0.944 | 0.422 |

  One base entry at `self_play_ratio = 0.65` solves to `[noise, self, self]`, so
  that 36% premium was priced into a third of every batch -- 520 of g0002's 1,600
  training samples. It is not reachable by lowering the ratio: `compose_opponents`
  clamps to 0.9, which against a one-entry base still floors the noise share at 10%.
  Removing the entry is the only way out.
- Why the session moves and the prefix does not. `self_play_after` is read as
  `if self_play_after and batch_index >= self_play_after` and cannot be 0 -- 0 is
  its disable value -- so batch 0 never consults the pool. With an empty base that
  is the "free-market collection needs at least one opponent" raise. Inheriting
  `animallots` past batch 30 is what makes the empty base legal;
  `test_the_mirror_is_pure_and_its_one_unreachable_batch_is_inherited_past` pins it.
- Carried over and expected to age out: `buffer_episodes = 1200` is roughly three
  generations, so the ring still holds noise-opponent transitions for about that
  long. Early v41 batches are a mix whatever the config says.
- The risk this takes on, stated so the revert has a trigger. v38 already recorded
  it -- "the policy stops playing the market it is graded in -- the failure
  `train_dqn.py` records at 17 consecutive rejections" -- and v41 goes one step
  past v38, which still had the noise entry. The gate's exits are
  `benchmark_grace_min_money = 0.0` and the manifest's
  `promotion_ratchet_grace_after_rejections`. **If promotions stall, put the
  pilkwang donor in `opponents`, not the noise agent.** The donor is the mix entry
  v38's own table argues for; the noise agent is what this session removed.
- Second risk, from reseeding. A freshly reseeded slot restarts near zero and the
  mature pool it now faces is 24 live members spanning 4,140-19,347. The noise
  entry used to be the one opponent in a cold-started learner's range. Watch
  whether reseeded slots recover or sit at the floor the way v40's slots 05, 06, 08
  and 09 did (`lr_score` 189, 316, 378, 198 at cutoff 20, flat from cutoff 10).
- What to watch: greedy money against pilkwang across cutoffs, against v38's
  2,500-3,900 band and its 3,164 -> 9,096 donor-in run. This session is the direct
  test of v38's open question -- if a pure mirror falls back into that band, the
  donor was carrying it.

## v42 -- `market-free-landhandle-v42`

- Changed (weak): `BUY_LAND` leaves the argmax. `order_space.LAND_HANDLE_SET_POINT`
  decides it instead, on `handle(s) = Q(s, BUY_LAND) - Q(s, NOOP)`, set point 0.0.
  The reward, the network shape and every checkpoint are untouched, so `SLOT_PREFIX`
  stays `animallots` and v41's lineages continue. `None` restores the single argmax
  and is the revert.
- Why a ranking is the wrong instrument, and it is structural rather than a tuning
  failure:
  - **A capital decision priced on a marginal scale.** Every other row is a turn's
    worth of coins; this one commits 1,000-4,000 against a payback 10-25 days out.
    `roadmap.md:284` measures the result -- `BUY_LAND` at *median rank 33 of 104
    feasible actions*. It does not need to be rank 1, it needs to be "yes" at most
    twice in 720 turns, and an argmax cannot express that.
  - **The contest is an artefact.** `BUY_LAND` is a market op resolved atomically
    against `maxMarketOrdersPerTurn = 10`, so the simulator never makes it compete
    with a farmer op. Ranking it against SELL and BUY_SEED is a consequence of this
    module emitting one order per decision, not of the game.
- Measured against `NOOP` rather than as a bare score. A bare threshold has no fixed
  scale to sit on: Q drifts by orders of magnitude across a run, so a constant that
  binds at batch 10 is inert by batch 200. `NOOP` is feasible every turn and is the
  value of doing nothing, so the difference is "what expansion is worth over idling"
  and carries its own units.
- The gate is unchanged and is still the safety net. The handle can only fire from a
  state `_land_is_survivable` has cleared -- mid-season, twelve days left, no
  outstanding seed deficit, cash covering price plus a day's hiring. That is close to
  the donor's own `LAND_OPEN_DAYS = (5, 9)`, and the donor finishes above 100,000.
- What this does **not** do, stated so it is not re-litigated later. It does not
  amortize the outlay: land still opens no lot, `lots.py` names `BUY_LAND` only as a
  turn to exclude, and the 1,000-4,000 coins still reach the policy solely through
  the terminal outcome. `LAND_POTENTIAL_WEIGHT` and `LAND_PURCHASE_BONUS` are
  untouched. This session moves *when land is selected*, nothing about how it is
  valued.
- The known interaction, and the first thing to check. `choose_order` scores through
  whatever noise NoisyNet has drawn, and a *comparison* absorbs a perturbation that a
  *threshold crossing* does not -- land can now fire on a noisy draw where before it
  had to out-rank 103 rows. **`buy_land_rate` against
  `land_tiles_planted_per_purchase` is the check. A rate that jumps while tiles stay
  flat is the handle firing on noise; raise the set point, do not re-tighten the
  gate.**
- Baselines to beat, from `training_batches` on the `animallots` prefix. v40 held
  `buy_land_rate` near 0.09 at 1.5-6 tiles per purchase over batches 1-20, rising to
  0.075 at 6.6 tiles by batch 30; the first two pure-mirror batches of v41 read 0.175
  and 0.206 at 11.1 and 11.0 tiles. The long-run references are v26 `logmoney`, which
  peaked at 0.218 by batch 40 and decayed to 0.008 by batch 120 and never recovered,
  and v33 `landbonus`, which climbed monotonically to 0.37 at 9-10 tiles. Against 25
  tiles per quadrant, even v33's best works under half of what it buys.

## v43 -- `market-free-landheuristic-v43`

- Changed (strong): `BUY_LAND` comes off the network entirely.
  `order_space.land_score` decides it and `choose_order` never reads the row.
  `LAND_PURCHASE_BONUS` 0.035 -&gt; 0.0 with it, so this is a reward move and
  `SLOT_PREFIX` goes `animallots` -&gt; `landheur`; the ten v40-v42 lineages end here.
- Why the v42 handle went. It asked `Q(BUY_LAND) - Q(NOOP)` and the row it read is
  dead: `buy_land_rate` ran 0.000-0.050 across all ten slots over batches 351-390,
  one purchase per twenty to two hundred episodes, against v33's 0.37 and v41's
  0.206. A gated row is starved of gradient, so its Q is arbitrary in exactly the
  states where the gate opens, and no set point placed on an arbitrary number is a
  decision rule. Every lever tried before it was an exploration lever and the
  measurements refused all of them -- 212 seats at sigma 3.159 bought zero
  quadrants, and every purchase on record happened at sigma &lt;= 0.871.
- The score, three factors multiplied so any one at zero is a veto:

      horizon = clip((days_left - 12) / (30 - 12), 0, 1)
      worked  = plants / max(1, plants + plantable)
      runway  = clip((money - price - reserve) / reserve, 0, 1)

  plus the old readiness vetoes, unchanged: day &gt;= 1, twelve days left, at most
  `MAX_EXTRA_LAND`, and no outstanding seed deficit.
- **Why it cannot reproduce the day-0 ruin, which is the point.** `horizon` alone
  peaks on day 0 -- the timing that put 649 of 707 purchases in the 0-3,000 money
  bucket -- and `worked` forbids it, because nothing is planted on turn 0 whatever
  the wallet says. Walked over a plant-up-then-earn trajectory the first purchase
  lands on **day 8**, inside the donor's own `LAND_OPEN_DAYS = (5, 9)`, with
  nothing hardcoding a day.
- The reserve is the tile term. Hands are wiped nightly and re-hired on a
  fibonacci ladder, so a day's wage is the whole prefix: 143 coins at 10 hands,
  609 at 13. A quadrant doubles the area, which takes `_target_hands` from its
  floor of 10 to the cap of 13 and the daily bill 143 -&gt; 609, a 4.3x jump. The
  second quadrant saturates against `MAX_HANDS` and costs no more in wages -- it
  costs 1,000 more in price instead.
- `LAND_RESERVE_DAYS = 1.0`, the risk-taking end of that trade and a deliberate
  choice rather than a default. One day is the smallest reserve that is still a
  reserve -- a farm that cannot cover tomorrow's crew cannot work what it just
  bought -- and it leaves v33's 1,965-coin wallet still able to buy. Traced over
  real episodes the first purchase lands on **day 4-5 at 84% of tiles worked**,
  where 2.0 put it at day 5-8. The failure it takes on is v33's: purchases rising
  while `land_tiles_planted_per_purchase` stays flat, and this constant is the
  answer to that. `land_gate_open_rate` (new, per batch) near zero is the opposite
  failure and the set point is the answer to that one.
- **The two-quadrant cap is gone.** `donor.MAX_EXTRA_LAND = 2` is read out of a
  frozen file -- `donor.py` is byte-identical to the scorer's copy and
  license-pinned -- and its stated reason, "nothing on record has come close to
  working two", is false on this database: `land_accepted / land_buying_episodes`
  reaches 1.72-1.81 on `animallots_04`, `animallots_08`, `widecap_06` and
  `lotbasis_08`, so most buyers on those slots already take both and the cap binds.
  The leaderboard settles the direction -- its best performers own all four
  quadrants. `order_space.LAND_MAX_EXTRA = len(LAND_PRICES)` is the agent's own
  number now; there is no fourth purchase, four quadrants is the whole board.
- Raising it is safe because the score is a score rather than a right. The third
  quadrant costs 4,000 and adds 25 tiles to a farm whose crew is already at
  `MAX_HANDS = 13`, so it buys area with no labour to work it -- and that shows up
  in `land_utilisation`, which falls the moment the second quadrant goes unworked
  and refuses the third on its own. Traced over three episodes the first clears on
  day 5 at 84% worked and the second on day 15 at 94%; the third never cleared,
  declined on runway at 4,000 + reserve rather than by any cap. Watch
  `land_accepted / land_buying_episodes` against
  `land_tiles_planted_per_purchase` -- the ratio rising while tiles per purchase
  falls is the third being bought and abandoned, and the answer is to put
  `LAND_MAX_EXTRA` back to 2, not to touch the set point.
- **Utilisation is read off the tiles, not off the role plan, and a traced episode
  is why.** `planting.market_influenced_roles` rewrites every vacant crop role from
  the seed actually held, so a farm holding seed for three tiles has a plan wanting
  three tiles -- and a role-based ratio called twenty bare tiles "nearly full". The
  first trace opened the gate on day 2 holding 3,000 untouched coins, which is
  exactly the purchase the score exists to refuse. The seed veto still reads the
  effective plan; the two are deliberately measured off different things.
- Two cleanups that follow from the row leaving the head. `rare.sigma_scale_vector`
  pins `BUY_LAND` to 1.0 -- it is the rarest row there is and sat at the
  `rare_sigma_cap` on every v30/v31 measurement, so widening it now spends the
  whole budget on a column that cannot change a decision. And `clone.CloneStats`
  counts a masked `BUY_LAND` as `orders_masked_land` rather than
  `orders_masked_infeasible`: the donor buys quadrants this heuristic declines,
  which is the two policies disagreeing rather than the mask being wrong.
- **The batch-0 raise is gone**, and it is what made the prefix move legal. With
  `opponents = []` and `self_play_after` read as `batch_index &gt;= 1`, batch 0 of a
  fresh `model_id` had no mix at all and collection raised. It seeds the quiet
  network there now -- one batch against an opponent that emits nothing, the same
  baseline the promotion gate already treats as a real incumbent. A rule that
  could not be followed was protecting nothing.
- One wrinkle in the batch history, recorded so it is not mistaken for signal.
  **Batches 1-6 were collected under the pre-trace land logic** -- reserve 2.0 and
  role-based utilisation -- and batches 7 onward under the shipped one. Against
  `buffer_episodes = 1200` that is 96 of 1,200 episodes a slot and ages out inside
  a generation, but the boundary is visible in the columns and is not a policy
  improvement:

    | batch | 1-6 | 7 | 8 | 9 | 10 |
    | --- | --- | --- | --- | --- | --- |
    | `buy_land_rate` | 0.038-0.063 | 0.113 | 0.188 | 0.169 | 0.231 |
    | tiles per purchase | 2.1-8.5 | 7.7 | 8.6 | 12.0 | 10.2 |
    | `land_day` | 2.3-7.8 | 6.2 | 5.9 | 8.5 | 9.5 |

  The rate roughly quadrupling at the cut is `LAND_RESERVE_DAYS` going 2.0 -&gt; 1.0.
  What is not explained by the cut, and is the first real read on the change: tiles
  worked per purchase settling at 10-12 of 25 against v40's 1.5-6.6 and v33's best
  of 9-10, with `land_day` moving *later* rather than toward 0.
- The pass condition, and it is not "money goes up": **land-buying seats at or
  above non-buyers' median money**. The v9 baseline is 304 against 13,844 and no
  session in this family has closed that gap. Secondary reads are
  `land_tiles_planted_per_purchase` above v33's 9-10 of 25 and `land_day` in the
  3-10 band rather than 0.
- What would end the line rather than tune it: the gate firing at a healthy rate,
  the tiles getting worked, and buyers still finishing below non-buyers. That says
  extra land does not pay inside a 30-day season under a frozen field scheduler,
  and the answer is to mask `BUY_LAND` off entirely.

## The v3c fork is undocumented, and this is where it resumes

`herd-v1`, `refusal-v1`, `sigma-v1`, `credit-v1`, `flow-v1`, `splitcredit-v1`,
`housing-v1`, `recurrent-v1` and `splitladder-v1` are all league sessions on
`train_market_free_v2/_v3/_v3b/_v3c.py` and none of them has an entry here.
Their history lives in trainer header comments, which is exactly what rule 3 at
the top of this file forbids. Backfilling them is its own job; `cashweight-v1`
starts the practice again rather than waiting for it.

## v44 -- `market-free-cashherd-v1`

- Changed (strong): `reward.OUTCOME_LOG_SCALE` 5.0 -&gt; 2.5, on the v3c fork. One
  constant, nothing else in `reward.py`. `SLOT_PREFIX` goes `split` -&gt; `cash` and
  the ten `splitladder-v1` lineages end here. The ladder demonstration arm is
  dropped with it, so the roster is uniform.
- **What `splitladder-v1` proved, which is that the wrong quantity was solved.**
  At cutoff 120 `unrecovered_seed_share` and `unrecovered_animal_share` read 0.0
  on nine of ten slots, seed margin +36,934 an episode and animal margin +5,272 --
  Block F's split-credit objective *met* -- with a best slot on 10,868 money.
  Decomposing that slot's `own_mean_return` of -0.261: outcome +0.252, the four lot
  legs +0.155 net, therefore about -0.67 of penalty. `split_02` runs about -1.11.
  This is v26's "the objective was 98% penalty" unfixed, and the part that binds is
  the ratio inside the positive half: trading stock pays +0.155 while growing the
  purse from the 3,000 stake to 10,868 pays +0.252.
- **The behaviour that ratio buys, against 1,050 mirrored ladder replays.** Per
  episode-seat we move 218,563 coins of gross revenue against a ladder champion's
  136,737 and give back 214,104 against its 40,588 -- a 2% net margin against 70%.
  34.7% of episodes finish under the stake; the ladder's finish under it 0% of the
  time. Money is under 100 coins on 40.8% of turns. The policy is not failing to
  trade, it reinvests everything it earns.
- **The constant was measured, not argued.** `episode_rewards` is a pure function
  of a trace, so 300 real episodes of `splitladder-v1` (generations &gt;= 8, both
  seats, median 5,270) were re-scored under candidates before anything was edited.
  Pearson / Spearman of the discounted return against final money, and the mean
  return of a season that made money minus one that lost it:

      S      pearson  spearman  win-lose
      1.0     0.776    0.954     1.413
      1.5     0.789    0.960     1.186
      2.0     0.785    0.955     1.017
      2.5     0.773    0.947     0.891
      3.0     0.759    0.935     0.795
      5.0     0.701    0.888     0.568

  **2.5 rather than the 1.5 optimum**, because 1.5 fits the distribution this run
  happens to produce and that is the failure every earlier value of this constant
  died of. 56,000 -&gt; 124,000 is worth 0.026 of reward at 1.5, 0.079 at 2.5.
- **Measured on the same 300 and rejected**: scaling the four lot weights down to
  put more of the stream on `outcome` buys +0.005 pearson and costs 0.034 of the
  win/lose separation. The lot legs are the only action-differentiable money term
  in the stream -- `outcome` is an episode-level scalar, identical for every action
  at a state, reaching the argmax only through the `n_step` bootstrap -- so
  shrinking them removes signal. Also rejected as an alignment fix:
  `INSOLVENCY_GRACE_TURNS` 24 -&gt; 48 lifts the mean return -0.341 -&gt; -0.246 and
  leaves the correlation flat. It is a level shift and deserves its own session.
- **Why no demonstration arm.** `splitladder-v1` pinned a 96-episode ladder pack on
  slots 00-04 and the value function diverged on contact. `max_abs_q` by cutoff,
  seeded against control:

      cutoff   20     30     60    100    130
      seeded  0.49  20.82  26.98  21.33  18.35
      control 0.57   0.50   0.45   0.49   0.60

  Cutoff 30 is the first update carrying `demonstrations`; `mean_loss` steps
  0.0013 -&gt; 1.194 on the same row and never returns, against control at 0.0005 all
  session. Every reward term is bounded, so a correct Q is single digits. The cause
  is the loss and not the pack: `updater._behavior_clone_step` runs `cross_entropy`
  straight on the Q head's logits, whose optimum is the chosen logit at infinity --
  `grad_clip_norm` bounds the step, not the fixed point -- for 100 steps a batch
  against the TD loop's 48, and the target net is then synced to the inflated head.
  So the seeded-vs-control read is void rather than negative, and no pack should be
  pinned again until the clone loss is bounded. DQfD's large-margin hinge is the
  form, and the roadmap already carries it as unbuilt -- filed there as a way to
  make the clone *stronger*, which the table above inverts.
- **Two numbers that were being read wrong, both from the arena, 12 seeds, both
  orientations.** The gate's `b_money_mean` of 123,661 is pilkwang **uncontested**:
  against a real competitor -- itself -- it makes 69,534 mean and 56,779 median.
  And our own field path is a second leak independent of the market head: swapping
  `donor._role_plan` for `herd.field_roles` and keeping the donor's market rules
  unmodified scores 48,638 and lets pilkwang run to 121,165 at a 0.00 win rate. So
  the honest reference for a contested episode is 56-70k, not 124k, and
  `herd.field_roles` + `planting.market_influenced_roles` own part of the gap.
- Landed with it, both narrowings rather than levers, and both recorded so the
  session is not read as one change. `order_space.stock_ceiling` shuts every SELL
  rung above the smallest that clears the shed -- deduplication, since the shed
  bounds the fill and those rungs are one action spelled several ways, which is the
  proof `feasible_actions` demands. The cost it removes is on the reward, not the
  fill: `PendingTurn.apply` quotes at the full quantity, so an oversized rung
  inflates `expected_delta` and `order_refused` -- a **turn**-level test at
  `FILL_FRACTION` -- reads a turn whose legs all filled as refused, charging
  `REJECTED_ORDER_PENALTY` and latching the state feature. 2,113 of 7,910 sell
  units an episode had no stock behind them. And `updater._large_margin_loss`
  replaces the clone phase's cross-entropy; inert here at
  `demonstration_episodes = 0`, landed so the next demonstration arm is readable.
- **The field overlays, attributed.** `sim.arena`, 12 seeds, both orientations, all
  four running the donor's market rules so only the role plan differs:

      field role plan              ours   pilkwang   win
      donor._role_plan only      69,534     69,534   0.42
      + herd overlay             48,950    121,532   0.00
      + planting overlay         79,617     83,162   0.54
      + both (= field_roles)     48,638    121,165   0.00

  `herd.herd_influenced_roles` is the whole leak -- alone it costs 20,584 and stops
  the seat contesting the market at all. `planting.market_influenced_roles` is the
  opposite, +10,083 and a winning record against pilkwang. `main.py` runs both, so
  the fork is paying 20k for the half that loses. Not changed here: it is an
  inference-path change, the animal lot weights were fitted with the overlay in
  place, and it deserves its own session.
- Pass condition: `own_solvent_fraction` mean over 0.50 against the `splitladder-v1`
  control half's 0.276, and `own_mean_return` crossing zero on a majority of slots,
  **with `order_diagnostics.trading_rate` held within ~25% of 0.392**. The third
  clause is the guard that matters. A slot that stops trading converges on the
  3,000 stake and every trading slot in `splitladder-v1` finishes below it, so
  silence raises `lr_score` mechanically -- `split_03` is doing exactly that at
  cutoff 120-130 on a `trading_rate` of 0.017.

## v45 -- the herd overlay's strip, folded into v44

Landed into `market-free-cashherd-v1` rather than run on its own, because it is an
inference change measured in the arena rather than a training lever that needs a
league to read it.

- Changed: `herd.herd_influenced_roles` pass two is gone. It stripped every
  speculative animal role -- a role on a tile with no structure and no animal --
  which is the donor's four-pasture opening.
- **Attributed within the overlay before being changed.** `sim.arena`, both
  orientations, donor market rules on both sides, everything else equal:

      variant                    ours   pilkwang   win   seeds
      overlay as shipped       50,240    122,515  0.00      24
      no herd overlay          79,343     79,493  0.56      24
      pass two off entirely    83,525     86,111  0.35      24
      pass two payback-gated   83,525     86,111  0.35      24
      overlay, WARM_SLOTS=2    47,706    117,723  0.00      12
      overlay, WARM_SLOTS=4    74,105    103,176  0.17      12

  The strip is the whole of the module's cost -- removing it alone beats removing
  the overlay entirely, because passes one, three and four are worth keeping.
  `WARM_SLOTS` is not the knob: 2 does nothing and 4 only approximates rebuilding
  the donor's pastures by a worse route.
- **The payback gate is bit-identical to deletion**, to the last digit over 48
  episodes, which is what identified the real reason the strip was redundant:
  `donor._field_jobs` emits `BUILD_` only at `left >= animal_rule["first"] + 2` and
  not at all during liquidation, so a speculative role the season cannot pay back
  produces no job and costs nothing. The strip bought nothing the donor's own build
  window was not already refusing. Shipped without the gate rather than with a
  branch that never fires -- rule 2.
- **Why the original argument failed.** "52% of episodes never buy the animal" was
  measured on a learner that was not buying animals, and the strip is part of why:
  it deletes the housing, the housing is what makes a purchase land. Both sides of
  the fork buy now -- `splitladder-v1` averages 9.7 cows, 7.4 sheep, 8.9 geese an
  episode. `docs/dqn_market_free_animal_products.md` carries the correction beside
  its original claim.
- Not changed: passes one, three and four, `WARM_SLOTS`, `warm_slot_species`, the
  ten `HERD_VIABILITY_DIM` state columns. The coop the donor's planner cannot reach
  is still this module's reason to exist.

## v3e -- `market-free-mark-v1`

The v3d reward is the bare cash delta plus the placed-animal potential and the
terminal ruin charge, and it has a credit-timing hole: a harvest in the shed is
worth exactly nothing in the label until the turn it sells, so the only place
"sell" is crisp is the last `n_step = 24` turns, where replay bootstraps with
discount zero and the target is realised cash alone. Measured on ruin-v2 and
open-v1 (60 most recent seats each; learner is the `_match_policy` family,
opponent pilkwang; the scripts read `episode_payloads.replay_gzip` directly):

    metric                                     learner ruin-v2  pilkwang  learner open-v1  pilkwang
    shed value at quotes, day 27                       12,925     9,808           10,548     8,974
    unsold value at terminal, median                    1,540         0            2,449         0
    sale price / day-0 quote, days 20-27, median         1.14      1.14             0.92      1.14
    sale price / day-0 quote, days 28-29, median         1.17      1.26             1.03      1.26
    share of units sold on days 28-29                    0.13      0.27
    money on day 20                                     4,064    12,934

It hoards, dumps late into the prices it has already depressed, and still leaves
produce on the table. The dump itself is not the fault -- pilkwang dumps harder --
the price is.

v3e keeps the cash stream and adds one potential-based term, `STOCK_MARK_WEIGHT`
(`training/reward.py`): held produce, shed plus carried, marked at its constant base
price `donor.MARKET[item][0]` through `market_state.HELD_STOCK_BASIS`, the same
linear-read-of-`turn_state` shape as the animal potential, so every stored pack
relabels for free. A harvest is credited `+base` the turn it lands, a sale at `q`
pays `q - base`, unsold stock is written off at the terminal, and holding costs
`(1 - gamma) * phi` a turn (~0.12 over 300 turns on a 10k shed). Base is a constant,
so there is no quote to pump, which is what ruled out v2's mark-to-market. The
optimum is unchanged; what moves is when the label says it.

Graded on three new per-episode columns from `diagnostics._sale_use`, routed onto
`model_updates.metrics_json` through `_BATCH_LOT_KEYS`: `late_sale_share`,
`sale_price_over_base` (1.0 is a sale at the mark), `unsold_terminal_value`; and on
`own_mean_cash_return`, the cash-only sum `own_mean_return` stopped being when the
animal potential landed. Wanted: late-phase price toward pilkwang's 1.14-1.26,
terminal unsold median toward 0, `trading_rate` inside 0.30-0.47. A collapse of
`trading_rate` or `unsold_terminal_value` climbing with `sale_price_over_base` --
refusing every sale under base and eating the write-off -- means the mark is too
loud: 0.5 next.

Result, mark-v1 (seq 720, 35 updates, 10 promotions): the strongest run on record.
Cold start to 34.2k contested money by seq 100, PROMOTED 40,202 at seq 600 against
pilkwang's 109k (ruin-v2's best at the same seq was 25.9k). `sale_price_over_base`
1.17 -> 1.39, `late_sale_share` 0.13 -> 0.28, `max_abs_q` 1-4 all run. Then flat:
seq 300-700 is 21 updates at mean candidate 31.4k, sd 5.2k, own median 35-39k, so the
two late promotions sit inside the noise. `trading_rate` averaged 0.195 against the
0.20 floor and 20 of 25 rejections cite `activity_gate_failed` (seq 120 rejected at
36,967, above the bar, for a rate of 0.196). `unsold_terminal_value` did *not* close:
2.0-4.5k on every rejected update from seq 500 while the price climbed -- the "too
loud" clause above, except halving the weight (mark-v3, `dqn_market_free_v3f`,
`STOCK_MARK_WEIGHT = 0.5`) scored 13-17k against 32-35k at the same seq, worse on
every column. mark-v2 (floor 0.10) drifted to 0.51 wire orders and 18-21k by seq 300.

## v3e -- `market-free-mark-v4`

Three moves, one session. `UNSOLD_WRITE_OFF_WEIGHT = 1.5` (`training/reward.py`):
the potential leaves a hoarded unit reward-neutral (credited on landing, written off
at T), so nothing in the label spoke against holding to the end; the excess half is
charged on `target` only, like the ruin term, and moves the optimum on purpose.
Reward-grade, so `SLOT_PREFIX` moves to `mark4`. `viability_min_orders_per_decision`
0.20 -> 0.17, under mark-v1's band and above where mark-v2 collapsed.
`resume_reset_after_rejections` 3 -> 2: tripping it resumes from the newest
checkpoint rather than rolling back, so every rejection *under* the limit is the one
that discards 960 steps -- mark-v1 threw away ~240 of its last 400 batches that way.
Graded on `unsold_terminal_value` toward 0 with `sale_price_over_base` held at or
above 1.3, candidate money clearing 40.2k, `trading_rate` staying above 0.17 with
`wire_orders_per_decision` not falling under ~0.8.

