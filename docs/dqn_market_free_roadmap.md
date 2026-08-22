# dqn_market_free — unblock promotion, then the v2 roadmap

What each session changed and measured is in
[dqn_market_free_sessions.md](dqn_market_free_sessions.md). This file is the
argument about what to build next.

Standings against the public ladder, and which layer the deficit is in: [dqn_market_free_ladder_standing.md](dqn_market_free_ladder_standing.md)

## Context

`dqn_market_free` was rewritten at `a1c297e` as a single-action DQN: a 186-wide
flat Q head, one market order per turn, argmax over a feasibility mask.

The first league run under that design, `market-free-simple-v2`, is finished and
its verdict is unambiguous. Over 666 batches and 33 model updates it promoted
**nothing**, and the policy it learned was worse than not trading at all:

| | |
| --- | --- |
| quiet (never-trade) baseline, 100 paired seeds | 3,000.0 mean money |
| learner, same seeds | **0.725** mean, 0.0 median |
| `hard_viability.head_to_head.passed` | `false` (floors 2,400 mean / 2,250 median) |
| margin vs pilkwang | −139,128, win rate 0.000 |
| learner money by generation | g0 **1,661** → g1 5 → g5 156 → g12 1,030 → g17 0 |

Generation 0 — untrained — was the best that policy ever was.

**The mechanism, from a rollout money trace:**

```
turn:  0     1     2     3     4  ...  719
$:  3000  2857   417   177     7  ...    0
```

Action attribution for that episode: `HIRE 13` × 18 = −4,026 coins. Hire cost is
fibonacci in `hires_today`, so one `HIRE 13` is 609 coins, and hands are wiped at
every day boundary — a daily rent, not an asset. Three of those orders ate the
whole stake by turn 4, and `feasible_actions` then left nothing but NOOP for the
remaining 715 turns. That dead absorbing state was ~99% of the replay buffer.

Three causes, all now addressed (see **What changed**, below):

1. `feasible_actions` opened all thirteen `HIRE` rungs when only the *first* hand
   was affordable, so the wallet-draining rung stayed available at every wealth
   level.
2. The shaping prices every sensible purchase at exactly zero by design, so Q was
   a flat plateau and its argmax was arbitrary. `REJECTED_ORDER_PENALTY` breaks
   that plateau for refused orders only, and a hire *does* move money, so it was
   never classed as refused.
3. `tanh(margin / 100_000)` against a margin of −128k in every single episode is
   the constant −0.86. A constant on the terminal reward shifts every Q value
   uniformly and can never change an argmax — the exact failure that constant's
   own docstring records for its predecessor.

Intended outcome: a run that keeps its capital past day one, can promote a
checkpoint at all, and a ranked, pre-argued list of what v2 becomes once it does.

## What changed (`market-free-selfplay-v3`)

Landed together, because none of them is observable without the others.

- **`order_space.feasible_actions`** gates each `HIRE` rung on the cumulative
  fibonacci cost of all `n` hands. Stricter than `_do_hire`, which commits what
  it can and stops — deliberately: `HIRE k` is its own index on the same ladder,
  so no outcome the simulator produces becomes unreachable.
- **`training/reward.py`** adds `INSOLVENCY_FLOOR` / `INSOLVENCY_PENALTY`, a
  per-turn charge for a wallet with no runway. Bankruptcy is an *opportunity*
  cost, and the money delta cannot express one; this prices the state locally
  instead of asking a 24-step bootstrap to carry it across 700 turns of dead
  state. Not policy-invariant, on the same terms as the refusal charge.
- **`TERMINAL_MARGIN_SCALE` 100,000 → 10,000**, sized to the self-play margin
  distribution rather than the scripted field's.
- **`updater._self_play_pool`** follows the lineage rather than the release. The
  scorer's shared helper filters on `promoted`, which made the mirror
  unreachable on a run that promoted nothing in 33 updates. Release is gated; the
  behaviour distribution is not.
- **`train_market_free.py`** runs one anchor (`configurable_crop_surrogate`,
  32k — the weakest available) with `self_play_ratio = 0.65`, so two of every
  three rollouts face one of this model's five most recent checkpoints.

First measurement, batches 1–8: `mean_final_money` 1,414–2,103 with a max of
4,748, against 7–39 at the same point on `simple_v2`. The bankruptcy collapse is
gone; whether it can now *win* is the open question.

## What changed again (`market-free-outcome-v4`, then `market-free-mirror-v5`)

`selfplay_v3` ran 240 batches over 13 generations and promoted none of its 12
updates. It did not fail the way `simple_v2` did: gate money against the quiet
3,000 baseline climbed 182 → 506 → 1,573 → 2,020 across the run, so the solvency
work above bought something real — just not the 2,400 mean / 2,250 median floors.
Its `mean_loss` sat at 0.002 throughout, a Q head carrying almost no TD signal.

`outcome_v4` dropped the money delta and the shaping (`reward_mode =
"outcome_only"`) and pinned 200 surrogate demonstrations at `demo_fraction 0.25`.
**It promoted.** Update `cutoff_batch_seq = 40` released at 5,984 money against
the quiet baseline's 3,000, win rate 0.94, `hard_viability.passed = true` — the
first checkpoint this family has ever shipped, and the answer to the Phase 1
question below: the blocker was the policy, not the rule, and no cold-start
promotion path was needed.

Then it diverged. The next three updates produced candidates finishing at 1.8,
2.5 and 0.8 coins while `mean_loss` climbed 4.48 → 5.94 → 9.63, against a reward
stream bounded in roughly [-3, +2].

**The mechanism.** Under `outcome_only` the whole stream is
`tanh(margin / TERMINAL_MARGIN_SCALE)` plus two ~0.002 charges, and against
anything scripted that `tanh` is a *constant*:

| source | share of a minibatch | margin | outcome |
| --- | --- | --- | --- |
| pinned demonstrations | 25% | +20k–31k | **+1.0** |
| anchor rollouts | ~26% | −20k–31k | **−1.0** |
| mirror rollouts | ~49% | hundreds | ~±0.05 |

About half of every batch carried a flat ±1.0 label on states the network cannot
tell apart — both seats open at 3,000 coins under the same frozen donor field
policy. No action explains that spread; it is irreducible target variance, and
the Double-DQN max-bootstrap compounds it. `selfplay_v3` never showed it because
the money delta anchored the regression there.

`outcome_baseline` was documented at `reward.py` as the lever for a saturated
`tanh`, but a baseline is one scalar and this distribution is bimodal — no value
of it centres both modes. It never ran at anything but `0.0` and has since been
removed from the code; the `0.0` entry left in `train_market_free.FREE_CONFIG` is
inert, kept only so `ensure_session`'s experiment digest keeps the sessions whose
stored manifests already carry it resumable.

So `market-free-mirror-v5` removes both saturated sources rather than trying to
centre them: no scripted opponent (`opponents = []`), no demonstrations
(`demonstration_episodes = 0`), and `configurable_crop_surrogate` out of
`benchmarks` as well. What is left is the mirror, whose margins are the only ones
`TERMINAL_MARGIN_SCALE` was ever sized to discriminate, and pilkwang as the
release objective the gate measures. Two live risks, taken knowingly: this is the
reward-only control that promoted nothing on `selfplay_v3`, and `train_dqn.py`
records 17 consecutive rejections from a curriculum with no static opponent.

Three supporting changes landed with it:

- **`mean_abs_q` / `max_abs_q` in `metrics_json`.** The divergence was only ever
  visible as `mean_loss` rising, which cannot distinguish a harder target from an
  exploding one, and by the time the gate rejects a candidate the update that
  started it is several checkpoints back on the lineage.
- **The lineage snapback now fires on a run that never promoted.**
  `lineage_checkpoint` snapped back to `released()` after
  `resume_reset_after_rejections`, but only `if fallback is not None` — so on a
  run with nothing released the guard was silently inert, exactly where it was
  needed. `selfplay_v3` took twelve consecutive rejections at a limit of five that
  way. With no release, `None` is the reset and the caller cold-starts.
- **The streak restarts at a recorded reset, not only at a promotion.** The job
  scorer's copy always read it that way; the market scorer's did not, so once the
  limit tripped it tripped on every later update. `gate_json.lineage_reset` is now
  written by the free-market updater so the query can see it.

- **Exploration schedule doubled** (`epsilon_decay_transitions` 2M → 4M). With
  neither demonstrations nor a scripted opponent, epsilon is the only thing left
  putting unfamiliar states in the buffer, and `total_transitions` is scoped by
  `model_id` — inherited along with the lineage, so the run would otherwise have
  resumed at 0.063 and hit the 0.01 floor three generations in.

## What changed again (`market-free-explore-v7`)

`noisy_v6` ran 240 batches over 12 updates and promoted none of them. It did not
fail on the reward this time. It failed on its own controller.

| update | `noisy_argmax_flip_rate` | `noisy_sigma_scale_target` | target |
| --- | --- | --- | --- |
| 220 | 0.278 | 0.264 | 0.05 |
| 200 | 0.394 | 0.378 | 0.05 |
| 180 | 0.343 | 0.170 | 0.05 |

Every multiplier below 1.0. `training_batches.metadata_json.actor_sigma_scales` on
the last batch reads `[0.326, 0.199, 0.121, 0.074, 0.045, 0.027, 0.017, 0.010]` —
not one of the eight actors played at the learned sigma, and the greediest played
at a hundredth of it. Meanwhile every gate reported `trading_rate: 0.0` and
`a_money_mean: 3000.0` against a `reference_money_mean` of 3000.0: the policy was
being rejected for tying the never-trade baseline, with `mean_abs_q` 0.04–0.09 and
`mean_loss` 0.0005.

**The mechanism, and it was predicted in writing.** `agent.toml` said of
`noisy_flip_rate_target`:

> this cannot prevent the NOOP collapse. A collapsed policy sits on a flat
> plateau where every action is near-tied, so flips are cheap and its flip rate
> reads *high* while it trades nothing — targeting the rate there would turn
> sigma down.

That is exactly what happened. The knob was documented as unable to catch the NOOP
collapse; what the run showed is stronger — on a collapsed policy it does not
merely fail to help, it actively throttles the exploration that might escape.

So the controller moves onto the quantity `updater.py` already computed and
labelled "Reported, never controlled on": `noisy_flip_to_order_rate`, the share of
states the greedy policy would sit out that the perturbation talks into a real
order. The near-ties that inflate the flip rate are precisely what keeps this one
low, so it cannot be fooled the same way. `flip_rate_curve` now sweeps it across
the whole sigma ladder rather than reading it at the learned sigma only — a
diagnostic measured at one point cannot be controlled on — and
`noisy_order_rate_target` selects it, taking precedence over
`noisy_flip_rate_target`. Exactly one curve may move sigma; a run with both set
could not say afterwards which number moved it.

Two supporting changes:

- **`resolve_sigma_scale` no longer overshoots a reachable target.** Its upper
  early-out was `target >= points[-1][1]`, so a target the ladder exactly reaches
  returned the *largest* scale sharing that rate instead of the smallest. Dead
  weight on the argmax flip rate, which is strictly increasing and never ties; the
  flip-to-order rate saturates at 1.0 once every NOOP state is talked out, so the
  tie is the normal case there. Now `>`, and the reachable target falls through to
  the interpolating loop.
- **`noisy_cadence` "day" → "episode".** The n-step pairing that chose "day" is
  satisfied strictly harder by one policy per 720-turn season. What it buys is
  temporally extended commitment — the property separating deep exploration from
  dithering. A perturbation redrawn every 24 turns cannot hold a plan across the
  48–96 turn BUY_SEED-to-harvest cycle, which `agent.toml` already recorded as the
  reason "episode" was the standing follow-up. The cost is real and taken
  knowingly: 8 distinct policies per batch against day cadence's 480, and one bad
  draw wastes a whole episode rather than 24 turns.

Same model name (`outcome_v4`) again. The reward is untouched, neither change
alters the checkpoint format, and `_self_play_pool` is scoped by `model_id`.

### The BUY_LAND measurement that motivated it

Land purchase is not under-explored, it is extinct, and the extinction is what a
Q-gap-bounded explorer with `epsilon = 0` does when an action has been punished:

```
gen  3: 348 seats bought land   <- spike
gen  4-7: 52, 44, 27, 39
gen  8-26: 0                    <- 19 consecutive generations
gen 36-57: 1..31                <- NoisyNet trickle
gen 58-59: 0
```

The gradient is right about the action *as offered*. 707 land-buying seats average
723 money (median **7**, win rate 0.287) against 30,985 non-buyers at 6,367 (median
5,512, win rate 0.609), and 649 of the 707 sit in the 0–3,000 bucket — with zero
land buys among the 1,439 seats above 12,000.

But that is a timing artifact. The donor buys land too (`donor.py:1927-1939`,
`LAND_OPEN_DAYS = (5, 9)`, `MAX_EXTRA_LAND = 2`, only with `money >= cost +
reserve` and `left >= 12`) and finishes above 100,000. `order_space.py:332-334`
opens the action the instant `money >= LAND_PRICES[extra]` — turn 0, day 0, 1,000
of a 3,000 stake — so the network only ever sampled the ruinous variant. The state
can tell the two apart (`market_state.py:484-485` encodes `day / total_days` and
`hour`), so this is sampling, not expressibility.

Nothing in v7 addresses that directly. Gating `BUY_LAND` on day and cash reserve
in `feasible_actions`, mirroring the donor, is the obvious follow-up and has
in-repo precedent — the HIRE fibonacci gate fixed the identical `simple_v2`
collapse — but it makes the mask encode preference where `order_space.py:263`
says it encodes feasibility. Worth revisiting once the controller fix shows what
the policy does with sigma restored.

## What v7 measured, and what `market-free-burst-v8` does about it

`explore-v7` worked, and it worked immediately.

| | v6, last two generations | v7, first two generations |
| --- | --- | --- |
| mean money | ~3,000 (the NOOP fixed point) | 11,407 → 12,025 |
| `mean_abs_q` | 0.04–0.09 | 0.56–0.60 |
| seats buying land | **0** of 1,068 | 38 of 532, then 56 of 534 |
| promotions | 0 in 12 updates | **1 in 2** |

The promoting update resolved `noisy_sigma_scale_target` to **1.096** — above 1.0,
which is the pass condition stated below and the first time this run family has
collected at more than the learned sigma.

What v7 did *not* do is make those land purchases any good:

| v7 | seats | mean | median | max | win rate |
| --- | --- | --- | --- | --- | --- |
| no land | 998 | 12,745 | 13,844 | 29,794 | 0.655 |
| bought land | 94 | 1,107 | **304** | 17,250 | 0.021 |

Every purchase lands on day 0. `feasible_actions` opens `BUY_LAND` the moment
`money >= LAND_PRICES[extra]`, so 1,000 of a 3,000 stake buys it before any
revenue exists, and the donor's day-5-with-reserve version is never sampled.
More exploration at the same setpoint just generates more day-0 ruin and drives
`Q(BUY_LAND)` back down faster.

**The structural reason the setpoint cannot find it.** `actor_sigma_scales` is
`alpha ** (-i/(N-1))`: actor 0 plays the controller's resolved base and the other
seven play *less*. Measured on v7's last batches, the whole ladder is
`[0.574, 0.350, 0.213, 0.130, 0.079, 0.048, 0.029, 0.018]`. No episode anywhere
in any batch explores harder than the setpoint, so the rarest behaviours are
exactly the ones collection is blind to.

`market-free-burst-v8` adds a schedule rather than raising the setpoint:
`noisy_burst_every = 5`, `noisy_burst_order_rate = 0.20` against the ordinary
0.05. Every fifth round rebases the whole ladder onto the sigma that lands on the
burst rate, preserving its shape — the near-greedy bottom rung is what gives
`trading_rate` and the money diagnostics an honest read, and a burst that
flattened the ladder would remove it on precisely the rounds hardest to interpret.

**Sized as a rate, not as a sigma multiplier**, and that distinction is measured
rather than stylistic. A multiplier drifts twice over: the base it scales moves
every update as the controller tracks the setpoint, and what a given scale buys
moves with the network. `resolve_sigma_scale` reads the burst off the same
`noisy_flip_to_order_curve` the setpoint uses, so both re-size together.

The rate also puts a measured ceiling on how hard a burst may go, and the first
draft of this change did not have one. v7's curve:

| sigma scale | flip rate | flip-to-order |
| --- | --- | --- |
| 1.0 | 0.200 | 0.089 |
| 2.0 | 0.352 | 0.192 |
| 4.0 | 0.647 | 0.536 |
| 8.0 | 0.858 | **0.739** |

A burst at scale 8 would place an order on three of every four turns the policy
would have sat out. At 720 decisions with ~97 of 186 actions spending money that
is the epsilon purchase pump restated: the episode bankrupts in its first turns
and the remaining ~700 are the dead NOOP absorbing state that was 99% of
`simple_v2`'s buffer. A burst round has to search, not detonate. 0.20 resolves to
a scale near 2.0 on that curve — against a setpoint scale of 0.658, roughly three
times the exploration and nowhere near the cliff:

```
ordinary: [0.658, 0.401, 0.244, 0.149, 0.091, 0.055, 0.034, 0.021]
burst:    [2.034, 1.240, 0.756, 0.461, 0.281, 0.171, 0.104, 0.064]
```

**Rounds rather than actors, and that is the design.** Raising the ladder inside
*every* batch would work too, and would charge for it continuously: `priority_alpha`
is 0.0, so high-sigma episodes are sampled uniformly beside the good ones, and a
permanent noise floor in every minibatch is the epsilon purchase pump with a new
name — the exact failure NoisyNet was turned on to remove. Alternating concentrates
the cost. A burst round fills the ring with exploratory trajectories; the four
ordinary rounds after it refill the FIFO with on-policy data while
`reservoir_fraction = 0.35` keeps a sample of the burst alive to learn from.

Cadence. Five because `rounds_per_generation` is 20 against `min_new_batches` 20,
so a generation is exactly one update and every update trains on four
consolidation rounds per burst — 20% of the batches collected, and rather less
than 20% of the ring once the FIFO has churned. Every-other would put a burst in
half the buffer permanently, which is the noise floor the schedule exists to
avoid.

The phase counts `training_batches` for the `model_id`, so it is continuous across
restarts and session ids rather than restarting at a burst on every relaunch. Same
model name again — the reward, the checkpoint format and the self-play pool are all
untouched.

**Read the run by splitting on the flag.** A burst round is *meant* to look bad on
`mean_final_money`, so `metadata_json ->> 'noisy_burst'` is load-bearing rather
than decorative: without it the money series alternates between two regimes with
nothing in the row to say why, and the ordinary rounds' progress disappears under
the average.

```sql
SELECT metadata_json ->> 'noisy_burst' AS burst,
       count(*) AS batches,
       round(avg((metadata_json ->> 'mean_final_money')::numeric), 0) AS money
FROM training_batches
WHERE metadata_json ->> 'noisy_burst' IS NOT NULL
GROUP BY 1
```

What would make it a failure: ordinary-round money falling toward the burst
rounds' — that is the noise floor leaking into consolidation, and the answer is a
longer `noisy_burst_every`, not a smaller scale. What would make it a success is
narrower than "money goes up": a land purchase appearing at day ≥ 5 *with* the
episode staying solvent. Bursts raise the sampling rate of every rare action
equally; they do not make the day-0 variant any less available, so if the timed
purchase still never appears, the remaining levers are `noisy_layers = "all"` for
genuinely state-dependent exploration, or the mask day-gate.

## What `burst-v8` measured: nothing, and why (`market-free-burst-v9`)

The bursts fired exactly as scheduled — `batch_seq` 1326, 1331, 1336, 1341, at a
resolved base of 1.892 against the setpoint's 0.822 — and changed nothing.

| v8, 560 seats | seats | bought land | mean money | median | max |
| --- | --- | --- | --- | --- | --- |
| ordinary rounds | 452 | 22 (4.9%) | 13,014 | 16,079 | 29,809 |
| burst rounds | 108 | 5 (**4.6%**) | 13,771 | 16,483 | 24,203 |

The land rate is flat and burst money came out *higher*, not lower. The design
predicted burst rounds would look visibly worse; they did not, which means they
were not reaching new behaviour at all.

**Attribution note, because it is easy to get wrong.** v8's ceiling is not new:
its best episode is 29,809 against v7's 29,794, fifteen coins apart, and no batch
*mean* has exceeded 17,774. The large gains — 3,000 to 12k+, two promotions —
belong to the v7 controller retarget and episode cadence, not to the bursts.

**Cause: the ladder diluted the burst.** Rebasing preserved the 32× Ape-X spread,
so a burst round ran

```
ordinary: 0.822, 0.502, 0.306, 0.187, 0.114, 0.070, 0.043, 0.026
burst:    1.892, 1.155, 0.705, 0.430, 0.263, 0.160, 0.098, 0.059
```

Only **two of eight** burst rungs exceed an ordinary round's *hardest* actor —
from the third rung down a "burst" episode explores less than an ordinary one. At
one round in five that is roughly 5% of the run actually exploring, which cannot
move a 4.6% land rate.

`market-free-burst-v9` fixes both halves:

- **Burst rounds are flat.** Every actor collects at the burst base, so all eight
  episodes outexplore every ordinary one instead of two of them. The ladder's own
  justification does not survive the move to episode cadence: it exists because one
  temperature per batch fills the buffer with near-copies at 81% mean pairwise
  action agreement, and that was measured under *epsilon*, where a shared rate
  really does produce near-copies. Under per-episode noise, eight episodes at one
  sigma are eight different perturbed policies. The near-greedy read the ladder
  also supplied comes from the four ordinary rounds between every burst.
- **`noisy_burst_order_rate` 0.20 → 0.35**, still under the knee at scale 4 where
  `flip_to_order` reaches 0.536 and episodes bankrupt in their first turns.

Resolved against v7's curve, that is a burst base of 2.750 with every actor on it,
against an ordinary ladder topping out at 0.658.

What would still count as failure: burst and ordinary rounds again
indistinguishable on the land rate. At that point the burst has been given a flat
ladder and 4× the setpoint rate and still found nothing, and the conclusion is that
undirected weight noise cannot reach the timed purchase however large it is —
which is what `noisy_layers = "head"` predicts, since factorized head noise picks
actions almost state-independently. The next lever is then `"all"` or the mask
day-gate, not a larger burst.

## What `burst-v9` settled, and `market-free-landgate-v10`

v9 ran the flat burst at a resolved sigma of 3.159 and answered the exploration
question against itself.

Per-episode attribution is what did it, and it needs no new instrumentation:
`training_batches.metadata_json -> 'seeds'` is 16 long, one seed twice per pair,
and `actor_sigma_scales` is indexed by pair, so joining
`jsonb_array_elements(...) WITH ORDINALITY` on `(ord - 1) / 2` to
`episode_players.seed` gives the sigma every episode was actually played at. Over
batches ≥ 1361, the first collected under the flat-burst build:

| round | σ | seats | broke (<500) | **land** | mean money |
| --- | --- | --- | --- | --- | --- |
| **burst** | **3.159** | 212 | **104 (49%)** | **0** | 4,892 |
| burst | 2.129 | 56 | 16 (29%) | 2 | 7,726 |
| ordinary | 0.871 | 108 | 32 (30%) | 6 | 10,901 |
| ordinary | 0.531 | 104 | 8 (8%) | 8 | 11,647 |
| ordinary | 0.197 | 108 | 28 (26%) | 8 | 11,753 |
| ordinary | 0.045 | 108 | 20 (19%) | 10 | 12,991 |
| ordinary | 0.027 | 104 | 18 (17%) | 8 | 13,081 |

**Noise does not buy land.** Zero purchases in 212 seats at 3.159. Every land
purchase in the run happened at σ ≤ 0.871 and the *rate* is highest at the lowest
rungs — 9.3% at 0.045 against 5.6% at 0.871. Land is chosen by the greedy policy,
not found by exploration. The mechanism is the one the reward comments have
circled since `simple_v2`: 97 of 186 actions spend money, so at that perturbation
the wallet is gone in the first turns and `money >= LAND_PRICES[extra]` never
holds again. Bankruptcy arrives before day 5 does.

Read round-level averages with suspicion here. Aggregated by round, v9's bursts
looked *better* than ordinary rounds on money — an artifact of a 26-seat window.
The per-episode split reverses it.

So the binding constraint was never the perturbation. It was the action space, and
v10 changes that instead:

- **`feasible_actions` gates `BUY_LAND`** on day and a cash reserve, mirroring
  `donor.py:1927-1939` and reading its constants rather than restating them:
  `day >= LAND_OPEN_DAYS[extra]`, `TOTAL_DAYS - day >= 12`, `money >= price +
  300/500`, capped at `MAX_EXTRA_LAND`. This is the **first mask rule that removes
  reachable outcomes**, which the docstring now says outright — the HIRE rung rule
  stayed inside the feasibility claim precisely because `HIRE k` is its own index,
  and a day gate cannot make that claim. The simulator-parity test keeps checking
  the direction that still matters: the mask may forbid a purchase the game would
  accept, never offer one it refuses.
- **`noisy_burst_order_rate` 0.35 → 0.10**, resolving to σ 1.308 on the curve
  update 1362 recorded — just above the ordinary ladder's 0.871 top rung.
- **`noisy_burst_scale_limit` 1.5.** The rate alone cannot bound the burst,
  because what a rate costs in sigma depends on a curve that moves with the
  network; `noisy_sigma_scale_limit = 8.0` is what let 0.35 reach 3.159. The
  ordinary controller keeps its own looser limit.
- **`resume_reset_after_rejections` 5 → 2.** Independent of the above and
  predating every burst: across v7, v8 and v9 the run oscillates rather than
  diverging monotonically — it promotes, then produces candidates at 1,335 / 975 /
  3,877 against a released incumbent scoring ~17,000, and sometimes recovers
  before a limit of 5 fires. That recovery is what made 5 look adequate. Every
  rejected update in between is a generation collected from a policy the gate has
  already judged far worse than one sitting released and unused.

One consequence worth knowing while reading the first generation: replay artifacts
store `order_mask` as computed at collection time and the updater reads those for
both `mask` and `next_mask`, so episodes already in the ring still say `BUY_LAND`
was feasible on day 0. At `buffer_episodes = 400` and 16 episodes a batch the ring
churns in ~25 batches, so it self-heals inside about one generation.

**The pass condition is narrow.** Not "money goes up" — land buyers ceasing to be
the bankrupt ones. The v9 baseline is a 304 median for land-buying seats against
13,844 for the rest; the target is land buyers at or above the non-buyers' median.
If the gate produces well-timed purchases that still lose money, the conclusion is
that extra land does not pay inside 30 days under a frozen field scheduler, and
the action should be masked off entirely rather than timed.

## The lineage reset was half a reset

`resume_reset_after_rejections` fired for the first time on `landgate-v10`, and
what followed showed the guard only ever rolled back a third of the state.

| update | candidate | released incumbent | `lineage_reset` |
| --- | --- | --- | --- |
| 1382 | 8,494 | 17,197 | false |
| 1403 | **357** | 18,339 | **true** |
| 1423 | 401 | 18,901 | false |

The reset resumed from an 18k released checkpoint and one generation of training
produced a 357 policy — and then the next twenty batches were *collected* from
that 357 policy, at 3–8k against the 11–16k before it.

Three things carried the discarded trajectory forward:

1. **The buffer.** `fill_buffer(buffer, context.all_batches)` rebuilds from recent
   batches whatever checkpoint was loaded, so the restored weights train
   immediately on ~400 episodes generated by the policy just thrown away.
2. **The lineage pointer.** The resetting update still trains, still writes a
   `model_training_checkpoints` row, and the lookup takes the newest by
   `created_at`. The reset also zeroes the streak — correctly, the old trajectory
   is gone — so one update later the counter reads healthy while the lineage
   points at the failed restart's own output.
3. **The self-play pool.** `_self_play_pool` takes the five most recent
   checkpoints promoted or not, so the mirror opponents were the discarded policy
   too.

**Fixed at (2), which subsumes (1) and (3).** `reset_since_promotion` holds the
lineage on the release until something promotes, rather than for a single update.
Skipping only the resetting update's own checkpoint would have been worse — the
next newest is the diverged row the reset existed to escape. Because collection
follows the lineage as well, holding the release means the ring refills with the
released policy's episodes and the self-play pool stops mirroring the wreckage, so
the next attempt trains from good weights on good data.

The cost, taken knowingly: this freezes the behaviour distribution while the gate
says no, which `lineage_checkpoint` argues against in general and
`_rejections_since_restart` restates as "resetting buys another `limit` updates,
not a permanent policy". That reasoning stops applying once
`resume_reset_after_rejections` consecutive rejections have already proven the
trajectory bad — a frozen release beats following a policy the gate scored fifty
times worse.

Landed in **both** copies. The job scorer's `_reset_since_restart` and the market
scorer's `reset_since_promotion` are deliberately identical: these two files
diverging on exactly this logic is what produced the streak bug recorded under
`mirror-v5`.

## Guards that were inert exactly where they were needed

Three separate guards in this family were written to fire "once there is a
released incumbent" and were therefore silently off on a run that had never
released — which is the state each of them exists to protect. Worth naming as a
pattern, because it has now cost three debugging sessions.

1. **The lineage snapback** (recorded under `mirror-v5`): `lineage_checkpoint`
   snapped back to `released()` only `if fallback is not None`, so a run that
   promoted nothing carried one diverged trajectory through twelve consecutive
   rejections at a limit of five.
2. **The streak that feeds it**: `rejections_since_promotion` did not stop at a
   recorded reset in the market scorer's copy, so once the limit tripped it
   tripped forever.
3. **The activity floor**, fixed now. `require_activity = active is not None`
   meant `viability_min_orders_per_decision` did nothing until something had been
   promoted. `market-free-noisy-v6` recorded the shape directly:
   `trading_rate: 0.0` beside `hard_viability.passed: true`.

The third is the one that decides what becomes an incumbent in the first place,
so being inert at cold start is the worst place for it to be inert. With
`opponents = [seeded_random]` and nothing released, the head-to-head runs against
a policy that reliably bankrupts — `evaluation_opponent` has never recorded a
non-zero `final_money` — so a pure-NOOP candidate keeps its 3,000 stake, wins
3,000-to-0 significantly, and passed hard viability unopposed. It is unconditional
now. Rejection does not stop training, and 0.02 is cleared by any policy that
trades at all, against 0.29 measured on a live trading run;
`viability_min_orders_per_decision = 0` is the deliberate off switch.

The floor reads `trading_rate`, the share of turns the *simulator acted on*, not
the wire order count: `HIRE 13` is one decision that expands to thirteen rows, so
a wire-count floor reads 0.13 orders per decision for a policy that acted once in
a hundred turns.

**Which arena carries it, corrected after `market-free-population-v25`.** Making
it unconditional was right; applying it to the *cold-start head-to-head* was not,
and the two are separable. That match is against the quiet baseline, which offers
no counterparty, so the floor measures nothing there. At cutoff 143 learner_01,
learner_02 and learner_09 traded at 0.633, 0.332 and 0.398 against the benchmark
and emitted 0, 210 and 0 orders against the baseline; five of ten slots had never
released after 143 batches and `population_members` held one row for the whole
run, so every graduation took the `restart` branch and the mature group stayed
empty. The floor is now unconditional on the index-0 benchmark and conditional on
a release for the head-to-head, falling back to requiring it when no benchmark is
configured. The v6 shape is still caught: a pure-NOOP candidate reads 0.0 against
pilkwang too. Do not re-merge the two arenas without re-reading this paragraph —
the pattern above makes "unconditional everywhere" look like the safe direction,
and on the head-to-head it is what blocked the first release.

**And the degeneracy check alongside it, same run, same cause.**
`degenerate_head_to_head` inferred the NOOP collapse from a zero-width paired
margin. That inference is false: round-trip trades are cash-neutral against an
unchanged market, so `learner_05` placed 16,686 orders over 35,950 decisions at
cutoff 143 — a trading rate of 0.464 — and still finished every paired seed on
exactly 3,000.0 against a baseline doing the same. It passed the ratchet at 987
benchmark money and passed hard viability, and this check alone rejected it. It
now reads the candidate's own `trading_rate` through `_collapsed_against_baseline`
rather than the margin, and shares the activity floor's off switch.

**A note on the reset hold, which nearly repeated the same mistake in reverse.**
Holding the lineage on the release until a promotion is only safe when there *is*
a release. With nothing promoted, `released()` is `None`, so the hold would return
a cold start on every update from the first reset onward — and a run that
reinitializes every update never trains far enough to promote, which is the only
thing that clears the hold. That is the reinitialize-every-update treadmill
`rejections_since_promotion` already warned about. The hold is conditional on
`context.active_checkpoint is not None`; with nothing released the streak alone
governs and each reset buys another `limit` updates of runway.

The two changes compose in the safe direction: the activity floor makes a first
release *harder*, and the reset hold only engages once one exists. A fresh run
therefore trains freely until it produces something that actually trades, and only
then does the stricter lineage discipline begin to apply.

## Scoring the exploration work, and `market-free-idleburst-v12`

Five sessions of changes, scored against what was actually measured:

| change | verdict |
| --- | --- |
| **v7** controller onto flip-to-order, `noisy_cadence = "episode"` | **helped decisively** — 3,000 (NOOP) → 11–12k, `mean_abs_q` 0.04 → 0.56, two promotions in three updates |
| v8 periodic burst @ 0.20 (σ 1.892) | neutral — 4.6% of burst seats bought land against 4.9% ordinary, over 560 seats |
| v9 flat burst @ 0.35 (σ 3.159) | harmful — 49% of 212 seats bankrupt against 8–26% ordinary, zero land, mean money 4,892 |
| v10 land gate | land purchases → 0 in 666 seats; money 8,960 against 13,160 |
| v10 `resume_reset_after_rejections` 5 → 2 | harmful on a never-released run — a wipe, not a snapback |

Only v7 helped. The rest tuned exploration while the measurements kept saying
exploration was not the binding constraint.

**The land gate is reverted.** `feasible_actions` is pure feasibility again. The
evidence that motivated it stands and is worth keeping: land-buying seats finished
at a 304 median against 13,844 for the rest, and 649 of 707 purchases landed in
the 0–3,000 money bucket, because `LAND_PRICES` opens at 1,000 against a 3,000
stake and the purchase is offered on turn 0. What the gate could not show is that
forbidding it *helps* — under it the policy stopped buying land at all rather than
buying it later, and money fell. Revisit from that evidence, not from scratch.

**The burst is repointed from a schedule to idleness.** A NOOP policy produces no
learning signal at all, so bankruptcy is strictly the better failure: it at least
generates transitions with a gradient in them. That inverts the objection to σ
3.159 — it was right on a healthy policy and is the point on an idle one, because
the episodes it replaces were worth nothing and it stops by itself the moment
trading resumes.

`noisy_idle_trading_rate = 0.005` (well under `viability_min_orders_per_decision`,
so it answers genuine silence rather than caution) fires the burst when the
previous batch's `order_diagnostics.trading_rate` is at or below it.
`noisy_burst_every = 0` retires the periodic path; its code and tests stay, since
the idle trigger reuses them and a tested negative result is worth keeping.

Two details that matter when reading it:

- `last_batch_trading_rate` returns **None**, not 0.0, when nothing is recorded.
  0.0 *is* the idle reading, so defaulting to it would burst on the first batch of
  a cold start before the policy had any chance to trade.
- `metadata_json ->> 'noisy_burst_reason'` (`"idle"` / `"schedule"` / null) and
  `noisy_idle_trading_rate_seen` are recorded per batch. A burst round is meant to
  look terrible on money, and there are now two different reasons it could be.

**The v7-era policy is gone.** Checkpoint `2d12c2877b8279562e90fd31`, the 17–18k
incumbent, is absent from disk; the 13 surviving `outcome_v4` checkpoints all
postdate the collapse. v7's result was code *plus* an inherited 1,300-batch
lineage, and only the code half is recoverable. Every run from here cold-starts.

## Ratings: there are none, by construction

`rating_snapshots` and `generation_stats` are empty for every session in this
league, and that is not a broken pipeline. `configs/dqn_market_free_league.toml`
sets `pairing_mode = "trainer_only"` with `population_size = 0`, so nothing ever
inserts into `generation_members`, and `_close_generation`
(`kaggriculture_lab/league.py`) reads its members from that table before writing
any snapshot. The dashboard already special-cases `trainer_only` and shows round
counts where a leaderboard would be. Do not "fix" it; `leader_conservative: null`
on `list_training_sessions` is the expected reading.

The substitute per-generation measure is the head-to-head money table:

```sql
SELECT g.generation_index, me.family_id, opp.family_id AS opponent,
       count(*) AS n, avg(ep.win::int) AS winrate,
       avg(ep.final_money) AS my_money, avg(ep.opponent_money) AS opp_money
FROM episode_players ep
JOIN agent_revisions me ON me.revision_id = ep.strategy_id
JOIN agent_revisions opp ON opp.revision_id = ep.opponent_id
JOIN pairing_episodes pe ON pe.match_id = ep.match_id
JOIN league_pairings lp ON lp.pairing_id = pe.pairing_id
JOIN generations g ON g.generation_id = lp.generation_id
WHERE g.session_id = %s AND NOT ep.failed
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
```

`episodes` has no `pairing_id`; the link is through `pairing_episodes.match_id`.

## The lineage stopped moving, and `market-free-population-v23`

`falloff_v1` did not diverge, collapse, or bankrupt itself. It sat still. Over
batches 1100–1688 — 590 batches, roughly 30 updates — every series is flat:

| 50-batch bucket | 1100 | 1200 | 1300 | 1400 | 1500 | 1600 | 1650 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rollout `mean_final_money` | 4,997 | 5,745 | 6,872 | 5,074 | 6,527 | 5,559 | 4,498 |

`mean_loss` held between 0.0014 and 0.0017 across updates 1321–1662 and
`mean_abs_q` between 0.19 and 0.33. Benchmark money against pilkwang ran 0.9 to
1,208 from a 3,000 stake, with two promotions in the last twelve updates each
immediately followed by rejections.

Every exploration change from v7 through v20 aimed at the *behaviour
distribution* — the sigma setpoint, the periodic burst, the idle burst, the hot
rungs, the rare-action widening. Only v7 helped. What had stopped moving was the
trajectory, and a trajectory cannot be pushed out of its basin by the gradient
that put it there.

**Two learner groups replace the single ever-resuming lineage.** Ten named slots
train in parallel under one declared model — the harness already namespaces them
as `dqn_market_free.raw_order_q_v1@learner_00…09` and gives each its own
lineage, artifacts and batches, so no harness change was needed. A slot whose
plateau-driven learning rate decays past `lr_graduate_cut` is *ended* rather than
nursed: its released checkpoint graduates into `population_members`, the mature
group drops its lowest scorer to hold `mature_capacity = 30`, and the slot
restarts from the best mature member with its learning rate back at zero and 10%
of its weights redrawn.

**The learning rate is the graduation clock, and the plateau is what drives it.**
Warmup climbs linearly from zero to `learning_rate` over `lr_warmup_updates`;
after that, `lr_plateau_patience` updates without a `lr_plateau_min_delta`
relative improvement halve it. At 1e-4 down to a cut of 1e-5 that is four
halvings, and at patience 4 that is at least sixteen stalled updates — sixteen
generations here — before a slot can end. Against a run that took 57 consecutive
rejections without anything ending, that is generous rather than trigger-happy.

Progress is measured on a **windowed median of the slot's own benchmark money**,
never on `mean_loss`. Loss is not a progress signal on this family and is not
even the right sign — a policy reaching harder states has a *larger* TD residual
— and a single arena draw is not a number here: `falloff_v1` measured 61 of them
on one unchanged checkpoint and got everything from 0.93 to 427. A plateau
detector reading single draws finds an improvement by luck every few updates and
never decays, which would reproduce the stagnation inside the mechanism meant to
end it. The plateau counter also does not run during warmup, where a slot is
deliberately under-trained and its scores describe the ramp rather than the
policy.

**What graduates is the released checkpoint, never the candidate.** A mature
member should be a policy the gate has already accepted, and a slot that decayed
to its cut is by definition one whose recent candidates stopped clearing the bar.
A slot that never released contributes nothing and simply restarts. Its score is
the windowed ratchet reference the gate already computed for that incumbent, so
membership costs no arena time — the same reason `_released_benchmark_money`
reads a recorded measurement instead of replaying the incumbent.

**The reseed is a whole reset, and that is the part with a scar.** "The lineage
reset was half a reset" above records what happens otherwise: a restored 18k
checkpoint trained immediately on ~400 episodes from the 357-coin policy just
discarded, and the mirror kept playing it. So a reseed carries two watermarks for
the *slot's whole life*, not for the one update that consumes it —
`reseed_batch_floor` keeps pre-reseed batches out of the buffer, and
`reseed_after_checkpoint_id` keeps pre-reseed checkpoints out of the self-play
pool, which falls back to the mature group while the slot has none of its own.
The optimizer state and step counters are dropped rather than restored.

The perturbed weights are **materialized as an artifact** rather than redrawn on
each side from a shared seed, so `collect` and `update` load identical bytes. Two
code paths agreeing only as long as they keep agreeing is the class of drift the
`to_numpy` silent-skip bug in this file already cost a debugging session to.

**`resume_reset_after_rejections` goes to 0, and not out of caution.** The
snapback and the population answer the same question differently, and leaving
both on is actively harmful: the snapback restores the slot's own incumbent,
whose score is exactly what the plateau detector is reading, so `lr_best_score`
gets pinned by the mechanism meant to rescue the slot and the slot never reaches
its cut.

**The hot actor ladder is retired with it.** `noisy_actor_hot`,
`noisy_hot_order_rate` and `noisy_hot_scale_limit` are gone and `_noisy` now
raises on them rather than ignoring them; `cold_order_diagnostics`, which existed
only to keep the idle trigger reading past them, goes too. Per-episode
attribution refused them twice: on `burst-v9`, 212 seats at sigma 3.159 finished
49% bankrupt against 8–26% on the ordinary ladder and bought *zero* land, with
every land purchase in the run happening at sigma ≤ 0.871; on `falloff-v18` the
hot rungs' land rate was 4.6% against the ordinary rungs' 4.9%. 97 of 186 actions
spend money, so a widened policy is broke before a rare action is affordable.
`rare_sigma_gain` widens single rows and stays — it is a different mechanism, and
`rare.sigma_scale_vector` leaves NOOP and every well-sampled row at exactly 1.0.

**The cost, stated plainly.** `_update_session_models` iterates the roster in one
process and each slot collects its own `rounds_per_generation` batches, so a
generation takes **ten times** what it did. `jobs` stays at 8 inside each
collector because the slots run one after another, not beside each other.

**And the concern this was built through.** `mean_abs_q` at 0.2 with `mean_loss`
at 0.0015 is a Q head carrying almost no TD signal. A population fixes a *stuck*
trajectory, not a flat one. If all ten slots decay to `lr_graduate_cut` with
benchmark money still under 100, the binding constraint is the reward and the
answer is Phase 3 below, not more slots.

**The pass condition is narrow, and it is not "money goes up".** It is the mature
group's *best* score strictly increasing across at least two graduations — the
ratchet the single lineage could not produce.

```sql
SELECT declared_model_id, count(*) FILTER (WHERE retired_at IS NULL) AS live,
       max(score) FILTER (WHERE retired_at IS NULL) AS best,
       min(score) FILTER (WHERE retired_at IS NULL) AS worst
FROM population_members GROUP BY 1
```

```sql
SELECT model_id, cutoff_batch_seq,
       metrics_json ->> 'lr'              AS lr,
       metrics_json ->> 'lr_stage'        AS stage,
       metrics_json ->> 'lr_score'        AS score,
       metrics_json ->> 'lr_best_score'   AS best,
       metrics_json ->> 'lr_stall_updates' AS stall,
       metrics_json -> 'population' ->> 'action' AS action
FROM model_updates ORDER BY started_at DESC LIMIT 30
```

Failure shapes to watch for, each with a different answer:

- **`lr_stage` stuck at `warmup`** — the score read is wired to the wrong
  benchmark index and `_settle_slot` is receiving `None` every update.
- **Every slot graduating within a generation or two of each other** — the
  plateau terms are too tight for the measurement noise, not evidence about the
  policy. Raise `lr_plateau_patience` before touching anything else.
- **`population.action = "restart"` on every graduation** — no slot is ever
  releasing, so the mature group stays empty and ten slots are cold-starting in
  parallel. That is a gate problem, not a population problem.
- **Batches after a reseed still naming the graduated checkpoint in
  `metadata_json ->> 'behavior_checkpoint_id'`** — the marker is not reaching
  `collect`, and the reset is a third of a reset again.

## What `population-v25` measured, and `market-free-logmoney-v26`

v25 ran 11 generations, 110 batches and 17,600 episodes with zero failures, and
produced a policy worse than never trading.

| | |
| --- | --- |
| benchmark money, 14 measured batches | 98 – 2,717 from a 3,000 stake |
| pilkwang, same series | ~142,000 |
| win rate | **0.000 at every batch** |
| `trading_rate` | 0.00 → 0.78 |
| `own_solvent_fraction` | 0.003 – 0.028 throughout |
| promotions | 10 of 108 updates; 3 slots never released |

It learned to act, and every action lost. The population question is **not**
answered: `population.action` is `train` on all 109 updates, `_graduates` needs
four LR halvings at patience 4 — sixteen stalled updates — and only eleven exist.
The four `population_members` rows are the *contribution* path
(`release_age >= population_entry_updates`), not graduations. What v25 settled is
upstream of the roster.

**Three causes, landed together because none is observable without the others.**

**1. The outcome was a margin, and in the mirror the margin is zero.**
`self_play_ratio` is 0.65 and `league_pairings.mean_margin` was exactly 0.000 in
all eleven generations — both seats run the same policy off the same stake under
the same frozen donor field. So two of every three rollouts carried no outcome
term at all, and `IDLE_PENALTY` and `INSOLVENCY_PENALTY` were the entire
objective. Their joint optimum is *emit an order every turn and stay a coin above
the floor*, which is the measured policy exactly. `reward.py:145-150` records this
same failure once already, at `IDLE_PENALTY = 0.0005`; it recurred because the
term those charges were sized against is structurally absent in the mirror.

**2. `tanh(margin / 10_000)` capped ambition at ~20k.** Against the bankrupt
seeded-random opponent the margin is own money, and that tanh is 0.76 at 10k,
0.96 at 20k, 0.999 at 40k. Every policy from the low twenties to pilkwang's 142k
was reward-identical. `explore-v7` plateaued at 11–12k and `falloff-v18` at 5–6k,
both sitting on the flat part. **The stall this document attributes to a stuck
trajectory — and built a ten-slot population to escape — is also what a saturated
objective looks like**, and the population had not begun to act when v25 produced
the same shape from a cold start.

**3. The gate judged the policy fifty times outside its training distribution.**
`market_state.py` fed `opponent_money / 10_000` and `money / (money + opponent +
1)` raw. Every training opponent holds 0–3,000, so those features lived in ~[0,
0.3] and ~0.5 in every transition ever buffered; pilkwang, which appears in no
rollout, finishes on 142,000 — 14.2 and 0.019. Measured rather than inferred: one
unchanged `learner_04` checkpoint read `trading_rate` 0.805 against pilkwang and
0.055 against its own incumbent **in the same update**. Own money was already
`log1p(money)/10` one line above; the asymmetry was the tell.

**What v26 changes.**

- **`OUTCOME_LOG_SCALE = 5.0` replaces `TERMINAL_MARGIN_SCALE`.** The outcome is
  `tanh(log((final + 1) / (stake + 1)) / 5)`, with the stake read off `money[0]`
  rather than declared. Scale-free, so there is no saturation point to place, and
  opponent-independent, so the mirror, the scripted opponent and the benchmark all
  score the same way — the bimodal-by-source target variance diagnosed under
  `outcome_v4` has no mechanism left. tanh(1) is a 167× season, 500,000 coins,
  deliberately well past pilkwang so beating the benchmark is a point on the slope.

      final money  3,000   6,000   12,000   40,000   142,000   500,000
      outcome       0.00    0.14     0.27     0.48      0.65      0.77

  It is also the number the gate already promotes on. Training on margin while
  gating on `benchmark_release_min_money` is the mismatch that put the old
  constant in an impossible position to begin with.

  **The cost, stated because it is the thing to watch.** The bottom tightened: a
  3,000-to-1,898 slide now scores −0.091 against a fully idle season's −0.072, so
  idling still beats churning but by 0.019 rather than 0.080. Break-even is 2,092
  coins. If `trading_rate` climbs while money falls, re-read this ratio before
  touching `IDLE_PENALTY`.

- **`priority_alpha` 0.0 → 0.7**, its documented intended value and the one
  `priority_beta`'s measurements were taken at. `demo_fraction` is conditional on a
  demonstrator and there is none, so until this moved, *nothing* was putting
  solvent transitions in a batch — a uniform draw over a buffer that is 99.7%
  insolvent teaches the only policy consistent with it.

- **`market_state` puts opponent wealth on the same log scale as our own**, and
  takes the wealth share as a ratio of logs. That moves 0.89 → 0.38 across the
  whole range the old form spent between 0.9997 and 0.019. The lagged money deltas
  are clipped to [-1, 1], both legs symmetrically; the own leg has never left that
  range on any recorded run, so the clip is inert there today. `MARKET_STATE_DIM`
  is unchanged at 238 and the numpy/torch parity test is untouched.

- **New model names (`logmoney_00…09`) and a new session.** `all_batches`,
  `_self_play_pool`, `lineage_checkpoint` and the slot LR state are all scoped by
  `model_id`, and a shaped return is not comparable across `OUTCOME_LOG_SCALE` —
  every v25 episode is mislabelled and every v25 checkpoint was fitted to the old
  labels. Renaming severs all four at once. The roster stays at ten; it is now ten
  independent reads on a reward change.

**The pass condition is narrow, and it is not "money goes up".** It is
**benchmark money above the 3,000 idle baseline on a majority of slots**, which no
session in this family has ever produced — every one of them has been beaten by
never trading. `own_solvent_fraction` above 0.2 is the same statement read off the
rollouts instead of the arena.

Failure shapes, each with a different answer:

- **`trading_rate` climbing while benchmark money falls** — cause 1 recurring
  under a reward that no longer permits it, which would mean the charges are now
  too large relative to the tightened bottom. Re-read `OUTCOME_LOG_SCALE`'s
  break-even, not `IDLE_PENALTY`.
- **Benchmark money flat at 3,000 with `trading_rate` 0.0** — the NOOP attractor,
  and the absolute outcome makes idling score exactly 0.0 rather than positive.
  That is `viability_min_orders_per_decision`'s case, not the reward's.
- **The two arenas still disagreeing on `trading_rate`** — the state fix did not
  reach far enough, and the remaining OOD feature is own money's *linear* leg at
  `market_state.py:517`, deliberately left alone here because this agent has never
  been rich enough for it to bite.
- **`mean_abs_q` climbing past the reward bound** — priority replay concentrating
  the draw too hard. `priority_beta` is the dial, and 0.7/0.5 is the pair the
  effective-sample-size measurement was taken at.

If all of it holds and benchmark money still sits under the stake, the reward is
no longer the binding constraint and the next lever is the one already written up
under **Out of scope, worth recording**: pilkwang as a *training* opponent so its
seats persist and become clonable, plus a DQfD large-margin term on
`_behavior_clone_step`. It was deferred from v26 deliberately — landing it
alongside three objective changes would have confounded all four.

## Land is not a learned decision any more

Six sessions tried to make the Q head value `BUY_LAND` and the last of them,
`landhandle-v42`, tried to make the argmax stop being the obstacle — the row came
out of the ranking and hung on `Q(BUY_LAND) - Q(NOOP)` instead. That did not take,
because the row it read is dead. Over batches 351–390, all ten slots:

| | |
| --- | --- |
| `buy_land_rate`, per batch | 0.000 – 0.050 |
| the same column on v33 `landbonus` | 0.37 |
| the same column on v41's first pure-mirror batches | 0.206 |

One purchase per twenty to two hundred episodes. The mechanism is the one
`order_space.py` has recorded since the v10 gate: a gated row is starved of
gradient, so its Q is arbitrary in exactly the states where the gate opens, and a
set point placed on an arbitrary number is not a decision rule. Everything tried
against it before v42 was an exploration lever, and the per-episode attribution
refused all of them — 212 seats at σ 3.159 bought *zero* quadrants, and every
purchase on record happened at σ ≤ 0.871.

**`market-free-landheuristic-v43` decides it in code.** `order_space.land_score`
multiplies three factors — lateness, utilisation of the land already owned, and
runway past the wage bill the extra tiles will need — and `choose_order` never
reads the row. The product matters: `horizon` alone peaks on day 0, which is the
timing that put 649 of 707 purchases in the 0–3,000 money bucket, and `worked`
vetoes it because nothing is planted on turn 0 whatever the wallet says. Walked
over a plant-up-then-earn season the first purchase lands on day 8, inside the
donor's own `LAND_OPEN_DAYS = (5, 9)`, with no day hardcoded anywhere.

`LAND_PURCHASE_BONUS` goes to 0.0 with it — it existed to tilt an argmax that no
longer ranks the row — which makes this a reward move, so `SLOT_PREFIX` moves.
That had been impossible: a fresh prefix restarts at batch 0, where the empty
`opponents` base reached the updater with no mix and raised. It seeds the quiet
network there now. Full argument and the pass condition are in
[the sessions log](dqn_market_free_sessions.md).

## Phase 1 — promotion, if it is still blocked

**Implemented as `promotion_rule = "benchmark_ratchet"`, and the shape it took was
the opposite of the one below: the problem was not a gate too strict to release,
it was a gate that had been switched off.** Under `promotion_rule = "always"`,
`falloff_v1` released nine consecutive updates, three of them with
`hard_viability.passed = false`. Its head-to-head win rate against its own
predecessor read 0.29, 0.00, 0.16, 1.00, 0.00, 0.38, 0.59, 0.94 — each comparison
locally honest, and the series meaningless, because the incumbent it measures
against *is* the previous update. Against the fixed pilkwang benchmark the same
run scored 2,979 → 54 → 20 → 1,146 → 18 → 359 → 1 → 11 → 107 → 96 → 146, win rate
0.00 throughout, and `export-active` was pointing at the 107-coin checkpoint whose
benchmark median was 1.

The absolute floor sketched below is what the fix kept: the release decision now
reads the candidate's own money in the index-0 benchmark series against
`benchmark_release_min_money`, plus a `benchmark_money_ratchet` non-regression
term against the incumbent's *recorded* score for the same opponent — no extra
arena time, since that number is already in the incumbent's `gate_json`. The
head-to-head stays computed and advisory. Two anti-stall mechanisms bound how long
it can block: `promotion_ratchet_grace_after_rejections` stands the relative term
down after a streak (never the floor), and `resume_reset_after_rejections = 2`
snaps the lineage back to something known good.

The money floors had the same drift — their reference is the incumbent's own seat,
which at `falloff_v1@80` put the mean floor at 58 coins — and on a cold start the
reference was the quiet baseline, which never trades and so keeps its full 3,000
stake, making `0.80 ×` it a bar no trading policy clears.
`viability_cold_start_reference_money` replaces it; `coldstart_v1@140` (candidate
mean 1,633 / median 2,218, benchmark money 894) is the update that would have
given that run its first incumbent instead of seven rejections and no escape.

The premise as originally written, and the `outcome_v4` finding that settled it:

**Settled, and the answer was no rule change.** `outcome_v4` promoted at
`cutoff_batch_seq = 40` under the existing `significant` rule with
`incumbent_checkpoint_id` null — a 5,984-to-3,000 win, not a tie — so the
cold-start path argued below was never needed. The blocker was the policy, as the
`simple_v2` gate said (`degenerate_head_to_head: false`, a challenger genuinely
losing 0.725 to 3,000). Kept here because the reasoning is still the right
reasoning if a future run stalls on a genuine tie rather than a loss.

The premise as originally written:

**Cold-start promotion rule.** The `significant` rule is correct for incumbent
replacement and questionable for the first release, where the baseline is a
synthetic quiet policy and a tie is a plausible outcome. Add a distinct cold-start
path in the gate: when `incumbent_checkpoint_id` is null and the head-to-head is
degenerate (zero-width CI), fall through to the hard-viability result plus an
absolute floor on the challenger's own money rather than requiring a significant
paired win.

Files: the gate construction in `agents/dqn_market_free/training/updater.py`
(the block that emits `gate_json` with `kind`,
`significant_incumbent_improvement_and_hard_viability`,
`degenerate_head_to_head`), and a new cold-start knob in the `[models.config]`
section of `agents/dqn_market_free/agent.toml` alongside the existing
`promotion_rule` / `viability_*` keys.

**Read the gate before changing it.** Confirm from `gate_json` which of
`head_to_head_significantly_better` and `hard_viability.passed` is false, and on
which check. A rule change aimed at the wrong one buys nothing.

## Phase 2 — cheap architecture wins (no MDP change)

Both preserve the action space, the replay format, `choose_order`, and the mask.
Both change the checkpoint format.

**Hard constraint on anything in this phase:** `agents/dqn_market_free/policy.py`
(numpy, ships in the Kaggle bundle) and
`agents/dqn_market_free/training/network.py` (torch) must produce bit-identical
outputs for identical weights. The parity test in
`tests/test_dqn_market_free.py` pins it, `network.build()` deliberately reads
`HIDDEN` and the layer order out of `policy.py` so sizes cannot drift, and
`to_numpy` / `from_numpy` must round-trip. Every change below is therefore two
implementations plus an export path, not one.

1. **Dueling head.** At 720 decisions per episode most turns are "nothing worth
   doing" and V dominates A — the regime dueling exists for. Smallest change with
   a real prior behind it.
2. **Quantity-ordinal head.** 173 of the 186 columns are (op, item) × quantity and
   currently share zero parameters: `SELL WHEAT 12` and `SELL WHEAT 16` are
   unrelated outputs. Replace the final `nn.Linear(hidden, ACTION_COUNT)` with a
   composed head — embeddings over the ~17 (op, item) groups and the 11-level
   `QUANTITIES` ladder, scored against the trunk. Output stays 186 flat scalars;
   argmax, mask, and `ACTIONS` indexing are untouched. Group membership is already
   available from `order_space._indices_by_item`.
3. **NoisyNet — landed, `market-free-noisy-v6`.** Done first of the three
   deliberately: all three rewrite `to_numpy`/`from_numpy`, and that function had a
   silent-skip bug — it selected layers with `isinstance(layer, nn.Linear)` and
   *dropped* anything else, so a new layer type produced a short network that
   `has_inert_hidden_trunk` waves through (it checks `layers[:-1]`, sees a nonzero
   238×128 weight, passes) and `save` writes. The first evidence was a shape error
   inside an arena worker one update later. It now raises on any unrecognised
   parameter-bearing module, and `from_numpy` length-checks instead of letting
   `zip` truncate. Doing that once, up front, is what makes 1 and 2 safe.

   What it changes and why: epsilon's explore branch never consults the network —
   uniform over feasible actions with NOOP removed — so an exploring turn is
   *forced* to place an order and 97 of the 186 actions spend money. The rate of
   ruinous purchases is therefore a property of the action space, not of what the
   policy has learned, which is the `simple_v2` bankruptcy driver restated. Weight
   noise is bounded by the Q gap instead: at `sigma0/√128 = 0.044` the head
   perturbation is ~0.5·‖h‖ against Q in single digits, so it flips near-ties and
   cannot reach an action scored far below the best. Measured in
   `test_noise_explores_like_the_policy_not_like_a_coin`: 0 of 2,000 noisy draws
   reached a `BUY_LAND` at mu −10 against a best of +10.

   Head-only (`noisy_layers = "head"`). That keeps `has_inert_hidden_trunk`
   correct as written — under `"all"`, a trunk with μ=0 and σ>0 is *not* inert but
   the guard fires and `_reject_inert_network` raises with a message that is
   factually wrong, so widening it is a same-commit requirement for that setting.
   `"all"` is the honest follow-up: head-only factorized noise correlates all 186
   action perturbations through one 128-dim `f_in`, so it is trunk noise that buys
   genuinely state-dependent exploration.

   **Corrected on measurement.** The design draft argued σ = 0 is a stationary
   point (`dL/dσ = L'(W)·E[ε] = 0`), by analogy with the all-zero-MLP argument in
   `policy.py`. That is false: built at `sigma0 = 0.0`, σ moved 1.8e-2 within 20
   Adam steps. Only the expectation vanishes — a per-minibatch `L'_ij · ε_ij` does
   not, and Adam rescales it to a ~lr-sized step. σ is still born in torch, but for
   the plainer reason that it keeps `quiet_start` σ-free so generation 0 still
   trades nothing.

   **New failure mode this introduces:** σ anneals toward zero, exploration stops,
   the run goes greedy-only, and nothing else in `metrics_json` moves. Watch
   `noisy_argmax_flip_rate` (the share of states where the perturbation changed the
   decision — the number directly comparable to epsilon) and `sigma_ratio_*`. The
   clone optimizer excludes σ from its parameter group because the clone phase runs
   *before* the TD loop and cross-entropy strictly prefers σ→0.

   **That failure mode is now controlled, not only watched** (`noisy_flip_rate_target`,
   default 0.05, declared in `agent.toml` rather than in `train_market_free.py` because
   `ensure_session` digests `experiment.training_agents` and a new key there strands v6).
   Each update sweeps a geometric ladder of σ multipliers through the same held-out
   probe and records the whole curve as `noisy_flip_rate_curve`; `resolve_sigma_scale`
   log-interpolates the multiplier that lands on the target and writes it as
   `noisy_sigma_scale_target`; the next round's `collect` reads it back out of
   `model_updates` and multiplies it through the Ape-X actor ladder, so actor 0 plays
   at the target rate and the spread runs down from there. A lookup against a measured
   curve rather than a controller with gains: nothing to tune, nothing to persist,
   nothing to oscillate.

   Two things it deliberately does not do. It cannot rescue σ = 0 — no multiplier can,
   so `resolve_sigma_scale` returns 1.0 on a flat curve instead of saturating at the
   limit and reading as a controller doing its job. And it widens the gap between the
   collection σ and the σ the TD loop trains at; that gap already existed under the
   Ape-X spread, but it is now larger and configurable, so read `mean_loss` against
   `mean_abs_q`/`max_abs_q`.

   **What it does not fix:** the NOOP attractor (and `epsilon = 0` removes a floor
   on trading, so `viability_min_orders_per_decision` matters more, not less), the
   `outcome_only` target-variance divergence, and the v3 round-trip loop. This is
   one collapse driver removed, not a cure.

   The flip-rate controller does not touch the NOOP attractor either, and the two are
   easy to confuse. A collapsed policy sits on the flat plateau the shaping creates —
   every sensible purchase at exactly zero immediate reward — where flips are cheap, so
   its *flip rate reads high* while it trades nothing, and targeting the rate there
   would turn σ down. `noisy_flip_to_order_rate` is logged for that: the share of
   states the greedy policy would sit out that the perturbation talks into a real
   order. Reported, never controlled on.

   Resume across the change goes through `_load_online_state`, which remaps
   `N.weight` → `N.weight_mu` and skips both optimizer restores — σ adds tensors to
   the parameter group, so `optimizer.load_state_dict` cannot span it regardless of
   `strict`. Same model name (`outcome_v4`) is load-bearing, for the reason
   recorded in `train_market_free.py`: `_self_play_pool` is scoped by `model_id`, so
   a fresh name has an empty pool on batch 0 and a pure-mirror run cannot cold-start.

## Phase 3 — distributional value

The return distribution is bimodal by measurement — a bankruptcy mode at 0 and a
solvent mode reaching ~10k, with `own_solvent_fraction 0.0` on the rejected
update. A scalar mean-Q cannot represent "usually fine, occasionally ruinous," and
the mean is dragged by the bankruptcy mode.

`demo_fraction` (0.25) was described here as papering over that pathology. It is
not: it only applies to pinned demonstrations, `demonstration_episodes` is 0, and
the share therefore resolves to zero on every update this run has taken. The
pathology has been unmitigated the whole time. `priority_alpha` is the answer to
the *sampling* half of it — rank the agent's own episodes by shaped return and
draw the good ones more often, with an importance weight undoing the bias — and
it composes with, rather than replaces, this phase. Reweighting the draw changes
which returns the head is fitted on; a distributional head changes what the head
can say about them.

Quantile-regression head (QR-DQN) over the same 186 actions; argmax on the mean of
quantiles so `choose_order` is unchanged. Keeps Double DQN — the existing target
in `updater.py` (online picks the argmax, target scores it, ~line 780) carries over
with the gather widened across the quantile dimension.

Follow-on this unlocks: the viability gate can read a risk statistic (CVaR over
the lower quantiles) instead of a mean money floor.

## Phase 4 — the actual v2: sequential intra-turn orders

Recovers exactly what the v1 rewrite gave up, without reviving what v0 died of.

The simulator accepts 10 orders per turn (`donor.MAX_MARKET_ORDERS`); v1 uses one
slot, and the only multi-row action is `HIRE n`, which is still a single index
whose `to_wire()` expands (`order_space.py:101`).

Not slots. **Autoregressive**: up to k sub-decisions per turn — emit an order,
re-encode state with it tentatively applied (money decremented, shed updated, mask
recomputed), emit the next, until STOP or k. Each sub-decision is its own
transition with its own Q(s, a) and its own TD target. v0 collapsed because total Q
was a *sum over slots*, so no slot received individuated credit and every slot's
argmax landed on the same order — one good buy became a tenfold spend
(`order_space.py:14-22`). Here each sub-decision is a genuine MDP step, so the
credit is individuated by construction.

Two measured payoffs:

- Demonstration `over_budget` drops from 27,770 of 55,652 orders toward zero;
  `mapped_fraction` 0.501 → ~1.0. Roughly doubles the behaviour-clone signal with
  no new collection. `_chosen_index` in `training/clone.py:104-118` is where the
  drop happens today.
- Same-turn sell→buy becomes expressible. Today it must sequence across turns and
  the market moves in between.

Costs and open decisions: k forward passes per turn; a `_post_order_state` model
(precedent — `donor._post_field_storage` already models the simulator's
intermediate storage so `feasible_actions` can see a hand's mid-turn DROP);
replay format becomes intra-turn; and `n_step = 24` needs an explicit decision on
whether 24 counts turns or sub-steps.

This changes the MDP and can re-break the agent. It is deliberately last, and
gated on Phase 1 having produced an actually-promoted checkpoint to regress
against.

## Out of scope, worth recording

The donor field scheduler is frozen; only the market layer learns. Episodes show
8 plants lost and 22.4 shed items unsold at the deadline — production-side losses
no market policy can fix. Joint field+market is a different agent, not a version
of this model.

Also unexplained, flagged not fixed: the run requested
`demonstration_episodes: 200` (session `config_json`) and the update reported 64.
`demonstration_episodes` is a `LIMIT`, and `clone.DEMONSTRATION_SQL` filters on
`final_money >= demonstration_min_money` first, so the number of *eligible* seats
is the real cap — which makes the count a measurement, not a discrepancy.

**What re-enabling demonstrations would run into, measured on `noisy_v6`.** The
mechanism is complete rather than stubbed (`updater.py:1073-1087` →
`clone.load_demonstrations`), and `replay.py:388-392` means `demo_fraction = 0.0`
lets pinned rows drive the behaviour-clone phase while never entering a TD target
— which sidesteps the saturated `+1.0` label diagnosed above without giving up the
clone. The blocker is the data:

| candidate family | seats | ≥ 10k | ≥ 5k |
| --- | --- | --- | --- |
| `dqn_market_free_match_policy` (own mirror) | 19,120 | 4,099 | 10,744 |
| `dqn_market_free` (trainer seat) | 12,534 | 1,877 | — |
| `evaluation_opponent` (seeded random) | 6,586 | 0 | 0 |
| `competitor_pilkwang_structured_economic_policy` | **0** | — | — |

pilkwang has no rows at all — `league_opponents` is off and it appears only as a
promotion-gate benchmark, whose matches are not persisted as league episodes. And
self-imitation cannot substitute for the case that matters here: the learner's own
high-money episodes contain *zero* `BUY_LAND` orders, so cloning them re-teaches
the local optimum.

If taken up, the route is pilkwang as a training opponent rather than a harness
change — `_opponent_policy` (`agents/dqn_market_scorer/training/progress.py:35-41`)
resolves a `main.py` spec with a sibling `agent.toml` to its real family, so its
seats would persist as `competitor_pilkwang_structured_economic_policy` and become
clonable. Pair it with a DQfD large-margin term on `_behavior_clone_step`
(`updater.py:379-405`): the current phase is plain cross-entropy, which the TD loop
washes out, and the margin term is what makes a demonstrated action *stay* above
its alternatives.

## Verification

Query the lab DB directly (container `kaggriculture-postgres`, database
`kaggriculture`, user `kag_admin`) or via the MCP endpoint at
`http://127.0.0.1:8000/mcp`.

- **The self-play mix passes** when `training_batches.metadata_json ->>
  'self_play_episodes'` is non-zero from batch 20 onward — it was zero on every
  one of `simple_v2`'s 666 batches — and `mean_final_money` holds above the
  3,000 stake's neighbourhood across a whole generation rather than decaying to
  zero within one, which is what `simple_v2` did after g0.
- **Phase 1 passed** on `outcome_v4`: `model_training_checkpoints` has a
  `promoted = true` row and `training_models.active_checkpoint_id` is non-null.
  It took the ordinary `significant` path, not a cold-start one.
- **`mirror-v5` has no scripted opponent left** when this returns only
  `dqn_market_free*` families:

  ```sql
  SELECT DISTINCT opp.family_id
  FROM episode_players ep
  JOIN agent_revisions opp ON opp.revision_id = ep.opponent_id
  JOIN pairing_episodes pe USING (match_id)
  JOIN league_pairings lp USING (pairing_id)
  JOIN generations g USING (generation_id)
  WHERE g.session_id = 'market-free-mirror-v5'
  ```

- **The divergence is gone** when `metrics_json.mean_abs_q` holds within an order
  of magnitude of the reward bound across three consecutive updates and
  `mean_loss` stops rising. That is the pass/fail on the constant-label
  diagnosis. If loss still climbs with both saturated sources removed, the cause
  is elsewhere and Phase 3 should not be started on top of it.
- **`explore-v7` stopped throttling sigma** when
  `metrics_json.noisy_sigma_scale_target` is at or above 1.0 while the policy is
  still collapsed, and `training_batches.metadata_json.actor_sigma_scales` no
  longer has its *top* rung below 0.33. That is the direct pass/fail on the
  controller change; the v6 baseline to compare against is
  `[0.326 … 0.010]`. `noisy_sigma_control_signal` says which curve moved it, and
  `noisy_flip_to_order_curve` and `noisy_flip_rate_curve` are both recorded so the
  two controllers can be read against each other on the same update.

  ```sql
  SELECT cutoff_batch_seq,
         metrics_json->>'noisy_sigma_control_signal' AS signal,
         metrics_json->>'noisy_argmax_flip_rate'     AS flip,
         metrics_json->>'noisy_flip_to_order_rate'   AS to_order,
         metrics_json->>'noisy_sigma_scale_target'   AS scale
  FROM model_updates ORDER BY started_at DESC LIMIT 10
  ```

- **Exploration is reaching the buffer** when `mean_final_money` on
  `training_batches.metadata_json` stops reading exactly `3000.0`. Movement in
  either direction means the actors are trading again; holding at 3,000.0 means
  the change bought nothing. Expect the capital actions that reappear to be
  *unprofitable* at first — the day-0 `BUY_LAND` timing problem is untouched by
  v7, and whether they appear at all is the measurement.
- **The gate can see a live policy** when `gate_json.hard_viability.trading_rate`
  and `wire_orders_per_decision` are non-zero. Both are 0.0 throughout v6, which
  is why `viability_min_orders_per_decision` never bit.
- **No regression in the harness**: `episodes` stays at 0 `failed`,
  `model_round_hooks` all COMPLETE, `model_updates.error_text` empty.
- **Phases 2-4 each**: `pytest tests/test_dqn_market_free.py` — the numpy/torch
  parity test is the gate on every architecture change — plus
  `tests/test_train_market_free.py`. Then a fresh league session and compare
  `episode_players.final_money` mean/median and solvent fraction against the
  baselines in the Context table above, at generation boundaries only
  (`generations.status = 'COMPLETE'`).
- **Promotion-relevant comparison** is `pairing_results.paired_margin`, not raw
  per-seat money — it already cancels seat advantage.
- **Phase 4 specifically**: `metrics_json.demonstrations.mapped_fraction` on the
  first update should move from ~0.50 toward 1.0. If it does not, the
  autoregressive path is not consuming the demonstrator's later orders and the
  change bought nothing.
