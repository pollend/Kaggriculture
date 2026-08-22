# dqn_market_free — what the Kaggle ladder episodes say

Session history is in [dqn_market_free_sessions.md](dqn_market_free_sessions.md),
the argument about what to build next in
[dqn_market_free_roadmap.md](dqn_market_free_roadmap.md), and one specific
unbuilt niche in
[dqn_market_free_animal_products.md](dqn_market_free_animal_products.md), and
where each layer stands against the ladder in
[dqn_market_free_ladder_standing.md](dqn_market_free_ladder_standing.md). This
file is the read of the *public ladder* replays rather than the league database,
because the ladder is the only place the policy meets 24 different human agents
instead of its own training opponent.

Status: **review only, nothing implemented.** Superseded as a *standing* by
[dqn_market_free_ladder_standing.md](dqn_market_free_ladder_standing.md), which
reads 1,050 mirrored replays rather than these 24 episodes. The five failures
below are still the sharpest per-behaviour read on the account and are why this
file stays.

Subject: submission `55602278`, `landheur_08` checkpoint `eda32fa2`, public score
560.7 — the best-scoring submission on the account at the time of writing. All 24
completed public episodes were downloaded and parsed whole.

```bash
kaggle competitions submissions -c kaggriculture          # find the submission id
kaggle competitions episodes 55602278 --format json       # 24 public + 1 validation
kaggle competitions replay <episode_id> -p <dir>          # ~20 MB each
```

One trap, because it silently inverts this whole file: **a replay's
`steps[i].observation` is the state *after* `steps[i].action`.** Any question of
the form "could the agent afford what it just ordered" has to read money and
prices from `steps[i-1]`. Read post-action instead and the policy appears to
submit a third of its buys against a wallet it has already emptied — an artifact
of reading the wallet after the spend, not a behaviour.

## The record

| | ours | opponents |
| --- | --- | --- |
| episodes won | **10 / 24** | 14 / 24 |
| mean final money | 49,758 | 59,199 |
| median final money | 52,627 | 59,888 |
| mean margin | **−9,441** | |

Both extremes are ours: +41,229 against `tine.sh agent`, −62,751 against
`D S S Kumar`. Every episode ended `DONE`/`DONE`, so nothing below is a crash,
a timeout, or an invalid-action ejection. The policy plays the full 720 steps and
loses on merit.

## Five failures, worst first

### 1. A third of our buys are sized past the wallet and part-fill it to zero

Measured against the *pre-action* state — `steps[i].observation` is the state
after `steps[i].action`, so the affordability of an order has to be read from
`steps[i-1]`:

| `BUY_PRODUCT` orders | ours | opponents |
| --- | --- | --- |
| refused (cash below one unit) | **0.0%** | 2.1% |
| part-filled (cash below `unit × qty`) | **31.5%** | 3.9% |
| total over 24 episodes | 6,709 | 3,131 |

The feasibility mask is not the problem and should not be touched: it opens
`BUY_PRODUCT` at `money >= _price_at(item, stock - 1)`, one unit, deliberately,
because the simulator part-fills. Nothing we emit is refused. What the policy
does instead is pick a quantity rung that costs more than the wallet holds on
roughly a third of its buys, and the simulator drains the wallet filling what it
can. Step 100 of `94310103`: 1,095 money, `BUY_PRODUCT FERTILIZER 12` quoted at
1,176, leaves 8. Forty steps later the identical order against 1,093.

That is legal, and on a rising quote it is even defensible. But it is how the
cash floor in finding 2 gets hit, over and over, by construction.

### 2. Cash starvation is the default state, not an accident

| | ours | opponents |
| --- | --- | --- |
| steps under 100 money | **18.4%** | 8.5% |
| steps under 1,000 money | 42.0% | 31.5% |
| mean peak stock value held | 13,617 | 13,101 |
| mean stock value held | 2,238 | 677 |

`corr(margin, min_money_over_episode) = +0.48`. The loop visible in the replays
is buy-product-with-nearly-all-cash, hold, sell later, repeat. Peak inventory
value is the same as the opponents' but our *average* holding is 3.3× theirs, so
the capital sits in stock instead of turning over.

### 3. Money is backloaded and never compounds

Mean money gained in the final quarter of the episode: **39,979 ours** against
29,357 for opponents. A typical loss has quartile cash of 429 / 2,916 / 18,498 —
that is three quarters of the episode played broke, then a liquidation sprint.
The opponents are ahead by mid-game and their winnings buy more land and seed.

### 4. Dead stock at the buzzer

| | ours | opponents |
| --- | --- | --- |
| terminal shed value | **3,572** | 275 |
| terminal unplanted seeds | 22.0 | 11.5 |

One episode ends holding 100 units — `shedCapacity`, the cap itself. There is no
liquidation window: the policy is still trading its inventory strategy when the
episode stops paying out.

### 5. Hired hands idle, and the farm is under-planted in the losses

| | ours | opponents |
| --- | --- | --- |
| hand actions that are `PASS` | **21.4%** | 15.0% |
| hires per step | 6.85 | 7.01 |
| mean planted tiles | 31.8 | 24.0 |

We pay for the same labour and use less of it. `corr(margin, planted_tiles) =
+0.54`; in the losses planting drops to 13–29 tiles while the winner runs 41–56.
In the −62,751 loss our hands passed 1,515 times while `D S S Kumar` watered 837
and planted 143 — same hire count, opposite use.

Note the opponent column here is a mix of 24 different agents, several of which
are weak (one planted 2.9 tiles and lost by 41k). The comparison that matters is
against the winners, not the mean.

## The collapse case

Episode `94310103`, seed-mate `Friaseus`: **3,648 against 47,077**, our worst
absolute finish.

- 38% of steps under 100 money, 66% under 1,000
- 88 of 207 `BUY_PRODUCT` orders (42%) sized past the wallet and part-filled
- 13.3 tiles planted against their 55.7
- the same `BUY_PRODUCT FERTILIZER 12` re-emitted against ~1,095 money at a
  ~1,170 quote, at step 100 and again at step 140, each time leaving single
  digits behind
- their money curve climbs 3,000 → 36,608 monotonically after day 3; ours never
  leaves the floor

Nothing failed and nothing was refused. The policy spent its wallet down to the
floor on market position inside the first hundred hours, could not fund a farm
out of what was left, and never recovered the base.

## What to do about it

Ordered by leverage per unit of work. 3 is a deterministic wrapper and is correct
without retraining. 1 and 2 change what the policy may do, so they change the
action distribution a trained head expects — measure them against the benchmark
before shipping, and expect a retrain to get the full value.

1. **Size `BUY_PRODUCT` rungs against the wallet, not just the first unit.**
   `feasible_actions` already does exactly this for `BUY_SEED` and `HIRE` —
   `quantities * price <= money`, rungs above the allowance shut rather than
   clamped, for the aliasing reason given in that comment. `BUY_PRODUCT` is the
   one buy column left open on a single-unit test, and it is the column running
   31.5% part-fills. Applying the seed rule there is a small, local change to a
   function that already carries the pattern. Note it is a *policy* change, not a
   correctness fix: part-fills are legal, and the module's stated contract is
   that it only forbids actions under proof. A quantity rung that cannot commit
   in full is not unreachable — the lower rung reaches the same outcome — so the
   proof rule holds, which is the argument the seed and hire rows already use.
2. **Hold a cash reserve floor.** Keep a fixed share of net worth liquid so farm
   operations stay funded while the market position is open. This is the
   difference between the two money curves in the collapse case.
3. **Liquidate on a terminal window.** Sell shed contents and stop buying seed in
   the last ~24 steps. Worth 3,572 mean, and the 100-unit episode is worth more.
4. **Charge for idle labour.** Penalise `PASS` hand actions in shaping, or hire
   fewer hands. 21.4% idle at 6.85 hires per step is a standing cost with no
   output.

## How to tell whether it worked

Re-run this review on a later submission's episodes and compare the same numbers.
The three that should move without any learning:

- `BUY_PRODUCT` part-fill share → should approach the opponents' 3.9% from 31.5%
- terminal shed value → should approach 0 from 3,572
- steps under 100 money → should fall from 18.4%

Then the earned ones:

- mean margin, currently −9,441
- episodes won, currently 10/24
- planted tiles in *losing* episodes, currently 13–29

Rank the checkpoint you submit on `challenger_benchmarks[0].a_money_median`, not
`lr_score`, for the reason given in
[dqn_market_free_animal_products.md](dqn_market_free_animal_products.md): the
arena score swings ~48% per update against 11.7% for the rollout mean. Note also
that ladder score and benchmark money disagree in level — `eda32fa2` benchmarks
around 43.5k against the pilkwang competitor and scores 560.7 on the ladder, so
the ladder is the only place these 24 opponents exist and the only place this
review's numbers can be reproduced.
