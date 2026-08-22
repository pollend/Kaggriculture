# dqn_market_free_v3c

One Q-network over a flat order space, behind a GRU that remembers the season.
The donor's market layer is deleted, not modulated — this is the unscaffolded
counterpart to `dqn_market_scorer`.

## Why

`dqn_market_scorer` learns a residual on top of a hand-written market policy: the
donor decides what to buy and how much, the network scales that demand and holds
back stock. Every hard rule stays in code — price thresholds, phase gates, the
survival feed floor, terminal liquidation, cash reserves — so the measured margin
answers "can a learned modifier improve pilkwang?" and nothing more.

This agent answers the other question. `main.py` never calls
`donor._market_actions`, and never calls `donor._schedule_market_adjustment`
either. The network has to learn from reward alone that it cannot spend money it
does not have, that animals starve without WHEAT, that stock held past the last
day is worthless, and that the field policy idles if nobody buys seed.

The two share a benchmark — the unmodified pilkwang snapshot — so the pair is a
direct measurement of what the scaffolding is worth.

## What is fixed

`donor.py` is byte-for-byte identical to `agents/dqn_market_scorer/donor.py`, an
Apache-2.0 derivative of the pilkwang competitor snapshot; see `NOTICE` and
`LICENSE-pilkwang`, and `tests/test_dqn_market_free_v3c.py` pins the copy. The
derivative's Q-guidance parameter is **not used here**: `main.py` calls
`donor._role_plan(obs, farm)` with no guidance, so the field policy runs exactly
as upstream wrote it. The local copy exists so this submission bundle does not
depend on a `competitor_*` path or on the scorer.

The donor keeps animal placement, field jobs, routing, crop care, harvest, and
liquidation. `planting.py` is the one narrow adapter: seed inventory the market
model actually bought may fill *vacant or weed-covered* crop roles. Live plants
and every animal role stay the donor's.

## What is learned

`order_space.py` enumerates `ACTION_COUNT` structurally possible orders — one
flat list, index 0 is NOOP. Game-invalid product buys and inventory the field
policy can never create or consume are excluded, because they are not exploration.
Everything state-dependent — affordability, sale sizing, land timing, the feed
floor, terminal liquidation — is left for the network.

**One order per turn, 720 decisions an episode.** The earlier design factored the
action into ten independent slots over a shared candidate grid with total Q as
the sum; the TD target only ever saw that sum, so no slot got its own credit,
every slot's argmax picked the same order, and one good buy became a tenfold
spend. Measured over fourteen updates, every candidate produced that way finished
a 3,000-coin season near zero. A single order per turn makes it an ordinary
Q(s, a) again, and turn-scale decisions supply the throughput.

The state schema is the scorer's (`market_state.py`): prices, market inventory,
the shed, the herd, and the donor's own planning state — its seed deficit, animal
deficit, hand target, phase.

## What v3 added: a learned memory

The simulator's observation is a pure snapshot. It carries no price series, no
previous market inventory, and — the one that costs most — no sign that last
turn's order was refused. v2 answered that with 30 hand-built columns: a 24-turn
lagged difference over four series, a one-hot of the last op, and three refusal
counters. One lag, chosen in advance.

v3 deletes that block and puts a GRU in front of the head instead:

```
turn t:      state_t ──► GRU ──► h_t
decision k:  concat(h_t, state_t,k) ──► MLP ──► Q(199)
```

Three things about the shape are load-bearing:

- **The GRU ticks once per turn, not once per sub-decision.** A turn holds up to
  eleven orders, so a per-decision tick would make the sequence 8,000 steps long
  and its length a function of how much the policy currently trades. The turn is
  also the unit reward is paid on and the unit `n_step` counts.
- **The head reads the raw decision state as well as the latent.** That is what
  keeps the cold start gradable without routing the argument through the GRU
  (see `OrderQNetwork.quiet_start`), and what lets the intra-turn block reach the
  head unmediated.
- **The GRU's input is the same 280-column vector the head reads**, encoded once at the
  top of the turn with block E carrying the *previous* turn's finished
  `PendingTurn`. That is what replaces the refusal counters: the recurrence sees
  that an order was emitted and sees the money that followed, and differences the
  two itself.

Why this and not more history columns: the failures in
`docs/dqn_market_free_kaggle_episode_review.md` are all judgements about a
trajectory rather than a turn — capital sitting in stock instead of turning over,
money backloaded and never compounding, stock still in the shed at the buzzer.

## What v3c adds: the market, decomposed

The recurrence is real — zeroing the hidden state on a trained v3b checkpoint costs
33–36% of final money over 240 seeded episodes — and it is still not enough. At
matched batch count v2's hand-built history outscores v3b roughly two to one
(17,563 against 9,026 at batch 240).

The engine says why. `market_price` is a pure function of market inventory, with no
random term anywhere: prices move because somebody traded or because the town
bought. Measured over six stored replays — on a turn where neither seat traded,
prices move *only* on the town shop tick, and 71.6% of every (turn, product) cell
is predicted exactly by town drain alone. So a lagged price difference is three
quantities summed, of which two are known exactly, and the third — the opponent's
order flow — is the only thing in the market that is genuinely hidden at decision
time.

Block F (`market_state.market_flow_block`, 48 columns, 232 → 280) hands over the
split instead of the sum:

- **F1, 27.** Last turn's trade flow per product, this agent's own emitted units,
  and the same flow summed over a day. Inventory moved, minus the town's purchase,
  minus what this agent queued, is the opponent. Validated against the replays'
  *other* seat: exact on 86.2% of cells, with 9.8% of the rest a part-filled order.
  Fills are never reported by the engine, which is why own emitted stays its own
  column rather than being subtracted here.
- **F2, 3.** Turns to the shop tick and to the town-centre tick, and whether one
  lands next turn. Every unit of exogenous movement is on those two clocks.
- **F3, 18.** Units sellable before the quote crosses 0.9 and 0.75 of base — the
  impact curve inverted, which is the question a sell order actually asks, and the
  quantity the donor's sell rule walks to one unit at a time.

Block F reads the same SELL price curve block A already evaluates. Sharing that one
pass rather than repeating it is worth 30% of episode wall clock.

Training is R2D2: the collector stores the hidden state every
`collect.HIDDEN_STRIDE` turns, replay draws a contiguous chunk of
`burn_in + train + n_step` turns, a no-grad burn-in refreshes the stale stored
state, and the gradient flows over the training window only. `training/replay.py`
carries the chunk geometry and why the tail exists; `training/network.py` carries
why the recurrence is never noisy and why that decides the TD step's ordering.

The numpy mirror in `policy.py` is the half that ships. A recurrence that
disagrees with its torch twin compounds over 720 ticks rather than costing one
decision, so `GruCell.step` documents the two association orders that silently
break parity, and `tests/test_dqn_market_free_v3c.py` pins both against a
deliberately-wrong reference.

The cold start is `OrderQNetwork.zeros()`, which ties every action and so takes
index 0: an agent that never trades. Unlike the scorer's donor-default cold
start, this one is deliberately weak — it is the honest baseline for an
unscaffolded market.

## The two collapses

Both failure modes complete every episode without raising, so neither shows up as
an error. Watch them per batch in `metadata_json` and per update in
`metrics_json`:

| signal | meaning |
| --- | --- |
| `orders_per_decision` → 0 | the NOOP collapse: it stopped trading |
| `mean_final_money` → 0 | the bankruptcy collapse: it traded itself broke |

`agent.toml` gates both on release: money floors relative to the incumbent's own
seat in the same paired head-to-head, plus
`viability_min_orders_per_decision`, applied only once a checkpoint exists
because the cold start is silent by design. `diagnostics.py` and
`training/collect.py` record what separates the two from a healthy run.

## Training

```bash
python train_market_free_v3.py
python -m agents.dqn_market_free_v3c.cli model status --name recur_00
```

`recur_00` is one of the ten named learner slots (`recur_00` .. `recur_09`,
`SLOT_PREFIX` in `train_market_free_v3.py`), each owning the effective model ID
`dqn_market_free_v3c.raw_order_q_v1@recur_NN`. The v3 roster runs
`demonstration_episodes = 0`, so every slot is the reward-only arm: no pilkwang
demonstrations are pinned and no behaviour-clone phase runs. Turning them on pins
solvent pilkwang episodes and clones their order choices before the Double-DQN
fine-tuning.

Nothing writes `weights.npz` on its own. After reviewing a passed release gate,
export explicitly:

```bash
python -m agents.dqn_market_free_v3c.cli model export-active \
  --name recur_00 --output agents/dqn_market_free_v3c/weights.npz \
  --confirm-release
```

`weights.npz` is intentionally absent from `agent.toml`'s `code_paths` — the
league bundles this agent as an anchor long before any checkpoint exists, and a
declared-but-missing path fails that bundle. Inference finds the file by path
when it is there, and `export_submission.py` refuses to build a Kaggle archive
without it, since a bundle with no weights ships the silent network.

## Layout

| file | role |
| --- | --- |
| `donor.py` | field policy, run with default (upstream) guidance |
| `order_space.py` | the flat order list, argmax and epsilon-greedy selection |
| `market_state.py` | state features, shared schema with the scorer |
| `planting.py` | lets bought seed fill vacant donor crop roles |
| `policy.py` | numpy inference network (`weights.npz`) |
| `main.py` | the agent: donor field + raw learned orders |
| `diagnostics.py` | order-activity and solvency summaries |
| `evaluation.py` | `evaluate_episode` hook |
| `train.py` | the trainer the league calls on each closed round |
| `cli.py` | `model status` / `model export-active` |
| `export_submission.py` | builds `submission/` and the Kaggle `.tar.gz` |
| `match_agent.py` | training-only exploring wrapper that records decisions |
| `training/` | collector, replay, demonstrations, updater. Never bundled. |
