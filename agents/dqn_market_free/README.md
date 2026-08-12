# dqn_market_free

One Q-network over a flat order space. The donor's market layer is deleted, not
modulated — this is the unscaffolded counterpart to `dqn_market_scorer`.

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
`LICENSE-pilkwang`, and `tests/test_dqn_market_free.py` pins the copy. The
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
python train_market_free.py
python -m agents.dqn_market_free.cli model status --name simple_v1
```

`simple_v1` is a named training instance owning the effective model ID
`dqn_market_free.raw_order_q_v1@simple_v1`. It pins solvent pilkwang
demonstrations and clones their order choices before Double-DQN fine-tuning; the
bare manifest (`demonstration_episodes = 0`) is the reward-only control.

Nothing writes `weights.npz` on its own. After reviewing a passed release gate,
export explicitly:

```bash
python -m agents.dqn_market_free.cli model export-active \
  --name simple_v1 --output agents/dqn_market_free/weights.npz \
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
| `export_submission.py` | builds `submission/` and the Kaggle `.tgz` |
| `match_agent.py` | training-only exploring wrapper that records decisions |
| `training/` | collector, replay, demonstrations, updater. Never bundled. |
