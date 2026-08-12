# dqn_market_scorer

One Q-network. The field policy is frozen; only the market is learned.

## Why

`dqn_job_scorer` trains two Q-networks off one scalar reward — a per-turn job net
and a daily market net. They share the credit for every dollar, so each one's
target is polluted by what the other did, and the variance shows up as
checkpoints that swing wildly between promotions. This agent removes the coupling
by freezing a field policy that already wins and learning only the market.

## What is frozen

`donor.py` is a byte-for-byte copy of the pilkwang competitor snapshot
(`agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py`),
redistributed under Apache-2.0 — see `NOTICE` and `LICENSE-pilkwang`. A test pins
its sha256 to the snapshot, so it cannot drift.

It was picked on measured strength *and* on structure. The four strongest
competitors are frozen 719-step tapes with field and market interleaved in one
recording: 158 tape `PLANT`s funded by exactly 18 tape `BUY_SEED` orders, and a
hand-roster length that tracks the tape's own hires. Replacing their market queue
desynchronises the field tape, so there is no policy to keep. pilkwang is a real
online policy whose dependency runs field → market and never back, and which
gates every market-supplied resource at runtime — `seeds.get(crop, 0) > 0` before
`PLANT` (`donor.py:918`), live `farm["hands"]` (`:1132`), live
`unlocked_quadrants` (`:499`). It degrades gracefully when the market buys
differently instead of desynchronising.

`main.py` calls `_role_plan` then `_unit_actions` unmodified. Only the market
layer is replaced.

## What is learned

The action is factored into `1 + len(PRODUCTS) = 10` factors, value-decomposed
(total Q is the sum over factors).

**Factor 0** picks a `MarketModifier` — six scalars that *modulate the donor's own
demand* rather than replacing it. Choosing crop targets independently would let
the network buy MELON while `_role_plan` intends to plant STRAWBERRY; scaling the
donor's stated deficit cannot.

| knob | levels | effect |
| --- | --- | --- |
| `seed_scale` | 0.0, 0.5, 1.0, 1.5 | scales `_seed_needs` counts |
| `hand_delta` | −4, −2, 0, +2 | offsets `_target_hands`, clamped to the donor's labour ceiling |
| `animal_gate` | block / donor | veto only — it can never grant permission the donor withheld |
| `land_bias` | defer / donor / now | shifts the `LAND_OPEN_DAYS` gate by ±days |
| `feed_days` | 2, 3, 4 | replaces `FEED_STOCK_DAYS` when sizing the wheat feed buy |
| `cash_floor` | ×0.5, ×1, ×2 | scales the seed and operating cash reserves |

38 curated rows, not the 864-row cross. **Row 0 is the donor default**, so a
zero-weight network scores everything equally, every argmax falls to index 0, and
the cold start plays unmodified pilkwang — a strong agent, not a random one. A
test asserts action-for-action equality over a full episode.

**Factors 1..9** pick a per-product sell reserve from `(0, 1, 2, 4, 8, 16)`,
clipped to the post-field shed. `market_exec.py` withholds it before the donor
sizes the sale, so the donor's price thresholds still apply to what is left: the
network can hold inventory but can never force a dump at a bad quote. Reserves
are ignored once the donor is liquidating.

Decisions are taken twice a day (hours 0 and 12), 60 per episode. The market net
is the only learner here, so it needs the samples; hour 0 covers the donor's
`hour <= 2` hiring window and hour 12 revisits the reserves.

## Training

```bash
kaglab league run --config configs/dqn_market_league.toml
python -m agents.dqn_market_scorer.cli model status
```

The trainer (`train.py`) collects rollout batches on each closed league round and
fits on the accumulated replay. The benchmark defaults to **the donor itself**:
with the field policy held identical on both sides, that match measures the
learned market layer and nothing else.

Rollout episodes land in `episodes`/`episode_players` under
`pairing_kind = 'market_rollout'`, so what the learner actually played is
answerable in SQL rather than only inferable from an artifact.

## Layout

| file | role |
| --- | --- |
| `donor.py` | frozen field policy, verbatim third-party copy |
| `market_exec.py` | derivative of `donor._market_actions` with the knobs threaded in |
| `market_policy.py` | state/action features, candidate enumeration, factored argmax |
| `policy.py` | numpy inference network (`weights.npz`) |
| `main.py` | the agent: donor field + learned market |
| `match_agent.py` | training-only exploring wrapper that records decisions |
| `training/` | collector, replay buffer, updater. Never bundled. |
