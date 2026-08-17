# dqn_market_scorer

One Q-network. A donor-derived field policy and its market executor share one
bounded economic decision.

## Policy kinds

The model ID is `dqn_market_scorer.joint_residual_q_v1`, and `donor` is not used
as a catch-all role. SQL and model metadata distinguish:

| kind | responsibility |
| --- | --- |
| `base_market_planner` | constructs coherent market demand from the donor rules |
| `joint_residual_q` | learned scorer selecting field/market modifiers and sell reserves |
| `q_guided_field_policy` | donor field work with learned expansion gates |
| `promotion_baseline` | unmodified donor opponent used by the release gate |
| `training_opponent` | curriculum opponent in exploratory rollouts |
| `fixed_benchmark` | additional non-regression opponent |

## Why

`dqn_job_scorer` trains two Q-networks off one scalar reward — a per-turn job net
and a daily market net. They share the credit for every dollar, so each one's
target is polluted by what the other did, and the variance shows up as
checkpoints that swing wildly between promotions. This agent removes the second
network: one economic Q decision guides donor field expansion and market actions.

## Donor-derived field policy

`donor.py` is an Apache-2.0 derivative of the pilkwang competitor snapshot
(`agents/competitor_pilkwang_kaggriculture_structured_economic_policy/main.py`),
with its upstream and derivative digests recorded in `NOTICE`. The modification
lets Q-selected expansion gates flow into `_role_plan`; see `LICENSE-pilkwang`.

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

`main.py` builds a provisional donor plan, selects the Q modifier, and recomputes
the role plan on the same turn. A blocked gate never abandons existing crops or
animals; it only prevents new expansion.

## What is learned

The action is factored into `1 + len(PRODUCTS) = 10` factors, value-decomposed
(total Q is the sum over factors).

**Factor 0** picks a `MarketModifier` — six scalars that *modulate the donor's own
field expansion and demand* rather than replacing its coherent plan. The same
choice reaches `_role_plan` and `market_exec`, so field work and purchases agree.

| knob | levels | effect |
| --- | --- | --- |
| `seed_scale` | 0.0, 0.5, 1.0, 1.5 | gates new crop roles and scales `_seed_needs` counts |
| `hand_delta` | −4, −2, 0, +2 | offsets `_target_hands`, clamped to the donor's labour ceiling |
| `animal_gate` | block / donor | gates new herd roles and purchases while retaining existing animals |
| `land_bias` | defer / donor / now | shifts the `LAND_OPEN_DAYS` gate by ±days |
| `feed_days` | 2, 3, 4 | replaces `FEED_STOCK_DAYS` when sizing the wheat feed buy |
| `cash_floor` | ×0.5, ×1, ×2 | scales the seed and operating cash reserves |

38 curated rows, not the 864-row cross. **Row 0 is the donor default**, so the
zero-Q cold start scores everything equally, every argmax falls to index 0, and
plays unmodified pilkwang — a strong agent, not a random one. Its hidden trunk
is initialized normally and only its output layer is zeroed, preserving live
gradients. A test asserts action-for-action equality over a full episode.

**Factors 1..9** pick a per-product sell reserve from `(0, 1, 2, 4, 8, 16)`,
clipped to the post-field shed. `market_exec.py` withholds it before the donor
sizes the sale, so the donor's price thresholds still apply to what is left: the
network can hold inventory but can never force a dump at a bad quote. Reserves
are ignored once the donor is liquidating.

Decisions are taken twice a day (hours 0 and 12), 60 per episode. The joint net
is the only learner; hour 0 covers expansion and hiring, while hour 12 revisits
field expansion and reserves.

## Training

```bash
python train_market_scorer.py
python -m agents.dqn_market_scorer.cli model status --name joint03
```

`joint03` is a named training instance. It owns the effective model ID
`dqn_market_scorer.joint_residual_q_v1@joint03`, so its batches, checkpoints and
self-play history are its own. Nothing has been trained under it yet — the
database holds no rows for this agent at all. After reviewing a passed release
gate, export it explicitly:

```bash
python -m agents.dqn_market_scorer.cli model export-active \
  --name joint03 --output agents/dqn_market_scorer/weights.npz \
  --confirm-release
```

The trainer (`train.py`) collects rollout batches on each closed league round and
fits on the accumulated replay. The benchmark defaults to **the unmodified donor
itself**, so the paired margin measures the complete Q-guided residual against
the policy it derives from.

Rollout episodes land in `episodes`/`episode_players` under
`pairing_kind = 'market_rollout'`, so what the learner actually played is
answerable in SQL rather than only inferable from an artifact.

## Layout

| file | role |
| --- | --- |
| `donor.py` | donor-derived field policy with Q-guided role expansion |
| `market_exec.py` | derivative of `donor._market_actions` with the knobs threaded in |
| `market_policy.py` | state/action features, candidate enumeration, factored argmax |
| `policy.py` | numpy inference network (`weights.npz`) |
| `main.py` | the agent: donor field + learned market |
| `evaluation.py` | `evaluate_episode` hook: margin trend and re-evaluation flag |
| `train.py` | the trainer the league calls on each closed round |
| `cli.py` | `model status` / `model export-active` for a named instance |
| `export_submission.py` | builds `submission/` and the Kaggle `.tgz` from `code_paths` |
| `match_agent.py` | training-only exploring wrapper that records decisions |
| `training/` | collector, replay buffer, updater. Never bundled. |
