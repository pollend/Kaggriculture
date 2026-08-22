# dqn_market_free — where each layer stands against the public ladder

Session history is in [dqn_market_free_sessions.md](dqn_market_free_sessions.md),
the argument about what to build next in
[dqn_market_free_roadmap.md](dqn_market_free_roadmap.md), one unbuilt niche in
[dqn_market_free_animal_products.md](dqn_market_free_animal_products.md), and the
read of one submission's 24 public episodes in
[dqn_market_free_kaggle_episode_review.md](dqn_market_free_kaggle_episode_review.md).

This file answers a different question, and one worth re-answering whenever the
lineage moves: **is this agent competitive, and if not, which layer is the
deficit in.** It is a standings file, not a plan.

Measured 2026-08-21, against `market-free-splitladder-v1` at batch 126 and the
1,050-replay mirror in `data/kaggle_episodes` (top 50 a day by `avg_score`, 22
days, **148** distinct ladder agents, 2,100 seats -- the "31" in
`train_market_free_v3c.py` counts the demonstration pack, which filters to seats
over 50,000, not the mirror).

## The standings

| configuration | final money | how measured |
| --- | --- | --- |
| Public ladder, top-50 slice | **97,716** median | 2,100 seats; p10 58,399, p90 137,347 |
| Our field layer + **the donor's** market rules | **83,525** | 48 arena episodes, matched seeds, win 0.35 against pilkwang's 86,111 |
| Pilkwang's complete agent, against itself | 69,534 mean / 56,779 median | 24 arena episodes, both orientations |
| Best submission this account ever made (`landheur_08`) | 49,758 | 24 public episodes, won 10, public score 560.7 |
| **The learned market head, against pilkwang** | **10,634** | `gate_json` at cutoff 120, 200 episodes, won **0** |

Two readings, pointing opposite ways, and both are the point of this file.

**The scaffolding is ladder-grade.** Our field layer driving the donor's market
rules scores 83,525 — inside ~15% of the ladder median, and *above pilkwang's own
complete agent playing itself*. On the board, this agent farms like a contender.
That is recent: it is what removing `herd.herd_influenced_roles`' pass-two strip
bought (50,240 → 83,525, `market-free-cashherd-v1`).

**The learned market head is 8x behind its own donor on that same board.** 10,634
against 83,525. Every other deficit in this lineage is rounding error beside it.

## The rating scale, which does not rescue the money

| | value |
| --- | --- |
| Ladder top `avg_score`, last 5 days | 3,133 – 3,202 |
| Median of the top-50 slice | 2,862 – 3,010 |
| Our best public submission | **560.7** |

Money and rating track each other on the ladder — the top-50-by-score episodes
carry both seats at 90k-140k — so the rating gap is the money gap and not a
scoring artifact. Nothing here is lost to an Elo quirk.

## One number that was being read wrong

`model_updates.gate_json -> 'challenger_benchmarks' -> 0 -> 'b_money_mean'` reads
~123,661 for pilkwang. **That is pilkwang playing an opponent that leaves the
market alone**, which is what our head does. Against a real competitor — itself —
pilkwang makes 69,534. Quoting the gate figure as the benchmark's strength
inflates every gap estimate; it made a 5-6x deficit look like 12x and pointed the
work at the wrong layer for a while. When sizing a gap, score the benchmark
against a real competitor first.

## What the 2026-08-21 changes buy, and what they do not

Four defects were real and are fixed:

- `reward.OUTCOME_LOG_SCALE` 5.0 → 2.5, so the return tracks money (Pearson 0.701
  → 0.773, Spearman 0.888 → 0.947, win/lose separation +57%, on 300 re-scored
  episodes).
- `updater._large_margin_loss` replaces a cross-entropy over Q-values that had no
  fixed point and ran `max_abs_q` at 20-27 for a whole session.
- `order_space.stock_ceiling` and one SELL per item per turn, which stop the wire
  being spent on rungs that commit identically.
- `herd`'s pass-two strip, worth 33,285.

**None of them is evidence the learned head can close 8x.** The reward fix makes
money *learnable*; it does not show it gets learned. Three of the four are
scaffolding and objective repair. The lineage's own ceiling is the caution:
`herd-v1` reached 19,637 at batch 300, the best any session on this fork has
managed, and that is still 4x short of the donor's market rules on our own board.

## What would close it

The premise is the deficit. `agents/dqn_market_free_v3c/main.py` calls
`donor._unit_actions` for the field and then declines to call
`donor._market_actions` (`donor.py:1749`) — implemented, in the same module,
shipping in the same bundle, never invoked — and asks a network to rediscover it
from noise instead.

Every queued arm is a way to help it rediscover: demonstrations teach it to
imitate the donor, a donor opponent gives it a market to learn against, a
curriculum shapes the search. All of them pay the imitation gap.

**The residual arm does not.** Donor rules author the baseline order list; the
network accepts, vetoes and adds on top. It attaches where `commit_land` already
pre-commits an order into `PendingTurn` (`main.py:284-287`), so the loop, the
mask, the replay format, the reward and the action space are all unchanged and
only the starting `pending` differs. Floor: the 83,525 measured above. Upside:
whatever the head learns beyond it.

It is the one change that starts in ladder range rather than asking a 10,634
policy to become an 83,525 one. The costs are real and worth stating: it ends the
market_free premise, mints a new revision through `match_agent.toml` `code_paths`,
makes the v3/v3b/v3c lineage non-comparable, and stales any demonstration pack
rolled without it — a pack built for the full order list labels decisions the
network is no longer asked to make.

Recommended order: run `market-free-cashherd-v1` far enough to confirm the reward
change trains, then the residual arm, and treat demonstrations as the fallback
rather than the next step.

## How to re-run these numbers

The arena figures, per configuration, both orientations:

```python
from sim.arena import play_series
out = play_series(spec, donor_spec, range(1_000_000, 1_000_024),
                  {"episodeSteps": 720}, jobs=6)   # do NOT pass reduce=, it clears replay
```

Read `MatchResult.seats[seat].reward` — `final_money` is only populated by a
reducer and is `None` otherwise, which is easy to mistake for a failed episode.

The ladder figures walk `data/kaggle_episodes/<date>/<episode_id>.json.gz` and
pair each action with the **previous** step's observation:
`steps[i].observation` is the state *after* `steps[i].action`, and reading it
post-action inverts every affordability question.

The learned head's figures come from
`model_updates.gate_json -> 'challenger_benchmarks' -> 0` — `a_money_mean` is the
candidate, `b_money_mean` the benchmark, with `win_rate`, `margin_ci` and
`order_diagnostics` beside them.

## Caveats

- The 83,525 is against pilkwang only, over 48 episodes. The ladder mirror holds 148
  distinct agents and some will exploit differently; a single-opponent arena
  number is an upper bound on how well the configuration generalises.
- The ladder mirror is the **top 50 a day**, so 97,716 is the median of a strong
  slice and not of the ladder. The competition's own median `avg_score` across all
  episodes runs below the slice's.
- `landheur_08`'s 49,758 was scored against whatever the ladder paired it with,
  not against pilkwang, so it is not directly comparable to the rows above it.
- Money is a proxy for the leaderboard, not the leaderboard. Rating is Elo-ish
  over pairings.
