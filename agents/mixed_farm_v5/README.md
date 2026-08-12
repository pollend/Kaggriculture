# Mixed-Farm Surrogate v5

Strategy attempt **#5**. This deterministic policy expands the configurable crop
surrogate into the full farm economy: crops, geese, cows, sheep, structures,
feed, care, fertilizer, land, labor, market impact, and terminal liquidation.

Unlike the receding-horizon planner, it does not perform beam search. A greedy
remaining-season portfolio chooses asset targets and a deadline-aware scheduler
turns them into executable field and market work. Every economic preference is
represented by the versioned `Strategy` schema and can be evolved by the
persistent league.

The `_v5` modules are the controller. `strategy.py` and `portfolio.py` are the
older economics they still delegate to — `strategy_v5.Strategy.to_legacy()`
reuses that validation and sampling prior, and `portfolio_v5.build_plan()` calls
into the greedy remaining-season targets. The v4 controller, market, and profile
loader are gone, along with the frozen v4 acceptance baseline: v4 was not
competitive.

V5 values crop, animal, feed, land, and sale intents before creating tasks. Labor
is then sized only for that accepted backlog, including travel and setup work.
There is no minimum-hand floor, timed land expansion, `target_hands`, or asset ↔
labor feedback loop. Normal crews stop at 10; hands 11–16 require twice the
normal marginal ROI. Every hire must leave the configured cash reserve intact,
fit the 5–20% daily hire budget, cover its exact Fibonacci cost, have at least
four useful turns left, and have unassigned profitable work. Emergency feed may
cross the reserve; discretionary assets and labor may not.

## Local evaluation

```bash
python -m sim.arena match \
  agents/mixed_farm_v5/main.py \
  agents/competitor_prvsiyan_kaggriculture_frontier_the_moon_counts_melons/main.py \
  --seeds 20
```

## Isolated training

```bash
python train_mixed_farm.py
python -m agents.mixed_farm_v5.cli model status \
  --name v5
```

The runner labels its session `mixed_farm_v5` and owns a separate artifact
namespace. Generation zero is drawn entirely from the bounded prior — v3/v4
champions are no longer inherited, since they were tuned against a labor schema
v5 discards and the lineage was not competitive. Every generation then gives
every live policy paired coverage against all twelve imported competitors, plus
separate population and historical-policy rounds.

Surrogate promotion changes only the active training checkpoint; the shipped
default is whatever `strategy_v5.Strategy()` declares, and changing it is an
explicit reviewed edit. Schema v4 discards the v3/v4 minimum-labor and timed
land fields and replaces coarse workload sizing with bounded utilization, labor
ROI, and hire-spend preferences. The Random Forest also receives exact
Fibonacci labor cost, capacity, workload, reserve, portfolio, and normalized
preference features. It proposes nothing until it has at least 60 rows, core
normalized validation error at most 0.75, and productive/idle MAE at most 0.08.
Proposals must predict at least 20% productive hand actions and at most 30% idle
hand actions. Moon Counts Melons is an equal-weight anchor and final
no-regression gate, not a specially weighted training target.

## Progression report

The report reads the live league database without modifying it and writes PNG,
SVG, CSV, JSON, and Markdown artifacts:

```bash
python -m agents.mixed_farm_v5.progression \
  --session-id configurable-mixed-farm-v5-training \
  --output artifacts/mixed_farm_v5/progression_v5
```

It plots population results by generation and decompresses the selected
champion's matched Moon replays for per-day labor, assets, money, and action
utilization. The latest completed generation supplies the default champion;
partial generations are shown only as progression data.
