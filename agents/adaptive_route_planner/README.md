# Adaptive Route Planner

This is strategy attempt **#2**: a trainable, observation-driven planner based
on the strongest operational ideas visible in the imported Tetsutani competitor
snapshot. It keeps season-scale goals for labor, land, livestock, crops, and
liquidation, but schedules every worker from current state instead of replaying
a fixed action trace.

The default profile is deliberately route-shaped: early wheat and livestock,
land expansion near days 7 and 11, a large daily workforce, strawberry and
melon scaling, price-impact-aware sales, and a terminal cash conversion phase.
Evolution may tune every goal and priority through `Strategy`.

Training uses `configs/adaptive_route_league.toml`. Model administration remains
family-specific:

```bash
python -m agents.adaptive_route_planner.cli model status
python -m agents.adaptive_route_planner.cli model update
```

## Attribution

The phase structure, actor-local disruption recovery, sale-slot price-impact
ordering, and terminal-value principles were derived from tetsu2131's
"Adaptive Farming Strategy for Kaggriculture" snapshot in
`agents/competitor_tetsutani_adaptive_farming_strategy_for_kaggriculture/`,
licensed Apache-2.0. This family does not embed or replay that competitor's
compressed 719-action trajectory.

