# Configurable Mixed-Farm Surrogate

Strategy attempt **#4**. This deterministic policy expands the configurable crop
surrogate into the full farm economy: crops, geese, cows, sheep, structures,
feed, care, fertilizer, land, labor, market impact, and terminal liquidation.

Unlike the receding-horizon planner, it does not perform beam search. A greedy
remaining-season portfolio chooses asset targets and a deadline-aware scheduler
turns them into executable field and market work. Every economic preference is
represented by the versioned `Strategy` schema and can be evolved by the
persistent league.

## Local evaluation

```bash
python -m sim.arena match \
  agents/configurable_mixed_farm_surrogate/main.py \
  agents/competitor_prvsiyan_kaggriculture_frontier_the_moon_counts_melons/main.py \
  --seeds 20
```

## Isolated training

```bash
python train_mixed_farm.py
python -m agents.configurable_mixed_farm_surrogate.cli model status \
  --name v3
```

The runner labels its sessions `mixed_farm_v3` and owns a separate artifact namespace. Surrogate
promotion changes only the active training checkpoint; shipping a selected
strategy still requires an explicit reviewed update to `profiles/balanced.json`.
Schema v2 makes the operating limits themselves learnable: crop floor and herd
support, opening crop and animal tranches, the livestock unlock point, labor
capacity, and the land-utilization trigger are Random Forest input features.

## Progression report

The report reads the live league database without modifying it and writes PNG,
SVG, CSV, JSON, and Markdown artifacts:

```bash
python -m agents.configurable_mixed_farm_surrogate.progression \
  --session-id configurable-mixed-farm-v3-training \
  --output artifacts/configurable_mixed_farm_surrogate/progression_v3
```

It plots population results by generation and decompresses the selected
champion's matched Moon replays for per-day labor, assets, money, and action
utilization. The latest completed generation supplies the default champion;
partial generations are shown only as progression data.
