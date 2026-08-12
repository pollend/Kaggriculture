# Receding Horizon Planner

Strategy attempt **#3**. This agent searches a three-day aggregate economic
horizon, then jointly assigns the current field tasks to the farmer and hands.
It covers crops, geese, cows, sheep, structures, feed, care, fertilizer, land,
labor, market price impact, and terminal liquidation.

## Architecture

1. `planner.py` runs a 64-state beam search and produces today's portfolio
   targets. It replans on day boundaries and important world-state changes.
2. `scheduler.py` expands those targets into the currently executable task
   frontier and uses rectangular Hungarian assignment across all units.
3. `market.py` uses the same plan for budgeted purchases, feed reserves,
   price-aware sales, hiring, and land acquisition.
4. `strategy.py` contains the parameters evolved by the persistent league.
5. `train.py` records one batch per closed generation and delegates checkpoint
   fitting to the random-forest surrogate in `training.py`.

The inference package is standard library only. Training code imports pandas,
scikit-learn, and the experiment lab, but is absent from `code_paths` and every
submission archive.

## Local evaluation

```bash
python -m sim.arena match \
  agents/receding_horizon_planner/main.py starter --seeds 20
```

## Parameter tuning

The separate constants-driven runner labels its sessions `planner` and uses globally
unique model IDs in the shared PostgreSQL database:

```bash
python train_planner.py
```

Edit the constants at the top of `train_planner.py` to change the population,
rounds, generations, episode length, parallel jobs, candidate pool, final arena,
or environment synchronization. The league automatically tries the model's
`on_batch` update after each `on_round` hook; two closed generations are needed
before the first surrogate checkpoint can be trained. Re-running the script with the
same session ID and database resumes its hook ledger, batches, update watermark, and
active checkpoint. Pointing it at another database intentionally starts a new learning
history.

Inspect or manually retry the planner's sole model without affecting other families:

```bash
python -m agents.receding_horizon_planner.cli model status
python -m agents.receding_horizon_planner.cli model update
```

## Submission

```bash
python agents/receding_horizon_planner/export_submission.py
```

This recreates `agents/receding_horizon_planner/submission/` and writes
`agents/receding_horizon_planner/receding_horizon_planner_submission.tgz` with
a top-level `main.py` and the exact inference allowlist from `agent.toml`.
