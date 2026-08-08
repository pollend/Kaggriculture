# Configurable Crop Surrogate

This is strategy attempt **#1**.

It uses one deterministic crop scheduler whose labor, crop-selection, watering,
harvesting, selling, routing, and market-awareness parameters are configurable.
The experiment pipeline generates many profiles, evaluates them in staged
tournaments, and trains Random Forest surrogate models to propose the next
batch.

## Entry points

- `main.py` — balanced default and submission candidate
- `variants/aggressive.py` — labor-heavy, early-harvest profile
- `variants/conservative.py` — cash-preserving, low-labor profile
- `variants/market_timer.py` — price-aware profile
- `profiles/*.json` — validated strategy parameters used by those entry points
- `agent.toml` — family identity, behavior-code fingerprint inputs, training adapter,
  and post-episode evaluator
- `train.py` — lab-facing `on_round` / `on_batch` trainer object
- `training.py` — mutation, crossover, surrogate fitting, and held-out gating
- `evaluation.py` — `main.py` hook that derives training metrics from DuckDB replays

Run a named variant directly from the repository root:

```bash
python -m sim.arena match agents/configurable_crop_surrogate/main.py starter --seeds 10
python -m sim.arena league \
  agents/configurable_crop_surrogate/main.py \
  agents/configurable_crop_surrogate/variants/aggressive.py \
  agents/configurable_crop_surrogate/variants/market_timer.py \
  starter --seeds 10
```

For a quick persistent training pass:

```bash
.venv/bin/kaglab league init \
  --config configs/league_smoke.toml --session-id crop-smoke
.venv/bin/kaglab league run --session-id crop-smoke
.venv/bin/kaglab league leaderboard --session-id crop-smoke
.venv/bin/kaglab league evaluate --session-id crop-smoke --force
```

Each completed league generation is registered as one surrogate training batch by
`train.py`. `league run` checks the update threshold automatically, so after two new
generations it fits a challenger and promotes it only when its held-out MAE is no worse
than the active surrogate. `python -m agents.configurable_crop_surrogate.cli model
update` is the manual recovery equivalent, not a required extra step after every
generation; use `model status` to inspect its batch count and active checkpoint.
Generation creation uses the active checkpoint and falls back to mutation/crossover
before the first promotion.

The batches, update watermark, candidates, and active checkpoint are stored under the
model ID in the selected DuckDB catalog. Resuming or starting another session against
that catalog reuses them; a separate database starts a separate learning history.

The configurable agent keeps routing targets and action counters in an `EpisodeState`
owned by that callable. It resets when step zero starts a new game; persistent learning
state lives in DuckDB and immutable checkpoint artifacts instead of module globals.
