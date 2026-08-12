# Mixed-Farm Random-Forest Market

This experiment holds the configurable mixed-farm portfolio and field scheduler
fixed by snapshotting them while learning the economic market policy. One forest chooses daily seed,
animal, wheat-reserve, and land targets.  A second forest chooses product reserves
whenever the shed contains sellable inventory.

The deterministic executor still owns emergency feed, final liquidation, hiring,
affordability, season viability, board capacity, and the ten-order cap.

The field policy it holds fixed is `Strategy()` — the committed defaults in
`strategy.py`, exposed as `training.FROZEN_STRATEGY`. Nothing is inherited from
another family, so no other session has to exist before this one runs. The warm-start
batch is collected by watching the bare agent play, which falls back to
`heuristic_market`, and that same heuristic is the baseline the promotion gate scores
against: identical field policy, hand-written market orders versus learned ones.

Start or resume the trainer-only experiment once the PostgreSQL lab stack is up:

```bash
uv run python train_rf_market.py
```

Run a promoted checkpoint directly with:

```text
agents.mixed_farm_rf_market.agent:make_agent?weights=artifacts/.../checkpoint.joblib
```

The checkpoint is a versioned Joblib bundle containing the frozen mixed-farm
strategy, purchase forest, sale forest, and dataset identity.
