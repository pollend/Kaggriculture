# Mixed-Farm Random-Forest Market

This experiment holds the configurable mixed-farm portfolio and field scheduler
fixed by snapshotting them while learning the economic market policy. One forest chooses daily seed,
animal, wheat-reserve, and land targets.  A second forest chooses product reserves
whenever the shed contains sellable inventory.

The deterministic executor still owns emergency feed, final liquidation, hiring,
affordability, season viability, board capacity, and the ten-order cap.

Start or resume the trainer-only experiment after the PostgreSQL lab stack and at
least one mixed-farm v4 generation are available:

```bash
uv run python train_rf_market.py
```

Run a promoted checkpoint directly with:

```text
agents.mixed_farm_rf_market.agent:make_agent?weights=artifacts/.../checkpoint.joblib
```

The checkpoint is a versioned Joblib bundle containing the frozen mixed-farm
strategy, purchase forest, sale forest, and dataset identity.
