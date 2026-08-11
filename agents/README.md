# Agent catalog

Folders without a `competitor_` prefix are strategies developed in this repository.
Folders beginning with `competitor_` are fixed third-party Kaggle notebook snapshots.
Each competitor directory contains its generated `main.py`, original notebook, raw
Kaggle metadata, provenance hashes, and a league `agent.toml` manifest.

Run the fixed competitors in the direct round-robin arena with:

```bash
.venv/bin/python -m sim.arena league \
  agents/configurable_crop_surrogate/main.py \
  agents/competitor_*/main.py \
  starter --seeds 10
```

The persistent league loads the same snapshots as non-trainable anchors from
`configs/league.toml`. `competitor_*` files are deliberately absent from `kaglab bundle`.

One exception, deliberate and licensed: `dqn_market_scorer` ships a byte-for-byte copy
of the pilkwang snapshot as its own `donor.py`, because it freezes that field policy and
learns only the market layer on top. The copy carries the upstream Apache-2.0 licence
(`LICENSE-pilkwang`) and a `NOTICE` recording provenance and the derivative changes, and
a test pins its sha256 to the snapshot. No `competitor_*` path is ever bundled.
