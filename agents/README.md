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
`configs/league.toml`. Competitor files are deliberately absent from `kaglab bundle`.
