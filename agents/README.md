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

Two deliberate, licensed exceptions exist: `dqn_market_scorer` and its
`dqn_market_free` ablation each ship a local `donor.py` derived from the pilkwang
snapshot, so that neither submission bundle depends on a `competitor_*` path. The
two copies are byte-for-byte identical to each other — not to the snapshot, which
they modify to accept Q-selected expansion gates. `dqn_market_scorer` uses those
gates; `dqn_market_free` passes the defaults and so runs the field policy
unchanged. Each carries the upstream Apache-2.0 licence (`LICENSE-pilkwang`) and a
`NOTICE` recording provenance and derivative changes; tests pin the copies to one
another and to the digests in those notices. No `competitor_*` path is ever
bundled.
