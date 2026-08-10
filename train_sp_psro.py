#!/usr/bin/env python3
"""Compatibility launcher: PSRO is paused in favor of fixed-opponent DQN."""

from __future__ import annotations

from train_fixed_dqn import main

if __name__ == "__main__":
    print(
        "[train_sp_psro] PSRO is paused; launching train_fixed_dqn.py. "
        "Use train_sp_psro_population.py to resume the original population run.",
        flush=True,
    )
    raise SystemExit(main())
