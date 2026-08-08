"""Reproducible stand-ins for the built-in kaggriculture agents.

`kaggriculture.random_agent` builds a fresh unseeded `random.Random()` on every
call, so seeding the global module does not make it reproducible and two runs of
the same seed diverge. Anything that needs a stochastic-but-repeatable opponent
uses `make_seeded_random` instead, reached through the ordinary spec syntax:

    kaggriculture_lab.opponents:make_seeded_random?seed=7

This lives outside `sim` because it is knowledge about one game's agents, not
about running matches.
"""

from __future__ import annotations

import random
from collections.abc import Callable


def make_seeded_random(config: dict | int | None = None) -> Callable[[dict], dict]:
    """Reproducible stand-in for the built-in `random` agent.

    Mirrors `kaggriculture.random_agent` exactly, but draws from a single
    `random.Random` seeded once, so an episode replays identically.

    Takes the spec config dict as a single positional argument, which is the
    factory contract `sim.registry.resolve` uses. An int is accepted too, so
    it stays pleasant to call directly from Python.
    """
    if isinstance(config, dict):
        seed = int(config.get("seed", 0))
    else:
        seed = int(config or 0)
    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

    rng = random.Random(seed)
    farmer_ops = ["NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "PASS"]

    def agent(obs):
        farms = obs.get("farms", [])
        player = obs.get("player", 0)
        private = obs.get("private", {}) or {}
        farm = farms[player] if farms and player < len(farms) else None
        if farm is None:
            return {"farmer": ["PASS"], "hands": [], "market": []}

        market = []
        seeds = private.get("seeds", {})

        affordable = [c for c in CROPS if CROPS[c]["seed"] <= farm["money"]]
        if affordable and rng.random() < 0.1:
            market.append(["BUY_SEED", rng.choice(affordable), 1])

        available = [c for c, n in seeds.items() if n > 0]
        if available and rng.random() < 0.3:
            farmer = ["PLANT", rng.choice(available)]
        else:
            farmer = [rng.choice(farmer_ops)]

        hands = [[rng.choice(farmer_ops)] for _ in farm.get("hands", [])]
        return {"farmer": farmer, "hands": hands, "market": market}

    return agent
