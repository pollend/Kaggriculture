"""One transcription of the rules, checked against every agent's private copy.

Agents deliberately do not share code, so each keeps its own crop table. That is
a real choice, not sloppiness -- but it is only safe if a typo in one copy fails
here instead of quietly costing a season. The table below is transcribed from
the Object Types and Price Function tables in `README.md`; if the competition
changes the rules, this is the one place to update, and every agent's copy will
fail until it is brought in line.

An agent that legitimately models a crop differently should be excluded here
explicitly, with a comment saying why -- not by loosening the check.
"""

from __future__ import annotations

import pytest

# crop -> (seed cost, base price, first yield day, max yield day, ongoing)
README_CROPS = {
    "WHEAT": (10, 25, 2, 4, False),
    "CARROT": (20, 35, 2, 3, False),
    "TOMATO": (50, 60, 8, 11, True),
    "STRAWBERRY": (100, 120, 10, 16, True),
    "MELON": (80, 250, 10, 12, False),
}

# Unfertilized peak yield, which is what both agents plan against. The README's
# "Max Yield" column lists the *fertilized* cap for wheat and carrot (6 and 4),
# reachable only with FERTILIZE, so the planning values differ on purpose.
README_UNFERTILIZED_YIELD = {
    "WHEAT": 4,
    "CARROT": 3,
    "TOMATO": 4,
    "STRAWBERRY": 4,
    "MELON": 6,
}

# Every agent module that keeps a private CROP_DATA copy.
CROP_TABLES = [
    "agents.configurable_crop_surrogate.agent",
    "agents.dqn_job_scorer.rules",
    "agents.receding_horizon_planner.rules",
]


def _crop_data(module_path: str) -> dict:
    import importlib

    return importlib.import_module(module_path).CROP_DATA


@pytest.mark.parametrize("module_path", CROP_TABLES)
def test_agent_crop_table_matches_the_readme(module_path):
    table = _crop_data(module_path)
    assert set(table) == set(README_CROPS), f"{module_path} has the wrong crop set"

    for crop, (seed, base, first, peak, ongoing) in README_CROPS.items():
        entry = table[crop]
        assert entry["seed"] == seed, f"{module_path}:{crop} seed cost"
        assert entry["base"] == base, f"{module_path}:{crop} base price"
        assert entry["first"] == first, f"{module_path}:{crop} first yield day"
        assert entry["peak"] == peak, f"{module_path}:{crop} max yield day"
        assert bool(entry["ongoing"]) is ongoing, f"{module_path}:{crop} ongoing flag"
        assert entry["yield"] == README_UNFERTILIZED_YIELD[crop], f"{module_path}:{crop} yield"


def test_every_copy_agrees_with_every_other():
    """Catches two copies drifting together away from the README."""
    tables = {path: _crop_data(path) for path in CROP_TABLES}
    reference_path, reference = next(iter(tables.items()))
    for path, table in tables.items():
        for crop in README_CROPS:
            for key in ("seed", "base", "first", "peak", "yield", "ongoing"):
                assert table[crop][key] == reference[crop][key], (
                    f"{path} disagrees with {reference_path} on {crop}.{key}"
                )


def test_board_geometry_matches_across_agents():
    """The shed sits at the four centre tiles; a wrong copy strands every unit."""
    from agents.configurable_crop_surrogate.agent import _shed_tiles
    from agents.dqn_job_scorer.rules import shed_tiles
    from agents.receding_horizon_planner.rules import shed_tiles as planner_shed_tiles

    for board_size in (10, 8, 6):
        assert shed_tiles(board_size) == _shed_tiles(board_size)
        assert planner_shed_tiles(board_size) == _shed_tiles(board_size)
    assert shed_tiles(10) == ((4, 4), (5, 4), (4, 5), (5, 5))
