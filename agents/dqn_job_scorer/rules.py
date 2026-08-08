"""Game facts this agent needs. Deliberately a private copy.

Every agent owns its own transcription of the rules rather than sharing one, so
an agent directory is self-contained and bundles without dragging in the rest of
the repo. The cost is that these tables exist in more than one place;
`tests/test_game_facts.py` asserts every copy against the values documented in
`README.md`, so a typo in one copy fails loudly instead of drifting.

Source: the Object Types and Price Function tables in README.md.
"""

from __future__ import annotations

from typing import Any

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")

# "yield" is the unfertilized peak, which is what this agent plans against.
CROP_DATA: dict[str, dict[str, Any]] = {
    "WHEAT": {"seed": 10, "base": 25, "first": 2, "peak": 4, "yield": 4, "ongoing": False},
    "CARROT": {"seed": 20, "base": 35, "first": 2, "peak": 3, "yield": 3, "ongoing": False},
    "TOMATO": {"seed": 50, "base": 60, "first": 8, "peak": 11, "yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "base": 120, "first": 10, "peak": 16, "yield": 4, "ongoing": True},
    "MELON": {"seed": 80, "base": 250, "first": 10, "peak": 12, "yield": 6, "ongoing": False},
}

SHOP_DEMAND: dict[str, set[str]] = {
    "BAKERY": {"WHEAT"},
    "PIZZA_SHOP": {"WHEAT", "TOMATO"},
    "BRUNCH_SPOT": {"WHEAT", "STRAWBERRY"},
    "ICE_CREAM_SHOP": {"WHEAT", "STRAWBERRY"},
    "PET_CAFE": {"CARROT"},
    "SMOOTHIE_SHOP": {"STRAWBERRY"},
    "FARMERS_MARKET": {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY"},
}

# Every product starts the season at this market inventory; price is `base` here.
MARKET_I0 = 10_000.0

# Opposite corners of the default 10x10 board.
MAX_MANHATTAN = 18.0

MOVE_FOR_DELTA = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}


def get(mapping: Any, key: str, default: Any = None) -> Any:
    """Read from an observation that may be a dict or a kaggle Struct."""
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan distance. The board has no impassable tiles -- locked quadrants
    are passable -- so this is the exact travel cost, and no pathfinding exists."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def shed_tiles(board_size: int) -> tuple[tuple[int, int], ...]:
    """The four centre tiles adjacent to the shed. The shed is not itself a tile."""
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def step_toward(position: tuple[int, int], target: tuple[int, int]) -> list[str]:
    """One move toward the target.

    Horizontal first: deterministic, and it gets a hand off the locked east-side
    spawn tile quickly.
    """
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    delta = (1 if dx > 0 else -1, 0) if dx else (0, 1 if dy > 0 else -1)
    return [MOVE_FOR_DELTA[delta]]
