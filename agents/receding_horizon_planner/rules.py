"""Submission-local Kaggriculture constants and geometry helpers."""

from __future__ import annotations

import math
from typing import Any

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_ORDER = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")

CROP_DATA = {
    "WHEAT": {
        "seed": 10,
        "base": 25,
        "first": 2,
        "peak": 4,
        "last": 4,
        "yield": 4,
        "max": 6,
        "interval": 0,
        "ongoing": False,
    },
    "CARROT": {
        "seed": 20,
        "base": 35,
        "first": 2,
        "peak": 3,
        "last": 3,
        "yield": 3,
        "max": 4,
        "interval": 0,
        "ongoing": False,
    },
    "TOMATO": {
        "seed": 50,
        "base": 60,
        "first": 8,
        "peak": 11,
        "last": 11,
        "yield": 4,
        "max": 4,
        "interval": 1,
        "ongoing": True,
    },
    "STRAWBERRY": {
        "seed": 100,
        "base": 120,
        "first": 10,
        "peak": 16,
        "last": 16,
        "yield": 4,
        "max": 4,
        "interval": 2,
        "ongoing": True,
    },
    "MELON": {
        "seed": 80,
        "base": 250,
        "first": 10,
        "peak": 12,
        "last": 12,
        "yield": 6,
        "max": 6,
        "interval": 0,
        "ongoing": False,
    },
}

ANIMAL_DATA = {
    "GOOSE": {"cost": 300, "structure": "COOP", "first": 4, "interval": 1, "max": 4, "product": "EGG", "base": 50},
    "COW": {"cost": 400, "structure": "PASTURE", "first": 8, "interval": 2, "max": 6, "product": "MILK", "base": 160},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first": 6, "interval": 3, "max": 6, "product": "WOOL", "base": 200},
}

LAND_ORDER = ("NE", "SW", "SE")
LAND_PRICES = (1000, 2000, 4000)
MARKET_I0 = 10_000
PRICE_FLOOR = 1

MARKET_PARAMS = {
    "WHEAT": {
        "base": 25,
        "I0": MARKET_I0,
        "T": 400,
        "below_func": "sqrt",
        "below_target": 0.80,
        "above_func": "log",
        "above_target": 0.20,
    },
    "CARROT": {
        "base": 35,
        "I0": MARKET_I0,
        "T": 450,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sqrt",
        "above_target": 0.70,
    },
    "TOMATO": {
        "base": 60,
        "I0": MARKET_I0,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "sqrt",
        "above_target": 0.60,
    },
    "STRAWBERRY": {
        "base": 120,
        "I0": MARKET_I0,
        "T": 100,
        "below_func": "sqrt",
        "below_target": 0.70,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "MELON": {
        "base": 250,
        "I0": MARKET_I0,
        "T": 300,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.60,
    },
    "EGG": {
        "base": 50,
        "I0": MARKET_I0,
        "T": 332,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "log",
        "above_target": 0.20,
    },
    "MILK": {
        "base": 160,
        "I0": MARKET_I0,
        "T": 122,
        "below_func": "sqrt",
        "below_target": 0.60,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "WOOL": {
        "base": 200,
        "I0": MARKET_I0,
        "T": 105,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.20,
    },
    "FERTILIZER": {
        "base": 100,
        "I0": MARKET_I0,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "linear",
        "above_target": 0.40,
    },
}

SHOP_DEMAND = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def shed_tiles(board_size: int) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def quadrant(x: int, y: int, board_size: int) -> str:
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def unlocked_empty_tiles(farm: dict) -> list[tuple[int, int]]:
    unlocked = set(farm.get("unlocked_quadrants") or ())
    tiles = farm.get("tiles") or []
    size = len(tiles)
    return [
        (x, y)
        for y, row in enumerate(tiles)
        for x, tile in enumerate(row)
        if tile is None and quadrant(x, y, size) in unlocked
    ]


def step_toward(position: tuple[int, int], target: tuple[int, int]) -> list[str]:
    x, y = position
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _shape(name: str, value: float) -> float:
    value = max(0.0, value)
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    return value


def market_parameters(obs: Any, item: str) -> dict[str, Any]:
    defaults = MARKET_PARAMS[item]
    overrides = ((get(obs, "market", {}) or {}).get("params", {}) or {}).get(item, {}) or {}
    return {**defaults, **overrides}


def market_price(item: str, inventory: float, obs: Any = None) -> int:
    params = market_parameters(obs, item) if obs is not None else MARKET_PARAMS[item]
    base = float(params["base"])
    initial = float(params["I0"])
    capacity = float(params["T"])
    if inventory < initial:
        name = str(params["below_func"])
        amplitude = float(params["below_target"]) * base / _shape(name, capacity)
        price = base + amplitude * _shape(name, initial - inventory)
    else:
        name = str(params["above_func"])
        amplitude = float(params["above_target"]) * base / _shape(name, capacity)
        price = base - amplitude * _shape(name, inventory - initial)
    return max(PRICE_FLOOR, int(round(price)))


def next_land(farm: dict) -> tuple[str, int] | None:
    unlocked = set(farm.get("unlocked_quadrants") or ())
    for name, price in zip(LAND_ORDER, LAND_PRICES):
        if name not in unlocked:
            return name, price
    return None


def hire_cost(hires_today: int, multiplier: int = 1) -> int:
    left = right = 1
    for _ in range(max(0, hires_today)):
        left, right = right, left + right
    return multiplier * left
