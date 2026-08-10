"""Submission-local game constants and geometry for the mixed-farm agent."""

from __future__ import annotations

from typing import Any

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")

CROP_DATA: dict[str, dict[str, int | bool]] = {
    "WHEAT": {"seed": 10, "base": 25, "first": 2, "peak": 4, "last": 4, "yield": 4, "max": 6, "interval": 0, "ongoing": False},
    "CARROT": {"seed": 20, "base": 35, "first": 2, "peak": 3, "last": 3, "yield": 3, "max": 4, "interval": 0, "ongoing": False},
    "TOMATO": {"seed": 50, "base": 60, "first": 8, "peak": 11, "last": 11, "yield": 4, "max": 4, "interval": 1, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "base": 120, "first": 10, "peak": 16, "last": 16, "yield": 4, "max": 4, "interval": 2, "ongoing": True},
    "MELON": {"seed": 80, "base": 250, "first": 10, "peak": 12, "last": 12, "yield": 6, "max": 6, "interval": 0, "ongoing": False},
}

ANIMAL_DATA: dict[str, dict[str, int | str]] = {
    "GOOSE": {"cost": 300, "structure": "COOP", "first": 4, "interval": 1, "max": 4, "product": "EGG", "base": 50},
    "COW": {"cost": 400, "structure": "PASTURE", "first": 8, "interval": 2, "max": 6, "product": "MILK", "base": 160},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first": 6, "interval": 3, "max": 6, "product": "WOOL", "base": 200},
}

LAND_PRICES = (1000, 2000, 4000)
SHOP_DEMAND = {
    "BAKERY": {"WHEAT"},
    "PIZZA_SHOP": {"WHEAT", "TOMATO"},
    "BRUNCH_SPOT": {"WHEAT", "STRAWBERRY", "EGG", "MILK"},
    "ICE_CREAM_SHOP": {"STRAWBERRY", "MILK"},
    "PET_CAFE": {"CARROT", "EGG"},
    "SMOOTHIE_SHOP": {"STRAWBERRY"},
    "FARMERS_MARKET": set(PRODUCTS) - {"FERTILIZER"},
    "YARN_STORE": {"WOOL"},
}

MOVE_FOR_DELTA = {(0, -1): "NORTH", (0, 1): "SOUTH", (1, 0): "EAST", (-1, 0): "WEST"}


def get(mapping: Any, key: str, default: Any = None) -> Any:
    return mapping.get(key, default) if isinstance(mapping, dict) else getattr(mapping, key, default)


def distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def step_toward(position: tuple[int, int], target: tuple[int, int]) -> list[str]:
    dx, dy = target[0] - position[0], target[1] - position[1]
    delta = (1 if dx > 0 else -1, 0) if dx else (0, 1 if dy > 0 else -1)
    return [MOVE_FOR_DELTA[delta]]


def shed_tiles(board_size: int) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def nearest_shed(position: tuple[int, int], board_size: int) -> tuple[int, int]:
    return min(shed_tiles(board_size), key=lambda tile: (distance(position, tile), tile))


def unlocked_empty_tiles(farm: dict) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y, row in enumerate(farm.get("tiles", []) or [])
        for x, tile in enumerate(row)
        if tile is None
    ]


def count_assets(farm: dict) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    crops = {name: 0 for name in CROPS}
    animals = {name: 0 for name in ANIMALS}
    structures = {"COOP": 0, "PASTURE": 0}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT" and tile.get("crop") in crops:
                crops[str(tile["crop"])] += 1
            if tile.get("kind") in structures:
                structures[str(tile["kind"])] += 1
            if tile.get("animal") in animals:
                animals[str(tile["animal"])] += 1
    return crops, animals, structures


def town_counts(obs: Any) -> dict[str, int]:
    counts = {item: 0 for item in PRODUCTS}
    for shop in (get(obs, "town", {}) or {}).get("unlocked_shops", []) or []:
        for item in SHOP_DEMAND.get(shop, ()):
            counts[item] += 1
    return counts


def opponent_supply(obs: Any, player: int) -> dict[str, int]:
    counts = {item: 0 for item in PRODUCTS}
    farms = get(obs, "farms", []) or []
    for index, farm in enumerate(farms):
        if index == player:
            continue
        for row in farm.get("tiles", []) or []:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                crop = tile.get("crop")
                if crop in counts:
                    counts[str(crop)] += 1
                animal = tile.get("animal")
                if animal in ANIMAL_DATA:
                    counts[str(ANIMAL_DATA[str(animal)]["product"])] += 1
    return counts


def hire_cost(hires_today: int, multiplier: int = 1) -> int:
    left, right = 1, 1
    for _ in range(max(0, hires_today)):
        left, right = right, left + right
    return left * multiplier

