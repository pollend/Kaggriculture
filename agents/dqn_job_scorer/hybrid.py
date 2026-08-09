"""Deterministic economic regime and portfolio controls around the DQN."""

from __future__ import annotations

from dataclasses import dataclass

from agents.dqn_job_scorer.rules import ANIMAL_DATA, ANIMALS, CROP_DATA, PRODUCT_BASE, SHOP_DEMAND, get

MODES = ("CROP", "MIXED", "ANIMAL")


@dataclass(frozen=True, slots=True)
class HybridPlan:
    mode: str
    animal: str
    target_animals: int
    target_crops: int
    wheat_reserve: int
    crop: str = "WHEAT"
    target_hands: int = 3
    target_land: int = 1
    sell_reserves: tuple[int, ...] = ()


def _town_weights(obs) -> dict[str, int]:
    weights = {item: 0 for item in PRODUCT_BASE}
    for shop in (get(obs, "town", {}) or {}).get("unlocked_shops", []) or []:
        for item in SHOP_DEMAND.get(shop, ()):
            weights[item] += 1
    return weights


def _opponent_supply(obs, player: int) -> dict[str, int]:
    supply = {item: 0 for item in PRODUCT_BASE}
    for index, farm in enumerate(get(obs, "farms", []) or []):
        if index == player:
            continue
        for row in farm.get("tiles", []) or []:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                if tile.get("kind") == "PLANT" and tile.get("crop") in supply:
                    supply[str(tile["crop"])] += 1
                animal = tile.get("animal")
                if animal in ANIMAL_DATA:
                    supply[str(ANIMAL_DATA[str(animal)]["product"])] += 1
    return supply


def choose_plan(obs, farm: dict, previous_mode: str | None = None, episode_steps: int = 720, turns_per_day: int = 24) -> HybridPlan:
    """Choose a stable portfolio from public economics, never hidden state."""
    player = int(get(obs, "player", 0))
    step = int(get(obs, "step", 0))
    days_left = max(0.0, (episode_steps - step) / max(1, turns_per_day))
    prices = (get(obs, "market", {}) or {}).get("prices", {}) or {}
    demand = _town_weights(obs)
    opponent = _opponent_supply(obs, player)

    crop_scores: dict[str, float] = {}
    for crop, data in CROP_DATA.items():
        if days_left < float(data["first"]):
            crop_scores[crop] = -1e9
            continue
        price = float(prices.get(crop, data["base"]))
        revenue = price * float(data["yield"]) * (1.0 + 0.18 * demand[crop])
        crop_scores[crop] = (revenue - float(data["seed"])) / (3.0 + float(data["peak"])) / (1.0 + 0.08 * opponent[crop])

    animal_scores: dict[str, float] = {}
    wheat_price = float(prices.get("WHEAT", PRODUCT_BASE["WHEAT"]))
    for animal, data in ANIMAL_DATA.items():
        product = str(data["product"])
        if days_left < float(data["first"]) + 2.0:
            animal_scores[animal] = -1e9
            continue
        cycles = 1 + int((days_left - float(data["first"])) // float(data["interval"]))
        price = float(prices.get(product, data["base"]))
        revenue = cycles * 2.0 * price * (1.0 + 0.18 * demand[product])
        feed = days_left * wheat_price
        animal_scores[animal] = (revenue - feed - float(data["cost"])) / max(1.0, days_left * 3.0) / (1.0 + 0.08 * opponent[product])

    best_crop_score = max(crop_scores.values())
    animal = max(ANIMALS, key=lambda name: (animal_scores[name], name))
    best_animal_score = animal_scores[animal]
    if best_animal_score >= 1.15 * max(0.01, best_crop_score):
        proposed = "ANIMAL"
    elif best_crop_score >= 1.15 * max(0.01, best_animal_score):
        proposed = "CROP"
    else:
        proposed = "MIXED"

    # Hysteresis: only dislodge an established pure regime for a 10% advantage.
    if previous_mode == "ANIMAL" and best_animal_score * 1.10 >= best_crop_score:
        proposed = previous_mode
    elif previous_mode == "CROP" and best_crop_score * 1.10 >= best_animal_score:
        proposed = previous_mode

    units = 1 + len(farm.get("hands", []) or [])
    cap = max(1, min(2 * units, int(0.60 * 24 * units / 3.0)))
    target_animals = 0 if proposed == "CROP" else min(cap, units if proposed == "MIXED" else 2 * units)
    target_crops = 6 * units if proposed == "CROP" else 3 * units if proposed == "MIXED" else 2 * units
    crop = max(CROP_DATA, key=lambda name: (crop_scores[name], name))
    return HybridPlan(
        proposed,
        animal,
        target_animals,
        target_crops,
        target_animals * 2,
        crop,
        3,
        len(farm.get("unlocked_quadrants", []) or ["NW"]),
    )
