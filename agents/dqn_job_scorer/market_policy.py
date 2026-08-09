"""Factored daily Q-policy for purchases and inventory reserves.

The market action is intentionally not a raw ten-order tuple.  One factor
chooses a portfolio target and one factor per product chooses how many units to
retain.  The deterministic market executor turns those targets into legal,
affordable orders and owns survival and terminal overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from agents.dqn_job_scorer.hybrid import HybridPlan, choose_plan
from agents.dqn_job_scorer.rules import (
    ANIMAL_DATA,
    ANIMALS,
    CROP_DATA,
    CROPS,
    MARKET_I0,
    PRODUCT_BASE,
    PRODUCTS,
    SHOP_DEMAND,
    get,
)

MARKET_STATE_DIM = 91
MARKET_ACTION_DIM = 30
MARKET_FEATURE_DIM = MARKET_STATE_DIM + MARKET_ACTION_DIM
RESERVE_LEVELS = (0, 1, 2, 4, 8, 16)
LAND_PRICES = (1000, 2000, 4000)


@dataclass(frozen=True, slots=True)
class MarketCandidates:
    state: np.ndarray
    purchase_plans: tuple[HybridPlan, ...]
    purchase_static: np.ndarray
    reserve_values: tuple[tuple[int, ...], ...]
    reserve_static: np.ndarray
    reserve_mask: np.ndarray

    @property
    def factor_count(self) -> int:
        return 1 + len(PRODUCTS)


@dataclass(frozen=True, slots=True)
class MarketDecision:
    plan: HybridPlan
    chosen: tuple[int, ...]
    candidates: MarketCandidates


def _counts(farm: dict, private: dict) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    crops = {name: int((private.get("seeds", {}) or {}).get(name, 0)) for name in CROPS}
    animals = {name: int((private.get("shed", {}) or {}).get(name, 0)) for name in ANIMALS}
    structures = {"COOP": 0, "PASTURE": 0}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT" and tile.get("crop") in crops:
                crops[str(tile["crop"])] += 1
            animal = tile.get("animal")
            if animal in animals:
                animals[str(animal)] += 1
            if tile.get("kind") in structures and not animal:
                structures[str(tile["kind"])] += 1
    for slot in private.get("inventories", []) or []:
        for name in ANIMALS:
            animals[name] += int((slot or {}).get(name, 0))
    return crops, animals, structures


def market_state_features(obs, farm: dict, private: dict, episode_steps: int, turns_per_day: int) -> np.ndarray:
    step = int(get(obs, "step", 0))
    day = int(get(obs, "day", step // max(1, turns_per_day)))
    hour = int(get(obs, "hour", step % max(1, turns_per_day)))
    total_days = max(1.0, episode_steps / max(1, turns_per_day))
    money = float(farm.get("money", 0))
    player = int(get(obs, "player", 0))
    farms = get(obs, "farms", []) or []
    opponent_money = max(
        (float(other.get("money", 0)) for i, other in enumerate(farms) if i != player),
        default=0.0,
    )
    private_seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    carried = {item: 0 for item in (*PRODUCTS, *ANIMALS)}
    for slot in private.get("inventories", []) or []:
        for item in carried:
            carried[item] += int((slot or {}).get(item, 0))
    market = get(obs, "market", {}) or {}
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}
    crops, animals, structures = _counts(farm, private)
    planted = empty = weeds = 0
    opponent_crops = {name: 0 for name in CROPS}
    opponent_animals = {name: 0 for name in ANIMALS}
    pending = {item: 0 for item in PRODUCTS}
    for i, other in enumerate(farms):
        for row in other.get("tiles", []) or []:
            for tile in row:
                if i == player:
                    if tile is None:
                        empty += 1
                    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                        weeds += 1
                    elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                        planted += 1
                        crop = tile.get("crop")
                        if crop in pending:
                            pending[str(crop)] += int(tile.get("yield_units", 0))
                    if isinstance(tile, dict) and tile.get("animal") in ANIMAL_DATA:
                        animal = str(tile["animal"])
                        pending[str(ANIMAL_DATA[animal]["product"])] += int(tile.get("yield_units", 0))
                        pending["FERTILIZER"] += int(bool(tile.get("fertilizer_available")))
                elif isinstance(tile, dict):
                    if tile.get("kind") == "PLANT" and tile.get("crop") in opponent_crops:
                        opponent_crops[str(tile["crop"])] += 1
                    if tile.get("animal") in opponent_animals:
                        opponent_animals[str(tile["animal"])] += 1
    demand = {item: 0 for item in PRODUCTS}
    for shop in (get(obs, "town", {}) or {}).get("unlocked_shops", []) or []:
        for item in SHOP_DEMAND.get(shop, ()):
            if item in demand:
                demand[item] += 1
    shed_total = sum(int(value) for value in shed.values())
    values = [
        day / total_days,
        hour / max(1, turns_per_day),
        step / max(1, episode_steps),
        max(0, episode_steps - step) / max(1, episode_steps),
        np.log1p(max(0.0, money)) / 10.0,
        money / 10_000.0,
        opponent_money / 10_000.0,
        money / (money + opponent_money + 1.0),
    ]
    values += [int(private_seeds.get(item, 0)) / 10.0 for item in CROPS]
    values += [int(shed.get(item, 0)) / 50.0 for item in PRODUCTS]
    values += [carried[item] / 25.0 for item in PRODUCTS]
    values += [carried[item] / 10.0 for item in ANIMALS]
    values += [float(prices.get(item, PRODUCT_BASE[item])) / PRODUCT_BASE[item] for item in PRODUCTS]
    values += [(float(inventory.get(item, MARKET_I0)) - MARKET_I0) / 1000.0 for item in PRODUCTS]
    values += [
        len(farm.get("hands", []) or []) / 3.0,
        int(farm.get("hires_today", 0)) / 10.0,
        planted / 100.0,
        empty / 100.0,
        weeds / 100.0,
        len(farm.get("unlocked_quadrants", []) or ["NW"]) / 4.0,
        shed_total / 100.0,
        max(0, 100 - shed_total) / 100.0,
    ]
    values += [animals[item] / 10.0 for item in ANIMALS]
    values += [structures[item] / 10.0 for item in ("COOP", "PASTURE")]
    values += [opponent_crops[item] / 25.0 for item in CROPS]
    values += [opponent_animals[item] / 10.0 for item in ANIMALS]
    values += [pending[item] / 25.0 for item in PRODUCTS]
    values += [demand[item] / 4.0 for item in PRODUCTS]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (MARKET_STATE_DIM,):
        raise AssertionError(f"MARKET_STATE_DIM is {MARKET_STATE_DIM}, built {result.shape[0]}")
    return result


def _purchase_static(plan: HybridPlan, current_land: int, money: float) -> np.ndarray:
    row = np.zeros(MARKET_ACTION_DIM, dtype=np.float32)
    row[0] = 1.0
    row[2 + (0 if plan.crop not in CROPS else 1 + CROPS.index(plan.crop))] = 1.0
    animal_cursor = 2 + 1 + len(CROPS)
    row[animal_cursor + (0 if plan.target_animals <= 0 else 1 + ANIMALS.index(plan.animal))] = 1.0
    product_cursor = animal_cursor + 1 + len(ANIMALS)
    cursor = product_cursor + len(PRODUCTS)
    row[cursor:cursor + 8] = (
        plan.target_hands / 3.0,
        plan.target_crops / 24.0,
        plan.target_animals / 8.0,
        plan.wheat_reserve / 20.0,
        plan.target_land / 4.0,
        max(0, plan.target_land - current_land),
        min(money, 10_000.0) / 10_000.0,
        1.0,
    )
    return row


def _reserve_static(product_name: str, reserve: int, available: int) -> np.ndarray:
    row = np.zeros(MARKET_ACTION_DIM, dtype=np.float32)
    row[1] = 1.0
    animal_cursor = 2 + 1 + len(CROPS)
    product_cursor = animal_cursor + 1 + len(ANIMALS)
    row[product_cursor + PRODUCTS.index(product_name)] = 1.0
    cursor = product_cursor + len(PRODUCTS)
    row[cursor + 5] = reserve / 100.0
    row[cursor + 6] = reserve / max(1.0, float(available))
    row[cursor + 7] = 1.0
    return row


def enumerate_market_candidates(obs, farm: dict, private: dict, episode_steps: int = 720, turns_per_day: int = 24) -> MarketCandidates:
    heuristic = choose_plan(obs, farm, None, episode_steps, turns_per_day)
    crops, animals, _ = _counts(farm, private)
    current_crops = sum(crops.values())
    current_animals = sum(animals.values())
    current_hands = len(farm.get("hands", []) or [])
    current_land = len(farm.get("unlocked_quadrants", []) or ["NW"])
    step = int(get(obs, "step", 0))
    days_left = max(0.0, (episode_steps - step) / max(1, turns_per_day))
    viable_crops = [name for name in CROPS if days_left >= float(CROP_DATA[name]["first"])]
    viable_animals = [name for name in ANIMALS if days_left >= float(ANIMAL_DATA[name]["first"]) + 2.0]
    plans: list[HybridPlan] = [heuristic]
    hold = HybridPlan(
        "CROP" if current_animals == 0 else "MIXED",
        heuristic.animal,
        current_animals,
        current_crops,
        max(current_animals, heuristic.wheat_reserve),
        heuristic.crop,
        current_hands,
        current_land,
    )
    plans.append(hold)
    for hands in range(current_hands, 4):
        units = 1 + hands
        for crop, density in product(viable_crops, (3, 6)):
            plans.append(HybridPlan("CROP", heuristic.animal, current_animals, max(current_crops, density * units), max(current_animals, 0), crop, hands, current_land))
        for animal, density, feed_days in product(viable_animals, (1, 2), (1, 2)):
            target_animals = max(current_animals, density * units)
            plans.append(HybridPlan("ANIMAL", animal, target_animals, max(current_crops, 2 * units), target_animals * feed_days, heuristic.crop, hands, current_land))
    for crop, animal in product(viable_crops, viable_animals):
        units = 1 + max(current_hands, min(3, heuristic.target_hands))
        plans.append(HybridPlan("MIXED", animal, max(current_animals, units), max(current_crops, 3 * units), max(current_animals, units) * 2, crop, max(current_hands, heuristic.target_hands), current_land))
    if current_land < 4:
        plans.append(HybridPlan(heuristic.mode, heuristic.animal, heuristic.target_animals, heuristic.target_crops, heuristic.wheat_reserve, heuristic.crop, heuristic.target_hands, current_land + 1))
    plans = list(dict.fromkeys(plans))
    purchase_static = np.stack([_purchase_static(plan, current_land, float(farm.get("money", 0))) for plan in plans])
    shed = private.get("shed", {}) or {}
    reserve_values: list[tuple[int, ...]] = []
    reserve_rows = np.zeros((len(PRODUCTS), len(RESERVE_LEVELS) + 1, MARKET_ACTION_DIM), dtype=np.float32)
    reserve_mask = np.zeros((len(PRODUCTS), len(RESERVE_LEVELS) + 1), dtype=bool)
    for item_index, item in enumerate(PRODUCTS):
        available = max(0, int(shed.get(item, 0)))
        values = tuple(dict.fromkeys((*[min(available, level) for level in RESERVE_LEVELS], available)))
        reserve_values.append(values)
        for value_index, value in enumerate(values):
            reserve_rows[item_index, value_index] = _reserve_static(item, value, available)
            reserve_mask[item_index, value_index] = True
    return MarketCandidates(
        market_state_features(obs, farm, private, episode_steps, turns_per_day),
        tuple(plans),
        purchase_static,
        tuple(reserve_values),
        reserve_rows,
        reserve_mask,
    )


def assemble_market(state: np.ndarray, action_static: np.ndarray) -> np.ndarray:
    shape = action_static.shape[:-1] + (state.shape[0],)
    return np.concatenate((np.broadcast_to(state, shape), action_static), axis=-1).astype(np.float32, copy=False)


def choose_market(candidates: MarketCandidates, network, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> MarketDecision:
    purchase_q = network(assemble_market(candidates.state, candidates.purchase_static))
    reserve_q = network(assemble_market(candidates.state, candidates.reserve_static)).reshape(
        candidates.reserve_static.shape[:2]
    )
    explore = epsilon > 0.0 and rng is not None and rng.random() < epsilon
    purchase = int(rng.integers(len(candidates.purchase_plans))) if explore else int(np.argmax(purchase_q))
    chosen = [purchase]
    reserves: list[int] = []
    for item_index, values in enumerate(candidates.reserve_values):
        index = int(rng.integers(len(values))) if explore else int(np.argmax(reserve_q[item_index, :len(values)]))
        chosen.append(index)
        reserves.append(values[index])
    base = candidates.purchase_plans[purchase]
    plan = HybridPlan(base.mode, base.animal, base.target_animals, base.target_crops, base.wheat_reserve, base.crop, base.target_hands, base.target_land, tuple(reserves))
    return MarketDecision(plan, tuple(chosen), candidates)
