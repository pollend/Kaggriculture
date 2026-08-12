"""Market-state encoder for the unscaffolded market learner.

This is the state half of the scorer's feature schema, kept local so the free
agent and its bundled donor share one module instance and one set of planning
semantics. The action encoding is intentionally not copied: this agent scores
raw simulator orders instead of donor-market modifiers.
"""

from __future__ import annotations

import numpy as np

from agents.dqn_market_free import donor

CROPS: tuple[str, ...] = tuple(donor.CROPS)
ANIMALS: tuple[str, ...] = tuple(donor.ANIMALS)
PRODUCTS: tuple[str, ...] = tuple(donor.PRODUCTS)
PHASES: tuple[str, ...] = ("BOOTSTRAP", "COMPOUND", "REALIZE", "CRISIS", "LIQUIDATE")

MARKET_STATE_DIM = 117


def _owned_animals(farm: dict, private: dict) -> dict[str, int]:
    placed = donor._farm_animal_counts(farm)
    return {
        animal: int(placed.get(animal, 0)) + int(donor._private_item_total(private, animal))
        for animal in ANIMALS
    }


def _structures(farm: dict) -> dict[str, int]:
    counts = {"COOP": 0, "PASTURE": 0}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") in counts and "animal" not in tile:
                counts[str(tile["kind"])] += 1
    return counts


def _tile_census(obs, player: int) -> tuple[int, int, int, dict[str, int], dict[str, int], dict[str, int]]:
    planted = empty = weeds = 0
    opponent_crops = {name: 0 for name in CROPS}
    opponent_animals = {name: 0 for name in ANIMALS}
    pending = {item: 0 for item in PRODUCTS}
    for index, other in enumerate(obs.get("farms", []) or []):
        for row in other.get("tiles", []) or []:
            for tile in row:
                if index == player:
                    if tile is None:
                        empty += 1
                    elif isinstance(tile, dict) and tile.get("kind") == "WEED":
                        weeds += 1
                    elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                        planted += 1
                        crop = tile.get("crop")
                        if crop in pending:
                            pending[str(crop)] += int(tile.get("yield_units", 0) or 0)
                    if isinstance(tile, dict) and tile.get("animal") in donor.ANIMALS:
                        rule = donor.ANIMALS[str(tile["animal"])]
                        pending[str(rule["product"])] += int(tile.get("yield_units", 0) or 0)
                        pending["FERTILIZER"] += int(bool(tile.get("fertilizer_available")))
                elif isinstance(tile, dict):
                    if tile.get("kind") == "PLANT" and tile.get("crop") in opponent_crops:
                        opponent_crops[str(tile["crop"])] += 1
                    if tile.get("animal") in opponent_animals:
                        opponent_animals[str(tile["animal"])] += 1
    return planted, empty, weeds, opponent_crops, opponent_animals, pending


def market_state_features(
    obs,
    farm: dict,
    private: dict,
    roles: dict,
    field: dict,
    *,
    config=None,
    episode_steps: int = 720,
    turns_per_day: int = 24,
) -> np.ndarray:
    """Public market state plus the donor's own planning state.

    The donor block is the point of the whole design: the network is choosing
    how much of the donor's stated demand to fund, so it has to see that demand.
    """

    step = int(obs.get("step", 0) or 0)
    day = int(obs.get("day", step // max(1, turns_per_day)) or 0)
    hour = int(obs.get("hour", step % max(1, turns_per_day)) or 0)
    total_days = max(1.0, episode_steps / max(1, turns_per_day))
    money = float(farm.get("money", 0) or 0)
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", []) or []
    opponent_money = max(
        (float(other.get("money", 0) or 0) for index, other in enumerate(farms) if index != player),
        default=0.0,
    )
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    carried = {item: 0 for item in (*PRODUCTS, *ANIMALS)}
    for slot in private.get("inventories", []) or []:
        for item in carried:
            carried[item] += int((slot or {}).get(item, 0) or 0)
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}
    planted, empty, weeds, opponent_crops, opponent_animals, pending = _tile_census(obs, player)
    owned = _owned_animals(farm, private)
    structures = _structures(farm)
    shed_total = sum(max(0, int(value or 0)) for value in shed.values())

    values = [
        day / total_days,
        hour / max(1, turns_per_day),
        step / max(1, episode_steps),
        max(0, episode_steps - step) / max(1, episode_steps),
        float(np.log1p(max(0.0, money))) / 10.0,
        money / 10_000.0,
        opponent_money / 10_000.0,
        money / (money + opponent_money + 1.0),
    ]
    values += [int(seeds.get(item, 0) or 0) / 10.0 for item in CROPS]
    values += [int(shed.get(item, 0) or 0) / 50.0 for item in PRODUCTS]
    values += [carried[item] / 25.0 for item in PRODUCTS]
    values += [carried[item] / 10.0 for item in ANIMALS]
    values += [
        float(prices.get(item, donor.MARKET[item][0]) or donor.MARKET[item][0]) / float(donor.MARKET[item][0])
        for item in PRODUCTS
    ]
    values += [
        (float(inventory.get(item, donor.MARKET_I0) or donor.MARKET_I0) - donor.MARKET_I0) / 1000.0
        for item in PRODUCTS
    ]
    values += [
        len(farm.get("hands", []) or []) / float(donor.MAX_HANDS),
        int(farm.get("hires_today", 0) or 0) / float(donor.MAX_HANDS),
        planted / 100.0,
        empty / 100.0,
        weeds / 100.0,
        len(farm.get("unlocked_quadrants", []) or ["NW"]) / 4.0,
        shed_total / 100.0,
        max(0, 100 - shed_total) / 100.0,
    ]
    values += [owned[item] / 10.0 for item in ANIMALS]
    values += [structures[item] / 10.0 for item in ("COOP", "PASTURE")]
    values += [opponent_crops[item] / 25.0 for item in CROPS]
    values += [opponent_animals[item] / 10.0 for item in ANIMALS]
    values += [pending[item] / 25.0 for item in PRODUCTS]
    values += [donor._town_demand_per_day(obs, item) / 24.0 for item in PRODUCTS]

    # --- donor planning state -------------------------------------------------
    needs = donor._seed_needs(obs, farm, private, roles)
    values += [int(needs.get(item, 0)) / 10.0 for item in CROPS]
    role_targets = {animal: 0 for animal in ANIMALS}
    for kind, item in roles.values():
        if kind == "ANIMAL" and item in role_targets:
            role_targets[item] += 1
    values += [max(0, role_targets[item] - owned[item]) / 8.0 for item in ANIMALS]
    values.append(donor._target_hands(obs, farm, private, roles) / float(donor.MAX_HANDS))

    summary = donor._survey(farm, private, roles, day)
    phase = donor._policy_phase(obs, farm, private, summary)
    values += [1.0 if phase == name else 0.0 for name in PHASES]
    values += [
        summary["shed_load"] / 100.0,
        summary["carried_load"] / 100.0,
        summary["at_risk_animals"] / 8.0,
        summary["at_risk_crops"] / 16.0,
        summary["open_structures"] / 8.0,
        summary["structures_todo"] / 16.0,
        summary["plantable"] / 32.0,
        summary["unfed"] / 16.0,
    ]

    shed_capacity = int(donor._cfg(config, "shedCapacity", donor.SHED_CAPACITY))
    post_shed, _ = donor._post_field_storage(private, field, shed_capacity)
    occupancy = sum(max(0, int(value or 0)) for value in post_shed.values())
    values += [
        occupancy / max(1.0, float(shed_capacity)),
        max(0, shed_capacity - occupancy) / max(1.0, float(shed_capacity)),
    ]
    values += [
        max(0, donor.TOTAL_DAYS - day) / float(donor.TOTAL_DAYS),
        1.0 if field.get("liquidation") else 0.0,
    ]

    result = np.asarray(values, dtype=np.float32)
    if result.shape != (MARKET_STATE_DIM,):
        raise AssertionError(f"MARKET_STATE_DIM is {MARKET_STATE_DIM}, built {result.shape[0]}")
    return result

