"""Factored economic Q-policy layered on the donor-derived field policy.

The action is not a raw ten-order tuple, and it is not a portfolio target
either. The donor already computes what it wants -- seed counts from its role
plan, a hand target from its job backlog, an animal deficit from its herd plan.
Choosing those independently would let the network buy MELON while the field
policy intends to plant STRAWBERRY. So the learned action *modulates the donor's
own demand and field expansion*: one factor picks a `MarketModifier` (which new
crop and herd roles to activate, and how much demand to fund), and one factor
per product picks how many units to retain from the shed before the donor's sale
sizing runs.

Total Q is the sum over the 1 + len(PRODUCTS) factors, which is how
`training/updater.py` trains it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from agents.dqn_market_scorer import donor

CROPS: tuple[str, ...] = tuple(donor.CROPS)
ANIMALS: tuple[str, ...] = tuple(donor.ANIMALS)
PRODUCTS: tuple[str, ...] = tuple(donor.PRODUCTS)
PHASES: tuple[str, ...] = ("BOOTSTRAP", "COMPOUND", "REALIZE", "CRISIS", "LIQUIDATE")

# Stable semantic kinds for the four policies that are easy to conflate when
# they all contain the word "donor" in prose. These values are persisted in
# revision roles, training-batch metadata, and update gates; keep them distinct.
POLICY_KINDS: dict[str, str] = {
    "base_market_planner": "base_market_planner",
    "market_learner": "joint_residual_q",
    "field_policy": "q_guided_field_policy",
    "promotion_baseline": "promotion_baseline",
    "training_opponent": "training_opponent",
    "fixed_benchmark": "fixed_benchmark",
}

RESERVE_LEVELS = (0, 1, 2, 4, 8, 16)

MARKET_STATE_DIM = 117
MARKET_ACTION_DIM = 20
MARKET_FEATURE_DIM = MARKET_STATE_DIM + MARKET_ACTION_DIM


@dataclass(frozen=True, slots=True)
class MarketModifier:
    """Scalars threaded into donor role planning and the market executor.

    The all-default instance reproduces the donor exactly, which is what makes
    an untrained network play pilkwang rather than noise.
    """

    seed_scale: float = 1.0
    hand_delta: int = 0
    animal_gate: int = 1  # 0 vetoes animal purchases, 1 leaves the donor gate alone
    land_bias: int = 0  # -1 defers the land gate, 0 donor, +1 opens it early
    feed_days: int = donor.FEED_STOCK_DAYS
    cash_floor: float = 1.0


DEFAULT_MODIFIER = MarketModifier()

SEED_SCALES = (0.0, 0.5, 1.0, 1.5)
HAND_DELTAS = (-4, -2, 0, 2)
ANIMAL_GATES = (0, 1)


def enumerate_modifiers() -> tuple[MarketModifier, ...]:
    """The bounded candidate grid, donor default first.

    Row 0 must stay the default: a zero-weight network scores every row equally
    and `select` breaks ties to the lowest index.
    """

    rows: list[MarketModifier] = [DEFAULT_MODIFIER]
    for seed_scale in SEED_SCALES:
        for hand_delta in HAND_DELTAS:
            for animal_gate in ANIMAL_GATES:
                candidate = MarketModifier(
                    seed_scale=seed_scale, hand_delta=hand_delta, animal_gate=animal_gate
                )
                if candidate != DEFAULT_MODIFIER:
                    rows.append(candidate)
    for land_bias in (-1, 1):
        rows.append(replace(DEFAULT_MODIFIER, land_bias=land_bias))
    for feed_days in (2, 4):
        rows.append(replace(DEFAULT_MODIFIER, feed_days=feed_days))
    for cash_floor in (0.5, 2.0):
        rows.append(replace(DEFAULT_MODIFIER, cash_floor=cash_floor))
    return tuple(rows)


MODIFIERS = enumerate_modifiers()


@dataclass(frozen=True, slots=True)
class MarketCandidates:
    state: np.ndarray  # float32 [MARKET_STATE_DIM]
    modifiers: tuple[MarketModifier, ...]
    purchase_static: np.ndarray  # float32 [P, MARKET_ACTION_DIM]
    purchase_mask: np.ndarray  # bool    [P]
    reserve_values: tuple[tuple[int, ...], ...]
    reserve_static: np.ndarray  # float32 [len(PRODUCTS), R, MARKET_ACTION_DIM]
    reserve_mask: np.ndarray  # bool    [len(PRODUCTS), R]

    @property
    def factor_count(self) -> int:
        return 1 + len(PRODUCTS)


@dataclass(frozen=True, slots=True)
class MarketDecision:
    modifier: MarketModifier
    reserves: dict[str, int]
    chosen: tuple[int, ...]
    candidates: MarketCandidates


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


_PRODUCT_CURSOR = 8
_RESERVE_CURSOR = _PRODUCT_CURSOR + len(PRODUCTS)


def modifier_action_features(modifier: MarketModifier) -> np.ndarray:
    row = np.zeros(MARKET_ACTION_DIM, dtype=np.float32)
    row[0] = 1.0
    row[2:8] = (
        modifier.seed_scale,
        modifier.hand_delta / 4.0,
        float(modifier.animal_gate),
        float(modifier.land_bias),
        modifier.feed_days / 4.0,
        modifier.cash_floor,
    )
    row[_RESERVE_CURSOR + 2] = 1.0
    return row


def reserve_action_features(product: str, reserve: int, available: int) -> np.ndarray:
    row = np.zeros(MARKET_ACTION_DIM, dtype=np.float32)
    row[1] = 1.0
    row[_PRODUCT_CURSOR + PRODUCTS.index(product)] = 1.0
    row[_RESERVE_CURSOR] = reserve / 100.0
    row[_RESERVE_CURSOR + 1] = reserve / max(1.0, float(available))
    row[_RESERVE_CURSOR + 2] = 1.0
    return row


def enumerate_market_candidates(
    obs,
    farm: dict,
    private: dict,
    roles: dict,
    field: dict,
    *,
    config=None,
    episode_steps: int = 720,
    turns_per_day: int = 24,
) -> MarketCandidates:
    state = market_state_features(
        obs,
        farm,
        private,
        roles,
        field,
        config=config,
        episode_steps=episode_steps,
        turns_per_day=turns_per_day,
    )
    purchase_static = np.stack([modifier_action_features(modifier) for modifier in MODIFIERS])
    purchase_mask = np.ones(len(MODIFIERS), dtype=bool)

    # Reserves are sized against the shed the donor's market layer will actually
    # see, which is post-field: this turn's DROP/PLACE has already landed.
    shed_capacity = int(donor._cfg(config, "shedCapacity", donor.SHED_CAPACITY))
    post_shed, _ = donor._post_field_storage(private, field, shed_capacity)
    width = len(RESERVE_LEVELS) + 1
    reserve_values: list[tuple[int, ...]] = []
    reserve_static = np.zeros((len(PRODUCTS), width, MARKET_ACTION_DIM), dtype=np.float32)
    reserve_mask = np.zeros((len(PRODUCTS), width), dtype=bool)
    for item_index, item in enumerate(PRODUCTS):
        available = max(0, int(post_shed.get(item, 0) or 0))
        values = tuple(dict.fromkeys((*(min(available, level) for level in RESERVE_LEVELS), available)))
        reserve_values.append(values)
        for value_index, value in enumerate(values):
            reserve_static[item_index, value_index] = reserve_action_features(item, value, available)
            reserve_mask[item_index, value_index] = True
    return MarketCandidates(
        state,
        MODIFIERS,
        purchase_static,
        purchase_mask,
        tuple(reserve_values),
        reserve_static,
        reserve_mask,
    )


def assemble_market(state: np.ndarray, action_static: np.ndarray) -> np.ndarray:
    shape = action_static.shape[:-1] + (state.shape[0],)
    return np.concatenate((np.broadcast_to(state, shape), action_static), axis=-1).astype(
        np.float32, copy=False
    )


def choose_market(
    candidates: MarketCandidates,
    network,
    epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
) -> MarketDecision:
    """Argmax per factor, one shared exploration coin.

    One coin rather than one per factor: the factors are trained as a value
    decomposition of a single joint action, so exploring them independently
    samples joint actions the greedy policy would never visit together.
    """

    purchase_q = network(assemble_market(candidates.state, candidates.purchase_static))
    reserve_q = network(assemble_market(candidates.state, candidates.reserve_static)).reshape(
        candidates.reserve_static.shape[:2]
    )
    explore = epsilon > 0.0 and rng is not None and rng.random() < epsilon
    purchase = int(rng.integers(len(candidates.modifiers))) if explore else int(np.argmax(purchase_q))
    chosen = [purchase]
    reserves: dict[str, int] = {}
    for item_index, values in enumerate(candidates.reserve_values):
        index = (
            int(rng.integers(len(values)))
            if explore
            else int(np.argmax(reserve_q[item_index, : len(values)]))
        )
        chosen.append(index)
        reserves[PRODUCTS[item_index]] = int(values[index])
    return MarketDecision(candidates.modifiers[purchase], reserves, tuple(chosen), candidates)
