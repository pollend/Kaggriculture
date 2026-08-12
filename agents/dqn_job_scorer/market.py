"""Safety-first market controller for the adaptive hybrid DQN.

The network scores field jobs only; everything on the market channel is rules.
This is a copy of the logic the configurable crop agent uses, not a shared
import -- the two agents are free to diverge here, and tuning one must not
silently move the other.

Behaviour is intentionally identical to `ConfigurableAgent._market_orders` at
its default `Strategy`, so the refactor anchor reproduces exactly. Divergence
from here on is a deliberate change, not an accident of the move.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.dqn_job_scorer.hybrid import HybridPlan
from agents.dqn_job_scorer.rules import ANIMAL_DATA, ANIMALS, CROP_DATA, CROPS, PRODUCT_BASE, PRODUCTS, SHOP_DEMAND, get

MAX_ORDERS = 10
LAND_PRICES = (1000, 2000, 4000)


@dataclass(frozen=True, slots=True)
class MarketConfig:
    """Defaults match the crop agent's default Strategy, so the two start level."""

    max_hands: int = 3
    hire_until_hour: int = 2
    cash_reserve: int = 300
    target_planted_tiles: int = 12
    sell_policy: str = "IMMEDIATE"          # IMMEDIATE | PRICE_THRESHOLD | SHED_PRESSURE
    sell_price_ratio: float = 1.0
    shed_pressure_fraction: float = 0.65
    endgame_liquidation_turns: int = 48
    crop_selection: str = "MARKET_ADJUSTED"  # PROFIT_PER_ACTION | PROFIT_PER_DAY | MARKET_ADJUSTED
    crop_weights: tuple[float, ...] = (1.0, 1.0, 0.6, 0.4, 0.5)  # ordered as CROPS
    use_town_forecast: bool = True
    town_demand_weight: float = 0.5
    use_opponent_supply: bool = True
    opponent_supply_weight: float = 0.3
    shed_capacity: int = 100


def _town_crop_counts(obs) -> dict[str, int]:
    counts = {crop: 0 for crop in CROPS}
    town = get(obs, "town", {}) or {}
    for shop in get(town, "unlocked_shops", []) or []:
        for crop in SHOP_DEMAND.get(shop, ()):
            if crop in counts:
                counts[crop] += 1
    return counts


def _opponent_crop_counts(obs, player: int) -> dict[str, int]:
    counts = {crop: 0 for crop in CROPS}
    for index, farm in enumerate(get(obs, "farms", []) or []):
        if index == player:
            continue
        for row in farm.get("tiles", []):
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    crop = tile.get("crop")
                    if crop in counts:
                        counts[crop] += 1
    return counts


def crop_score(crop: str, obs, player: int, config: MarketConfig) -> float:
    data = CROP_DATA[crop]
    prices = (get(obs, "market", {}) or {}).get("prices", {})
    price = float(prices.get(crop, data["base"]))
    gross = price * float(data["yield"]) - float(data["seed"])
    actions = 3.0 + float(data["peak"])  # plant + care days + harvest + rough travel
    days = float(data["peak"]) + 1.0

    if config.crop_selection == "PROFIT_PER_DAY":
        economic = gross / days
    else:
        economic = gross / actions
    if config.crop_selection == "MARKET_ADJUSTED":
        economic *= price / float(data["base"])

    weight = config.crop_weights[CROPS.index(crop)]
    score = max(0.01, economic) * weight
    if config.use_town_forecast:
        score *= 1 + config.town_demand_weight * _town_crop_counts(obs)[crop] / 4
    if config.use_opponent_supply:
        supply = _opponent_crop_counts(obs, player)[crop]
        score /= 1 + config.opponent_supply_weight * supply / 25
    return score


def crop_order(obs, player: int, config: MarketConfig) -> list[str]:
    return sorted(CROPS, key=lambda crop: (-crop_score(crop, obs, player, config), crop))


def market_orders(
    obs,
    farm: dict,
    private: dict,
    step: int,
    hour: int,
    endgame: bool,
    config: MarketConfig,
    episode_steps: int,
    turns_per_day: int,
    plan: HybridPlan | None = None,
) -> list[list]:
    """Build this turn's ordered market queue, capped at `MAX_ORDERS`."""
    orders: list[list] = []
    shed = private.get("shed", {}) or {}
    prices = (get(obs, "market", {}) or {}).get("prices", {})
    shed_total = sum(int(count) for count in shed.values())
    player = int(get(obs, "player", 0))

    plan = plan or HybridPlan("CROP", "GOOSE", 0, config.target_planted_tiles, 0)

    owned_animals = 0
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in ANIMALS:
                owned_animals += 1
    survival_wheat = owned_animals
    learned_reserves = (
        dict(zip(PRODUCTS, plan.sell_reserves)) if len(plan.sell_reserves) == len(PRODUCTS) else None
    )
    money = float(farm.get("money", 0))
    spendable = max(0.0, money - config.cash_reserve)
    carried_wheat = sum(int((slot or {}).get("WHEAT", 0)) for slot in private.get("inventories", []) or [])
    wheat_target = max(survival_wheat, plan.wheat_reserve)
    wheat_shortfall = max(0, wheat_target - int(shed.get("WHEAT", 0)) - carried_wheat)
    wheat_price = int(prices.get("WHEAT", CROP_DATA["WHEAT"]["base"]))
    affordable_wheat = min(wheat_shortfall, int(spendable // max(1, wheat_price)))

    # Survival gets the first market slot. Otherwise a full learned sell queue
    # could silently push feed past the environment's ten-order cap.
    if affordable_wheat and not endgame:
        orders.append(["BUY_PRODUCT", "WHEAT", affordable_wheat])
        spendable -= affordable_wheat * wheat_price

    # Terminal inventory has no value. Sell everything before considering buys.
    for item in PRODUCTS:
        count = int(shed.get(item, 0))
        if count <= 0:
            continue
        if learned_reserves is not None and not endgame:
            reserve = int(learned_reserves[item])
        else:
            reserve = plan.wheat_reserve if item == "WHEAT" and not endgame else 0
            reserve = min(3, plan.target_crops) if item == "FERTILIZER" and not endgame else reserve
        if item == "WHEAT" and not endgame:
            reserve = max(reserve, survival_wheat)
        count = max(0, count - reserve)
        if count <= 0:
            continue
        base = float(PRODUCT_BASE[item])
        ratio = float(prices.get(item, base)) / base
        sell = endgame or learned_reserves is not None or config.sell_policy == "IMMEDIATE"
        sell |= learned_reserves is None and config.sell_policy == "PRICE_THRESHOLD" and ratio >= config.sell_price_ratio
        sell |= learned_reserves is None and config.sell_policy == "SHED_PRESSURE" and (
            shed_total >= config.shed_capacity * config.shed_pressure_fraction or ratio >= config.sell_price_ratio
        )
        if sell:
            orders.append(["SELL", item, count])

    if endgame:
        return orders[:MAX_ORDERS]

    unlocked = len(farm.get("unlocked_quadrants", []) or ["NW"])
    if plan.target_land > unlocked and unlocked <= len(LAND_PRICES) and len(orders) < MAX_ORDERS:
        land_cost = LAND_PRICES[unlocked - 1]
        if spendable >= land_cost:
            orders.append(["BUY_LAND"])
            spendable -= land_cost

    hires = len(farm.get("hands", []) or [])
    if hour <= config.hire_until_hour:
        # Hire cost is farmHandCostMult * fib(hires_today): 1, 1, 2, 3, 5, 8, ...
        fib_a, fib_b = 1, 1
        for _ in range(int(farm.get("hires_today", hires))):
            fib_a, fib_b = fib_b, fib_a + fib_b
        target_hands = min(config.max_hands, max(hires, plan.target_hands))
        while hires < target_hands and spendable >= fib_a and len(orders) < MAX_ORDERS:
            orders.append(["HIRE"])
            spendable -= fib_a
            fib_a, fib_b = fib_b, fib_a + fib_b
            hires += 1

    active = sum(
        1
        for row in farm.get("tiles", []) or []
        for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "PLANT"
    )
    seeds = private.get("seeds", {}) or {}
    # Purchased animals remain useful only when the labor-aware plan supports them.
    animals = {name: int(shed.get(name, 0)) for name in ANIMALS}
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("animal") in animals:
                animals[str(tile["animal"])] += 1
    for slot in private.get("inventories", []) or []:
        for name in ANIMALS:
            animals[name] += int((slot or {}).get(name, 0))
    deficit = max(0, plan.target_animals - sum(animals.values()))
    if deficit and len(orders) < MAX_ORDERS:
        animal = plan.animal
        cost = int(ANIMAL_DATA[animal]["cost"])
        quantity = min(deficit, int(spendable // cost))
        if quantity:
            orders.append(["BUY_ANIMAL", animal, quantity])
            spendable -= quantity * cost

    needed = max(0, plan.target_crops - active - sum(int(n) for n in seeds.values()))
    order = [plan.crop, *(crop for crop in crop_order(obs, player, config) if crop != plan.crop)]

    while needed and len(orders) < MAX_ORDERS:
        bought = False
        for crop in order:
            cost = int(CROP_DATA[crop]["seed"])
            remaining_days = (episode_steps - step - 1) / turns_per_day
            # A crop that cannot reach first yield before the season ends is dead money.
            if remaining_days < int(CROP_DATA[crop]["first"]) or spendable < cost:
                continue
            orders.append(["BUY_SEED", crop, 1])
            spendable -= cost
            needed -= 1
            bought = True
            break
        if not bought:
            break
    return orders[:MAX_ORDERS]
