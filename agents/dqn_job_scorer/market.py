"""Market orders for the DQN agent: selling, hiring, seed buying.

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

from agents.dqn_job_scorer.rules import CROP_DATA, CROPS, SHOP_DEMAND, get

MAX_ORDERS = 10


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
) -> list[list]:
    """Build this turn's ordered market queue, capped at `MAX_ORDERS`."""
    orders: list[list] = []
    shed = private.get("shed", {}) or {}
    prices = (get(obs, "market", {}) or {}).get("prices", {})
    shed_total = sum(int(count) for count in shed.values())
    player = int(get(obs, "player", 0))

    for crop in CROPS:
        count = int(shed.get(crop, 0))
        if count <= 0:
            continue
        ratio = float(prices.get(crop, CROP_DATA[crop]["base"])) / float(CROP_DATA[crop]["base"])
        sell = endgame or config.sell_policy == "IMMEDIATE"
        sell |= config.sell_policy == "PRICE_THRESHOLD" and ratio >= config.sell_price_ratio
        sell |= config.sell_policy == "SHED_PRESSURE" and (
            shed_total >= config.shed_capacity * config.shed_pressure_fraction
            or ratio >= config.sell_price_ratio
        )
        if sell:
            orders.append(["SELL", crop, count])

    if endgame:
        return orders[:MAX_ORDERS]

    money = float(farm.get("money", 0))
    spendable = max(0.0, money - config.cash_reserve)

    hires = len(farm.get("hands", []) or [])
    if hour <= config.hire_until_hour:
        # Hire cost is farmHandCostMult * fib(hires_today): 1, 1, 2, 3, 5, 8, ...
        fib_a, fib_b = 1, 1
        for _ in range(int(farm.get("hires_today", hires))):
            fib_a, fib_b = fib_b, fib_a + fib_b
        while hires < config.max_hands and spendable >= fib_a and len(orders) < MAX_ORDERS:
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
    needed = max(0, config.target_planted_tiles - active - sum(int(n) for n in seeds.values()))
    order = crop_order(obs, player, config)

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
