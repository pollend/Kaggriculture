"""The donor's market executor with the learned knobs threaded through.

DERIVATIVE WORK. `market_actions` below is a modified copy of `_market_actions`
in `donor.py` (lines 1706-1931), which is Apache-2.0 licensed work by Pilkwang
Kim. See NOTICE for provenance and the list of changes.

Every deviation from the original is marked `# LEARNED:`. The decision rules,
their ordering, and the order budget are otherwise untouched -- the network
scales the donor's demand, it does not replace the donor's judgement. Legality,
affordability, survival feed and terminal liquidation stay the executor's job.
"""

from __future__ import annotations

from agents.dqn_market_scorer import donor
from agents.dqn_market_scorer.market_policy import DEFAULT_MODIFIER, MarketModifier

# How far the land-purchase day gate moves when the network biases it.
LAND_BIAS_DAYS = {-1: 6, 0: 0, 1: -4}


def _hand_cap(obs, farm: dict) -> int:
    """The donor's own labor ceiling, so `hand_delta` cannot exceed it."""

    return (
        donor._LABOR_GATE_INSIDE_CAP
        if donor._labor_gate_active(obs, farm)
        else donor._LABOR_GATE_OUTSIDE_CAP
    )


def market_actions(
    obs,
    config,
    farm: dict,
    private: dict,
    roles: dict,
    field: dict,
    *,
    modifier: MarketModifier = DEFAULT_MODIFIER,
    reserves: dict[str, int] | None = None,
) -> list[list]:
    reserves = reserves or {}
    day = int(obs.get("day", 0) or 0)
    hour = int(obs.get("hour", 0) or 0)
    left = donor.TOTAL_DAYS - day
    money = float(farm.get("money", 0) or 0)
    shed_capacity = int(donor._cfg(config, "shedCapacity", donor.SHED_CAPACITY))
    shed, post_field_inventories = donor._post_field_storage(private, field, shed_capacity)
    market_inventory = dict((obs.get("market", {}) or {}).get("inventory", {}) or {})
    max_orders = int(donor._cfg(config, "maxMarketOrdersPerTurn", donor.MAX_MARKET_ORDERS))
    summary = donor._survey(farm, private, roles, day)
    phase = donor._policy_phase(obs, farm, private, summary)
    orders: list[list] = []
    occupancy = sum(max(0, int(value or 0)) for value in shed.values())
    shed_load = occupancy / float(max(1, shed_capacity))
    animal_pipeline = summary["animals"] + sum(summary["animal_stock"].values())
    # LEARNED: feed horizon is a knob rather than the fixed FEED_STOCK_DAYS.
    feed_floor = animal_pipeline * modifier.feed_days
    total_wheat = int(shed.get("WHEAT", 0) or 0) + sum(
        int(inventory.get("WHEAT", 0) or 0) for inventory in post_field_inventories
    )

    # LEARNED: the donor's sell loop runs before its liquidation early-return, so
    # the retention guard has to be explicit here. Holding stock on the last day
    # or during a liquidation would strand it worthless.
    liquidating = bool(field.get("liquidation")) or left <= 1

    sells = []
    for item in donor.PRODUCTS:
        have = int(shed.get(item, 0) or 0)
        if item == "WHEAT" and left > 2:
            have = min(have, max(0, total_wheat - feed_floor))
        # LEARNED: retain the chosen reserve before the donor sizes the sale. The
        # donor's price thresholds then apply to what is left, so the network can
        # hold inventory but can never force a dump at a bad quote.
        if not liquidating:
            have = max(0, have - int(reserves.get(item, 0) or 0))
        if have <= 0:
            continue
        raw_inventory = market_inventory.get(item)
        inventory = donor.MARKET_I0 if raw_inventory is None else int(raw_inventory)
        quantity = donor._sell_quantity(item, have, inventory, day, shed_load, obs)
        if quantity <= 0:
            continue
        proceeds = sum(donor._price_at(item, inventory + offset, obs) for offset in range(quantity))
        # Orders are filled unit by unit and interleaved between seats, so on a
        # product both seats sell this turn the earlier slot takes the better
        # prices. Rank by the decay our own order inflicts on its quote -- the
        # slot where arriving late costs most -- rather than by gross proceeds,
        # which is dominated by the base price and carries no information about
        # the local slope.
        impact = quantity * max(
            0.0,
            donor._price_at(item, inventory, obs) - donor._price_at(item, inventory + quantity, obs),
        )
        sells.append((impact, proceeds, item, quantity))
    sells.sort(reverse=True)
    for _impact, proceeds, item, quantity in sells:
        if len(orders) >= max_orders:
            break
        orders.append(["SELL", item, quantity])
        money += 0.85 * proceeds
        occupancy = max(0, occupancy - quantity)
        shed[item] = max(0, int(shed.get(item, 0) or 0) - quantity)
        raw_inventory = market_inventory.get(item)
        inventory = donor.MARKET_I0 if raw_inventory is None else int(raw_inventory)
        market_inventory[item] = inventory + quantity

    if field["liquidation"] or left <= 1:
        if day >= 29 and hour <= 1:
            terminal_jobs = donor._field_jobs(obs, farm, private, roles, liquidation=True)
            target = min(8, len(terminal_jobs))
            hires = int(farm.get("hires_today", 0) or 0)
            while hires < target and len(orders) < max_orders:
                cost = donor._fib(hires)
                if money < cost + 20:
                    break
                orders.append(["HIRE"])
                money -= cost
                hires += 1
        return orders[:max_orders]

    placed = donor._farm_animal_counts(farm)
    role_targets = {animal: 0 for animal in donor.ANIMALS}
    for kind, item in roles.values():
        if kind == "ANIMAL" and item in role_targets:
            role_targets[item] += 1
    owned = {
        animal: placed[animal] + donor._private_item_total(private, animal)
        for animal in donor.ANIMALS
    }
    # LEARNED: animal_gate == 0 vetoes capital spend on livestock outright. It
    # only ever removes permission -- the donor's own phase gate still has to
    # agree -- so the network cannot buy a herd during a crisis.
    animal_capital_open = bool(modifier.animal_gate) and (
        phase in {"BOOTSTRAP", "COMPOUND"}
        or (
            phase == "CRISIS"
            and summary["at_risk_animals"] == 0
            and summary["shed_load"] + summary["carried_load"] < 95
            and summary["open_structures"] > 0
            and animal_pipeline < summary["animals"] + summary["open_structures"]
        )
    )
    if animal_capital_open and day <= donor.ANIMAL_PURCHASE_LAST_DAY and left >= 8:
        purchase_order = sorted(
            ("COW", "SHEEP"),
            key=lambda animal: (
                donor._livestock_score(
                    obs, animal, owned[animal], donor._opponent_animal_counts(obs)[animal]
                ),
                role_targets[animal] - owned[animal],
                animal == "COW",
            ),
            reverse=True,
        )
        for animal in purchase_order:
            if len(orders) >= max_orders:
                break
            missing = max(0, role_targets[animal] - owned[animal])
            if missing <= 0:
                continue
            # LEARNED: cash_floor scales the operating reserve the donor keeps
            # back before committing to livestock.
            operating_reserve = int(
                round((80 if sum(owned.values()) < donor.CORE_HERD else 220) * modifier.cash_floor)
            )
            quantity = min(
                missing,
                2,
                max(0, shed_capacity - occupancy),
                max(0, int((money - operating_reserve) // donor.ANIMALS[animal]["cost"])),
            )
            if quantity > 0:
                orders.append(["BUY_ANIMAL", animal, quantity])
                money -= quantity * donor.ANIMALS[animal]["cost"]
                occupancy += quantity
                owned[animal] += quantity

    total_wheat = int(shed.get("WHEAT", 0) or 0) + sum(
        int(inventory.get("WHEAT", 0) or 0) for inventory in post_field_inventories
    )
    planned_herd = sum(owned.values())
    # LEARNED: same feed horizon knob as the sale floor above.
    desired_wheat = max(planned_herd * modifier.feed_days, 8 if planned_herd > 0 else 0)
    if desired_wheat > total_wheat and len(orders) < max_orders and planned_herd > 0:
        raw_inventory = market_inventory.get("WHEAT")
        inventory = donor.MARKET_I0 if raw_inventory is None else int(raw_inventory)
        emergency_reserve = 0 if summary["at_risk_animals"] else 80
        quantity = 0
        cost = 0
        limit = min(desired_wheat - total_wheat, max(0, shed_capacity - occupancy))
        for offset in range(limit):
            unit = donor._price_at("WHEAT", inventory - offset - 1, obs)
            if money - cost - unit < emergency_reserve:
                break
            cost += unit
            quantity += 1
        if quantity > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", quantity])
            money -= cost
            occupancy += quantity

    extra_land = max(0, len(farm.get("unlocked_quadrants", ["NW"])) - 1)
    if (
        phase in {"BOOTSTRAP", "COMPOUND"}
        and extra_land < donor.MAX_EXTRA_LAND
        # LEARNED: land_bias shifts the day the quadrant becomes eligible.
        and day >= max(0, donor.LAND_OPEN_DAYS[extra_land] + LAND_BIAS_DAYS[modifier.land_bias])
        and left >= 12
        and len(orders) < max_orders
    ):
        cost = donor.LAND_PRICES[extra_land]
        reserve = 300 if extra_land == 0 else 500
        if money >= cost + reserve:
            orders.append(["BUY_LAND"])
            money -= cost

    needs = donor._seed_needs(obs, farm, private, roles)
    # LEARNED: fund seed_scale of the role plan's stated seed deficit.
    needs = {crop: int(round(count * modifier.seed_scale)) for crop, count in needs.items()}
    # LEARNED: cash_floor scales the cash the donor keeps back from seed spend.
    seed_reserve = int(round((80 if day <= 4 else 150) * modifier.cash_floor))
    seed_order = (
        ("MELON",)
        if day == 0
        else ("MELON", "WHEAT", "STRAWBERRY", "CARROT", "TOMATO")
    )
    for crop in seed_order:
        if len(orders) >= max_orders or needs.get(crop, 0) <= 0:
            continue
        cost = donor.CROPS[crop]["seed"]
        quantity = min(needs[crop], 25, max(0, int((money - seed_reserve) // cost)))
        if quantity > 0:
            orders.append(["BUY_SEED", crop, quantity])
            money -= quantity * cost

    if hour <= 2:
        # LEARNED: hand_delta offsets the donor's labor target, clamped to the
        # donor's own ceiling so the network cannot outspend the labor gate.
        target_hands = donor._target_hands(obs, farm, private, roles)
        target_hands = max(0, min(_hand_cap(obs, farm), target_hands + modifier.hand_delta))
        hires = int(farm.get("hires_today", 0) or 0)
        while hires < target_hands and len(orders) < max_orders:
            cost = donor._fib(hires)
            if money < max(20, 3 * cost):
                break
            orders.append(["HIRE"])
            money -= cost
            hires += 1

    return orders[:max_orders]
