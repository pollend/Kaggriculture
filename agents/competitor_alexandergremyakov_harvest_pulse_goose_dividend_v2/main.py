import math


CROP_RULES = {
    "WHEAT": (10, 2, 4, 24, False),
    "CARROT": (20, 2, 3, 25, False),
    "TOMATO": (50, 8, 0, 20, True),
    "STRAWBERRY": (100, 10, 0, 18, True),
    "MELON": (80, 10, 12, 16, False),
}

MIX = (
    ("MELON", 13),
    ("CARROT", 4),
    ("WHEAT", 4),
    ("TOMATO", 2),
    ("STRAWBERRY", 2),
)

SELL_ORDER = (
    "MELON",
    "STRAWBERRY",
    "TOMATO",
    "CARROT",
    "WHEAT",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)

MARKET_RULES = {
    "WHEAT": (25, 10000, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 10000, 450, "log", 0.20, "sqrt", 0.70),
    "TOMATO": (60, 10000, 200, "linear", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 10000, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 10000, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 10000, 332, "linear", 0.40, "log", 0.20),
    "MILK": (160, 10000, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 10000, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 10000, 200, "linear", 0.40, "linear", 0.40),
}

TARGET_HANDS = 7
LAND_RESERVE = 100
LAND_PRICES = (1000, 2000, 4000)
ANIMAL_COSTS = {
    "GOOSE": 300,
    "COW": 400,
    "SHEEP": 500,
}
FARM_HAND_COST_MULT = 1
MAX_MARKET_ORDERS = 10
SHED_CAPACITY = 100
GOOSE_COST = ANIMAL_COSTS["GOOSE"]
WHEAT_RESERVE = 5
MOVE_ORDER = (
    ("NORTH", 0, -1),
    ("EAST", 1, 0),
    ("SOUTH", 0, 1),
    ("WEST", -1, 0),
)

DEFAULT_FEATURES = {
    "early_ne": True,
    "land_day": 0,
    "full_land": True,
    "staggered_hiring": False,
    "safe_routing": True,
    "routing_mode": "hybrid",
    "global_matching": True,
    "stable_assignment": False,
    "target_hands": TARGET_HANDS,
    "sticky_followups": True,
    "storage_threshold": 82,
    "storage_cap": 90,
    "urgent_storage_cap": 99,
    "recall_all": True,
    "drop_hour": 20,
    "terminal_return": True,
    "retire_ongoing": "harvest_only",
    "retire_slack": 12,
    "retire_opportunistic": False,
    "retire_assignment_protect": False,
    "contract_horizon": True,
    "projected_wave_recall": False,
    "wave_window": 16,
    "expiry_lead": None,
    "expiry_slack": 4,
    "expiry_slack_mode": "pair",
    "use_ne_corridor_plot": False,
    "ne_far_mix": (
        "CARROT",
        "CARROT",
        "TOMATO",
        "TOMATO",
        "STRAWBERRY",
        "STRAWBERRY",
    ),
    "horizon_purchases": True,
    "terminal_ops": True,
    "terminal_hires": False,
    "endgame_wheat": False,
    "all_season_jit": False,
    "tight_wheat_early": False,
    "goose": True,
    "target_geese": 2,
    "second_goose_buy_day": 3,
    "second_goose_buy_deadline": 8,
    "second_goose_place_cutoff": 16,
    "computed_sales": True,
    "hold_scarce": True,
    "pace_melons": True,
    "melon_demand_share": 0.65,
    "fertilize_tomato": False,
    "fertilize_melon": False,
    "fertilize_strawberry": True,
    "fertilize_strawberry_second": True,
    "fertilizer_value_ratio": 1.10,
}


def _direct_move(source, target):
    x, y = source
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


def _route(tiles, source, target, routing_mode):
    if source == target:
        return 0, None
    direct = _direct_move(source, target)
    if routing_mode == "direct":
        return (
            abs(source[0] - target[0]) + abs(source[1] - target[1]),
            direct,
        )

    if routing_mode == "hybrid":
        deltas = {
            "NORTH": (0, -1),
            "EAST": (1, 0),
            "SOUTH": (0, 1),
            "WEST": (-1, 0),
        }
        dx, dy = deltas[direct[0]]
        nxt = (source[0] + dx, source[1] + dy)
        if (
            0 <= nxt[0] < len(tiles)
            and 0 <= nxt[1] < len(tiles)
            and tiles[nxt[1]][nxt[0]] != "LOCKED"
        ):
            return (
                abs(source[0] - target[0]) + abs(source[1] - target[1]),
                direct,
            )

    size = len(tiles)
    queue = [source]
    previous = {source: None}
    arrived_by = {}
    head = 0

    while head < len(queue):
        current = queue[head]
        head += 1
        directions = sorted(
            MOVE_ORDER,
            key=lambda item: (
                abs(current[0] + item[1] - target[0])
                + abs(current[1] + item[2] - target[1]),
                MOVE_ORDER.index(item),
            ),
        )
        for action, dx, dy in directions:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in previous:
                continue
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if tiles[nxt[1]][nxt[0]] == "LOCKED":
                continue
            previous[nxt] = current
            arrived_by[nxt] = action
            if nxt == target:
                cursor = nxt
                distance = 1
                while previous[cursor] != source:
                    cursor = previous[cursor]
                    distance += 1
                return distance, [arrived_by[cursor]]
            queue.append(nxt)
    return None, None


def _hands_view(farm, private):
    hands = list(farm.get("hands", []) or [])
    if not hands:
        return None, None
    farm_view = dict(farm)
    farm_view["farmer"] = hands[0]
    farm_view["hands"] = hands[1:]
    private_view = dict(private)
    private_view["inventories"] = list(
        private.get("inventories", []) or []
    )[1:]
    return farm_view, private_view


def _plan(
    board_size,
    include_ne,
    full_land,
    use_ne_corridor_plot=False,
    ne_far_mix=(),
    target_geese=2,
):
    half = max(1, board_size // 2)
    coop = (half - 1, half - 1)
    core = [
        (x, y)
        for y in range(half)
        for x in range(half)
        if (x, y) != coop
    ]
    core.sort(
        key=lambda p: (
            -(abs(p[0] - coop[0]) + abs(p[1] - coop[1])),
            p[1],
            p[0],
        )
    )

    crops = []
    for crop, count in MIX:
        crops.extend([crop] * count)
    plan = dict(zip(core, crops))

    if include_ne:
        shed = (half, half - 1)
        corridor = {shed}
        if not use_ne_corridor_plot:
            corridor.add((half, max(0, half - 2)))
        extra = [
            (x, y)
            for y in range(half)
            for x in range(half, board_size)
            if (x, y) not in corridor
        ]
        extra.sort(
            key=lambda p: (
                abs(p[0] - shed[0]) + abs(p[1] - shed[1]),
                p[1],
                p[0],
            )
        )
        limit = len(extra) if full_land else min(15, len(extra))
        selected = extra[:limit]
        plan.update({position: "WHEAT" for position in selected})
        far_mix = tuple(ne_far_mix or ())
        if far_mix:
            for position, crop in zip(
                reversed(selected),
                far_mix,
            ):
                plan[position] = crop
    if target_geese >= 2:
        plan.pop((half + 1, half - 1), None)
    return plan


def _task(
    tile,
    planned_crop,
    day,
    hour,
    step,
    seeds,
    sticky_followups,
    retire_ongoing,
    expiry_lead,
):
    if tile == "LOCKED":
        return None
    if tile is None:
        last_day = CROP_RULES[planned_crop][3]
        plant_hour = 18 if sticky_followups == "cutoff18" else 21
        if (
            hour <= plant_hour
            and day <= last_day
            and seeds.get(planned_crop, 0) > 0
        ):
            priority = (
                2
                if (
                    planned_crop == "TOMATO"
                    and day >= 14
                    and int(seeds.get("TOMATO", 0)) > 1
                )
                else 3
            )
            return priority, ["PLANT", planned_crop]
        return None
    if not isinstance(tile, dict):
        return None
    if tile.get("kind") == "WEED":
        return 2, ["DIG"]
    if tile.get("kind") != "PLANT":
        return None

    crop = tile.get("crop", planned_crop)
    _, first_day, harvest_day, _, ongoing = CROP_RULES.get(
        crop, CROP_RULES[planned_crop]
    )
    age = day - tile.get("planted_day", day)
    amount = tile.get("yield_units", 0)

    if ongoing and retire_ongoing and tile.get("max_lifespan_step", -1) >= 0:
        if amount > 0:
            return 0, ["HARVEST"]
        if retire_ongoing == "harvest_only":
            return None
        return 0, ["DIG"]
    expiry = int(tile.get("max_lifespan_step", -1))
    if (
        not ongoing
        and expiry >= 0
        and expiry_lead is not None
        and step + int(expiry_lead) >= expiry
    ):
        target = 4 if crop == "WHEAT" else 3 if crop == "CARROT" else 0
        if (
            not tile.get("watered_today", False)
            and tile.get("consecutive_unwatered", 0) >= 1
        ):
            return -3, ["WATER"]
        if (
            target
            and amount < target
            and not tile.get("watered_today", False)
        ):
            return -2, ["WATER"]
        if amount > 0:
            return -2, ["HARVEST"]
    if crop == "MELON" and age >= 10 and amount >= 6:
        return 0, ["HARVEST"]
    if not ongoing:
        if age >= harvest_day and amount > 0:
            target = 4 if crop == "WHEAT" else 3 if crop == "CARROT" else 0
            if target and amount < target and not tile.get("watered_today", False):
                return 0, ["WATER"]
            return 0, ["HARVEST"]
        if not tile.get("watered_today", False):
            if sticky_followups and tile.get("consecutive_unwatered", 0) >= 1:
                if sticky_followups == "priority0":
                    return 0, ["WATER"]
                if sticky_followups == "new_vs_old":
                    return (
                        -1 if tile.get("planted_day") == day else 0,
                        ["WATER"],
                    )
                return -3, ["WATER"]
            return 1, ["WATER"]
        return None
    if not tile.get("watered_today", False):
        if sticky_followups and tile.get("consecutive_unwatered", 0) >= 1:
            if sticky_followups == "priority0":
                return 0, ["WATER"]
            if sticky_followups == "new_vs_old":
                return (
                    -1 if tile.get("planted_day") == day else 0,
                    ["WATER"],
                )
            return -3, ["WATER"]
        return 1, ["WATER"]
    if age >= first_day and amount > 0:
        return 0, ["HARVEST"]
    return None


def _candidate_tasks(obs, farm, private, plan, features):
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    step = int(obs.get("step", day * 24 + hour))
    seeds = private.get("seeds", {}) or {}
    tiles = farm["tiles"]
    tasks = []

    for position, crop in plan.items():
        x, y = position
        result = _task(
            tiles[y][x],
            crop,
            day,
            hour,
            step,
            seeds,
            features["sticky_followups"],
            features["retire_ongoing"],
            features["expiry_lead"],
        )
        if result is not None:
            priority, action = result
            tile_expiry = (
                int(tiles[y][x].get("max_lifespan_step", -1))
                if isinstance(tiles[y][x], dict)
                else -1
            )
            tile_crop = (
                tiles[y][x].get("crop", crop)
                if isinstance(tiles[y][x], dict)
                else crop
            )
            expiry_urgent = bool(
                features["expiry_lead"] is not None
                and tile_expiry >= 0
                and not CROP_RULES.get(
                    tile_crop,
                    CROP_RULES[crop],
                )[4]
                and step + int(features["expiry_lead"]) >= tile_expiry
            )
            tile_ongoing = CROP_RULES.get(
                tile_crop,
                CROP_RULES[crop],
            )[4]
            urgent = priority < 0
            tasks.append(
                {
                    "priority": priority,
                    "position": position,
                    "action": action,
                    "urgent": urgent,
                    "crop": (
                        tiles[y][x].get("crop")
                        if isinstance(tiles[y][x], dict)
                        else crop
                    ),
                    "units": (
                        int(tiles[y][x].get("yield_units", 0))
                        if isinstance(tiles[y][x], dict)
                        else 0
                    ),
                    "expiry_urgent": expiry_urgent,
                    "expiry": tile_expiry,
                    "final_annual": bool(
                        tile_expiry >= 0
                        and not tile_ongoing
                        and action[0] in ("WATER", "HARVEST")
                    ),
                    "remaining_actions": (
                        1
                        if action[0] == "PLANT"
                        or (action[0] == "WATER" and not tile_ongoing)
                        else 0
                    ),
                    "retirement_harvest": bool(
                        features["retire_ongoing"]
                        and tile_ongoing
                        and tile_expiry >= 0
                        and action[0] == "HARVEST"
                    ),
                }
            )

    tasks.sort(
        key=lambda task: (
            task["priority"],
            task["position"][1],
            task["position"][0],
        )
    )

    seed_budget = dict(seeds)
    kept = []
    for task in tasks:
        action = task["action"]
        if action[0] == "PLANT":
            crop = action[1]
            if seed_budget.get(crop, 0) <= 0:
                continue
            seed_budget[crop] = seed_budget.get(crop, 0) - 1
        kept.append(task)
    return kept


def _pair_data(
    farm,
    positions,
    tasks,
    safe_routing,
    step,
    hour,
    expiry_slack,
    expiry_slack_mode,
    retire_slack,
    retire_opportunistic,
    contract_horizon,
):
    tiles = farm["tiles"]
    routes = {}
    nearest = {}
    day_end = step - hour + 23
    for worker, source in enumerate(positions):
        source = tuple(source)
        for task_idx, task in enumerate(tasks):
            distance, move = _route(
                tiles,
                source,
                task["position"],
                safe_routing,
            )
            if distance is None:
                continue
            effective_expiry = task.get("expiry", -1)
            if effective_expiry >= 0 and effective_expiry % 24 == 0:
                effective_expiry -= 1
            if (
                contract_horizon
                and hour + distance >= 24
            ):
                continue
            if (
                contract_horizon
                and 0 <= effective_expiry <= day_end
                and step
                + distance
                + task.get("remaining_actions", 0)
                > effective_expiry
            ):
                continue
            routes[(worker, task_idx)] = (distance, move)
            nearest[task_idx] = min(
                distance,
                nearest.get(task_idx, distance),
            )

    pairs = {}
    for (worker, task_idx), (distance, move) in routes.items():
        task = tasks[task_idx]
        effective_expiry = task.get("expiry", -1)
        if effective_expiry >= 0 and effective_expiry % 24 == 0:
            effective_expiry -= 1
        urgency_distance = (
            nearest[task_idx]
            if expiry_slack_mode == "nearest"
            else distance
        )
        deadline_urgent = bool(
            expiry_slack is not None
            and task.get("final_annual")
            and effective_expiry
            - step
            - urgency_distance
            - task["remaining_actions"]
            <= int(expiry_slack)
        )
        retirement_urgent = bool(
            task.get("retirement_harvest")
            and (
                (retire_opportunistic and distance == 0)
                or effective_expiry
                - step
                - nearest[task_idx]
                <= int(retire_slack)
            )
        )
        pair_urgent = deadline_urgent or retirement_urgent
        travel_weight = (
            10000 if task.get("urgent") or pair_urgent else 100
        )
        urgency_bonus = (
            -2000000
            if deadline_urgent and not task.get("urgent")
            else -1000000
            if retirement_urgent and not task.get("urgent")
            else 0
        )
        cost = (
            task["priority"] * 1000000
            + urgency_bonus
            + distance * travel_weight
            + task["position"][1] * len(tiles)
            + task["position"][0]
        )
        pairs[(worker, task_idx)] = (
            cost,
            distance,
            move,
            pair_urgent,
            (
                effective_expiry
                - step
                - urgency_distance
                - task["remaining_actions"]
                if pair_urgent
                else None
            ),
        )
    return pairs


def _global_assign(positions, tasks, pairs):
    workers = len(positions)
    empty = tuple([-1] * workers)
    states = {0: (0, empty)}

    for task_idx in range(len(tasks)):
        updated = dict(states)
        for mask, (total, assignments) in states.items():
            for worker in range(workers):
                if mask & (1 << worker):
                    continue
                pair = pairs.get((worker, task_idx))
                if pair is None:
                    continue
                next_mask = mask | (1 << worker)
                next_total = total + pair[0]
                current = updated.get(next_mask)
                if current is not None and current[0] <= next_total:
                    continue
                selected = list(assignments)
                selected[worker] = task_idx
                updated[next_mask] = (next_total, tuple(selected))
        states = updated

    best_mask, (_, best) = min(
        states.items(),
        key=lambda item: (
            -item[0].bit_count(),
            item[1][0],
            item[1][1],
        ),
    )
    if best_mask == 0:
        return empty
    return best


def _stable_assign(positions, tasks, pairs):
    workers = len(positions)
    task_count = len(tasks)
    if not workers:
        return ()
    if not task_count or not pairs:
        return tuple([-1] * workers)

    integer_costs = {}
    for key, pair in pairs.items():
        worker, task_idx = key
        if not (0 <= worker < workers and 0 <= task_idx < task_count):
            continue
        try:
            cost = int(pair[0])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("assignment costs must be finite integers") from error
        if cost != pair[0]:
            raise ValueError("assignment costs must be finite integers")
        integer_costs[key] = cost
    if not integer_costs:
        return tuple([-1] * workers)

    base = task_count + 2
    tie_scale = base ** workers
    maximum = max(abs(cost) for cost in integer_costs.values())
    cardinality = (2 * workers * maximum + 2) * tie_scale
    forbidden = (workers + 1) * cardinality
    columns = task_count + workers
    costs = []
    for worker in range(workers):
        weight = base ** (workers - worker - 1)
        row = []
        for task_idx in range(task_count):
            cost = integer_costs.get((worker, task_idx))
            if cost is None:
                row.append(forbidden)
            else:
                row.append(
                    cost * tie_scale
                    + (task_idx + 1) * weight
                )
        row.extend([cardinality] * workers)
        costs.append(row)

    u = [0] * (workers + 1)
    v = [0] * (columns + 1)
    matched = [0] * (columns + 1)
    path = [0] * (columns + 1)
    infinity = forbidden * (workers + columns + 2) + 1

    for worker in range(1, workers + 1):
        matched[0] = worker
        column = 0
        minimum = [infinity] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column] = True
            active = matched[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = (
                    costs[active - 1][candidate - 1]
                    - u[active]
                    - v[candidate]
                )
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    path[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[matched[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched[column] == 0:
                break
        while True:
            previous = path[column]
            matched[column] = matched[previous]
            column = previous
            if column == 0:
                break

    result = [-1] * workers
    for column in range(1, columns + 1):
        worker = matched[column]
        if not worker or column > task_count:
            continue
        task_idx = column - 1
        if (worker - 1, task_idx) in integer_costs:
            result[worker - 1] = task_idx
    return tuple(result)


def _select_assign(positions, tasks, pairs, stable=False):
    solver = _stable_assign if stable else _global_assign
    return solver(positions, tasks, pairs)


def _greedy_assign(positions, tasks, pairs):
    assignments = [-1] * len(positions)
    reserved = set()
    for worker in range(len(positions)):
        choices = [
            (pairs[(worker, task_idx)][0], task_idx)
            for task_idx in range(len(tasks))
            if task_idx not in reserved and (worker, task_idx) in pairs
        ]
        if choices:
            _, task_idx = min(choices)
            assignments[worker] = task_idx
            reserved.add(task_idx)
    return tuple(assignments)


def _stock(mapping):
    return sum(int(amount) for amount in (mapping or {}).values())


def _shape(name, value):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    return value


def _price_at(obs, item, inventory):
    base, i0, throughput, below_fn, below_target, above_fn, above_target = (
        MARKET_RULES[item]
    )
    patch = ((obs.get("market", {}) or {}).get("params", {}) or {}).get(
        item,
        {},
    )
    if patch:
        base = patch.get("base", base)
        i0 = patch.get("I0", i0)
        throughput = patch.get("T", throughput)
        below_fn = patch.get("below_func", below_fn)
        below_target = patch.get("below_target", below_target)
        above_fn = patch.get("above_func", above_fn)
        above_target = patch.get("above_target", above_target)
    if inventory < i0:
        amp = below_target * base / max(
            1e-9,
            _shape(below_fn, throughput),
        )
        price = base + amp * _shape(below_fn, i0 - inventory)
    else:
        amp = above_target * base / max(
            1e-9,
            _shape(above_fn, throughput),
        )
        price = base - amp * _shape(above_fn, inventory - i0)
    return max(1, int(round(price)))


def _item_stock(private, item):
    total = int((private.get("shed", {}) or {}).get(item, 0))
    for inventory in private.get("inventories", []) or []:
        total += int((inventory or {}).get(item, 0))
    return total


def _inventory(private, index):
    inventories = private.get("inventories", []) or []
    return inventories[index] or {} if index < len(inventories) else {}


def _shed_access(farm):
    size = len(farm["tiles"])
    half = size // 2
    return [
        position
        for position in (
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        )
        if farm["tiles"][position[1]][position[0]] != "LOCKED"
    ]


def _goose_positions(farm):
    half = max(1, len(farm.get("tiles", [])) // 2)
    return (half - 1, half - 1), (half + 1, half - 1)


def _goose_position(farm):
    return _goose_positions(farm)[0]


def _goose_tile(farm):
    x, y = _goose_position(farm)
    return farm["tiles"][y][x]


def _goose_stock(farm, private):
    placed = 0
    for x, y in _goose_positions(farm):
        tile = farm["tiles"][y][x]
        placed += int(
            isinstance(tile, dict) and tile.get("animal") == "GOOSE"
        )
    return placed + _item_stock(private, "GOOSE")


def _goose_exists(farm, private):
    return _goose_stock(farm, private) > 0


def _routing_mode(features):
    return (
        features["routing_mode"]
        if features["safe_routing"]
        else "direct"
    )


def _route_action(obs, farm, source, target, action, features):
    distance, move = _route(
        farm["tiles"],
        tuple(source),
        tuple(target),
        _routing_mode(features),
    )
    if distance is None:
        return None
    if (
        features.get("contract_horizon")
        and int(obs.get("hour", 0)) + distance >= 24
    ):
        return None
    return action if distance == 0 else move


def _nearest_shed_route(farm, source, features):
    routes = []
    for target in _shed_access(farm):
        distance, move = _route(
            farm["tiles"],
            tuple(source),
            target,
            _routing_mode(features),
        )
        if distance is not None:
            routes.append((distance, target[1], target[0], move))
    return min(routes) if routes else None


def _fib(index):
    a, b = 1, 1
    for _ in range(max(0, int(index))):
        a, b = b, a + b
    return a


class _MarketLedger:
    def __init__(
        self,
        obs,
        farm,
        private,
        max_orders=MAX_MARKET_ORDERS,
        hire_mult=FARM_HAND_COST_MULT,
    ):
        self.obs = obs
        self.cash = float(farm.get("money", 0))
        self.shed = dict(private.get("shed", {}) or {})
        self.seeds = dict(private.get("seeds", {}) or {})
        self.market_inventory = dict(
            ((obs.get("market", {}) or {}).get("inventory", {}) or {})
        )
        self.unlocked = len(farm.get("unlocked_quadrants", []) or [])
        self.hires_today = int(farm.get("hires_today", 0))
        self.hands = len(farm.get("hands", []) or [])
        self.order_count = 0
        self.max_orders = max(1, int(max_orders))
        self.hire_mult = int(hire_mult)

    def copy(self):
        other = object.__new__(_MarketLedger)
        other.obs = self.obs
        other.cash = self.cash
        other.shed = dict(self.shed)
        other.seeds = dict(self.seeds)
        other.market_inventory = dict(self.market_inventory)
        other.unlocked = self.unlocked
        other.hires_today = self.hires_today
        other.hands = self.hands
        other.order_count = self.order_count
        other.max_orders = self.max_orders
        other.hire_mult = self.hire_mult
        return other

    def _adopt(self, other):
        self.cash = other.cash
        self.shed = other.shed
        self.seeds = other.seeds
        self.market_inventory = other.market_inventory
        self.unlocked = other.unlocked
        self.hires_today = other.hires_today
        self.hands = other.hands
        self.order_count = other.order_count

    def _quantity(self, order):
        if not isinstance(order, list) or len(order) < 3:
            return 0
        try:
            quantity = int(order[2])
        except (TypeError, ValueError):
            return 0
        return max(0, quantity)

    def _commit_unit(self, op, item, reserve=0):
        if op == "SELL":
            if item not in self.market_inventory or self.shed.get(item, 0) <= 0:
                return False
            inventory = self.market_inventory[item]
            price = _price_at(self.obs, item, inventory)
            self.shed[item] -= 1
            self.cash += price
            if price > 1:
                self.market_inventory[item] = inventory + 1
            return True
        if op == "BUY_PRODUCT":
            if item not in ("WHEAT", "FERTILIZER"):
                return False
            inventory = int(self.market_inventory.get(item, 0))
            price = _price_at(self.obs, item, inventory - 1)
            if self.cash - price < reserve:
                return False
            self.cash -= price
            self.shed[item] = self.shed.get(item, 0) + 1
            self.market_inventory[item] = inventory - 1
            return True
        if op == "BUY_SEED":
            if item not in CROP_RULES:
                return False
            price = CROP_RULES[item][0]
            if self.cash - price < reserve:
                return False
            self.cash -= price
            self.seeds[item] = self.seeds.get(item, 0) + 1
            return True
        if op == "BUY_ANIMAL":
            if item not in ANIMAL_COSTS:
                return False
            price = ANIMAL_COSTS[item]
            if self.cash - price < reserve:
                return False
            self.cash -= price
            self.shed[item] = self.shed.get(item, 0) + 1
            return True
        return False

    def apply(self, order, reserve=0):
        if self.order_count >= self.max_orders:
            return 0
        self.order_count += 1
        if not isinstance(order, list) or not order:
            return 0
        op = order[0]
        if op == "HIRE":
            price = self.hire_mult * _fib(self.hires_today)
            if self.cash - price < reserve:
                return 0
            self.cash -= price
            self.hires_today += 1
            self.hands += 1
            return 1
        if op == "BUY_LAND":
            extra = self.unlocked - 1
            if extra < 0 or extra >= len(LAND_PRICES):
                return 0
            price = LAND_PRICES[extra]
            if self.cash - price < reserve:
                return 0
            self.cash -= price
            self.unlocked += 1
            return 1
        quantity = self._quantity(order)
        if quantity <= 0 or len(order) < 2:
            return 0
        committed = 0
        for _ in range(quantity):
            if not self._commit_unit(op, order[1], reserve):
                break
            committed += 1
        return committed

    def affordable_quantity(self, op, item, limit, reserve=0):
        if self.order_count >= self.max_orders:
            return 0
        trial = self.copy()
        committed = 0
        for _ in range(max(0, int(limit))):
            if not trial._commit_unit(op, item, reserve):
                break
            committed += 1
        return committed

    def try_apply_full(self, order, reserve=0):
        trial = self.copy()
        committed = trial.apply(order, reserve)
        expected = 1 if order and order[0] in ("HIRE", "BUY_LAND") else trial._quantity(order)
        if committed != expected or committed <= 0:
            return False
        self._adopt(trial)
        return True


def _project_market_orders(
    obs,
    farm,
    private,
    orders,
    max_orders=MAX_MARKET_ORDERS,
    hire_mult=FARM_HAND_COST_MULT,
):
    ledger = _MarketLedger(
        obs,
        farm,
        private,
        max_orders,
        hire_mult,
    )
    committed = [ledger.apply(order) for order in orders]
    return ledger, committed


def _fertilizer_candidates(obs, farm, features, require_watered=True):
    day = int(obs.get("day", 0))
    if day >= 29:
        return []
    fertilizer_price = int(
        ((obs.get("market", {}) or {}).get("prices", {}) or {}).get(
            "FERTILIZER",
            100,
        )
    )
    extra_units = {"TOMATO": 3, "STRAWBERRY": 2, "MELON": 0}
    candidates = []

    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if (
                not isinstance(tile, dict)
                or tile.get("kind") != "PLANT"
            ):
                continue
            crop = tile.get("crop")
            planted = int(tile.get("planted_day", day))
            age = day - planted
            enabled = (
                (
                    crop == "TOMATO"
                    and features["fertilize_tomato"]
                    and age == 7
                )
                or (
                    crop == "STRAWBERRY"
                    and (
                        (
                            features["fertilize_strawberry"]
                            and age == 9
                        )
                        or (
                            features["fertilize_strawberry_second"]
                            and age == 13
                        )
                    )
                )
                or (
                    crop == "MELON"
                    and features["fertilize_melon"]
                    and age == 6
                )
            )
            if not enabled:
                continue
            if (
                require_watered
                and not tile.get("watered_today", False)
            ):
                continue
            if int(tile.get("fertilized_until_day", -1)) >= day:
                continue
            if crop == "MELON":
                held = int(tile.get("yield_units", 0))
                normal = max(0, 10 - max(6, age) + 1)
                extra_units[crop] = max(0, 6 - held - normal)
            route = _nearest_shed_route(farm, (x, y), features)
            if route is None or 2 * route[0] + 2 > 23:
                continue
            harvest_step = (day + 1) * 24
            expiry = int(tile.get("max_lifespan_step", -1))
            if day + 1 > 28 or (expiry >= 0 and harvest_step >= expiry):
                continue
            product_price = int(
                ((obs.get("market", {}) or {}).get("prices", {}) or {}).get(
                    crop,
                    MARKET_RULES[crop][0],
                )
            )
            value = extra_units[crop] * product_price
            if value >= fertilizer_price * float(
                features["fertilizer_value_ratio"]
            ):
                candidates.append(
                    (-value, route[0], y, x)
                )
    candidates.sort()
    return [(item[3], item[2]) for item in candidates]


def _animal_tiles(farm, features):
    limit = max(1, int(features.get("target_geese", 1)))
    result = []
    for position in _goose_positions(farm)[:limit]:
        x, y = position
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("animal") == "GOOSE":
            result.append((position, tile))
    return result


def _shed_action(farm, position, action, features):
    route = _nearest_shed_route(farm, position, features)
    if route is None:
        return None
    return action if route[0] == 0 else route[3]


def _goose_action(obs, farm, private, features):
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    position = tuple(farm.get("farmer", [0, 0]))
    limit = max(1, int(features.get("target_geese", 1)))
    targets = _goose_positions(farm)[:limit]
    inventory = _inventory(private, 0)

    for target in targets:
        x, y = target
        tile = farm["tiles"][y][x]
        if tile is None:
            return _route_action(
                obs,
                farm,
                position,
                target,
                ["BUILD_COOP"],
                features,
            )
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return _route_action(
                obs,
                farm,
                position,
                target,
                ["DIG"],
                features,
            )

    cutoff = int(features.get("second_goose_place_cutoff", 16))
    for index, target in enumerate(targets):
        x, y = target
        tile = farm["tiles"][y][x]
        if not isinstance(tile, dict) or tile.get("kind") != "COOP":
            continue
        if tile.get("animal") == "GOOSE":
            continue
        if index and hour > cutoff:
            if int(inventory.get("GOOSE", 0)) > 0:
                return _shed_action(
                    farm,
                    position,
                    ["DROP"],
                    features,
                )
            continue
        if int(inventory.get("GOOSE", 0)) > 0:
            return _route_action(
                obs,
                farm,
                position,
                target,
                ["PLACE", "GOOSE"],
                features,
            )
        if int((private.get("shed", {}) or {}).get("GOOSE", 0)) > 0:
            return _shed_action(
                farm,
                position,
                ["PICKUP", "GOOSE", 1],
                features,
            )

    animals = _animal_tiles(farm, features)
    if day >= 29:
        if _stock(inventory) > 0:
            route = _nearest_shed_route(farm, position, features)
            if route is None or hour + route[0] > 22:
                return None
            return ["DROP"] if route[0] == 0 else route[3]
        for target, tile in animals:
            if tile.get("fertilizer_available", False):
                return _route_action(
                    obs,
                    farm,
                    position,
                    target,
                    ["COLLECT_FERTILIZER"],
                    features,
                )
            if int(tile.get("yield_units", 0)) > 0:
                return _route_action(
                    obs,
                    farm,
                    position,
                    target,
                    ["HARVEST"],
                    features,
                )
        return None

    unfed = [
        (target, tile)
        for target, tile in animals
        if not tile.get("fed_today", False)
    ]
    if unfed:
        target, _ = unfed[0]
        if int(inventory.get("WHEAT", 0)) > 0:
            return _route_action(
                obs,
                farm,
                position,
                target,
                ["FEED"],
                features,
            )
        shed_wheat = int((private.get("shed", {}) or {}).get("WHEAT", 0))
        if shed_wheat > 0:
            return _shed_action(
                farm,
                position,
                ["PICKUP", "WHEAT", min(len(unfed), shed_wheat)],
                features,
            )

    fertilizer_targets = _fertilizer_candidates(obs, farm, features)
    if int(inventory.get("FERTILIZER", 0)) > 0 and fertilizer_targets:
        return _route_action(
            obs,
            farm,
            position,
            fertilizer_targets[0],
            ["FERTILIZE"],
            features,
        )

    for target, tile in animals:
        if tile.get("fed_today", False) and not tile.get("cared_today", False):
            return _route_action(
                obs,
                farm,
                position,
                target,
                ["CARE"],
                features,
            )

    for target, tile in animals:
        if tile.get("fertilizer_available", False):
            return _route_action(
                obs,
                farm,
                position,
                target,
                ["COLLECT_FERTILIZER"],
                features,
            )

    if (
        fertilizer_targets
        and int((private.get("shed", {}) or {}).get("FERTILIZER", 0)) > 0
    ):
        return _shed_action(
            farm,
            position,
            ["PICKUP", "FERTILIZER", 1],
            features,
        )

    for target, tile in animals:
        if int(tile.get("yield_units", 0)) >= 4 or (
            day >= 27 and int(tile.get("yield_units", 0)) > 0
        ):
            return _route_action(
                obs,
                farm,
                position,
                target,
                ["HARVEST"],
                features,
            )

    if _stock(inventory) > 0:
        return _shed_action(farm, position, ["DROP"], features)
    return None


def _shed_routes(
    farm,
    positions,
    private,
    obs,
    features,
    protected=(),
    projected_units=0,
):
    threshold = features.get("storage_threshold")
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    terminal = bool(features.get("terminal_return")) and day >= 29
    if threshold is None and not terminal:
        return {}

    tiles = farm["tiles"]
    size = len(tiles)
    half = size // 2
    access = [
        (half - 1, half - 1),
        (half, half - 1),
        (half - 1, half),
        (half, half),
    ]
    access = [
        position
        for position in access
        if tiles[position[1]][position[0]] != "LOCKED"
    ]
    inventories = private.get("inventories", []) or []
    pressure = _stock(private.get("shed", {})) + sum(
        _stock(inventory) for inventory in inventories
    )
    forced = {}
    candidates = []

    for worker, source in enumerate(positions):
        if worker in protected and not terminal:
            continue
        inventory = inventories[worker] if worker < len(inventories) else {}
        amount = _stock(inventory)
        if amount <= 0:
            continue

        routes = []
        for target in access:
            distance, move = _route(
                tiles,
                tuple(source),
                target,
                (
                    features["routing_mode"]
                    if features["safe_routing"]
                    else "direct"
                ),
            )
            if distance is not None:
                routes.append((distance, target[1], target[0], move))
        if not routes:
            continue

        distance, _, _, move = min(routes)
        if (
            features.get("contract_horizon")
            and hour + distance > 23
        ):
            continue
        candidates.append(
            (
                distance,
                -amount,
                worker,
                amount,
                ["DROP"] if distance == 0 else move,
            )
        )

    if terminal:
        return {item[2]: item[4] for item in candidates}
    if threshold is None:
        return {}
    projected_relief = bool(
        features.get("projected_wave_recall")
        and pressure + projected_units > 90
    )
    if pressure < int(threshold) and not projected_relief:
        return {}
    if features.get("recall_all"):
        return {item[2]: item[4] for item in candidates}

    needed = pressure - max(0, int(threshold) - 10)
    covered = 0
    for _, _, worker, amount, action in sorted(candidates):
        forced[worker] = action
        covered += amount
        if covered >= needed:
            break
    return forced


def _unit_actions(obs, farm, private, plan, features):
    positions = [farm["farmer"], *farm.get("hands", [])]
    step = int(
        obs.get(
            "step",
            int(obs.get("day", 0)) * 24 + int(obs.get("hour", 0)),
        )
    )
    hour = int(obs.get("hour", step % 24))
    routing_mode = (
        features["routing_mode"] if features["safe_routing"] else "direct"
    )
    tasks = _candidate_tasks(obs, farm, private, plan, features)
    preliminary_pairs = _pair_data(
        farm,
        positions,
        tasks,
        routing_mode,
        step,
        hour,
        features.get("expiry_slack"),
        features.get("expiry_slack_mode", "pair"),
        features.get("retire_slack", 2),
        features.get("retire_opportunistic", True),
        features.get("contract_horizon", False),
    )
    if features["global_matching"]:
        preliminary = _select_assign(
            positions,
            tasks,
            preliminary_pairs,
            features.get("stable_assignment", False),
        )
    else:
        preliminary = _greedy_assign(positions, tasks, preliminary_pairs)
    protected = {
        worker
        for worker, task_idx in enumerate(preliminary)
        if task_idx >= 0
        and (
            tasks[task_idx].get("urgent")
            or (
                preliminary_pairs[(worker, task_idx)][3]
                and (
                    not features.get("projected_wave_recall")
                    or preliminary_pairs[(worker, task_idx)][4] <= 2
                )
            )
            or (
                features.get("retire_assignment_protect")
                and tasks[task_idx].get("retirement_harvest")
            )
        )
    }
    wave_window = int(features.get("wave_window", 16))
    projected_units = 0
    if features.get("projected_wave_recall"):
        for task in tasks:
            expiry = int(task.get("expiry", -1))
            if expiry >= 0 and expiry % 24 == 0:
                expiry -= 1
            if (
                (
                    task.get("final_annual")
                    or task.get("retirement_harvest")
                )
                and 0 <= expiry - step <= wave_window
            ):
                projected_units += int(task.get("units", 0))
    forced = _shed_routes(
        farm,
        positions,
        private,
        obs,
        features,
        protected,
        projected_units,
    )
    active_workers = [
        worker for worker in range(len(positions)) if worker not in forced
    ]
    active_positions = [positions[worker] for worker in active_workers]
    if not forced:
        pairs = preliminary_pairs
        assignments = preliminary
    else:
        pairs = _pair_data(
            farm,
            active_positions,
            tasks,
            routing_mode,
            step,
            hour,
            features.get("expiry_slack"),
            features.get("expiry_slack_mode", "pair"),
            features.get("retire_slack", 2),
            features.get("retire_opportunistic", True),
            features.get("contract_horizon", False),
        )
        if features["global_matching"]:
            assignments = _select_assign(
                active_positions,
                tasks,
                pairs,
                features.get("stable_assignment", False),
            )
        else:
            assignments = _greedy_assign(active_positions, tasks, pairs)

    actions = [["PASS"] for _ in positions]
    for worker, action in forced.items():
        actions[worker] = action
    immediate_harvests = []
    for local_worker, task_idx in enumerate(assignments):
        worker = active_workers[local_worker]
        if task_idx < 0:
            continue
        task = tasks[task_idx]
        _, distance, move, deadline_urgent, _ = pairs[
            (local_worker, task_idx)
        ]
        if (
            not distance
            and task["action"][0] == "HARVEST"
            and features.get("storage_cap") is not None
        ):
            price = int(
                ((obs.get("market", {}) or {}).get("prices", {}) or {}).get(
                    task["crop"],
                    0,
                )
            )
            immediate_harvests.append(
                (
                    (
                        0
                        if (
                            task.get("expiry_urgent")
                            or deadline_urgent
                            or task.get("urgent")
                        )
                        else 1
                    ),
                    -(price * task["units"]),
                    worker,
                    task,
                    deadline_urgent,
                )
            )
            continue
        actions[worker] = move if distance else task["action"]

    if features.get("storage_cap") is not None:
        pressure = _stock(private.get("shed", {})) + sum(
            _stock(item)
            for item in (private.get("inventories", []) or [])
        )
        current_pressure = pressure
        for _, _, worker, task, deadline_urgent in sorted(
            immediate_harvests
        ):
            limit = (
                int(features.get("urgent_storage_cap", 99))
                if (
                    task.get("expiry_urgent")
                    or deadline_urgent
                    or task.get("urgent")
                )
                else int(features["storage_cap"])
            )
            if current_pressure + task["units"] <= limit:
                actions[worker] = task["action"]
                current_pressure += task["units"]
    return actions


def _terminal_tasks(obs, farm):
    day = int(obs.get("day", 0))
    tasks = []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if (
                not isinstance(tile, dict)
                or tile.get("kind") != "PLANT"
                or int(tile.get("yield_units", 0)) <= 0
            ):
                continue
            crop = tile.get("crop")
            first_day = CROP_RULES.get(crop, (0, 0))[1]
            if day - int(tile.get("planted_day", day)) < first_day:
                continue
            tasks.append(
                {
                    "priority": 0,
                    "position": (x, y),
                    "action": ["HARVEST"],
                }
            )
    return tasks


def _terminal_unit_actions(obs, farm, private, features):
    hour = int(obs.get("hour", 0))
    positions = [farm["farmer"], *farm.get("hands", [])]
    inventories = list(private.get("inventories", []) or [])
    while len(inventories) < len(positions):
        inventories.append({})
    actions = [None] * len(positions)
    available = []

    for worker, source in enumerate(positions):
        if _stock(inventories[worker]) <= 0:
            available.append(worker)
            continue
        route = _nearest_shed_route(farm, source, features)
        if route is None or hour + route[0] > 22:
            actions[worker] = ["PASS"]
        else:
            actions[worker] = ["DROP"] if route[0] == 0 else route[3]

    tasks = _terminal_tasks(obs, farm)
    open_positions = [positions[worker] for worker in available]
    pairs = {}
    remaining = 23 - hour
    for local_worker, source in enumerate(open_positions):
        for task_idx, task in enumerate(tasks):
            distance, move = _route(
                farm["tiles"],
                tuple(source),
                task["position"],
                _routing_mode(features),
            )
            if distance is None or 2 * distance + 2 > remaining:
                continue
            cost = (
                distance * 100
                + task["position"][1] * len(farm["tiles"])
                + task["position"][0]
            )
            pairs[(local_worker, task_idx)] = (
                cost,
                distance,
                move,
            )
    assignments = _select_assign(
        open_positions,
        tasks,
        pairs,
        features.get("stable_assignment", False),
    )

    for local_worker, task_idx in enumerate(assignments):
        worker = available[local_worker]
        if task_idx < 0:
            actions[worker] = ["PASS"]
            continue
        task = tasks[task_idx]
        _, distance, move = pairs[(local_worker, task_idx)]
        actions[worker] = move if distance else task["action"]
    return [action or ["PASS"] for action in actions]


def _integrated_unit_actions(obs, farm, private, plan, features):
    day = int(obs.get("day", 0))
    goose_action = (
        _goose_action(obs, farm, private, features)
        if features["goose"]
        else None
    )
    terminal = features["terminal_ops"] and day >= 29

    if goose_action is None:
        actions = (
            _terminal_unit_actions(obs, farm, private, features)
            if terminal
            else _unit_actions(obs, farm, private, plan, features)
        )
        return actions, False

    farm_view, private_view = _hands_view(farm, private)
    if farm_view is None:
        return [goose_action], True
    hand_actions = (
        _terminal_unit_actions(
            obs,
            farm_view,
            private_view,
            features,
        )
        if terminal
        else _unit_actions(
            obs,
            farm_view,
            private_view,
            plan,
            features,
        )
    )
    return [goose_action, *hand_actions], True


def _seed_needs(farm, private, plan, include_locked_ne):
    half = len(farm["tiles"]) // 2
    needs = {}
    for (x, y), crop in plan.items():
        tile = farm["tiles"][y][x]
        available = tile is None or (
            include_locked_ne and x >= half and tile == "LOCKED"
        )
        if available:
            needs[crop] = needs.get(crop, 0) + 1

    seeds = private.get("seeds", {}) or {}
    return {
        crop: max(0, amount - int(seeds.get(crop, 0)))
        for crop, amount in needs.items()
    }


def _horizon_seed_needs(obs, farm, private, plan, include_locked_ne):
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    half = len(farm["tiles"]) // 2
    needs = {}
    if hour > 20:
        return needs

    for (x, y), crop in plan.items():
        tile = farm["tiles"][y][x]
        available = tile is None or (
            include_locked_ne and x >= half and tile == "LOCKED"
        )
        if not available or day > CROP_RULES[crop][3]:
            continue
        needs[crop] = needs.get(crop, 0) + 1
    seeds = private.get("seeds", {}) or {}
    return {
        crop: max(0, amount - int(seeds.get(crop, 0)))
        for crop, amount in needs.items()
    }


def _assignment_seed_needs(
    obs,
    farm,
    private,
    plan,
    features,
    farmer_busy,
    distance_zero,
):
    if int(obs.get("hour", 0)) > 20:
        return {}
    unit_farm = farm
    unit_private = private
    if farmer_busy:
        unit_farm, unit_private = _hands_view(farm, private)
        if unit_farm is None:
            return {}
    positions = [
        unit_farm["farmer"],
        *unit_farm.get("hands", []),
    ]
    synthetic = dict(unit_private)
    synthetic["seeds"] = {
        crop: len(plan) for crop in CROP_RULES
    }
    tasks = _candidate_tasks(
        obs,
        unit_farm,
        synthetic,
        plan,
        features,
    )
    step = int(
        obs.get(
            "step",
            int(obs.get("day", 0)) * 24
            + int(obs.get("hour", 0)),
        )
    )
    hour = int(obs.get("hour", step % 24))
    pairs = _pair_data(
        unit_farm,
        positions,
        tasks,
        _routing_mode(features),
        step,
        hour,
        features.get("expiry_slack"),
        features.get("expiry_slack_mode", "pair"),
        features.get("retire_slack", 2),
        features.get("retire_opportunistic", True),
        features.get("contract_horizon", False),
    )
    assignments = _select_assign(
        positions,
        tasks,
        pairs,
        features.get("stable_assignment", False),
    )
    desired = {}
    for worker, task_idx in enumerate(assignments):
        if task_idx < 0:
            continue
        task = tasks[task_idx]
        action = task["action"]
        if action[0] != "PLANT":
            continue
        distance = pairs[(worker, task_idx)][1]
        if distance_zero and distance != 0:
            continue
        crop = action[1]
        desired[crop] = desired.get(crop, 0) + 1
    seeds = private.get("seeds", {}) or {}
    return {
        crop: max(0, amount - int(seeds.get(crop, 0)))
        for crop, amount in desired.items()
    }


def _land_day(features):
    value = features.get("land_day")
    if value is None:
        return 0 if features["early_ne"] else 11
    return int(value)


def _visible_opponent_production(obs, item):
    player = int(obs.get("player", 0))
    product_animal = {
        "EGG": "GOOSE",
        "MILK": "COW",
        "WOOL": "SHEEP",
    }
    count = 0
    for index, other in enumerate(obs.get("farms", []) or []):
        if index == player:
            continue
        for row in other.get("tiles", []) or []:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                if (
                    tile.get("kind") == "PLANT"
                    and tile.get("crop") == item
                ):
                    count += 1
                if tile.get("animal") == product_animal.get(item):
                    count += 1
    return count


def _town_demand_per_day(obs, item):
    day = int(obs.get("day", 0))
    demand = 2 * (4 if day >= 20 else 2 if day >= 10 else 1)
    shop_products = {
        "BAKERY": ("EGG", "WHEAT"),
        "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
        "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
        "YARN_STORE": ("WOOL",),
        "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
        "PET_CAFE": ("CARROT",),
        "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
        "FARMERS_MARKET": (
            "WHEAT",
            "CARROT",
            "TOMATO",
            "STRAWBERRY",
        ),
    }
    for shop in (obs.get("town", {}) or {}).get(
        "unlocked_shops",
        [],
    ) or []:
        products = shop_products.get(shop, ())
        if item in products:
            demand += 12 if len(products) == 1 else 6
    return demand


def _days_to_next_supply(farm, item, day):
    waits = []
    for row in farm.get("tiles", []) or []:
        for tile in row:
            if (
                isinstance(tile, dict)
                and tile.get("kind") == "PLANT"
                and tile.get("crop") == item
            ):
                age = day - int(tile.get("planted_day", day))
                waits.append(max(1, CROP_RULES[item][1] - age))
    return min(waits) if waits else min(10, max(1, 29 - day))


def _sale_regime(obs, farm, item, occupancy, features):
    day = int(obs.get("day", 0))
    if item == "FERTILIZER" or day >= 27 or occupancy >= 70:
        return "RACE"
    if not features["hold_scarce"]:
        return "RACE"
    inventory = int(
        ((obs.get("market", {}) or {}).get("inventory", {}) or {}).get(
            item,
            10000,
        )
    )
    if (
        _visible_opponent_production(obs, item) > 0
        or inventory >= MARKET_RULES[item][1]
    ):
        return "RACE"
    return "HOLD"


def _sale_quantity(
    obs,
    farm,
    item,
    available,
    occupancy,
    features,
    force=False,
):
    available = int(available)
    if available <= 0:
        return 0
    if force:
        return available
    inventory = int(
        ((obs.get("market", {}) or {}).get("inventory", {}) or {}).get(
            item,
            10000,
        )
    )
    profitable = 0
    for offset in range(available):
        if _price_at(obs, item, inventory + offset) <= 1:
            break
        profitable += 1
    if profitable <= 0:
        return 0
    if item == "MELON" and features["pace_melons"]:
        horizon = _days_to_next_supply(
            farm,
            item,
            int(obs.get("day", 0)),
        )
        demand = _town_demand_per_day(obs, item)
        opponent_front_run = min(
            12,
            _visible_opponent_production(obs, item),
        )
        paced = int(
            math.ceil(
                demand
                * horizon
                * float(features["melon_demand_share"])
            )
        ) + opponent_front_run
        headroom_relief = max(0, occupancy - 80)
        return min(
            profitable,
            max(1, paced, headroom_relief),
        )
    if _sale_regime(obs, farm, item, occupancy, features) == "RACE":
        return profitable
    if int(obs.get("day", 0)) >= 22:
        return profitable
    if occupancy >= 60:
        return min(profitable, max(1, occupancy - 50))
    return 0


def _sale_revenue(obs, item, quantity):
    inventory = int(
        ((obs.get("market", {}) or {}).get("inventory", {}) or {}).get(
            item,
            10000,
        )
    )
    return sum(
        _price_at(obs, item, inventory + offset)
        for offset in range(quantity)
    )


def _projected_storage(
    farm,
    private,
    unit_actions,
    capacity=SHED_CAPACITY,
):
    shed = dict(private.get("shed", {}) or {})
    inventories = [
        dict(inventory or {})
        for inventory in private.get("inventories", []) or []
    ]
    positions = [farm["farmer"], *farm.get("hands", [])]
    access = set(_shed_access(farm))
    for worker, action in enumerate(unit_actions):
        if (
            worker >= len(inventories)
            or worker >= len(positions)
            or not action
            or tuple(positions[worker]) not in access
        ):
            continue
        if action[0] == "PICKUP" and len(action) >= 2:
            item = action[1]
            requested = int(action[2]) if len(action) >= 3 else 1
            take = min(
                max(0, requested),
                int(shed.get(item, 0)),
            )
            if take > 0:
                shed[item] = int(shed.get(item, 0)) - take
                inventories[worker][item] = (
                    int(inventories[worker].get(item, 0)) + take
                )
            continue
        if action[0] != "DROP":
            continue
        for item, amount in inventories[worker].items():
            room = max(0, int(capacity) - _stock(shed))
            take = min(max(0, int(amount)), room)
            if take > 0:
                shed[item] = shed.get(item, 0) + take
        inventories[worker] = {}
    return shed, inventories


def _market_actions(
    obs,
    farm,
    private,
    plan,
    features,
    unit_actions,
    farmer_busy,
):
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    shed, inventories = _projected_storage(
        farm,
        private,
        unit_actions,
    )
    occupancy = _stock(shed)
    carried = sum(_stock(item) for item in inventories)
    pressure = occupancy + carried
    force = day >= 29 or pressure >= 90
    orders = []
    market_private = dict(private)
    market_private["shed"] = shed
    market_private["inventories"] = inventories
    ledger = _MarketLedger(obs, farm, market_private)
    fertilizer_reserve = (
        min(
            len(
                _fertilizer_candidates(
                    obs,
                    farm,
                    features,
                    require_watered=False,
                )
            ),
            int(shed.get("FERTILIZER", 0)),
        )
        if day < 29
        else 0
    )
    carried_wheat = sum(
        int(inventory.get("WHEAT", 0))
        for inventory in inventories
    )
    goose_tile = _goose_tile(farm) if features["goose"] else None
    total_wheat = int(shed.get("WHEAT", 0)) + carried_wheat
    feed_due = int(
        isinstance(goose_tile, dict)
        and goose_tile.get("animal") == "GOOSE"
        and day < 29
        and not goose_tile.get("fed_today", False)
        and total_wheat > 0
    )
    carried_wheat_after_feed = max(0, carried_wheat - feed_due)

    for item in SELL_ORDER:
        amount = int(shed.get(item, 0))
        if item == "WHEAT" and features["goose"] and day < 29:
            amount -= max(
                0,
                WHEAT_RESERVE - carried_wheat_after_feed,
            )
        if item == "FERTILIZER":
            amount -= fertilizer_reserve
        if amount <= 0 or len(orders) >= 10:
            continue
        quantity = (
            _sale_quantity(
                obs,
                farm,
                item,
                amount,
                pressure,
                features,
                force,
            )
            if features["computed_sales"]
            else (
                amount
                if day >= 29 or item != "MELON"
                else min(amount, 2)
            )
        )
        if quantity > 0:
            order = ["SELL", item, quantity]
            if ledger.try_apply_full(order):
                orders.append(order)
                occupancy -= quantity
                pressure -= quantity

    if day >= 29:
        return orders

    unlocked = "NE" in farm.get("unlocked_quadrants", [])
    wants_land = not unlocked and day >= _land_day(features)
    locked_ne_needs = (
        _horizon_seed_needs(obs, farm, private, plan, True)
        if features["horizon_purchases"]
        else _seed_needs(farm, private, plan, True)
    )
    seed_cost = sum(
        CROP_RULES[crop][0] * amount
        for crop, amount in locked_ne_needs.items()
    )
    buy_land = wants_land and ledger.try_apply_full(
        ["BUY_LAND"],
        seed_cost + LAND_RESERVE,
    )
    if buy_land:
        orders.append(["BUY_LAND"])

    if (
        features["goose"]
        and day <= 2
        and not _goose_exists(farm, private)
        and ledger.try_apply_full(
            ["BUY_ANIMAL", "GOOSE", 1],
            200,
        )
    ):
        orders.append(["BUY_ANIMAL", "GOOSE", 1])
        occupancy += 1

    if features["goose"]:
        wheat_missing = max(
            0,
            WHEAT_RESERVE + feed_due - total_wheat,
        )
        regular = ledger.affordable_quantity(
            "BUY_PRODUCT",
            "WHEAT",
            wheat_missing,
            150,
        )
        emergency = (
            ledger.affordable_quantity(
                "BUY_PRODUCT",
                "WHEAT",
                min(1, wheat_missing),
            )
            if total_wheat <= 0
            else 0
        )
        affordable = min(
            wheat_missing,
            max(regular, emergency),
            max(0, 100 - occupancy),
        )
        order = ["BUY_PRODUCT", "WHEAT", affordable]
        if affordable > 0 and ledger.try_apply_full(order):
            orders.append(order)
            occupancy += affordable

    needs = (
        _horizon_seed_needs(
            obs,
            farm,
            private,
            plan,
            unlocked or buy_land,
        )
        if features["horizon_purchases"]
        else _seed_needs(
            farm,
            private,
            plan,
            unlocked or buy_land,
        )
    )
    if features["all_season_jit"]:
        needs = _assignment_seed_needs(
            obs,
            farm,
            private,
            plan,
            features,
            farmer_busy,
            False,
        )
    if features["horizon_purchases"]:
        immediate = _assignment_seed_needs(
            obs,
            farm,
            private,
            plan,
            features,
            farmer_busy,
            True,
        )
        for crop in CROP_RULES:
            tight_day = CROP_RULES[crop][3] - int(
                crop in ("MELON", "TOMATO", "STRAWBERRY")
                or (
                    crop == "WHEAT"
                    and features["tight_wheat_early"]
                )
            )
            if day >= tight_day:
                needs[crop] = immediate.get(crop, 0)
            if (
                crop in ("MELON", "TOMATO", "STRAWBERRY")
                and day == CROP_RULES[crop][3]
                and hour > 14
            ):
                needs[crop] = 0

    purchased = {}
    for crop in ("MELON", "CARROT", "WHEAT", "TOMATO", "STRAWBERRY"):
        wanted = needs.get(crop, 0)
        amount = ledger.affordable_quantity(
            "BUY_SEED",
            crop,
            wanted,
            LAND_RESERVE,
        )
        order = ["BUY_SEED", crop, amount]
        if amount > 0 and ledger.try_apply_full(order):
            orders.append(order)
            purchased[crop] = amount

    pending_geese = sum(
        int(order[2]) if len(order) >= 3 else 1
        for order in orders
        if len(order) >= 2 and order[:2] == ["BUY_ANIMAL", "GOOSE"]
    )
    needs_second = _goose_stock(farm, private) + pending_geese < 2
    if (
        features["goose"]
        and int(features.get("target_geese", 1)) >= 2
        and int(features.get("second_goose_buy_day", 3)) <= day
        <= int(features.get("second_goose_buy_deadline", 8))
        and float(farm.get("money", 0)) >= GOOSE_COST + 300
        and needs_second
        and ledger.try_apply_full(
            ["BUY_ANIMAL", "GOOSE", 1],
            300,
        )
    ):
        orders.append(["BUY_ANIMAL", "GOOSE", 1])

    target_hands = int(features["target_hands"])
    can_hire = day < 29
    if features["horizon_purchases"]:
        task_private = dict(private)
        task_private["seeds"] = dict(private.get("seeds", {}) or {})
        for crop, amount in purchased.items():
            task_private["seeds"][crop] = (
                int(task_private["seeds"].get(crop, 0)) + amount
            )
        tasks = _candidate_tasks(
            obs,
            farm,
            task_private,
            plan,
            features,
        )
        crop_farmer = 0 if farmer_busy else 1
        target_hands = min(
            target_hands,
            max(0, len(tasks) - crop_farmer),
        )
        can_hire = hour <= 20 and bool(tasks)
    if (
        can_hire
        and len(farm.get("hands", [])) < target_hands
    ):
        missing = target_hands - len(farm.get("hands", []))
        hires = 1 if features["staggered_hiring"] else missing
        for _ in range(hires):
            if not ledger.try_apply_full(["HIRE"]):
                break
            orders.append(["HIRE"])
    return orders


import math
from dataclasses import dataclass


PRODUCTS = tuple(MARKET_RULES)
CROP_PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": (
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
    ),
}
TOWN_CENTER_PRODUCTS = tuple(
    product for product in PRODUCTS if product != "FERTILIZER"
)
TOWN_CENTER_DEMAND_SCHEDULE = ((20, 4), (10, 2), (0, 1))
PRODUCT_ANIMAL = {
    "EGG": "GOOSE",
    "MILK": "COW",
    "WOOL": "SHEEP",
}


@dataclass(frozen=True)
class ForecastSettings:
    turns_per_day: int = 24
    shop_sell_interval: int = 4
    town_center_sell_interval: int = 12
    episode_steps: int = 720
    opponent_units_per_source_day: float = 0.0


@dataclass(frozen=True)
class DemandForecast:
    product: str
    start_step: int
    stop_step: int
    revealed_shops: tuple
    town_center_events: int
    shop_events: int
    town_center_units: int
    shop_units: int
    total_units: int


@dataclass(frozen=True)
class OpponentSignal:
    product: str
    crop_sources: int
    animal_sources: int
    visible_field_units: int
    source_count: int
    pressure_units: int


@dataclass(frozen=True)
class ProductValue:
    product: str
    current_inventory: int
    current_marginal_value: int
    demand_units: int
    demand_only_inventory: int
    demand_only_marginal_value: int
    opponent_sources: int
    opponent_pressure_units: int
    forecast_inventory: int
    forecast_marginal_value: int


DEFAULT_FORECAST_SETTINGS = ForecastSettings()


def _settings(settings):
    return settings or DEFAULT_FORECAST_SETTINGS


def _step(obs, settings):
    turns_per_day = max(1, int(settings.turns_per_day))
    return int(
        obs.get(
            "step",
            int(obs.get("day", 0)) * turns_per_day
            + int(obs.get("hour", 0)),
        )
    )


def revealed_shops(obs):
    shops = (obs.get("town", {}) or {}).get("unlocked_shops", []) or []
    return tuple(shop for shop in shops if shop in SHOP_PRODUCTS)


def town_center_multiplier(day):
    day = int(day)
    return next(
        multiplier
        for threshold, multiplier in TOWN_CENTER_DEMAND_SCHEDULE
        if day >= threshold
    )


def forecast_demand(obs, horizon_steps, settings=None):
    settings = _settings(settings)
    horizon_steps = max(0, int(horizon_steps))
    start = max(0, _step(obs, settings))
    action_stop = max(0, int(settings.episode_steps) - 1)
    stop = min(action_stop, start + horizon_steps)
    turns_per_day = max(1, int(settings.turns_per_day))
    shop_interval = max(1, int(settings.shop_sell_interval))
    center_interval = max(1, int(settings.town_center_sell_interval))
    shops = revealed_shops(obs)
    rows = {
        product: {
            "town_center_events": 0,
            "shop_events": 0,
            "town_center_units": 0,
            "shop_units": 0,
        }
        for product in PRODUCTS
    }

    for event_step in range(start, stop):
        if event_step % center_interval == 0:
            multiplier = town_center_multiplier(
                event_step // turns_per_day
            )
            for product in TOWN_CENTER_PRODUCTS:
                rows[product]["town_center_events"] += 1
                rows[product]["town_center_units"] += multiplier
        if event_step % shop_interval == 0:
            for shop in shops:
                products = SHOP_PRODUCTS[shop]
                multiplier = 2 if len(products) == 1 else 1
                for product in products:
                    rows[product]["shop_events"] += 1
                    rows[product]["shop_units"] += multiplier

    return {
        product: DemandForecast(
            product=product,
            start_step=start,
            stop_step=stop,
            revealed_shops=shops,
            town_center_events=row["town_center_events"],
            shop_events=row["shop_events"],
            town_center_units=row["town_center_units"],
            shop_units=row["shop_units"],
            total_units=row["town_center_units"] + row["shop_units"],
        )
        for product, row in rows.items()
    }


def _market_shape(name, value):
    value = max(0.0, float(value))
    if name == "linear":
        return value
    if name == "sq":
        return value * value
    if name == "sqrt":
        return math.sqrt(value)
    if name == "log":
        return math.log1p(value)
    if name == "log10":
        return math.log10(1.0 + value)
    return value


def _market_rule(obs, product):
    rule = list(MARKET_RULES[product])
    patch = ((obs.get("market", {}) or {}).get("params", {}) or {}).get(
        product,
        {},
    )
    keys = (
        "base",
        "I0",
        "T",
        "below_func",
        "below_target",
        "above_func",
        "above_target",
    )
    for index, key in enumerate(keys):
        if key in patch:
            rule[index] = patch[key]
    return tuple(rule)


def price_at_inventory(obs, product, inventory):
    (
        base,
        target_inventory,
        throughput,
        below_shape,
        below_target,
        above_shape,
        above_target,
    ) = _market_rule(obs, product)
    inventory = int(inventory)
    if inventory < target_inventory:
        amplitude = below_target * base / max(
            1e-9,
            _market_shape(below_shape, throughput),
        )
        price = base + amplitude * _market_shape(
            below_shape,
            target_inventory - inventory,
        )
    else:
        amplitude = above_target * base / max(
            1e-9,
            _market_shape(above_shape, throughput),
        )
        price = base - amplitude * _market_shape(
            above_shape,
            inventory - target_inventory,
        )
    return max(1, int(round(price)))


def visible_opponent_signal(obs, product, horizon_steps, settings=None):
    settings = _settings(settings)
    player = int(obs.get("player", 0))
    crop_sources = 0
    animal_sources = 0
    visible_field_units = 0
    animal = PRODUCT_ANIMAL.get(product)

    for index, farm in enumerate(obs.get("farms", []) or []):
        if index == player:
            continue
        for row in farm.get("tiles", []) or []:
            for tile in row:
                if not isinstance(tile, dict):
                    continue
                crop_match = (
                    product in CROP_PRODUCTS
                    and tile.get("kind") == "PLANT"
                    and tile.get("crop") == product
                )
                animal_match = animal and tile.get("animal") == animal
                if crop_match:
                    crop_sources += 1
                if animal_match:
                    animal_sources += 1
                if crop_match or animal_match:
                    visible_field_units += max(
                        0,
                        int(tile.get("yield_units", 0)),
                    )

    source_count = crop_sources + animal_sources
    start = max(0, _step(obs, settings))
    action_stop = max(0, int(settings.episode_steps) - 1)
    effective_steps = max(
        0,
        min(
            int(horizon_steps),
            action_stop - start,
        ),
    )
    turns_per_day = max(1, int(settings.turns_per_day))
    future_units = math.ceil(
        source_count
        * effective_steps
        / turns_per_day
        * settings.opponent_units_per_source_day
    )
    pressure_units = (
        visible_field_units + future_units if effective_steps else 0
    )
    return OpponentSignal(
        product=product,
        crop_sources=crop_sources,
        animal_sources=animal_sources,
        visible_field_units=visible_field_units,
        source_count=source_count,
        pressure_units=pressure_units,
    )


def forecast_product_values(obs, horizon_steps, settings=None):
    settings = _settings(settings)
    demand = forecast_demand(obs, horizon_steps, settings)
    inventory = (obs.get("market", {}) or {}).get("inventory", {}) or {}
    result = {}
    for product in PRODUCTS:
        current_inventory = int(
            inventory.get(product, MARKET_RULES[product][1])
        )
        opponent = visible_opponent_signal(
            obs,
            product,
            horizon_steps,
            settings,
        )
        demand_only_inventory = current_inventory - demand[product].total_units
        forecast_inventory = (
            demand_only_inventory + opponent.pressure_units
        )
        result[product] = ProductValue(
            product=product,
            current_inventory=current_inventory,
            current_marginal_value=price_at_inventory(
                obs,
                product,
                current_inventory,
            ),
            demand_units=demand[product].total_units,
            demand_only_inventory=demand_only_inventory,
            demand_only_marginal_value=price_at_inventory(
                obs,
                product,
                demand_only_inventory,
            ),
            opponent_sources=opponent.source_count,
            opponent_pressure_units=opponent.pressure_units,
            forecast_inventory=forecast_inventory,
            forecast_marginal_value=price_at_inventory(
                obs,
                product,
                forecast_inventory,
            ),
        )
    return result


def rank_product_values(values):
    return tuple(
        sorted(
            values.values(),
            key=lambda row: (
                -row.forecast_marginal_value,
                -row.demand_units,
                -row.current_marginal_value,
                row.product,
            ),
        )
    )


from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketFlowForecast:
    product: str
    window_steps: int
    observed_steps: int
    coverage: float
    ready: bool
    shrinkage: float
    historical_net_units: int
    horizon_steps: int
    forecast_net_units: float


class MarketHistory:
    def __init__(
        self,
        settings=None,
        max_window=120,
        min_observations=24,
        shrinkage_steps=12,
    ):
        self.settings = _settings(settings)
        self.max_window = max(1, int(max_window))
        self.min_observations = max(0, int(min_observations))
        self.shrinkage_steps = max(0, int(shrinkage_steps))
        self.reset()

    def reset(self):
        self._last_step = None
        self._last_inventory = None
        self._last_demand = None
        self._last_fingerprint = None
        self._flows = deque(maxlen=self.max_window)

    def _fingerprint(self, obs, step, inventory):
        return (
            step,
            int(obs.get("player", 0)),
            tuple(inventory[product] for product in PRODUCTS),
            tuple(sorted(revealed_shops(obs))),
        )

    def _remember(self, obs, step, inventory, fingerprint):
        demand = forecast_demand(obs, 1, self.settings)
        self._last_step = step
        self._last_inventory = inventory
        self._last_demand = {
            product: demand[product].total_units for product in PRODUCTS
        }
        self._last_fingerprint = fingerprint

    def update(self, obs):
        step = max(0, _step(obs, self.settings))
        market = obs.get("market", {}) or {}
        raw_inventory = market.get("inventory", {}) or {}
        inventory = {
            product: int(raw_inventory.get(product, 0)) for product in PRODUCTS
        }
        fingerprint = self._fingerprint(obs, step, inventory)

        if self._last_step is None or step < self._last_step:
            self.reset()
            self._remember(obs, step, inventory, fingerprint)
            return None
        if step == self._last_step:
            if fingerprint != self._last_fingerprint:
                self.reset()
                self._remember(obs, step, inventory, fingerprint)
            return None
        if step != self._last_step + 1:
            self.reset()
            self._remember(obs, step, inventory, fingerprint)
            return None

        flow = {
            product: inventory[product]
            - self._last_inventory[product]
            + self._last_demand[product]
            for product in PRODUCTS
        }
        self._flows.append(
            (
                self._last_step,
                tuple(flow[product] for product in PRODUCTS),
            )
        )
        self._remember(obs, step, inventory, fingerprint)
        return flow

    def forecast(self, product, horizon_steps, window_steps=48):
        if product not in PRODUCTS:
            raise KeyError(product)
        window_steps = max(1, min(self.max_window, int(window_steps)))
        horizon_steps = max(0, int(horizon_steps))
        if self._last_step is None:
            effective_horizon = 0
        else:
            action_stop = max(0, int(self.settings.episode_steps) - 1)
            effective_horizon = max(
                0,
                min(horizon_steps, action_stop - self._last_step),
            )
        product_index = PRODUCTS.index(product)
        selected = list(self._flows)[-window_steps:]
        historical = sum(row[1][product_index] for row in selected)
        observed = len(selected)
        ready = bool(observed and observed >= self.min_observations)
        shrinkage = (
            observed / (observed + self.shrinkage_steps)
            if observed
            else 0.0
        )
        projected = 0.0
        if ready and observed:
            projected = (
                historical
                * effective_horizon
                / observed
                * shrinkage
            )
        return MarketFlowForecast(
            product=product,
            window_steps=window_steps,
            observed_steps=observed,
            coverage=observed / window_steps,
            ready=ready,
            shrinkage=shrinkage,
            historical_net_units=historical,
            horizon_steps=effective_horizon,
            forecast_net_units=projected,
        )


def make_agent(**overrides):
    features = dict(DEFAULT_FEATURES)
    features.update(overrides)

    def run(obs):
        farms = obs.get("farms", []) or []
        player = int(obs.get("player", 0))
        if player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}

        farm = farms[player]
        private = obs.get("private", {}) or {}
        board_size = len(farm.get("tiles", [])) or 10
        include_ne = (
            int(obs.get("day", 0)) >= _land_day(features)
            or "NE" in farm.get("unlocked_quadrants", [])
        )
        plan = _plan(
            board_size,
            include_ne,
            features["full_land"],
            features.get("use_ne_corridor_plot", False),
            features.get("ne_far_mix", ()),
            features.get("target_geese", 1) if features["goose"] else 0,
        )
        actions, farmer_busy = _integrated_unit_actions(
            obs,
            farm,
            private,
            plan,
            features,
        )
        return {
            "farmer": actions[0] if actions else ["PASS"],
            "hands": actions[1:],
            "market": _market_actions(
                obs,
                farm,
                private,
                plan,
                features,
                actions,
                farmer_busy,
            ),
        }

    return run


agent = make_agent()
