"""Order-activity diagnostics for the free-market policy.

The one number that separates this agent's two degenerate attractors is how
often it trades: never is the NOOP collapse, on almost every turn is the
bankruptcy collapse. These metrics make that visible without changing or
filtering the policy's actions.

Activity is counted by *effect*, not by emission, and that distinction is the
whole point of this module. Invalid market orders are silent no-ops in the
simulator, so a policy can emit an order every turn and change nothing. One
measured run did exactly that: 57% of its decisions were `SELL WOOL 16` while it
owned no sheep and therefore no wool, 40% were NOOP, and 98.4% of everything it
emitted was rejected. Counting emissions reported 59% activity for a policy whose
real activity was 1%, and the promotion gate's activity floor -- which exists
precisely to catch a silent policy -- read that 59% and passed it.

Effect is inferred from the money delta on the turn. The agent's own orders are
the only thing that moves its cash, and every order the simulator accepts moves
it by at least one coin: prices floor at 1, quantities start at 1, and hiring,
land and seed all cost. So a turn that emitted an order and moved no money had
that order refused.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from agents.dqn_market_free import lots


@dataclass(frozen=True, slots=True)
class OrderDiagnostics:
    #: Turns the policy was asked for an action.
    decisions: int = 0
    #: Simulator order rows emitted. Not a decision count: `HIRE n` is one
    #: decision that expands to n rows, so this can exceed `decisions`.
    orders: int = 0
    #: Decisions that emitted at least one order, accepted or not.
    emitting_decisions: int = 0
    #: Decisions whose order the simulator actually acted on. This is the one
    #: the activity floor reads, and the only one a do-nothing policy cannot
    #: inflate.
    effective_decisions: int = 0

    def __add__(self, other: OrderDiagnostics) -> OrderDiagnostics:
        return OrderDiagnostics(
            **{name: getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__}
        )

    def as_dict(self) -> dict[str, int | float]:
        """The counts plus the rates, named so they cannot be confused.

        `trading_rate` is the real activity and what the gate reads.
        `emitted_rate` above it is how much the policy *tried* to do, and the
        gap between the two is `rejected_order_rate` -- a large gap means the
        policy is talking to itself.
        """

        decisions = max(1, self.decisions)
        payload: dict[str, int | float] = dict(asdict(self))
        payload["trading_rate"] = self.effective_decisions / decisions
        payload["emitted_rate"] = self.emitting_decisions / decisions
        payload["wire_orders_per_decision"] = self.orders / decisions
        payload["rejected_order_rate"] = (
            1.0 - self.effective_decisions / self.emitting_decisions
            if self.emitting_decisions
            else 0.0
        )
        return payload


@dataclass(frozen=True, slots=True)
class BehaviorCounts:
    """What a policy *did*, as opposed to how often it did anything.

    `OrderDiagnostics` above answers the activity question and deliberately
    collapses every op into one counter, because the two attractors it separates
    -- NOOP and bankruptcy -- are both visible in the rate alone. This one exists
    for the opposite question: two policies trading at an identical rate can be
    running completely different strategies, and the pool cannot hold them apart
    unless something records which.

    Kept separate from `OrderDiagnostics` rather than added to it because that
    dataclass sums itself over `__dataclass_fields__`, which works only while
    every field is a number. `seeds` is a mapping, so folding it in there would
    break `__add__` for every existing caller.
    """

    #: Episodes aggregated, so rates can be taken without a second count.
    episodes: int = 0
    #: Accepted or not -- this counts what the policy *chose*, because the
    #: question is which strategy it is running, not which one it can afford.
    land: int = 0
    #: `BUY_SEED` quantity per crop. Quantity rather than decision count, so a
    #: policy hedging one token crop against a staple does not read as diversified.
    seeds: dict[str, int] = field(default_factory=dict)

    # Land *use*. `land` above says the policy asked for a quadrant, which is all
    # the archive cell needs. It cannot distinguish the two explanations for land
    # losing money -- that expanding is genuinely negative inside 30 days, or that
    # the field scheduler never works the tiles -- and those want opposite fixes.
    # Everything below is conditioned on purchases the simulator *accepted*,
    # because only those cost money and gained tiles.

    #: Quadrants that actually appeared in `unlocked_quadrants`.
    land_accepted: int = 0
    #: In-game day and cash immediately *before* each accepted purchase. Summed
    #: rather than averaged so the fold stays associative; `as_dict` divides.
    land_day_sum: int = 0
    land_money_sum: float = 0.0
    #: Distinct tiles inside a bought quadrant ever seen holding a crop after the
    #: purchase. Against 25 tiles per quadrant this is the direct answer to
    #: whether bought land is worked at all.
    land_tiles_planted: int = 0
    #: DIGs spent clearing weeds inside a bought quadrant -- the standing labour
    #: cost of an expansion, paid before it returns anything.
    land_weed_digs: int = 0
    #: Idle fraction after the first accepted purchase minus the fraction before.
    #: Positive means the seat bought capacity it could not dispatch to.
    land_idle_delta_sum: float = 0.0
    #: Episodes with at least one accepted purchase: the denominator for the two
    #: per-episode figures above, which `episodes` would understate.
    land_episodes: int = 0

    # Seed lots, from `lots.py`. `seeds` above says what was bought;
    # these say whether it came back and how long it took, which is the only way
    # to tell `LOT_POTENTIAL_WEIGHT` moving purchases *earlier* from it simply
    # buying *more*. The ledger no longer feeds the reward -- the potential is
    # state-only and needs no attribution -- so this is the whole of what it is
    # for, and it is why the matching still has to be right.

    #: Purchase lots opened, and how many of them ever saw a matched sale.
    seed_lots: int = 0
    seed_lots_matched: int = 0
    #: Coins in and coins out, across matched lots. Summed rather than ratioed so
    #: the fold stays associative, the same as the land sums above.
    seed_spend: float = 0.0
    seed_proceeds: float = 0.0
    #: Turns from a lot's purchase to its first matched sale, summed over matched
    #: lots. The distance the credit used to have to cross.
    seed_holding_turn_sum: float = 0.0

    def __add__(self, other: BehaviorCounts) -> BehaviorCounts:
        seeds = dict(self.seeds)
        for crop, quantity in other.seeds.items():
            seeds[crop] = seeds.get(crop, 0) + quantity
        return BehaviorCounts(
            episodes=self.episodes + other.episodes,
            land=self.land + other.land,
            seeds=seeds,
            land_accepted=self.land_accepted + other.land_accepted,
            land_day_sum=self.land_day_sum + other.land_day_sum,
            land_money_sum=self.land_money_sum + other.land_money_sum,
            land_tiles_planted=self.land_tiles_planted + other.land_tiles_planted,
            land_weed_digs=self.land_weed_digs + other.land_weed_digs,
            land_idle_delta_sum=self.land_idle_delta_sum + other.land_idle_delta_sum,
            land_episodes=self.land_episodes + other.land_episodes,
            seed_lots=self.seed_lots + other.seed_lots,
            seed_lots_matched=self.seed_lots_matched + other.seed_lots_matched,
            seed_spend=self.seed_spend + other.seed_spend,
            seed_proceeds=self.seed_proceeds + other.seed_proceeds,
            seed_holding_turn_sum=self.seed_holding_turn_sum + other.seed_holding_turn_sum,
        )

    @property
    def distinct_crops(self) -> int:
        return sum(1 for quantity in self.seeds.values() if quantity > 0)

    @property
    def top_crop_share(self) -> float:
        """How concentrated the planting is. 1.0 is a monoculture.

        0.0 when nothing was planted, which reads as "not concentrated" and is
        the safe direction: a policy that never buys a seed is caught by the
        activity floor, and inflating its concentration here would let it occupy
        the monoculture cell that the archive refuses to reserve.
        """

        total = sum(self.seeds.values())
        return max(self.seeds.values()) / total if total > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        purchases = max(1, self.land_accepted)
        buyers = max(1, self.land_episodes)
        return {
            "episodes": self.episodes,
            "land": self.land,
            "land_per_episode": self.land / max(1, self.episodes),
            "seeds": dict(self.seeds),
            "distinct_crops": self.distinct_crops,
            "top_crop_share": self.top_crop_share,
            "land_accepted": self.land_accepted,
            "land_episodes": self.land_episodes,
            # Per accepted purchase, not per episode: a seat that bought twice
            # made two timing decisions and both should weigh.
            "land_day": self.land_day_sum / purchases,
            "land_money": self.land_money_sum / purchases,
            "land_tiles_planted": self.land_tiles_planted,
            "land_tiles_planted_per_purchase": self.land_tiles_planted / purchases,
            "land_weed_digs": self.land_weed_digs,
            "land_weed_digs_per_episode": self.land_weed_digs / buyers,
            "post_land_idle_delta": self.land_idle_delta_sum / buyers,
            "seed_lots": self.seed_lots,
            "seed_lots_matched": self.seed_lots_matched,
            "seed_spend": self.seed_spend,
            "seed_proceeds": self.seed_proceeds,
            "seed_margin": self.seed_proceeds - self.seed_spend,
            # Over *matched* lots only. Dividing by every lot would report a lot
            # that never sold as having sold instantly, which is the reading that
            # makes an over-buying policy look fast.
            "seed_holding_turns": self.seed_holding_turn_sum / max(1, self.seed_lots_matched),
            "seed_lots_matched_share": self.seed_lots_matched / max(1, self.seed_lots),
            "unrecovered_seed_share": (
                max(0.0, 1.0 - self.seed_proceeds / self.seed_spend)
                if self.seed_spend > 0
                else 0.0
            ),
        }


def _normalise_order(order: Any) -> tuple[str, str | None, int] | None:
    if not isinstance(order, (list, tuple)) or not order:
        return None
    op = str(order[0])
    item = str(order[1]) if len(order) >= 2 else None
    try:
        quantity = int(order[2]) if len(order) >= 3 else 0
    except (TypeError, ValueError):
        quantity = 0
    return op, item, quantity


def diagnose_order_lists(
    order_lists: Iterable[Iterable[Any]],
    money_deltas: Sequence[float] | None = None,
) -> OrderDiagnostics:
    """Summarise emitted market orders, one iterable per decision turn.

    `money_deltas` is the per-turn change in the acting player's cash, aligned
    with `order_lists`. Without it nothing can be counted as effective: the
    result fails closed, so a caller that forgets to pass it reports zero
    activity and trips the gate rather than quietly reporting emissions as
    trades, which is the failure this replaced.

    Opponents and stored replays are not held to this agent's one-order rule, so
    a decision may still carry several orders here; they are all counted.
    """

    total = OrderDiagnostics()
    for index, raw_orders in enumerate(order_lists):
        orders = [row for raw in raw_orders if (row := _normalise_order(raw)) is not None]
        moved = (
            money_deltas is not None
            and index < len(money_deltas)
            and abs(float(money_deltas[index])) > 1e-9
        )
        total += OrderDiagnostics(
            decisions=1,
            orders=len(orders),
            emitting_decisions=1 if orders else 0,
            effective_decisions=1 if orders and moved else 0,
        )
    return total


def _replay_orders(replay: dict[str, Any], seat: int) -> tuple[list[list[Any]], list[float]]:
    """One seat's per-turn order lists and the money delta each one moved."""

    steps = replay.get("steps") or []
    order_lists: list[list[Any]] = []
    money: list[float] = []
    # Step zero is the environment reset. An action stored on steps[i] is the
    # action taken from steps[i - 1]'s observation, matching clone.py -- so the
    # money it moved is the difference between those two observations.
    for index in range(1, len(steps)):
        step = steps[index]
        if not isinstance(step, list) or not (0 <= seat < len(step)):
            continue
        state = step[seat] or {}
        action = state.get("action") if isinstance(state, dict) else None
        order_lists.append((action or {}).get("market", []) if isinstance(action, dict) else [])
        money.append(_seat_money(steps[index - 1], seat))
        if index == len(steps) - 1:
            money.append(_seat_money(step, seat))

    deltas = [money[i + 1] - money[i] for i in range(len(money) - 1)] if len(money) > 1 else []
    return order_lists, deltas


#: What a replay with no readable trace contributes: nothing, not a zero lot.
_LOT_USE_ZERO: dict[str, Any] = {
    "seed_lots": 0,
    "seed_lots_matched": 0,
    "seed_spend": 0.0,
    "seed_proceeds": 0.0,
    "seed_holding_turn_sum": 0.0,
}


def _market_op(step: Any, seat: int) -> str | None:
    """The op of one seat's market order on `step`, or None if it placed none.

    One action per turn, so the list is either empty or a run of the same op --
    `Order.to_wire` expands `HIRE 8` into eight rows of `HIRE`. The first row is
    therefore the whole decision.
    """

    if not isinstance(step, list) or not (0 <= seat < len(step)):
        return None
    state = step[seat] or {}
    action = state.get("action") if isinstance(state, dict) else None
    orders = (action or {}).get("market") or [] if isinstance(action, dict) else []
    for order in orders:
        if isinstance(order, (list, tuple)) and order:
            return str(order[0])
    return None


def _lot_use(replay: dict[str, Any], seat: int) -> dict[str, Any]:
    """Seed lots matched to the sales that paid for them, from one seat's replay.

    The ledger itself lives in `lots.py` and is shared with the reward
    path on purpose: the columns here have to describe the same matching the
    reward acted on, or a run could be reading one attribution and training on
    another.

    The trace is built from steps `0 .. n-2`, which is the pre-action sample for
    each recorded decision -- the same convention `clone.py` uses to rebuild a
    demonstration and the same one `Recorder.record_turn` writes live.
    """

    steps = replay.get("steps") or []
    if len(steps) < 3:
        return dict(_LOT_USE_ZERO)

    seeds: list[list[int]] = []
    standing: list[list[int]] = []
    stock: list[list[int]] = []
    money: list[float] = []
    ops: list[str | None] = []
    for index in range(len(steps) - 1):
        observation = _seat_observation(steps[index], seat)
        private = observation.get("private") or {}
        farms = observation.get("farms") or []
        player = int(observation.get("player", seat) or seat)
        farm = farms[player] if 0 <= player < len(farms) else {}
        held_seeds, held_stock = lots.holdings(private if isinstance(private, dict) else {})
        seeds.append(held_seeds)
        standing.append(lots.standing_plants(farm if isinstance(farm, dict) else {}))
        stock.append(held_stock)
        money.append(_seat_money(steps[index], seat))
        # The op chosen *from* this observation, which `_replay_orders` explains
        # is stored on the following step. `lots.match` needs it to tell a
        # BUY_PRODUCT from a harvest that happened to land on a spending turn.
        ops.append(_market_op(steps[index + 1], seat))

    ledger = lots.match(seeds, stock, money, ops, standing)
    summary = ledger.summary()
    # `not lot.bought` matters and is not decoration: WHEAT is both a crop and a
    # `BUY_PRODUCT` item, so without it a bought-wheat lot lands in this sum while
    # `summary["seed_lots_matched"]` -- which excludes bought lots -- leaves it out
    # of the count. The batch column divides the two, so the mean came out above
    # the 720-turn season it is measured in: 1,010 on `lotbasis` batches 0-9.
    matched = [
        lot
        for lot in ledger.lots
        if lot.item in lots.CROP_ITEMS and not lot.bought and lot.first_sale_turn is not None
    ]
    return {
        "seed_lots": int(summary["seed_lots"]),
        "seed_lots_matched": int(summary["seed_lots_matched"]),
        "seed_spend": float(summary["seed_spend"]),
        "seed_proceeds": float(summary["seed_proceeds"]),
        # Summed rather than the episode's median, because `BehaviorCounts` folds
        # across episodes and a median of medians is not a median.
        "seed_holding_turn_sum": float(
            sum(lot.first_sale_turn - lot.buy_turn for lot in matched)
        ),
    }


def diagnose_replay(replay: dict[str, Any], seat: int) -> OrderDiagnostics:
    """Read the actions taken by ``seat`` from an environment replay.

    Money is read from the same steps, so the effect of each order is known
    without the caller supplying anything.
    """

    order_lists, deltas = _replay_orders(replay, seat)
    return diagnose_order_lists(order_lists, deltas)


def behavior_from_replay(replay: dict[str, Any], seat: int) -> BehaviorCounts:
    """One seat's strategy fingerprint from an environment replay.

    Emissions rather than effects, which is the opposite of `diagnose_replay`'s
    rule and deliberate. That function measures whether the policy is doing
    anything, so an order the simulator refused has to count as nothing. This one
    measures *which* strategy the policy is running, and a refused `BUY_LAND` is
    still the policy asking for land -- it says the row is reachable, which is
    exactly what the archive is trying not to lose.
    """

    order_lists, _ = _replay_orders(replay, seat)
    land = 0
    seeds: dict[str, int] = {}
    for raw_orders in order_lists:
        for raw in raw_orders:
            order = _normalise_order(raw)
            if order is None:
                continue
            op, item, quantity = order
            if op == "BUY_LAND":
                land += 1
            elif op == "BUY_SEED" and item:
                seeds[item] = seeds.get(item, 0) + max(1, quantity)
    return BehaviorCounts(
        episodes=1,
        land=land,
        seeds=seeds,
        **_land_use(replay, seat),
        **_lot_use(replay, seat),
    )


def _seat_observation(step: Any, seat: int) -> dict[str, Any]:
    if not isinstance(step, list) or not (0 <= seat < len(step)):
        return {}
    observation = (step[seat] or {}).get("observation") or {}
    return observation if isinstance(observation, dict) else {}


def _seat_farm(step: Any, seat: int) -> dict[str, Any]:
    observation = _seat_observation(step, seat)
    farms = observation.get("farms") or []
    player = int(observation.get("player", seat) or seat)
    if not (0 <= player < len(farms)):
        player = seat
    if not (0 <= player < len(farms)):
        return {}
    farm = farms[player] or {}
    return farm if isinstance(farm, dict) else {}


def _seat_money(step: Any, seat: int) -> float:
    return float(_seat_farm(step, seat).get("money", 0) or 0)


def _quadrant_of(x: int, y: int, board_size: int) -> str:
    """Which 5x5 corner a tile falls in, matching the simulator's own split."""

    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _quadrant_cells(quadrant: str, board_size: int) -> Iterable[tuple[int, int]]:
    """Every `(x, y)` in one quadrant, so a scan never sweeps the whole board."""

    half = board_size // 2
    ys = range(0, half) if quadrant.startswith("N") else range(half, board_size)
    xs = range(0, half) if quadrant.endswith("W") else range(half, board_size)
    return ((x, y) for y in ys for x in xs)


def _unit_ops(action: Any) -> list[str]:
    """One op per worker, farmer first -- the order `farm["hands"]` is stored in."""

    if not isinstance(action, dict):
        return []
    rows = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    return [str(row[0]) if isinstance(row, (list, tuple)) and row else "PASS" for row in rows]


def _unit_positions(farm: dict[str, Any]) -> list[Any]:
    return [farm.get("farmer"), *(farm.get("hands") or [])]


_LAND_USE_ZERO: dict[str, Any] = {
    "land_accepted": 0,
    "land_day_sum": 0,
    "land_money_sum": 0.0,
    "land_tiles_planted": 0,
    "land_weed_digs": 0,
    "land_idle_delta_sum": 0.0,
    "land_episodes": 0,
}


def _land_use(replay: dict[str, Any], seat: int) -> dict[str, Any]:
    """Whether accepted land purchases were ever worked, from one seat's replay.

    A purchase is detected as a quadrant appearing in `unlocked_quadrants`
    rather than as a `BUY_LAND` emission, because the simulator silently
    no-ops an unaffordable one and a refused order neither costs money nor
    gains tiles. Day and cash are read from the step *before* the quadrant
    appears, which is the state the policy actually decided from.

    Cheap enough to run on every arena replay: the tile scan is confined to the
    quadrants already bought, so it is at most 75 lookups per turn rather than a
    full board sweep.
    """

    steps = replay.get("steps") or []
    if len(steps) < 2:
        return dict(_LAND_USE_ZERO)

    bought: dict[str, int] = {}
    day_sum = 0
    money_sum = 0.0
    weed_digs = 0
    planted: set[tuple[int, int]] = set()
    # Idle is the PASS share of unit actions, matching `episode_metrics`, split
    # at the first accepted purchase so the two halves are comparable.
    units_before = passes_before = 0
    units_after = passes_after = 0
    bought_yet = False

    previous_quadrants = set(_seat_farm(steps[0], seat).get("unlocked_quadrants") or ["NW"])

    for index in range(1, len(steps)):
        pre_farm = _seat_farm(steps[index - 1], seat)
        post_farm = _seat_farm(steps[index], seat)
        state = steps[index][seat] if isinstance(steps[index], list) and seat < len(steps[index]) else {}
        action = (state or {}).get("action")
        ops = _unit_ops(action)

        if bought_yet:
            units_after += len(ops)
            passes_after += sum(1 for op in ops if op == "PASS")
        else:
            units_before += len(ops)
            passes_before += sum(1 for op in ops if op == "PASS")

        # Weed clearing inside land already bought. The worker stood where the
        # *pre* state says it did, and dug the tile the pre state describes.
        pre_tiles = pre_farm.get("tiles") or []
        if bought and pre_tiles:
            board_size = len(pre_tiles)
            for position, op in zip(_unit_positions(pre_farm), ops):
                if op != "DIG" or not isinstance(position, (list, tuple)) or len(position) < 2:
                    continue
                x, y = int(position[0]), int(position[1])
                if not (0 <= y < len(pre_tiles) and 0 <= x < len(pre_tiles[y])):
                    continue
                if _quadrant_of(x, y, board_size) not in bought:
                    continue
                tile = pre_tiles[y][x]
                if isinstance(tile, dict) and tile.get("kind") == "WEED":
                    weed_digs += 1

        quadrants = set(post_farm.get("unlocked_quadrants") or ["NW"])
        for quadrant in sorted(quadrants - previous_quadrants):
            bought[quadrant] = index
            day_sum += int(_seat_observation(steps[index - 1], seat).get("day", 0) or 0)
            money_sum += float(pre_farm.get("money", 0) or 0)
            bought_yet = True
        previous_quadrants = quadrants

        post_tiles = post_farm.get("tiles") or []
        if bought and post_tiles:
            for quadrant in bought:
                for x, y in _quadrant_cells(quadrant, len(post_tiles)):
                    if (x, y) in planted or not (0 <= y < len(post_tiles)):
                        continue
                    row = post_tiles[y]
                    if not (0 <= x < len(row)):
                        continue
                    tile = row[x]
                    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                        planted.add((x, y))

    if not bought:
        return dict(_LAND_USE_ZERO)

    idle_before = passes_before / units_before if units_before else 0.0
    idle_after = passes_after / units_after if units_after else 0.0
    return {
        "land_accepted": len(bought),
        "land_day_sum": day_sum,
        "land_money_sum": money_sum,
        "land_tiles_planted": len(planted),
        "land_weed_digs": weed_digs,
        "land_idle_delta_sum": idle_after - idle_before,
        "land_episodes": 1,
    }
