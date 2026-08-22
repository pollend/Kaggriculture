"""Order-activity diagnostics for the free-market policy.

The number that separates this agent's two degenerate attractors is how often it
trades: never is the NOOP collapse, almost every turn is the bankruptcy collapse.

Activity is counted by *effect*, not by emission, and that distinction is the
whole point of the module. Invalid market orders are silent no-ops, so a policy
can emit an order every turn and change nothing -- one measured run sat on
`SELL WOOL 16` for 57% of its decisions while owning no sheep, had 98.4% of its
orders refused, and reported 59% activity against a real rate of 1%. The
promotion gate's activity floor read that 59% and passed it.

Effect is inferred from the money delta: the agent's own orders are the only
thing that moves its cash, and every accepted order moves it by at least a coin.
So a turn that emitted an order and moved no money had it refused.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from agents.dqn_market_free_v2 import lots


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

    Two policies trading at an identical rate can be running completely different
    strategies, and the pool cannot hold them apart unless something records which.

    Kept separate from `OrderDiagnostics` because that dataclass sums itself over
    `__dataclass_fields__`, which works only while every field is a number. `seeds`
    is a mapping, so folding it in would break `__add__`.
    """

    #: Episodes aggregated, so rates can be taken without a second count.
    episodes: int = 0
    #: Accepted or not -- this counts what the policy *chose*, because the
    #: question is which strategy it is running, not which one it can afford.
    land: int = 0
    #: `BUY_SEED` quantity per crop. Quantity rather than decision count, so a
    #: policy hedging one token crop against a staple does not read as diversified.
    seeds: dict[str, int] = field(default_factory=dict)

    # Land *use*, conditioned on purchases the simulator *accepted*. `land` above
    # cannot distinguish the two explanations for land losing money -- expanding is
    # genuinely negative inside 30 days, or the scheduler never works the tiles --
    # and those want opposite fixes.

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

    # Seed lots, from `lots.py`. `seeds` above says what was bought; these say
    # whether it came back and how long it took, which is the only way to tell
    # `LOT_POTENTIAL_WEIGHT` moving purchases *earlier* from it simply buying more.

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
    #: Coins `reward.SEED_WASTE_WEIGHT` charged back onto buy turns, summed per lot
    #: and clamped at zero per lot, which is why it is not just `spend - proceeds`:
    #: a lot sold above cost does not offset a lot that was wasted.
    seed_waste_cost: float = 0.0
    #: Coins `reward.SEED_GAIN_WEIGHT` credited back onto buy turns, the mirror of
    #: the above and capped at each lot's own cost. Not `seed_margin` clamped
    #: upward: the cap is what keeps the reward pair bounded by the seed spend, and
    #: a column that dropped it would over-report what the term actually paid.
    seed_gain_credit: float = 0.0

    # Animal lots, the same six columns for a bought cow or sheep. Kept apart from
    # the seed ones rather than summed into them: an animal is a durable asset whose
    # capacity is a deadline rather than a yield, its holding distance is several
    # times longer, and a single column would hide a herd programme paying for a seed
    # programme that was not working.

    #: Units of each animal the seat *asked* for, by species, the way `seeds` counts
    #: crops. Emissions, not fills: this is the column that answers "is the policy
    #: still buying geese", and the aggregate columns below cannot -- v1 ran 54
    #: generations with `animal_spend` climbing from 160 to 6,100 coins an episode
    #: while goose purchases fell from 10.3 an episode to 0.05 and EGG sales to zero.
    #: A herd that is all cow and a balanced one read identically without this.
    animals: dict[str, int] = field(default_factory=dict)

    animal_lots: int = 0
    animal_lots_matched: int = 0
    animal_spend: float = 0.0
    animal_proceeds: float = 0.0
    animal_holding_turn_sum: float = 0.0
    #: Coins `reward.ANIMAL_WASTE_WEIGHT` charged and `reward.ANIMAL_GAIN_WEIGHT`
    #: credited, clamped and capped per lot exactly as the seed pair is. The two to
    #: watch against each other: a herd that never gets placed shows the first
    #: climbing with the second flat.
    animal_waste_cost: float = 0.0
    animal_gain_credit: float = 0.0
    #: The same three coin columns split by species, so a species that is bought and
    #: never pays is separable from one that is never bought. Dicts rather than one
    #: column per animal for the same reason `seeds` is a dict: the roster is
    #: `donor.ANIMALS` and this file must not restate it.
    animal_lots_by_species: dict[str, int] = field(default_factory=dict)
    animal_spend_by_species: dict[str, float] = field(default_factory=dict)
    animal_proceeds_by_species: dict[str, float] = field(default_factory=dict)

    def __add__(self, other: BehaviorCounts) -> BehaviorCounts:
        seeds = dict(self.seeds)
        for crop, quantity in other.seeds.items():
            seeds[crop] = seeds.get(crop, 0) + quantity

        def _merge(left: dict, right: dict) -> dict:
            merged = dict(left)
            for key, value in right.items():
                merged[key] = merged.get(key, 0) + value
            return merged

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
            seed_waste_cost=self.seed_waste_cost + other.seed_waste_cost,
            seed_gain_credit=self.seed_gain_credit + other.seed_gain_credit,
            animal_lots=self.animal_lots + other.animal_lots,
            animal_lots_matched=self.animal_lots_matched + other.animal_lots_matched,
            animal_spend=self.animal_spend + other.animal_spend,
            animal_proceeds=self.animal_proceeds + other.animal_proceeds,
            animal_holding_turn_sum=(
                self.animal_holding_turn_sum + other.animal_holding_turn_sum
            ),
            animal_waste_cost=self.animal_waste_cost + other.animal_waste_cost,
            animal_gain_credit=self.animal_gain_credit + other.animal_gain_credit,
            animals=_merge(self.animals, other.animals),
            animal_lots_by_species=_merge(
                self.animal_lots_by_species, other.animal_lots_by_species
            ),
            animal_spend_by_species=_merge(
                self.animal_spend_by_species, other.animal_spend_by_species
            ),
            animal_proceeds_by_species=_merge(
                self.animal_proceeds_by_species, other.animal_proceeds_by_species
            ),
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
            # In coins, not reward units. The conversion needs `MONEY_SCALE` and the
            # two weights, and this module is bundled into the submission while
            # `training/` is not, so it must not import from there.
            "seed_waste_cost": self.seed_waste_cost,
            "seed_waste_cost_per_episode": self.seed_waste_cost / max(1, self.episodes),
            "seed_gain_credit": self.seed_gain_credit,
            "seed_gain_credit_per_episode": self.seed_gain_credit / max(1, self.episodes),
            "animal_lots": self.animal_lots,
            "animal_lots_matched": self.animal_lots_matched,
            "animal_spend": self.animal_spend,
            "animal_proceeds": self.animal_proceeds,
            "animal_margin": self.animal_proceeds - self.animal_spend,
            "animal_holding_turns": (
                self.animal_holding_turn_sum / max(1, self.animal_lots_matched)
            ),
            "animal_lots_matched_share": self.animal_lots_matched / max(1, self.animal_lots),
            "unrecovered_animal_share": (
                max(0.0, 1.0 - self.animal_proceeds / self.animal_spend)
                if self.animal_spend > 0
                else 0.0
            ),
            "animal_waste_cost": self.animal_waste_cost,
            "animal_waste_cost_per_episode": self.animal_waste_cost / max(1, self.episodes),
            "animal_gain_credit": self.animal_gain_credit,
            "animal_gain_credit_per_episode": (
                self.animal_gain_credit / max(1, self.episodes)
            ),
            # Flattened per species rather than nested, because every consumer of
            # this dict -- the gate's `metrics_json`, the dashboard, an SQL query
            # against `gate_json` -- reads scalars.
            **{
                f"animals_{animal.lower()}_per_episode": count / max(1, self.episodes)
                for animal, count in sorted(self.animals.items())
            },
            **{
                f"animal_lots_{animal.lower()}": count
                for animal, count in sorted(self.animal_lots_by_species.items())
            },
            **{
                f"animal_margin_{animal.lower()}": (
                    self.animal_proceeds_by_species.get(animal, 0.0) - spend
                )
                for animal, spend in sorted(self.animal_spend_by_species.items())
            },
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
    "seed_waste_cost": 0.0,
    "seed_gain_credit": 0.0,
    "animal_lots": 0,
    "animal_lots_matched": 0,
    "animal_spend": 0.0,
    "animal_proceeds": 0.0,
    "animal_holding_turn_sum": 0.0,
    "animal_waste_cost": 0.0,
    "animal_gain_credit": 0.0,
    "animal_lots_by_species": {},
    "animal_spend_by_species": {},
    "animal_proceeds_by_species": {},
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
    owned: list[list[int]] = []
    produce_room: list[list[int]] = []
    days: list[int] = []
    money: list[float] = []
    ops: list[str | None] = []
    for index in range(len(steps) - 1):
        observation = _seat_observation(steps[index], seat)
        view = lots.seat_view(observation, seat)
        farm, private = view if view is not None else ({}, {})
        # The same builder `collect.Recorder.record_turn` and `clone` use, so the
        # column here describes the trace the reward acted on rather than a
        # look-alike. The holdings and animal traces are wanted; the quadrant count
        # rides along and is discarded, while the day is kept because an animal lot's
        # capacity is a deadline rather than a fixed yield.
        held = lots.sample(observation, farm, private, index)
        seeds.append(held.seeds)
        standing.append(held.standing)
        stock.append(held.stock)
        owned.append(held.owned)
        produce_room.append(held.produce_room)
        days.append(held.day)
        money.append(_seat_money(steps[index], seat))
        # The op chosen *from* this observation, which `_replay_orders` explains
        # is stored on the following step. `lots.match` needs it to tell a
        # BUY_PRODUCT from a harvest that happened to land on a spending turn.
        #
        # The *wire* op, deliberately unlike the reward path, which records
        # `ACTIONS[chosen].op`. The two differ only when an order the action space
        # cannot represent was played: this column describes what the seat did, the
        # reward describes what the learner chose.
        ops.append(_market_op(steps[index + 1], seat))

    ledger = lots.match(
        seeds,
        stock,
        money,
        ops,
        standing,
        owned=owned,
        produce_room=produce_room,
        days=days,
    )
    summary = ledger.summary()
    matched = [
        lot for lot in ledger.seed_lots() if lot.first_sale_turn is not None
    ]
    animal_matched = [
        lot for lot in ledger.animal_lots() if lot.first_sale_turn is not None
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
        # The coins the two non-invariant lot legs actually moved on buy turns this
        # episode, read off the same `unrecovered()` and `surplus()` the reward
        # consumes. Both sides have to be visible or a run cannot tell a working
        # pair from one where only the charge ever fires.
        "seed_waste_cost": float(summary["unrecovered_seed_cost"]),
        "seed_gain_credit": float(summary["recovered_seed_surplus"]),
        "animal_lots": int(summary["animal_lots"]),
        "animal_lots_matched": int(summary["animal_lots_matched"]),
        "animal_spend": float(summary["animal_spend"]),
        "animal_proceeds": float(summary["animal_proceeds"]),
        "animal_holding_turn_sum": float(
            sum(lot.first_sale_turn - lot.buy_turn for lot in animal_matched)
        ),
        "animal_waste_cost": float(summary["unrecovered_animal_cost"]),
        "animal_gain_credit": float(summary["recovered_animal_surplus"]),
        # `_Lot.animal` carries the species the lot was opened for, so the split is
        # a group-by over the same lots the aggregates sum, not a second walk.
        "animal_lots_by_species": _count_by(
            (str(lot.animal) for lot in ledger.animal_lots()), 1
        ),
        "animal_spend_by_species": _sum_by(
            (str(lot.animal), lot.units * lot.unit_cost) for lot in ledger.animal_lots()
        ),
        "animal_proceeds_by_species": _sum_by(
            (str(lot.animal), lot.proceeds) for lot in ledger.animal_lots()
        ),
    }


def _count_by(keys: Any, weight: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + weight
    return counts


def _sum_by(pairs: Any) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key, value in pairs:
        totals[key] = totals.get(key, 0.0) + float(value)
    return totals


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
    animals: dict[str, int] = {}
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
            elif op == "BUY_ANIMAL" and item:
                animals[item] = animals.get(item, 0) + max(1, quantity)
    return BehaviorCounts(
        episodes=1,
        land=land,
        seeds=seeds,
        animals=animals,
        **_land_use(replay, seat),
        **_lot_use(replay, seat),
    )


def arena_reduction(result: Any, seat: int) -> tuple[OrderDiagnostics, BehaviorCounts] | None:
    """Both replay aggregates for one seat, folded inside the worker.

    Passed to `sim.arena.play_series(..., reduce=...)` by name, so it runs in the
    process that played the episode rather than in the updater thread that
    collects the series. Two things move with it: the replay stops being pickled
    back (it is the largest thing a `MatchResult` carries and the gate never
    persists it), and these two walks -- `_lot_use` alone rebuilds the whole
    per-turn trace and re-runs `lots.match` -- stop being serialised by the GIL
    against every other model updating at the same time.

    A failed match reduces to None: `_arena_summary` already drops incomplete
    pairs, so nothing reads it, and a crashed episode's replay is not worth
    walking.
    """

    if getattr(result, "failed", False):
        return None
    replay = getattr(result, "replay", None)
    if not isinstance(replay, dict):
        return None
    return diagnose_replay(replay, seat), behavior_from_replay(replay, seat)


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
