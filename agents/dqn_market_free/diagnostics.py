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
from dataclasses import asdict, dataclass
from typing import Any


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


def diagnose_replay(replay: dict[str, Any], seat: int) -> OrderDiagnostics:
    """Read the actions taken by ``seat`` from an environment replay.

    Money is read from the same steps, so the effect of each order is known
    without the caller supplying anything.
    """

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
    return diagnose_order_lists(order_lists, deltas)


def _seat_money(step: Any, seat: int) -> float:
    if not isinstance(step, list) or not (0 <= seat < len(step)):
        return 0.0
    observation = (step[seat] or {}).get("observation") or {}
    farms = observation.get("farms") or []
    player = int(observation.get("player", seat) or seat)
    if not (0 <= player < len(farms)):
        player = seat
    if not (0 <= player < len(farms)):
        return 0.0
    return float((farms[player] or {}).get("money", 0) or 0)
