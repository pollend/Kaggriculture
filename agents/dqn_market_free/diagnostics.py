"""Order-activity diagnostics for the free-market policy.

The one number that separates this agent's two degenerate attractors is how
often it trades: zero orders per decision is the NOOP collapse, ordering on
almost every turn is the bankruptcy collapse. These metrics make that visible
without changing or filtering the policy's actions.

An earlier version also counted duplicate and mutually contradictory orders
*within* one decision, because the policy scored ten order slots independently
and could repeat itself across them without any slot seeing the others. A
decision now carries at most one order, so those counts are identically zero and
have been removed rather than reported as constants.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OrderDiagnostics:
    #: Turns the policy was asked for an action.
    decisions: int = 0
    #: Simulator order rows emitted. Not a decision count: `HIRE n` is one
    #: decision that expands to n rows, so this can exceed `decisions`.
    orders: int = 0
    #: Decisions that emitted at least one order. This is the one bounded by
    #: `decisions`, and it is what the activity floor reads.
    trading_decisions: int = 0

    def __add__(self, other: OrderDiagnostics) -> OrderDiagnostics:
        return OrderDiagnostics(
            **{name: getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__}
        )

    def as_dict(self) -> dict[str, int | float]:
        """The counts plus both rates, named so they cannot be confused.

        There used to be one `orders_per_decision` here, described everywhere as
        bounded by 1.0 because a decision emits at most one order. That is false:
        `HIRE n` is a single decision whose wire form is n rows, so the ratio ran
        to 13.0 on a full hire. The bounded quantity is `trading_rate`; the wire
        ratio keeps its own name so a reader cannot mistake one for the other.
        """

        payload: dict[str, int | float] = dict(asdict(self))
        payload["trading_rate"] = self.trading_decisions / max(1, self.decisions)
        payload["wire_orders_per_decision"] = self.orders / max(1, self.decisions)
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


def diagnose_order_lists(order_lists: Iterable[Iterable[Any]]) -> OrderDiagnostics:
    """Summarise emitted market orders, one iterable per decision turn.

    Opponents and stored replays are not held to this agent's one-order rule, so
    a decision may still carry several orders here; they are all counted.
    """

    total = OrderDiagnostics()
    for raw_orders in order_lists:
        orders = [row for raw in raw_orders if (row := _normalise_order(raw)) is not None]
        total += OrderDiagnostics(
            decisions=1, orders=len(orders), trading_decisions=1 if orders else 0
        )
    return total


def diagnose_replay(replay: dict[str, Any], seat: int) -> OrderDiagnostics:
    """Read the actions taken by ``seat`` from an environment replay."""

    order_lists: list[list[Any]] = []
    # Step zero is the environment reset. An action stored on steps[i] is the
    # action taken from steps[i - 1]'s observation, matching clone.py.
    for step in (replay.get("steps") or [])[1:]:
        if not isinstance(step, list) or not (0 <= seat < len(step)):
            continue
        state = step[seat] or {}
        action = state.get("action") if isinstance(state, dict) else None
        order_lists.append((action or {}).get("market", []) if isinstance(action, dict) else [])
    return diagnose_order_lists(order_lists)
