"""Post-episode hook: judge the free-market checkpoint on accumulated evidence.

The league fires this after every episode this agent plays. Unlike the agent it
has PostgreSQL access, so it can look at the revision's own history rather than
the single noisy episode that triggered it.

What it watches for is specific to an unscaffolded market. The scorer's failure
mode is a gradual regression; this agent has two abrupt attractors instead --
emit nothing and sit on the starting float, or emit everything and go bankrupt --
and a rolling money mean is what tells them apart from an ordinary bad run.

Deliberately imports nothing from `kaggriculture_lab` at runtime.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any

from agents.dqn_market_free_v3d.diagnostics import diagnose_replay

if TYPE_CHECKING:
    from kaggriculture_lab.hooks import EpisodeHookContext, EpisodeReader

SCHEMA_VERSION = 1

#: Episodes needed before a verdict means anything.
MIN_HISTORY = 20

#: Rolling window used for the trend comparison.
WINDOW = 40

#: Re-evaluate when the recent window's mean money margin falls this far below
#: the older window's.
REGRESSION_MARGIN = 1500.0

#: Starting cash. An agent parked within this band of it has stopped trading.
STARTING_MONEY = 3000.0
IDLE_BAND = 50.0

#: Below this the run has spent itself into the floor.
INSOLVENT_MONEY = 500.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def evaluate_episode(context: EpisodeHookContext, reader: EpisodeReader) -> dict[str, Any]:
    """Summarise this episode and flag whether the checkpoint should be re-evaluated."""

    summary = reader.summary()
    metrics = summary["metrics"]
    order_diagnostics = diagnose_replay(reader.replay(), context.seat)
    history = reader.history(context.revision_id, limit=WINDOW * 2)

    played = [row for row in history if not row["failed"]]
    margins = [float(row["money_difference"] or 0.0) for row in played]
    wins = [float(row["win"] or 0.0) for row in played]
    finals = [float(row["final_money"] or 0.0) for row in played]

    half = len(margins) // 2
    recent, older = margins[:half], margins[half:]
    recent_mean, older_mean = _mean(recent), _mean(older)
    regression = older_mean - recent_mean

    enough = len(played) >= MIN_HISTORY and len(older) > 0
    idle = [value for value in finals if abs(value - STARTING_MONEY) <= IDLE_BAND]
    broke = [value for value in finals if value < INSOLVENT_MONEY]
    idle_fraction = len(idle) / len(finals) if finals else 0.0
    insolvent_fraction = len(broke) / len(finals) if finals else 0.0

    collapsed = enough and (idle_fraction > 0.9 or insolvent_fraction > 0.9)
    should_reevaluate = bool(enough and (regression >= REGRESSION_MARGIN or collapsed))

    if not enough:
        reason = "insufficient history"
    elif idle_fraction > 0.9:
        reason = f"policy emits no orders: {idle_fraction:.0%} of episodes ended on the float"
    elif insolvent_fraction > 0.9:
        reason = f"policy is bankrupting itself: {insolvent_fraction:.0%} of episodes under {INSOLVENT_MONEY:.0f}"
    elif regression >= REGRESSION_MARGIN:
        reason = f"mean margin fell {regression:,.0f} over the last {len(recent)} episodes"
    else:
        reason = "stable"

    return {
        "schema_version": SCHEMA_VERSION,
        "revision_id": context.revision_id,
        "final_money": summary["final_money"],
        "money_difference": summary["money_difference"],
        "win": summary["win"],
        "productive_action_percentage": metrics.get("productive_action_percentage", 0.0),
        "movement_percentage": metrics.get("movement_percentage", 0.0),
        "plants_lost_inferred": metrics.get("plants_lost_inferred", 0),
        "order_diagnostics": order_diagnostics.as_dict(),
        "episodes_seen": len(played),
        "rolling_win_rate": _mean(wins),
        "recent_margin_mean": recent_mean,
        "older_margin_mean": older_mean,
        "margin_regression": regression,
        # The two collapse detectors, reported every episode so a run that has
        # stopped trading is answerable in SQL rather than inferred from money.
        "idle_fraction": idle_fraction,
        "insolvent_fraction": insolvent_fraction,
        "collapsed": collapsed,
        "should_reevaluate": should_reevaluate,
        "reevaluate_reason": reason,
    }
