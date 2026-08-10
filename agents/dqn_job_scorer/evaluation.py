"""Post-episode hook: decide whether the Q-network is due for re-evaluation.

The league fires this after every episode this agent plays. It has read-only
PostgreSQL access, so unlike the agent itself it can look at the games this revision
has already played and judge the checkpoint on accumulated evidence rather than
on the single episode that triggered it.

What it is for: DQN training is not monotonic. The dqn02 run peaked around round
45 and collapsed by round 75, and a per-episode number is far too noisy to see
that -- a good checkpoint loses plenty of individual games. A rolling window over
the revision's own history is the smallest thing that can.

Deliberately imports nothing from `kaggriculture_lab` at runtime. The hook
contract is duck-typed, so the agent directory stays self-contained and
bundleable; the lab types are pulled in only for type checking.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- typing only, never imported at runtime
    from kaggriculture_lab.hooks import EpisodeHookContext, EpisodeReader

SCHEMA_VERSION = 1

# Episodes needed before a verdict means anything. Below this the window is
# reported but never used to trigger anything.
MIN_HISTORY = 20

# Rolling window used for the trend comparison.
WINDOW = 40

# Re-evaluate when the recent window's mean money margin falls this far below
# the older window's. Scaled to the game: a few hundred coins is noise, a few
# thousand is the collapse signature.
REGRESSION_MARGIN = 1500.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def evaluate_episode(context: EpisodeHookContext, reader: EpisodeReader) -> dict[str, Any]:
    """Summarise this episode and flag whether the checkpoint should be re-evaluated."""
    summary = reader.summary()
    metrics = summary["metrics"]
    history = reader.history(context.revision_id, limit=WINDOW * 2)

    played = [row for row in history if not row["failed"]]
    margins = [float(row["money_difference"] or 0.0) for row in played]
    wins = [float(row["win"] or 0.0) for row in played]

    # Split whatever history we have down the middle rather than at a fixed
    # offset. Slicing at WINDOW left `older` empty for anything under 41
    # episodes, so the check silently reported "insufficient history" well past
    # the MIN_HISTORY it advertises -- it would never have fired in a short
    # league. WINDOW now only caps how much history is pulled.
    half = len(margins) // 2
    recent, older = margins[:half], margins[half:]
    recent_mean, older_mean = _mean(recent), _mean(older)
    regression = older_mean - recent_mean

    enough = len(played) >= MIN_HISTORY and len(older) > 0
    should_reevaluate = bool(enough and regression >= REGRESSION_MARGIN)

    return {
        "schema_version": SCHEMA_VERSION,
        "revision_id": context.revision_id,
        # This episode.
        "final_money": summary["final_money"],
        "money_difference": summary["money_difference"],
        "win": summary["win"],
        "productive_action_percentage": metrics.get("productive_action_percentage", 0.0),
        "movement_percentage": metrics.get("movement_percentage", 0.0),
        "plants_lost_inferred": metrics.get("plants_lost_inferred", 0),
        # Candidate-set truncation is deliberately absent. It is an internal
        # count from the agent's own enumerator, and the league's metrics are
        # derived from the replay, which cannot see it. Reporting
        # `metrics.get("job_truncations", 0)` here would emit a constant zero
        # that reads like data. It is logged per round by the trainer instead.
        # Accumulated evidence.
        "episodes_seen": len(played),
        "rolling_win_rate": _mean(wins),
        "recent_margin_mean": recent_mean,
        "older_margin_mean": older_mean,
        "margin_regression": regression,
        "should_reevaluate": should_reevaluate,
        "reevaluate_reason": (
            f"mean margin fell {regression:,.0f} over the last {len(recent)} episodes"
            if should_reevaluate
            else ("insufficient history" if not enough else "stable")
        ),
    }
