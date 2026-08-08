"""Training hooks for the receding-horizon planner. The lab enters this agent here.

Same shape as the crop surrogate: the league generation is the unit of training
data, and fitting happens once enough generations have closed. `training.py`
holds the evolution adapter and the surrogate fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.receding_horizon_planner.training import PlannerSurrogateTrainer

if TYPE_CHECKING:
    from kaggriculture_lab.model_trainers import RoundContext, TrainingBatchWriter
    from kaggriculture_lab.model_updates import (
        ModelTrainingReader,
        ModelUpdateContext,
        ModelUpdateResult,
    )


class PlannerTrainer:
    def __init__(self) -> None:
        self._surrogate = PlannerSurrogateTrainer()

    def on_round(
        self,
        context: RoundContext,
        reader: ModelTrainingReader,
        writer: TrainingBatchWriter,
    ) -> None:
        """Ratings exist only once the generation closes, so rounds are no-ops."""

        if context.scope != "generation":
            return
        row = reader.fetchone(
            "SELECT count(*) FROM rating_snapshots WHERE generation_id = ? AND NOT is_frozen",
            [context.generation_id],
        )
        writer.record(
            source_kind="league_generation",
            source_id=context.generation_id,
            sample_count=int(row[0]) if row else 0,
            metadata={
                "session_id": context.session_id,
                "generation_id": context.generation_id,
            },
        )

    def on_batch(self, context: ModelUpdateContext, reader: ModelTrainingReader) -> ModelUpdateResult:
        return self._surrogate.on_batch(context, reader)


trainer = PlannerTrainer()
