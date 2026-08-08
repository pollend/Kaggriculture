"""Training hooks for the crop surrogate. The lab enters this agent here.

Two calls, and nothing else crosses the boundary: `on_round` when the league
closes a round or a generation, `on_batch` when enough new data has accumulated
to be worth fitting. `training.py` holds the actual model work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.configurable_crop_surrogate.training import CropSurrogateUpdater

if TYPE_CHECKING:
    from kaggriculture_lab.model_trainers import RoundContext, TrainingBatchWriter
    from kaggriculture_lab.model_updates import (
        ModelTrainingReader,
        ModelUpdateContext,
        ModelUpdateResult,
    )


class CropSurrogateTrainer:
    def __init__(self) -> None:
        self._updater = CropSurrogateUpdater()

    def on_round(
        self,
        context: RoundContext,
        reader: ModelTrainingReader,
        writer: TrainingBatchWriter,
    ) -> None:
        """One closed generation of rated strategies is one training batch.

        Individual rounds carry no ratings -- those are computed once, when the
        generation closes -- so a round-scope call has nothing to record.
        """

        if context.scope != "generation":
            return
        row = reader.fetchone(
            "SELECT count(*) FROM rating_snapshots "
            "WHERE generation_id = ? AND NOT is_frozen",
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

    def on_batch(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        return self._updater.update(context, reader)


trainer = CropSurrogateTrainer()
