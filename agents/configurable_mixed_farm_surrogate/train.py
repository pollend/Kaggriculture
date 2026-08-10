"""League-owned training hooks for the mixed-farm surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.configurable_mixed_farm_surrogate.training import MixedFarmSurrogateTrainer

if TYPE_CHECKING:
    from kaggriculture_lab.model_trainers import RoundContext, TrainingBatchWriter
    from kaggriculture_lab.model_updates import ModelTrainingReader, ModelUpdateContext, ModelUpdateResult


class MixedFarmTrainer:
    def __init__(self) -> None:
        self._surrogate = MixedFarmSurrogateTrainer()

    def on_round(self, context: RoundContext, reader: ModelTrainingReader, writer: TrainingBatchWriter) -> None:
        if context.scope != "generation":
            return
        row = reader.fetchone(
            "SELECT count(*) FROM rating_snapshots WHERE generation_id = %s AND NOT is_frozen",
            [context.generation_id],
        )
        writer.record(
            source_kind="league_generation",
            source_id=context.generation_id,
            sample_count=int(row[0]) if row else 0,
            metadata={"session_id": context.session_id, "generation_id": context.generation_id},
        )

    def on_batch(self, context: ModelUpdateContext, reader: ModelTrainingReader) -> ModelUpdateResult:
        return self._surrogate.on_batch(context, reader)


trainer = MixedFarmTrainer()

