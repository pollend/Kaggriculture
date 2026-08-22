"""Training hooks for the free-market agent. The lab enters this agent here."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.dqn_market_free_v3c.training.updater import FreeMarketCollector, FreeMarketUpdater

if TYPE_CHECKING:
    from kaggriculture_lab.model_trainers import RoundContext, TrainingBatchWriter
    from kaggriculture_lab.model_updates import (
        ModelTrainingReader,
        ModelUpdateContext,
        ModelUpdateResult,
    )


class FreeMarketTrainer:
    def __init__(self) -> None:
        self._collector = FreeMarketCollector()
        self._updater = FreeMarketUpdater()

    def on_round(
        self,
        context: RoundContext,
        reader: ModelTrainingReader,
        writer: TrainingBatchWriter,
    ) -> None:
        if context.scope == "generation":
            return
        self._collector.collect(context, reader, writer)

    def on_batch(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        return self._updater.update(context, reader)


trainer = FreeMarketTrainer()
