"""Training hooks for the free-market agent. The lab enters this agent here."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.dqn_market_free_v3c.training.updater import (
    FreeMarketCollector,
    FreeMarketSelector,
    FreeMarketUpdater,
)

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
        self._selector = FreeMarketSelector()

    def on_round(
        self,
        context: RoundContext,
        reader: ModelTrainingReader,
        writer: TrainingBatchWriter,
    ) -> None:
        # Generation scope is the prune. The lab fires it once per generation from
        # `_close_generation_training`, after every slot's updates have landed --
        # which is the only moment all of this generation's checkpoints and their
        # scores exist. It used to be discarded here.
        if context.scope == "generation":
            self._selector.select(context, reader)
            return
        self._collector.collect(context, reader, writer)

    def on_batch(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        return self._updater.update(context, reader)


trainer = FreeMarketTrainer()
