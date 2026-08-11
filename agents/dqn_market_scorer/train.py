"""Training hooks for the market scorer. The lab enters this agent here.

The exploration loop lives behind `on_round`: the league's round schedule is the
clock, and each closed round buys `batches_per_round` rollout batches. Because
the learner is off-policy it does not care who the league's population is, only
that time has passed and the behaviour policy has moved.

`training/` holds the collector, the replay buffer, and the trainer proper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.dqn_market_scorer.training.progress import MarketProgressEvaluator
from agents.dqn_market_scorer.training.updater import MarketCollector, MarketUpdater

if TYPE_CHECKING:
    from kaggriculture_lab.model_trainers import RoundContext, TrainingBatchWriter
    from kaggriculture_lab.model_updates import (
        ModelTrainingReader,
        ModelUpdateContext,
        ModelUpdateResult,
    )


class MarketTrainer:
    def __init__(self) -> None:
        self._collector = MarketCollector()
        self._progress = MarketProgressEvaluator()
        self._updater = MarketUpdater()

    def on_round(
        self,
        context: RoundContext,
        reader: ModelTrainingReader,
        writer: TrainingBatchWriter,
    ) -> None:
        if context.scope == "generation":
            self._progress.observe(context, writer, phase="end")
            return
        if context.round_index == 0:
            self._progress.observe(context, writer, phase="start")
        self._collector.collect(context, reader, writer)

    def on_batch(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        return self._updater.update(context, reader)


trainer = MarketTrainer()
