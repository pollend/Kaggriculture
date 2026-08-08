"""Evolution operators for the configurable crop strategy family."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import fields
from enum import StrEnum
from typing import Any

from agents.configurable_crop_surrogate.strategy import Strategy, sample_strategy
from kaggriculture_lab.identity import AgentRevision
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
)


class ConfigurableCropAdapter:
    family_id = "configurable_crop_surrogate"

    def decode(self, revision: AgentRevision) -> Strategy:
        if revision.params_json is None:
            raise ValueError("configurable crop revision has no parameters")
        return Strategy.from_dict(json.loads(revision.params_json))

    def encode(self, strategy: Strategy) -> dict[str, Any]:
        return strategy.to_features()

    def sample(self, rng: random.Random) -> Strategy:
        return sample_strategy(rng)

    def mutate(self, parent: Strategy, rng: random.Random) -> Strategy:
        values = parent.to_dict()
        field_names = [field.name for field in fields(Strategy)]
        mutations = rng.randint(1, max(1, len(field_names) // 4))
        donor = sample_strategy(rng).to_dict()
        for name in rng.sample(field_names, mutations):
            old = values[name]
            replacement = donor[name]
            if isinstance(old, bool) or isinstance(old, str):
                values[name] = replacement
            elif isinstance(old, int):
                values[name] = round((old + replacement) / 2)
            else:
                values[name] = round(0.7 * float(old) + 0.3 * float(replacement), 6)
        return Strategy.from_dict(values)

    def crossover(self, left: Strategy, right: Strategy, rng: random.Random) -> Strategy:
        left_values = left.to_dict()
        right_values = right.to_dict()
        child: dict[str, Any] = {}
        for name in left_values:
            a, b = left_values[name], right_values[name]
            if isinstance(a, bool) or isinstance(a, str) or isinstance(a, StrEnum):
                child[name] = rng.choice((a, b))
            elif isinstance(a, int):
                child[name] = round((a + b) / 2) if rng.random() < 0.5 else rng.choice((a, b))
            else:
                child[name] = round((float(a) + float(b)) / 2, 6)
        return self.mutate(Strategy.from_dict(child), rng)

    def propose(
        self,
        count: int,
        parents: list[AgentRevision],
        rng: random.Random,
        reader: ModelTrainingReader,
        candidate_pool: int = 2_000,
    ) -> list[Strategy]:
        """Rank a random candidate pool with this family's surrogate model.

        Called by the league when filling the `surrogate` share of a generation.
        Which model is used, and how candidates are scored, is this family's
        business -- the league previously hardcoded both, which meant a second
        agent family could not breed at all.

        Returns an empty list when no checkpoint is active yet; the league falls
        back to mutation.
        """
        row = reader.fetchone(
            """
            SELECT mtc.inference_artifact_path
            FROM training_models tm
            JOIN model_training_checkpoints mtc
              ON mtc.checkpoint_id = tm.active_checkpoint_id
            WHERE tm.model_id = ?
            """,
            [f"{self.family_id}.surrogate"],
        )
        if not row:
            return []

        import joblib
        import pandas as pd

        payload = joblib.load(row[0])
        candidates = [self.sample(rng) for _ in range(candidate_pool)]
        frame = pd.DataFrame([self.encode(item) for item in candidates])
        scores = payload["model"].predict(frame[payload["feature_columns"]])
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [candidate for _, candidate in ranked[:count]]


adapter = ConfigurableCropAdapter()


class CropSurrogateUpdater:
    """Train and gate the strategy-quality surrogate from closed generations."""

    @staticmethod
    def _pipeline(frame, config: dict[str, Any]):
        from pandas.api.types import is_numeric_dtype
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        feature_columns = [
            column
            for column in frame.columns
            if column not in {"target", "generation", "revision_id"}
        ]
        categorical = [
            column for column in feature_columns if not is_numeric_dtype(frame[column])
        ]
        transformer = ColumnTransformer(
            [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
            remainder="passthrough",
        )
        model = Pipeline(
            [
                ("features", transformer),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(config.get("n_estimators", 300)),
                        min_samples_leaf=int(config.get("min_samples_leaf", 2)),
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        return model, feature_columns, categorical

    def update(
        self, context: ModelUpdateContext, reader: ModelTrainingReader
    ) -> ModelUpdateResult:
        import joblib
        import pandas as pd
        from sklearn.metrics import mean_absolute_error

        generation_ids = [
            batch.source_id
            for batch in context.all_batches
            if batch.source_kind == "league_generation"
        ]
        if len(generation_ids) < 2:
            raise ValueError("crop surrogate requires at least two closed generations")
        placeholders = ",".join("?" for _ in generation_ids)
        rows = reader.fetchall(
            f"""
            SELECT ar.params_json, rs.rating - 2 * rs.deviation AS target,
                   g.generation_index, ar.revision_id, g.generation_id
            FROM rating_snapshots rs
            JOIN generations g ON g.generation_id = rs.generation_id
            JOIN agent_revisions ar ON ar.revision_id = rs.revision_id
            WHERE g.generation_id IN ({placeholders})
              AND ar.params_json IS NOT NULL AND NOT rs.is_frozen
            ORDER BY g.generation_index, ar.revision_id
            """,
            generation_ids,
        )
        records = [
            {
                **json.loads(row[0]),
                "target": float(row[1]),
                "generation": int(row[2]),
                "revision_id": str(row[3]),
                "generation_id": str(row[4]),
            }
            for row in rows
        ]
        frame = pd.DataFrame.from_records(records)
        if len(frame) < 6:
            raise ValueError("at least six rated strategy rows are required")

        newest_generation = generation_ids[-1]
        validation = frame[frame["generation_id"] == newest_generation].copy()
        train = frame[frame["generation_id"] != newest_generation].copy()
        if train.empty or validation.empty:
            raise ValueError("training and validation generations must both contain rows")
        frame = frame.drop(columns=["generation_id"])
        train = train.drop(columns=["generation_id"])
        validation = validation.drop(columns=["generation_id"])

        candidate, columns, categorical = self._pipeline(train, context.config)
        candidate.fit(train[columns], train["target"])
        candidate_mae = float(
            mean_absolute_error(
                validation["target"], candidate.predict(validation[columns])
            )
        )
        incumbent_mae: float | None = None
        if context.active_checkpoint is not None:
            incumbent_payload = joblib.load(
                context.active_checkpoint.inference_artifact_path
            )
            incumbent_mae = float(
                mean_absolute_error(
                    validation["target"],
                    incumbent_payload["model"].predict(
                        validation[incumbent_payload["feature_columns"]]
                    ),
                )
            )
        promote = incumbent_mae is None or candidate_mae <= incumbent_mae

        # The gated score is honest; the persisted candidate is then refit on all
        # available rows so a promoted model does not discard its validation data.
        persisted, columns, categorical = self._pipeline(frame, context.config)
        persisted.fit(frame[columns], frame["target"])
        checkpoint_id = hashlib.sha256(
            f"{context.model_id}\0{context.update_id}\0{context.dataset_sha256}".encode()
        ).hexdigest()[:24]
        destination = context.artifacts / "model_checkpoints" / context.model_id
        destination.mkdir(parents=True, exist_ok=True)
        artifact = destination / f"{checkpoint_id}.joblib"
        joblib.dump(
            {
                "model": persisted,
                "feature_columns": columns,
                "dataset_sha256": context.dataset_sha256,
            },
            artifact,
        )
        artifact_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        metrics = {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "all_rows": len(frame),
            "candidate_mae": candidate_mae,
            "incumbent_mae": incumbent_mae,
            "validation_generation": newest_generation,
            "feature_columns": columns,
            "categorical_columns": categorical,
        }
        gate = {
            "kind": "held_out_mae_non_regression",
            "passed": promote,
            "candidate_mae": candidate_mae,
            "incumbent_mae": incumbent_mae,
        }
        evaluation = ModelEvaluation(
            kind="held_out_generation",
            metrics=metrics,
            passed=promote,
            opponent_checkpoint_id=(
                context.active_checkpoint.checkpoint_id
                if context.active_checkpoint is not None
                else None
            ),
        )
        return ModelUpdateResult(
            checkpoint_id=checkpoint_id,
            trainer_artifact_path=str(artifact),
            inference_artifact_path=str(artifact),
            artifact_sha256=artifact_digest,
            metrics=metrics,
            gate=gate,
            promote=promote,
            evaluations=(evaluation,),
        )


# The manifest names `train.py`, which constructs this itself. `adapter` above
# is a different seam -- evolution, not training -- and is still resolved by path.
