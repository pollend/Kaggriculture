"""Evolution and conservative-rating surrogate for adaptive route profiles."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import fields
from typing import Any

from agents.adaptive_route_planner.strategy import Strategy, sample_strategy
from kaggriculture_lab.identity import AgentRevision
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
)

FAMILY_ID = "adaptive_route_planner"


class AdaptiveRouteTrainingAdapter:
    family_id = FAMILY_ID

    def decode(self, revision: AgentRevision) -> Strategy:
        if revision.params_json is None:
            raise ValueError("adaptive route revision has no parameters")
        return Strategy.from_dict(json.loads(revision.params_json))

    def encode(self, strategy: Strategy) -> dict[str, Any]:
        return strategy.to_features()

    def sample(self, rng: random.Random) -> Strategy:
        return sample_strategy(rng)

    def mutate(self, parent: Strategy, rng: random.Random) -> Strategy:
        values = parent.to_dict()
        donor = sample_strategy(rng).to_dict()
        names = [field.name for field in fields(Strategy)]
        for name in rng.sample(names, rng.randint(1, max(1, len(names) // 4))):
            old, replacement = values[name], donor[name]
            if isinstance(old, int):
                values[name] = round((old + int(replacement)) / 2)
            else:
                values[name] = round(0.7 * float(old) + 0.3 * float(replacement), 6)
        if values["second_land_day"] <= values["first_land_day"]:
            values["second_land_day"] = min(16, values["first_land_day"] + 1)
            if values["second_land_day"] <= values["first_land_day"]:
                values["first_land_day"] = values["second_land_day"] - 1
        overflow = values["cow_target"] + values["sheep_target"] - 16
        if overflow > 0:
            values["sheep_target"] -= overflow
        return Strategy.from_dict(values)

    def crossover(self, left: Strategy, right: Strategy, rng: random.Random) -> Strategy:
        left_values, right_values = left.to_dict(), right.to_dict()
        child: dict[str, Any] = {}
        for name, left_value in left_values.items():
            right_value = right_values[name]
            if isinstance(left_value, int):
                child[name] = rng.choice((left_value, right_value)) if rng.random() < 0.5 else round((left_value + int(right_value)) / 2)
            else:
                child[name] = round((float(left_value) + float(right_value)) / 2, 6)
        # Normalization is centralized in mutation's invariant repair.
        try:
            seed = Strategy.from_dict(child)
        except ValueError:
            child["second_land_day"] = max(child["first_land_day"] + 1, child["second_land_day"])
            if child["cow_target"] + child["sheep_target"] > 16:
                child["sheep_target"] = 16 - child["cow_target"]
            seed = Strategy.from_dict(child)
        return self.mutate(seed, rng)

    def propose(
        self,
        count: int,
        parents: list[AgentRevision],
        rng: random.Random,
        reader: ModelTrainingReader,
        model_id: str | None = None,
        candidate_pool: int = 10_000,
    ) -> list[Strategy]:
        del parents
        if model_id is None:
            return []
        row = reader.fetchone(
            """
            SELECT mtc.inference_artifact_path
            FROM training_models tm
            JOIN model_training_checkpoints mtc
              ON mtc.checkpoint_id = tm.active_checkpoint_id
            WHERE tm.model_id = %s
            """,
            [model_id],
        )
        if not row:
            return []
        import joblib
        import pandas as pd

        payload = joblib.load(row[0])
        candidates = [self.sample(rng) for _ in range(candidate_pool)]
        frame = pd.DataFrame([self.encode(candidate) for candidate in candidates])
        scores = payload["model"].predict(frame[payload["feature_columns"]])
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [candidate for _, candidate in ranked[:count]]


adapter = AdaptiveRouteTrainingAdapter()


class AdaptiveRouteSurrogateTrainer:
    @staticmethod
    def _pipeline(frame, config: dict[str, Any]):
        from sklearn.ensemble import RandomForestRegressor

        columns = [column for column in frame.columns if column not in {"target", "generation", "revision_id"}]
        model = RandomForestRegressor(
            n_estimators=int(config.get("n_estimators", 300)),
            min_samples_leaf=int(config.get("min_samples_leaf", 2)),
            n_jobs=-1,
            random_state=int(config.get("seed", 42)),
        )
        return model, columns

    def on_batch(self, context: ModelUpdateContext, reader: ModelTrainingReader) -> ModelUpdateResult:
        import joblib
        import pandas as pd
        from sklearn.metrics import mean_absolute_error

        generation_ids = [batch.source_id for batch in context.all_batches if batch.source_kind == "league_generation"]
        if len(generation_ids) < 2:
            raise ValueError("adaptive route surrogate requires two closed generations")
        placeholders = ",".join("%s" for _ in generation_ids)
        rows = reader.fetchall(
            f"""
            SELECT ar.params_json, rs.rating - 2 * rs.deviation AS target,
                   g.generation_index, ar.revision_id, g.generation_id
            FROM rating_snapshots rs
            JOIN generations g USING (generation_id)
            JOIN agent_revisions ar USING (revision_id)
            WHERE g.generation_id IN ({placeholders}) AND ar.family_id = %s
              AND ar.params_json IS NOT NULL AND NOT rs.is_frozen
            ORDER BY g.generation_index, ar.revision_id
            """,
            [*generation_ids, FAMILY_ID],
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
            raise ValueError("at least six rated adaptive route rows are required")
        newest = generation_ids[-1]
        validation = frame[frame["generation_id"] == newest].drop(columns=["generation_id"])
        train = frame[frame["generation_id"] != newest].drop(columns=["generation_id"])
        all_rows = frame.drop(columns=["generation_id"])
        if train.empty or validation.empty:
            raise ValueError("training and validation generations must both contain rows")

        candidate, columns = self._pipeline(train, context.config)
        candidate.fit(train[columns], train["target"])
        candidate_mae = float(mean_absolute_error(validation["target"], candidate.predict(validation[columns])))
        incumbent_mae: float | None = None
        if context.active_checkpoint is not None:
            incumbent = joblib.load(context.active_checkpoint.inference_artifact_path)
            incumbent_mae = float(mean_absolute_error(
                validation["target"],
                incumbent["model"].predict(validation[incumbent["feature_columns"]]),
            ))
        promote = incumbent_mae is None or candidate_mae <= incumbent_mae

        persisted, columns = self._pipeline(all_rows, context.config)
        persisted.fit(all_rows[columns], all_rows["target"])
        checkpoint_id = hashlib.sha256(
            f"{context.model_id}\0{context.update_id}\0{context.dataset_sha256}".encode()
        ).hexdigest()[:24]
        destination = context.artifacts / "checkpoints"
        destination.mkdir(parents=True, exist_ok=True)
        artifact = destination / f"{checkpoint_id}.joblib"
        joblib.dump({"model": persisted, "feature_columns": columns, "dataset_sha256": context.dataset_sha256}, artifact)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        metrics = {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "all_rows": len(all_rows),
            "candidate_mae": candidate_mae,
            "incumbent_mae": incumbent_mae,
            "validation_generation": newest,
            "feature_columns": columns,
        }
        gate = {"kind": "held_out_mae_non_regression", "passed": promote, "candidate_mae": candidate_mae, "incumbent_mae": incumbent_mae}
        evaluation = ModelEvaluation(
            "held_out_generation",
            metrics,
            promote,
            context.active_checkpoint.checkpoint_id if context.active_checkpoint else None,
        )
        return ModelUpdateResult(
            checkpoint_id,
            str(artifact),
            str(artifact),
            digest,
            metrics,
            gate,
            promote,
            (evaluation,),
        )

