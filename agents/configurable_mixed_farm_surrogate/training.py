"""Evolution and Random Forest surrogates for mixed-farm strategies."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import fields
from typing import Any

from agents.configurable_mixed_farm_surrogate.strategy import Strategy, sample_strategy
from kaggriculture_lab.identity import AgentRevision
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
)

FAMILY_ID = "configurable_mixed_farm_surrogate"
MOON_FAMILY_ID = "competitor_prvsiyan_moon_counts_melons"
MODEL_ID = f"{FAMILY_ID}.surrogate"


class MixedFarmTrainingAdapter:
    family_id = FAMILY_ID

    def decode(self, revision: AgentRevision) -> Strategy:
        if revision.params_json is None:
            raise ValueError("mixed-farm revision has no parameters")
        return Strategy.from_dict(json.loads(revision.params_json))

    def encode(self, strategy: Strategy) -> dict[str, Any]:
        return strategy.to_features()

    def sample(self, rng: random.Random) -> Strategy:
        return sample_strategy(rng)

    def mutate(self, parent: Strategy, rng: random.Random) -> Strategy:
        values = parent.to_dict()
        names = [field.name for field in fields(Strategy)]
        donor = sample_strategy(rng).to_dict()
        for name in rng.sample(names, rng.randint(1, max(1, len(names) // 4))):
            old, replacement = values[name], donor[name]
            values[name] = (
                round((int(old) + int(replacement)) / 2)
                if isinstance(old, int)
                else round(0.7 * float(old) + 0.3 * float(replacement), 6)
            )
        return Strategy.from_dict(values)

    def crossover(self, left: Strategy, right: Strategy, rng: random.Random) -> Strategy:
        child: dict[str, Any] = {}
        right_values = right.to_dict()
        for name, left_value in left.to_dict().items():
            right_value = right_values[name]
            if isinstance(left_value, int):
                child[name] = (
                    round((left_value + int(right_value)) / 2)
                    if rng.random() < 0.5
                    else rng.choice((left_value, right_value))
                )
            else:
                child[name] = round((float(left_value) + float(right_value)) / 2, 6)
        return self.mutate(Strategy.from_dict(child), rng)

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
        import numpy as np
        import pandas as pd

        payload = joblib.load(row[0])
        candidates = [self.sample(rng) for _ in range(candidate_pool)]
        frame = pd.DataFrame([self.encode(candidate) for candidate in candidates])
        columns = payload["feature_columns"]
        core = payload["core_model"].predict(frame[columns])
        moon_model = payload.get("moon_model")
        if moon_model is None:
            scores = core
        else:
            moon = moon_model.predict(frame[columns])
            core_std = float(np.std(core)) or 1.0
            moon_std = float(payload.get("moon_target_std", 1.0)) or 1.0
            core_z = (core - float(np.mean(core))) / core_std
            moon_z = (moon - float(payload.get("moon_target_mean", 0.0))) / moon_std
            weight = float(payload.get("moon_weight", 0.30))
            scores = (1.0 - weight) * core_z + weight * moon_z
        ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
        return [candidate for _, candidate in ranked[:count]]


adapter = MixedFarmTrainingAdapter()


class MixedFarmSurrogateTrainer:
    @staticmethod
    def _model(config: dict[str, Any]):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=int(config.get("n_estimators", 300)),
            min_samples_leaf=int(config.get("min_samples_leaf", 2)),
            n_jobs=-1,
            random_state=int(config.get("seed", 42)),
        )

    @staticmethod
    def _records(reader: ModelTrainingReader, generation_ids: list[str]):
        import pandas as pd

        placeholders = ",".join("%s" for _ in generation_ids)
        rows = reader.fetchall(
            f"""
            WITH episode_rollup AS (
                SELECT lp.generation_id, ep.strategy_id AS revision_id,
                       avg(ep.final_money) AS final_money
                FROM league_pairings lp
                JOIN pairing_episodes pe USING (pairing_id)
                JOIN episode_players ep USING (match_id)
                WHERE lp.generation_id IN ({placeholders}) AND NOT ep.failed
                GROUP BY lp.generation_id, ep.strategy_id
            ), moon_rollup AS (
                SELECT lp.generation_id, ep.strategy_id AS revision_id,
                       avg(ep.money_difference) AS moon_margin
                FROM league_pairings lp
                JOIN pairing_episodes pe USING (pairing_id)
                JOIN episode_players ep USING (match_id)
                JOIN agent_revisions opponent ON opponent.revision_id = ep.opponent_id
                WHERE lp.generation_id IN ({placeholders}) AND NOT ep.failed
                  AND opponent.family_id = %s
                GROUP BY lp.generation_id, ep.strategy_id
            )
            SELECT ar.params_json, rs.rating - 2 * rs.deviation AS conservative,
                   gs.mean_margin, er.final_money, mr.moon_margin,
                   g.generation_index, ar.revision_id, g.generation_id
            FROM rating_snapshots rs
            JOIN generations g USING (generation_id)
            JOIN generation_stats gs USING (generation_id, revision_id)
            JOIN agent_revisions ar USING (revision_id)
            JOIN episode_rollup er USING (generation_id, revision_id)
            LEFT JOIN moon_rollup mr USING (generation_id, revision_id)
            WHERE g.generation_id IN ({placeholders})
              AND ar.family_id = %s AND ar.params_json IS NOT NULL AND NOT rs.is_frozen
            ORDER BY g.generation_index, ar.revision_id
            """,
            [*generation_ids, *generation_ids, MOON_FAMILY_ID, *generation_ids, FAMILY_ID],
        )
        records = [
            {
                **json.loads(row[0]),
                "conservative": float(row[1]),
                "mean_margin": float(row[2]),
                "final_money": float(row[3]),
                "moon_margin": None if row[4] is None else float(row[4]),
                "generation": int(row[5]),
                "revision_id": str(row[6]),
                "generation_id": str(row[7]),
            }
            for row in rows
        ]
        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return frame
        for source, weight in (("conservative", 0.45), ("mean_margin", 0.35), ("final_money", 0.20)):
            means = frame.groupby("generation_id")[source].transform("mean")
            deviations = frame.groupby("generation_id")[source].transform("std").fillna(0.0).replace(0.0, 1.0)
            frame[f"{source}_z"] = (frame[source] - means) / deviations
            frame[f"{source}_weighted"] = frame[f"{source}_z"] * weight
        frame["target"] = (
            frame["conservative_weighted"]
            + frame["mean_margin_weighted"]
            + frame["final_money_weighted"]
        )
        return frame

    def on_batch(self, context: ModelUpdateContext, reader: ModelTrainingReader) -> ModelUpdateResult:
        import joblib
        import numpy as np
        from sklearn.metrics import mean_absolute_error

        generation_ids = [
            batch.source_id for batch in context.all_batches
            if batch.source_kind == "league_generation"
        ]
        if len(generation_ids) < 2:
            raise ValueError("mixed-farm surrogate requires at least two closed generations")
        frame = self._records(reader, generation_ids)
        if len(frame) < 6:
            raise ValueError("at least six rated mixed-farm strategies are required")

        newest = generation_ids[-1]
        excluded = {
            "target", "generation", "revision_id", "generation_id", "moon_margin",
            "conservative", "mean_margin", "final_money",
            "conservative_z", "mean_margin_z", "final_money_z",
            "conservative_weighted", "mean_margin_weighted", "final_money_weighted",
        }
        columns = [column for column in frame.columns if column not in excluded]
        train = frame[frame["generation_id"] != newest]
        validation = frame[frame["generation_id"] == newest]
        if train.empty or validation.empty:
            raise ValueError("training and validation generations must both contain rows")

        core_candidate = self._model(context.config)
        core_candidate.fit(train[columns], train["target"])
        core_mae = float(mean_absolute_error(validation["target"], core_candidate.predict(validation[columns])))
        core_scale = float(np.std(validation["target"])) or 1.0
        candidate_error = core_mae / core_scale

        moon_min_rows = int(context.config.get("moon_min_rows", 20))
        moon_weight = float(context.config.get("moon_weight", 0.30))
        moon_train = train.dropna(subset=["moon_margin"])
        moon_validation = validation.dropna(subset=["moon_margin"])
        moon_candidate = None
        moon_mae = None
        if moon_train["revision_id"].nunique() >= moon_min_rows and not moon_validation.empty:
            moon_candidate = self._model({**context.config, "seed": int(context.config.get("seed", 42)) + 1})
            moon_candidate.fit(moon_train[columns], moon_train["moon_margin"])
            moon_mae = float(
                mean_absolute_error(
                    moon_validation["moon_margin"], moon_candidate.predict(moon_validation[columns])
                )
            )
            moon_scale = float(np.std(moon_validation["moon_margin"])) or float(np.std(moon_train["moon_margin"])) or 1.0
            candidate_error = (1.0 - moon_weight) * candidate_error + moon_weight * moon_mae / moon_scale

        incumbent_error: float | None = None
        if context.active_checkpoint is not None:
            incumbent = joblib.load(context.active_checkpoint.inference_artifact_path)
            incumbent_core_mae = float(
                mean_absolute_error(
                    validation["target"], incumbent["core_model"].predict(validation[incumbent["feature_columns"]])
                )
            )
            incumbent_error = incumbent_core_mae / core_scale
            incumbent_moon = incumbent.get("moon_model")
            if incumbent_moon is not None and not moon_validation.empty:
                incumbent_moon_mae = float(
                    mean_absolute_error(
                        moon_validation["moon_margin"],
                        incumbent_moon.predict(moon_validation[incumbent["feature_columns"]]),
                    )
                )
                moon_scale = float(np.std(moon_validation["moon_margin"])) or float(incumbent.get("moon_target_std", 1.0)) or 1.0
                incumbent_error = (
                    (1.0 - moon_weight) * incumbent_error
                    + moon_weight * incumbent_moon_mae / moon_scale
                )
        promote = incumbent_error is None or candidate_error <= incumbent_error

        core_persisted = self._model(context.config)
        core_persisted.fit(frame[columns], frame["target"])
        moon_rows = frame.dropna(subset=["moon_margin"])
        moon_persisted = None
        if moon_rows["revision_id"].nunique() >= moon_min_rows:
            moon_persisted = self._model({**context.config, "seed": int(context.config.get("seed", 42)) + 1})
            moon_persisted.fit(moon_rows[columns], moon_rows["moon_margin"])

        checkpoint_id = hashlib.sha256(
            f"{context.model_id}\0{context.update_id}\0{context.dataset_sha256}".encode()
        ).hexdigest()[:24]
        destination = context.artifacts / "checkpoints"
        destination.mkdir(parents=True, exist_ok=True)
        artifact = destination / f"{checkpoint_id}.joblib"
        moon_mean = float(moon_rows["moon_margin"].mean()) if not moon_rows.empty else 0.0
        moon_std = float(moon_rows["moon_margin"].std()) if len(moon_rows) > 1 else 1.0
        joblib.dump(
            {
                "core_model": core_persisted,
                "moon_model": moon_persisted,
                "feature_columns": columns,
                "moon_target_mean": moon_mean,
                "moon_target_std": moon_std or 1.0,
                "moon_weight": moon_weight,
                "dataset_sha256": context.dataset_sha256,
            },
            artifact,
        )
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        metrics = {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "all_rows": len(frame),
            "moon_rows": len(moon_rows),
            "moon_strategies": int(moon_rows["revision_id"].nunique()),
            "core_mae": core_mae,
            "moon_mae": moon_mae,
            "candidate_weighted_error": candidate_error,
            "incumbent_weighted_error": incumbent_error,
            "validation_generation": newest,
            "feature_columns": columns,
        }
        gate = {
            "kind": "held_out_weighted_mae_non_regression",
            "passed": promote,
            "candidate_error": candidate_error,
            "incumbent_error": incumbent_error,
        }
        evaluation = ModelEvaluation(
            "held_out_generation",
            metrics,
            promote,
            context.active_checkpoint.checkpoint_id if context.active_checkpoint is not None else None,
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


trainer = MixedFarmSurrogateTrainer()

