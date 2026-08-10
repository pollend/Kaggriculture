"""Evolution and Random Forest surrogates for mixed-farm strategies."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import fields
from typing import Any

from agents.mixed_farm_v5.strategy_v5 import Strategy, sample_strategy
from kaggriculture_lab.identity import AgentRevision
from kaggriculture_lab.model_updates import (
    ModelEvaluation,
    ModelTrainingReader,
    ModelUpdateContext,
    ModelUpdateResult,
)

FAMILY_ID = "mixed_farm_v5"
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

    def initialize_population(
        self,
        count: int,
        rng: random.Random,
        reader: ModelTrainingReader,
        seed_session_ids: tuple[str, ...],
    ) -> list[tuple[Strategy, str]]:
        """Open generation zero from the bounded prior alone.

        V5 no longer inherits v3/v4 champions.  Those policies were tuned
        against a labor schema that v5 discards, and the lineage they seeded was
        not competitive, so every run starts from fresh samples.
        """

        return [(self.sample(rng), "random") for _ in range(count)]

    def mutate(self, parent: Strategy, rng: random.Random) -> Strategy:
        if not isinstance(parent, Strategy):
            parent = Strategy.from_dict(parent.to_dict())
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
        values["max_hands"] = min(16, int(values["max_hands"]))
        return Strategy.from_dict(values)

    def crossover(self, left: Strategy, right: Strategy, rng: random.Random) -> Strategy:
        if not isinstance(left, Strategy):
            left = Strategy.from_dict(left.to_dict())
        if not isinstance(right, Strategy):
            right = Strategy.from_dict(right.to_dict())
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
        child["max_hands"] = min(16, int(child["max_hands"]))
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
        quality = payload.get("quality", {})
        if (
            int(quality.get("rows", 0)) < 60
            or float(quality.get("core_error", float("inf"))) > 0.75
            or float(quality.get("productive_mae", float("inf"))) > 0.08
            or float(quality.get("idle_mae", float("inf"))) > 0.08
        ):
            return []
        usable_parents = [
            self.decode(parent)
            for parent in parents
            if parent.family_id == FAMILY_ID and parent.params_json is not None
        ]
        global_count = candidate_pool - candidate_pool // 2
        candidates = [self.sample(rng) for _ in range(global_count)]
        while len(candidates) < candidate_pool:
            if len(usable_parents) >= 2 and len(candidates) % 2:
                left, right = rng.sample(usable_parents, 2)
                candidates.append(self.crossover(left, right, rng))
            elif usable_parents:
                candidates.append(self.mutate(rng.choice(usable_parents), rng))
            else:
                candidates.append(self.sample(rng))
        frame = pd.DataFrame([self.encode(candidate) for candidate in candidates])
        columns = payload["feature_columns"]
        model = payload["core_model"]
        matrix = frame[columns].to_numpy(dtype=float)
        tree_predictions = np.asarray(
            [tree.predict(matrix) for tree in model.estimators_], dtype=float
        )
        means = np.mean(tree_predictions, axis=0)
        deviations = np.std(tree_predictions, axis=0)
        scale = np.ptp(matrix, axis=0)
        scale[scale == 0] = 1.0
        normalized = (matrix - np.min(matrix, axis=0)) / scale
        productive_model = payload["productive_model"]
        idle_model = payload["idle_model"]
        productive_predictions = productive_model.predict(matrix)
        idle_predictions = idle_model.predict(matrix)
        trusted = set(np.where((productive_predictions >= 0.20) & (idle_predictions <= 0.30))[0])
        selected: list[int] = []

        def add_ranked(ranking: list[int], amount: int, *, threshold: float = 0.15) -> None:
            start = len(selected)
            for enforce_distance in (True, False):
                for index in ranking:
                    if len(selected) - start >= amount or len(selected) >= target:
                        return
                    if index in selected:
                        continue
                    if index not in trusted:
                        continue
                    if enforce_distance and selected and min(
                        float(np.sqrt(np.mean((normalized[index] - normalized[chosen]) ** 2)))
                        for chosen in selected
                    ) < threshold:
                        continue
                    selected.append(index)

        exploitation = math.ceil(count * 0.5)
        uncertainty_aware = math.ceil((count - exploitation) * 0.5)
        exploration = count - exploitation - uncertainty_aware
        target = exploitation
        add_ranked(list(np.argsort(-means)), exploitation)
        target += uncertainty_aware
        add_ranked(list(np.argsort(-(means + 0.5 * deviations))), uncertainty_aware)
        target += exploration
        eligible = np.where(means >= float(np.median(means)))[0]
        add_ranked(
            [int(index) for index in eligible[np.argsort(-deviations[eligible])]],
            exploration,
        )
        if len(selected) < count:
            target = count
            add_ranked(list(np.argsort(-means)), count - len(selected), threshold=0.0)
        return [candidates[index] for index in selected[:count]]


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
        import numpy as np
        import pandas as pd

        placeholders = ",".join("%s" for _ in generation_ids)
        rows = reader.fetchall(
            f"""
            WITH episode_rollup AS (
                SELECT lp.generation_id, ep.strategy_id AS revision_id,
                       avg(ep.final_money) AS final_money,
                       avg(ep.total_unit_actions) AS total_actions,
                       avg(ep.productive_actions) AS productive_actions,
                       avg(ep.movement_actions) AS movement_actions,
                       avg(ep.plants_lost_inferred) AS plants_lost,
                       avg(COALESCE(
                           ((ep.metrics_json->'market_orders')->>'HIRE')::DOUBLE PRECISION,
                           0
                       )) AS hire_orders,
                       avg(COALESCE((ee.result_json->>'hand_productive_percentage')::DOUBLE PRECISION, 0)) AS hand_productive,
                       avg(COALESCE((ee.result_json->>'hand_idle_percentage')::DOUBLE PRECISION, 1)) AS hand_idle,
                       sum(COALESCE((ee.result_json->>'reserve_violations')::INTEGER, 0)) AS reserve_violations
                FROM league_pairings lp
                JOIN pairing_episodes pe USING (pairing_id)
                JOIN episode_players ep USING (match_id)
                LEFT JOIN episode_evaluations ee ON ee.match_id = ep.match_id
                  AND ee.revision_id = ep.strategy_id AND ee.seat = ep.seat
                  AND ee.status = 'COMPLETE'
                WHERE lp.generation_id IN ({placeholders}) AND NOT ep.failed
                GROUP BY lp.generation_id, ep.strategy_id
            )
            SELECT ar.params_json, rs.rating - 2 * rs.deviation AS conservative,
                   gs.mean_margin, er.final_money, er.total_actions,
                   er.productive_actions, er.movement_actions, er.plants_lost,
                   er.hire_orders, er.hand_productive, er.hand_idle,
                   er.reserve_violations, g.generation_index, ar.revision_id,
                   g.generation_id, ar.source
            FROM rating_snapshots rs
            JOIN generations g USING (generation_id)
            JOIN generation_stats gs USING (generation_id, revision_id)
            JOIN agent_revisions ar USING (revision_id)
            JOIN episode_rollup er USING (generation_id, revision_id)
            WHERE g.generation_id IN ({placeholders})
              AND ar.family_id = %s AND ar.params_json IS NOT NULL AND NOT rs.is_frozen
            ORDER BY g.generation_index, ar.revision_id
            """,
            [*generation_ids, *generation_ids, FAMILY_ID],
        )
        def labor_values(row: tuple[Any, ...]) -> tuple[float, float, int, int, str, str, str]:
            if len(row) >= 16:
                return float(row[9]), float(row[10]), int(row[11]), int(row[12]), str(row[13]), str(row[14]), str(row[15])
            return 0.0, 1.0, 0, int(row[9]), str(row[10]), str(row[11]), str(row[12])

        records = [
            {
                **adapter.encode(Strategy.from_dict(json.loads(row[0]))),
                "conservative": float(row[1]),
                "mean_margin": float(row[2]),
                "final_money": float(row[3]),
                "total_actions": float(row[4]),
                "productive_actions": float(row[5]),
                "movement_actions": float(row[6]),
                "plants_lost": float(row[7]),
                "hire_orders": float(row[8]),
                "hand_productive": labor_values(row)[0],
                "hand_idle": labor_values(row)[1],
                "reserve_violations": labor_values(row)[2],
                "generation": labor_values(row)[3],
                "revision_id": labor_values(row)[4],
                "generation_id": labor_values(row)[5],
                "source": labor_values(row)[6],
            }
            for row in rows
        ]
        frame = pd.DataFrame.from_records(records)
        if frame.empty:
            return frame
        anchor_rows = reader.fetchall(
            f"""
            SELECT lp.generation_id, ep.strategy_id AS revision_id,
                   opponent.family_id AS opponent_family,
                   avg(ep.money_difference) AS paired_margin
            FROM league_pairings lp
            JOIN pairing_episodes pe USING (pairing_id)
            JOIN episode_players ep USING (match_id)
            JOIN agent_revisions policy ON policy.revision_id = ep.strategy_id
            JOIN agent_revisions opponent ON opponent.revision_id = ep.opponent_id
            WHERE lp.generation_id IN ({placeholders}) AND NOT ep.failed
              AND lp.pairing_kind = 'anchor' AND policy.family_id = %s
            GROUP BY lp.generation_id, ep.strategy_id, opponent.family_id
            """,
            [*generation_ids, FAMILY_ID],
        )
        if not anchor_rows:
            raise ValueError("mixed-farm v4 training requires fixed-anchor episode rows")
        anchors = pd.DataFrame.from_records(
            anchor_rows,
            columns=("generation_id", "revision_id", "opponent_family", "margin"),
        )
        anchors["margin"] = anchors["margin"].astype(float)
        grouped = anchors.groupby(["generation_id", "opponent_family"])["margin"]
        means = grouped.transform("mean")
        deviations = grouped.transform("std").fillna(0.0).replace(0.0, 1.0)
        anchors["margin_z"] = (anchors["margin"] - means) / deviations
        summary = anchors.groupby(["generation_id", "revision_id"]).agg(
            anchor_macro=("margin_z", "mean"),
            anchor_floor=("margin_z", lambda values: float(np.quantile(values, 0.25))),
            anchor_opponents=("opponent_family", "nunique"),
        ).reset_index()
        moon = (
            anchors[anchors["opponent_family"] == MOON_FAMILY_ID]
            .groupby(["generation_id", "revision_id"])["margin"]
            .mean()
            .rename("moon_margin")
            .reset_index()
        )
        frame = frame.merge(summary, on=["generation_id", "revision_id"], how="left")
        frame = frame.merge(moon, on=["generation_id", "revision_id"], how="left")
        if frame[["anchor_macro", "anchor_floor"]].isna().any().any():
            raise ValueError("every mixed-farm policy must have fixed-anchor results")
        for source, weight in (
            ("conservative", 0.40),
            ("anchor_macro", 0.25),
            ("anchor_floor", 0.20),
            ("final_money", 0.15),
        ):
            means = frame.groupby("generation_id")[source].transform("mean")
            deviations = frame.groupby("generation_id")[source].transform("std").fillna(0.0).replace(0.0, 1.0)
            frame[f"{source}_z"] = (frame[source] - means) / deviations
            frame[f"{source}_weighted"] = frame[f"{source}_z"] * weight
        frame["target"] = (
            frame["conservative_weighted"]
            + frame["anchor_macro_weighted"]
            + frame["anchor_floor_weighted"]
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
        if len(frame) < 60:
            raise ValueError("at least sixty rated mixed-farm strategies are required")

        newest = generation_ids[-1]
        columns = list(adapter.encode(Strategy()).keys())
        train = frame[frame["generation_id"] != newest]
        validation = frame[frame["generation_id"] == newest]
        if train.empty or validation.empty:
            raise ValueError("training and validation generations must both contain rows")

        core_candidate = self._model(context.config)
        core_candidate.fit(train[columns], train["target"])
        core_mae = float(mean_absolute_error(validation["target"], core_candidate.predict(validation[columns])))
        core_scale = float(np.std(validation["target"])) or 1.0
        candidate_error = core_mae / core_scale
        productive_candidate = self._model({**context.config, "seed": int(context.config.get("seed", 42)) + 2})
        idle_candidate = self._model({**context.config, "seed": int(context.config.get("seed", 42)) + 3})
        productive_candidate.fit(train[columns], train["hand_productive"])
        idle_candidate.fit(train[columns], train["hand_idle"])
        productive_mae = float(mean_absolute_error(validation["hand_productive"], productive_candidate.predict(validation[columns])))
        idle_mae = float(mean_absolute_error(validation["hand_idle"], idle_candidate.predict(validation[columns])))

        moon_min_rows = int(context.config.get("moon_min_rows", 20))
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

        incumbent_error: float | None = None
        if context.active_checkpoint is not None:
            incumbent = joblib.load(context.active_checkpoint.inference_artifact_path)
            incumbent_core_mae = float(
                mean_absolute_error(
                    validation["target"], incumbent["core_model"].predict(validation[incumbent["feature_columns"]])
                )
            )
            incumbent_error = incumbent_core_mae / core_scale
        quality_passed = candidate_error <= 0.75 and productive_mae <= 0.08 and idle_mae <= 0.08
        promote = quality_passed and (incumbent_error is None or candidate_error <= incumbent_error)

        core_persisted = self._model(context.config)
        core_persisted.fit(frame[columns], frame["target"])
        productive_persisted = self._model({**context.config, "seed": int(context.config.get("seed", 42)) + 2})
        idle_persisted = self._model({**context.config, "seed": int(context.config.get("seed", 42)) + 3})
        productive_persisted.fit(frame[columns], frame["hand_productive"])
        idle_persisted.fit(frame[columns], frame["hand_idle"])
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
                "productive_model": productive_persisted,
                "idle_model": idle_persisted,
                "moon_model": moon_persisted,
                "feature_columns": columns,
                "moon_target_mean": moon_mean,
                "moon_target_std": moon_std or 1.0,
                "dataset_sha256": context.dataset_sha256,
                "quality": {"rows": len(frame), "core_error": candidate_error,
                            "productive_mae": productive_mae, "idle_mae": idle_mae},
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
            "anchor_opponents": int(frame["anchor_opponents"].max()),
            "core_mae": core_mae,
            "moon_mae": moon_mae,
            "candidate_weighted_error": candidate_error,
            "productive_mae": productive_mae,
            "idle_mae": idle_mae,
            "incumbent_weighted_error": incumbent_error,
            "validation_generation": newest,
            "feature_columns": columns,
            "feature_importances": dict(
                sorted(
                    zip(columns, core_persisted.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )[:20]
            ),
            "validation_behavior": {
                name: float(validation[name].mean())
                for name in (
                    "hire_orders",
                    "total_actions",
                    "productive_actions",
                    "movement_actions",
                    "plants_lost",
                    "hand_productive",
                    "hand_idle",
                    "reserve_violations",
                )
            },
        }
        gate = {
            "kind": "held_out_performance_and_labor_quality",
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
