"""Elo, Random Forest surrogates, and next-generation candidate proposals."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from agents.configurable_crop_surrogate.experiment import ExperimentConfig
from agents.configurable_crop_surrogate.strategy import Strategy, sample_strategy
from kaggriculture_lab.storage import ExperimentStore


def calculate_elo(store: ExperimentStore, run_id: str, k_factor: float = 24.0) -> dict[str, tuple[float, int]]:
    rows = store.connection.execute(
        """
        SELECT stage, seed, match_id, seat0_id, seat1_id, p.money_difference
        FROM episodes e
        JOIN episode_players p USING (match_id, run_id, stage, seed)
        WHERE e.run_id = %s AND p.seat = 0 AND NOT e.failed
        ORDER BY CASE stage WHEN 'stage1' THEN 1 WHEN 'stage2' THEN 2 ELSE 3 END,
                 seed, match_id
        """,
        [run_id],
    ).fetchall()
    ratings: dict[str, float] = {}
    games: dict[str, int] = {}
    for _, _, _, left, right, difference in rows:
        ratings.setdefault(left, 1500.0)
        ratings.setdefault(right, 1500.0)
        games[left] = games.get(left, 0) + 1
        games[right] = games.get(right, 0) + 1
        actual = 1.0 if difference > 0 else 0.0 if difference < 0 else 0.5
        expected = 1.0 / (1.0 + 10 ** ((ratings[right] - ratings[left]) / 400.0))
        change = k_factor * (actual - expected)
        ratings[left] += change
        ratings[right] -= change
    result = {key: (rating, games[key]) for key, rating in ratings.items()}
    store.save_ratings(run_id, result)
    return result


def _strategy_frame(strategies: list[Strategy], prefix: str = "") -> pd.DataFrame:
    return pd.DataFrame([strategy.to_features(prefix) for strategy in strategies])


def _pipeline(frame: pd.DataFrame, seed: int) -> Pipeline:
    categorical = [name for name in frame.columns if not is_numeric_dtype(frame[name])]
    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )
    forest = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([("features", preprocessor), ("model", forest)])


def _evaluate(
    frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 4:
        raise ValueError("at least four independent strategy groups are required to train a surrogate")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_index, test_index = next(splitter.split(frame, target, groups))
    train_x, test_x = frame.iloc[train_index], frame.iloc[test_index]
    train_y, test_y = target[train_index], target[test_index]

    model = _pipeline(frame, seed)
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    dummy = DummyRegressor(strategy="mean").fit(train_x, train_y)
    dummy_predicted = dummy.predict(test_x)
    permutation = permutation_importance(
        model, test_x, test_y, n_repeats=10, random_state=seed, n_jobs=-1
    )
    importances = sorted(
        (
            {"feature": feature, "importance": float(value)}
            for feature, value in zip(frame.columns, permutation.importances_mean, strict=True)
        ),
        key=lambda row: row["importance"],
        reverse=True,
    )
    report = {
        "rows": len(frame),
        "groups": len(unique_groups),
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "mae": float(mean_absolute_error(test_y, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(test_y, predicted))),
        "r2": float(r2_score(test_y, predicted)) if len(test_y) > 1 else 0.0,
        "rank_correlation": float(pd.Series(test_y).rank().corr(pd.Series(predicted).rank())),
        "dummy_mae": float(mean_absolute_error(test_y, dummy_predicted)),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)) or 1.0,
        "permutation_importance": importances,
    }
    # Refit the persisted model on every available row after honest held-out evaluation.
    model.fit(frame, target)
    return model, report


def _quality_dataset(store: ExperimentStore, run_id: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = store.connection.execute(
        """
        SELECT p.strategy_id, p.money_difference, s.params_json
        FROM episode_players p
        JOIN strategies s USING (strategy_id)
        WHERE p.run_id = %s AND p.stage = 'stage1' AND NOT p.failed
          AND p.strategy_id NOT LIKE 'builtin:%'
        ORDER BY p.strategy_id, p.match_id
        """,
        [run_id],
    ).fetchall()
    strategies = [Strategy.from_dict(json.loads(row[2])) for row in rows]
    return (
        _strategy_frame(strategies),
        np.asarray([float(row[1]) for row in rows]),
        np.asarray([row[0] for row in rows]),
    )


def _matchup_dataset(store: ExperimentStore, run_id: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    rows = store.connection.execute(
        """
        SELECT p.strategy_id, p.opponent_id, p.money_difference,
               mine.params_json, theirs.params_json, p.match_id
        FROM episode_players p
        JOIN strategies mine ON mine.strategy_id = p.strategy_id
        JOIN strategies theirs ON theirs.strategy_id = p.opponent_id
        WHERE p.run_id = %s AND p.stage IN ('stage2', 'stage3') AND NOT p.failed
        ORDER BY p.match_id, p.seat
        """,
        [run_id],
    ).fetchall()
    if len(rows) < 20:
        return None
    mine = [Strategy.from_dict(json.loads(row[3])) for row in rows]
    theirs = [Strategy.from_dict(json.loads(row[4])) for row in rows]
    frame = pd.concat(
        [_strategy_frame(mine, "mine_"), _strategy_frame(theirs, "opponent_")], axis=1
    )
    groups = np.asarray(["::".join(sorted((row[0], row[1]))) for row in rows])
    return frame, np.asarray([float(row[2]) for row in rows]), groups


def train_surrogates(
    store: ExperimentStore,
    run_id: str,
    artifacts: str | Path = "artifacts",
) -> dict[str, dict[str, Any]]:
    run_row = store.connection.execute(
        "SELECT seed FROM runs WHERE run_id = %s", [run_id]
    ).fetchone()
    if run_row is None:
        raise ValueError(f"unknown run_id: {run_id}")
    seed = int(run_row[0])
    directory = Path(artifacts) / "models" / run_id
    directory.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict[str, Any]] = {}
    quality_x, quality_y, quality_groups = _quality_dataset(store, run_id)
    quality, quality_report = _evaluate(quality_x, quality_y, quality_groups, seed)
    quality_path = directory / "quality.joblib"
    joblib.dump({"model": quality, "report": quality_report, "columns": list(quality_x)}, quality_path)
    quality_id = hashlib.sha256(f"{run_id}:quality".encode()).hexdigest()[:20]
    store.save_model(quality_id, run_id, "quality", str(quality_path), quality_report)
    reports["quality"] = quality_report

    matchup_data = _matchup_dataset(store, run_id)
    if matchup_data is not None and len(np.unique(matchup_data[2])) >= 4:
        matchup_x, matchup_y, matchup_groups = matchup_data
        matchup, matchup_report = _evaluate(matchup_x, matchup_y, matchup_groups, seed + 1)
        matchup_path = directory / "matchup.joblib"
        joblib.dump(
            {"model": matchup, "report": matchup_report, "columns": list(matchup_x)}, matchup_path
        )
        matchup_id = hashlib.sha256(f"{run_id}:matchup".encode()).hexdigest()[:20]
        store.save_model(matchup_id, run_id, "matchup", str(matchup_path), matchup_report)
        reports["matchup"] = matchup_report

    (directory / "report.json").write_text(json.dumps(reports, indent=2, sort_keys=True))
    return reports


def _forest_predictions(bundle: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pipeline: Pipeline = bundle["model"]
    transformed = pipeline.named_steps["features"].transform(frame[bundle["columns"]])
    forest: RandomForestRegressor = pipeline.named_steps["model"]
    predictions = np.vstack([tree.predict(transformed) for tree in forest.estimators_])
    return predictions.mean(axis=0), predictions.std(axis=0)


def _diversity_key(strategy: Strategy) -> tuple[Any, ...]:
    values = strategy.to_dict()
    keys: list[Any] = []
    ranges = {
        "max_hands": (0, 8), "hire_until_hour": (0, 6), "cash_reserve": (0, 1500),
        "target_planted_tiles": (4, 25), "water_priority": (0.5, 3),
        "emergency_water_multiplier": (2, 12), "sell_price_ratio": (0.7, 1.5),
        "shed_pressure_fraction": (0.25, 0.9), "movement_cost_weight": (0, 2),
        "town_demand_weight": (0, 2), "opponent_supply_weight": (0, 2),
        "endgame_liquidation_turns": (12, 72),
    }
    for crop in ("wheat", "carrot", "tomato", "strawberry", "melon"):
        ranges[f"{crop}_weight"] = (0, 2)
    for name, value in values.items():
        if name in ranges:
            low, high = ranges[name]
            keys.append(min(9, int(10 * (float(value) - low) / (high - low))))
        else:
            keys.append(value)
    return tuple(keys)


def propose_strategies(
    store: ExperimentStore,
    run_id: str,
    artifacts: str | Path = "artifacts",
    candidate_pool: int | None = None,
    proposal_count: int | None = None,
) -> list[Strategy]:
    run_row = store.connection.execute(
        "SELECT seed, config_json FROM runs WHERE run_id = %s", [run_id]
    ).fetchone()
    if run_row is None:
        raise ValueError(f"unknown run_id: {run_id}")
    config = ExperimentConfig(**json.loads(run_row[1]))
    candidate_pool = candidate_pool or config.candidate_pool
    proposal_count = proposal_count or config.proposal_count
    rng = random.Random(int(run_row[0]) + 900_000)
    existing = set(store.load_strategies())
    candidates: dict[str, Strategy] = {}
    while len(candidates) < candidate_pool:
        strategy = sample_strategy(rng)
        if strategy.strategy_id not in existing:
            candidates[strategy.strategy_id] = strategy
    strategies = list(candidates.values())
    frame = _strategy_frame(strategies)

    model_rows = store.connection.execute(
        "SELECT kind, artifact_path FROM models WHERE run_id = %s ORDER BY created_at DESC",
        [run_id],
    ).fetchall()
    paths = {kind: path for kind, path in model_rows}
    if "quality" not in paths:
        raise ValueError("quality model not found; run analysis first")
    quality_bundle = joblib.load(paths["quality"])
    quality_mean, quality_std = _forest_predictions(quality_bundle, frame)
    quality_report = quality_bundle["report"]
    quality_z = (quality_mean - quality_report["target_mean"]) / quality_report["target_std"]

    matchup_mean = np.full(len(strategies), np.nan)
    if "matchup" in paths:
        matchup_bundle = joblib.load(paths["matchup"])
        elites_rows = store.connection.execute(
            """
            SELECT r.strategy_id FROM ratings r JOIN strategies s USING (strategy_id)
            WHERE r.run_id = %s ORDER BY r.rating DESC LIMIT 20
            """,
            [run_id],
        ).fetchall()
        # The store hands back raw parameter dicts; this family decodes its own.
        elites = [
            Strategy.from_dict(params)
            for params in store.load_strategies(row[0] for row in elites_rows).values()
        ]
        if elites:
            totals = np.zeros(len(strategies))
            for elite in elites:
                opponent = pd.DataFrame([elite.to_features("opponent_")] * len(strategies))
                mine = _strategy_frame(strategies, "mine_")
                pair_frame = pd.concat([mine, opponent], axis=1)
                totals += matchup_bundle["model"].predict(pair_frame[matchup_bundle["columns"]])
            matchup_mean = totals / len(elites)
            matchup_report = matchup_bundle["report"]
            matchup_z = (matchup_mean - matchup_report["target_mean"]) / matchup_report["target_std"]
            combined = 0.4 * quality_z + 0.6 * matchup_z
        else:
            combined = quality_z
    else:
        combined = quality_z

    exploit_count = max(1, round(proposal_count * 0.8))
    explore_count = max(0, proposal_count - exploit_count)
    selected: list[tuple[int, str]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for index in np.argsort(-combined):
        key = _diversity_key(strategies[int(index)])
        if key in seen_keys:
            continue
        selected.append((int(index), "exploit"))
        seen_keys.add(key)
        if len(selected) >= exploit_count:
            break
    threshold = float(np.median(combined))
    for index in np.argsort(-quality_std):
        if combined[int(index)] < threshold:
            continue
        key = _diversity_key(strategies[int(index)])
        if key in seen_keys:
            continue
        selected.append((int(index), "explore"))
        seen_keys.add(key)
        if len(selected) >= exploit_count + explore_count:
            break

    proposed = [strategies[index] for index, _ in selected]
    store.add_strategies(proposed, "proposed", run_id)
    rows = []
    for rank, (index, acquisition) in enumerate(selected, 1):
        rows.append(
            {
                "strategy_id": strategies[index].strategy_id,
                "rank": rank,
                "acquisition": acquisition,
                "predicted_quality": float(quality_mean[index]),
                "predicted_matchup": None if np.isnan(matchup_mean[index]) else float(matchup_mean[index]),
                "uncertainty": float(quality_std[index]),
            }
        )
    store.save_proposals(run_id, rows)
    directory = Path(artifacts) / "proposals" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "strategies.json").write_text(
        json.dumps(
            [{**row, "strategy": strategy.to_dict()} for row, strategy in zip(rows, proposed, strict=True)],
            indent=2,
            sort_keys=True,
        )
    )
    return proposed


def analyze_run(
    store: ExperimentStore, run_id: str, artifacts: str | Path = "artifacts"
) -> dict[str, dict[str, Any]]:
    calculate_elo(store, run_id)
    return train_surrogates(store, run_id, artifacts)
