"""Replay extraction, Random Forest fitting, rollout collection, and gating."""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from agents.mixed_farm_rf_market.policy import (
    MarketPolicyBundle,
    enumerate_purchase_candidates,
    market_state_features,
    purchase_feature_matrix,
    reserve_candidates,
    sale_action_features,
)
from agents.mixed_farm_rf_market.portfolio import build_plan
from agents.mixed_farm_rf_market.rules import (
    ANIMAL_DATA,
    ANIMALS,
    CROP_DATA,
    CROPS,
    PRODUCTS,
    count_assets,
)
from agents.mixed_farm_rf_market.strategy import Strategy
from kaggriculture_lab.model_updates import ModelEvaluation, ModelUpdateResult

FAMILY_ID = "mixed_farm_rf_market"
MODEL_ID = f"{FAMILY_ID}.market_policy"
SELF_SPEC = "agents.mixed_farm_rf_market.agent:make_agent"

# The field policy this experiment holds fixed while it learns the market. It is a
# committed default rather than another family's champion: the surrogate families
# come and go, and reaching into their sessions made this trainer unrunnable the
# moment one was retired. `agent.py` falls back to `heuristic_market` whenever it
# has no weights, so the bare spec below plays exactly this strategy with the
# hand-written market the forests are meant to replace -- which makes it both the
# warm-start data source and the honest promotion baseline.
FROZEN_STRATEGY = Strategy()
HEURISTIC_SPEC = SELF_SPEC


@dataclass(frozen=True, slots=True)
class DecisionDataset:
    purchase_x: np.ndarray
    purchase_y: np.ndarray
    purchase_group: np.ndarray
    sale_x: np.ndarray
    sale_y: np.ndarray
    sale_group: np.ndarray

    @property
    def episodes(self) -> int:
        groups = {*self.purchase_group.tolist(), *self.sale_group.tolist()}
        return len(groups)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            purchase_x=self.purchase_x,
            purchase_y=self.purchase_y,
            purchase_group=self.purchase_group,
            sale_x=self.sale_x,
            sale_y=self.sale_y,
            sale_group=self.sale_group,
        )

    @classmethod
    def load(cls, path: str | Path) -> DecisionDataset:
        with np.load(path, allow_pickle=False) as payload:
            return cls(*(np.asarray(payload[name]) for name in (
                "purchase_x", "purchase_y", "purchase_group",
                "sale_x", "sale_y", "sale_group",
            )))


def _empty_dataset() -> DecisionDataset:
    return DecisionDataset(
        np.empty((0, 0), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype="U1"),
        np.empty((0, 0), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype="U1"),
    )


def concatenate(datasets: Iterable[DecisionDataset]) -> DecisionDataset:
    items = [item for item in datasets if item.purchase_x.size or item.sale_x.size]
    if not items:
        return _empty_dataset()

    def rows(name: str) -> np.ndarray:
        values = [getattr(item, name) for item in items if getattr(item, name).size]
        return np.concatenate(values, axis=0) if values else np.empty((0, 0), dtype=np.float32)

    return DecisionDataset(
        rows("purchase_x"), rows("purchase_y"), rows("purchase_group"),
        rows("sale_x"), rows("sale_y"), rows("sale_group"),
    )


def split_by_episode(dataset: DecisionDataset, count: int) -> list[DecisionDataset]:
    """Partition complete episodes without splitting their decision rows."""

    groups = sorted({*dataset.purchase_group.tolist(), *dataset.sale_group.tolist()})
    count = max(1, min(count, len(groups)))
    buckets = [set(groups[index::count]) for index in range(count)]
    parts: list[DecisionDataset] = []
    for bucket in buckets:
        purchase_mask = np.isin(dataset.purchase_group, list(bucket))
        sale_mask = np.isin(dataset.sale_group, list(bucket))
        parts.append(
            DecisionDataset(
                dataset.purchase_x[purchase_mask],
                dataset.purchase_y[purchase_mask],
                dataset.purchase_group[purchase_mask],
                dataset.sale_x[sale_mask],
                dataset.sale_y[sale_mask],
                dataset.sale_group[sale_mask],
            )
        )
    return parts


def _seat_state(raw_step: Any, seat: int) -> dict[str, Any]:
    if not isinstance(raw_step, list) or seat >= len(raw_step):
        return {}
    state = raw_step[seat]
    return state if isinstance(state, dict) else {}


def _money_difference(obs: dict, seat: int) -> float:
    farms = obs.get("farms", []) or []
    if seat >= len(farms):
        return 0.0
    own = float(farms[seat].get("money", 0))
    other = max((float(farm.get("money", 0)) for index, farm in enumerate(farms) if index != seat), default=0.0)
    return own - other


def _daily_purchase_orders(steps: list[Any], seat: int) -> dict[int, list[list[Any]]]:
    by_day: dict[int, list[list[Any]]] = {}
    learned = {"BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "BUY_LAND"}
    previous_day: int | None = None
    for raw in steps:
        state = _seat_state(raw, seat)
        obs = state.get("observation") or {}
        # A replay step stores the state *after* its own action, so the day an order
        # was decided in is the previous step's. Bucketing by this step's day would
        # shift every end-of-day order into the following day.
        decided_day = previous_day if previous_day is not None else int(obs.get("day", 0))
        previous_day = int(obs.get("day", 0))
        for order in (state.get("action") or {}).get("market", []) or []:
            if order and order[0] in learned:
                by_day.setdefault(decided_day, []).append(list(order))
    return by_day


def _purchase_signature(orders: list[list[Any]]) -> tuple[dict[str, int], dict[str, int], int, int]:
    crops = {name: 0 for name in CROPS}
    animals = {name: 0 for name in ANIMALS}
    wheat = land = 0
    for order in orders:
        if order[0] == "BUY_SEED" and len(order) >= 3 and order[1] in crops:
            crops[str(order[1])] += max(0, int(order[2]))
        elif order[0] == "BUY_ANIMAL" and len(order) >= 3 and order[1] in animals:
            animals[str(order[1])] += max(0, int(order[2]))
        elif order[:2] == ["BUY_PRODUCT", "WHEAT"] and len(order) >= 3:
            wheat += max(0, int(order[2]))
        elif order[0] == "BUY_LAND":
            land += 1
    return crops, animals, wheat, land


def _infer_purchase_index(candidates, farm: dict, private: dict, orders: list[list[Any]]) -> int:
    actual_crops, actual_animals, actual_wheat, actual_land = _purchase_signature(orders)
    current_crops, current_animals, _ = count_assets(farm)
    for crop in CROPS:
        current_crops[crop] += int((private.get("seeds", {}) or {}).get(crop, 0))
    for animal in ANIMALS:
        current_animals[animal] += int((private.get("shed", {}) or {}).get(animal, 0))
        current_animals[animal] += sum(
            int((slot or {}).get(animal, 0)) for slot in private.get("inventories", []) or []
        )
    current_wheat = int((private.get("shed", {}) or {}).get("WHEAT", 0)) + sum(
        int((slot or {}).get("WHEAT", 0)) for slot in private.get("inventories", []) or []
    )
    current_land = max(1, len(farm.get("unlocked_quadrants", []) or ["NW"]))

    def distance(candidate) -> float:
        expected_crops = {
            name: max(0, candidate.crop_targets()[name] - current_crops[name]) for name in CROPS
        }
        expected_animals = {
            name: max(0, candidate.animal_targets()[name] - current_animals[name]) for name in ANIMALS
        }
        score = sum(
            abs(math.log1p(expected_crops[name]) - math.log1p(actual_crops[name])) for name in CROPS
        )
        score += sum(
            abs(math.log1p(expected_animals[name]) - math.log1p(actual_animals[name])) for name in ANIMALS
        )
        score += abs(math.log1p(max(0, candidate.wheat_reserve - current_wheat)) - math.log1p(actual_wheat))
        score += 4.0 * abs(int(candidate.target_land > current_land) - int(actual_land > 0))
        return score

    return min(range(len(candidates)), key=lambda index: (distance(candidates[index]), index))


def extract_replay_dataset(
    replay: dict[str, Any],
    seat: int,
    strategy: Strategy,
    group: str,
    *,
    episode_steps: int = 720,
    turns_per_day: int = 24,
) -> DecisionDataset:
    steps = replay.get("steps", []) or []
    if not steps:
        return _empty_dataset()
    final_obs = _seat_state(steps[-1], seat).get("observation") or {}
    final_margin = _money_difference(final_obs, seat)
    purchase_by_day = _daily_purchase_orders(steps, seat)
    seen_days: set[int] = set()
    purchase_x: list[np.ndarray] = []
    purchase_y: list[float] = []
    sale_x: list[np.ndarray] = []
    sale_y: list[float] = []

    previous_obs: dict[str, Any] | None = None

    for raw in steps:
        state = _seat_state(raw, seat)
        # A replay step stores the state *after* its own action was applied. The state
        # a decision was actually taken from is therefore the previous step's: pairing
        # an action with its own step's observation trains both forests to predict a
        # purchase from a bank that has already paid for it, and drops every sale that
        # emptied its shed slot because the post-sale shed reads as zero.
        obs = previous_obs
        previous_obs = state.get("observation") or {}
        if obs is None:
            continue
        farms = obs.get("farms", []) or []
        if seat >= len(farms):
            continue
        farm = farms[seat]
        private = obs.get("private", {}) or {}
        plan = build_plan(obs, farm, strategy, episode_steps=episode_steps, turns_per_day=turns_per_day)
        state_row = market_state_features(
            obs, farm, private, plan, episode_steps=episode_steps, turns_per_day=turns_per_day
        )
        target = final_margin - _money_difference(obs, seat)
        day = int(obs.get("day", 0))
        if day not in seen_days:
            candidates = enumerate_purchase_candidates(
                obs, farm, private, plan, strategy,
                episode_steps=episode_steps, turns_per_day=turns_per_day,
            )
            index = _infer_purchase_index(
                candidates, farm, private, purchase_by_day.get(day, [])
            )
            purchase_x.append(purchase_feature_matrix(state_row, candidates)[index])
            purchase_y.append(target)
            seen_days.add(day)

        action = state.get("action") or {}
        sold = {name: 0 for name in PRODUCTS}
        for order in action.get("market", []) or []:
            if order and order[0] == "SELL" and len(order) >= 3 and order[1] in sold:
                sold[str(order[1])] += max(0, int(order[2]))
        shed = private.get("shed", {}) or {}
        prices = (obs.get("market", {}) or {}).get("prices", {}) or {}
        survival = sum(count_assets(farm)[1].values())
        for item in PRODUCTS:
            available = max(0, int(shed.get(item, 0)))
            if available <= 0:
                continue
            # Preserve every actual sale but thin repetitive hold observations.
            # Otherwise a stocked shed contributes nine nearly identical rows on
            # every turn and overwhelms the scarce decisions to sell.
            step = int(obs.get("step", 0))
            if sold[item] <= 0 and step % 4:
                continue
            inferred = max(0, available - sold[item])
            levels = reserve_candidates(item, available, survival)
            reserve = min(levels, key=lambda value: (abs(value - inferred), value))
            base = 100.0
            if item in CROPS:
                base = float(CROP_DATA[item]["base"])
            else:
                base = next(
                    (float(data["base"]) for data in ANIMAL_DATA.values() if data["product"] == item),
                    100.0,
                )
            ratio = float(prices.get(item, base)) / max(1.0, base)
            sale_x.append(sale_action_features(state_row, item, available, reserve, ratio))
            sale_y.append(target)

    group_width = max(1, len(group))
    return DecisionDataset(
        np.asarray(purchase_x, dtype=np.float32),
        np.asarray(purchase_y, dtype=np.float32),
        np.asarray([group] * len(purchase_x), dtype=f"U{group_width}"),
        np.asarray(sale_x, dtype=np.float32),
        np.asarray(sale_y, dtype=np.float32),
        np.asarray([group] * len(sale_x), dtype=f"U{group_width}"),
    )


def _file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _spec(path: str | Path, *, epsilon: float = 0.0, seed: int = 0) -> str:
    return f"{SELF_SPEC}?weights={quote(str(path))}&epsilon={epsilon}&policy_seed={seed}"


def _collect_dataset(
    policy_for_seed: Callable[[int], str],
    strategy: Strategy,
    opponents: list[str],
    seeds: list[int],
    jobs: int,
    group_prefix: str,
) -> DecisionDataset:
    from sim.arena import play_series

    if not opponents:
        raise ValueError("at least one opponent is required to collect decisions")
    datasets: list[DecisionDataset] = []
    for index, seed in enumerate(seeds):
        opponent = opponents[index % len(opponents)]
        pair = play_series(policy_for_seed(seed), opponent, [seed], {}, jobs=jobs)[seed]
        forward, reverse = pair
        if not forward.failed and forward.replay is not None:
            datasets.append(
                extract_replay_dataset(forward.replay, 0, strategy, f"{group_prefix}:{seed}:0")
            )
        if not reverse.failed and reverse.replay is not None:
            datasets.append(
                extract_replay_dataset(reverse.replay, 1, strategy, f"{group_prefix}:{seed}:1")
            )
    return concatenate(datasets)


def bootstrap_dataset(opponents: list[str], seeds: list[int], jobs: int) -> DecisionDataset:
    """Warm start by watching the frozen heuristic play.

    The agent falls back to `heuristic_market` whenever it has no weights, so the
    bare spec is exactly the policy the forests are meant to improve on. Collecting
    from it keeps the experiment self-contained -- no other family's session has to
    exist, or still exist, before this one can run.
    """

    dataset = _collect_dataset(
        lambda _: HEURISTIC_SPEC, FROZEN_STRATEGY, opponents, seeds, jobs, "bootstrap"
    )
    if not dataset.purchase_x.size and not dataset.sale_x.size:
        raise ValueError("heuristic bootstrap produced no successful replays")
    return dataset


def rollout_dataset(
    bundle_path: str | Path,
    opponents: list[str],
    seeds: list[int],
    epsilon: float,
    jobs: int,
) -> DecisionDataset:
    bundle = MarketPolicyBundle.load(bundle_path)
    return _collect_dataset(
        lambda seed: _spec(bundle_path, epsilon=epsilon, seed=seed),
        bundle.strategy,
        opponents,
        seeds,
        jobs,
        "rollout",
    )


def fit_forests(dataset: DecisionDataset, config: dict[str, Any]):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupShuffleSplit

    def fit_one(
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        seed: int,
        maximum_rows: int,
    ):
        if len(x) > maximum_rows:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(len(x), size=maximum_rows, replace=False))
            x, y, groups = x[selected], y[selected], groups[selected]
        unique = np.unique(groups)
        if len(x) < 10 or len(unique) < 2:
            raise ValueError("each market forest requires at least ten rows from two episodes")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_index, validation_index = next(splitter.split(x, y, groups))
        model = RandomForestRegressor(
            n_estimators=int(config.get("n_estimators", 300)),
            criterion=str(config.get("criterion", "squared_error")),
            max_depth=config.get("max_depth"),
            min_samples_leaf=int(config.get("min_samples_leaf", 5)),
            max_features=float(config.get("max_features", 0.7)),
            bootstrap=bool(config.get("bootstrap", True)),
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(x[train_index], y[train_index])
        validation_predictions = model.predict(x[validation_index])
        validation_mae = float(mean_absolute_error(y[validation_index], validation_predictions))
        validation_scale = float(np.std(y[validation_index])) or 1.0
        tree_predictions = np.asarray(
            [tree.predict(x[validation_index]) for tree in model.estimators_], dtype=float
        )
        persisted = RandomForestRegressor(
            n_estimators=int(config.get("n_estimators", 300)),
            criterion=str(config.get("criterion", "squared_error")),
            max_depth=config.get("max_depth"),
            min_samples_leaf=int(config.get("min_samples_leaf", 5)),
            max_features=float(config.get("max_features", 0.7)),
            bootstrap=bool(config.get("bootstrap", True)),
            n_jobs=-1,
            random_state=seed,
        )
        persisted.fit(x, y)
        persisted.n_jobs = 1
        return persisted, {
            "rows": len(x),
            "episodes": len(unique),
            "validation_rows": len(validation_index),
            "validation_mae": validation_mae,
            "validation_normalized_mae": validation_mae / validation_scale,
            "validation_tree_disagreement": float(np.mean(np.std(tree_predictions, axis=0))),
            "top_feature_importances": {
                str(index): float(persisted.feature_importances_[index])
                for index in np.argsort(-persisted.feature_importances_)[:20]
            },
        }

    seed = int(config.get("seed", 42))
    purchase, purchase_metrics = fit_one(
        dataset.purchase_x,
        dataset.purchase_y,
        dataset.purchase_group,
        seed,
        int(config.get("max_purchase_rows", 50_000)),
    )
    sale, sale_metrics = fit_one(
        dataset.sale_x,
        dataset.sale_y,
        dataset.sale_group,
        seed + 1,
        int(config.get("max_sale_rows", 250_000)),
    )
    return purchase, sale, {"purchase": purchase_metrics, "sale": sale_metrics}


def _paired_margins(spec: str, opponent: str, seeds: range, jobs: int) -> list[float]:
    from sim.arena import play_series

    results = play_series(spec, opponent, seeds, {}, jobs)
    values = []
    for forward, reverse in results.values():
        if forward.failed or reverse.failed or forward.margin is None or reverse.margin is None:
            continue
        values.append((float(forward.margin) - float(reverse.margin)) / 2.0)
    return values


def paired_gate(
    candidate_path: str | Path,
    baseline_spec: str,
    incumbent_path: str | Path | None,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    opponents = [str(value) for value in config.get("promotion_opponents", [])]
    count = int(config.get("promotion_seeds", 10))
    if not opponents or count <= 0:
        return True, {"kind": "disabled", "passed": True}
    jobs = max(1, int(config.get("jobs", 1)))
    seed_base = int(config.get("promotion_seed_base", 1_000_000))
    seeds = range(seed_base, seed_base + count)
    candidate_spec = _spec(candidate_path)
    references = {"baseline": baseline_spec}
    if incumbent_path is not None:
        references["incumbent"] = _spec(incumbent_path)
    report: dict[str, Any] = {"kind": "paired_broad_market_gate", "opponents": {}}
    differences: dict[str, list[float]] = {name: [] for name in references}
    moon_differences: list[float] = []
    for opponent in opponents:
        candidate = _paired_margins(candidate_spec, opponent, seeds, jobs)
        row: dict[str, Any] = {"candidate_margin": statistics.fmean(candidate) if candidate else None}
        for name, reference_spec in references.items():
            reference = _paired_margins(reference_spec, opponent, seeds, jobs)
            paired = [left - right for left, right in zip(candidate, reference)]
            mean = statistics.fmean(paired) if paired else float("-inf")
            row[f"vs_{name}"] = mean
            differences[name].append(mean)
            if "prvsiyan" in opponent and name == "baseline":
                moon_differences = paired
        report["opponents"][opponent] = row

    baseline_values = differences["baseline"]
    macro = statistics.fmean(baseline_values) if baseline_values else float("-inf")
    macro_low = macro
    if len(baseline_values) > 1:
        macro_low -= 1.96 * statistics.stdev(baseline_values) / math.sqrt(len(baseline_values))
    moon_low = float("-inf")
    if moon_differences:
        moon_low = statistics.fmean(moon_differences)
        if len(moon_differences) > 1:
            moon_low -= 1.96 * statistics.stdev(moon_differences) / math.sqrt(len(moon_differences))
    worst = min(baseline_values, default=float("-inf"))
    incumbent_macro = (
        statistics.fmean(differences["incumbent"])
        if "incumbent" in differences and differences["incumbent"]
        else 0.0
    )
    passed = (
        macro > 0.0
        and macro_low > -5_000.0
        and moon_low > -5_000.0
        and worst >= -15_000.0
        and incumbent_macro >= 0.0
    )
    report.update(
        {
            "passed": passed,
            "macro_improvement": macro,
            "macro_ci95_low": macro_low,
            "moon_ci95_low": moon_low,
            "worst_opponent_improvement": worst,
            "incumbent_improvement": incumbent_macro,
        }
    )
    return passed, report


class RFMarketTrainer:
    def on_round(self, context, reader, writer) -> None:
        if context.scope != "round":
            return
        config = context.config
        batch_index = int(reader.fetchone(
            "SELECT count(*) FROM training_batches WHERE model_id = %s", [context.model_id]
        )[0])
        directory = Path(context.artifacts) / "training_batches"
        source_id = f"market-{batch_index:08d}"
        path = directory / f"{source_id}.npz"
        opponents = [str(value) for value in config.get("opponents", ["starter"])]
        if batch_index == 0:
            count = int(config.get("bootstrap_episodes", 40))
            seed_base = int(config.get("seed", 42))
            dataset = bootstrap_dataset(
                opponents,
                list(range(seed_base, seed_base + count)),
                max(1, int(config.get("jobs", 1))),
            )
            parts = split_by_episode(
                dataset, int(config.get("bootstrap_batches", 5))
            )
            for offset, part in enumerate(parts):
                part_source_id = f"market-{offset:08d}"
                part_path = directory / f"{part_source_id}.npz"
                part.save(part_path)
                writer.record(
                    source_kind="rf_market_bootstrap",
                    source_id=part_source_id,
                    sample_count=part.episodes,
                    artifact_path=part_path,
                    artifact_sha256=_file_digest(part_path),
                    metadata={
                        "episodes": part.episodes,
                        "purchase_rows": len(part.purchase_y),
                        "sale_rows": len(part.sale_y),
                        "epsilon": None,
                        "session_id": context.session_id,
                        "generation_id": context.generation_id,
                        "round_index": context.round_index,
                    },
                )
            return
        else:
            checkpoint_path = (
                context.active_checkpoint.inference_artifact_path
                if context.active_checkpoint is not None
                else None
            )
            if checkpoint_path is None:
                row = reader.fetchone(
                    """
                    SELECT inference_artifact_path FROM model_training_checkpoints
                    WHERE model_id = %s ORDER BY created_at DESC LIMIT 1
                    """,
                    [context.model_id],
                )
                checkpoint_path = row[0] if row else None
            if checkpoint_path is None:
                raise ValueError("bootstrap model must be trained before rollout collection")
            decay = max(1, int(config.get("epsilon_decay_batches", 100)))
            fraction = min(1.0, (batch_index - 1) / decay)
            epsilon = float(config.get("epsilon_start", 0.30)) + fraction * (
                float(config.get("epsilon_end", 0.05)) - float(config.get("epsilon_start", 0.30))
            )
            seed = int(config.get("seed", 42)) + batch_index * 1009
            count = int(config.get("episodes_per_batch", 8))
            seeds = list(range(seed, seed + count))
            dataset = rollout_dataset(
                checkpoint_path, opponents, seeds, epsilon, max(1, int(config.get("jobs", 1)))
            )
            source_kind = "rf_market_rollout"
        dataset.save(path)
        writer.record(
            source_kind=source_kind,
            source_id=source_id,
            sample_count=dataset.episodes,
            artifact_path=path,
            artifact_sha256=_file_digest(path),
            metadata={
                "episodes": dataset.episodes,
                "purchase_rows": len(dataset.purchase_y),
                "sale_rows": len(dataset.sale_y),
                "epsilon": epsilon,
                "session_id": context.session_id,
                "generation_id": context.generation_id,
                "round_index": context.round_index,
            },
        )

    def on_batch(self, context, reader) -> ModelUpdateResult:
        batches = [batch for batch in context.all_batches if batch.artifact_path]
        max_batches = int(context.config.get("max_training_batches", 200))
        datasets = [DecisionDataset.load(batch.artifact_path) for batch in batches[-max_batches:]]
        dataset = concatenate(datasets)
        purchase, sale, fit_metrics = fit_forests(dataset, context.config)
        strategy, baseline_spec = FROZEN_STRATEGY, HEURISTIC_SPEC
        checkpoint_id = hashlib.sha256(
            f"{context.model_id}\0{context.update_id}\0{context.dataset_sha256}".encode()
        ).hexdigest()[:24]
        artifact = Path(context.artifacts) / "checkpoints" / f"{checkpoint_id}.joblib"
        bundle = MarketPolicyBundle(strategy, purchase, sale, context.dataset_sha256)
        bundle.dump(artifact)
        incumbent_path = (
            context.active_checkpoint.inference_artifact_path
            if context.active_checkpoint is not None else None
        )
        promote, gate = paired_gate(
            artifact, baseline_spec, incumbent_path, context.config
        )
        digest = _file_digest(artifact)
        metrics = {
            **fit_metrics,
            "episodes": dataset.episodes,
            "purchase_rows": len(dataset.purchase_y),
            "sale_rows": len(dataset.sale_y),
            "frozen_strategy_id": strategy.strategy_id,
            "dataset_sha256": context.dataset_sha256,
            "gate": gate,
        }
        evaluation = ModelEvaluation(
            "paired_broad_market_gate",
            gate,
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


trainer = RFMarketTrainer()
