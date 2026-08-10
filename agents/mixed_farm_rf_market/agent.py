"""Mixed-farm field agent whose market targets come from Random Forests."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agents.mixed_farm_rf_market.heuristic_market import market_orders as heuristic_market_orders
from agents.mixed_farm_rf_market.market import learned_market_orders
from agents.mixed_farm_rf_market.policy import (
    MarketPolicyBundle,
    PurchasePlan,
    choose_purchase,
    choose_reserves,
    enumerate_purchase_candidates,
    market_state_features,
)
from agents.mixed_farm_rf_market.portfolio import build_plan
from agents.mixed_farm_rf_market.rules import ANIMALS, count_assets, get
from agents.mixed_farm_rf_market.scheduler import assign_actions, enumerate_tasks
from agents.mixed_farm_rf_market.strategy import Strategy


@dataclass(slots=True)
class EpisodeState:
    player: int | None = None
    last_step: int = -1
    action_counts: dict[str, int] = field(default_factory=dict)
    commitments: dict[int, tuple[Any, ...]] = field(default_factory=dict)

    def reset(self, player: int) -> None:
        self.player = player
        self.last_step = -1
        self.action_counts.clear()
        self.commitments.clear()


@dataclass(slots=True)
class DecisionTrace:
    purchase_rows: list[Any] = field(default_factory=list)
    sale_rows: list[Any] = field(default_factory=list)


class MixedFarmRFMarketAgent:
    def __init__(
        self,
        bundle: MarketPolicyBundle,
        *,
        epsilon: float = 0.0,
        policy_seed: int = 0,
        episode_steps: int = 720,
        turns_per_day: int = 24,
        recorder: Any | None = None,
    ) -> None:
        self.bundle = bundle
        self.strategy = bundle.strategy
        self.epsilon = max(0.0, min(1.0, float(epsilon)))
        self.rng = random.Random(policy_seed)
        self.episode_steps = episode_steps
        self.turns_per_day = turns_per_day
        self.recorder = recorder
        self.state = EpisodeState()
        self._day = -1
        self._purchase: PurchasePlan | None = None

    def __call__(self, obs: Any, configuration: Any | None = None) -> dict[str, list]:
        farms = get(obs, "farms", []) or []
        player = int(get(obs, "player", 0))
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        farm = farms[player]
        private = get(obs, "private", {}) or {}
        step = int(get(obs, "step", 0))
        day = int(get(obs, "day", 0))
        if step == 0 or self.state.player != player or step <= self.state.last_step:
            self.state.reset(player)
            self._day = -1
            self._purchase = None
            if self.recorder is not None and hasattr(self.recorder, "reset"):
                self.recorder.reset()

        self.episode_steps = int(get(configuration, "episodeSteps", self.episode_steps))
        self.turns_per_day = int(get(configuration, "turnsPerDay", self.turns_per_day))
        hand_multiplier = int(get(configuration, "farmHandCostMult", 1))
        plan = build_plan(
            obs,
            farm,
            self.strategy,
            episode_steps=self.episode_steps,
            turns_per_day=self.turns_per_day,
        )
        state_row = market_state_features(
            obs,
            farm,
            private,
            plan,
            episode_steps=self.episode_steps,
            turns_per_day=self.turns_per_day,
        )

        if self._purchase is None or day != self._day:
            candidates = enumerate_purchase_candidates(
                obs,
                farm,
                private,
                plan,
                self.strategy,
                episode_steps=self.episode_steps,
                turns_per_day=self.turns_per_day,
            )
            # Candidate zero is the hold plan.  Prefer the candidate closest to
            # the heuristic target before a first checkpoint exists.
            fallback = min(
                range(len(candidates)),
                key=lambda index: sum(
                    abs(candidates[index].crop_targets()[name] - plan.target_crops[name]) for name in plan.target_crops
                ) + sum(
                    abs(candidates[index].animal_targets()[name] - plan.target_animals[name]) for name in plan.target_animals
                ) + abs(candidates[index].wheat_reserve - plan.wheat_reserve),
            )
            self._purchase, purchase_index, purchase_row = choose_purchase(
                state_row,
                candidates,
                self.bundle.purchase_model,
                epsilon=self.epsilon,
                rng=self.rng,
                fallback=fallback,
            )
            self._day = day
            if self.recorder is not None and hasattr(self.recorder, "record_purchase"):
                self.recorder.record_purchase(step, purchase_index, purchase_row)

        prices = (get(obs, "market", {}) or {}).get("prices", {}) or {}
        _, current_animals, _ = count_assets(farm)
        owned_animal_total = sum(current_animals.values())
        shed = private.get("shed", {}) or {}
        inventories = private.get("inventories", []) or []
        owned_animal_total += sum(
            int(shed.get(animal, 0))
            + sum(int((slot or {}).get(animal, 0)) for slot in inventories)
            for animal in ANIMALS
        )
        reserves, sale_rows = choose_reserves(
            state_row,
            shed,
            prices,
            self.bundle.sale_model,
            survival_wheat=owned_animal_total,
            epsilon=self.epsilon,
            rng=self.rng,
        )
        if self.recorder is not None and hasattr(self.recorder, "record_sales"):
            self.recorder.record_sales(step, sale_rows)

        if self.bundle.purchase_model is None and self.bundle.sale_model is None and self.epsilon == 0.0:
            orders = heuristic_market_orders(
                obs,
                farm,
                private,
                plan,
                self.strategy,
                episode_steps=self.episode_steps,
                turns_per_day=self.turns_per_day,
                farm_hand_cost_mult=hand_multiplier,
            )
        else:
            orders = learned_market_orders(
                obs,
                farm,
                private,
                plan,
                self._purchase,
                reserves,
                self.strategy,
                episode_steps=self.episode_steps,
                turns_per_day=self.turns_per_day,
                farm_hand_cost_mult=hand_multiplier,
            )

        positions = [tuple(farm["farmer"]), *(tuple(position) for position in farm.get("hands", []) or [])]
        inventories = list(inventories)
        while len(inventories) < len(positions):
            inventories.append({})
        tasks = enumerate_tasks(
            obs,
            farm,
            private,
            plan,
            positions,
            inventories,
            self.strategy,
            episode_steps=self.episode_steps,
            turns_per_day=self.turns_per_day,
        )
        actions, self.state.commitments = assign_actions(
            tasks, positions, inventories, step, self.strategy, self.state.commitments
        )
        self.state.last_step = step
        return {"farmer": actions[0], "hands": actions[1:], "market": orders[:10]}


def make_agent(config: dict[str, Any] | None = None) -> Callable[[Any], dict]:
    values = dict(config or {})
    weights = values.pop("weights", None)
    epsilon = float(values.pop("epsilon", 0.0))
    policy_seed = int(values.pop("policy_seed", 0))
    episode_steps = int(values.pop("episode_steps", 720))
    turns_per_day = int(values.pop("turns_per_day", 24))
    recorder = values.pop("recorder", None)
    if weights:
        bundle = MarketPolicyBundle.load(str(weights))
    else:
        strategy = Strategy.from_dict(values) if values else Strategy()
        bundle = MarketPolicyBundle(strategy, None, None)
    return MixedFarmRFMarketAgent(
        bundle,
        epsilon=epsilon,
        policy_seed=policy_seed,
        episode_steps=episode_steps,
        turns_per_day=turns_per_day,
        recorder=recorder,
    )
