import json
import random

import pytest

from agents.configurable_crop_surrogate._profile import load_strategy
from agents.configurable_crop_surrogate.strategy import Strategy, generate_population, sample_strategy


def test_strategy_round_trip_and_stable_id():
    strategy = sample_strategy(random.Random(7))
    restored = Strategy.from_dict(json.loads(strategy.canonical_json()))
    assert restored == strategy
    assert restored.strategy_id == strategy.strategy_id


def test_population_is_seeded_and_unique():
    first = generate_population(20, 99)
    second = generate_population(20, 99)
    assert [item.strategy_id for item in first] == [item.strategy_id for item in second]
    assert len({item.strategy_id for item in first}) == 20


def test_strategy_rejects_invalid_values_and_unknown_fields():
    with pytest.raises(ValueError, match="max_hands"):
        Strategy(max_hands=9)
    with pytest.raises(ValueError, match="unknown"):
        Strategy.from_dict({"not_a_parameter": 1})


@pytest.mark.parametrize("name", ["balanced", "aggressive", "conservative", "market_timer"])
def test_checked_in_profiles_are_valid(name):
    assert load_strategy(name).strategy_id

