"""`sim.fastenv` shadows two pieces of `kaggle_environments`. Pin both to it.

These are the tests that make a library upgrade loud: if `toJSON` grows a field
or `process_schema` changes what it rejects, the fast paths stop matching and
say so here rather than silently changing what a training run records.
"""

from __future__ import annotations

import copy

import pytest
from kaggle_environments import make

from sim import fastenv
from sim.runner import ENV_NAME, run_match

SPECS = ("agents/dqn_market_free/main.py", "starter")
STEPS = 120


def _played_env():
    from sim import registry

    fastenv.install()
    env = make(ENV_NAME, configuration={"episodeSteps": STEPS, "seed": 7}, debug=False)
    env.run([registry.resolve(spec) for spec in SPECS])
    return env


def test_to_json_matches_the_library():
    """The copy-free snapshot must be the same document the library builds."""
    env = _played_env()
    assert fastenv.to_json(env) == env.toJSON()


def test_to_json_is_a_live_view_not_a_copy():
    """States the point of the change: no deep copy of the retained steps."""
    env = _played_env()
    assert fastenv.to_json(env)["steps"] is env.steps


def test_fast_process_schema_agrees_with_the_library():
    """Same accept/reject verdict, and the same defaulted data, on both."""
    from kaggle_environments.utils import process_schema, structify

    schema = structify(
        {
            "type": "object",
            "properties": {
                "farmer": {"type": "array", "default": ["PASS"]},
                "hands": {"type": "array", "default": []},
                "count": {"type": "integer", "default": 0},
            },
        }
    )
    cases = [
        {},
        {"farmer": ["MOVE", 1, 2]},
        {"count": "not-an-integer"},
        {"hands": [["PASS"]], "count": 3},
    ]
    for case in cases:
        # The stock function is patched at this point; compare against the
        # original implementation rather than the name.
        expected_error, expected_data = _stock_process_schema(schema, copy.deepcopy(case))
        error, data = fastenv.fast_process_schema(schema, copy.deepcopy(case))
        assert (error is None) == (expected_error is None), case
        assert data == expected_data, case
    assert process_schema is not None


def _stock_process_schema(schema, data):
    import jsonschema
    from kaggle_environments.utils import default_schema

    data = default_schema(schema, copy.deepcopy(data))
    try:
        jsonschema.validate(data, schema)
    except Exception as err:  # noqa: BLE001
        return str(err), data
    return None, data


@pytest.mark.parametrize("seed", [4, 19])
def test_episodes_are_identical_with_the_fast_paths_disabled(seed, monkeypatch):
    """The whole point: same episode, same replay, either way."""
    fast = run_match(SPECS, seed=seed, config={"episodeSteps": STEPS})

    monkeypatch.setenv("KAGG_NO_FASTENV", "1")
    fastenv.uninstall()
    try:
        from kaggle_environments import core as kaggle_core

        assert kaggle_core.process_schema is not fastenv.fast_process_schema
        slow = run_match(SPECS, seed=seed, config={"episodeSteps": STEPS})
    finally:
        monkeypatch.delenv("KAGG_NO_FASTENV", raising=False)
        fastenv.install()

    assert fast.replay["steps"] == slow.replay["steps"]
    # Everything but `id`, which is a fresh uuid4 per environment either way.
    assert {k: v for k, v in fast.replay.items() if k != "id"} == {
        k: v for k, v in slow.replay.items() if k != "id"
    }
    assert [s.reward for s in fast.seats] == [s.reward for s in slow.seats]
    assert [s.final_money for s in fast.seats] == [s.final_money for s in slow.seats]
