"""Two costs `kaggle_environments` pays per step that buy this repo nothing.

Neither changes what an episode does -- `tests/test_fastenv.py` plays episodes
both ways and compares the replays -- and both are large. Measured on one
720-step self-play episode under cProfile:

| cost | per episode |
| --- | --- |
| kaggriculture's own game logic | 0.08 s |
| re-validating the action *schema* every step | 0.81 s |
| `toJSON` deep-copying all 720 steps | 1.01 s |

1. **`process_schema`** calls `jsonschema.validate(data, schema)`, which
   re-runs `cls.check_schema(schema)` on every call. The schema is a constant;
   only the data changes. jsonschema's own docstring says to build a validator
   once when validating many instances against one schema, and the profile
   agrees -- `check_schema` was 0.807 s of `validate`'s 0.845 s, so 95% of
   action validation was validating the schema rather than the action. The
   replacement validates identically, against the same validator class
   `jsonschema.validate` would have chosen.

2. **`toJSON`** deep-copies the entire environment, all 720 retained steps
   included, to hand back a snapshot that cannot be disturbed by later stepping.
   `sim.runner.run_match` calls it once, at the end, and drops the environment
   immediately afterwards, so there is nothing left to disturb it.

Set `KAGG_NO_FASTENV=1` to run the stock library instead -- the flag is the
first thing to try if an episode ever looks wrong.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import jsonschema
from kaggle_environments import core as kaggle_core
from kaggle_environments import utils as kaggle_utils

#: `id(schema) -> (schema, validator)`. Keyed on identity because a schema is an
#: unhashable dict, and sound because the tuple holds the schema itself: a live
#: entry pins its key's object, so that id cannot be recycled underneath us.
_VALIDATORS: dict[int, tuple[Any, Any]] = {}

#: An environment builds its own schema objects, so a process playing episodes
#: back to back would otherwise accumulate one set per episode. Cleared whole
#: rather than evicted piecewise, which keeps the identity argument above intact.
_VALIDATOR_CAP = 64

_installed = False

#: The library's own `process_schema`, captured before anything is swapped, so
#: `uninstall` restores the real thing rather than whatever was bound last.
_STOCK_PROCESS_SCHEMA = kaggle_utils.process_schema


def disabled() -> bool:
    return os.environ.get("KAGG_NO_FASTENV", "") not in ("", "0")


def _validator_for(schema: Any) -> Any:
    key = id(schema)
    entry = _VALIDATORS.get(key)
    if entry is not None:
        return entry[1]
    if len(_VALIDATORS) >= _VALIDATOR_CAP:
        _VALIDATORS.clear()
    validator = jsonschema.validators.validator_for(schema)(schema)
    _VALIDATORS[key] = (schema, validator)
    return validator


def fast_process_schema(
    schema: Any, data: Any, use_default: bool = True
) -> tuple[str | None, Any]:
    """`kaggle_environments.utils.process_schema` without the per-call `check_schema`."""

    error = None
    if use_default is True:
        data = kaggle_utils.default_schema(schema, copy.deepcopy(data))
    try:
        _validator_for(schema).validate(data)
    except Exception as err:  # noqa: BLE001 -- the stock function stringifies anything
        error = str(err)
    return error, data


def to_json(env: Any) -> dict[str, Any]:
    """`Environment.toJSON` without the deep copy.

    The field list mirrors the library's, so an upgrade that adds a key here
    would go unnoticed -- `test_to_json_matches_the_library` compares this
    against the real `toJSON` on a played episode and fails when they diverge.
    """

    if disabled():
        return env.toJSON()
    spec = env.specification
    return {
        "id": env.id,
        "name": spec.name,
        "title": spec.title,
        "description": spec.description,
        "version": spec.version,
        "module_version": kaggle_core.__version__,
        "configuration": env.configuration,
        "specification": {
            "action": spec.action,
            "agents": spec.agents,
            "configuration": spec.configuration,
            "info": spec.info,
            "observation": spec.observation,
            "reward": spec.reward,
        },
        "steps": env.steps,
        "rewards": [state.reward for state in env.steps[-1]],
        "statuses": [state.status for state in env.steps[-1]],
        "schema_version": 1,
        "info": env.info,
    }


def install() -> None:
    """Swap in the cached-validator `process_schema`. Idempotent.

    `core` binds the name at import (`from .utils import process_schema`), so
    both modules have to be patched or the step loop keeps the slow one.
    """

    global _installed
    if _installed or disabled():
        return
    kaggle_utils.process_schema = fast_process_schema
    kaggle_core.process_schema = fast_process_schema
    _installed = True


def uninstall() -> None:
    """Restore the library's own `process_schema`. Idempotent.

    Exists so the equivalence test can actually play an episode the stock way in
    a process that has already installed -- patching is global and an env var
    checked at install time cannot undo it.
    """

    global _installed
    kaggle_utils.process_schema = _STOCK_PROCESS_SCHEMA
    kaggle_core.process_schema = _STOCK_PROCESS_SCHEMA
    _VALIDATORS.clear()
    _installed = False
