"""Three costs `kaggle_environments` pays per step that buy this repo nothing.

None of them changes what an episode does -- `tests/test_fastenv.py` plays
episodes both ways and compares the replays -- and all are large. Measured on one
720-step self-play episode under cProfile:

| cost | per episode |
| --- | --- |
| kaggriculture's own game logic | 0.08 s |
| re-validating the action *schema* every step | 0.81 s |
| `toJSON` deep-copying all 720 steps | 1.01 s |
| deep-copying the state for every agent call | 3.71 s |

That last row was measured later, on `dqn_market_free` against the promotion
benchmark rather than on self-play: 3.71 s of a 9.36 s episode, against 1.16 s
for the whole learner. The framework was costing three times what the agent did.

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

3. **`__get_shared_state`** deep-copies a seat's whole state before every single
   agent call, then walks the schema deleting hidden fields and splicing in
   shared ones. The copy exists so the walk has something safe to mutate, but
   `Agent.act` immediately runs `structify` over the observation, which clones
   every dict and list on the way to the agent regardless. So the deep copy is
   isolation that is about to be redone: only the nodes the walk actually touches
   need to be new. The replacement copies those and shares the rest.

Set `KAGG_NO_FASTENV=1` to run the stock library instead -- the flag is the
first thing to try if an episode ever looks wrong.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import jsonschema
from kaggle_environments import agent as kaggle_agent
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

#: Likewise for the shared-state builder, captured under its mangled name.
_STOCK_GET_SHARED_STATE = kaggle_core.Environment._Environment__get_shared_state

#: And for `structify`, which three modules bind by value at import.
_STOCK_STRUCTIFY = kaggle_utils.structify

#: The only two types `structify` does anything to. Everything else is returned
#: unchanged, so it never has to be handed to the function at all.
_CONTAINERS = (list, dict)


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


def fast_structify(o: Any) -> Any:
    """`kaggle_environments.utils.structify` without the redundant dict builds.

    The stock version ends every dict node with `Struct(**{...})`, which builds
    the comprehension's dict, unpacks it into a fresh kwargs dict, filters that
    into a third inside `Struct.__init__`, and only then fills the instance --
    four dicts per node, on a tree with a few hundred of them, twice per agent
    call. This fills the `Struct` directly and keeps the rest identical: same
    dispatch order, same recursion, same dropping of an entry literally named
    `items`, which `Struct.__init__` excludes because it would shadow
    `dict.items`.

    Scalars are also filtered at the call site rather than recursed into. A tile
    is a handful of numbers under a handful of keys, so the tree is mostly
    leaves, and a leaf costs a whole function call to be handed straight back.
    """

    if isinstance(o, list):
        return [
            fast_structify(value) if isinstance(value, _CONTAINERS) else value
            for value in o
        ]
    if isinstance(o, dict):
        entries = {
            key: (fast_structify(value) if isinstance(value, _CONTAINERS) else value)
            for key, value in o.items()
            if key != "items"
        }
        clone = kaggle_utils.Struct()
        dict.update(clone, entries)
        clone.__dict__.update(entries)
        return clone
    return o


def _struct_shell(source: Any) -> Any:
    """A one-level copy of `source` that a deep copy of it would be indistinguishable from.

    `Struct` keeps every entry twice -- once as a dict item, once in `__dict__` --
    and `deepcopy` restores both, because a dict subclass reduces to its items
    plus its instance dict. Both halves are copied here for the same reason:
    `Environment.__get_shared_state` is read by item in `act_agent` and by
    attribute at `core.py:484`.
    """

    clone = kaggle_utils.Struct()
    dict.update(clone, source)
    clone.__dict__.update(getattr(source, "__dict__", None) or source)
    return clone


def _shared_state(shared_state: Any, state: Any, schema_props: Any) -> Any:
    """`Environment.__get_shared_state`'s deep copy and mutating walk, in one pass.

    The library deep-copies the whole state and then walks the schema deleting
    hidden fields and splicing in shared ones. Only the nodes that walk touches
    need to be new: everything else is handed straight to `Agent.act`, which
    calls `structify` on the observation and so clones every dict and list it
    gives the agent anyway. Sharing the untouched subtrees with the environment
    is therefore invisible -- nothing downstream mutates them.

    Item assignment only, no `__dict__` sync, because that is precisely what the
    stock walk does to its deep copy.
    """

    result = _struct_shell(state)
    for key, prop in schema_props.items():
        if kaggle_utils.get(prop, bool, path=["hidden"], fallback=False):
            if key in result:
                del result[key]
        elif kaggle_utils.get(prop, bool, path=["shared"], fallback=False):
            result[key] = shared_state[key]
        elif kaggle_utils.has(prop, dict, path=["properties"]):
            result[key] = _shared_state(
                shared_state[key], state[key], prop["properties"]
            )
    return result


def fast_get_shared_state(self: Any, position: int) -> Any:
    """`Environment.__get_shared_state` without the per-call deep copy.

    The schema is memoised on the environment as well. The library's own
    `__state_schema` property tries to cache into `__state_schema_value`, but the
    assignment is name-mangled to `_Environment__state_schema_value` while the
    `hasattr` guard tests the unmangled string, so the cache never hits and the
    schema is rebuilt on every call -- twice per step per agent.
    """

    schema = getattr(self, "_fastenv_state_schema", None)
    if schema is None:
        schema = self._Environment__state_schema["properties"]
        self._fastenv_state_schema = schema
    return _shared_state(self.state[0], self.state[position], schema)


def install() -> None:
    """Swap in the cached-validator `process_schema` and the copy-free shared state.

    Idempotent. `core` binds `process_schema` at import
    (`from .utils import process_schema`), so both modules have to be patched or
    the step loop keeps the slow one. `__get_shared_state` is patched under its
    mangled name because that is what the class body compiled its call sites to.
    """

    global _installed
    if _installed or disabled():
        return
    kaggle_utils.process_schema = fast_process_schema
    kaggle_core.process_schema = fast_process_schema
    kaggle_core.Environment._Environment__get_shared_state = fast_get_shared_state
    kaggle_utils.structify = fast_structify
    kaggle_core.structify = fast_structify
    kaggle_agent.structify = fast_structify
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
    kaggle_core.Environment._Environment__get_shared_state = _STOCK_GET_SHARED_STATE
    kaggle_utils.structify = _STOCK_STRUCTIFY
    kaggle_core.structify = _STOCK_STRUCTIFY
    kaggle_agent.structify = _STOCK_STRUCTIFY
    _VALIDATORS.clear()
    _installed = False
