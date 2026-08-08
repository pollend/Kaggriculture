"""Resolve an agent spec string to a callable.

Specs are resolved inside the worker process, never pickled across it, so a
spec is always a plain string.

Supported forms::

    pass                      built-in
    starter                   built-in
    random                    built-in (NOT reproducible -- see below)
    main.py                   file path; uses `agent` if present, else last callable
    pkg.mod:agent             importable module attribute
    pkg.mod:make_agent?a=1    factory called with a config dict

Built-in names come from the environment itself, so this module knows about
no particular game. A reproducible stochastic opponent is reached through the
factory form, e.g. ``kaggriculture_lab.opponents:make_seeded_random?seed=7``.

The `?k=v` suffix works on any form. For a factory (`module:name` whose
attribute is callable and declares a parameter) the parsed dict is passed as a
keyword-free single positional argument. For every other form the dict is
exported as JSON in the ``KAGG_AGENT_CONFIG`` environment variable *before*
the module is imported, so a submission-shaped `main.py` can read it at import
time and stay a bare one-arg `agent(obs)`.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import unquote

CONFIG_ENV_VAR = "KAGG_AGENT_CONFIG"
DEFAULT_ENV = "kaggriculture"

# Built-ins whose output differs between two runs of the same seed, so a caller
# can warn rather than silently produce unreproducible results. The name is the
# environment's, not ours: `kaggriculture.random_agent` builds a fresh unseeded
# `random.Random()` per call, so seeding the global module does not help.
NONDETERMINISTIC_BUILTINS = frozenset({"random"})


@lru_cache(maxsize=8)
def builtin_agents(env_name: str = DEFAULT_ENV) -> dict[str, Any]:
    """The environment's own registered agents, by name.

    Read from the environment rather than imported from a specific game module,
    so this resolver stays usable for any kaggle-environments env.
    """
    from kaggle_environments import make

    return dict(make(env_name).agents)


def _coerce(value: str) -> Any:
    """Turn a query-string value into a number/bool/None where it clearly is one."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def split_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """Split ``target?k=v&j=2`` into ``("target", {"k": v, "j": 2})``.

    Percent-escapes are decoded but ``+`` is left alone. `parse_qsl` would read
    it as a space, which silently corrupts any value containing one -- most
    painfully a checkpoint path like ``runs/best/round0045_margin+2071.npz``.
    These values are file paths and scalars, not HTML form data.
    """
    target, _, query = spec.partition("?")
    config: dict[str, Any] = {}
    for part in query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        config[unquote(key)] = _coerce(unquote(value))
    return target, config


def _load_path(path: Path) -> Any:
    """Import a .py file and return its `agent` attribute, else its last callable."""
    module_name = f"_kagg_agent_{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import agent file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # The real submission contract is a module-level `agent(obs)`; fall back to
    # the last defined callable to match how kaggle-environments loads sources.
    if callable(getattr(module, "agent", None)):
        return module.agent
    callables = [v for v in vars(module).values() if callable(v) and getattr(v, "__module__", None) == module_name]
    if not callables:
        raise ValueError(f"no callable agent found in {path}")
    return callables[-1]


def _is_factory(obj: Any) -> bool:
    """A factory is a callable that does not itself look like an agent.

    An agent takes exactly one required positional parameter (`obs`). Anything
    else that is callable is treated as something to call to *get* an agent.
    """
    if not callable(obj):
        return False
    try:
        params = inspect.signature(obj).parameters.values()
    except (TypeError, ValueError):
        return False
    required = [p for p in params if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(required) != 1


def resolve(spec: str, env_name: str = DEFAULT_ENV) -> Callable[[dict], dict]:
    """Resolve a spec string to a one-argument agent callable."""
    target, config = split_spec(spec)

    if target in builtin_agents(env_name):
        return builtin_agents(env_name)[target]

    # Config must be visible before import so a submission-shaped module can
    # read it at import time.
    previous = os.environ.get(CONFIG_ENV_VAR)
    if config:
        os.environ[CONFIG_ENV_VAR] = json.dumps(config)
    try:
        if ":" in target:
            module_name, _, attr = target.partition(":")
            obj = getattr(importlib.import_module(module_name), attr)
        else:
            path = Path(target)
            if not path.exists():
                raise ValueError(f"unknown agent spec: {spec!r} (no such file)")
            obj = _load_path(path)

        return obj(config) if _is_factory(obj) else obj
    finally:
        if config:
            if previous is None:
                os.environ.pop(CONFIG_ENV_VAR, None)
            else:
                os.environ[CONFIG_ENV_VAR] = previous


def agent_config() -> dict[str, Any]:
    """Read the injected sweep config. For use from inside an agent module."""
    raw = os.environ.get(CONFIG_ENV_VAR)
    return json.loads(raw) if raw else {}


def is_deterministic(spec: str) -> bool:
    """False for specs known to differ between two runs of the same seed."""
    return split_spec(spec)[0] not in NONDETERMINISTIC_BUILTINS
