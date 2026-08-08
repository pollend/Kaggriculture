"""Family-specific candidate generation, behind a parameter-agnostic interface.

The league breeds populations without knowing what a "strategy" is for any
particular agent. Everything family-shaped -- how parameters are sampled,
mutated, crossed, encoded for a surrogate model, and proposed in bulk -- lives
in the agent's own adapter, reached through `AgentManifest.training_adapter`.

Parameters are opaque to this module on purpose. Typing this Protocol against
one agent's dataclass is what previously made every agent's prototype changes
able to break the others.
"""

from __future__ import annotations

import importlib
import random
from typing import Any, Protocol, TypeVar

from kaggriculture_lab.identity import AgentManifest, AgentRevision

# Whatever the family uses for parameters. The league only ever passes these
# values back to the same adapter, so it never needs to know the type.
Params = TypeVar("Params")


class TrainingAdapter(Protocol[Params]):
    family_id: str

    def decode(self, revision: AgentRevision) -> Params: ...

    def encode(self, params: Params) -> dict[str, Any]: ...

    def sample(self, rng: random.Random) -> Params: ...

    def mutate(self, parent: Params, rng: random.Random) -> Params: ...

    def crossover(self, left: Params, right: Params, rng: random.Random) -> Params: ...


def propose(
    adapter: Any,
    count: int,
    parents: list[Any],
    rng: random.Random,
    reader: Any,
) -> list[Any]:
    """Ask the adapter for `count` model-guided candidates, best first.

    Optional: an adapter that has no surrogate model simply does not define
    `propose`, and the caller falls back to mutation. Keeping this out of the
    league is what lets one family use a random forest, another use a Q-network,
    and a third use nothing at all.
    """
    hook = getattr(adapter, "propose", None)
    if hook is None:
        return []
    return list(hook(count=count, parents=parents, rng=rng, reader=reader))


def load_training_adapter(manifest: AgentManifest) -> TrainingAdapter | None:
    if not manifest.training_adapter:
        return None
    module_name, separator, attribute = manifest.training_adapter.partition(":")
    if not separator:
        raise ValueError("training_adapter must use module:attribute syntax")
    adapter = getattr(importlib.import_module(module_name), attribute)
    if adapter.family_id != manifest.family_id:
        raise ValueError("training adapter family_id does not match agent manifest")
    return adapter


def allocate_generation(count: int) -> dict[str, int]:
    """Deterministic largest-remainder allocation for the 20/50/20/10 mix."""

    weights = {"elite": 0.20, "evolution": 0.50, "surrogate": 0.20, "random": 0.10}
    raw = {key: count * weight for key, weight in weights.items()}
    allocated = {key: int(value) for key, value in raw.items()}
    remaining = count - sum(allocated.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - allocated[key]), key))
    for key in order[:remaining]:
        allocated[key] += 1
    if count >= 10 and allocated["random"] == 0:
        donor = max((key for key in allocated if key != "random"), key=allocated.get)
        allocated[donor] -= 1
        allocated["random"] = 1
    return allocated
