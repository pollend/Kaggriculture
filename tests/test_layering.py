"""Pins the package dependency direction.

    sim/               -> nothing in this repo   game-agnostic evaluation
    kaggriculture_lab/ -> sim                    research infrastructure
    agents/<name>/     -> nothing in this repo   self-contained agents

The rule that matters most is the last one: no agent may import another agent.
Prototyping a change to one agent must not be able to break a different one, and
that only holds if it is checked. Duplication between agents is deliberate --
see `tests/test_game_facts.py` for the guard that keeps the copies honest.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCAL_PACKAGES = {"sim", "kaggriculture_lab", "agents"}

# Training and research code inside an agent may use the lab and the arena: it
# plugs into `model_updates`/`hooks` and evaluates with `sim.arena` by design.
# It is never part of a submission bundle, which the manifest's `code_paths`
# allowlist enforces separately.
INFRA_ALLOWED_SUFFIXES = (
    "train.py",
    "training.py",
    "analysis.py",
    "experiment.py",
    "cli.py",
    "evaluation.py",
)
INFRA_ALLOWED_DIRS = ("training",)

# SP-PSRO is deliberately an alternate training curriculum and artifact layer
# over the DQN job scorer. Keeping that relationship explicit here preserves
# the no-accidental-cross-agent rule for every other family.
AGENT_EXTENSIONS = {("sp_psro_job_scorer", "dqn_job_scorer")}


def _module_imports(path: Path) -> set[str]:
    """Top-level runtime imports. `if TYPE_CHECKING:` blocks are excluded --
    they cost nothing at runtime and exist only to type the hook signatures."""
    tree = ast.parse(path.read_text())

    type_checking_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            name = getattr(test, "id", None) or getattr(test, "attr", None)
            if name == "TYPE_CHECKING":
                for child in ast.walk(node):
                    type_checking_nodes.add(id(child))

    found: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in type_checking_nodes:
            continue
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return {name for name in found if name.split(".")[0] in LOCAL_PACKAGES}


def _source_files(package: str) -> list[Path]:
    return [
        path
        for path in (REPO / package).rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def _is_infra(path: Path) -> bool:
    return path.name.endswith(INFRA_ALLOWED_SUFFIXES) or any(
        part in INFRA_ALLOWED_DIRS for part in path.parts
    )


def test_sim_does_not_know_about_the_lab_or_any_agent():
    """`sim` is the measurement rig; it must stay reusable and game-agnostic."""
    offences = [
        f"{path.relative_to(REPO)} imports {module}"
        for path in _source_files("sim")
        for module in _module_imports(path)
        if module.split(".")[0] in {"kaggriculture_lab", "agents"}
    ]
    assert not offences, "sim must not depend on the lab or on agents:\n" + "\n".join(offences)


def test_no_agent_imports_another_agent():
    """The load-bearing rule: agents are independent, duplication and all."""
    offences = []
    for path in _source_files("agents"):
        owner = path.relative_to(REPO).parts[1]
        for module in _module_imports(path):
            parts = module.split(".")
            if (
                parts[0] == "agents"
                and len(parts) > 1
                and parts[1] != owner
                and (owner, parts[1]) not in AGENT_EXTENSIONS
            ):
                offences.append(f"{path.relative_to(REPO)} imports {module}")
    assert not offences, "an agent imported another agent:\n" + "\n".join(offences)


def test_agent_inference_code_does_not_import_the_lab_or_sim():
    """Submission-path modules must be importable with nothing but numpy."""
    offences = []
    for path in _source_files("agents"):
        if _is_infra(path):
            continue
        for module in _module_imports(path):
            if module.split(".")[0] in {"kaggriculture_lab", "sim"}:
                offences.append(f"{path.relative_to(REPO)} imports {module}")
    assert not offences, (
        "agent inference code must not import the lab or sim "
        "(training/research modules are exempt):\n" + "\n".join(offences)
    )


def test_lab_does_not_depend_on_any_agent():
    """Generic infrastructure typed against one family is what broke prototyping."""
    offences = [
        f"{path.relative_to(REPO)} imports {module}"
        for path in _source_files("kaggriculture_lab")
        for module in _module_imports(path)
        if module.split(".")[0] == "agents"
    ]
    assert not offences, (
        "kaggriculture_lab must reach agents through manifests and adapters, "
        "never by import:\n" + "\n".join(offences)
    )
