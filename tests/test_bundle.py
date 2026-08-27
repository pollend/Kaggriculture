"""A submission bundle must stand alone.

The real failure this catches is silent: an agent imports the lab or its own
trainer, works perfectly in the repo, and then dies on Kaggle where neither
exists. Unpacking the tarball into an empty directory and importing it with the
repo off `sys.path` is the only check that actually proves it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

from agents.dqn_job_scorer.export_submission import build_submission
from kaggriculture_lab.cli import main as cli_main
from kaggriculture_lab.identity import AgentManifest

REPO = Path(__file__).resolve().parent.parent

# Agents we own and ship. Vendored competitor agents are not built here.
OWNED_AGENTS = [
    "agents/configurable_crop_surrogate/agent.toml",
    "agents/dqn_job_scorer/agent.toml",
    "agents/dqn_market_free_v2/agent.toml",
    "agents/dqn_market_free_v3/agent.toml",
    "agents/dqn_market_free_v3b/agent.toml",
    "agents/dqn_market_free_v3c/agent.toml",
    "agents/dqn_market_free_v3d/agent.toml",
    "agents/receding_horizon_planner/agent.toml",
]

# Anything here in a bundled submission means it would fail on Kaggle, or drag
# a training dependency into a inference-only artifact.
FORBIDDEN = {"torch", "kaggriculture_lab", "sim", "sklearn", "pandas", "duckdb", "joblib"}


def _build(manifest_path: str, tmp_path: Path) -> Path:
    archive = tmp_path / "submission.tar.gz"
    assert cli_main(["bundle", "--agent", manifest_path, "--output", str(archive)]) == 0
    unpacked = tmp_path / "unpacked"
    unpacked.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(unpacked, filter="data")
    return unpacked


@pytest.mark.parametrize("manifest_path", OWNED_AGENTS)
def test_bundle_imports_and_plays_standalone(manifest_path, tmp_path):
    unpacked = _build(manifest_path, tmp_path)
    assert (unpacked / "main.py").is_file(), "Kaggle requires a top-level main.py"

    probe = textwrap.dedent(f"""
        import sys
        sys.path = [p for p in sys.path if p != {str(REPO)!r}]
        sys.path.insert(0, ".")
        import main

        assert callable(main.agent), "main.agent is not callable"

        leaked = sorted({{m.split(".")[0] for m in sys.modules}} & {FORBIDDEN!r})
        assert not leaked, f"submission pulled in {{leaked}}"

        obs = {{
            "player": 0, "step": 0, "day": 0, "hour": 0,
            "farms": [
                {{"money": 3000.0, "tiles": [[None] * 10 for _ in range(10)],
                  "farmer": [4, 4], "hands": [],
                  "unlocked_quadrants": ["NW"], "hires_today": 0}},
                {{"money": 3000.0, "tiles": [[None] * 10 for _ in range(10)],
                  "farmer": [4, 4], "hands": [],
                  "unlocked_quadrants": ["NW"], "hires_today": 0}},
            ],
            "market": {{"inventory": {{}}, "prices": {{}}}},
            "town": {{"unlocked_shops": []}},
            "private": {{"shed": {{}}, "seeds": {{}}, "inventories": [{{}}]}},
        }}
        action = main.agent(obs)
        assert set(action) == {{"farmer", "hands", "market"}}, action
        assert len(action["market"]) <= 10
        print("ok")
    """)

    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=unpacked, capture_output=True, text=True, env={"PATH": ""}
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("manifest_path", OWNED_AGENTS)
def test_manifest_code_paths_all_exist(manifest_path):
    """A stale `code_paths` entry silently omits a module from the submission."""
    manifest = AgentManifest.load(manifest_path)
    missing = [path for path in manifest.code_paths if not (REPO / path).is_file()]
    assert not missing, f"{manifest_path} lists files that do not exist: {missing}"


def _module_scope_imports(tree: ast.AST):
    """Import nodes that run at import time, skipping function bodies.

    A deferred import inside a function is a different failure: `main.__getattr__`
    resolves `evaluate_episode` lazily, so a missing module there breaks the
    lab-side evaluator hook rather than the Kaggle inference path, and several
    shipped agents rely on that being optional. Module-scope imports have no such
    escape -- they run on `import main` and take the whole submission with them.
    """

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        else:
            yield from _module_scope_imports(node)


def _reachable(package_dir: Path, package: str, entry: str = "main") -> set[str]:
    """Modules in `package_dir` reachable by *module-scope* import from `entry`.

    Parsed rather than imported so this holds for agents whose inference stack is
    not installed in the test environment.
    """

    seen: set[str] = set()
    stack = [entry]
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        source = package_dir / f"{module}.py"
        if not source.is_file():
            continue
        for node in _module_scope_imports(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if base == package:
                    stack.extend(alias.name for alias in node.names)
                elif base.startswith(f"{package}."):
                    stack.append(base.split(".")[-1])
            elif isinstance(node, ast.Import):
                stack.extend(
                    alias.name.split(".")[-1]
                    for alias in node.names
                    if alias.name.startswith(f"{package}.")
                )
    return {module for module in seen if (package_dir / f"{module}.py").is_file()}


@pytest.mark.parametrize("manifest_path", OWNED_AGENTS)
def test_manifest_declares_every_module_the_package_imports(manifest_path):
    """The reverse of the check above, and the one that actually reached Kaggle.

    A declared-but-missing file fails the build loudly. An undeclared but
    *imported* module builds a bundle that looks complete and raises `ImportError`
    on the first turn -- which is how `dqn_market_free_v3d` shipped without
    `labor.py`, imported at module scope by both `market_state.py` and
    `order_space.py`.

    `test_bundle_imports_and_plays_standalone` catches this only for modules the
    happy path touches; this catches it for every module-scope import.

    Deferred imports are deliberately out of scope -- `configurable_crop_surrogate`,
    `dqn_job_scorer`, `dqn_market_free_v2` and `receding_horizon_planner` all reach
    `evaluation.py`/`lots.py` only through `main.__getattr__`, which Kaggle never
    calls. That is a real gap for a bundled league anchor whose evaluator runs, but
    it is not what breaks a submission.
    """

    package_dir = REPO / Path(manifest_path).parent
    package = ".".join(Path(manifest_path).parent.parts)
    declared = set(AgentManifest.load(manifest_path).code_paths)
    undeclared = sorted(
        f"{Path(manifest_path).parent.as_posix()}/{module}.py"
        for module in _reachable(package_dir, package)
        if f"{Path(manifest_path).parent.as_posix()}/{module}.py" not in declared
    )
    assert not undeclared, (
        f"{manifest_path} omits modules the package imports: {undeclared}"
    )


def test_bundle_excludes_training_code(tmp_path):
    """`code_paths` is an allowlist, so the trainer is excluded by construction."""
    unpacked = _build("agents/dqn_job_scorer/agent.toml", tmp_path)
    assert not (unpacked / "agents/dqn_job_scorer/training").exists()


def test_dqn_submission_exporter_creates_matching_minimal_folder_and_archive(tmp_path):
    output_dir = tmp_path / "submission"
    archive_path = tmp_path / "dqn_job_scorer_submission.tgz"

    built_dir, built_archive = build_submission(output_dir, archive_path)
    manifest = AgentManifest.load("agents/dqn_job_scorer/agent.toml")
    expected = {"main.py", *manifest.code_paths}
    unpacked_files = {path.relative_to(built_dir).as_posix() for path in built_dir.rglob("*") if path.is_file()}

    with tarfile.open(built_archive, "r:gz") as archive:
        archived_files = {member.name for member in archive.getmembers() if member.isfile()}

    assert unpacked_files == expected
    assert archived_files == expected
    assert not any("training" in Path(path).parts for path in archived_files)


def test_agent_owned_exporter_matches_the_manifest_allowlist(tmp_path):
    """`export_submission.py` is a second path to a submission.

    It bypasses `kaglab bundle`, so without this the two could drift and the
    agent-owned one could ship a file the manifest never vetted -- or miss one it
    needs. Both must produce exactly the manifest's allowlist plus `main.py`.
    """
    import subprocess
    import sys
    import tarfile

    script = REPO / "agents/dqn_job_scorer/export_submission.py"
    if not script.is_file():
        pytest.skip("agent does not ship its own exporter")

    subprocess.run([sys.executable, str(script)], cwd=REPO, check=True, capture_output=True)
    archive = REPO / "agents/dqn_job_scorer/dqn_job_scorer_submission.tgz"
    assert archive.is_file()

    with tarfile.open(archive) as handle:
        names = {name for name in handle.getnames() if not name.endswith("/")}

    manifest = AgentManifest.load("agents/dqn_job_scorer/agent.toml")
    assert names == {*manifest.code_paths, "main.py"}, (
        f"exporter and manifest disagree: "
        f"only-in-archive={sorted(names - {*manifest.code_paths, 'main.py'})} "
        f"only-in-manifest={sorted({*manifest.code_paths} - names)}"
    )
    assert not any("training" in name for name in names)
