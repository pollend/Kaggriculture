"""Build the minimal Market Free Kaggle submission.

Run from anywhere inside or outside the repository:

    python agents/dqn_market_free_v3e/export_submission.py

The exporter recreates ``submission/`` beside this file and writes a matching
``dqn_market_free_v3e_submission.tar.gz``. Both contain the inference files named by
``agent.toml``, the trained ``weights.npz``, and the top-level ``main.py``
required by Kaggle.

``weights.npz`` is deliberately *not* in ``agent.toml``'s ``code_paths``: the
league bundles this agent as an anchor long before a checkpoint has been
exported, and a declared-but-missing path fails that bundle outright. A Kaggle
submission is the opposite case -- shipping this agent without weights ships a
policy that never trades, because `main.load_network` falls back to a silent
network rather than a strong donor. So the file is required here and only here.
"""

from __future__ import annotations

import ast
import os
import shutil
import tarfile
import tempfile
import tomllib
from pathlib import Path
from typing import Any

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parents[1]
MANIFEST_PATH = AGENT_DIR / "agent.toml"
SUBMISSION_DIR = AGENT_DIR / "submission"
#: `.tar.gz`, not `.tgz`. Kaggle's submission endpoint rejects the short
#: extension with a 400 and no useful message, so v2's archives had to be renamed
#: by hand every time. The tarball's contents are identical either way.
ARCHIVE_PATH = AGENT_DIR / "dqn_market_free_v3e_submission.tar.gz"
WEIGHTS = Path("agents/dqn_market_free_v3e/weights.npz")
PACKAGE = "agents.dqn_market_free_v3e"


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


def _reachable_modules(entry: str = "main") -> set[str]:
    """Every module in this package reachable by import from ``entry``.

    Walks the AST rather than importing: the exporter has to run in a checkout
    where `torch` may be absent, and importing `main` would drag the whole
    inference stack in.

    Both import spellings this package uses are followed --
    ``from agents.dqn_market_free_v3e import labor`` and
    ``from agents.dqn_market_free_v3e.labor import ...``.

    Module-scope only. A deferred import inside a function is how
    `main.__getattr__` reaches the evaluator hook, which Kaggle never calls, and
    treating that as required would reject manifests that are correct for a
    submission.
    """

    seen: set[str] = set()
    stack = [entry]
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        source = AGENT_DIR / f"{module}.py"
        if not source.is_file():
            continue
        for node in _module_scope_imports(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if base == PACKAGE:
                    stack.extend(alias.name for alias in node.names)
                elif base.startswith(f"{PACKAGE}."):
                    stack.append(base.split(".")[-1])
            elif isinstance(node, ast.Import):
                stack.extend(
                    alias.name.split(".")[-1]
                    for alias in node.names
                    if alias.name.startswith(f"{PACKAGE}.")
                )
    return {module for module in seen if (AGENT_DIR / f"{module}.py").is_file()}


def _manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _inference_files(manifest: dict[str, Any]) -> tuple[Path, ...]:
    declared = tuple(Path(value) for value in manifest.get("code_paths", ()))
    if not declared:
        raise ValueError(f"{MANIFEST_PATH} does not declare any code_paths")

    missing: list[Path] = []
    for relative in declared:
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"code_path must stay inside the repository: {relative}")
        if "training" in relative.parts:
            raise ValueError(f"training code cannot be included in a submission: {relative}")
        if not (REPO_ROOT / relative).is_file():
            missing.append(relative)

    if missing:
        joined = ", ".join(path.as_posix() for path in missing)
        raise FileNotFoundError(f"agent.toml declares missing files: {joined}")

    package_main = Path("agents/dqn_market_free_v3e/main.py")
    if package_main not in declared:
        raise ValueError(f"agent.toml must include {package_main.as_posix()}")

    # The reverse of the existence check above, and the one that actually bites.
    # A missing *declared* file fails the build; an undeclared *reachable* one
    # built a bundle that looked fine and raised `ImportError` on Kaggle's first
    # turn. `labor.py` shipped that way -- imported at module scope by both
    # `market_state.py` and `order_space.py`, listed nowhere.
    undeclared = sorted(
        f"agents/dqn_market_free_v3e/{module}.py"
        for module in _reachable_modules()
        if Path(f"agents/dqn_market_free_v3e/{module}.py") not in declared
    )
    if undeclared:
        joined = ", ".join(undeclared)
        raise ValueError(
            f"agent.toml omits modules the package imports: {joined}. "
            "A submission built without them raises ImportError on Kaggle."
        )

    if not (REPO_ROOT / WEIGHTS).is_file():
        raise FileNotFoundError(
            f"{WEIGHTS.as_posix()} does not exist: a submission built without it "
            "runs the silent fallback network and never trades. Export a reviewed "
            "checkpoint first:\n"
            "  python -m agents.dqn_market_free_v3e.cli model export-active "
            f"--output {WEIGHTS.as_posix()} --confirm-release"
        )
    if WEIGHTS in declared:
        return declared
    return (*declared, WEIGHTS)


def _top_level_main(manifest: dict[str, Any]) -> str:
    entry_point = str(manifest.get("entry_point", ""))
    module, separator, attribute = entry_point.partition(":")
    if not separator or attribute != "agent" or not module.endswith(".main"):
        raise ValueError(f"unsupported submission entry_point: {entry_point!r}")
    package = module.split(".", 1)[0]
    target = os.path.join(*module.split(".")) + ".py"
    # `kaggle_environments` runs this file through `exec(code_object, {})` and
    # only *appends* its directory to `sys.path`. Two consequences, both Kaggle-
    # only and neither reproducible from a repo checkout:
    #
    #   * there is no `__file__` to locate ourselves with, so the directory is
    #     recovered by looking for the package on the path instead;
    #   * every bundled environment directory is already ahead of us, and
    #     `envs/lux_ai_s3/` ships its own `agents.py`, so a plain
    #     `from agents... import` binds *that* module and dies on its relative
    #     import.
    #
    # So claim the name: move our own directory to the front and drop anything
    # already bound under the package, leaving the import below to re-resolve.
    #
    # `get_last_callable` returns the last callable in these globals, so nothing
    # callable may be defined after `agent`.
    return f"""import os
import sys

_TARGET = {target!r}
for _candidate in [*sys.path, os.getcwd()]:
    if _candidate and os.path.isfile(os.path.join(_candidate, _TARGET)):
        _HERE = os.path.abspath(_candidate)
        break
else:
    _HERE = os.getcwd()

while _HERE in sys.path:
    sys.path.remove(_HERE)
sys.path.insert(0, _HERE)
for _shadow in [
    _name
    for _name in sys.modules
    if _name == {package!r} or _name.startswith({package + "."!r})
]:
    del sys.modules[_shadow]

from {module} import agent as agent
"""


def _safe_generated_directory(path: Path) -> None:
    resolved = path.resolve()
    protected = {REPO_ROOT.resolve(), AGENT_DIR.resolve(), Path(resolved.anchor)}
    if resolved in protected or path.is_symlink():
        raise ValueError(f"refusing to replace unsafe output directory: {path}")


def build_submission(
    output_dir: Path = SUBMISSION_DIR,
    archive_path: Path = ARCHIVE_PATH,
) -> tuple[Path, Path]:
    """Create an unpacked submission folder and equivalent ``.tar.gz`` archive."""

    output_dir = Path(output_dir)
    archive_path = Path(archive_path)
    _safe_generated_directory(output_dir)

    manifest = _manifest()
    included = _inference_files(manifest)
    archive_members = (Path("main.py"), *included)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".free-submission-", dir=output_dir.parent))
    archive_fd, temporary_archive_name = tempfile.mkstemp(
        prefix=".free-submission-",
        suffix=".tar.gz",
        dir=archive_path.parent,
    )
    os.close(archive_fd)
    temporary_archive = Path(temporary_archive_name)

    try:
        (staging / "main.py").write_text(_top_level_main(manifest), encoding="utf-8")
        for relative in included:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative, destination)

        with tarfile.open(temporary_archive, "w:gz") as archive:
            for relative in archive_members:
                archive.add(staging / relative, arcname=relative.as_posix(), recursive=False)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
        os.replace(temporary_archive, archive_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        temporary_archive.unlink(missing_ok=True)
        raise

    return output_dir, archive_path


def main() -> int:
    output_dir, archive_path = build_submission()
    files = sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file())
    print(f"Submission folder: {output_dir}")
    print(f"Submission archive: {archive_path}")
    print(f"Included files ({len(files)}):")
    for path in files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
