"""Build the minimal Market Free Kaggle submission.

Run from anywhere inside or outside the repository:

    python agents/dqn_market_free/export_submission.py

The exporter recreates ``submission/`` beside this file and writes a matching
``dqn_market_free_submission.tgz``. Both contain the inference files named by
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
ARCHIVE_PATH = AGENT_DIR / "dqn_market_free_submission.tgz"
WEIGHTS = Path("agents/dqn_market_free/weights.npz")


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

    package_main = Path("agents/dqn_market_free/main.py")
    if package_main not in declared:
        raise ValueError(f"agent.toml must include {package_main.as_posix()}")

    if not (REPO_ROOT / WEIGHTS).is_file():
        raise FileNotFoundError(
            f"{WEIGHTS.as_posix()} does not exist: a submission built without it "
            "runs the silent fallback network and never trades. Export a reviewed "
            "checkpoint first:\n"
            "  python -m agents.dqn_market_free.cli model export-active "
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
    return f"from {module} import agent as agent\n"


def _safe_generated_directory(path: Path) -> None:
    resolved = path.resolve()
    protected = {REPO_ROOT.resolve(), AGENT_DIR.resolve(), Path(resolved.anchor)}
    if resolved in protected or path.is_symlink():
        raise ValueError(f"refusing to replace unsafe output directory: {path}")


def build_submission(
    output_dir: Path = SUBMISSION_DIR,
    archive_path: Path = ARCHIVE_PATH,
) -> tuple[Path, Path]:
    """Create an unpacked submission folder and equivalent ``.tgz`` archive."""

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
        suffix=".tgz",
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
