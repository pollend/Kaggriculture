"""Build the minimal Receding Horizon Planner Kaggle submission."""

from __future__ import annotations

import io
import shutil
import tarfile
import tomllib
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parents[1]
MANIFEST = AGENT_DIR / "agent.toml"
OUTPUT_DIR = AGENT_DIR / "submission"
ARCHIVE = AGENT_DIR / "receding_horizon_planner_submission.tgz"


def build_submission(
    output_dir: Path = OUTPUT_DIR,
    archive_path: Path = ARCHIVE,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    archive_path = Path(archive_path)
    if output_dir.resolve() in {REPO_ROOT.resolve(), AGENT_DIR.resolve()} or output_dir.is_symlink():
        raise ValueError(f"unsafe submission directory: {output_dir}")
    with MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)
    paths = tuple(Path(value) for value in manifest["code_paths"])
    if any(path.is_absolute() or ".." in path.parts or "training" in path.parts for path in paths):
        raise ValueError("submission allowlist contains an unsafe or training path")
    missing = [path for path in paths if not (REPO_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing submission files: {missing}")

    module, separator, attribute = str(manifest["entry_point"]).partition(":")
    if not separator or attribute not in {"agent", "make_agent"}:
        raise ValueError("unsupported manifest entry point")
    # Kaggle needs a ready-to-call root agent, so use the package's default main.
    package = ".".join(AGENT_DIR.relative_to(REPO_ROOT).parts)
    shim = f"from {package}.main import agent as agent\n".encode()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for relative in paths:
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    (output_dir / "main.py").write_bytes(shim)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("main.py")
        info.size = len(shim)
        archive.addfile(info, io.BytesIO(shim))
        for relative in paths:
            archive.add(REPO_ROOT / relative, arcname=relative.as_posix(), recursive=False)
    return output_dir, archive_path


def main() -> int:
    output, archive = build_submission()
    files = [path for path in output.rglob("*") if path.is_file()]
    print(f"Submission folder: {output}")
    print(f"Submission archive: {archive}")
    print(f"Included files: {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
