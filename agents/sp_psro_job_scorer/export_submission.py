"""Build a submission containing the selected SP-PSRO champion only."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import tomllib
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parents[1]
MANIFEST = AGENT_DIR / "agent.toml"
WEIGHTS = AGENT_DIR / "weights.npz"
SUBMISSION = AGENT_DIR / "submission"
ARCHIVE = AGENT_DIR / "sp_psro_job_scorer_submission.tgz"


def build_submission(output_dir: Path = SUBMISSION, archive_path: Path = ARCHIVE) -> tuple[Path, Path]:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            "no released champion; run `python -m agents.sp_psro_job_scorer.cli "
            "model export-active --output agents/sp_psro_job_scorer/weights.npz "
            "--confirm-release` first"
        )
    with MANIFEST.open("rb") as stream:
        manifest = tomllib.load(stream)
    declared = [Path(item) for item in manifest["code_paths"]]
    for path in declared:
        if path.is_absolute() or ".." in path.parts or not (REPO_ROOT / path).is_file():
            raise ValueError(f"invalid code_path: {path}")
        if "training" in path.parts:
            raise ValueError(f"training file declared for submission: {path}")
    output_dir = Path(output_dir)
    archive_path = Path(archive_path)
    staging = Path(tempfile.mkdtemp(prefix=".sp-psro-submission-", dir=output_dir.parent))
    archive_fd, temporary_name = tempfile.mkstemp(prefix=".sp-psro-submission-", suffix=".tgz", dir=archive_path.parent)
    os.close(archive_fd)
    temporary = Path(temporary_name)
    try:
        (staging / "main.py").write_text("from agents.sp_psro_job_scorer.main import agent as agent\n")
        for relative in declared:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative, destination)
        destination = staging / "agents/sp_psro_job_scorer/weights.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEIGHTS, destination)
        with tarfile.open(temporary, "w:gz") as archive:
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                archive.add(path, arcname=path.relative_to(staging).as_posix())
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
        os.replace(temporary, archive_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        temporary.unlink(missing_ok=True)
        raise
    return output_dir, archive_path


def main() -> int:
    output, archive = build_submission()
    print(f"Submission folder: {output}")
    print(f"Submission archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
