#!/usr/bin/env python3
"""Mirror Kaggriculture episode replays from the public Kaggle datasets.

The competition publishes one dataset per day (``kaggle/kaggriculture-episodes-<date>``)
holding ~700 replay JSONs of ~30 MB each, plus a ``manifest.csv`` ranking that day's
episodes by ``avg_score``.  A separate index dataset
(``kaggle/kaggriculture-episodes-index``) lists which days exist.

This script walks the index, takes the top ``--top`` episodes of each day by
``avg_score``, and stores each replay gzipped on disk.  Gzip buys roughly 55x, so a
30 MB replay lands at ~550 KB.  Downloads are unauthenticated, resumable (an existing
output file is skipped) and run across a small thread pool.

Nothing here touches PostgreSQL; ingesting the mirror into ``episode_payloads`` is a
separate step.

    python scripts/pull_kaggle_episodes.py --top 50
    python scripts/pull_kaggle_episodes.py --top 100 --since 2026-08-15 --workers 8
    python scripts/pull_kaggle_episodes.py --list
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

API = "https://www.kaggle.com/api/v1/datasets/download"
INDEX_DATASET = "kaggle/kaggriculture-episodes-index"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data" / "kaggle_episodes"

_print_lock = threading.Lock()


def _log(message: str) -> None:
    with _print_lock:
        print(message, file=sys.stderr, flush=True)


def _fetch(dataset: str, file_name: str, *, retries: int = 4, timeout: int = 300) -> bytes:
    """Download one file out of a public dataset, unwrapping Kaggle's zip if present.

    Kaggle returns the raw bytes for a single small file but zips larger ones, and
    which of the two you get is not advertised in the response headers, so sniff.
    """
    url = f"{API}/{dataset}?file_name={urllib.parse.quote(file_name)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "kaggriculture-lab"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            break
        except (urllib.error.URLError, TimeoutError, OSError) as error:  # noqa: PERF203
            last = error
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    else:  # pragma: no cover - the loop always breaks or raises
        raise RuntimeError(str(last))

    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            if len(names) != 1:
                raise RuntimeError(f"{dataset}/{file_name}: expected 1 zip member, got {names}")
            return archive.read(names[0])
    return payload


def _read_csv(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


@dataclass(frozen=True)
class Day:
    date: str
    dataset: str
    episode_count: int
    total_bytes: int
    top_avg_score: float
    median_avg_score: float


def load_index(root: Path, *, refresh: bool) -> list[Day]:
    """Fetch (or reuse) the index manifest naming every published day."""
    cached = root / "index_manifest.csv"
    if refresh or not cached.exists():
        payload = _fetch(INDEX_DATASET, "manifest.csv")
        root.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(payload)
    else:
        payload = cached.read_bytes()

    days = []
    for row in _read_csv(payload):
        slug = row["daily_dataset_slug"]
        days.append(
            Day(
                date=row["date"],
                dataset=slug if "/" in slug else f"kaggle/{slug}",
                episode_count=int(row["episode_count"]),
                total_bytes=int(row["total_bytes"]),
                top_avg_score=float(row["top_avg_score"]),
                median_avg_score=float(row["median_avg_score"]),
            )
        )
    return sorted(days, key=lambda day: day.date)


def load_day_manifest(day: Day, root: Path, *, refresh: bool) -> list[dict[str, str]]:
    """Fetch (or reuse) one day's per-episode manifest, ranked by avg_score."""
    cached = root / day.date / "manifest.csv"
    if refresh or not cached.exists():
        payload = _fetch(day.dataset, "manifest.csv")
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(payload)
    else:
        payload = cached.read_bytes()

    rows = _read_csv(payload)
    rows.sort(key=lambda row: float(row["avg_score"]), reverse=True)
    return rows


def download_episode(day: Day, episode_id: str, destination: Path) -> int:
    """Download one replay and gzip it into place.  Returns bytes written."""
    payload = _fetch(day.dataset, f"{episode_id}.json")
    # Fail loudly on a truncated or error-page response rather than caching garbage.
    json.loads(payload)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with gzip.open(temporary, "wb", compresslevel=6) as handle:
        handle.write(payload)
    temporary.replace(destination)
    return destination.stat().st_size


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024 or unit == "TB":
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="mirror directory (default: data/kaggle_episodes)")
    parser.add_argument("--top", type=int, default=50, help="episodes per day, highest avg_score first (default: %(default)s)")
    parser.add_argument("--since", help="skip days before this ISO date")
    parser.add_argument("--until", help="skip days after this ISO date")
    parser.add_argument("--date", action="append", help="only this date; repeatable")
    parser.add_argument("--workers", type=int, default=6, help="concurrent downloads (default: %(default)s)")
    parser.add_argument("--min-score", type=float, help="skip episodes below this avg_score")
    parser.add_argument("--refresh-manifests", action="store_true", help="re-fetch manifests instead of reusing the cache")
    parser.add_argument("--list", action="store_true", help="print the index and what would be pulled, then exit")
    parser.add_argument("--dry-run", action="store_true", help="resolve the work list without downloading replays")
    args = parser.parse_args(argv)

    root: Path = args.root
    days = load_index(root, refresh=args.refresh_manifests or args.list)

    if args.date:
        wanted = set(args.date)
        days = [day for day in days if day.date in wanted]
    if args.since:
        days = [day for day in days if day.date >= args.since]
    if args.until:
        days = [day for day in days if day.date <= args.until]
    if not days:
        _log("no days matched the filters")
        return 1

    if args.list:
        print(f"{'date':12} {'episodes':>9} {'raw':>9} {'top':>10} {'median':>10}  dataset")
        for day in days:
            print(
                f"{day.date:12} {day.episode_count:>9} {_human(day.total_bytes):>9} "
                f"{day.top_avg_score:>10.1f} {day.median_avg_score:>10.1f}  {day.dataset}"
            )
        return 0

    jobs: list[tuple[Day, str, Path]] = []
    skipped = 0
    for day in days:
        rows = load_day_manifest(day, root, refresh=args.refresh_manifests)
        if args.min_score is not None:
            rows = [row for row in rows if float(row["avg_score"]) >= args.min_score]
        if args.top > 0:
            rows = rows[: args.top]
        for row in rows:
            episode_id = row["episode_id"]
            destination = root / day.date / f"{episode_id}.json.gz"
            if destination.exists():
                skipped += 1
                continue
            jobs.append((day, episode_id, destination))

    _log(f"{len(days)} day(s), {len(jobs)} episode(s) to fetch, {skipped} already mirrored")
    if args.dry_run or not jobs:
        return 0

    free = shutil.disk_usage(root if root.exists() else Path.cwd()).free
    _log(f"free disk: {_human(free)}")

    written = 0
    failures: list[tuple[str, str]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(download_episode, day, episode_id, destination): (day, episode_id)
            for day, episode_id, destination in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            day, episode_id = futures[future]
            try:
                written += future.result()
            except Exception as error:  # noqa: BLE001 - one bad episode must not kill the run
                failures.append((f"{day.date}/{episode_id}", str(error)))
                _log(f"  FAIL {day.date}/{episode_id}: {error}")
                continue
            if index % 25 == 0 or index == len(futures):
                elapsed = time.monotonic() - started
                rate = index / elapsed if elapsed else 0.0
                _log(f"  {index}/{len(futures)} stored={_human(written)} {rate:.1f} ep/s")

    _log(f"done: {len(futures) - len(failures)} stored ({_human(written)}), {len(failures)} failed")
    for name, error in failures[:20]:
        _log(f"  {name}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
