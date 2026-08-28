"""Convert mirrored Kaggle ladder replays into pinned demonstrations.

`scripts/pull_kaggle_episodes.py` mirrors the competition's public episode datasets
to `data/kaggle_episodes/<date>/<episode_id>.json.gz`, keeping each day's top slice
by `avg_score`. Those replays are the only place this policy's action space meets
agents that are neither itself nor the one donor: collection runs `opponents = []`,
so every rollout opponent is a mirror of the same lineage.

`clone.record_from_replay` already reads exactly this document -- the ladder replay
and `episode_payloads.replay_gzip` are the same `kaggle-environments` envelope, and
only the source of bytes differs. What this module adds is the walk over the mirror,
the seat filter, and a *cache*, because the conversion is far too expensive to redo
per update: a replay inflates to ~30 MB of JSON and the walk re-encodes 720 turns of
features, against an update that otherwise reads a few compact `.npz` artifacts.

So a pack is built once, offline:

    python -m agents.dqn_market_free_v3e.training.ladder --dry-run --limit 40
    python -m agents.dqn_market_free_v3e.training.ladder \\
        --out artifacts/dqn_market_free_v3e/demonstrations/ladder-001 --limit 256

and read back by `load_ladder_demonstrations`, which re-rolls every hidden state
under the network about to train on it (`clone.rehidden`). Nothing here is reachable
from the inference path, and nothing here writes to PostgreSQL: these episodes are
not league episodes, and minting `episode_players` rows for them would put ladder
opponents into every rating query and dashboard rollup that counts them.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agents.dqn_market_free_v3e.policy import GruCell, OrderQNetwork
from agents.dqn_market_free_v3e.training.clone import CloneStats, record_from_replay, rehidden
from agents.dqn_market_free_v3e.training.collect import (
    EpisodeRecord,
    load_episode_batch,
    save_episode_batch,
)
from agents.dqn_market_free_v3e.training.reward import relabel

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "data" / "kaggle_episodes"
#: Records per shard. The pack is a directory rather than one file so neither the
#: build nor the load ever holds the whole thing: a heavily-trading ladder seat is
#: ~10 MB of record, and 256 of them would be 2.5 GB resident at the moment of
#: `np.savez_compressed`.
SHARD_RECORDS = 16
SIDECAR = "pack.json"


@dataclass(frozen=True, slots=True)
class MirrorEpisode:
    """One mirrored replay, with the ladder rank that selected it."""

    episode_id: str
    date: str
    path: Path
    avg_score: float


def iter_mirror_episodes(
    root: Path | str = DEFAULT_ROOT,
    *,
    min_avg_score: float | None = None,
    days: int | None = None,
) -> Iterator[MirrorEpisode]:
    """Walk the mirror in descending ladder score.

    The manifests list every episode the day published (700-930 of them); the mirror
    holds only the slice `pull_kaggle_episodes.py` was asked for, so rows without a
    file on disk are skipped rather than treated as an error. `days` takes the most
    recent N dates, which is how a run picks the meta it is actually facing.
    """

    root = Path(root)
    dates = sorted(
        (entry.name for entry in root.iterdir() if entry.is_dir()),
        reverse=True,
    )
    if days is not None:
        dates = dates[: max(0, days)]
    episodes: list[MirrorEpisode] = []
    for date in dates:
        manifest = root / date / "manifest.csv"
        if not manifest.is_file():
            continue
        with manifest.open(newline="") as stream:
            for row in csv.DictReader(stream):
                score = float(row["avg_score"])
                if min_avg_score is not None and score < min_avg_score:
                    continue
                path = root / date / f"{row['episode_id']}.json.gz"
                if not path.is_file():
                    continue
                episodes.append(MirrorEpisode(str(row["episode_id"]), date, path, score))
    episodes.sort(key=lambda item: item.avg_score, reverse=True)
    yield from episodes


def seat_finals(replay: dict[str, Any]) -> list[float]:
    """Final money per seat, read the way the clone walk reads it.

    Through the observation's own `player` index rather than the seat number, as
    `clone.record_from_replay` does at its terminal read: the two agree in every
    replay seen so far, and disagreeing would rank a seat by the other's wallet.
    `rewards` carries the same numbers, but only when the episode ended DONE.
    """

    steps = replay.get("steps") or []
    if not steps:
        return []
    final = steps[-1]
    finals: list[float] = []
    for seat in range(len(final)):
        obs = final[seat].get("observation") or {}
        farms = obs.get("farms") or []
        # `or seat` would be wrong here: seat 1 reading `player: 0` is exactly the
        # disagreement this indirection exists to handle, and 0 is falsy.
        declared = obs.get("player", seat)
        player = int(seat if declared is None else declared)
        if not (0 <= player < len(farms)):
            player = seat
        finals.append(float(farms[player].get("money", 0) or 0) if 0 <= player < len(farms) else 0.0)
    return finals


def _as_dict(stats: CloneStats) -> dict[str, Any]:
    return {
        "episodes": stats.episodes,
        "decisions": stats.decisions,
        "orders_seen": stats.orders_seen,
        "orders_mapped": stats.orders_mapped,
        "mapped_fraction": stats.mapped_fraction,
        "orders_unrepresentable": stats.orders_unrepresentable,
        "orders_masked_infeasible": stats.orders_masked_infeasible,
        "orders_land": stats.orders_land,
    }


def _from_dict(payload: dict[str, Any]) -> CloneStats:
    return CloneStats(
        episodes=int(payload.get("episodes", 0)),
        decisions=int(payload.get("decisions", 0)),
        orders_seen=int(payload.get("orders_seen", 0)),
        orders_mapped=int(payload.get("orders_mapped", 0)),
        orders_unrepresentable=int(payload.get("orders_unrepresentable", 0)),
        orders_masked_infeasible=int(payload.get("orders_masked_infeasible", 0)),
        orders_land=int(payload.get("orders_land", 0)),
    )


def _merge(left: CloneStats, right: CloneStats) -> CloneStats:
    return CloneStats(
        episodes=left.episodes + right.episodes,
        decisions=left.decisions + right.decisions,
        orders_seen=left.orders_seen + right.orders_seen,
        orders_mapped=left.orders_mapped + right.orders_mapped,
        orders_unrepresentable=left.orders_unrepresentable + right.orders_unrepresentable,
        orders_masked_infeasible=left.orders_masked_infeasible + right.orders_masked_infeasible,
        orders_land=left.orders_land + right.orders_land,
    )


#: One GRU per worker process, built once. Only the *shape* matters here: every
#: `turn_hidden` this module writes is a placeholder that `load_ladder_demonstrations`
#: overwrites with a roll under the network that is about to train, so the weights
#: that produced it are irrelevant and are recorded in the sidecar as such.
_PLACEHOLDER_GRU: GruCell | None = None


def _placeholder_gru() -> GruCell:
    global _PLACEHOLDER_GRU
    if _PLACEHOLDER_GRU is None:
        _PLACEHOLDER_GRU = OrderQNetwork.quiet_start(seed=0).gru
    return _PLACEHOLDER_GRU


def _convert(job: tuple[str, str, str, float, float, float, tuple]) -> list[dict[str, Any]]:
    """Convert every qualifying seat of one replay. Runs in a worker process.

    Returns plain dicts rather than `EpisodeRecord`s so the parent can hold the
    provenance without the arrays when it is only counting (`--dry-run`).
    """

    episode_id, date, path, avg_score, min_final_money, gamma, players = job
    try:
        with gzip.open(path, "rb") as stream:
            replay = json.loads(stream.read())
    except (OSError, ValueError) as error:
        return [{"episode_id": episode_id, "error": str(error)}]
    finals = seat_finals(replay)
    names = [
        str(agent.get("Name", "")) for agent in (replay.get("info") or {}).get("Agents") or []
    ]
    statuses = replay.get("statuses") or []
    gru = _placeholder_gru()
    out: list[dict[str, Any]] = []
    for seat, final_money in enumerate(finals):
        if final_money < min_final_money:
            continue
        # One player's seats only, when asked. The blended pack fits an *average* of
        # every player it holds, and where two of them take different actions in
        # similar states no single action clears the large-margin hinge -- which is why
        # `mean_behavior_clone_loss` floors at the margin no matter how evenly the seats
        # were picked. A single-player pack is the control for that.
        if players and (names[seat] if seat < len(names) else "") not in players:
            continue
        # A crashed or ejected seat played a truncated game; its wallet is not a
        # demonstration of anything. Both seats of a ladder episode read DONE in
        # every replay reviewed, so this is a guard rather than a filter.
        if seat < len(statuses) and str(statuses[seat]) not in {"DONE", "ACTIVE"}:
            continue
        record, stats = record_from_replay(replay, seat, gamma, gru=gru)
        if record is None:
            continue
        out.append(
            {
                "episode_id": episode_id,
                "date": date,
                "seat": seat,
                "avg_score": avg_score,
                "final_money": final_money,
                "name": names[seat] if seat < len(names) else "",
                "stats": stats,
                "record": record,
            }
        )
    return out


def build_demo_pack(
    out: Path | str | None,
    *,
    root: Path | str = DEFAULT_ROOT,
    limit: int = 256,
    min_final_money: float = 0.0,
    min_avg_score: float | None = None,
    days: int | None = None,
    players: tuple[str, ...] | None = None,
    gamma: float = 0.999,
    workers: int | None = None,
    shard_records: int = SHARD_RECORDS,
    progress=None,
) -> tuple[CloneStats, list[dict[str, Any]]]:
    """Convert the top of the mirror into a pack directory.

    `out=None` converts and throws the records away, which is the dry run: it
    measures `mapped_fraction` and the money distribution without spending the disk
    or the time to compress. Returns the totals and the per-record provenance.
    """

    root = Path(root)
    destination = Path(out) if out is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        for stale in sorted(destination.glob("part-*.npz")):
            stale.unlink()
    seats_of = tuple(players or ())
    jobs = [
        (episode.episode_id, episode.date, str(episode.path), episode.avg_score,
         float(min_final_money), float(gamma), seats_of)
        for episode in iter_mirror_episodes(root, min_avg_score=min_avg_score, days=days)
    ]
    totals = CloneStats()
    provenance: list[dict[str, Any]] = []
    pending: list[EpisodeRecord] = []
    shards = 0
    errors: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal shards, pending
        if destination is None or not pending:
            pending = []
            return
        save_episode_batch(pending, destination / f"part-{shards:04d}.npz")
        shards += 1
        pending = []

    max_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) // 2)
    executor = ProcessPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
    try:
        mapped = (
            executor.map(_convert, jobs, chunksize=1)
            if executor is not None
            else map(_convert, jobs)
        )
        for results in mapped:
            for item in results:
                if "error" in item:
                    errors.append(item)
                    continue
                totals = _merge(totals, item["stats"])
                record = item.pop("record")
                stats = item.pop("stats")
                # The whole `CloneStats` per record, not a summary: shards are
                # written in ladder order and `demonstration_episodes` reads the
                # head of the pack, so the totals that describe *what an update
                # actually loaded* have to be re-summable over a prefix.
                item["stats"] = _as_dict(stats)
                item["decisions"] = stats.decisions
                item["mapped_fraction"] = stats.mapped_fraction
                provenance.append(item)
                if destination is not None:
                    pending.append(record)
                    if len(pending) >= shard_records:
                        flush()
                if progress is not None:
                    progress(len(provenance), limit, item)
            if len(provenance) >= limit:
                break
    finally:
        if executor is not None:
            # The generator above is abandoned on the `break`, so the pool has
            # outstanding futures for jobs nobody will read. Cancel them rather
            # than waiting out a queue sized by the whole mirror.
            executor.shutdown(wait=False, cancel_futures=True)
    flush()

    if destination is not None:
        sidecar = {
            "source": "kaggle_ladder_mirror",
            "root": str(root),
            "records": len(provenance),
            "shards": shards,
            "shard_records": shard_records,
            "gamma": gamma,
            "min_final_money": min_final_money,
            "players": list(players) if players else None,
            "min_avg_score": min_avg_score,
            "days": days,
            # Written by `OrderQNetwork.quiet_start(seed=0)` and meaningless:
            # `load_ladder_demonstrations` re-rolls every one of them under the
            # network that is about to train. Recorded so a reader who finds the
            # column does not mistake it for a policy's trajectory.
            "turn_hidden": "placeholder; re-rolled by clone.rehidden on load",
            "stats": _as_dict(totals),
            "errors": errors,
            "provenance": provenance,
        }
        (destination / SIDECAR).write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return totals, provenance


def pack_stats(pack: Path | str, limit: int | None = None) -> CloneStats:
    """The conversion totals recorded when the pack was built.

    `limit` re-sums over the first `limit` records instead of returning the
    pack-wide totals, because that is what an update loading the head of a larger
    pack actually holds -- and a `demonstrations.episodes` that disagreed with the
    number of pinned records would be a metrics row nobody could reconcile.
    """

    payload = json.loads((Path(pack) / SIDECAR).read_text())
    if limit is None:
        return _from_dict(payload.get("stats") or {})
    provenance = payload.get("provenance") or []
    chosen = limit if isinstance(limit, (list, tuple, set)) else range(min(limit, len(provenance)))
    totals = CloneStats()
    for index in chosen:
        if 0 <= int(index) < len(provenance):
            totals = _merge(totals, _from_dict(provenance[int(index)].get("stats") or {}))
    return totals


def stratified_indices(provenance: list[dict[str, Any]], limit: int) -> list[int]:
    """`limit` record indices spread across competitors, best seat of each first.

    Round-robin over players rather than the head of the money order, and that is the
    whole point. The pack is built in descending ladder order and `ladder-001`'s head
    is half two players -- 34 of the first 128 seats are one competitor and 30 another
    -- so the clone was fitting a blend weighted by who happened to rank highest, and
    two players supplied 50% of every pinned batch.

    A blend is still what it fits; the round-robin only makes it an even one. Where two
    competitors disagree at a state, no single action satisfies the large-margin hinge,
    and the loss floors at the margin no matter how the seats were chosen. What this
    buys is coverage: `limit` seats now reach as many distinct strategies as the pack
    holds before any player contributes a second.

    Ties inside a player break on money, so a player's best season is taken first.
    """

    by_player: dict[str, list[int]] = {}
    for index, item in enumerate(provenance):
        by_player.setdefault(str(item.get("name") or ""), []).append(index)
    for indices in by_player.values():
        indices.sort(
            key=lambda index: -float(provenance[index].get("final_money") or 0.0)
        )
    # Players in descending order of their best seat, so a short `limit` still takes
    # the strongest strategies rather than whichever name sorts first.
    order = sorted(
        by_player,
        key=lambda name: -float(provenance[by_player[name][0]].get("final_money") or 0.0),
    )
    chosen: list[int] = []
    depth = 0
    while len(chosen) < limit:
        added = False
        for name in order:
            if depth < len(by_player[name]):
                chosen.append(by_player[name][depth])
                added = True
                if len(chosen) >= limit:
                    break
        if not added:
            break
        depth += 1
    return chosen


def load_ladder_demonstrations(
    pack: Path | str,
    *,
    gru: GruCell,
    limit: int = 96,
    stratified: bool = True,
) -> tuple[list[EpisodeRecord], CloneStats]:
    """Read a pack and re-roll its hidden states under `gru`.

    Signature deliberately mirrors `clone.load_demonstrations` so the updater's
    branch is the source and nothing downstream of `buffer.add(..., pinned=True)`
    has to know which one it got.

    `gru` is required for the same reason `record_from_replay` requires it: a stored
    hidden state written by another network passes every width and stride check
    `load_episode_batch` makes, and then feeds mid-episode chunks a state the policy
    never occupied.
    """

    directory = Path(pack)
    if not directory.is_dir():
        raise FileNotFoundError(f"demonstration pack {directory} is not a directory")

    # Which records to take. `stratified` spreads them over competitors; the head of
    # the ladder order is the fallback for a pack whose sidecar carries no names.
    #
    # Relabelled on the way in, for the same reason the hidden state is re-rolled:
    # a shard stores whatever `reward.py` said when the pack was built, and the
    # pack outlives that. `ladder-001` carried the old shaped stream at
    # `MONEY_SCALE = 20_000` into a session training on flat terminal labels at
    # 250_000 -- a quarter of every TD batch fitted to a different objective at a
    # 12.5x scale gap, and no metric said so.
    wanted: set[int] | None = None
    picked: list[int] = []
    if stratified:
        payload = json.loads((directory / SIDECAR).read_text())
        provenance = payload.get("provenance") or []
        if provenance and any(item.get("name") for item in provenance):
            picked = stratified_indices(provenance, limit)
            wanted = set(picked)

    records: list[EpisodeRecord] = []
    if wanted is None:
        # Shards are written in descending ladder order, so taking the head of the
        # first shards is taking the best-ranked episodes rather than an arbitrary
        # subset -- which is what makes `demonstration_episodes` a meaningful dial
        # against a pack built once at a larger limit.
        for shard in sorted(directory.glob("part-*.npz")):
            for record in load_episode_batch(shard):
                records.append(rehidden(relabel(record), gru))
                if len(records) >= limit:
                    return records, pack_stats(directory, len(records))
    else:
        # Provenance is written in shard order, so a record index is a position in the
        # concatenation of the shards. Only the wanted ones are re-hidden, which is the
        # expensive half -- a GRU roll over every turn of every episode.
        by_index: dict[int, EpisodeRecord] = {}
        cursor = 0
        for shard in sorted(directory.glob("part-*.npz")):
            for record in load_episode_batch(shard):
                if cursor in wanted:
                    by_index[cursor] = rehidden(relabel(record), gru)
                cursor += 1
            if len(by_index) >= len(wanted):
                break
        records = [by_index[index] for index in picked if index in by_index]
        if records:
            return records, pack_stats(directory, [index for index in picked if index in by_index])
    if not records:
        raise ValueError(f"demonstration pack {directory} holds no episodes")
    return records, pack_stats(directory, len(records))


def _report(totals: CloneStats, provenance: list[dict[str, Any]]) -> str:
    money = np.asarray([item["final_money"] for item in provenance], dtype=np.float64)
    lines = [
        f"records                  {len(provenance)}",
        f"decisions                {totals.decisions}",
        f"orders_seen              {totals.orders_seen}",
        f"orders_mapped            {totals.orders_mapped}",
        f"mapped_fraction          {totals.mapped_fraction:.4f}",
        f"orders_unrepresentable   {totals.orders_unrepresentable}",
        f"orders_masked_infeasible {totals.orders_masked_infeasible}",
        f"orders_land              {totals.orders_land}",
    ]
    if money.size:
        lines += [
            f"final_money min/med/max  {money.min():,.0f} / "
            f"{np.median(money):,.0f} / {money.max():,.0f}",
            f"final_money p10/p90      {np.percentile(money, 10):,.0f} / "
            f"{np.percentile(money, 90):,.0f}",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, help="pack directory to write")
    parser.add_argument("--dry-run", action="store_true", help="convert and report, write nothing")
    parser.add_argument("--limit", type=int, default=256, help="seat-trajectories to keep")
    parser.add_argument("--min-money", type=float, default=0.0)
    parser.add_argument("--min-avg-score", type=float)
    parser.add_argument("--days", type=int, help="most recent N mirrored days only")
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--shard-records", type=int, default=SHARD_RECORDS)
    args = parser.parse_args(argv)

    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run")

    def progress(done: int, target: int, item: dict[str, Any]) -> None:
        print(
            f"[{done:>4}/{target}] {item['date']} {item['episode_id']} seat {item['seat']} "
            f"money {item['final_money']:>9,.0f} mapped {item['mapped_fraction']:.3f} "
            f"{item['name']}",
            file=sys.stderr,
            flush=True,
        )

    totals, provenance = build_demo_pack(
        None if args.dry_run else args.out,
        root=args.root,
        limit=args.limit,
        min_final_money=args.min_money,
        min_avg_score=args.min_avg_score,
        days=args.days,
        gamma=args.gamma,
        workers=args.workers,
        shard_records=args.shard_records,
        progress=progress,
    )
    print(_report(totals, provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
