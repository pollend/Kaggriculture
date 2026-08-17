"""Safe post-episode evaluation hooks executed by the league parent process."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EpisodeHookContext:
    session_id: str
    generation_id: str
    pairing_id: str
    match_id: str
    revision_id: str
    opponent_revision_id: str
    seat: int


class EpisodeReader:
    """Narrow view over one persisted episode."""

    def __init__(
        self,
        connection: Any,
        match_id: str,
        seat: int,
        replay: dict[str, Any] | None = None,
    ):
        self._connection = connection
        self.match_id = match_id
        self.seat = seat
        self._replay = replay

    def summary(self) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT status, failed, reward, final_money, opponent_money,
                   money_difference, win, metrics_json
            FROM episode_players WHERE match_id = %s AND seat = %s
            """,
            [self.match_id, self.seat],
        ).fetchone()
        if row is None:
            raise ValueError(f"episode player not found: {self.match_id} seat {self.seat}")
        return {
            "status": row[0],
            "failed": row[1],
            "reward": row[2],
            "final_money": row[3],
            "opponent_money": row[4],
            "money_difference": row[5],
            "win": row[6],
            "metrics": json.loads(row[7]),
        }

    def replay(self) -> dict[str, Any]:
        """The full replay for this episode.

        Every episode keeps one, trainer rollouts included, so an absent payload
        is a real fault rather than a storage policy and is raised as such. It
        was briefly otherwise: rollouts were recorded without a payload to save
        disk, the hook fired for them anyway, and every replay-reading evaluator
        turned those episodes into `episode_evaluations` ERROR rows.

        The hook fires immediately after the episode was persisted, in the
        process that played it, so the caller hands the still-live replay in
        rather than making this read back the multi-megabyte payload it just
        compressed. Otherwise the first call fetches and decompresses once and
        both seats share the result -- the two hooks for one episode used to
        gunzip and re-parse the same ~4 MB of JSON twice.
        """

        if self._replay is None:
            row = self._connection.execute(
                "SELECT replay_gzip FROM episode_payloads WHERE match_id = %s",
                [self.match_id],
            ).fetchone()
            if row is None or row[0] is None:
                raise ValueError(f"replay payload not found: {self.match_id}")
            self._replay = json.loads(gzip.decompress(bytes(row[0])))
        return self._replay

    def fetchall(
        self, sql: str, parameters: list[Any] | tuple[Any, ...] = ()
    ) -> list[tuple[Any, ...]]:
        """Run one query across the whole store.

        An evaluator often needs more than the episode it was fired for -- "is
        this revision trending down over its last N games" cannot be answered
        from a single match.
        """
        return self._connection.execute(sql, list(parameters)).fetchall()

    def fetchone(
        self, sql: str, parameters: list[Any] | tuple[Any, ...] = ()
    ) -> tuple[Any, ...] | None:
        rows = self.fetchall(sql, parameters)
        return rows[0] if rows else None

    def history(self, revision_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """This revision's most recent completed episodes, newest first.

        Excludes the current match so a caller can compare "now" against "before"
        without double-counting it.
        """
        current = self.fetchone(
            """
            SELECT g.session_id, g.generation_index, lp.round_index,
                   lp.pairing_id, pe.orientation
            FROM pairing_episodes pe
            JOIN league_pairings lp USING (pairing_id)
            JOIN generations g USING (generation_id)
            WHERE pe.match_id = %s
            """,
            [self.match_id],
        )
        if current is None:
            # Non-league runs have no scheduling metadata. Keep their stable
            # legacy ordering, but league hooks always use schedule chronology.
            rows = self.fetchall(
                """
                SELECT ep.match_id, ep.seed, ep.seat, ep.failed, ep.final_money,
                       ep.opponent_money, ep.money_difference, ep.win
                FROM episode_players ep
                WHERE ep.strategy_id = %s AND ep.match_id != %s
                ORDER BY ep.match_id DESC
                LIMIT %s
                """,
                [revision_id, self.match_id, limit],
            )
        else:
            session_id, generation_index, round_index, pairing_id, orientation = current
            rows = self.fetchall(
                """
                SELECT ep.match_id, ep.seed, ep.seat, ep.failed, ep.final_money,
                       ep.opponent_money, ep.money_difference, ep.win
                FROM episode_players ep
                JOIN pairing_episodes pe USING (match_id)
                JOIN league_pairings lp USING (pairing_id)
                JOIN generations g USING (generation_id)
                WHERE ep.strategy_id = %s AND g.session_id = %s
                  AND (g.generation_index, lp.round_index, lp.pairing_id, pe.orientation)
                      < (%s, %s, %s, %s)
                ORDER BY g.generation_index DESC, lp.round_index DESC,
                         lp.pairing_id DESC, pe.orientation DESC
                LIMIT %s
                """,
                [
                    revision_id,
                    session_id,
                    generation_index,
                    round_index,
                    pairing_id,
                    orientation,
                    limit,
                ],
            )
        keys = ("match_id", "seed", "seat", "failed", "final_money",
                "opponent_money", "money_difference", "win")
        return [dict(zip(keys, row)) for row in rows]


EpisodeEvaluator = Callable[[EpisodeHookContext, EpisodeReader], dict[str, Any]]


#: Fingerprints by source path, keyed on what a rewrite would change. The digest
#: is part of every `episode_evaluations` row's identity, so it has to track edits
#: to the evaluator -- but it fired twice per episode, re-reading and re-hashing
#: the whole source file each time, for a file that changes between runs at most.
_FINGERPRINTS: dict[str, tuple[tuple[int, int], str]] = {}


def _source_fingerprint(evaluator: EpisodeEvaluator) -> str:
    source_file = inspect.getsourcefile(evaluator)
    if not source_file:
        return hashlib.sha256(inspect.getsource(evaluator).encode()).hexdigest()
    stat = Path(source_file).stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _FINGERPRINTS.get(source_file)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    digest = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    _FINGERPRINTS[source_file] = (stamp, digest)
    return digest


def load_evaluator(path: str) -> tuple[EpisodeEvaluator, str]:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError("episode evaluator must use module:attribute syntax")
    evaluator = getattr(importlib.import_module(module_name), attribute)
    if not callable(evaluator):
        raise TypeError(f"episode evaluator is not callable: {path}")
    return evaluator, _source_fingerprint(evaluator)
