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

    def __init__(self, connection: Any, match_id: str, seat: int):
        self._connection = connection
        self.match_id = match_id
        self.seat = seat

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
        """

        row = self._connection.execute(
            "SELECT replay_gzip FROM episode_payloads WHERE match_id = %s", [self.match_id]
        ).fetchone()
        if row is None or row[0] is None:
            raise ValueError(f"replay payload not found: {self.match_id}")
        return json.loads(gzip.decompress(bytes(row[0])))

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


def load_evaluator(path: str) -> tuple[EpisodeEvaluator, str]:
    module_name, separator, attribute = path.partition(":")
    if not separator:
        raise ValueError("episode evaluator must use module:attribute syntax")
    evaluator = getattr(importlib.import_module(module_name), attribute)
    if not callable(evaluator):
        raise TypeError(f"episode evaluator is not callable: {path}")
    source_file = inspect.getsourcefile(evaluator)
    if source_file:
        digest = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    else:
        digest = hashlib.sha256(inspect.getsource(evaluator).encode()).hexdigest()
    return evaluator, digest
