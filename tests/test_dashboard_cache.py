"""The watermark gate, the render cache, and the event stream's pacing.

Together these are what stop an open browser tab from costing a full page render
every second, so each is tested for the property it contributes: the watermark
only holds still when nothing was written, the cache collapses repeat renders, and
the stream neither renders nor spins when the database is quiet.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard.cache import RenderCache, page_key
from kaggriculture_lab.dashboard.errors import DashboardBadRequest
from kaggriculture_lab.dashboard.events import StreamLimit, stream_dashboard
from kaggriculture_lab.dashboard.repository import DashboardRepository
from kaggriculture_lab.database import DatabaseTarget, connect_postgres
from kaggriculture_lab.storage import ExperimentStore
from tests.postgres_support import postgres_database


@pytest.fixture
def database(tmp_path: Path) -> DatabaseTarget:
    target = postgres_database(tmp_path, "cache")
    with ExperimentStore(target) as store:
        store.connection.execute(
            """
            INSERT INTO training_sessions (session_id, league, name, status, seed, config_json)
            VALUES ('s', 'cache', 'Cache league', 'RUNNING', 1, '{}')
            """
        )
    return target


@pytest.fixture
def connection(database: DatabaseTarget):
    live = connect_postgres(database, initialize=False)
    try:
        yield live
    finally:
        live.close()


# --------------------------------------------------------------------------- #
# Watermark
# --------------------------------------------------------------------------- #


def test_the_watermark_holds_still_while_nothing_is_written(connection):
    first = lq.watermark(connection)
    connection.execute("SELECT count(*) FROM training_sessions")
    connection.execute("SELECT count(*) FROM episodes")
    assert lq.watermark(connection) == first


def test_the_watermark_moves_on_any_write(connection):
    before = lq.watermark(connection)
    connection.execute(
        """
        INSERT INTO training_sessions (session_id, league, name, status, seed, config_json)
        VALUES ('s2', 'cache', 'Another', 'RUNNING', 2, '{}')
        """
    )
    assert lq.watermark(connection) != before


def test_the_watermark_moves_on_an_in_place_update(connection):
    """Ratings are upserted onto the same key, so a count-based watermark would miss them."""
    before = lq.watermark(connection)
    connection.execute("UPDATE training_sessions SET status = 'COMPLETE' WHERE session_id = 's'")
    assert lq.watermark(connection) != before


def test_the_watermark_moves_immediately_rather_than_on_a_stats_flush(connection):
    """No sleep: a page reloaded straight after a write must not be served stale.

    `pg_stat_user_tables` would still be reporting the previous counts here --
    the collector flushes about once a second -- which is why the watermark reads
    the transaction counter instead.
    """
    before = lq.watermark(connection)
    connection.execute(
        """
        INSERT INTO training_sessions (session_id, league, name, status, seed, config_json)
        VALUES ('s-now', 'cache', 'Immediate', 'RUNNING', 9, '{}')
        """
    )
    assert lq.watermark(connection) != before


# --------------------------------------------------------------------------- #
# Render cache
# --------------------------------------------------------------------------- #


def test_query_order_does_not_split_one_page_into_two_entries():
    assert page_key("/g", {"a": ["1"], "b": ["2"]}) == page_key("/g", {"b": ["2"], "a": ["1"]})
    assert page_key("/g", {}) == "/g"
    assert page_key("/g", {"a": ["1"]}) != page_key("/g", {"a": ["2"]})


def test_a_cached_page_is_returned_only_at_the_same_watermark():
    cache = RenderCache()
    cache.put("/", 5, "<main>one</main>")
    assert cache.get("/", 5) == "<main>one</main>"
    assert cache.get("/", 6) is None
    assert cache.stats()["hits"] == 1


def test_the_cache_evicts_least_recently_used_entries_over_its_limit():
    cache = RenderCache(max_bytes=40)
    cache.put("a", 1, "x" * 20)
    cache.put("b", 1, "y" * 20)
    cache.get("a", 1)  # touch a, so b is now the least recent
    cache.put("c", 1, "z" * 20)
    assert cache.get("b", 1) is None
    assert cache.get("a", 1) is not None
    assert cache.stats()["bytes"] <= 40


def test_a_document_larger_than_the_whole_cache_is_not_stored():
    cache = RenderCache(max_bytes=10)
    cache.put("a", 1, "x" * 100)
    assert cache.get("a", 1) is None
    assert cache.stats()["entries"] == 0


def test_replacing_an_entry_does_not_double_count_its_bytes():
    cache = RenderCache(max_bytes=1000)
    cache.put("a", 1, "x" * 100)
    cache.put("a", 2, "y" * 10)
    assert cache.stats()["bytes"] == 10
    assert cache.stats()["entries"] == 1


# --------------------------------------------------------------------------- #
# Event stream
# --------------------------------------------------------------------------- #


class FakeHandler:
    """Just enough of the HTTP handler for the stream to write into."""

    def __init__(self, repository: DashboardRepository):
        self.repository = repository
        self.render_cache = RenderCache()
        self.wfile = io.BytesIO()
        self.headers: dict[str, str] = {}
        self.status = None
        self.sent_text: list[tuple] = []

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers[key] = value

    def end_headers(self):
        pass

    def _send_text(self, status, value, *, head, headers=None):
        self.sent_text.append((status, value))


class CountingRepository(DashboardRepository):
    """Counts how many statements each stream tick issues."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statements = 0

    def connect_dedicated(self):
        outer = self

        class Counting:
            def __init__(self, connection):
                self._connection = connection

            def execute(self, query, parameters=None):
                outer.statements += 1
                return self._connection.execute(query, parameters)

            def __getattr__(self, name):
                return getattr(self._connection, name)

        from contextlib import contextmanager

        @contextmanager
        def lease():
            with DashboardRepository.connect_dedicated(self) as connection:
                yield Counting(connection)

        return lease()


def test_a_quiet_database_costs_one_statement_per_tick(database: DatabaseTarget):
    repository = CountingRepository(database)
    handler = FakeHandler(repository)
    try:
        stream_dashboard(
            handler,
            {"path": ["/"]},
            poll_seconds=0,
            max_poll_seconds=0,
            keepalive_seconds=1e9,
            max_ticks=1,
        )
        first_tick = repository.statements
        assert first_tick > 1  # the initial render reads real tables

        repository.statements = 0
        handler.wfile = io.BytesIO()
        stream_dashboard(
            handler,
            {"path": ["/"]},
            poll_seconds=0,
            max_poll_seconds=0,
            keepalive_seconds=1e9,
            max_ticks=4,
        )
        # First tick renders (the fragment is new to this stream); the three that
        # follow see an unchanged watermark and issue only that one query.
        assert repository.statements <= first_tick + 3
    finally:
        repository.close()


def test_a_write_produces_a_new_fragment(database: DatabaseTarget):
    repository = DashboardRepository(database)
    handler = FakeHandler(repository)
    writer = connect_postgres(database, initialize=False)
    try:
        stream_dashboard(
            handler, {"path": ["/"]}, poll_seconds=0, keepalive_seconds=1e9, max_ticks=1
        )
        assert handler.wfile.getvalue().count(b"event: dashboard.fragment") == 1

        writer.execute(
            """
            INSERT INTO training_sessions (session_id, league, name, status, seed, config_json)
            VALUES ('s3', 'cache', 'Third', 'RUNNING', 3, '{}')
            """
        )
        handler.wfile = io.BytesIO()
        stream_dashboard(
            handler, {"path": ["/"]}, poll_seconds=0, keepalive_seconds=1e9, max_ticks=1
        )
        body = handler.wfile.getvalue()
        assert b"event: dashboard.fragment" in body
        assert b"Third" in body
    finally:
        writer.close()
        repository.close()


def test_a_quiet_stream_sends_a_keepalive_on_wall_clock(database: DatabaseTarget):
    repository = DashboardRepository(database)
    handler = FakeHandler(repository)
    try:
        stream_dashboard(
            handler,
            {"path": ["/"]},
            poll_seconds=0,
            max_poll_seconds=0,
            keepalive_seconds=0,
            max_ticks=3,
        )
        assert b": keepalive" in handler.wfile.getvalue()
    finally:
        repository.close()


def test_streams_beyond_the_cap_are_refused(database: DatabaseTarget):
    repository = DashboardRepository(database)
    handler = FakeHandler(repository)
    limit = StreamLimit(limit=1)
    assert limit.claim()
    try:
        stream_dashboard(handler, {"path": ["/"]}, max_ticks=1, limit=limit)
        assert handler.sent_text and handler.sent_text[0][0] == 503
        assert handler.wfile.getvalue() == b""
    finally:
        limit.release()
        repository.close()


def test_a_finished_stream_gives_its_slot_back(database: DatabaseTarget):
    repository = DashboardRepository(database)
    handler = FakeHandler(repository)
    limit = StreamLimit(limit=1)
    try:
        stream_dashboard(
            handler, {"path": ["/"]}, poll_seconds=0, max_ticks=1, limit=limit
        )
        assert limit.live == 0
    finally:
        repository.close()


def test_the_stream_refuses_a_path_it_must_not_render(database: DatabaseTarget):
    repository = DashboardRepository(database)
    handler = FakeHandler(repository)
    try:
        with pytest.raises(DashboardBadRequest):
            stream_dashboard(handler, {"path": ["//evil.example/"]}, max_ticks=1)
    finally:
        repository.close()
