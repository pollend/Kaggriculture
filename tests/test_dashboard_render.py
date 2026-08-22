"""Markup tests that never open a connection.

Every page module splits `fetch` from `render`, so the markup can be driven from
hand-built rows. That makes the branches that are otherwise awkward to reach -- the
empty states, the trainer-only layout, an escaped name -- cheap to pin down, and
these run in milliseconds rather than waiting on PostgreSQL.
"""

from __future__ import annotations

import json

import pytest

from kaggriculture_lab.dashboard import routes
from kaggriculture_lab.dashboard.errors import DashboardBadRequest, DashboardNotFound
from kaggriculture_lab.dashboard.pages import episode as episode_page
from kaggriculture_lab.dashboard.pages import generation as generation_page
from kaggriculture_lab.dashboard.pages import session as session_page
from kaggriculture_lab.dashboard.pages import sessions as sessions_page


def _session_row(**overrides):
    row = {
        "session_id": "s/1",
        "league": "test-league",
        "name": "Test <script> League",
        "status": "COMPLETE",
        "seed": 17,
        "config_json": json.dumps({"rounds": 1}),
        "created_at": "2026-08-14 00:00:00",
    }
    return {**row, **overrides}


# --------------------------------------------------------------------------- #
# Sessions list
# --------------------------------------------------------------------------- #


def test_sessions_list_escapes_names_and_links_by_id():
    document = sessions_page.render(
        {
            "page": 1,
            "has_next": False,
            "sessions": [_session_row()],
            "rollups": {
                "s/1": {
                    "generations": 2,
                    "pairings": 10,
                    "completed_pairings": 9,
                    "episodes": 18,
                    "failed_episodes": 1,
                    "completed_rounds": 4,
                }
            },
            "leaders": {"s/1": {"display_name": "alpha", "conservative": 1300.5}},
        }
    )
    assert "Test &lt;script&gt; League" in document
    assert "<script> League" not in document
    # The id contains a slash, so the link has to be percent-encoded.
    assert 'href="/sessions/s%2F1"' in document
    assert "9 / 10" in document
    assert "alpha" in document


def test_sessions_list_shows_rounds_for_a_trainer_only_session():
    document = sessions_page.render(
        {
            "page": 1,
            "has_next": False,
            "sessions": [
                _session_row(
                    config_json=json.dumps(
                        {"pairing_mode": "trainer_only", "rounds_per_generation": 5}
                    )
                )
            ],
            "rollups": {
                "s/1": {
                    "generations": 3,
                    "pairings": 0,
                    "completed_pairings": 7,
                    "episodes": 12,
                    "failed_episodes": 0,
                    "completed_rounds": 11,
                }
            },
            "leaders": {},
        }
    )
    # Three generations at five rounds each, not a pairing fraction.
    assert "11 / 15 rounds" in document
    assert "7 pairings recorded" in document
    assert '<div class="progress"' not in document


def test_sessions_list_has_an_empty_state():
    document = sessions_page.render(
        {"page": 1, "has_next": False, "sessions": [], "rollups": {}, "leaders": {}}
    )
    assert "No league sessions have been recorded yet." in document
    assert "<table>" not in document


def test_sessions_list_falls_back_when_a_session_has_no_rollup():
    """A session created but never run has no rollup row at all."""
    document = sessions_page.render(
        {
            "page": 1,
            "has_next": False,
            "sessions": [_session_row()],
            "rollups": {},
            "leaders": {},
        }
    )
    assert "—" in document
    assert "0 / 0" in document


# --------------------------------------------------------------------------- #
# One session
# --------------------------------------------------------------------------- #


def _session_data(**overrides):
    config = overrides.pop("config", {"rounds": 1})
    data = {
        "session": _session_row(config_json=json.dumps(config)),
        "session_config": config,
        "generations": [
            {
                "generation_id": "s/1:g0",
                "generation_index": 0,
                "status": "COMPLETE",
                "created_at": "2026-08-14 00:00:00",
                "completed_at": "2026-08-14 01:00:00",
                "members": 4,
                "pairings": 6,
                "completed_pairings": 6,
                "episodes": 12,
                "failed_episodes": 0,
                "completed_rounds": 2,
            }
        ],
        "trend": [],
        "evaluation": {"evaluations": 4, "errors": 1, "reevaluation_requests": 2},
        "roster": [],
        "head_to_head": [],
        "completed_rounds": 0,
    }
    return {**data, **overrides}


def test_session_page_summarises_generations_and_evaluations():
    document = session_page.render(_session_data())
    assert 'href="/generations/s%2F1%3Ag0"' in document
    assert "1 / 4" in document  # hook errors over evaluations
    assert ">2<" in document  # re-evaluation requests
    assert "Ratings appear after a generation closes." in document  # empty chart


def test_session_page_counts_rounds_for_a_trainer_only_session():
    document = session_page.render(
        _session_data(
            config={"pairing_mode": "trainer_only", "rounds_per_generation": 3},
            completed_rounds=2,
        )
    )
    assert "2 / 3" in document  # one generation planned three rounds
    assert "Pairings recorded" in document


def test_head_to_head_is_relabelled_when_the_opponent_is_the_slot_itself():
    """A trainer-only session has no league opponent, so the bars are not a ranking.

    Every pairing is one learner against mirrors of its own checkpoints, and
    `_self_play_pool` deliberately does not filter those on `promoted`. The margin
    is therefore largest when the slot is doing worst: `market-free-cashherd-v1`
    ran `cashherd_02` at +18,148 and a 0.96 win rate against three checkpoints its
    own gate had refused, and the page presented that as the session's best record.
    """

    rows = [
        {"me": "learner", "opponent": "mirror", "opponent_name": "mirror",
         "pairings": 160, "mean_margin": 18148.0},
        {"me": "learner", "opponent": "learner", "opponent_name": "learner",
         "pairings": 160, "mean_margin": 0.0},
    ]

    self_play = session_page._head_to_head_section(rows, True)
    assert "Rollout matchups" in self_play
    assert "Not a quality ranking" in self_play
    assert "mirrors of its own checkpoints" in self_play
    assert "<h2>Head to head</h2>" not in self_play
    # Relabelled, never hidden -- the bars are still the record that was played.
    assert "<svg" in self_play

    league = session_page._head_to_head_section(rows, False)
    assert "<h2>Head to head</h2>" in league
    assert "Not a quality ranking" not in league
    # The default is the league reading, so no caller is silently switched over.
    assert session_page._head_to_head_section(rows) == league


def test_roster_prints_how_many_samples_the_money_stands_on():
    """Two measurements and eleven are not the same evidence, and only one row says so.

    The ratchet reference is a median over the slot's recent benchmark draws, and one
    arena series on these families is a draw rather than a number. `market-free-
    cashherd-v1` was configured for a window of eleven and ran every *promoting* slot
    at two, which is invisible if the page prints only the median.
    """

    row = {
        "model_id": "family.role@cashherd_03",
        "reseed_index": 0,
        "stage": "steady",
        "decay_steps": 0,
        "stall_updates": 1,
        "updates_since_reseed": 4,
        "latest_batch_seq": 50,
        "last_attempted_batch_seq": 40,
        "min_new_batches": 10,
        "trading_rate": 0.4,
        "last_money": 1750.93,
        "reference_samples": 2,
        "active_checkpoint_id": "abc123",
        "promotions": 4,
        "in_pool": False,
        "pool_score": None,
    }

    assert "2 samples" in session_page._roster_row(row)
    # A trainer that records no ratchet leaves the column blank rather than "0".
    assert "samples" not in session_page._roster_row({**row, "reference_samples": None})
    # And zero means the window has not run yet -- the incumbent's single stored
    # draw -- which is the least evidence on the page, not the absence of a reading.
    assert "single draw" in session_page._roster_row({**row, "reference_samples": 0})


def test_session_page_draws_the_rating_trend_when_there_is_one():
    document = session_page.render(
        _session_data(
            trend=[
                {"generation_index": 0, "champion_conservative": 1300, "median_conservative": 1200},
                {"generation_index": 1, "champion_conservative": 1400, "median_conservative": 1250},
            ]
        )
    )
    assert "<svg" in document
    assert 'aria-label="Rating trend"' in document
    assert "Champion conservative rating" in document


# --------------------------------------------------------------------------- #
# Generation panels
# --------------------------------------------------------------------------- #


def test_outcome_panel_is_empty_until_a_pairing_settles():
    assert "No pairings have been settled yet." in generation_page._outcome_panel({}, [])
    assert "No pairings have been settled yet." in generation_page._outcome_panel(
        {"settled_pairings": 0}, []
    )


def test_outcome_panel_reports_the_seat_gap():
    panel = generation_page._outcome_panel(
        {
            "settled_pairings": 4,
            "mean_paired_margin": 150.0,
            "median_paired_margin": 120.0,
            "paired_margin_stddev": 40.0,
            "min_paired_margin": -10.0,
            "max_paired_margin": 300.0,
            "drawn": 1,
        },
        [
            {"seat": 0, "episodes": 4, "mean_final_money": 3200, "mean_margin": 200, "win_rate": 0.75},
            {"seat": 1, "episodes": 4, "mean_final_money": 3000, "mean_margin": -200, "win_rate": 0.25},
        ],
    )
    assert "Seat-order bias" in panel
    assert "75.0%" in panel
    assert "-10 to 300" in panel


def test_outcome_panel_keeps_the_cards_when_no_episode_finished():
    """Settled pairings with no seat rows is possible mid-generation."""
    panel = generation_page._outcome_panel(
        {
            "settled_pairings": 1,
            "mean_paired_margin": 0,
            "median_paired_margin": 0,
            "paired_margin_stddev": None,
            "min_paired_margin": 0,
            "max_paired_margin": 0,
            "drawn": 1,
        },
        [],
    )
    assert "No completed episodes yet." in panel
    assert "Settled pairings" in panel


def test_agent_and_runtime_and_hook_panels_have_empty_states():
    assert "Per-agent statistics appear once episodes complete." in generation_page._agent_panel([])
    assert "Timings appear once episodes run." in generation_page._runtime_panel({}, [])
    assert "No post-episode evaluations were recorded." in generation_page._hook_panel({})


def test_runtime_panel_reports_a_failure_rate():
    panel = generation_page._runtime_panel(
        {
            "episodes": 4,
            "failed": 1,
            "total_wallclock": 120,
            "mean_wallclock": 30,
            "median_wallclock": 28,
            "p95_wallclock": 44,
            "max_wallclock": 45,
            "mean_steps": 700,
            "max_steps": 720,
        },
        [{"match_id": "m/1", "wallclock": 45, "steps": 720, "failed": True, "round_index": 2}],
    )
    assert "25.0%" in panel
    assert 'href="/episodes/m%2F1"' in panel
    assert "FAILED" in panel


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw", ["0", "-3", "abc", ""])
def test_page_numbers_must_be_positive_integers(raw):
    with pytest.raises(DashboardBadRequest):
        routes.parse_page({"page": [raw]})


def test_page_defaults_to_one():
    assert routes.parse_page({}) == 1


@pytest.mark.parametrize(
    "path",
    ["/sessions/", "/sessions/a/b", "/sessions/%00"],
)
def test_identifiers_reject_empty_slashed_and_nul_values(path):
    with pytest.raises(DashboardBadRequest):
        routes.identifier(path, "/sessions/")


def test_identifiers_are_percent_decoded():
    assert routes.identifier("/sessions/s%2F1", "/sessions/") == "s/1"
    assert routes.identifier("/episodes/m%2F1/replay", "/episodes/", "/replay") == "m/1"


def test_unknown_paths_are_not_found():
    with pytest.raises(DashboardNotFound):
        routes.resolve("/nope", {})


def test_generation_route_rejects_overlong_filters():
    with pytest.raises(DashboardBadRequest):
        routes.resolve("/generations/g", {"status": ["x" * 81]})


def test_resolve_returns_a_renderer_without_touching_the_database():
    render = routes.resolve("/sessions/s%2F1", {})
    assert callable(render)


@pytest.mark.parametrize(
    "target,allowed",
    [
        ("/", True),
        ("/generations/g?sort=round", True),
        ("//evil.example/", False),
        ("https://evil.example/", False),
        ("/events/dashboard", False),
        ("/dashboard.js", False),
        ("/dashboard.css", False),
        ("/healthz", False),
        ("/episodes/m/replay", False),
        ("/episodes/m/replay.json", False),
        ("/episodes/m", True),
    ],
)
def test_the_event_stream_only_follows_renderable_paths(target, allowed):
    assert routes.stream_allows(target) is allowed


# --------------------------------------------------------------------------- #
# Episode
# --------------------------------------------------------------------------- #


def _episode_data(**overrides):
    data = {
        "match_id": "m/1",
        "episode": {
            "match_id": "m/1",
            "seed": 17,
            "steps": 720,
            "wallclock": 1.5,
            "failed": False,
            "stage": "league:g",
            "orientation": 0,
            "pairing_id": "p",
            "generation_id": "s/1:g0",
            "revision_a": "r0",
            "revision_b": "r1",
            "agent_a_name": "alpha",
            "agent_a_strategy_id": "strategy-0",
            "agent_a_checkpoint_id": None,
            "agent_a_params_json": "{}",
            "agent_b_name": "beta",
            "agent_b_strategy_id": "strategy-1",
            "agent_b_checkpoint_id": None,
            "agent_b_params_json": "{}",
        },
        "players": [
            {
                "seat": seat,
                "status": "DONE",
                "final_money": 3200 - 200 * seat,
                "money_difference": 200 - 400 * seat,
                "reward": 1.0,
                "productive_action_percentage": 0.6,
                "movement_percentage": 0.3,
                "idle_percentage": 0.1,
                "plants_lost_inferred": 0,
                "final_shed_items": 1,
                "final_carried_items": 0,
                "metrics_json": "{}",
            }
            for seat in (0, 1)
        ],
        "evaluations": [],
        "has_replay": False,
    }
    return {**data, **overrides}


def test_episode_page_hides_the_replay_when_there_is_no_payload():
    document = episode_page.render(_episode_data())
    assert "Replay unavailable" in document
    assert "<iframe" not in document
    assert "No post-episode evaluations were recorded." in document


def test_episode_page_offers_the_replay_when_a_payload_exists():
    document = episode_page.render(_episode_data(has_replay=True))
    assert 'href="/episodes/m%2F1/replay.json"' in document
    assert 'src="/episodes/m%2F1/replay"' in document


def test_episode_page_orders_players_by_the_seat_they_played():
    """Orientation 1 means revision B held seat 0."""
    data = _episode_data()
    data["episode"] = {**data["episode"], "orientation": 1}
    document = episode_page.render(data)
    assert document.index("beta") < document.index("alpha")


def _roster_row(**overrides):
    row = {
        "model_id": "market_free@learner_03",
        "agent_name": "learner_03",
        "declared_model_id": "market_free",
        "active_checkpoint_id": "ckpt-7",
        "last_attempted_batch_seq": 32,
        "last_status": "PROMOTED",
        "cutoff_batch_seq": 40,
        "last_update_at": "2026-08-14 00:00:00",
        "min_new_batches": 4,
        "latest_batch_seq": 40,
        "reseed_index": 0,
        "updates_since_reseed": 8,
        "decay_steps": 1,
        "stall_updates": 2,
        "stage": "steady",
        "score": 900.0,
        "last_money": 4057.0,
        "trading_rate": 0.6171,
        "release_age": 5,
        "promotions": 2,
        "in_pool": False,
        "pool_score": None,
    }
    return {**row, **overrides}


def _batches_row(**overrides):
    row = {
        "model_id": "market_free@learner_03",
        "latest_batch_seq": 40,
        "last_attempted_batch_seq": 32,
        "min_new_batches": 4,
        "active_checkpoint_id": "ckpt-7",
    }
    return {**row, **overrides}


def test_session_page_links_every_model_that_trained_in_it():
    document = session_page.render(
        _session_data(roster=[_roster_row()])
    )
    assert 'href="/models/market_free%40learner_03"' in document
    assert "8 / 4" in document  # new batches against the update threshold
    assert "ckpt-7" in document


def test_the_roster_leads_with_the_slot_generation_marker():
    """The slot id survives every restart, so it alone cannot say which policy is in it."""

    document = session_page.render(
        _session_data(roster=[_roster_row(reseed_index=3)])
    )
    assert "#3" in document
    assert "reseed index" in document  # the legend explaining it


def test_the_roster_names_the_two_collapses_rather_than_leaving_two_numbers():
    idle = session_page.render(
        _session_data(
            roster=[_roster_row(trading_rate=0.0, last_money=2998.0)]
        )
    )
    broke = session_page.render(
        _session_data(
            roster=[_roster_row(trading_rate=0.0, last_money=5.6)]
        )
    )
    trading = session_page.render(
        _session_data(
            roster=[_roster_row(trading_rate=0.62, last_money=4057.0)]
        )
    )
    assert "idle" in idle and "broke" not in idle
    assert "broke" in broke and " idle" not in broke
    assert "idle" not in trading and "broke" not in trading


def test_the_roster_says_whether_a_release_reached_the_pool():
    out = session_page.render(
        _session_data(
            roster=[_roster_row(in_pool=True, pool_score=812.5)]
        )
    )
    assert "IN POOL" in out and "812.5" in out
    withheld = session_page.render(
        _session_data(roster=[_roster_row(in_pool=False)])
    )
    assert "IN POOL" not in withheld


def test_a_slot_with_no_release_says_so():
    document = session_page.render(
        _session_data(
            roster=[_roster_row(active_checkpoint_id=None, promotions=0)],
            models=[_batches_row(active_checkpoint_id=None)],
        )
    )
    assert "none" in document


def test_session_page_omits_the_models_panel_when_nothing_trained():
    assert "<h2>Models</h2>" not in session_page.render(_session_data())
