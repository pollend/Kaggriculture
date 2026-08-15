"""The failure triage page.

Its whole value is turning a count into a cause, so the tests are about the
grouping: a traceback must collapse to its exception line, and forty-seven copies
of one problem must read as one row with a count of forty-seven.
"""

from __future__ import annotations

import pytest

from kaggriculture_lab.dashboard import routes
from kaggriculture_lab.dashboard.pages import failures as failures_page

TRACEBACK = '''Traceback (most recent call last):
  File "/home/user/project/kaggriculture_lab/model_trainers.py", line 747, in run_round_hooks
    hook(context, ModelTrainingReader(store), writer)
  File "/home/user/project/agents/dqn_market_free/train.py", line 31, in on_round
    self._collector.collect(context, reader, writer)
RuntimeError: every free-market rollout episode failed'''

OTHER_TRACEBACK = '''Traceback (most recent call last):
  File "/home/user/project/other.py", line 12, in run
    thing()
ValueError: bad shape (3, 4)'''


def _data(**overrides):
    data = {
        "session": {"session_id": "s/1", "name": "Test League"},
        "hooks": [],
        "updates": [],
        "episodes": [],
        "pairings": [],
    }
    return {**data, **overrides}


def test_a_traceback_collapses_to_its_exception_line():
    assert failures_page.cause_of(TRACEBACK) == (
        "RuntimeError: every free-market rollout episode failed"
    )
    assert failures_page.cause_of(OTHER_TRACEBACK) == "ValueError: bad shape (3, 4)"


def test_a_one_line_error_is_its_own_cause():
    assert failures_page.cause_of("connection refused") == "connection refused"


def test_a_missing_error_text_is_named_rather_than_blank():
    assert failures_page.cause_of(None) == "no error text recorded"
    assert failures_page.cause_of("") == "no error text recorded"
    assert failures_page.cause_of("   \n  \n") == "no error text recorded"


def test_a_very_long_cause_is_truncated():
    cause = failures_page.cause_of("RuntimeError: " + "x" * 500)
    assert len(cause) <= failures_page.MAX_CAUSE_LENGTH
    assert cause.endswith("…")


def _section(document: str, heading: str) -> str:
    """The markup between one <h2> and the next."""
    start = document.index(f"<h2>{heading}</h2>")
    end = document.find("<h2>", start + 4)
    return document[start : end if end > 0 else len(document)]


def test_many_copies_of_one_problem_read_as_one_row():
    document = failures_page.render(
        _data(
            hooks=[
                {"model_id": "m", "error_text": TRACEBACK, "round_index": index}
                for index in range(47)
            ]
        )
    )
    assert ">47<" in document
    # One row, not forty-seven. (The cause string also appears inside the folded
    # traceback, since it is the line that traceback ends on.)
    section = _section(document, "Round hook errors")
    body = section[section.index("<tbody>") : section.index("</tbody>")]
    assert body.count("<tr>") == 1
    assert "Distinct causes" in document
    assert ">1<" in document


def test_distinct_causes_are_listed_most_frequent_first():
    document = failures_page.render(
        _data(
            updates=[
                {"model_id": "m", "error_text": OTHER_TRACEBACK, "update_id": "u1"},
                {"model_id": "m", "error_text": TRACEBACK, "update_id": "u2"},
                {"model_id": "m", "error_text": TRACEBACK, "update_id": "u3"},
            ]
        )
    )
    assert document.index("RuntimeError:") < document.index("ValueError:")


def test_the_full_traceback_stays_one_click_away():
    document = failures_page.render(_data(hooks=[{"model_id": "m", "error_text": TRACEBACK}]))
    assert "<details>" in document
    assert "run_round_hooks" in document  # the frames are kept, just folded


def test_failed_episodes_link_to_the_episode_and_its_generation():
    document = failures_page.render(
        _data(
            episodes=[
                {
                    "generation_id": "s/1:g0",
                    "generation_index": 0,
                    "match_id": "m/1",
                    "stage": "league:g0",
                    "steps": 12,
                    "round_index": 3,
                    "pairing_kind": "population",
                }
            ]
        )
    )
    assert 'href="/episodes/m%2F1"' in document
    assert 'href="/generations/s%2F1%3Ag0"' in document


def test_a_clean_session_says_so_in_every_section():
    document = failures_page.render(_data())
    for message in (
        "No round hooks failed.",
        "No model updates failed.",
        "No episode crashed.",
        "No pairing failed.",
    ):
        assert message in document


def test_a_capped_episode_list_says_so():
    episodes = [
        {
            "generation_id": "g",
            "generation_index": 0,
            "match_id": f"m{index}",
            "stage": "league:g",
            "steps": 1,
            "round_index": 0,
            "pairing_kind": "population",
        }
        for index in range(200)
    ]
    assert "Showing the 200 most recent" in failures_page.render(_data(episodes=episodes))


def test_the_route_matches_failures_before_the_session_page():
    render = routes.resolve("/sessions/s%2F1/failures", {})
    assert callable(render)
    # And a trailing slash normalises to the same route rather than 404ing.
    assert callable(routes.resolve("/sessions/s%2F1/failures/", {}))


def test_the_failures_route_still_rejects_a_bad_identifier():
    with pytest.raises(Exception):
        routes.resolve("/sessions//failures", {})
