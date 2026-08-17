"""The model page's rendering, driven from hand-built rows.

The page exists to answer "is the gate shipping anything?", so the tests here are
mostly about the parts that could quietly lie: a rejected candidate that looks like
a promotion, a truncated history that looks complete, a metrics blob whose keys
differ per agent family.
"""

from __future__ import annotations

import json

from kaggriculture_lab.dashboard.html import _status
from kaggriculture_lab.dashboard.pages import model as model_page


def _model(**overrides):
    row = {
        "model_id": "market_free@dqn",
        "family_id": "market_free",
        "family_name": "Market Free",
        "agent_type": "dqn",
        "agent_name": "dqn_market_free",
        "declared_model_id": "market_free",
        "trainer_path": "agents.dqn_market_free.train:trainer",
        "min_new_batches": 4,
        "config_json": json.dumps({"gamma": 0.99}),
        "active_checkpoint_id": "ckpt-3",
        "last_attempted_batch_seq": 30,
        "latest_batch_seq": 36,
        "created_at": "2026-08-01 00:00:00",
        "updated_at": "2026-08-14 00:00:00",
    }
    return {**row, **overrides}


def _update(seq: int, status: str, **overrides):
    row = {
        "update_id": f"u{seq}",
        "parent_checkpoint_id": None,
        "cutoff_batch_seq": seq,
        "new_batch_count": 4,
        "dataset_sha256": "d",
        "status": status,
        "candidate_checkpoint_id": f"ckpt-{seq}",
        "metrics_json": json.dumps({"mean_loss": 1.0 / seq, "gradient_steps": seq * 100}),
        "gate_json": json.dumps({"passed": status == "PROMOTED"}),
        "error_text": None,
        "started_at": "2026-08-10 00:00:00",
        "completed_at": "2026-08-10 00:01:00",
    }
    return {**row, **overrides}


def _data(**overrides):
    data = {
        "model": _model(),
        "active_checkpoint": {
            "checkpoint_id": "ckpt-3",
            "model_id": "market_free@dqn",
            "update_id": "u3",
            "parent_checkpoint_id": "ckpt-2",
            "trainer_artifact_path": "trainer.pt",
            "inference_artifact_path": "policy.pt",
            "artifact_sha256": "aaa",
            "dataset_sha256": "bbb",
            "metrics_json": json.dumps({"mean_loss": 0.3}),
            "gate_json": json.dumps({"passed": True}),
            "promoted": True,
            "created_at": "2026-08-12 00:00:00",
        },
        "updates": [_update(1, "PROMOTED"), _update(2, "REJECTED"), _update(3, "PROMOTED")],
        "checkpoints": [
            {
                "checkpoint_id": "ckpt-1",
                "update_id": "u1",
                "parent_checkpoint_id": None,
                "promoted": True,
                "metrics_json": "{}",
                "gate_json": "{}",
                "dataset_sha256": "d",
                "created_at": "2026-08-10 00:00:00",
            },
            {
                "checkpoint_id": "ckpt-3",
                "update_id": "u3",
                "parent_checkpoint_id": "ckpt-1",
                "promoted": True,
                "metrics_json": "{}",
                "gate_json": "{}",
                "dataset_sha256": "d",
                "created_at": "2026-08-12 00:00:00",
            },
        ],
        "evaluations": [
            {
                "evaluation_id": "e1",
                "update_id": "u3",
                "evaluation_kind": "incumbent_head_to_head",
                "opponent_checkpoint_id": "ckpt-1",
                "metrics_json": json.dumps({"margin": 120}),
                "passed": True,
                "created_at": "2026-08-12 00:00:00",
            }
        ],
        "batch_sources": [
            {
                "source_kind": "free_market_rollout",
                "batches": 36,
                "consumed": 30,
                "pending": 6,
                "samples": 7200,
            }
        ],
        "batches": [],
        "hooks": [
            {
                "session_id": "s/1",
                "scope": "round",
                "status": "COMPLETE",
                "hooks": 12,
                "latest_generation_index": 4,
                "latest_round_index": 7,
                "batches_recorded": 12,
            }
        ],
    }
    return {**data, **overrides}


def test_a_rejection_does_not_look_like_a_promotion():
    """Both used to render the same yellow pill, which made the column meaningless."""
    promoted = _status("PROMOTED")
    rejected = _status("REJECTED")
    assert promoted != rejected
    assert "good" in promoted
    assert "bad" not in rejected  # a rejection is the gate working, not a failure


def test_the_page_reports_progress_towards_the_next_update():
    document = model_page.render(_data())
    assert "Next update" in document
    # Six new batches against a threshold of four: the update is due.
    assert "4 / 4" in document
    assert "An update fires once" in document


def _section(document: str, heading: str) -> str:
    """The markup between one <h2> and the next, so ordering assertions are local."""
    start = document.index(f"<h2>{heading}</h2>")
    end = document.find("<h2>", start + 1)
    return document[start : end if end > 0 else len(document)]


def test_the_gate_timeline_colours_each_outcome():
    document = model_page.render(_data())
    section = _section(document, "Update gate")
    assert 'aria-label="Update outcomes by batch sequence"' in section
    assert section.count("<title>") == 3  # one tooltip per update
    assert model_page.UPDATE_STATUS_INK["REJECTED"] in section


def test_the_timeline_is_empty_rather_than_blank_before_the_first_update():
    document = model_page.render(_data(updates=[]))
    assert "The gate has not run yet." in document
    assert "No update has been attempted yet." in document


def test_metric_keys_are_discovered_rather_than_hard_coded():
    """dqn writes mean_loss, sp_psro writes beta_loss, mixed_farm writes core_mae."""
    updates = [
        _update(1, "PROMOTED", metrics_json=json.dumps({"beta_loss": 2.0, "nu_loss": 1.0})),
        _update(2, "REJECTED", metrics_json=json.dumps({"beta_loss": 1.5, "nu_loss": 0.9})),
    ]
    document = model_page.render(_data(updates=updates))
    assert "beta_loss" in document
    assert "nu_loss" in document


def test_a_metric_seen_only_once_is_not_plotted():
    updates = [
        _update(1, "PROMOTED", metrics_json=json.dumps({"mean_loss": 1.0, "one_off": 5})),
        _update(2, "PROMOTED", metrics_json=json.dumps({"mean_loss": 0.5})),
    ]
    document = model_page.render(_data(updates=updates))
    assert 'aria-label="mean_loss by batch sequence"' in document
    assert 'aria-label="one_off by batch sequence"' not in document


def test_non_numeric_metrics_are_skipped():
    updates = [
        _update(1, "PROMOTED", metrics_json=json.dumps({"lineage_reset": True, "curve": {"a": 1}})),
        _update(2, "PROMOTED", metrics_json=json.dumps({"lineage_reset": False, "curve": {"a": 2}})),
    ]
    document = model_page.render(_data(updates=updates))
    assert "Numeric update metrics appear" in document


def test_loss_metrics_are_plotted_before_bookkeeping_counters():
    updates = [
        _update(seq, "PROMOTED", metrics_json=json.dumps({
            "zzz_counter": seq,
            "mean_loss": 1.0 / seq,
            "own_mean_return": seq * 2,
            "aaa_counter": seq,
        }))
        for seq in (1, 2, 3)
    ]
    # Scoped to the chart grid: the update table below dumps the same metric names
    # inside its JSON blocks, in whatever order the trainer wrote them.
    section = _section(model_page.render(_data(updates=updates)), "Update metrics")
    assert section.index("mean_loss") < section.index("own_mean_return")
    assert section.index("own_mean_return") < section.index("aaa_counter")


def test_a_truncated_update_table_says_so():
    updates = [_update(seq, "REJECTED") for seq in range(1, 61)]
    document = model_page.render(_data(updates=updates))
    assert "Showing the 40 most recent of 60 updates" in document


def test_a_short_history_carries_no_truncation_note():
    assert "Showing the" not in model_page.render(_data())


def test_the_lineage_marks_the_active_checkpoint_and_its_root():
    document = model_page.render(_data())
    assert "ACTIVE" in document
    assert "root" in document


def test_a_parent_outside_the_page_is_called_out():
    checkpoints = [
        {
            "checkpoint_id": "ckpt-9",
            "update_id": "u9",
            "parent_checkpoint_id": "ckpt-long-ago",
            "promoted": True,
            "metrics_json": "{}",
            "gate_json": "{}",
            "dataset_sha256": "d",
            "created_at": "2026-08-13 00:00:00",
        }
    ]
    document = model_page.render(_data(checkpoints=checkpoints))
    assert "outside this page" in document


def test_the_gate_flag_falls_back_to_status_when_the_trainer_writes_no_passed_field():
    assert model_page._gate_flag({"status": "PROMOTED", "gate_json": "{}"}) == "yes"
    assert model_page._gate_flag({"status": "REJECTED", "gate_json": "{}"}) == "—"
    assert model_page._gate_flag({"status": "REJECTED", "gate_json": '{"passed": false}'}) == "no"
    assert model_page._gate_flag({"status": "ERROR", "gate_json": "not json"}) == "—"


def test_batches_are_split_at_the_watermark():
    document = model_page.render(_data())
    assert "free_market_rollout" in document
    assert ">30<" in document  # consumed
    assert ">6<" in document  # pending


def test_hooks_link_back_to_their_session():
    document = model_page.render(_data())
    assert 'href="/sessions/s%2F1"' in document


def test_every_section_has_an_empty_state():
    document = model_page.render(
        _data(
            active_checkpoint=None,
            updates=[],
            checkpoints=[],
            evaluations=[],
            batch_sources=[],
            hooks=[],
        )
    )
    for message in (
        "No checkpoint is active",
        "No update has been attempted yet.",
        "No checkpoints have been written yet.",
        "records no gate evaluations",
        "No training batches have been recorded.",
        "has not run a round hook",
    ):
        assert message in document
