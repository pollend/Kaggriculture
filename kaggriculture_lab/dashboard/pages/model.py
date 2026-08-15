"""One trainable model at `/models/<model_id>`.

Until this page existed, `training_models`, `training_batches`, `model_updates`,
`model_training_checkpoints` and `model_evaluations` were reachable only through
the MCP tool -- so "why did the gate reject the last twenty candidates?", which is
the question a trainer run actually turns on, could not be asked in the browser.

The gate timeline is the centre of the page. It is drawn against
`cutoff_batch_seq` rather than wall clock because training is bursty: a day of
nothing followed by forty updates in an hour spaces out sensibly by batch and
bunches unreadably by time.
"""

from __future__ import annotations

import json
from typing import Any

from kaggriculture_lab import league_queries as lq
from kaggriculture_lab.dashboard import charts
from kaggriculture_lab.dashboard.errors import DashboardNotFound
from kaggriculture_lab.dashboard.html import (
    _cards,
    _escape,
    _id_path,
    _layout,
    _number,
    _progress,
    _status,
)

HISTORY_LIMIT = 40

UPDATE_STATUS_INK = {
    "PROMOTED": charts.CHART_INK[5],
    "REJECTED": charts.CHART_INK[1],
    "ERROR": charts.CHART_INK[2],
    "RUNNING": charts.CHART_INK[3],
}


def fetch(connection: Any, model_id: str) -> dict[str, Any]:
    model = lq.model(connection, model_id)
    if model is None:
        raise DashboardNotFound(f"model not found: {model_id}")
    active_checkpoint = None
    if model["active_checkpoint_id"]:
        active_checkpoint = lq.checkpoint(connection, str(model["active_checkpoint_id"]))
    return {
        "model": model,
        "active_checkpoint": active_checkpoint,
        # Oldest-first and unlimited for the timeline: the shape of the whole
        # history is the point, and one rect per update stays legible far past
        # the length any table would.
        "updates": lq.model_updates(connection, model_id, oldest_first=True),
        "checkpoints": lq.model_checkpoints(connection, model_id, limit=HISTORY_LIMIT),
        "evaluations": lq.model_evaluations(connection, model_id, limit=HISTORY_LIMIT),
        "batch_sources": lq.model_batch_sources(connection, model_id),
        "batches": lq.model_batches(connection, model_id, limit=HISTORY_LIMIT),
        "hooks": lq.model_hook_status(connection, model_id),
    }


def _json_block(value: Any, summary: str) -> str:
    if value in (None, "", "{}"):
        return ""
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return f"<details><summary>{_escape(summary)}</summary><pre>{_escape(value)}</pre></details>"
    if not parsed:
        return ""
    text = json.dumps(parsed, indent=2, sort_keys=True)
    return f"<details><summary>{_escape(summary)}</summary><pre>{_escape(text)}</pre></details>"


def _gate_flag(update: dict[str, Any]) -> str:
    """Whether the gate passed, read from `gate_json` and falling back to status.

    `gate_json` has no shared schema: each trainer writes its own shape, and only
    some record `passed`. The status is the one field every family sets.
    """
    try:
        gate = json.loads(str(update.get("gate_json") or "{}"))
    except (TypeError, ValueError):
        gate = {}
    if isinstance(gate, dict) and isinstance(gate.get("passed"), bool):
        return "yes" if gate["passed"] else "no"
    return "yes" if update.get("status") == "PROMOTED" else "—"


def _gate_timeline(updates: list[dict[str, Any]]) -> str:
    if not updates:
        return charts.empty("The gate has not run yet.")
    rows = [
        {"seq": int(update["cutoff_batch_seq"] or 0), "status": update["status"]}
        for update in updates
    ]
    strip = charts.status_strip(
        rows,
        "seq",
        lambda row: UPDATE_STATUS_INK.get(str(row["status"]), charts.AXIS_INK),
        label="Update outcomes by batch sequence",
    )
    legend = " · ".join(
        f'<span style="color:{colour}">{_escape(status)}</span>'
        for status, colour in UPDATE_STATUS_INK.items()
    )
    return f'{strip}<p class="muted">{legend}</p>'


def _metric_trend(updates: list[dict[str, Any]]) -> str:
    """Small multiples over whatever numeric metrics this family happens to emit.

    The keys cannot be hard-coded: dqn writes `mean_loss`, sp_psro writes
    `beta_loss`, mixed_farm writes `core_mae`. They are discovered from the rows,
    kept only where at least two updates carry them, and each gets its own y
    scale -- a loss and a step count on one axis would flatten the loss to a line.
    """
    series: dict[str, list[tuple[int, float]]] = {}
    for update in updates:
        try:
            metrics = json.loads(str(update.get("metrics_json") or "{}"))
        except (TypeError, ValueError):
            continue
        if not isinstance(metrics, dict):
            continue
        sequence = int(update["cutoff_batch_seq"] or 0)
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            series.setdefault(str(key), []).append((sequence, float(value)))
    plottable = {key: points for key, points in series.items() if len(points) >= 2}
    if not plottable:
        return charts.empty("Numeric update metrics appear once the trainer records them.")

    # Six panels is the cap, so the ordering decides which six. Fitting error
    # first, then what the policy actually earned, then bookkeeping counters --
    # alphabetical order would spend the whole grid on step counts.
    def rank(key: str) -> tuple[int, str]:
        lowered = key.lower()
        if any(word in lowered for word in ("loss", "mae", "error")):
            return (0, key)
        if any(
            word in lowered
            for word in ("return", "money", "margin", "win", "rate", "reward", "score")
        ):
            return (1, key)
        return (2, key)

    panels = []
    for index, key in enumerate(sorted(plottable, key=rank)[: charts.MAX_SERIES]):
        points = sorted(plottable[key])
        rows = [{"seq": sequence, "value": value} for sequence, value in points]
        chart = charts.line_chart(
            rows,
            "seq",
            [("value", key, charts.CHART_INK[index % charts.MAX_SERIES])],
            label=f"{key} by batch sequence",
            compact=True,
        )
        panels.append(f'<div class="panel"><h3>{_escape(key)}</h3>{chart}</div>')
    return f'<div class="grid">{"".join(panels)}</div>'


def _checkpoint_lineage(checkpoints: list[dict[str, Any]], active_id: Any) -> str:
    """Checkpoints as a parent-to-child chain rather than a flat list.

    `parent_checkpoint_id` self-references, so the chain is where a lineage reset
    shows up -- a checkpoint whose parent is absent is a fresh start, and a flat
    list hides that entirely.
    """
    if not checkpoints:
        return '<div class="panel empty">No checkpoints have been written yet.</div>'
    known = {str(row["checkpoint_id"]) for row in checkpoints}
    ordered = sorted(checkpoints, key=lambda row: str(row["created_at"]))
    rows = []
    for checkpoint in ordered:
        parent = checkpoint["parent_checkpoint_id"]
        if parent is None:
            lineage = '<span class="muted">root</span>'
        elif str(parent) in known:
            lineage = f'<small>{_escape(parent)}</small>'
        else:
            lineage = f'<small>{_escape(parent)}</small> <span class="muted">(outside this page)</span>'
        badges = []
        if checkpoint["promoted"]:
            badges.append(_status("PROMOTED"))
        if active_id and str(checkpoint["checkpoint_id"]) == str(active_id):
            badges.append('<span class="status good">ACTIVE</span>')
        rows.append(
            f'<tr><td><small>{_escape(checkpoint["checkpoint_id"])}</small></td>'
            f'<td>{lineage}</td><td>{" ".join(badges) or "—"}</td>'
            f'<td>{_escape(checkpoint["created_at"])}</td>'
            f'<td>{_json_block(checkpoint.get("metrics_json"), "Metrics")}'
            f'{_json_block(checkpoint.get("gate_json"), "Gate")}</td></tr>'
        )
    return (
        '<div class="panel"><table><thead><tr><th>Checkpoint</th><th>Parent</th>'
        '<th>State</th><th>Created</th><th>Detail</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def render(data: dict[str, Any]) -> str:
    model = data["model"]
    model_id = str(model["model_id"])
    updates = data["updates"]
    latest = int(model["latest_batch_seq"] or 0)
    attempted = int(model["last_attempted_batch_seq"] or 0)
    minimum = int(model["min_new_batches"] or 0)
    new_batches = latest - attempted
    title = str(model["family_name"] or model["family_id"] or model_id)

    cards = _cards([
        ("Latest batch", f"{latest:,}"),
        ("Last attempted", f"{attempted:,}"),
        ("New batches", f"{new_batches:,}"),
        ("Update threshold", f"{minimum:,}"),
        ("Updates", f"{len(updates):,}"),
        (
            "Promoted",
            f"{sum(1 for update in updates if update['status'] == 'PROMOTED'):,}",
        ),
    ])
    # `_progress` reads as "how close is the next update", which is exactly the
    # `new_batches >= min_new_batches` test the updater applies.
    threshold = (
        f'<div class="panel"><h3>Next update</h3>{_progress(min(new_batches, minimum), minimum)}'
        f'<p class="muted">An update fires once {minimum:,} new batches have landed since the '
        f'last attempt.</p></div>'
        if minimum
        else ""
    )

    active = data["active_checkpoint"]
    active_html = (
        '<div class="panel empty">No checkpoint is active; the model is still running its '
        "first update.</div>"
        if active is None
        else (
            '<div class="panel">'
            f'<p><b>{_escape(active["checkpoint_id"])}</b> · {_status("PROMOTED" if active["promoted"] else "CANDIDATE")}</p>'
            f'<p class="muted">trainer {_escape(active["trainer_artifact_path"])}<br>'
            f'inference {_escape(active["inference_artifact_path"])}<br>'
            f'artifact sha {_escape(active["artifact_sha256"])}<br>'
            f'dataset sha {_escape(active["dataset_sha256"])}</p>'
            f'{_json_block(active.get("metrics_json"), "Metrics")}'
            f'{_json_block(active.get("gate_json"), "Gate")}</div>'
        )
    )

    # The chart above draws every update -- one rect each stays legible -- but the
    # table carries two JSON blobs per row, so it is capped. The cap is stated
    # rather than silent: a truncated table that looks complete is worse than a
    # shorter one that says so.
    listed_updates = list(reversed(updates))[:HISTORY_LIMIT]
    hidden = len(updates) - len(listed_updates)
    update_rows = "".join(
        f'<tr><td>{int(update["cutoff_batch_seq"] or 0):,}</td>'
        f'<td>{int(update["new_batch_count"] or 0):,}</td>'
        f'<td>{_status(update["status"])}</td><td>{_gate_flag(update)}</td>'
        f'<td><small>{_escape(update["candidate_checkpoint_id"] or "—")}</small></td>'
        f'<td>{_escape(update["started_at"])}</td>'
        f'<td>{_escape(update["error_text"] or "—")}</td>'
        f'<td>{_json_block(update.get("metrics_json"), "Metrics")}'
        f'{_json_block(update.get("gate_json"), "Gate")}</td></tr>'
        for update in listed_updates
    )
    truncated = (
        f'<p class="muted">Showing the {len(listed_updates)} most recent of '
        f'{len(updates)} updates; {hidden} older are on the chart above only.</p>'
        if hidden > 0
        else ""
    )
    update_table = (
        '<div class="panel"><table><thead><tr><th>Cutoff</th><th>Batches</th><th>Status</th>'
        '<th>Gate</th><th>Candidate</th><th>Started</th><th>Error</th><th>Detail</th>'
        f'</tr></thead><tbody>{update_rows}</tbody></table>{truncated}</div>'
        if updates
        else '<div class="panel empty">No update has been attempted yet.</div>'
    )

    evaluation_rows = "".join(
        f'<tr><td><small>{_escape(row["update_id"])}</small></td>'
        f'<td>{_escape(row["evaluation_kind"])}</td>'
        f'<td>{_status("PASSED" if row["passed"] else "REJECTED")}</td>'
        f'<td><small>{_escape(row["opponent_checkpoint_id"] or "—")}</small></td>'
        f'<td>{_escape(row["created_at"])}</td>'
        f'<td>{_json_block(row.get("metrics_json"), "Metrics")}</td></tr>'
        for row in data["evaluations"]
    )
    evaluation_table = (
        '<div class="panel"><table><thead><tr><th>Update</th><th>Kind</th><th>Result</th>'
        '<th>Opponent</th><th>Created</th><th>Detail</th></tr></thead>'
        f'<tbody>{evaluation_rows}</tbody></table></div>'
        if data["evaluations"]
        else '<div class="panel empty">This model records no gate evaluations.</div>'
    )

    source_rows = "".join(
        f'<tr><td>{_escape(row["source_kind"])}</td><td>{int(row["batches"] or 0):,}</td>'
        f'<td>{int(row["consumed"] or 0):,}</td><td>{int(row["pending"] or 0):,}</td>'
        f'<td>{_number(row["samples"])}</td></tr>'
        for row in data["batch_sources"]
    )
    batch_panel = (
        '<div class="panel"><table><thead><tr><th>Source</th><th>Batches</th>'
        '<th>Consumed</th><th>Pending</th><th>Samples</th></tr></thead>'
        f'<tbody>{source_rows}</tbody></table></div>'
        if data["batch_sources"]
        else '<div class="panel empty">No training batches have been recorded.</div>'
    )

    hook_rows = "".join(
        f'<tr><td><a href="{_escape(_id_path("sessions", row["session_id"]))}">'
        f'{_escape(row["session_id"])}</a></td><td>{_escape(row["scope"])}</td>'
        f'<td>{_status(row["status"])}</td><td>{int(row["hooks"] or 0):,}</td>'
        f'<td>{_number(row["latest_generation_index"])}</td>'
        f'<td>{_number(row["latest_round_index"])}</td>'
        f'<td>{_number(row["batches_recorded"])}</td></tr>'
        for row in data["hooks"]
    )
    hook_panel = (
        '<div class="panel"><table><thead><tr><th>Session</th><th>Scope</th><th>Status</th>'
        '<th>Hooks</th><th>Latest generation</th><th>Latest round</th><th>Batches</th>'
        f'</tr></thead><tbody>{hook_rows}</tbody></table></div>'
        if data["hooks"]
        else '<div class="panel empty">This model has not run a round hook.</div>'
    )

    body = f'''<h1>{_escape(title)}</h1>
<p class="muted">{_escape(model_id)} · {_escape(model["trainer_path"])} · agent {_escape(model["agent_name"] or "—")}</p>
{cards}{threshold}
<h2>Update gate</h2><div class="panel">{_gate_timeline(updates)}</div>
{update_table}
<h2>Update metrics</h2>{_metric_trend(updates)}
<h2>Active checkpoint</h2>{active_html}
<h2>Checkpoint lineage</h2>{_checkpoint_lineage(data["checkpoints"], model["active_checkpoint_id"])}
<h2>Gate evaluations</h2>{evaluation_table}
<h2>Training batches</h2>{batch_panel}
<h2>Round hooks</h2>{hook_panel}
<details class="panel"><summary>Model configuration</summary>
<pre>{_escape(json.dumps(json.loads(str(model["config_json"] or "{{}}")), indent=2, sort_keys=True))}</pre></details>'''
    return _layout(title, body)


def _model_page(connection: Any, model_id: str) -> str:
    return render(fetch(connection, model_id))
