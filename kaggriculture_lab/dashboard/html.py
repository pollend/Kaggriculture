"""Escaping, formatting and the small HTML fragments every page is built from.

Nothing here touches the database. Given rows, these produce markup; that split is
what lets the page renderers be tested without PostgreSQL.
"""

from __future__ import annotations

# Absolute import: this module shadows the stdlib name only inside the package.
import html
import json
import re
from typing import Any
from urllib.parse import quote, urlencode

from kaggriculture_lab.dashboard import charts
from kaggriculture_lab.dashboard.errors import DashboardUnavailable

PAGE_SIZE = 50


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _number(value: Any, digits: int = 0) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:,.{digits}f}"


def _percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def _duration(value: Any) -> str:
    if value is None:
        return "—"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.0f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m"


# `model_updates.status` is RUNNING / PROMOTED / REJECTED / ERROR, and a rejected
# candidate is not a failure -- it is the gate doing its job. It gets its own
# neutral class so it reads differently from both a promotion and an error;
# without that distinction the model page's most important column would be one
# uniform colour.
GOOD_STATUSES = frozenset({"COMPLETE", "DONE", "SUCCESS", "PROMOTED"})
BAD_STATUSES = frozenset({"ERROR", "FAILED"})
REJECTED_STATUSES = frozenset({"REJECTED", "SKIPPED"})


def _status(value: Any) -> str:
    label = str(value or "UNKNOWN")
    if label in GOOD_STATUSES:
        kind = "good"
    elif label in BAD_STATUSES:
        kind = "bad"
    elif label in REJECTED_STATUSES:
        kind = "muted-pill"
    else:
        kind = "warn"
    return f'<span class="status {kind}">{_escape(label)}</span>'


def _href(path: str, **query: Any) -> str:
    values = {key: value for key, value in query.items() if value not in (None, "", "all")}
    return path + (f"?{urlencode(values)}" if values else "")


def _id_path(prefix: str, identifier: Any) -> str:
    return f"/{prefix}/{quote(str(identifier), safe='')}"


def _agent_identity_html(
    display_name: Any,
    strategy_id: Any,
    revision_id: Any,
    checkpoint_id: Any,
    *,
    label: str | None = None,
    params_json: Any = None,
) -> str:
    heading = _escape(display_name or revision_id or "Unknown agent")
    policy_label = _policy_role_label(params_json, revision_id)
    if policy_label:
        heading = f'<span class="muted">{_escape(policy_label)}</span> · {heading}'
    if label:
        heading = f'<span class="muted">{_escape(label)}</span> · {heading}'
    details = []
    if strategy_id:
        details.append(f"strategy {_escape(strategy_id)}")
    if revision_id:
        details.append(f"revision {_escape(revision_id)}")
    if checkpoint_id:
        details.append(f"checkpoint {_escape(checkpoint_id)}")
    suffix = "<br>".join(f"<small>{item}</small>" for item in details)
    return f"<b>{heading}</b>" + (f"<br>{suffix}" if suffix else "")


def _policy_role_label(params_json: Any, revision_id: Any) -> str | None:
    try:
        params = json.loads(str(params_json)) if params_json else {}
    except (TypeError, ValueError):
        return None
    role = params.get("role")
    revision = str(revision_id or "")
    if role == "best_response":
        suffix = revision.removeprefix("beta-")[:8]
        return "Live best response" + (f" — β {suffix}" if suffix else " — β")
    if role != "population":
        return None
    if revision == "member-000000-initial":
        return "Frozen population — initial baseline"
    if revision == "live-nu":
        return "Live new policy — ν"
    match = re.fullmatch(r"member-(\d+)-(beta|nu-average)", revision)
    if match:
        iteration = int(match.group(1))
        kind = "β" if match.group(2) == "beta" else "averaged ν"
        return f"Frozen population — {kind} member {iteration}"
    return "Frozen population"


def _agent_replay_name(
    display_name: Any,
    strategy_id: Any,
    revision_id: Any,
    checkpoint_id: Any,
    params_json: Any = None,
    *,
    seat: int | None = None,
) -> str:
    name = str(display_name or "Agent")
    policy_label = _policy_role_label(params_json, revision_id)
    if policy_label:
        prefix = f"Seat {seat + 1} · " if seat is not None else ""
        return f"{prefix}{policy_label} · {name} (fixed for episode)"
    variant = strategy_id or checkpoint_id or revision_id
    return f"{name} · {variant}" if variant else name


def _layout(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_escape(title)} · Kaggriculture</title>
<link rel="stylesheet" href="/dashboard.css"><script src="/dashboard.js" defer></script></head><body><header><div class="bar"><a class="brand" href="/">Kaggriculture League</a>
<span class="muted">Shared PostgreSQL database</span></div></header><main>{body}</main></body></html>"""


def _main_fragment(document: str) -> str:
    start = document.find("<main>")
    end = document.rfind("</main>")
    if start < 0 or end < start:
        raise DashboardUnavailable("dashboard response did not contain a main element")
    return document[start : end + len("</main>")]


def _cards(items: list[tuple[str, str]]) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><span class="muted">{_escape(label)}</span><b>{value}</b></div>'
        for label, value in items
    ) + "</div>"


def _progress(done: int, total: int) -> str:
    fraction = min(1.0, done / total) if total else 0.0
    return f'<div class="progress" title="{done} of {total}"><i style="width:{100 * fraction:.1f}%"></i></div><small>{done:,} / {total:,}</small>'


def _pager(path: str, page: int, has_next: bool, **query: Any) -> str:
    previous = f'<a class="button" data-dashboard-nav href="{_escape(_href(path, page=page - 1, **query))}">← Previous</a>' if page > 1 else "<span></span>"
    following = f'<a class="button" data-dashboard-nav href="{_escape(_href(path, page=page + 1, **query))}">Next →</a>' if has_next else "<span></span>"
    return f'<div class="pager">{previous}<span class="muted">Page {page}</span>{following}</div>'


def _chart(trend: list[dict[str, Any]]) -> str:
    """The session page's rating trend: champion against median, per generation."""
    return charts.line_chart(
        trend,
        "generation_index",
        [
            ("champion_conservative", "Champion conservative rating", charts.CHART_INK[0]),
            ("median_conservative", "Median conservative rating", charts.CHART_INK[1]),
        ],
        label="Rating trend",
        empty_message="Ratings appear after a generation closes.",
    )
