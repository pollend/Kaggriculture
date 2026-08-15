"""Hand-rolled inline SVG charts.

There is no charting library here and there should not be one. Every mark below is
a `<polyline>`, `<rect>` or `<text>` with its geometry computed in Python and its
colour set as an SVG *presentation attribute* -- which is not a style, so none of
this touches the `script-src 'self'` / `style-src` policy the pages are served
under, and nothing has to be vendored.

Three primitives cover what the dashboard needs: a multi-series line, a bar (with
a diverging variant whose zero sits wherever zero falls), and a sparkline. Binning
is SQL's job -- `width_bucket` -- so a histogram is a bar chart over rows that
arrive already bucketed, and this module never grows a statistics layer.
"""

from __future__ import annotations

import html
from typing import Any, Sequence

# Matches the CSS custom properties, so a chart sits in the palette rather than
# beside it. Six is the cap: past that, series stop being distinguishable and the
# caller should fold the tail into an "other".
CHART_INK = ("#62d4c7", "#f0c66b", "#ff8181", "#a9b6ff", "#ffcf70", "#7de3ae")
AXIS_INK = "#9dafc8"
GRID_INK = "#263650"
ZERO_INK = "#4a5f80"

MAX_SERIES = len(CHART_INK)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def empty(message: str) -> str:
    """The one empty state every chart falls back to.

    A chart with no data must not render a degenerate axis: an empty pair of axes
    reads as "the value is zero" rather than "there is nothing yet".
    """
    return f'<div class="empty">{_escape(message)}</div>'


def scale(values: Sequence[float], *, pad: float = 0.1, min_span: float = 1.0) -> tuple[float, float]:
    """Low and high bounds for an axis, padded so marks do not touch the frame."""
    if not values:
        return 0.0, min_span
    low, high = min(values), max(values)
    padding = max(min_span * 10 * pad, (high - low) * pad)
    low, high = low - padding, high + padding
    if high - low < min_span:
        high = low + min_span
    return low, high


def _numbers(rows: list[dict[str, Any]], keys: Sequence[str]) -> list[float]:
    return [
        float(row[key])
        for row in rows
        for key in keys
        if row.get(key) is not None
    ]


def line_chart(
    rows: list[dict[str, Any]],
    x_key: str,
    series: Sequence[tuple[str, str, str]],
    *,
    x_label: str = "",
    label: str = "Trend",
    compact: bool = False,
    empty_message: str = "Nothing to plot yet.",
) -> str:
    """One or more series over an ordinal x axis.

    `series` is `(row key, legend label, colour)`. That list is the whole
    generalization: every caller differs only in which keys it plots.

    The x axis is ordinal -- position in `rows` -- never a time scale. Training is
    bursty, so generation index and batch sequence space the marks evenly where
    wall clock would bunch them.
    """
    series = list(series)[:MAX_SERIES]
    values = _numbers(rows, [key for key, _, _ in series])
    if not values:
        return empty(empty_message)

    width, height = (380, 170) if compact else (760, 230)
    left, top, right, bottom = (40, 26, 14, 30) if compact else (55, 20, 20, 38)
    gridlines = 3 if compact else 5
    low, high = scale(values, min_span=1.0)
    span = high - low
    steps = max(1, len(rows) - 1)
    plot_width = width - left - right
    plot_height = height - top - bottom

    def x_at(index: int) -> float:
        return left + index * plot_width / steps

    def y_at(value: float) -> float:
        return top + (high - value) * plot_height / span

    marks = []
    for key, _, colour in series:
        points = " ".join(
            f"{x_at(index):.1f},{y_at(float(row[key])):.1f}"
            for index, row in enumerate(rows)
            if row.get(key) is not None
        )
        if not points:
            continue
        marks.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="{2 if compact else 3}" '
            f'stroke-linejoin="round" points="{points}"/>'
        )
        # A one-point series has no line to draw, so give it a dot rather than
        # rendering an invisible chart.
        if points.count(" ") == 0:
            x, y = points.split(",")
            marks.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{colour}"/>')

    grid = "".join(
        f'<g><line x1="{left}" x2="{width - right}" '
        f'y1="{top + step * plot_height / (gridlines - 1):.1f}" '
        f'y2="{top + step * plot_height / (gridlines - 1):.1f}" stroke="{GRID_INK}"/>'
        f'<text x="{left - 8}" y="{top + step * plot_height / (gridlines - 1) + 4:.1f}" '
        f'fill="{AXIS_INK}" text-anchor="end">'
        f'{high - step * span / (gridlines - 1):.0f}</text></g>'
        for step in range(gridlines)
    )

    # Thin the x labels rather than letting them overlap into a smear.
    stride = max(1, len(rows) // (6 if compact else 14))
    ticks = "".join(
        f'<text x="{x_at(index):.1f}" y="{height - 12}" fill="{AXIS_INK}" '
        f'text-anchor="middle">{_escape(row.get(x_key))}</text>'
        for index, row in enumerate(rows)
        if index % stride == 0 or index == len(rows) - 1
    )

    legend = "".join(
        f'<text x="{left + offset * (plot_width / max(1, len(series)))}" y="14" fill="{colour}">'
        f"{_escape(legend_label)}</text>"
        for offset, (_, legend_label, colour) in enumerate(series)
    )
    axis_title = (
        f'<text x="{width - right}" y="{height - 12}" fill="{AXIS_INK}" text-anchor="end">'
        f"{_escape(x_label)}</text>"
        if x_label
        else ""
    )
    classes = "chart small" if compact else "chart"
    return (
        f'<svg class="{classes}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape(label)}">{grid}{ticks}{axis_title}'
        f'{"".join(marks)}{"" if compact else legend}</svg>'
    )


def bar_chart(
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    *,
    diverging: bool = False,
    horizontal: bool = False,
    label: str = "Distribution",
    empty_message: str = "Nothing to plot yet.",
    colour: str = CHART_INK[0],
    negative_colour: str = CHART_INK[2],
) -> str:
    """Bars over categories, optionally diverging around zero.

    `diverging` moves the baseline from the edge to wherever zero falls, and
    colours bars below it differently -- which is the whole difference between a
    histogram and a win/loss comparison, so it is a flag rather than a second
    function.

    `horizontal` is for long category labels (family names), where vertical bars
    would need rotated text.
    """
    values = _numbers(rows, [value_key])
    if not values:
        return empty(empty_message)

    if diverging:
        limit = max(abs(min(values)), abs(max(values))) or 1.0
        low, high = -limit, limit
    else:
        low, high = min(0.0, min(values)), max(values) or 1.0
    span = (high - low) or 1.0

    if horizontal:
        row_height = 26
        width, left, right, top, bottom = 760, 200, 60, 8, 24
        height = top + bottom + row_height * len(rows)
        plot_width = width - left - right
        zero_x = left + (0.0 - low) * plot_width / span
        marks = []
        for index, row in enumerate(rows):
            value = float(row.get(value_key) or 0.0)
            y = top + index * row_height + 4
            x = left + (min(value, 0.0) - low) * plot_width / span
            bar_width = abs(value) * plot_width / span
            marks.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width, 1):.1f}" '
                f'height="{row_height - 10}" fill="{negative_colour if value < 0 else colour}" rx="2"/>'
                f'<text x="{left - 8}" y="{y + row_height - 14:.1f}" fill="{AXIS_INK}" '
                f'text-anchor="end">{_escape(row.get(label_key))}</text>'
                f'<text x="{width - right + 6}" y="{y + row_height - 14:.1f}" fill="{AXIS_INK}">'
                f"{value:,.0f}</text>"
            )
        zero = (
            f'<line x1="{zero_x:.1f}" x2="{zero_x:.1f}" y1="{top}" y2="{height - bottom}" '
            f'stroke="{ZERO_INK}" stroke-width="1.5"/>'
        )
        return (
            f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{_escape(label)}">{zero}{"".join(marks)}</svg>'
        )

    width, height = 760, 220
    left, top, right, bottom = 55, 16, 16, 34
    plot_width = width - left - right
    plot_height = height - top - bottom
    slot = plot_width / max(1, len(rows))
    zero_y = top + (high - 0.0) * plot_height / span
    marks = []
    for index, row in enumerate(rows):
        value = float(row.get(value_key) or 0.0)
        value_y = top + (high - value) * plot_height / span
        y = min(value_y, zero_y)
        bar_height = max(abs(value_y - zero_y), 1.0)
        marks.append(
            f'<rect x="{left + index * slot + slot * 0.15:.1f}" y="{y:.1f}" '
            f'width="{slot * 0.7:.1f}" height="{bar_height:.1f}" '
            f'fill="{negative_colour if value < 0 else colour}" rx="2"/>'
        )
    stride = max(1, len(rows) // 12)
    ticks = "".join(
        f'<text x="{left + index * slot + slot / 2:.1f}" y="{height - 12}" fill="{AXIS_INK}" '
        f'text-anchor="middle">{_escape(row.get(label_key))}</text>'
        for index, row in enumerate(rows)
        if index % stride == 0
    )
    baseline = (
        f'<line x1="{left}" x2="{width - right}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" '
        f'stroke="{ZERO_INK}" stroke-width="1.5"/>'
    )
    axis = "".join(
        f'<text x="{left - 8}" y="{top + step * plot_height / 4 + 4:.1f}" fill="{AXIS_INK}" '
        f'text-anchor="end">{high - step * span / 4:,.0f}</text>'
        for step in range(5)
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape(label)}">{axis}{baseline}{"".join(marks)}{ticks}</svg>'
    )


def sparkline(
    values: Sequence[float], *, width: int = 120, height: int = 28, colour: str = CHART_INK[0]
) -> str:
    """A shape-only line for a table cell: no axes, no labels, no scale."""
    numbers = [float(value) for value in values if value is not None]
    if len(numbers) < 2:
        return ""
    low, high = min(numbers), max(numbers)
    span = (high - low) or 1.0
    steps = len(numbers) - 1
    points = " ".join(
        f"{index * width / steps:.1f},{height - (value - low) * height / span:.1f}"
        for index, value in enumerate(numbers)
    )
    return (
        f'<svg class="sparkline" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-hidden="true">'
        f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{points}"/></svg>'
    )


def status_strip(
    rows: list[dict[str, Any]],
    label_key: str,
    colour_of,
    *,
    label: str = "Status timeline",
    empty_message: str = "Nothing recorded yet.",
) -> str:
    """One coloured cell per row, in order: a timeline of outcomes at a glance.

    `colour_of(row)` returns the fill. Used for the model gate history, where the
    question is "how many of the last twenty were rejected" and each attempt is a
    category rather than a number.
    """
    if not rows:
        return empty(empty_message)
    width, height = 760, 62
    left, right, top = 8, 8, 8
    slot = (width - left - right) / max(1, len(rows))
    cells = "".join(
        f'<rect x="{left + index * slot + 1:.1f}" y="{top}" width="{max(slot - 2, 2):.1f}" '
        f'height="24" fill="{colour_of(row)}" rx="2"><title>{_escape(row.get(label_key))}</title></rect>'
        for index, row in enumerate(rows)
    )
    stride = max(1, len(rows) // 12)
    ticks = "".join(
        f'<text x="{left + index * slot + slot / 2:.1f}" y="{height - 12}" fill="{AXIS_INK}" '
        f'text-anchor="middle">{_escape(row.get(label_key))}</text>'
        for index, row in enumerate(rows)
        if index % stride == 0
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape(label)}">{cells}{ticks}</svg>'
    )
