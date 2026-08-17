"""The inline SVG chart primitives.

These render markup from numbers with no database and no browser, so the things
worth pinning down are the ones a screenshot would not catch: that an empty series
degrades to a message rather than a bare axis, that a diverging bar puts zero where
zero belongs, and that nothing emitted here needs a relaxed content policy.
"""

from __future__ import annotations

import re

from kaggriculture_lab.dashboard import charts


def test_a_chart_with_no_numbers_says_so_instead_of_drawing_empty_axes():
    """An empty pair of axes reads as "the value is zero", which is a different claim."""
    assert "<svg" not in charts.line_chart([], "x", [("y", "Y", charts.CHART_INK[0])])
    assert "<svg" not in charts.line_chart(
        [{"x": 0, "y": None}], "x", [("y", "Y", charts.CHART_INK[0])]
    )
    assert "<svg" not in charts.bar_chart([], "label", "value")
    assert charts.sparkline([]) == ""
    assert charts.sparkline([1.0]) == ""  # one point has no shape


def test_the_empty_message_is_escaped():
    assert "&lt;script&gt;" in charts.empty("<script>")


def test_a_line_chart_plots_one_polyline_per_series():
    document = charts.line_chart(
        [
            {"x": 0, "a": 10, "b": 5},
            {"x": 1, "a": 20, "b": 6},
            {"x": 2, "a": 15, "b": 7},
        ],
        "x",
        [("a", "Alpha", charts.CHART_INK[0]), ("b", "Beta", charts.CHART_INK[1])],
    )
    assert document.count("<polyline") == 2
    assert "Alpha" in document and "Beta" in document
    assert 'role="img"' in document and 'aria-label=' in document


def test_a_line_chart_skips_gaps_rather_than_plotting_them_as_zero():
    document = charts.line_chart(
        [{"x": 0, "a": 10}, {"x": 1, "a": None}, {"x": 2, "a": 30}],
        "x",
        [("a", "Alpha", charts.CHART_INK[0])],
    )
    points = re.search(r'points="([^"]+)"', document).group(1)
    assert len(points.split(" ")) == 2


def test_a_single_point_series_still_shows_something():
    document = charts.line_chart(
        [{"x": 0, "a": 10}], "x", [("a", "Alpha", charts.CHART_INK[0])]
    )
    assert "<circle" in document


def test_series_beyond_the_palette_are_dropped_rather_than_recoloured():
    series = [(f"k{index}", f"S{index}", charts.CHART_INK[0]) for index in range(10)]
    rows = [{"x": 0, **{f"k{index}": index for index in range(10)}}]
    document = charts.line_chart(rows, "x", series)
    assert document.count("<circle") <= charts.MAX_SERIES


def test_x_labels_are_thinned_rather_than_overlapped():
    rows = [{"x": index, "a": index} for index in range(200)]
    document = charts.line_chart(rows, "x", [("a", "Alpha", charts.CHART_INK[0])])
    assert document.count("<text") < 30


def test_a_compact_chart_drops_its_minimum_width_so_a_grid_can_hold_it():
    rows = [{"x": index, "a": index} for index in range(5)]
    assert 'class="chart small"' in charts.line_chart(
        rows, "x", [("a", "A", charts.CHART_INK[0])], compact=True
    )
    assert 'class="chart"' in charts.line_chart(rows, "x", [("a", "A", charts.CHART_INK[0])])


def test_a_plain_bar_chart_baselines_at_zero():
    document = charts.bar_chart(
        [{"label": "a", "value": 5}, {"label": "b", "value": 10}], "label", "value"
    )
    assert document.count("<rect") == 2
    assert "<line" in document  # the zero baseline


def test_a_diverging_bar_chart_colours_and_places_negatives_correctly():
    document = charts.bar_chart(
        [{"label": "loss", "value": -100}, {"label": "win", "value": 100}],
        "label",
        "value",
        diverging=True,
    )
    assert charts.CHART_INK[2] in document  # the negative colour
    assert charts.CHART_INK[0] in document
    # Symmetric bounds, so the two equal-and-opposite bars are the same length.
    widths = re.findall(r'<rect[^>]*height="([\d.]+)"', document)
    assert len(set(widths)) == 1


def test_a_horizontal_bar_chart_labels_each_row():
    document = charts.bar_chart(
        [{"family": "alpha", "margin": -50}, {"family": "beta", "margin": 120}],
        "family",
        "margin",
        diverging=True,
        horizontal=True,
    )
    assert "alpha" in document and "beta" in document
    assert document.count("<rect") == 2


def test_a_zero_valued_bar_is_still_visible():
    document = charts.bar_chart([{"label": "a", "value": 0}], "label", "value")
    assert '<rect' in document


def test_a_status_strip_carries_a_tooltip_per_cell():
    document = charts.status_strip(
        [{"seq": 1, "ok": True}, {"seq": 2, "ok": False}],
        "seq",
        lambda row: charts.CHART_INK[0] if row["ok"] else charts.CHART_INK[2],
    )
    assert document.count("<rect") == 2
    assert document.count("<title>") == 2
    assert "<svg" not in charts.status_strip([], "seq", lambda row: "#fff")


def test_scale_pads_and_never_collapses_to_a_zero_span():
    low, high = charts.scale([10.0, 10.0])
    assert high > low
    low, high = charts.scale([0.0, 100.0])
    assert low < 0 and high > 100


def test_nothing_emitted_needs_a_relaxed_content_policy():
    """Colours are SVG presentation attributes, not styles; no script, no external URL."""
    rows = [{"x": index, "a": index} for index in range(4)]
    documents = [
        charts.line_chart(rows, "x", [("a", "A", charts.CHART_INK[0])]),
        charts.bar_chart(rows, "x", "a"),
        charts.bar_chart(rows, "x", "a", diverging=True, horizontal=True),
        charts.sparkline([1, 2, 3]),
        charts.status_strip(rows, "x", lambda row: charts.CHART_INK[0]),
    ]
    for document in documents:
        assert "<script" not in document
        assert "style=" not in document
        assert "http://" not in document and "https://" not in document


def test_labels_are_escaped():
    document = charts.bar_chart([{"label": "<script>", "value": 1}], "label", "value")
    assert "<script>" not in document
    assert "&lt;script&gt;" in document
