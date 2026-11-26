"""Overlays, rugs and regression fits reach every facet panel."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from scipy.stats import linregress

from edalens.utils.plot_utils import (
    facet_panels,
    facet_regression,
    regression_stats,
    rug_position,
)
from test_visualization_plots import viz_app

TIPS = px.data.tips()
CATEGORICAL = ["sex", "smoker", "day", "time"]


def faceted(plot, **roles):
    app = viz_app(TIPS, plot, facet_col="sex", facet_row="smoker", **roles)
    app.session_state["current_categorical_cols"] = CATEGORICAL
    app.run()
    return app


def check(app, *labels):
    for box in app.checkbox:
        if box.label in labels:
            box.check()
    app.run()
    assert not app.exception
    return json.loads(app.get("plotly_chart")[-1].proto.spec)["data"]


def is_mean(trace) -> bool:
    return " mean" in trace.get("name", "")


def panels_of(traces) -> set:
    return {t.get("xaxis", "x") for t in traces}


def test_facet_panels_follow_plotly_express_rows():
    data = pd.DataFrame({"r": ["a", "a", "b"], "c": ["x", "y", "x"], "v": [1, 2, 3]})
    n_rows, n_cols, panels = facet_panels(data, "c", "r", bottom_up=True)
    assert (n_rows, n_cols) == (2, 2)
    # Plotly Express counts rows from the bottom: the first row value is row 2.
    assert [(row, col, len(sub)) for row, col, sub, _ in panels] == [
        (2, 1, 1),
        (2, 2, 1),
        (1, 1, 1),
        (1, 2, 0),
    ]


@pytest.mark.parametrize(
    "x, color", [("day", "None"), ("day", "time"), ("None", "None"), ("None", "time")]
)
def test_box_swarm_and_mean_ci_in_every_panel(x, color):
    data = check(
        faceted("boxes", x=x, y="total_bill", color=color),
        "Overlay Swarm of Points",
        "Show Band Interval (CI around means)",
    )
    boxes = [t for t in data if t["type"] == "box"]
    means = [t for t in data if is_mean(t)]
    assert {b.get("boxpoints") for b in boxes} == {"all"}
    assert panels_of(means) == {"x", "x2", "x3", "x4"}


def test_each_panel_mean_comes_from_its_own_data():
    data = check(
        faceted("boxes", x="day", y="total_bill"),
        "Show Band Interval (CI around means)",
    )
    panel = {
        t.get("xaxis", "x"): [
            p.split("=")[1] for p in t["hovertemplate"].split("<br>")[:2]
        ]
        for t in data
        if t["type"] == "box"
    }
    for trace in (t for t in data if is_mean(t)):
        smoker, sex = panel[trace.get("xaxis", "x")]
        subset = TIPS[(TIPS.smoker == smoker) & (TIPS.sex == sex)]
        expected = (
            subset.groupby("day")["total_bill"].mean().reindex(trace["x"]).to_numpy()
        )
        assert np.allclose(trace["y"], expected)


@pytest.mark.parametrize(
    "plot, roles, kind, rugs",
    [
        (
            "density 1",
            dict(x="total_bill", y="None"),
            None,
            8,
        ),  # x only, 2 colors x 4 panels.
        ("density 2", dict(x="total_bill", y="tip"), "kde", 16),
        (
            "density 2",
            dict(x="total_bill", y="tip"),
            "hist",
            8,
        ),  # The heatmap has no colors.
        ("scatter", dict(x="total_bill", y="tip"), None, 16),
        ("regression", dict(x="total_bill", y="tip"), None, 16),
    ],
)
def test_rugs_in_every_panel(plot, roles, kind, rugs):
    app = faceted(plot, color="time", **roles)
    if kind:
        next(s for s in app.selectbox if s.label == "Kind").set_value(kind)
        app.run()
    data = check(app, "Add Rug Plot?")
    rug = [t for t in data if t.get("name", "").endswith(" rug")]
    assert len(rug) == rugs and panels_of(rug) == {"x", "x2", "x3", "x4"}


def test_faceted_regression_has_a_fit_and_a_table_row_per_panel_and_color():
    app = faceted("regression", x="total_bill", y="tip", color="time")
    assert not app.exception
    data = json.loads(app.get("plotly_chart")[-1].proto.spec)["data"]
    fits = [t for t in data if t.get("name", "").startswith("Fit")]
    assert len(fits) == 8 and panels_of(fits) == {"x", "x2", "x3", "x4"}
    table = app.dataframe[0].value
    assert list(table.columns[:3]) == ["smoker", "sex", "time"] and len(table) == 8
    row = table[
        (table.smoker == "Yes") & (table.sex == "Female") & (table.time == "Dinner")
    ].iloc[0]
    subset = TIPS[
        (TIPS.smoker == "Yes") & (TIPS.sex == "Female") & (TIPS.time == "Dinner")
    ]
    expected = linregress(subset.total_bill, subset.tip)
    assert row["n"] == len(subset)
    assert row["Slope"] == pytest.approx(expected.slope) and row[
        "p-value"
    ] == pytest.approx(expected.pvalue)
    assert row["R²"] == pytest.approx(expected.rvalue**2)


def test_unfaceted_regression_also_shows_its_table():
    app = viz_app(TIPS, "regression", x="total_bill", y="tip")
    app.run()
    assert not app.exception and len(app.dataframe[0].value) == 1


def test_regression_stats_for_a_polynomial_and_too_few_points():
    frame = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0) ** 2})
    stats = regression_stats(frame, "x", "y", order=2)
    assert stats["R²"] == pytest.approx(1.0) and stats["x^2"] == pytest.approx(1.0)
    assert regression_stats(frame.head(2), "x", "y", order=2) is None


def test_facet_regression_without_facets_is_one_panel():
    fig, table = facet_regression(TIPS, "total_bill", "tip")
    assert len(table) == 1 and table.loc[0, "n"] == len(TIPS)
    assert rug_position([0, 10]) == pytest.approx(-0.6)


def layout_and_data(app):
    spec = json.loads(app.get("plotly_chart")[-1].proto.spec)
    return spec["layout"], spec["data"]


@pytest.mark.parametrize("facets", [True, False])
def test_colored_means_sit_on_their_own_boxes(facets):
    """Plotly shifts boxes by boxgap and grouped markers by scattergap: they must match."""
    extra = dict(facet_col="sex", facet_row="smoker") if facets else {}
    app = viz_app(TIPS, "boxes", x="day", y="total_bill", color="time", **extra)
    app.session_state["current_categorical_cols"] = CATEGORICAL
    app.run()
    check(app, "Show Band Interval (CI around means)")
    layout, data = layout_and_data(app)
    assert layout["scattermode"] == "group" and layout["scattergap"] == layout["boxgap"]
    box_groups = {t["offsetgroup"] for t in data if t["type"] == "box"}
    mean_groups = {t["offsetgroup"] for t in data if is_mean(t)}
    assert mean_groups == box_groups == {"Dinner", "Lunch"}


def test_uncolored_facet_boxes_are_not_squeezed_off_center():
    # Plotly Express would group the panels' single boxes into one alignment group.
    app = faceted("boxes", x="day", y="total_bill")
    check(app, "Show Band Interval (CI around means)")
    layout, _ = layout_and_data(app)
    assert layout["boxmode"] == "overlay"


def test_x_as_color_keeps_one_centered_box_per_category():
    for extra in ({}, dict(facet_col="sex")):
        app = viz_app(TIPS, "boxes", x="day", y="total_bill", color="day", **extra)
        app.session_state["current_categorical_cols"] = CATEGORICAL
        app.run()
        check(app, "Show Band Interval (CI around means)")
        layout, data = layout_and_data(app)
        # Nothing to place side by side: overlay, and no shifted (grouped) markers.
        assert layout["boxmode"] == "overlay" and layout.get("scattermode") != "group"
        means = [t for t in data if is_mean(t)]
        assert means and not any(t.get("offsetgroup") for t in means)
        assert {t["marker"]["color"] for t in means} == {
            t["marker"]["color"] for t in data if t["type"] == "box"
        }


def test_points_by_default_with_colors_or_facets_and_line_on_request():
    app = faceted("boxes", x="sex", y="total_bill", color="time")
    data = check(app, "Show Band Interval (CI around means)")
    means = [t for t in data if is_mean(t)]
    assert all(t["mode"] == "markers" and "error_y" in t for t in means)
    assert not any(t.get("fill") == "toself" for t in data)
    next(s for s in app.selectbox if s.label == "Mean ± CI shown as").set_value(
        "Line and band"
    )
    app.run()
    _, data = layout_and_data(app)
    assert any(t.get("fill") == "toself" for t in data)
    assert any(t.get("mode") == "lines+markers" for t in data if is_mean(t))
    # One color, no facets: the line and band stay the default.
    plain = viz_app(TIPS, "boxes", x="day", y="total_bill")
    plain.run()
    check(plain, "Show Band Interval (CI around means)")
    assert (
        next(s for s in plain.selectbox if s.label == "Mean ± CI shown as").value
        == "Line and band"
    )


@pytest.mark.parametrize("extra", [{}, dict(facet_col="sex")])
def test_line_and_band_with_x_as_color_joins_all_categories(extra):
    app = viz_app(TIPS, "boxes", x="day", y="total_bill", color="day", **extra)
    app.session_state["current_categorical_cols"] = CATEGORICAL
    app.run()
    check(app, "Show Band Interval (CI around means)")
    next(s for s in app.selectbox if s.label == "Mean ± CI shown as").set_value(
        "Line and band"
    )
    app.run()
    _, data = layout_and_data(app)
    lines = [t for t in data if is_mean(t)]
    panels = 2 if extra else 1
    assert len(lines) == panels and all(len(t["x"]) == 4 for t in lines)
    assert sum(t.get("fill") == "toself" for t in data) == panels
