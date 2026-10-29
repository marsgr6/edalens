"""Tests for the regression figure builder: fit methods, marginals, and validation."""

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from utils.plot_utils import (
    MARGINAL_KINDS,
    lowess_fit,
    numeric_heatmap_figure,
    polynomial_fit,
    regression_figure,
    scale_columns,
)


def sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 80)
    return pd.DataFrame(
        {
            "x": x,
            "y": np.sin(x) + rng.normal(0, 0.2, x.size),
            "group": np.where(np.arange(x.size) % 2, "a", "b"),
            "label": ["t"] * x.size,
        }
    )


def traces(fig, kind=None, name_prefix=None):
    return [
        trace
        for trace in fig.data
        if (kind is None or trace.type == kind)
        and (name_prefix is None or (trace.name or "").startswith(name_prefix))
    ]


def test_polynomial_fit_band_contains_the_fit():
    frame = sample_frame()
    fit = polynomial_fit(frame["x"].to_numpy(), frame["y"].to_numpy(), order=3, ci_level=95)
    assert list(fit.columns) == ["x", "fit", "lower", "upper"]
    assert (fit["lower"] <= fit["fit"]).all() and (fit["fit"] <= fit["upper"]).all()
    # Too few distinct x values for the order.
    assert polynomial_fit(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 2, 95) is None


def test_lowess_follows_a_nonlinear_signal_better_than_a_line():
    frame = sample_frame()
    x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
    smooth = lowess_fit(x, y, frac=0.2)
    assert smooth["x"].is_monotonic_increasing and len(smooth) == len(x)
    lowess_err = np.mean((np.interp(x, smooth["x"], smooth["fit"]) - np.sin(x)) ** 2)
    line = polynomial_fit(x, y, 1, 95)
    line_err = np.mean((np.interp(x, line["x"], line["fit"]) - np.sin(x)) ** 2)
    assert lowess_err < line_err / 5
    with pytest.raises(ValueError):
        lowess_fit(x, y, frac=0)


def test_default_matches_previous_behaviour_polynomial_with_band():
    fig = regression_figure(sample_frame(), "x", "y")
    assert len(traces(fig, "scatter", "Data")) == 1
    assert len(traces(fig, "scatter", "Fit")) == 1
    assert any(trace.fill == "toself" for trace in fig.data)
    assert not traces(fig, name_prefix="LOWESS")
    assert "95% Confidence Interval" in fig.layout.title.text


def test_lowess_only_and_both_methods_per_hue_group():
    frame = sample_frame()
    only = regression_figure(frame, "x", "y", methods=["LOWESS"], lowess_frac=0.3)
    assert len(traces(only, name_prefix="LOWESS")) == 1
    assert not any(trace.fill == "toself" for trace in only.data)

    both = regression_figure(frame, "x", "y", hue="group", methods=["Polynomial", "LOWESS"], order=2)
    assert len(traces(both, name_prefix="LOWESS")) == 2
    assert len(traces(both, name_prefix="Poly (order 2)")) == 2
    assert {trace.line.dash for trace in traces(both, name_prefix="LOWESS")} == {"dash"}


@pytest.mark.parametrize("kind", MARGINAL_KINDS)
def test_marginals_are_added_per_group_on_both_axes(kind):
    fig = regression_figure(sample_frame(), "x", "y", hue="group", marginal_x=kind, marginal_y=kind)
    expected = {"rug": "scatter", "histogram": "histogram", "box": "box", "violin": "violin"}[kind]
    # Marginals are hidden from the legend; two groups x two axes.
    marginal = [t for t in fig.data if t.type == expected and t.showlegend is False and getattr(t, "fill", None) != "toself"]
    assert len(marginal) == 4
    main = traces(fig, "scatter", "a")[0]

    def linked(axis_id, other_id):
        # Shared subplot axes have their own ids and are linked with ``matches``.
        axis = fig.layout[axis_id[0] + "axis" + axis_id[1:]]
        return axis_id == other_id or axis.matches == other_id

    top = [t for t in marginal if linked(t.xaxis, main.xaxis) and t.yaxis != main.yaxis]
    right = [t for t in marginal if linked(t.yaxis, main.yaxis) and t.xaxis != main.xaxis]
    assert len(top) == 2 and len(right) == 2


def test_single_marginal_uses_two_rows_only():
    fig = regression_figure(sample_frame(), "x", "y", marginal_x="histogram")
    assert len(fig._grid_ref) == 2 and len(fig._grid_ref[0]) == 1
    assert len(traces(fig, "histogram")) == 1


def test_box_and_violin_marginals_get_more_room_than_rug_and_grow_with_groups():
    frame = sample_frame()
    frame["three"] = np.arange(len(frame)) % 3

    def top_share(fig):
        main, top = fig.layout.yaxis3.domain, fig.layout.yaxis.domain
        return (top[1] - top[0]) / (main[1] - main[0])

    def right_share(fig):
        return fig.layout.xaxis4.domain[1] - fig.layout.xaxis4.domain[0]

    rug = regression_figure(frame, "x", "y", marginal_x="rug", marginal_y="rug")
    for kind in ("box", "violin"):
        one = regression_figure(frame, "x", "y", marginal_x=kind, marginal_y=kind)
        three = regression_figure(frame, "x", "y", hue="three", marginal_x=kind, marginal_y=kind)
        assert top_share(rug) < top_share(one) < top_share(three)
        assert one.layout.height < three.layout.height
        assert right_share(rug) < right_share(one) <= right_share(three) < 0.35


def test_box_and_violin_marginals_show_outliers_as_jittered_points():
    fig = regression_figure(sample_frame(), "x", "y", marginal_x="box", marginal_y="violin")
    box = traces(fig, "box")[0]
    violin = traces(fig, "violin")[0]
    assert box.boxpoints == "outliers" and violin.points == "outliers"
    assert box.jitter > 0 and violin.jitter > 0


def test_invalid_inputs_raise_value_error():
    frame = sample_frame()
    with pytest.raises(ValueError, match="not numeric"):
        regression_figure(frame, "label", "y")
    with pytest.raises(ValueError, match="not in the data"):
        regression_figure(frame, "missing", "y")
    with pytest.raises(ValueError, match="fit method"):
        regression_figure(frame, "x", "y", methods=["spline"])
    with pytest.raises(ValueError, match="marginal"):
        regression_figure(frame, "x", "y", marginal_x="ecdf")


def test_missing_values_are_dropped_pairwise():
    frame = sample_frame()
    frame.loc[::5, "y"] = np.nan
    fig = regression_figure(frame, "x", "y", methods=["LOWESS"], marginal_x="rug")
    points = traces(fig, "scatter", "Data")[0]
    assert len(points.x) == frame["y"].notna().sum()


def regression_page():
    from views.visualization import visualization

    visualization()


def test_regression_page_renders_lowess_with_marginals():
    frame = sample_frame()
    app = AppTest.from_function(regression_page, default_timeout=30)
    app.session_state["filtered_df"] = frame
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = ["x", "y"]
    app.session_state["current_categorical_cols"] = ["group", "label"]
    app.session_state["viz_state"] = {
        "plot_type": "regression", "x": "x", "y": "y", "color": "None",
        "facet_col": "None", "facet_row": "None", "size": "None",
    }
    app.session_state["_plot_type_selector"] = "regression"
    app.run()
    assert not app.exception

    next(w for w in app.multiselect if w.label == "Fit Methods").set_value(["Polynomial", "LOWESS"])
    next(w for w in app.selectbox if w.label == "Marginal Plot for X").set_value("violin")
    app.run()
    assert not app.exception
    assert any(s.label.startswith("LOWESS Fraction") for s in app.slider)
    # Choosing only the X marginal adds it to Y as well ("Same as X" default).
    assert marginal_axes(app) == {"violin": 2}

    next(w for w in app.selectbox if w.label == "Marginal Plot for Y").set_value("rug")
    app.run()
    assert not app.exception
    assert marginal_axes(app) == {"violin": 1, "scatter": 1}

    next(w for w in app.selectbox if w.label == "Marginal Plot for Y").set_value("None")
    app.run()
    assert marginal_axes(app) == {"violin": 1}


def marginal_axes(app: AppTest) -> dict:
    """Count marginal traces (hidden from the legend, not CI bands) by trace type."""
    spec = json.loads(app.get("plotly_chart")[-1].proto.spec)
    marginal = [
        t["type"] for t in spec["data"]
        if t.get("showlegend") is False and t.get("fill") != "toself"
    ]
    return dict(Counter(marginal))


# --- numeric heatmap ---------------------------------------------------------


def heatmap_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "city": rng.choice(["Quito", "Lima"], 12),  # non-numeric: must be ignored
            "temp": rng.normal(20, 3, 12),
            "pressure": rng.normal(1000, 50, 12),
            "rain": [np.nan, *rng.uniform(0, 5, 11)],
            "const": [7.0] * 12,
            "flag": [True, False] * 6,  # boolean: not a heatmap variable
        },
        index=[f"r{i}" for i in range(12)],
    )


def test_scale_columns_per_column_with_constant_and_missing_values():
    frame = heatmap_frame()[["temp", "pressure", "rain", "const"]]
    z = scale_columns(frame, "Z-score")
    assert np.allclose(z[["temp", "pressure"]].mean(), 0) and np.allclose(z[["temp", "pressure"]].std(ddof=0), 1)
    assert (z["const"] == 0).all()
    assert z["rain"].isna().sum() == 1
    mm = scale_columns(frame, "Min-max")
    assert mm[["temp", "pressure", "rain"]].min().eq(0).all() and mm[["temp", "pressure", "rain"]].max().eq(1).all()
    assert (mm["const"] == 0.5).all()
    pd.testing.assert_frame_equal(scale_columns(frame, "None"), frame.astype(float))
    with pytest.raises(ValueError):
        scale_columns(frame, "log")


def test_numeric_heatmap_ignores_non_numeric_columns_and_keeps_raw_values_for_hover():
    frame = heatmap_frame()
    fig = numeric_heatmap_figure(frame)
    heat = fig.data[0]
    assert list(heat.y) == ["temp", "pressure", "rain", "const"]
    assert list(heat.x) == list(frame.index)
    np.testing.assert_allclose(heat.customdata, frame[["temp", "pressure", "rain", "const"]].T.to_numpy())
    np.testing.assert_allclose(np.nanmean(heat.z[:2], axis=1), 0, atol=1e-12)  # z-scored by default
    assert "Value: %{customdata" in heat.hovertemplate
    assert heat.zmid == 0


def test_numeric_heatmap_column_selection_orientation_and_validation():
    frame = heatmap_frame()
    fig = numeric_heatmap_figure(frame, columns=["rain", "temp"], scaling="None", variables_on="x")
    heat = fig.data[0]
    assert list(heat.x) == ["temp", "rain"] and list(heat.y) == list(frame.index)
    np.testing.assert_allclose(heat.z, frame[["temp", "rain"]].to_numpy())
    with pytest.raises(ValueError, match="must be numeric"):
        numeric_heatmap_figure(frame, columns=["city"])
    with pytest.raises(ValueError, match="at least one numeric"):
        numeric_heatmap_figure(frame[["city"]])
    with pytest.raises(ValueError, match="sort"):
        numeric_heatmap_figure(frame, row_order="Sort by column")


def test_numeric_heatmap_sorting_and_clustering_reorder_observations():
    frame = heatmap_frame()
    fig = numeric_heatmap_figure(frame, row_order="Sort by column", sort_by="temp", descending=True)
    assert list(fig.data[0].x) == list(frame["temp"].sort_values(ascending=False).index)

    # Two clear blocks of observations must end up contiguous after clustering.
    blocks = pd.DataFrame({"a": [0, 10, 0, 10, 0, 10], "b": [0, 10, 0, 10, 0, 10]}, index=list("pqrstu"))
    order = list(numeric_heatmap_figure(blocks, row_order="Cluster", scaling="None").data[0].x)
    assert {tuple(sorted(order[:3])), tuple(sorted(order[3:]))} == {("p", "r", "t"), ("q", "s", "u")}

    clustered = numeric_heatmap_figure(frame, cluster_columns=True)
    assert sorted(clustered.data[0].y) == sorted(["temp", "pressure", "rain", "const"])


def test_heatmap_page_renders_with_mixed_columns():
    frame = heatmap_frame().drop(columns="flag")
    app = AppTest.from_function(regression_page, default_timeout=30)
    app.session_state["filtered_df"] = frame
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = ["temp", "pressure", "rain", "const"]
    app.session_state["current_categorical_cols"] = ["city"]
    app.session_state["viz_state"] = {
        "plot_type": "heatmap", "x": "None", "y": "temp", "color": "None",
        "facet_col": "None", "facet_row": "None", "size": "None",
    }
    app.session_state["_plot_type_selector"] = "heatmap"
    app.run()
    assert not app.exception
    assert not [w.value for w in app.warning if "must be numerical" in w.value]
    spec = json.loads(app.get("plotly_chart")[-1].proto.spec)
    assert spec["data"][0]["type"] == "heatmap"
    assert spec["data"][0]["y"] == ["temp", "pressure", "rain", "const"]

    next(w for w in app.selectbox if w.label == "Observation Order").set_value("Cluster")
    next(w for w in app.radio if w.label == "Variables On").set_value("X axis")
    app.run()
    assert not app.exception
    spec = json.loads(app.get("plotly_chart")[-1].proto.spec)
    assert spec["data"][0]["x"] == ["temp", "pressure", "rain", "const"]
