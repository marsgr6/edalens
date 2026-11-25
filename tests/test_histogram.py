"""Histogram: the step element draws outlines in the app palette, with facets."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from streamlit.testing.v1 import AppTest

from edalens.utils.plot_utils import step_histogram_frame

PALETTE = px.colors.qualitative.Set2


def visualization_page():
    from edalens.views.visualization import visualization

    visualization()


def tips() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        {
            "tip": rng.gamma(2, 1.5, n),
            "sex": rng.choice(["M", "F"], n),
            "time": rng.choice(["Lunch", "Dinner"], n),
        }
    )


def hist_app(**viz) -> AppTest:
    frame = tips()
    app = AppTest.from_function(visualization_page, default_timeout=30)
    app.session_state["filtered_df"] = frame
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = ["tip"]
    app.session_state["current_categorical_cols"] = ["sex", "time"]
    app.session_state["viz_state"] = {
        "plot_type": "histogram",
        "x": "tip",
        "y": "None",
        "color": "None",
        "facet_col": "None",
        "facet_row": "None",
        "size": "None",
        **viz,
    }
    app.session_state["_plot_type_selector"] = "histogram"
    return app


def chart(app: AppTest) -> dict:
    return json.loads(app.get("plotly_chart")[-1].proto.spec)


def select(app: AppTest, label: str):
    return next(s for s in app.selectbox if s.label == label)


def colors(spec: dict) -> dict:
    out = {}
    for trace in spec["data"]:
        color = (trace.get("line") or {}).get("color") or (
            trace.get("marker") or {}
        ).get("color")
        out.setdefault(trace["name"], color)
    return out


def test_step_uses_the_same_palette_as_bars():
    app = hist_app(color="sex")
    app.run()
    bar_colors = colors(chart(app))
    select(app, "Element").set_value("step")
    app.run()
    assert not app.exception
    spec = chart(app)
    assert {t["type"] for t in spec["data"]} == {"scatter"}
    assert all(t["line"]["shape"] == "hv" for t in spec["data"])
    assert colors(spec) == bar_colors
    assert set(bar_colors.values()) <= set(PALETTE)


def test_step_without_a_hue_is_not_plotly_blue():
    app = hist_app()
    app.run()
    select(app, "Element").set_value("step")
    app.run()
    line = chart(app)["data"][0]["line"]["color"]
    assert line == PALETTE[0]


def test_step_supports_facets():
    app = hist_app(color="sex", facet_col="time")
    app.run()
    select(app, "Element").set_value("step")
    app.run()
    assert not app.exception
    titles = {a["text"] for a in chart(app)["layout"]["annotations"]}
    assert titles == {"time=Lunch", "time=Dinner"}
    assert len(chart(app)["data"]) == 4  # 2 colors x 2 panels.


@pytest.mark.parametrize("stat, total", [("probability", 1.0), ("percent", 100.0)])
def test_cumulative_step_ends_at_the_whole(stat, total):
    steps = step_histogram_frame(tips(), "tip", "sex", stat=stat, cumulative=True)
    tops = steps.groupby("sex")["value"].max()
    assert tops.to_dict() == pytest.approx({"M": total, "F": total})


def test_density_steps_have_unit_area():
    frame = tips()
    steps = step_histogram_frame(frame, "tip", stat="density")
    edges = np.histogram_bin_edges(frame["tip"], bins="auto")
    heights = steps["value"].to_numpy()[1:-2]
    assert (heights * np.diff(edges)).sum() == pytest.approx(1.0)


def test_stacked_steps_add_up_within_a_panel():
    frame = tips()
    layered = step_histogram_frame(frame, "tip", "sex")
    stacked = step_histogram_frame(frame, "tip", "sex", stacked=True)
    top_color = stacked["sex"].iloc[-1]
    whole = step_histogram_frame(frame, "tip")
    assert stacked.loc[stacked["sex"] == top_color, "value"].max() == pytest.approx(
        whole["value"].max()
    )
    assert layered["value"].max() < stacked["value"].max()


def test_step_fill_opacity():
    app = hist_app(color="sex")
    app.run()
    select(app, "Element").set_value("step")
    app.run()
    slider = next(s for s in app.slider if s.label == "Fill opacity")
    assert slider.value == 0.3
    fills = {t["name"]: t.get("fillcolor") for t in chart(app)["data"]}
    assert all(t.get("fill") == "toself" for t in chart(app)["data"])
    # Each fill is its outline's color at the chosen opacity.
    assert (
        fills["F"] == "rgba(102,194,165,0.3)" and fills["M"] == "rgba(252,141,98,0.3)"
    )
    slider.set_value(0.0)
    app.run()
    assert all(t.get("fill") in (None, "none") for t in chart(app)["data"])


def test_bars_have_no_fill_option():
    app = hist_app()
    app.run()
    assert not [s for s in app.slider if s.label == "Fill opacity"]


@pytest.mark.parametrize(
    "multiple, barmode, shared",
    [("layer", "overlay", True), ("dodge", "group", False), ("stack", "stack", True)],
)
def test_multiple_modes_overlap_dodge_and_stack(multiple, barmode, shared):
    app = hist_app(color="sex")
    app.run()
    select(app, "Multiple").set_value(multiple)
    app.run()
    assert not app.exception  # "dodge" used to raise: it is "group" to Plotly.
    spec = chart(app)
    assert spec["layout"]["barmode"] == barmode
    groups = {t.get("offsetgroup") or "" for t in spec["data"]}
    # One shared offset group overlaps or stacks the colors; none lets them dodge.
    assert groups == ({"hist"} if shared else {""})
    assert all(not t.get("alignmentgroup") for t in spec["data"])


def test_layer_bars_are_translucent_by_default_and_adjustable():
    app = hist_app(color="sex")
    app.run()
    slider = next(s for s in app.slider if s.label == "Bar opacity")
    assert slider.value == 0.6
    assert {t["opacity"] for t in chart(app)["data"]} == {0.6}
    slider.set_value(0.35)
    app.run()
    assert {t["opacity"] for t in chart(app)["data"]} == {0.35}
    select(app, "Multiple").set_value("stack")
    app.run()
    assert next(s for s in app.slider if s.label == "Bar opacity").value == 1.0


def array(values) -> list:
    """Plotly >= 6 writes numeric arrays as base64 typed arrays; decode them to lists."""
    if isinstance(values, dict) and "bdata" in values:
        import base64

        return np.frombuffer(
            base64.b64decode(values["bdata"]), dtype=np.dtype(values["dtype"])
        ).tolist()
    return list(values)


def check(app: AppTest, label: str):
    return next(c for c in app.checkbox if c.label == label)


@pytest.mark.parametrize("common_norm", [False, True])
def test_step_common_norm_shares_one_total_per_panel(common_norm):
    frame = tips()
    steps = step_histogram_frame(
        frame, "tip", "sex", ("time",), stat="probability", common_norm=common_norm
    )
    # Each outline repeats its last height once and opens/closes at 0; the bins are the middle points.
    totals = steps.groupby(["time", "sex"])["value"].apply(lambda v: v.iloc[1:-2].sum())
    shares = frame.groupby(["time", "sex"]).size() / frame.groupby("time").size()
    expected = shares if common_norm else shares * 0 + 1
    assert totals.to_dict() == pytest.approx(expected.to_dict())


@pytest.mark.parametrize("stat, total", [("probability", 1), ("percent", 100)])
def test_bars_with_common_norm_add_up_to_one_total(stat, total):
    app = hist_app(color="sex")
    app.run()
    select(app, "Stat").set_value(stat)
    app.run()
    check(app, "Common Norm").check()
    app.run()
    assert not app.exception
    traces = chart(app)["data"]
    sums = {t["name"]: sum(array(t["y"])) for t in traces}
    shares = tips()["sex"].value_counts(normalize=True) * total
    assert sums == pytest.approx(shares.to_dict())
    assert {t["histfunc"] for t in traces} == {"sum"}
    assert chart(app)["layout"]["yaxis"]["title"]["text"] == stat.capitalize()


def test_common_norm_needs_a_color_and_a_normalized_stat():
    app = hist_app()
    app.run()
    select(app, "Stat").set_value("probability")
    app.run()
    assert check(app, "Common Norm").disabled  # No color.
    app = hist_app(color="sex")
    app.run()
    assert check(app, "Common Norm").disabled  # Counts are not normalized.


def area(x: list, y: list) -> float:
    """Trapezoid rule (np.trapezoid needs numpy 2; CI pins numpy 1.26)."""
    x, y = np.asarray(x), np.asarray(y)
    return float(((y[1:] + y[:-1]) / 2 * np.diff(x)).sum())


@pytest.mark.parametrize("common_norm", [False, True])
def test_density_curves_share_their_panels_area(common_norm):
    app = hist_app(color="sex")
    app.session_state["viz_state"]["plot_type"] = "density 1"
    app.session_state["_plot_type_selector"] = "density 1"
    app.run()
    if common_norm:
        check(app, "Common Norm").check()
        app.run()
    assert not app.exception
    areas = {t["name"]: area(array(t["x"]), array(t["y"])) for t in chart(app)["data"]}
    shares = tips()["sex"].value_counts(normalize=True)
    expected = shares if common_norm else shares * 0 + 1
    # The curves are drawn over the data range, so a little of each tail is cut off.
    assert areas == pytest.approx(expected.to_dict(), abs=0.06)
