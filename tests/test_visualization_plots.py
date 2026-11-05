"""Streamlit-level checks: facets, candlestick, waterfall, 2-D density rug, PDF export."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from edalens.utils import plot_utils as pu


def visualization_page():
    from edalens.views.visualization import visualization

    visualization()


def viz_app(frame: pd.DataFrame, plot_type: str, **viz) -> AppTest:
    app = AppTest.from_function(visualization_page, default_timeout=30)
    app.session_state["filtered_df"] = frame
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = frame.select_dtypes(
        "number"
    ).columns.tolist()
    app.session_state["current_categorical_cols"] = [
        c for c in frame.columns if c not in app.session_state["current_numeric_cols"]
    ]
    app.session_state["viz_state"] = {
        "plot_type": plot_type,
        "x": None,
        "y": None,
        "color": "None",
        "facet_col": "None",
        "facet_row": "None",
        "size": "None",
        **viz,
    }
    app.session_state["_plot_type_selector"] = plot_type
    return app


def chart(app: AppTest) -> dict:
    return json.loads(app.get("plotly_chart")[-1].proto.spec)


def tips() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "day": rng.choice(["Thu", "Fri", "Sat", "Sun"], n),
            "sex": rng.choice(["M", "F"], n),
            "tip": rng.normal(3, 1, n),
            "bill": rng.normal(20, 5, n),
        }
    )


def prices() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    price = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 60))
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "month": dates.strftime("%b"),
            "price": price,
        }
    )


def test_row_facets_leave_room_for_every_panels_axis_titles():
    app = viz_app(tips(), "lineplot", x="day", y="tip", facet_row="sex")
    app.run()
    assert not app.exception
    layout = chart(app)["layout"]
    assert (
        layout["xaxis"]["title"]["text"] == layout["xaxis2"]["title"]["text"] == "day"
    )
    # The gap between rows holds the upper x-axis title and the lower facet title.
    upper_bottom, lower_top = (
        layout["yaxis"]["domain"][0],
        layout["yaxis2"]["domain"][1],
    )
    assert (upper_bottom - lower_top) * layout["height"] >= 100


def test_lineplot_keeps_lines_whose_first_step_has_one_observation():
    frame = tips()
    frame.loc[
        frame.index[(frame["sex"] == "F") & (frame["day"] == "Fri")][1:], "day"
    ] = "Sat"
    app = viz_app(frame, "lineplot", x="day", y="tip", facet_row="sex")
    app.run()
    assert len(chart(app)["data"]) == 2  # One line per panel.


def test_candlestick_from_one_value_column_sorts_iso_dates():
    app = viz_app(prices(), "candlestick", x="month", y="price")
    app.run()
    assert not app.exception
    trace = chart(app)["data"][0]
    assert trace["type"] == "candlestick"
    assert trace["x"] == ["Jan", "Feb", "Mar"]  # Order of appearance for text.


def test_waterfall_has_a_total_bar():
    app = viz_app(tips(), "waterfall", x="day", y="tip")
    app.run()
    assert not app.exception
    trace = chart(app)["data"][0]
    assert trace["type"] == "waterfall"
    assert trace["measure"][-1] == "total" and trace["x"][-1] == "Total"


@pytest.mark.parametrize("kind", ["hist", "kde"])
def test_density_2_rug_is_on_both_axes(kind):
    app = viz_app(tips(), "density 2", x="tip", y="bill")
    app.run()
    next(s for s in app.selectbox if s.label == "Kind").set_value(kind)
    next(c for c in app.checkbox if c.label == "Add Rug Plot?").check()
    app.run()
    assert not app.exception
    rugs = [t for t in chart(app)["data"] if t["type"] == "box"]
    assert {"x" if "x" in t else "y" for t in rugs} == {"x", "y"}


def zone(app: AppTest, label: str):
    return next(s for s in app.selectbox if s.label == label)


def test_facet_row_follows_every_change():
    frame = tips()
    frame["smoker"] = np.where(frame.index % 3, "yes", "no")
    app = viz_app(frame, "lineplot", x="day", y="tip")
    app.run()
    for value in ["sex", "smoker", "sex", "None"]:
        zone(app, "Facet Row").set_value(value)
        app.run()
        notes = [a["text"] for a in chart(app)["layout"].get("annotations", [])]
        assert zone(app, "Facet Row").value == value
        assert all(n.startswith(f"{value}: ") for n in notes) and (
            bool(notes) == (value != "None")
        )
        # Another control rerunning the page keeps the choice.
        next(c for c in app.checkbox if c.label == "Show Markers").uncheck()
        app.run()
        assert zone(app, "Facet Row").value == value
        next(c for c in app.checkbox if c.label == "Show Markers").check()
        app.run()


def test_changing_the_plot_moves_x_and_y_to_valid_columns():
    app = viz_app(tips(), "lineplot", x="day", y="tip")
    app.run()
    zone(app, "Plot").set_value("scatter")
    app.run()
    assert not app.exception
    assert zone(app, "X").value in ("tip", "bill") and zone(app, "Y").value in (
        "tip",
        "bill",
    )


def test_plots_without_facets_say_so():
    app = viz_app(tips(), "regression", x="tip", y="bill", facet_row="sex")
    app.run()
    assert any("does not use Facet" in i.value for i in app.info)


def test_prepare_pdf_offers_the_rendered_chart(monkeypatch):
    from edalens.utils import ui

    monkeypatch.setattr(
        ui, "figure_pdf", lambda fig, context: b"%PDF-" + context.encode()
    )
    app = viz_app(tips(), "waterfall", x="day", y="tip")
    app.run()
    next(b for b in app.button if b.label == "📄 Prepare PDF").click()
    app.run()
    assert not app.exception
    download = app.get("download_button")[-1].proto
    assert download.label == "⬇️ Download PDF"


def test_pdf_button_is_hidden_in_the_browser(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")
    app = viz_app(tips(), "waterfall", x="day", y="tip")
    app.run()
    assert not app.exception
    assert not [b for b in app.button if b.label == "📄 Prepare PDF"]


def test_ohlc_from_columns_combines_rows_per_step():
    frame = pd.DataFrame(
        {
            "d": ["2023-01-02", "2023-01-01", "2023-01-02"],
            "o": [10, 5, 11],
            "h": [12, 6, 15],
            "l": [9, 4, 8],
            "c": [11, 5.5, 14],
        }
    )
    ohlc = pu.ohlc_frame(frame, "d", ohlc=["o", "h", "l", "c"])
    assert list(ohlc.index.strftime("%Y-%m-%d")) == ["2023-01-01", "2023-01-02"]
    assert ohlc.loc["2023-01-02"].tolist() == [10, 15, 8, 14]


def test_ohlc_from_one_value_column():
    frame = pd.DataFrame({"g": ["a", "a", "a", "b"], "v": [3, 7, 1, 2]})
    ohlc = pu.ohlc_frame(frame, "g", value="v")
    assert ohlc.loc["a"].tolist() == [3, 7, 1, 1]


@pytest.mark.parametrize(
    "aggregation, expected", [("Sum", [4, -2, 4]), ("Mean", [2, -2, 4])]
)
def test_waterfall_changes_follow_custom_order(aggregation, expected):
    frame = pd.DataFrame({"k": ["b", "a", "b", "c"], "y": [1, -2, 3, 4]})
    frame["k"] = pd.Categorical(frame["k"], categories=["b", "a", "c"], ordered=True)
    changes = pu.waterfall_frame(frame, "k", "y", aggregation)
    assert changes.index.tolist() == ["b", "a", "c"] and changes.tolist() == expected


@pytest.mark.parametrize("plot", ["lineplot", "bars"])
def test_time_is_here_accepts_text_dates_on_every_pandas(plot):
    # pandas 3 stores text as dtype "str", not "object"; it must still count as dates text.
    dates = pd.date_range("2024-01-01", periods=30, freq="D").strftime("%Y-%m-%d")
    frame = pd.DataFrame({"date": dates, "kind": ["a", "b"] * 15, "v": np.arange(30.0)})
    app = viz_app(frame, plot, x="date", y="v")
    app.run()
    next(c for c in app.checkbox if c.label == "Time is here").check()
    app.run()
    assert not app.exception
    assert not [w.value for w in app.warning if "requires" in w.value]
    assert len(chart(app)["data"][0]["x"]) == 30
