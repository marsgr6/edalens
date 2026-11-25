"""Lineplot: aggregation or raw values, date X, spread around the mean, style and width."""

import base64
import json

import numpy as np
import pandas as pd
import pytest

from edalens.utils import plot_utils as pu
from test_visualization_plots import tips, viz_app


def chart(app) -> dict:
    return json.loads(app.get("plotly_chart")[-1].proto.spec)


def array(values) -> list:
    """Plotly >= 6 writes numeric arrays as base64 typed arrays; decode them to lists."""
    if isinstance(values, dict) and "bdata" in values:
        return np.frombuffer(
            base64.b64decode(values["bdata"]), dtype=np.dtype(values["dtype"])
        ).tolist()
    return list(values)


def select(app, label: str):
    return next(s for s in app.selectbox if s.label == label)


def lines(app) -> list:
    return [t for t in chart(app)["data"] if t.get("fill") != "toself"]


def readings() -> pd.DataFrame:
    """Day-first text dates, out of order, two meters, one reading per meter and day."""
    days = pd.date_range("2024-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        {
            "date": np.repeat(days.strftime("%d/%m/%Y"), 2),
            "meter": ["a", "b"] * 40,
            "kwh": np.arange(80.0),
        }
    )
    return frame.sample(frac=1, random_state=0).reset_index(drop=True)


def test_no_time_is_here_checkbox():
    app = viz_app(tips(), "lineplot", x="day", y="tip")
    app.run()
    assert not app.exception
    assert not [c for c in app.checkbox if c.label == "Time is here"]
    assert select(app, "Aggregation").value == "Mean"


def test_text_dates_are_plotted_in_date_order_on_a_time_axis():
    app = viz_app(readings(), "lineplot", x="date", y="kwh", color="meter")
    app.run()
    assert not app.exception
    fig = chart(app)
    assert fig["layout"]["xaxis"]["type"] == "date"
    a = next(t for t in lines(app) if t["name"] == "a")
    dates = pd.to_datetime(a["x"])
    assert len(dates) == 40 and dates.is_monotonic_increasing
    assert dates[0] == pd.Timestamp("2024-01-01")  # Read day first.
    assert array(a["y"]) == list(
        np.arange(0.0, 80.0, 2)
    )  # One reading per day: the mean is the value.


def test_dates_grouped_by_month_are_summed():
    app = viz_app(readings(), "lineplot", x="date", y="kwh")
    app.run()
    select(app, "Aggregation").set_value("Sum")
    app.run()
    select(app, "Group dates in date by").set_value("Month")
    app.run()
    assert not app.exception
    (trace,) = lines(app)
    assert len(trace["x"]) == 2  # January and February.
    assert sum(array(trace["y"])) == pytest.approx(sum(range(80)))
    assert chart(app)["layout"]["yaxis"]["title"]["text"] == "Sum of kwh"


def test_raw_values_keep_every_row_in_x_order():
    app = viz_app(readings(), "lineplot", x="date", y="kwh")
    app.run()
    select(app, "Aggregation").set_value(pu.RAW_VALUES)
    app.run()
    assert not app.exception
    (trace,) = lines(app)
    assert len(trace["x"]) == 80
    assert pd.to_datetime(trace["x"]).is_monotonic_increasing
    assert chart(app)["layout"]["yaxis"]["title"]["text"] == "kwh"


@pytest.mark.parametrize("shown_as, band", [("Band", True), ("Error bars", False)])
def test_spread_around_the_mean(shown_as, band):
    app = viz_app(tips(), "lineplot", x="day", y="tip", color="sex")
    app.run()
    select(app, "Spread around the mean").set_value("95% CI")
    app.run()
    next(r for r in app.radio if r.label == "Show spread as").set_value(shown_as)
    app.run()
    assert not app.exception
    data = chart(app)["data"]
    bands = [t for t in data if t.get("fill") == "toself"]
    assert len(bands) == (2 if band else 0)
    assert all(("error_y" in t) != band for t in lines(app))


def test_x_order_is_shared_by_every_line():
    frame = tips()
    frame = frame[
        ~((frame["sex"] == "F") & (frame["day"] == "Thu"))
    ]  # F has no Thursday.
    app = viz_app(frame, "lineplot", x="day", y="tip", color="sex")
    app.run()
    assert not app.exception
    axis = chart(app)["layout"]["xaxis"]
    assert axis["type"] == "category" and axis["categoryarray"] == [
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    ]  # Calendar order.


def test_style_splits_lines_by_dash():
    frame = tips()
    frame["smoker"] = np.where(frame.index % 2, "yes", "no")
    app = viz_app(frame, "lineplot", x="day", y="tip", color="sex")
    app.run()
    select(app, "Line Style (Categorical)").set_value("smoker")
    app.run()
    assert not app.exception
    traces = lines(app)
    assert len(traces) == 4
    assert {t["line"]["dash"] for t in traces} == {"solid", "dash"}
    assert {t["name"] for t in traces} == {"M, no", "M, yes", "F, no", "F, yes"}


def test_width_follows_each_lines_mean():
    app = viz_app(tips(), "lineplot", x="day", y="tip", color="sex")
    app.run()
    select(app, "Line Width (Numeric)").set_value("bill")
    app.run()
    assert not app.exception
    assert sorted(t["line"]["width"] for t in lines(app)) == [1.0, 6.0]


def test_line_summary_interval_needs_two_values():
    frame = pd.DataFrame({"x": ["a", "a", "b"], "y": [1.0, 3.0, 5.0]})
    out = pu.line_summary(frame, "x", "y", interval="SD")
    assert out["value"].tolist() == [2.0, 5.0]
    assert out["lower"].iloc[0] == pytest.approx(2 - np.sqrt(2)) and np.isnan(
        out["lower"].iloc[1]
    )


def test_numeric_x_is_offered_and_plotted_in_numeric_order():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "dose": rng.choice([10, 2, 5, 20], 120),
            "group": rng.choice(["a", "b"], 120),
            "response": rng.normal(0, 1, 120),
        }
    )
    app = viz_app(frame, "lineplot", x="group", y="response")
    app.run()
    assert "dose" in select(app, "X").options
    select(app, "X").set_value("dose")
    app.run()
    assert not app.exception
    (trace,) = lines(app)
    assert array(trace["x"]) == [2, 5, 10, 20]
    assert chart(app)["layout"]["xaxis"].get("type") in (None, "linear", "-")
    expected = frame.groupby("dose")["response"].mean()
    assert array(trace["y"]) == pytest.approx(expected.tolist())
