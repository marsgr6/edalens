"""Bars: Y None counts rows per X; a numeric Y is aggregated per X; percentages."""

import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from edalens.utils.plot_utils import bar_summary


def visualization_page():
    from edalens.views.visualization import visualization

    visualization()


def energy() -> pd.DataFrame:
    """Two years of monthly consumption, two meters: several rows per month."""
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    rows = []
    for i, month in enumerate(months):
        for meter, base in (("A", 100.0), ("B", 50.0)):
            rows.append(
                {
                    "month": month.strftime("%m"),
                    "year": str(month.year),
                    "meter": meter,
                    "kwh": base + i,
                }
            )
    return pd.DataFrame(rows)


def bars_app(frame: pd.DataFrame, **viz) -> AppTest:
    app = AppTest.from_function(visualization_page, default_timeout=30)
    numeric = frame.select_dtypes("number").columns.tolist()
    app.session_state["filtered_df"] = frame
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = numeric
    app.session_state["current_categorical_cols"] = [
        c for c in frame.columns if c not in numeric
    ]
    app.session_state["viz_state"] = {
        "plot_type": "bars",
        "x": "month",
        "y": "None",
        "color": "None",
        "facet_col": "None",
        "facet_row": "None",
        "size": "None",
        **viz,
    }
    app.session_state["_plot_type_selector"] = "bars"
    return app


def chart(app: AppTest) -> dict:
    return json.loads(app.get("plotly_chart")[-1].proto.spec)


def array(values) -> list:
    """Plotly >= 6 writes numeric arrays as base64 typed arrays; decode them to lists."""
    if isinstance(values, dict) and "bdata" in values:
        import base64

        data = np.frombuffer(
            base64.b64decode(values["bdata"]), dtype=np.dtype(values["dtype"])
        )
        if "shape" in values:
            data = data.reshape([int(n) for n in str(values["shape"]).split(",")])
        return data.tolist()
    return list(values)


def select(app: AppTest, label: str):
    return next(s for s in app.selectbox if s.label == label)


def bar_values(app: AppTest) -> dict:
    """{(trace name, x): height} for every bar."""
    out = {}
    for trace in chart(app)["data"]:
        for x, y in zip(array(trace["x"]), array(trace["y"])):
            out[(trace.get("name") or "", x)] = y
    return out


def test_without_y_bars_count_rows_per_x():
    app = bars_app(energy())
    app.run()
    assert not app.exception
    values = bar_values(app)
    assert set(values.values()) == {4}  # 2 years x 2 meters per month.
    assert chart(app)["layout"]["yaxis"]["title"]["text"] == "Count"
    assert any("count the rows" in c.value for c in app.caption)


def test_numeric_y_is_summed_per_x_and_replaces_time_is_here():
    app = bars_app(energy(), y="kwh")
    app.run()
    assert not app.exception
    assert not [c for c in app.checkbox if c.label == "Time is here"]
    frame = energy()
    expected = frame.groupby("month")["kwh"].sum()
    values = bar_values(app)
    assert {x: v for (_, x), v in values.items()} == pytest.approx(expected.to_dict())
    assert chart(app)["layout"]["yaxis"]["title"]["text"] == "Sum of kwh"


def test_one_row_per_x_shows_the_values_themselves():
    frame = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "kwh": [310.0, 280.0, 295.0]})
    app = bars_app(frame, y="kwh")
    app.run()
    assert {x: v for (_, x), v in bar_values(app).items()} == {
        "Jan": 310,
        "Feb": 280,
        "Mar": 295,
    }


def test_percent_within_x_with_a_stacked_color():
    app = bars_app(energy(), y="kwh", color="meter")
    app.run()
    select(app, "Show as").set_value("Percent within X")
    next(r for r in app.radio if r.label == "Bar layout").set_value("Stacked")
    app.run()
    assert not app.exception
    spec = chart(app)
    assert spec["layout"]["barmode"] == "relative"
    # One offset group: the colors stack instead of standing side by side.
    assert len({t.get("offsetgroup") for t in spec["data"]}) == 1
    per_month = {}
    for (_, x), v in bar_values(app).items():
        per_month[x] = per_month.get(x, 0) + v
    assert all(v == pytest.approx(100) for v in per_month.values())


def test_mean_offers_error_bars_and_no_percentages():
    app = bars_app(energy(), y="kwh")
    app.run()
    select(app, "Aggregate kwh per bar").set_value("Mean")
    app.run()
    assert not [s for s in app.selectbox if s.label == "Show as"]
    select(app, "Error bars").set_value("95% CI")
    app.run()
    assert not app.exception
    assert array(chart(app)["data"][0]["error_y"]["array"])


def test_heatmap_crosses_x_with_the_color():
    app = bars_app(energy(), y="kwh", color="meter")
    app.run()
    select(app, "Plot Type").set_value("heatmap")
    app.run()
    assert not app.exception
    trace = chart(app)["data"][0]
    assert trace["type"] == "heatmap" and len(array(trace["z"])) == 12


def test_non_numeric_y_explains_itself():
    app = bars_app(energy(), y="year")
    app.session_state["risk_it_all"] = True
    app.run()
    assert any("numeric Y" in w.value for w in app.warning)


def test_summary_percent_of_total_is_per_facet_panel():
    frame = energy()
    summary = bar_summary(
        frame, "month", facets=("year", None), scale="Percent of total"
    )
    assert summary.groupby("year")["value"].sum().to_dict() == pytest.approx(
        {"2023": 100, "2024": 100}
    )


def test_summary_counts_skip_rows_missing_a_group():
    frame = pd.DataFrame({"x": ["a", "a", None, "b"], "hue": ["p", None, "q", "q"]})
    summary = bar_summary(frame, "x", hue="hue")
    assert sorted(map(tuple, summary.to_numpy().tolist())) == [
        ("a", "p", 1),
        ("b", "q", 1),
    ]


def test_summary_sd_error_bars():
    frame = pd.DataFrame({"x": ["a"] * 3, "v": [1.0, 2.0, 3.0]})
    summary = bar_summary(frame, "x", "v", aggregation="Mean", error="SD")
    assert summary.loc[0, "error"] == pytest.approx(np.std([1, 2, 3], ddof=1))


def test_month_and_weekday_names_start_in_calendar_order():
    from edalens.views.visualization import natural_sort

    assert natural_sort(["Mar", "Jan", "Dec", "Feb"]) == ["Jan", "Feb", "Mar", "Dec"]
    assert natural_sort(["sunday", "Monday", "Fri"]) == ["Monday", "Fri", "sunday"]
    assert natural_sort(["September", "Sept", "jan"]) == ["jan", "September", "Sept"]
    assert natural_sort(["May", "Maybe"]) == [
        "May",
        "Maybe",
    ]  # Not all names: alphabetical.
    assert natural_sort([3, 1, 2]) == [1, 2, 3]


def test_bars_over_month_names_follow_the_calendar():
    frame = pd.DataFrame(
        {"month": ["Mar", "Jan", "Feb", "Jan"], "kwh": [3.0, 1.0, 2.0, 1.0]}
    )
    app = bars_app(frame, y="kwh")
    app.run()
    trace = chart(app)["data"][0]
    order = chart(app)["layout"]["xaxis"].get("categoryarray") or array(trace["x"])
    assert list(order) == ["Jan", "Feb", "Mar"]
    assert dict(zip(array(trace["x"]), array(trace["y"]))) == {
        "Jan": 2,
        "Feb": 2,
        "Mar": 3,
    }


def daily_energy() -> pd.DataFrame:
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    return pd.DataFrame(
        {
            "date": days.strftime("%Y-%m-%d"),
            "meter": ["A", "B"] * (len(days) // 2) + ["A"] * (len(days) % 2),
            "kwh": 1.0,
        }
    )


@pytest.mark.parametrize("as_datetime", [False, True])
def test_a_date_x_groups_by_month_on_a_time_axis(as_datetime):
    frame = daily_energy()
    if as_datetime:  # A real datetime column is offered as X too.
        frame["date"] = pd.to_datetime(frame["date"])
    app = bars_app(frame, x="date", y="kwh")
    app.run()
    assert not app.exception
    assert "date" in next(s for s in app.selectbox if s.label == "X").options
    select(app, "Group dates in date by").set_value("Month")
    app.run()
    assert not app.exception
    spec = chart(app)
    trace = spec["data"][0]
    assert len(array(trace["x"])) == 24
    assert array(trace["x"])[0].startswith("2023-01-01")
    assert (
        sorted(array(trace["y"]))[0] == 28 and max(array(trace["y"])) == 31
    )  # Days per month, kWh = 1 each.
    assert spec["layout"]["xaxis"]["tickformat"] == "%b %Y"
    assert spec["layout"]["title"]["text"].startswith("Sum of kwh by date (month)")


def test_mean_per_year_with_error_bars_on_a_date_x():
    app = bars_app(daily_energy(), x="date", y="kwh")
    app.run()
    select(app, "Group dates in date by").set_value("Year")
    select(app, "Aggregate kwh per bar").set_value("Mean")
    app.run()
    select(app, "Error bars").set_value("SD")
    app.run()
    assert not app.exception
    trace = chart(app)["data"][0]
    assert array(trace["y"]) == [1.0, 1.0] and "error_y" in trace


def test_day_first_only_for_text_dates():
    app = bars_app(pd.DataFrame({"month": ["Jan", "Feb"], "kwh": [1.0, 2.0]}), y="kwh")
    app.run()
    assert not [c for c in app.checkbox if c.label.startswith("Day First")]
    assert not [s for s in app.selectbox if s.label.startswith("Group dates")]
    frame = pd.DataFrame(
        {"day": ["05/01/2024", "06/01/2024", "13/01/2024"], "kwh": [1.0, 2.0, 3.0]}
    )
    app = bars_app(frame, x="day", y="kwh")
    app.run()
    assert next(c for c in app.checkbox if c.label.startswith("Day First")).value
    assert array(chart(app)["data"][0]["x"])[0].startswith("2024-01-05")


def test_heatmap_rows_are_readable_grouped_dates():
    app = bars_app(daily_energy(), x="date", y="kwh", color="meter")
    app.run()
    select(app, "Plot Type").set_value("heatmap")
    select(app, "Group dates in date by").set_value("Quarter")
    app.run()
    assert not app.exception
    rows = array(chart(app)["data"][0]["y"])
    assert rows[:2] == ["2023 Q1", "2023 Q2"] and len(rows) == 8


def test_bars_open_as_a_count_plot():
    frame = energy()
    app = AppTest.from_function(visualization_page, default_timeout=30)
    app.session_state["filtered_df"] = frame
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = ["kwh"]
    app.session_state["current_categorical_cols"] = ["month", "year", "meter"]
    app.run()  # No saved plot state: the page's own defaults.
    assert not app.exception
    assert select(app, "Plot").value == "bars"
    assert (select(app, "X").value, select(app, "Y").value) == ("month", "None")
    assert chart(app)["layout"]["yaxis"]["title"]["text"] == "Count"


def test_switching_to_bars_counts_even_after_a_numeric_y():
    app = bars_app(energy(), plot_type="lineplot", y="kwh")
    app.session_state["_plot_type_selector"] = "lineplot"
    app.run()
    select(app, "Plot").set_value("bars")
    app.run()
    assert select(app, "Y").value == "None"
    assert chart(app)["layout"]["yaxis"]["title"]["text"] == "Count"
    select(app, "Y").set_value("kwh")  # A numeric Y is one pick away.
    app.run()
    assert chart(app)["layout"]["yaxis"]["title"]["text"] == "Sum of kwh"


def test_facet_panels_split_their_width_only_among_their_own_bars():
    # px aligns grouped bars across facets, reserving a slot for every color in every
    # panel (thin bars); each panel should only group the colors it shows.
    frame = pd.DataFrame(
        {
            "Year": ["2019", "2020"] * 3,
            "University": ["U1", "U1", "U2", "U2", "U3", "U3"],
            "Ranking": ["1", "1", "2", "2", "2", "2"],
            "Publications": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    app = bars_app(
        frame, x="Year", y="Publications", color="University", facet_col="Ranking"
    )
    app.run()
    assert not app.exception
    spec = chart(app)
    assert all(
        not t.get("alignmentgroup") and not t.get("offsetgroup") for t in spec["data"]
    )
    assert spec["layout"]["bargap"] == 0.15


def test_labels_are_one_size_and_horizontal_when_bars_are_few():
    from edalens.utils.plot_utils import bar_figure

    few = bar_summary(energy(), "month", "kwh")  # 12 bars.
    fig = bar_figure(few, "month", "Sum of kwh")
    trace = fig.data[0]
    assert (trace.textposition, trace.textangle, trace.constraintext) == (
        "outside",
        0,
        "none",
    )
    assert trace.textfont.size == 11
    assert fig.layout.yaxis.range[1] > few["value"].max()  # Room for the top label.


def test_labels_turn_vertical_when_bars_are_many_and_stay_inside_when_stacked():
    from edalens.utils.plot_utils import bar_figure

    many = bar_summary(energy(), "month", "kwh", hue="year", facets=("meter", None))
    fig = bar_figure(many, "month", "Sum of kwh", hue="year", facet_col="meter")
    assert {t.textangle for t in fig.data} == {-90}  # 24 bars per panel x 2 panels.
    stacked = bar_figure(
        many, "month", "Sum of kwh", hue="year", facet_col="meter", stacked=True
    )
    assert {t.textposition for t in stacked.data} == {"inside"}
    assert stacked.layout.uniformtext.mode == "hide"
