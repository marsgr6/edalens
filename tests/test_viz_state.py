"""Plot controls keep X/Y/hue and custom category orders across plot types."""

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def visualization_page():
    from edalens.views.visualization import visualization

    visualization()


def tips() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            "day": rng.choice(["Thu", "Fri", "Sat", "Sun"], n),
            "sex": rng.choice(["M", "F"], n),
            "tip": rng.normal(3, 1, n),
            "bill": rng.normal(20, 5, n),
        }
    )


def viz_app(**viz) -> AppTest:
    frame = tips()
    app = AppTest.from_function(visualization_page, default_timeout=30)
    app.session_state["filtered_df"] = frame
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = ["tip", "bill"]
    app.session_state["current_categorical_cols"] = ["day", "sex"]
    app.session_state["viz_state"] = {
        "plot_type": "lineplot",
        "x": "day",
        "y": "tip",
        "color": "None",
        "facet_col": "None",
        "facet_row": "None",
        "size": "None",
        **viz,
    }
    app.session_state["_plot_type_selector"] = app.session_state["viz_state"][
        "plot_type"
    ]
    return app


def zone(app: AppTest, label: str):
    return next(s for s in app.selectbox if s.label == label)


def switch(app: AppTest, plot: str) -> None:
    zone(app, "Plot").set_value(plot)
    app.run()
    assert not app.exception, app.exception


def mapping(app: AppTest) -> tuple:
    return tuple(zone(app, name).value for name in ("X", "Y", "Color"))


def order_widget(app: AppTest, column: str):
    """The "Custom Order" multiselect for ``column`` (labels end in ": col" or "(col)")."""
    return next(
        m
        for m in app.multiselect
        if m.label.startswith("Custom Order")
        and (m.label.endswith(f": {column}") or m.label.endswith(f"({column})"))
    )


def test_x_y_and_hue_come_back_after_plots_that_cannot_use_them():
    app = viz_app(color="sex")
    app.run()
    assert mapping(app) == ("day", "tip", "sex")
    switch(
        app, "scatter"
    )  # Needs a numeric X: tip moves over, Y takes the other numeric.
    assert mapping(app) == ("tip", "bill", "sex")
    switch(app, "histogram")  # No Y.
    assert mapping(app) == ("tip", "None", "sex")
    switch(app, "lineplot")
    assert mapping(app) == ("day", "tip", "sex")
    switch(app, "boxes")
    assert mapping(app) == ("day", "tip", "sex")
    switch(app, "bars")  # Bars keeps Y for its "Time is here" mode.
    assert mapping(app) == ("day", "tip", "sex")


def test_explicit_picks_are_remembered_per_role():
    app = viz_app()
    app.run()
    zone(app, "X").set_value("sex")
    app.run()
    zone(app, "Y").set_value("bill")
    app.run()
    switch(app, "scatter")
    assert mapping(app)[:2] == ("bill", "tip")
    switch(app, "catplot")
    assert mapping(app)[:2] == ("sex", "bill")


def test_custom_order_follows_the_column_across_roles_and_plots():
    app = viz_app(color="sex")
    app.run()
    order_widget(app, "sex").set_value(["M", "F"])
    app.run()
    assert order_widget(app, "sex").value == ["M", "F"]
    # Another control rerunning the page keeps it.
    next(c for c in app.checkbox if c.label == "Show Markers").uncheck()
    app.run()
    assert order_widget(app, "sex").value == ["M", "F"]
    for plot in ("boxes", "bars", "catplot"):
        switch(app, plot)
        assert order_widget(app, "sex").value == ["M", "F"], plot
    # The order of X too, including a category removed from it.
    order_widget(app, "day").set_value(["Sun", "Sat", "Fri"])
    app.run()
    switch(app, "lineplot")
    assert order_widget(app, "day").value == ["Sun", "Sat", "Fri"]


def test_new_data_starts_with_fresh_orders():
    app = viz_app(color="sex")
    app.run()
    order_widget(app, "sex").set_value(["M", "F"])
    app.run()
    app.session_state["data_version"] = (
        app.session_state["data_version"] + 1
        if "data_version" in app.session_state
        else 1
    )
    app.run()
    assert order_widget(app, "sex").value == ["F", "M"]


def chart(app: AppTest) -> dict:
    import json

    return json.loads(app.get("plotly_chart")[-1].proto.spec)


def hue_trace_order(app: AppTest) -> list:
    names = [t.get("name") for t in chart(app)["data"] if t.get("name") in ("M", "F")]
    return list(dict.fromkeys(names))


def test_traces_and_panels_follow_the_custom_order():
    app = viz_app(color="sex")
    app.run()
    order_widget(app, "sex").set_value(["M", "F"])
    app.run()
    for plot in ("lineplot", "bars", "boxes", "histogram", "catplot"):
        if plot != "lineplot":
            switch(app, plot)
        assert hue_trace_order(app) == ["M", "F"], plot
    # Facet panels too: the first row is the first category.
    switch(app, "lineplot")
    zone(app, "Color").set_value("None")
    zone(app, "Facet Row").set_value("sex")
    app.run()
    titles = [a["text"] for a in chart(app)["layout"]["annotations"]]
    assert titles == ["sex: M", "sex: F"]
