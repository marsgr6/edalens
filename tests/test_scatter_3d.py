"""Streamlit-level checks for the scatter plot's 3D mode."""

import json

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def visualization_page():
    from edalens.views.visualization import visualization

    visualization()


def scatter_app(frame: pd.DataFrame, **viz) -> AppTest:
    app = AppTest.from_function(visualization_page, default_timeout=30)
    app.session_state["filtered_df"] = frame
    app.session_state["operations"] = []
    app.session_state["current_numeric_cols"] = frame.select_dtypes("number").columns.tolist()
    app.session_state["current_categorical_cols"] = frame.select_dtypes("object").columns.tolist()
    app.session_state["viz_state"] = {
        "plot_type": "scatter", "x": "a", "y": "b", "color": "None",
        "facet_col": "None", "facet_row": "None", "size": "None", **viz,
    }
    app.session_state["_plot_type_selector"] = "scatter"
    return app


def sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(size=30),
            "b": rng.normal(size=30),
            "c": rng.normal(size=30),
            "w": rng.uniform(1, 5, 30),
            "kind": rng.choice(["p", "q"], 30),
        }
    )


def checkbox(app: AppTest, label: str):
    return next(item for item in app.checkbox if item.label == label)


def chart(app: AppTest) -> dict:
    return json.loads(app.get("plotly_chart")[-1].proto.spec)


def test_scatter_stays_2d_by_default():
    app = scatter_app(sample_frame())
    app.run()
    assert not app.exception
    assert {trace["type"] for trace in chart(app)["data"]} == {"scatter"}


def test_3d_scatter_uses_z_and_keeps_color_and_size():
    app = scatter_app(sample_frame(), color="kind", size="w")
    app.run()
    checkbox(app, "3D Scatter").check()
    app.run()
    z_select = next(item for item in app.selectbox if item.label == "Z Variable")
    assert z_select.options == ["c", "w"]  # numeric columns other than X and Y
    z_select.set_value("c")
    checkbox(app, "Show Projections on Walls").check()
    app.run()

    assert not app.exception
    spec = chart(app)
    traces = spec["data"]
    assert {trace["type"] for trace in traces} == {"scatter3d"}
    assert {trace["name"] for trace in traces} == {"p", "q"}  # one trace per hue group
    assert spec["layout"]["scene"]["zaxis"]["title"]["text"] == "c"
    assert all(trace["projection"]["z"]["show"] for trace in traces)


def test_3d_scatter_drops_facets_with_a_notice():
    app = scatter_app(sample_frame(), facet_col="kind")
    app.run()
    checkbox(app, "3D Scatter").check()
    app.run()

    assert not app.exception
    assert any("not available in 3D" in item.value for item in app.info)
    assert {trace["type"] for trace in chart(app)["data"]} == {"scatter3d"}


def test_3d_scatter_without_a_third_numeric_column_warns():
    app = scatter_app(sample_frame()[["a", "b", "kind"]])
    app.run()
    checkbox(app, "3D Scatter").check()
    app.run()

    assert not app.exception
    assert any("third numeric column" in item.value for item in app.warning)
