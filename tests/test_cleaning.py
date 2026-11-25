"""Load page cleaning: rename, order labels, parse dates, filter, bin, recode; imputer options."""

import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from edalens.utils.data_processing import (
    OperationError,
    apply_operation,
    apply_operations,
)


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "size": ["item 10", "item 2", "item 1", "item 2", None],
            "day": ["Tue", "Mon", "Tue", "Sun", "Mon"],
            "when": ["31/01/2024", "01/02/2024", "15/02/2024", "x", "01/03/2024"],
            "score": [1.0, 5.0, np.nan, 9.0, 3.0],
            "age": [10, 20, 30, 40, 50],
        }
    )


def test_rename_refuses_duplicates():
    out = apply_operation(
        frame(), {"type": "rename_columns", "mapping": {"score": "points"}}
    )
    assert "points" in out.columns and "score" not in out.columns
    with pytest.raises(OperationError):
        apply_operation(
            frame(), {"type": "rename_columns", "mapping": {"score": "age"}}
        )


def test_category_order_puts_unlisted_labels_last():
    out = apply_operation(
        frame(), {"type": "category_order", "column": "day", "order": ["Sun", "Mon"]}
    )
    assert (
        list(out["day"].cat.categories) == ["Sun", "Mon", "Tue"]
        and out["day"].cat.ordered
    )
    assert out["size"].isna().sum() == 1  # Other columns untouched.


def test_to_datetime_and_date_filter():
    out = apply_operation(
        frame(), {"type": "to_datetime", "column": "when", "dayfirst": True}
    )
    assert out["when"].dtype.kind == "M" and out["when"].isna().sum() == 1
    assert out.loc[1, "when"] == pd.Timestamp("2024-02-01")
    kept = apply_operation(
        out,
        {
            "type": "filter_rows",
            "column": "when",
            "min": "2024-02-01",
            "max": "2024-02-29",
        },
    )
    assert kept.index.tolist() == [1, 2]


def test_filters_by_range_and_values():
    data = frame()
    assert apply_operation(
        data, {"type": "filter_rows", "column": "age", "min": 20, "max": 40}
    ).index.tolist() == [1, 2, 3]
    out = apply_operation(
        data, {"type": "filter_rows", "column": "day", "values": ["Mon"]}
    )
    assert out["day"].tolist() == ["Mon", "Mon"]
    with pytest.raises(OperationError):
        apply_operation(data, {"type": "filter_rows", "column": "day", "values": []})


@pytest.mark.parametrize("method", ["equal-width", "quantile"])
def test_bins_are_ordered_categories(method):
    out = apply_operation(
        frame(),
        {
            "type": "bin_numeric",
            "column": "age",
            "bins": 2,
            "method": method,
            "name": "age_bin",
        },
    )
    labels = list(out["age_bin"].cat.categories)
    assert len(labels) == 2 and out["age_bin"].cat.ordered
    assert (
        out["age_bin"].tolist()[0] == labels[0]
        and out["age_bin"].tolist()[-1] == labels[1]
    )


def test_recode_merges_labels_and_keeps_category_order():
    data = apply_operation(
        frame(),
        {"type": "category_order", "column": "day", "order": ["Mon", "Tue", "Sun"]},
    )
    out = apply_operation(
        data,
        {
            "type": "recode_values",
            "column": "day",
            "mapping": {"Sun": "Weekend", "Tue": "Mon"},
        },
    )
    assert out["day"].tolist() == ["Mon", "Mon", "Mon", "Weekend", "Mon"]
    assert list(out["day"].cat.categories) == ["Mon", "Weekend"]


@pytest.mark.parametrize(
    "numeric, expected",
    [
        ("mean", (1 + 5 + 9 + 3) / 4),
        ("median", 4.0),
        ("zero", 0.0),
        ("interpolate", 7.0),
    ],
)
def test_numeric_imputers(numeric, expected):
    out = apply_operation(
        frame(), {"type": "impute", "numeric": numeric, "categorical": "none"}
    )
    assert out.loc[2, "score"] == pytest.approx(expected)
    assert out["size"].isna().sum() == 1  # Categorical left alone.


def test_knn_with_k_and_only_some_columns_and_missing_label():
    out = apply_operation(
        frame(),
        {
            "type": "impute",
            "numeric": "knn",
            "k": 1,
            "categorical": "missing",
            "columns": ["score", "size"],
        },
    )
    assert out.loc[2, "score"] in (5.0, 9.0)  # The nearest row by age.
    assert out.loc[4, "size"] == "Missing"


def test_old_impute_operations_replay_unchanged():
    old = apply_operations(frame(), [{"type": "impute"}])
    new = apply_operation(
        frame(), {"type": "impute", "numeric": "knn", "k": 5, "categorical": "mode"}
    )
    pd.testing.assert_frame_equal(old, new)


def test_drop_columns_with_too_many_missing():
    data = frame().assign(empty=[np.nan] * 4 + [1.0])
    assert (
        "empty"
        not in apply_operation(
            data, {"type": "drop_columns_missing", "max_pct": 50}
        ).columns
    )


def home_page():
    from edalens.views.home import home

    home()


def home_app() -> AppTest:
    data = frame()
    app = AppTest.from_function(home_page, default_timeout=30)
    app.session_state["raw_df"] = data
    app.session_state["filtered_df"] = data
    app.session_state["operations"] = []
    app.session_state["data_loaded"] = True
    app.session_state["data_version"] = 0
    app.session_state["current_numeric_cols"] = ["score", "age"]
    app.session_state["current_categorical_cols"] = ["size", "day", "when"]
    return app


def use(app: AppTest, tool: str) -> None:
    """Pick a cleaning tool (one is shown at a time)."""
    import edalens.views.home as home

    app.radio(key="clean_tool").set_value(getattr(home, tool))
    app.run()


def test_order_labels_on_the_load_page_reaches_the_plots():
    app = home_app()
    app.run()
    use(app, "ORDER")
    app.selectbox(key="clean_order_column").set_value("size")
    app.run()
    app.selectbox(key="clean_order_how").set_value("Alphabetical (A → Z)")
    app.run()
    assert app.multiselect(key="order_size_Alphabetical (A → Z)_0").value == [
        "item 1",
        "item 2",
        "item 10",
    ]
    app.button(key="order_apply_0").click()
    app.run()
    assert not app.exception
    data = app.session_state["filtered_df"]
    assert list(data["size"].cat.categories) == ["item 1", "item 2", "item 10"]

    from test_visualization_plots import viz_app

    viz = viz_app(data, "bars", x="size", y="None")
    viz.session_state["current_categorical_cols"] = ["size", "day", "when"]
    viz.run()
    assert not viz.exception
    order = next(
        m for m in viz.multiselect if m.label.startswith("Custom Order for X")
    ).value
    assert order == ["item 1", "item 2", "item 10"]
    axis = json.loads(viz.get("plotly_chart")[-1].proto.spec)["layout"]["xaxis"]
    assert axis["categoryarray"] == ["item 1", "item 2", "item 10"]


def test_cleaning_tools_apply_operations():
    app = home_app()
    app.run()
    app.selectbox(key="clean_rename_column").set_value(
        "score"
    )  # Rename is the first tool.
    app.run()
    app.text_input(key="rename_to_score_0").set_value("points")
    app.run()  # Typing re-runs the page; the button turns on.
    app.button(key="rename_apply_0").click()
    app.run()
    assert not app.exception and "points" in app.session_state["filtered_df"].columns
    use(app, "BINS")
    app.selectbox(key="clean_bin_column").set_value("age")
    app.run()
    app.button(key="bin_apply_1").click()
    app.run()
    assert not app.exception and "age_bin" in app.session_state["filtered_df"].columns
    assert [op["type"] for op in app.session_state["operations"]] == [
        "rename_columns",
        "bin_numeric",
    ]


def missing_page():
    from edalens.views.missing_data import missing_data_imputation

    missing_data_imputation()


def missing_app() -> AppTest:
    data = pd.DataFrame(
        {
            "a": [1, np.nan, 3, 4],
            "b": [1, 2, np.nan, 4.0],
            "c": ["x", None, "x", "y"],
            "e": [np.nan] * 3 + [1],
        }
    )
    app = AppTest.from_function(missing_page, default_timeout=30)
    app.session_state["filtered_df"] = data
    app.session_state["raw_df"] = data
    app.session_state["operations"] = []
    return app


def test_missing_page_imputes_with_the_chosen_methods():
    app = missing_app()
    app.run()
    app.selectbox(key="impute_numeric").set_value("median")
    app.selectbox(key="impute_categorical").set_value("missing")
    app.run()
    next(b for b in app.button if b.label == "Impute missing values").click()
    app.run()
    assert not app.exception
    data = app.session_state["filtered_df"]
    assert data.loc[1, "a"] == 3.0 and data.loc[1, "c"] == "Missing"
    assert app.session_state["operations"][-1]["numeric"] == "median"


def test_missing_page_drops_mostly_empty_columns():
    app = missing_app()
    app.run()
    next(b for b in app.button if b.label == "Drop these columns").click()
    app.run()
    assert not app.exception and "e" not in app.session_state["filtered_df"].columns


def kinds_of(block) -> list:
    return [type(child).__name__ for child in block.children.values()]


def blocks(node):
    """Every block (container) in the element tree."""
    for child in getattr(node, "children", {}).values():
        if hasattr(child, "children"):
            yield child
            yield from blocks(child)


def test_the_tool_stays_chosen_with_its_message_below_the_button():
    app = home_app()
    app.run()
    use(app, "BINS")
    app.selectbox(key="clean_bin_column").set_value("age")
    app.run()
    app.button(key="bin_apply_0").click()
    app.run()
    assert not app.exception
    assert app.radio(key="clean_tool").value == app.radio(key="clean_tool").options[4]
    assert [s.value for s in app.success] == [
        "Added age_bin, a categorical column in bin order."
    ]
    # Below the button, so the button just clicked does not move down.
    panel = next(block for block in blocks(app.main) if "Success" in kinds_of(block))
    kinds = kinds_of(panel)
    assert kinds.index("Success") > kinds.index("Button")


def test_pickers_stay_on_the_column_being_edited():
    app = home_app()
    app.run()
    use(app, "ORDER")
    app.selectbox(key="clean_order_column").set_value("day")
    app.selectbox(key="clean_order_how").set_value("Calendar (months, weekdays)")
    app.run()
    app.button(key="order_apply_0").click()
    app.run()
    assert list(app.session_state["filtered_df"]["day"].cat.categories) == [
        "Mon",
        "Tue",
        "Sun",
    ]
    # Still on day (not back to the first text column), with the same sort.
    assert app.selectbox(key="clean_order_column").value == "day"
    assert app.selectbox(key="clean_order_how").value == "Calendar (months, weekdays)"

    use(app, "BINS")
    app.selectbox(key="clean_bin_column").set_value("age")
    app.number_input(key="clean_bin_count").set_value(3)
    app.run()
    use(app, "RENAME")
    app.selectbox(key="clean_rename_column").set_value("age")
    app.run()
    app.text_input(key="rename_to_age_1").set_value("years")
    app.run()  # Typing re-runs the page; the button turns on.
    app.button(key="rename_apply_1").click()
    app.run()
    assert not app.exception
    # Pickers that were on the renamed column follow it; other settings stay.
    assert app.selectbox(key="clean_rename_column").value == "years"
    use(app, "BINS")
    assert app.selectbox(key="clean_bin_column").value == "years"
    assert app.number_input(key="clean_bin_count").value == 3
    use(app, "ORDER")
    assert app.selectbox(key="clean_order_column").value == "day"
