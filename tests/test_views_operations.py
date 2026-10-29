"""Streamlit-level checks for operation errors, undo, reset, and read-only views."""

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from streamlit.testing.v1 import AppTest


def home_page():
    from edalens.views.home import home

    home()


def missing_data_page():
    from edalens.views.missing_data import missing_data_imputation

    missing_data_imputation()


def visualization_page():
    from edalens.views.visualization import visualization

    visualization()


def loaded_app(page, frame: pd.DataFrame, tracking: bool = False) -> AppTest:
    app = AppTest.from_function(page, default_timeout=30)
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    app.session_state["raw_df"] = frame.copy()
    app.session_state["filtered_df"] = frame.copy()
    app.session_state["operations"] = []
    app.session_state["enable_operation_tracking"] = tracking
    app.session_state["current_numeric_cols"] = numeric
    app.session_state["current_categorical_cols"] = categorical
    app.session_state["data_loaded"] = True
    return app


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def tracking_checkbox(app: AppTest):
    return next(
        item
        for item in app.checkbox
        if item.label.startswith("Enable Operation Tracking")
    )


def test_failed_home_operation_shows_error_and_keeps_state():
    frame = pd.DataFrame({"id": ["a", "b"], "x": [1, 2]})
    app = loaded_app(home_page, frame)
    app.run()

    app.multiselect(key="melt_keys").set_value(["id", "x"])
    button(app, "Apply Melting").click()
    app.run()

    assert not app.exception
    assert any("non-ID column" in error.value for error in app.error)
    assert not any("Melting applied" in item.value for item in app.success)
    assert app.session_state["operations"] == []
    assert_frame_equal(app.session_state["filtered_df"], frame)


def test_failed_missing_data_operation_shows_error_and_keeps_state():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, "x"]})
    app = loaded_app(missing_data_page, frame)
    app.run()

    button(app, "Remove Rows with Any Missing Values").click()
    app.run()

    assert not app.exception
    assert any("every row" in error.value for error in app.error)
    assert not app.success
    assert app.session_state["operations"] == []
    assert_frame_equal(app.session_state["filtered_df"], frame)


def test_imputation_records_history_and_keeps_columns():
    frame = pd.DataFrame(
        {
            "when": pd.to_datetime(["2024-01-01", None, "2024-01-03"]),
            "v": [1.0, np.nan, 3.0],
        }
    )
    app = loaded_app(missing_data_page, frame)
    app.run()

    button(
        app, "Impute Missing Values with KNN Imputer (Numeric) and Mode (Categorical)"
    ).click()
    app.run()

    assert not app.exception
    assert app.session_state["operations"] == [{"type": "impute"}]
    assert app.session_state["is_imputed"] is True
    result = app.session_state["filtered_df"]
    assert list(result.columns) == ["when", "v"]
    assert result["v"].tolist() == [1.0, 2.0, 3.0]
    # Datetime columns are not imputed, so the page reports what remains.
    assert any("still" in warning.value for warning in app.warning)


def test_undo_of_only_operation_restores_raw_data_and_reset_clears_session():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    app = loaded_app(home_page, frame)
    app.run()
    tracking_checkbox(app).check()
    app.run()

    app.multiselect(key="select_columns").set_value(["a"])
    button(app, "Apply Column Selection").click()
    app.run()
    assert list(app.session_state["filtered_df"].columns) == ["a"]
    # The undo button is drawn above the operation controls, so it appears on
    # the rerun that follows the first recorded operation.
    app.run()

    button(app, "Undo Last Operation").click()
    app.run()

    assert not app.exception
    assert app.session_state["operations"] == []
    assert app.session_state["data_loaded"] is True
    assert_frame_equal(app.session_state["filtered_df"], frame)

    button(app, "Reset All").click()
    app.run()

    assert app.session_state["operations"] == []
    assert app.session_state["data_loaded"] is False
    assert "filtered_df" not in app.session_state
    assert "raw_df" not in app.session_state


def test_visualization_does_not_mutate_the_shared_working_frame():
    frame = pd.DataFrame(
        {"group": pd.Categorical(["a", "b", "a"]), "value": [1.0, 2.0, 3.0]}
    )
    app = loaded_app(visualization_page, frame)
    app.run()

    assert not app.exception
    assert app.session_state["filtered_df"]["group"].dtype == "category"
    assert_frame_equal(app.session_state["filtered_df"], frame)
