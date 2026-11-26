"""Save a session to JSON and open it again: data, steps, plot settings, orders, report."""

import json

import numpy as np
import pandas as pd
import pytest
import plotly.express as px
from streamlit.testing.v1 import AppTest

from edalens.utils.data_processing import run_operation, set_working_df
from edalens.utils.report import Report
from edalens.utils.session import SessionError, export_session, restore_session


def loaded_state() -> dict:
    raw = pd.DataFrame(
        {
            "day": ["Tue", "Mon", "Sun", "Mon"],
            "tip": [1.0, np.nan, 3.0, 4.0],
            "when": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
        }
    )
    state = {"raw_df": raw.copy(), "operations": [], "last_uploaded_file": "tips.csv"}
    set_working_df(state, raw.copy(), [])
    run_operation(
        state,
        {"type": "category_order", "column": "day", "order": ["Mon", "Tue", "Sun"]},
    )
    run_operation(state, {"type": "impute", "numeric": "mean", "categorical": "mode"})
    state["viz_state"] = {
        "plot_type": "lineplot",
        "x": "day",
        "y": "tip",
        "color": "None",
        "facet_col": "None",
        "facet_row": "None",
        "size": "None",
    }
    state["category_orders"] = {
        "day": {"order": ["Sun", "Mon", "Tue"], "known": ["Mon", "Tue", "Sun"]}
    }
    state["report"] = Report("Saved")
    state["report"].add(px.bar(x=[1, 2], y=[3, 4], title="Bars"))
    return state


def test_a_saved_session_opens_where_it_was_left():
    state = loaded_state()
    text = export_session(state)
    fresh = {"x_zone": "stale"}
    working = restore_session(fresh, text)
    # JSON keeps the dates; pandas 3 may read them back at another resolution (ns vs us).
    pd.testing.assert_frame_equal(working, state["filtered_df"], check_dtype=False)
    assert list(working["day"].cat.categories) == ["Mon", "Tue", "Sun"]
    assert working["when"].dtype.kind == "M"
    assert [op["type"] for op in fresh["operations"]] == ["category_order", "impute"]
    assert fresh["viz_state"]["plot_type"] == fresh["_plot_type_selector"] == "lineplot"
    assert "x_zone" not in fresh  # Plot widgets restart from the saved mappings.
    assert fresh["category_orders"]["day"]["order"] == ["Sun", "Mon", "Tue"]
    assert fresh["category_orders_version"] == fresh["data_version"]
    assert fresh["report"].items[0].title == "Bars"
    assert fresh["data_loaded"] and fresh["last_uploaded_file"] == "tips.csv"


def test_a_session_without_data_needs_matching_data():
    state = loaded_state()
    text = export_session(state, include_data=False)
    assert json.loads(text)["data"] is None
    with pytest.raises(SessionError, match="load the same data"):
        restore_session({}, text)
    with pytest.raises(SessionError, match="lacks columns"):
        restore_session({}, text, data=pd.DataFrame({"other": [1]}))
    working = restore_session({}, text, data=state["raw_df"])
    assert len(working) == 4


@pytest.mark.parametrize("text", ["not json", json.dumps({"format": "other"})])
def test_other_files_are_refused(text):
    with pytest.raises(SessionError):
        restore_session({}, text)


def test_steps_that_no_longer_apply_leave_the_session_unchanged():
    state = loaded_state()
    payload = json.loads(export_session(state))
    payload["operations"].append({"type": "select_columns", "columns": ["missing"]})
    target = {"keep": 1}
    with pytest.raises(SessionError, match="replayed"):
        restore_session(target, json.dumps(payload))
    assert target == {"keep": 1}


def panel_page():
    from edalens.views.session import session_panel

    session_panel()


def test_panel_prepares_a_download():
    state = loaded_state()
    app = AppTest.from_function(panel_page, default_timeout=30)
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    assert not app.get("download_button")
    next(b for b in app.button if b.label == "Prepare session file").click()
    app.run()
    assert not app.exception and app.get("download_button")


def test_text_dates_stay_text_so_day_first_is_not_lost():
    # pandas' read_json parses text columns named like "date" month-first.
    raw = pd.DataFrame(
        {
            "date": ["02/01/2024", "13/01/2024"],
            "stamp": pd.to_datetime(["2024-01-02", "2024-01-13"]),
        }
    )
    state = {"raw_df": raw.copy(), "operations": []}
    fresh = {}
    restore_session(fresh, export_session(state, include_data=True))
    assert fresh["raw_df"]["date"].tolist() == ["02/01/2024", "13/01/2024"]
    assert fresh["raw_df"]["stamp"].dtype.kind == "M"


def test_browser_preload_keeps_text_dates(tmp_path):
    from edalens.browser import frame_json
    from edalens.utils.data_processing import PRESET_DATA_ENV, load_preset_data

    path = tmp_path / "data.json"
    path.write_text(frame_json(pd.DataFrame({"DATE": ["02/01/2024", "13/01/2024"]})))
    state = {}
    assert load_preset_data(state, {PRESET_DATA_ENV: str(path)})
    assert state["raw_df"]["DATE"].tolist() == ["02/01/2024", "13/01/2024"]
