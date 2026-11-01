"""Time Series page: pandas-version-safe frequencies and the in-browser menu."""

import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from edalens.views import time_series


@pytest.mark.parametrize("granularity", ["Hourly", "Monthly", "Quarterly", "Yearly"])
def test_frequency_aliases_resample_on_this_pandas(granularity):
    # pandas 3 rejects "H", "M", "Q" and "Y"; the aliases must work on the installed pandas.
    series = pd.Series(
        range(800), index=pd.date_range("2020-01-01", periods=800, freq="D")
    )
    assert len(series.resample(time_series.FREQ[granularity]).mean()) > 0


def time_series_page():
    from edalens.views.time_series import time_series_analysis

    time_series_analysis()


def test_analysis_menu_is_a_radio_in_the_browser(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")  # As under Pyodide (stlite).
    app = AppTest.from_function(time_series_page, default_timeout=30)
    app.run()
    assert not app.exception
    menu = app.sidebar.radio[0]
    assert menu.value == "Load Data"
    assert "Forecasting" in menu.options
