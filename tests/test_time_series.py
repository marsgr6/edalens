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


def monthly_csv(years: int = 8) -> bytes:
    dates = pd.date_range("2010-01-01", periods=12 * years, freq="MS")
    values = [50 + 20 * ((i % 12) in (5, 6, 7)) + i * 0.1 for i in range(len(dates))]
    frame = pd.DataFrame({"FECHA": dates.strftime("%d/%m/%Y"), "rain": values})
    return frame.to_csv(index=False).encode()


@pytest.mark.parametrize("model", ["Holt-Winters (ETS)", "SARIMA"])
def test_forecasting_section_runs_statsmodels_models(monkeypatch, model):
    import requests

    class Response:
        content = monthly_csv()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        sys, "platform", "emscripten"
    )  # Radio menu, drivable in AppTest.
    app = AppTest.from_function(time_series_page, default_timeout=60)
    app.run()
    next(r for r in app.radio if r.label == "Select data source").set_value(
        "GitHub Link"
    )
    app.run()
    app.text_input[0].set_value("https://example.com/rain.csv")
    app.run()
    app.sidebar.radio[0].set_value("Forecasting")
    app.run()
    next(s for s in app.selectbox if s.label == "Model").set_value(model)
    app.run()

    assert not app.exception
    assert not app.warning, [w.value for w in app.warning]
    assert any("observations" in caption.value for caption in app.caption)
