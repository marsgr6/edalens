"""Time Series page: pandas-version-safe frequencies and the in-browser menu."""

import io
import json
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


def has_prophet() -> bool:
    try:
        import prophet  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize(
    "model",
    [
        "Holt-Winters (ETS)",
        "SARIMA",
        pytest.param(
            "Prophet",
            marks=pytest.mark.skipif(not has_prophet(), reason="prophet not installed"),
        ),
    ],
)
def test_forecasting_section_runs_every_model(monkeypatch, model):
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
    # Every model, Prophet included, flags anomalies on its residuals.
    assert any("anomalous points flagged" in c.value for c in app.caption)
    traces = [
        t.get("name")
        for t in json.loads(app.get("plotly_chart")[-1].proto.spec)["data"]
    ]
    assert {"Observed", "Fitted", "Forecast"} <= set(traces)


def test_loader_reads_iso_dates_with_day_first_selected(monkeypatch):
    import requests

    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    csv = (
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "v": range(40)})
        .to_csv(index=False)
        .encode()
    )

    class Response:
        content = csv

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(sys, "platform", "emscripten")
    app = AppTest.from_function(time_series_page, default_timeout=30)
    app.run()
    next(r for r in app.radio if r.label == "Select data source").set_value(
        "GitHub Link"
    )
    app.run()
    app.text_input[0].set_value("https://example.com/v.csv")
    app.run()
    assert next(
        r for r in app.radio if r.label == "Select date format"
    ).value.startswith("DD/MM")
    assert not app.exception and not app.warning, [w.value for w in app.warning]
    assert app.session_state["df"].index.equals(pd.DatetimeIndex(dates, name="date"))


def test_changepoints_run_for_several_variables(monkeypatch):
    # One Prophet model per variable; this branch once called a misspelled helper.
    pytest.importorskip("prophet")
    import requests

    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    trend = [i if i < 60 else 60 + 3 * (i - 60) for i in range(120)]
    csv = (
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "a": trend, "b": trend[::-1]})
        .to_csv(index=False)
        .encode()
    )

    class Response:
        content = csv

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(sys, "platform", "emscripten")
    app = AppTest.from_function(time_series_page, default_timeout=120)
    app.run()
    next(r for r in app.radio if r.label == "Select data source").set_value(
        "GitHub Link"
    )
    app.run()
    app.text_input[0].set_value("https://example.com/v.csv")
    app.run()
    app.sidebar.radio[0].set_value("Changepoint Analysis")
    app.run()
    next(c for c in app.sidebar.checkbox if "multiple" in c.label).check()
    app.run()
    app.sidebar.multiselect[0].set_value(["a", "b"])
    app.run()
    assert not app.exception, app.exception
    assert len(app.get("plotly_chart")) >= 2  # A changepoint chart per variable.


SECTIONS = [
    "Time Series Visualization",
    "Decomposition & ACF/PACF",
    "Anomaly Detection",
    "Temporal Profiles",
    "Calendar Heatmap",
    "Heatmap Profiles",
    "Forecasting",
    "Changepoint Analysis",
    "Trend Analysis",
]


def daily_csv(days: int = 730) -> bytes:
    import numpy as np

    dates = pd.date_range("2022-01-01", periods=days, freq="D")
    t = np.arange(days)
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "load": 100
            + 0.05 * t
            + 10 * np.sin(2 * np.pi * t / 365)
            + rng.normal(0, 2, days),
            "temp": 15
            + 8 * np.sin(2 * np.pi * (t - 30) / 365)
            + rng.normal(0, 1, days),
        }
    )
    return frame.to_csv(index=False).encode()


def section_app(monkeypatch) -> AppTest:
    import requests

    class Response:
        content = daily_csv()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        sys, "platform", "emscripten"
    )  # Radio menu, drivable in AppTest.
    app = AppTest.from_function(time_series_page, default_timeout=180)
    app.run()
    next(r for r in app.radio if r.label == "Select data source").set_value(
        "GitHub Link"
    )
    app.run()
    app.text_input[0].set_value("https://example.com/daily.csv")
    app.run()
    return app


@pytest.mark.parametrize(
    "several", [False, True], ids=["one variable", "two variables"]
)
@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_opens_with_its_defaults(monkeypatch, section, several):
    # Catches crashes (like a misspelled helper) in sections no other test opens.
    app = section_app(monkeypatch)
    app.sidebar.radio[0].set_value(section)
    app.run()
    if several:
        next(c for c in app.sidebar.checkbox if "multiple" in c.label).check()
        app.run()
        app.sidebar.multiselect[0].set_value(["load", "temp"])
        app.run()
    assert not app.exception, app.exception[0].message


@pytest.mark.parametrize(
    "section", ["Forecasting", "Decomposition & ACF/PACF", "Changepoint Analysis"]
)
def test_prophet_sections_run_with_the_browser_version(monkeypatch, section):
    """In browser mode Prophet is the numpy LiteProphet; every Prophet section uses it."""
    import requests

    from edalens.utils import prophet_lite

    monkeypatch.setenv(prophet_lite.LITE_ENV, "lite")
    monkeypatch.setattr(time_series, "Prophet", prophet_lite.LiteProphet)
    monkeypatch.setattr(time_series, "PROPHET_LITE", True)

    class Response:
        content = monthly_csv()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(sys, "platform", "emscripten")
    app = AppTest.from_function(time_series_page, default_timeout=60)
    app.run()
    next(r for r in app.radio if r.label == "Select data source").set_value(
        "GitHub Link"
    )
    app.run()
    app.text_input[0].set_value("https://example.com/rain.csv")
    app.run()
    app.sidebar.radio[0].set_value(section)
    app.run()
    if section == "Forecasting":
        next(s for s in app.selectbox if s.label == "Model").set_value("Prophet")
    elif section.startswith("Decomposition"):
        next(
            s for s in app.selectbox if s.label == "Select Decomposition Method"
        ).set_value("Prophet")
    app.run()
    assert not app.exception
    assert not app.warning, [w.value for w in app.warning]
    assert any(prophet_lite.LABEL in c.value for c in app.caption)
    assert app.get("plotly_chart")


def test_date_columns_and_day_month_order():
    from edalens.utils.data_processing import date_columns, guess_dayfirst

    frame = pd.DataFrame(
        {
            "when": ["31/01/2024", "01/02/2024", "15/02/2024"],
            "year": [2020, 2021, 2022],  # Numbers are not dates.
            "label": ["a", "b", "c"],
            "stamp": pd.date_range("2024-01-01", periods=3),
        }
    )
    assert date_columns(frame) == ["stamp", "when"]
    assert guess_dayfirst(frame["when"])  # 31/01 only reads day first.
    # Both orders read every value; only month first gives a regular (monthly) series.
    monthly = pd.Series(
        pd.date_range("2020-01-01", periods=24, freq="MS").strftime("%m/%d/%Y")
    )
    assert not guess_dayfirst(monthly)


def loaded_app(data: pd.DataFrame) -> AppTest:
    app = AppTest.from_function(time_series_page, default_timeout=60)
    app.session_state["filtered_df"] = data
    app.session_state["data_loaded"] = True
    app.session_state["data_version"] = 1
    return app


def monthly_frame() -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(monthly_csv()))


def test_load_data_dataset_is_the_default_source(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")
    app = loaded_app(monthly_frame())
    app.run()
    assert not app.exception
    source = next(r for r in app.radio if r.label == "Select data source")
    assert source.value == time_series.LOADED_SOURCE
    assert (
        next(s for s in app.selectbox if s.label == "Select the date column").value
        == "FECHA"
    )
    assert len(app.session_state["df"]) == 96
    assert isinstance(app.session_state["df"].index, pd.DatetimeIndex)


def test_other_sections_use_the_load_data_dataset_and_follow_its_changes(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")
    app = loaded_app(monthly_frame())
    app.run()
    app.sidebar.radio[0].set_value("Time Series Visualization")
    app.run()
    assert not app.exception and len(app.session_state["df"]) == 96
    # A cleaning step on the Load data page (a new data_version) reaches this page.
    app.session_state["filtered_df"] = monthly_frame().head(50)
    app.session_state["data_version"] = 2
    app.run()
    assert not app.exception and len(app.session_state["df"]) == 50


def test_a_dataset_without_dates_gets_a_message(monkeypatch):
    monkeypatch.setattr(sys, "platform", "emscripten")
    app = loaded_app(pd.DataFrame({"tip": [1.0, 2.5], "day": ["Sun", "Sat"]}))
    app.run()
    assert not app.exception and "df" not in app.session_state
    assert any("no date column" in w.value for w in app.warning)
    app.sidebar.radio[0].set_value("Forecasting")
    app.run()
    assert any("no date column" in i.value for i in app.info)
