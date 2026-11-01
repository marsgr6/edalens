"""statsmodels forecasting and residual Isolation Forest anomalies (no Streamlit)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from edalens.utils import forecasting as fc


def monthly_series(years: int = 10) -> pd.Series:
    """Trend plus a yearly cycle with mild noise, monthly."""
    index = pd.date_range("2010-01-01", periods=12 * years, freq="MS")
    t = np.arange(len(index))
    rng = np.random.default_rng(0)
    values = 50 + 0.2 * t + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, len(t))
    return pd.Series(values, index=index)


@pytest.mark.parametrize("method", fc.METHODS)
def test_forecast_has_the_horizon_and_an_interval_around_it(method):
    series = fc.regular_series(monthly_series())
    result = fc.fit_forecast(
        series, method, horizon=12, season=fc.season_length(series)
    )
    future = result[result["forecast"].notna()]
    assert len(future) == 12
    assert future.index[0] == series.index[-1] + series.index.freq
    assert (future["lower"] < future["forecast"]).all() and (
        future["forecast"] < future["upper"]
    ).all()
    # The model follows the series: one-step fits stay close to the observations.
    history = result.dropna(subset=["observed", "fitted"])
    assert (history["observed"] - history["fitted"]).abs().median() < 3


def test_season_length_comes_from_the_frequency_and_needs_two_seasons():
    assert fc.season_length(fc.regular_series(monthly_series())) == 12
    assert fc.season_length(fc.regular_series(monthly_series(years=1))) is None
    assert fc.season_length(fc.regular_series(monthly_series()), "Quarterly") == 4


def test_residual_anomalies_find_a_point_that_is_normal_in_size_but_wrong_for_its_season():
    series = monthly_series()
    peak = series.index[series.index.month == 1][6]  # A seasonal low...
    series[peak] = series.max() - 5  # ...given a typical summer-peak value.
    series = fc.regular_series(series)

    result = fc.fit_forecast(series, fc.ETS, horizon=6, season=12)
    flagged = fc.residual_anomalies(result, contamination=0.02)
    assert flagged[peak]

    # Isolation Forest on the raw values alone misses it: the value is not extreme.
    raw = IsolationForest(contamination=0.02, random_state=42).fit_predict(
        series.to_frame()
    )
    assert raw[series.index.get_loc(peak)] == 1


def test_irregular_dates_ask_for_a_summarization_frequency():
    index = pd.to_datetime(
        [
            "2020-01-01",
            "2020-01-03",
            "2020-01-04",
            "2020-01-09",
            "2020-01-10",
            "2020-01-17",
            "2020-01-18",
            "2020-01-30",
            "2020-02-02",
        ]
    )
    with pytest.raises(fc.ForecastError, match="summarization frequency"):
        fc.regular_series(pd.Series(range(len(index)), index=index))


def test_small_gaps_are_filled_on_the_regular_grid():
    series = monthly_series(years=3).drop(monthly_series(years=3).index[[5, 17]])
    regular = fc.regular_series(series)
    assert len(regular) == 36 and regular.notna().all()


def test_long_seasons_use_a_non_seasonal_sarima():
    index = pd.date_range("2018-01-01", periods=52 * 3, freq="W-SUN")
    series = pd.Series(np.sin(np.arange(len(index)) / 8) + 10, index=index)
    result = fc.fit_forecast(series, fc.SARIMA, horizon=4, season=52)
    assert result["forecast"].notna().sum() == 4


def test_figure_shows_observed_fit_forecast_interval_and_anomalies():
    series = fc.regular_series(monthly_series())
    result = fc.fit_forecast(series, fc.ETS, horizon=6, season=12)
    flagged = fc.residual_anomalies(result, contamination=0.05)
    fig = fc.forecast_figure(result, "t", "y", anomalies=flagged)
    names = [trace.name for trace in fig.data]
    assert names == [
        "Observed",
        "Fitted",
        "95% interval",
        "Forecast",
        "Anomaly (Isolation Forest)",
    ]
    assert len(fig.data[-1].x) == int(flagged.sum())
