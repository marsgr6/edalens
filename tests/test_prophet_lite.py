"""The numpy Prophet used in browser mode: same model, close to Prophet's results."""

import logging

import numpy as np
import pandas as pd
import pytest

from edalens.utils import prophet_lite
from edalens.utils.prophet_lite import LiteProphet


def daily(n: int = 900, seed: int = 1) -> pd.DataFrame:
    """A trend that turns at day 400, yearly and weekly cycles, noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = np.where(t < 400, 0.05 * t, 20 - 0.03 * (t - 400))
    season = 5 * np.sin(2 * np.pi * t / 365.25) + 2 * np.sin(2 * np.pi * t / 7)
    return pd.DataFrame(
        {
            "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
            "y": 50 + trend + season + rng.normal(0, 1, n),
        }
    )


def test_finds_the_trend_change_and_the_seasons():
    model = LiteProphet().fit(daily())
    assert set(model.seasonalities) == {
        "yearly",
        "weekly",
    }  # Daily data: no daily season.
    delta = np.abs(model.params["delta"][0])
    top = model.changepoints.iloc[delta.argmax()]
    assert abs((top - pd.Timestamp("2021-02-04")).days) < 40  # Day 400.
    assert (delta == 0).sum() > len(delta) / 2  # The Laplace prior keeps most at zero.
    fit = model.predict()
    assert {"trend", "yearly", "weekly", "yhat", "yhat_lower", "yhat_upper"} <= set(fit)
    assert np.std(daily()["y"] - fit["yhat"]) < 1.2  # About the noise level.


def test_future_band_widens_and_holds_the_level():
    model = LiteProphet(interval_width=0.9).fit(daily())
    future = model.make_future_dataframe(120)
    assert len(future) == 900 + 120 and future["ds"].iloc[-1] == pd.Timestamp(
        "2022-10-16"
    )
    forecast = model.predict(future)
    width = (forecast["yhat_upper"] - forecast["yhat_lower"]).to_numpy()
    assert (
        width[-1] > width[899] * 1.05
    )  # Trend uncertainty grows (Prophet: +14% here).
    assert (forecast["yhat_lower"] <= forecast["yhat"]).all()
    assert (forecast["yhat"] <= forecast["yhat_upper"]).all()


def test_settings_used_by_the_page():
    data = daily(200)
    model = LiteProphet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        n_changepoints=20,
        changepoint_range=0.95,
        changepoint_prior_scale=0.5,
    ).fit(data)
    assert list(model.seasonalities) == ["yearly"] and len(model.changepoints) == 20
    assert model.params["delta"].shape == (1, 20)
    assert model.changepoints.iloc[-1] <= data["ds"].iloc[int(200 * 0.95)]


def test_prophet_class_picks_lite_on_request(monkeypatch):
    monkeypatch.setenv(prophet_lite.LITE_ENV, "lite")
    assert prophet_lite.prophet_class() == (LiteProphet, True)


def test_close_to_real_prophet():
    prophet = pytest.importorskip("prophet")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    data = daily()
    results = []
    for cls in (prophet.Prophet, LiteProphet):
        model = cls(interval_width=0.95).fit(data.iloc[:-60])
        results.append(model.predict(model.make_future_dataframe(60)))
    real, lite = results
    assert np.corrcoef(real["yhat"], lite["yhat"])[0, 1] > 0.99
    assert np.max(np.abs(real["yhat"] - lite["yhat"])) < 0.1 * data["y"].std()
    width = lambda f: (f["yhat_upper"] - f["yhat_lower"]).mean()  # noqa: E731
    assert 0.8 < width(lite) / width(real) < 1.25
