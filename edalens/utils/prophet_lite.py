"""A small Prophet-compatible model in numpy, for browser mode.

Prophet fits its model with Stan, a compiled C++ program that cannot run in
WebAssembly (Pyodide), so ``edalens.walk(df, mode="browser")`` uses this class
in its place. It follows Prophet's additive model and API for what EDALens uses:

    y(t) = trend(t) + yearly(t) + weekly(t) + daily(t) + noise

* **Trend**: piecewise linear, ``k*t + m + sum(delta_j * (t - s_j)+)``, with
  ``n_changepoints`` candidate dates spread over the first
  ``changepoint_range`` of the history (as Prophet places them).
* **Changepoint prior**: Prophet puts a Laplace(0, ``changepoint_prior_scale``)
  prior on each ``delta`` and takes the MAP estimate. That is an L1 (lasso)
  penalty, solved here by iteratively reweighted least squares.
* **Seasonality**: Fourier series with Prophet's default orders (yearly 10,
  weekly 3, daily 4) and its "auto" rules for which ones to use.
* **Intervals**: like Prophet, future trend changes are simulated (changepoints
  as frequent as in the history, Laplace deltas of the fitted mean size) and
  observation noise is added; the band is the ``interval_width`` quantiles.

Not covered: multiplicative seasonality, logistic growth, holidays, extra
regressors and MCMC. Results are close to Prophet's but not identical.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

LITE_ENV = "EDALENS_PROPHET"  # "lite" uses this model even where prophet is installed.
LABEL = "Prophet (browser version)"


def prophet_class():
    """``(Prophet class, is_lite)``: real Prophet when installed, else LiteProphet."""
    import os

    if os.environ.get(LITE_ENV, "").lower() != "lite":
        try:
            from prophet import Prophet

            return Prophet, False
        except ImportError:  # Browser mode: Stan cannot run in WebAssembly.
            pass
    return LiteProphet, True


SEASONALITIES = {  # name: (period in days, Fourier order), Prophet's defaults.
    "yearly": (365.25, 10),
    "weekly": (7.0, 3),
    "daily": (1.0, 4),
}
SEASONALITY_PRIOR_SCALE = 10.0
UNCERTAINTY_SAMPLES = 300
IRLS_STEPS = 30
EPS = 1e-6


def _days(ds: pd.Series) -> np.ndarray:
    """Days since 1970, the time Prophet's Fourier terms use."""
    return pd.to_datetime(ds).astype("datetime64[ns]").astype("int64").to_numpy() / 1e9 / 86400


def _fourier(days: np.ndarray, period: float, order: int) -> np.ndarray:
    x = 2 * np.pi * np.outer(days, np.arange(1, order + 1)) / period
    return np.column_stack([f(x[:, i]) for i in range(order) for f in (np.sin, np.cos)])


class LiteProphet:
    """The subset of ``prophet.Prophet`` the Time Series page uses."""

    lite = True  # Lets the page say which model it ran.

    def __init__(
        self,
        growth: str = "linear",
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        interval_width: float = 0.8,
        uncertainty_samples: int = UNCERTAINTY_SAMPLES,
        **_ignored,
    ):
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.requested = {"yearly": yearly_seasonality, "weekly": weekly_seasonality,
                          "daily": daily_seasonality}
        self.changepoint_prior_scale = changepoint_prior_scale
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.history: Optional[pd.DataFrame] = None
        self.params: dict = {}

    # -- fitting ----------------------------------------------------------------

    def _seasonalities(self, ds: pd.Series) -> dict:
        """Prophet's "auto" rules: a season needs two full cycles and finer data."""
        span = (ds.max() - ds.min()) / pd.Timedelta(days=1)
        step = ds.sort_values().diff().dropna()
        step = step.min() / pd.Timedelta(days=1) if len(step) else np.inf
        auto = {"yearly": span >= 730, "weekly": span >= 14 and step < 7, "daily": span >= 2 and step < 1}
        chosen = {}
        for name, value in self.requested.items():
            period, order = SEASONALITIES[name]
            if value == "auto":
                value = auto[name]
            if value is True:
                chosen[name] = (period, order)
            elif value and not isinstance(value, bool):  # An int sets the Fourier order.
                chosen[name] = (period, int(value))
        return chosen

    def _t(self, ds: pd.Series) -> np.ndarray:
        return ((pd.to_datetime(ds) - self.start) / self.t_scale).to_numpy(dtype=float)

    def _design(self, ds: pd.Series) -> tuple:
        t = self._t(ds)
        hinge = np.maximum(0.0, t[:, None] - self.changepoints_t[None, :])
        days = _days(ds)
        blocks, slices, at = [], {}, 0
        for name, (period, order) in self.seasonalities.items():
            block = _fourier(days, period, order)
            blocks.append(block)
            slices[name] = slice(at, at + block.shape[1])
            at += block.shape[1]
        seasonal = np.hstack(blocks) if blocks else np.empty((len(t), 0))
        return t, np.column_stack([np.ones_like(t), t, hinge]), seasonal, slices

    def fit(self, df: pd.DataFrame) -> "LiteProphet":
        data = df[["ds", "y"]].copy()
        data["ds"] = pd.to_datetime(data["ds"])
        data = data.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
        fitted_rows = data["y"].notna()
        if fitted_rows.sum() < 2:
            raise ValueError("Dataframe has less than 2 non-NaN rows.")
        self.history = data
        self.start = data["ds"].min()
        self.t_scale = max(data["ds"].max() - self.start, pd.Timedelta(seconds=1))
        self.y_scale = float(np.abs(data["y"]).max()) or 1.0
        self.seasonalities = self._seasonalities(data["ds"])

        # Candidate changepoints: evenly spaced rows over the first changepoint_range.
        rows = int(np.floor(len(data) * self.changepoint_range))
        n = max(0, min(self.n_changepoints, rows - 1))
        if n:
            index = np.linspace(0, rows - 1, n + 1).round().astype(int)[1:]
            self.changepoints = data["ds"].iloc[index].reset_index(drop=True)
        else:
            self.changepoints = pd.Series([], dtype="datetime64[ns]")
        self.changepoints_t = self._t(self.changepoints) if n else np.empty(0)

        _, trend_x, seasonal_x, self.slices = self._design(data["ds"])
        x = np.hstack([trend_x, seasonal_x])[fitted_rows.to_numpy()]
        y = (data["y"][fitted_rows] / self.y_scale).to_numpy(dtype=float)
        n_delta, n_season = len(self.changepoints_t), seasonal_x.shape[1]
        delta_cols = slice(2, 2 + n_delta)

        # MAP: sum(r^2)/(2 s^2) + sum|delta|/tau + sum(beta^2)/(2*10^2). Times 2 s^2,
        # a least-squares problem with an L1 penalty on delta (reweighted to ridge).
        tau = max(self.changepoint_prior_scale, 1e-6)
        sigma2 = np.var(y - x[:, :2] @ np.linalg.lstsq(x[:, :2], y, rcond=None)[0]) or EPS
        w = np.zeros(x.shape[1])
        gram, xty = x.T @ x, x.T @ y
        for _ in range(IRLS_STEPS):
            penalty = np.zeros(x.shape[1])
            penalty[:2] = EPS
            penalty[delta_cols] = (2 * sigma2 / tau) / (2 * (np.abs(w[delta_cols]) + EPS))
            penalty[2 + n_delta:] = sigma2 / SEASONALITY_PRIOR_SCALE**2
            new = np.linalg.solve(gram + np.diag(penalty), xty)
            sigma2 = max(np.mean((y - x @ new) ** 2), EPS)
            if np.max(np.abs(new - w)) < 1e-7:
                w = new
                break
            w = new
        delta = w[delta_cols]
        delta[np.abs(delta) < 1e-5] = 0.0  # IRLS leaves tiny values where the lasso has zeros.
        self.coef = w
        self.coef[delta_cols] = delta
        self.sigma = float(np.sqrt(sigma2))
        # Shaped like Prophet's (samples x changepoints), in scaled units like Prophet's.
        self.params = {"k": np.array([[w[1]]]), "m": np.array([[w[0]]]), "delta": delta[None, :],
                       "sigma_obs": np.array([[self.sigma]]), "beta": w[2 + n_delta:][None, :]}
        self.n_season = n_season
        return self

    # -- prediction -------------------------------------------------------------

    def make_future_dataframe(self, periods: int, freq: str = "D", include_history: bool = True) -> pd.DataFrame:
        last = self.history["ds"].max()
        dates = pd.date_range(start=last, periods=periods + 1, freq=freq)
        dates = dates[dates > last][:periods]
        if include_history:
            dates = np.concatenate([self.history["ds"].to_numpy(), dates.to_numpy()])
        return pd.DataFrame({"ds": pd.to_datetime(dates)})

    def predict(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if self.history is None:
            raise ValueError("Model has not been fit.")
        ds = pd.to_datetime((self.history if df is None else df)["ds"]).reset_index(drop=True)
        t, trend_x, seasonal_x, _ = self._design(ds)
        n_trend = trend_x.shape[1]
        trend = trend_x @ self.coef[:n_trend]
        out = pd.DataFrame({"ds": ds, "trend": trend * self.y_scale})
        additive = np.zeros(len(ds))
        for name, cols in self.slices.items():
            part = seasonal_x[:, cols] @ self.coef[n_trend:][cols]
            out[name] = part * self.y_scale
            additive += part
        out["additive_terms"] = additive * self.y_scale
        out["yhat"] = (trend + additive) * self.y_scale
        lower, upper = self._interval(t, trend + additive)
        out["yhat_lower"], out["yhat_upper"] = lower * self.y_scale, upper * self.y_scale
        out["trend_lower"] = out["trend_upper"] = out["trend"]
        return out

    def _interval(self, t: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Prophet's simulation: new changepoints after the history, plus noise."""
        rng = np.random.default_rng(0)
        samples = self.uncertainty_samples
        noise = rng.normal(0.0, self.sigma, (samples, len(t)))
        future = t > 1
        delta = self.coef[2:2 + len(self.changepoints_t)]
        if future.any() and len(delta):
            rate = len(self.changepoints_t)  # Changepoints per unit of scaled time.
            scale = np.mean(np.abs(delta)) + 1e-8
            t_max = t.max()
            counts = rng.poisson(rate * (t_max - 1), samples)
            for i, count in enumerate(counts):
                if count:
                    where = rng.uniform(1, t_max, count)
                    size = rng.laplace(0, scale, count)
                    noise[i] += (np.maximum(0.0, t[:, None] - where[None, :]) * size).sum(axis=1)
        draws = yhat[None, :] + noise
        tail = (1 - self.interval_width) / 2
        return np.quantile(draws, tail, axis=0), np.quantile(draws, 1 - tail, axis=0)
