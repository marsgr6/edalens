"""Streamlit-free forecasting and residual anomaly detection for the Time Series page.

Uses statsmodels (Holt-Winters / ETS and SARIMA) and scikit-learn's Isolation
Forest, which also run in the browser (Pyodide), unlike Prophet. Figures are
Plotly, matching the rest of the app.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

ETS = "Holt-Winters (ETS)"
SARIMA = "SARIMA"
METHODS = (ETS, SARIMA)

# Season length per summarization frequency (None: no seasonality).
SEASON_BY_GRANULARITY = {
    "Hourly": 24,
    "Daily": 7,
    "Weekly": 52,
    "Monthly": 12,
    "Quarterly": 4,
    "Yearly": None,
}
# Season length per pandas frequency code, for raw (unsummarized) data.
SEASON_BY_FREQ = {
    "h": 24,
    "H": 24,
    "D": 7,
    "W": 52,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
}
# Seasonal SARIMA gets slow with long seasons (the browser is ~10x slower); above
# this it falls back to a non-seasonal ARIMA.
MAX_SARIMA_SEASON = 12
ANOMALY_COLOR = "#d62728"


# A frequency is accepted for gappy data when the dates fill this share of its grid.
MIN_GRID_COVERAGE = 0.8


class ForecastError(ValueError):
    """The series cannot be modelled; the message is shown to users."""


def _grid_frequency(index: pd.DatetimeIndex) -> Optional[str]:
    """A common frequency whose grid contains every date and is mostly filled.

    ``pd.infer_freq`` needs a complete index; real data often misses a few steps.
    """
    weekly = f"W-{index[0].day_name()[:3].upper()}"
    for freq in ("h", "D", weekly, "MS", "ME", "QS", "QE", "YS", "YE"):
        try:
            grid = pd.date_range(index[0], index[-1], freq=freq)
        except ValueError:  # Alias unknown to this pandas version.
            continue
        if (
            len(grid)
            and index.isin(grid).all()
            and len(index) >= MIN_GRID_COVERAGE * len(grid)
        ):
            return freq
    return None


def regular_series(series: pd.Series) -> pd.Series:
    """``series`` with a regular DatetimeIndex frequency and no gaps.

    Raises ``ForecastError`` when no frequency can be inferred (irregular raw data).
    """
    series = series.dropna()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if len(series) < 8:
        raise ForecastError(
            "At least 8 observations are needed to fit a forecasting model."
        )
    freq = (
        series.index.freq
        or pd.infer_freq(series.index)
        or _grid_frequency(series.index)
    )
    if freq is None:
        raise ForecastError(
            "The dates are not evenly spaced, so a forecasting model cannot be fitted. "
            "Choose a summarization frequency (e.g. Daily or Monthly) in the sidebar."
        )
    # Fill the occasional missing step so the models see a complete grid.
    return series.asfreq(freq).interpolate(limit_direction="both")


def season_length(series: pd.Series, granularity: str = "None") -> Optional[int]:
    """Season length for the series, or None when there are not two full seasons."""
    if granularity in SEASON_BY_GRANULARITY:
        season = SEASON_BY_GRANULARITY[granularity]
    else:
        code = pd.tseries.frequencies.to_offset(series.index.freq).name.split("-")[0]
        season = SEASON_BY_FREQ.get(code)
    return season if season and len(series) >= 2 * season else None


def fit_forecast(
    series: pd.Series,
    method: str,
    horizon: int,
    season: Optional[int],
    ci_level: float = 95,
) -> pd.DataFrame:
    """Fit ``method`` to a regular series and forecast ``horizon`` steps.

    Returns a frame indexed by date with ``observed`` (history only), ``fitted``
    (in-sample one-step predictions) and ``forecast``/``lower``/``upper``
    (future only, with a ``ci_level`` % prediction interval).
    """
    alpha = 1 - ci_level / 100
    try:
        if method == ETS:
            from statsmodels.tsa.exponential_smoothing.ets import ETSModel

            model = ETSModel(
                series.astype(float),
                error="add",
                trend="add",
                damped_trend=True,
                seasonal="add" if season else None,
                seasonal_periods=season,
            )
            result = model.fit(disp=False)
            fitted = result.fittedvalues
            prediction = result.get_prediction(
                start=len(series), end=len(series) + horizon - 1
            )
            frame = prediction.summary_frame(alpha=alpha)
            future = pd.DataFrame(
                {
                    "forecast": frame["mean"],
                    "lower": frame["pi_lower"],
                    "upper": frame["pi_upper"],
                }
            )
        elif method == SARIMA:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            seasonal = season if season and season <= MAX_SARIMA_SEASON else None
            model = SARIMAX(
                series.astype(float),
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, seasonal) if seasonal else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            fitted = result.fittedvalues
            # The first differenced steps have no real one-step prediction.
            fitted.iloc[: 1 + (seasonal or 0)] = np.nan
            frame = result.get_forecast(steps=horizon).summary_frame(alpha=alpha)
            future = pd.DataFrame(
                {
                    "forecast": frame["mean"],
                    "lower": frame["mean_ci_lower"],
                    "upper": frame["mean_ci_upper"],
                }
            )
        else:
            raise ForecastError(f"Unknown forecasting method: {method}")
    except (ValueError, np.linalg.LinAlgError) as exc:
        if isinstance(exc, ForecastError):
            raise
        raise ForecastError(
            f"{method} could not be fitted to this series ({exc})."
        ) from exc
    history = pd.DataFrame({"observed": series, "fitted": fitted})
    return pd.concat([history, future])


def residual_anomalies(result: pd.DataFrame, contamination: float = 0.05) -> pd.Series:
    """Isolation Forest on the model's residuals: True where a point is anomalous.

    A residual is what the model did not expect given trend and season, so
    this flags unusual points for their time, not just extreme values.
    """
    history = result.dropna(subset=["observed", "fitted"])
    flags = pd.Series(False, index=result.index[result["observed"].notna()])
    if len(history) < 10:
        return flags
    residual = (history["observed"] - history["fitted"]).to_numpy().reshape(-1, 1)
    labels = IsolationForest(contamination=contamination, random_state=42).fit_predict(
        residual
    )
    flags.loc[history.index] = labels == -1
    return flags


def forecast_figure(
    result: pd.DataFrame,
    title: str,
    y_label: str,
    ci_level: float = 95,
    anomalies: Optional[pd.Series] = None,
) -> go.Figure:
    """History, in-sample fit, forecast with its interval band and optional anomalies."""
    history = result[result["observed"].notna()]
    future = result[result["forecast"].notna()]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index, y=history["observed"], name="Observed", mode="lines"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["fitted"],
            name="Fitted",
            mode="lines",
            line=dict(dash="dot", width=1),
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([future.index, future.index[::-1]]),
            y=np.concatenate([future["upper"], future["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name=f"{ci_level:g}% interval",
        )
    )
    fig.add_trace(
        go.Scatter(x=future.index, y=future["forecast"], name="Forecast", mode="lines")
    )
    if anomalies is not None and anomalies.any():
        points = history.loc[anomalies[anomalies].index]
        fig.add_trace(
            go.Scatter(
                x=points.index,
                y=points["observed"],
                name="Anomaly (Isolation Forest)",
                mode="markers",
                marker=dict(
                    color=ANOMALY_COLOR,
                    size=8,
                    symbol="circle-open",
                    line=dict(width=2),
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        # Below the plot: five entries above it would overlap the title.
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(b=110),
    )
    return fig
