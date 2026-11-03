"""Write data/candlestick_prices.csv: one year of daily prices for the candlestick plot.

Run from the repository root: ``python scripts/make_candlestick_data.py``.
The data is synthetic (a seeded random walk), so the file is the same every run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "candlestick_prices.csv"
SEED = 7


def prices(start: str = "2025-01-02", days: int = 252) -> pd.DataFrame:
    """Business-day OHLC prices with volume, plus Week and Month labels to group by."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(start, periods=days)
    # Daily returns with a mild upward drift and a calmer second half of the year.
    volatility = np.where(np.arange(days) < days // 2, 0.018, 0.011)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0006, volatility)))
    # Each day opens near the previous close (small overnight gap).
    open_ = np.r_[100.0, close[:-1]] * (1 + rng.normal(0, 0.004, days))
    body_top = np.maximum(open_, close)
    body_bottom = np.minimum(open_, close)
    high = body_top * (1 + np.abs(rng.normal(0, 0.006, days)))
    low = body_bottom * (1 - np.abs(rng.normal(0, 0.006, days)))
    volume = rng.lognormal(13.5, 0.35, days) * (1 + 8 * np.abs(close / open_ - 1))
    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Week": dates.strftime("%G-W%V"),
            "Month": dates.strftime("%Y-%m"),
            "Quarter": "Q" + dates.quarter.astype(str),
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Volume": volume.round().astype(int),
            "Change": (close - open_).round(2),
        }
    )


if __name__ == "__main__":
    OUTPUT.parent.mkdir(exist_ok=True)
    frame = prices()
    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(frame)} rows to {OUTPUT.relative_to(ROOT)}")
