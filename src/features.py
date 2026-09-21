"""Past-only OHLCV feature engineering."""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


BASE_FEATURES = ["open", "high", "low", "close", "volume"]


def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Create strictly past/current-only features within each ticker.

    The input must contain one row per (Name, date), sorted by Name/date.
    No centered windows or future shifts are used here. The only future shift
    in the experiment is the target, which is created separately in data_prep.py.
    """
    required = {"date", "open", "high", "low", "close", "volume", "Name", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {sorted(missing)}")

    out = df.copy()
    g = out.groupby("Name", sort=False, group_keys=False)

    out["daily_return"] = g["close"].pct_change(1)
    out["return_2d"] = g["close"].pct_change(2)
    out["return_3d"] = g["close"].pct_change(3)
    out["return_5d"] = g["close"].pct_change(5)

    out["intraday_range"] = (out["high"] - out["low"]) / out["close"]
    prev_close = g["close"].shift(1)
    out["close_to_open_gap"] = (out["open"] / prev_close) - 1.0

    for n in (5, 10, 20):
        sma = g["close"].transform(lambda s, n=n: s.rolling(n, min_periods=n).mean())
        out[f"sma_{n}"] = sma
        out[f"close_sma_ratio_{n}"] = out["close"] / sma

    for n in (5, 10, 20):
        out[f"volatility_{n}"] = g["daily_return"].transform(
            lambda s, n=n: s.rolling(n, min_periods=n).std()
        )

    volume_ma20 = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    out["volume_ratio_20"] = out["volume"] / volume_ma20

    # RSI(14), Wilder-style EWMA. It uses current/past deltas only.
    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(out["Name"]).transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    avg_loss = loss.groupby(out["Name"]).transform(
        lambda s: s.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    # If average loss is exactly zero, RSI is conventionally 100.
    out.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100.0
    out.loc[(avg_loss == 0) & (avg_gain == 0), "rsi_14"] = 50.0

    for lag in range(1, 6):
        out[f"return_lag_{lag}"] = g["daily_return"].shift(lag)

    feature_names = (
        BASE_FEATURES
        + [
            "daily_return",
            "return_2d",
            "return_3d",
            "return_5d",
            "intraday_range",
            "close_to_open_gap",
            "sma_5",
            "sma_10",
            "sma_20",
            "close_sma_ratio_5",
            "close_sma_ratio_10",
            "close_sma_ratio_20",
            "volatility_5",
            "volatility_10",
            "volatility_20",
            "volume_ratio_20",
            "rsi_14",
            "return_lag_1",
            "return_lag_2",
            "return_lag_3",
            "return_lag_4",
            "return_lag_5",
        ]
    )

    if len(feature_names) != 27:
        raise AssertionError(f"Expected 27 features, got {len(feature_names)}")

    return out, feature_names
