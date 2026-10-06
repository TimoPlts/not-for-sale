"""Per-bar market facts for entry filters (causal; identical in backtests and live)."""

from __future__ import annotations

import math

import pandas as pd

from trading_lab.config import RiskConfig


def filter_columns(candles: pd.DataFrame, risk: RiskConfig) -> list[dict[str, float | None]]:
    """For each bar: ``{"trend_sma": simple average of the last N closes, up to and including the bar}``.

    A simple (not exponential) average does not depend on where the data
    window starts, so live paper trading sees exactly the backtest's values.
    """
    if risk.trend_filter_period <= 0:
        return [{} for _ in range(len(candles))]
    period = risk.trend_filter_period
    sma = candles["close"].rolling(period, min_periods=period).mean().to_numpy()
    return [{"trend_sma": float(v) if math.isfinite(v) else None} for v in sma]
