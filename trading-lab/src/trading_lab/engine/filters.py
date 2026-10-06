"""Per-bar market facts for entry filters (causal; identical in backtests and live)."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
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


def correlation_lookup(
    candles: Mapping[str, pd.DataFrame], risk: RiskConfig
) -> dict[str, dict[pd.Timestamp, dict[str, float | None]]]:
    """``{symbol: {bar time: {other symbol: correlation}}}`` of per-bar log returns.

    Rolling over ``risk.correlation_lookback`` bars up to and including the
    bar (causal, fixed window, so identical live and in backtests). Bars are
    aligned on time; a window with a gap gives None (unknown).
    """
    if risk.max_correlated_positions <= 0 or len(candles) < 2:
        return {sym: {} for sym in candles}
    closes = pd.DataFrame({sym: frame["close"] for sym, frame in candles.items()}).sort_index()
    returns = np.log(closes).diff()
    n = risk.correlation_lookback
    out: dict[str, dict[pd.Timestamp, dict[str, float | None]]] = {sym: {} for sym in candles}
    symbols = list(candles)
    for a_i, a in enumerate(symbols):
        for b in symbols[a_i + 1:]:
            corr = returns[a].rolling(n, min_periods=n).corr(returns[b])
            for ts, value in corr.items():
                v = float(value) if math.isfinite(value) else None
                out[a].setdefault(ts, {})[b] = v
                out[b].setdefault(ts, {})[a] = v
    return out
