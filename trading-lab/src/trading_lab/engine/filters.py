"""Per-bar market facts for entry filters (causal; identical in backtests and live)."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from trading_lab.config import RiskConfig, VotingConfig

WARMUP = "warm-up"


def trend_labels(close: pd.Series, trend_bars: int, slope_bars: int) -> np.ndarray:
    """Per bar: "up" (close above a rising ``trend_bars`` SMA), "down" (below a falling one),
    "sideways", or ``WARMUP`` before the average exists. Uses bars up to each bar only."""
    sma = close.rolling(trend_bars, min_periods=trend_bars).mean()
    rising, falling = sma > sma.shift(slope_bars), sma < sma.shift(slope_bars)
    return np.where(sma.isna(), WARMUP,
                    np.where((close > sma) & rising, "up", np.where((close < sma) & falling, "down", "sideways")))


def filter_columns(
    candles: pd.DataFrame, risk: RiskConfig, voting: VotingConfig | None = None
) -> list[dict[str, float | str | None]]:
    """Per-bar market facts, each from candles up to and including the bar:

    * ``trend_sma``: simple average of the last ``trend_filter_period`` closes (trend filter);
    * ``regime``: the trend label for regime-dependent strategy weights (``voting.regime_weights``).

    Simple (not exponential) averages do not depend on where the data window
    starts, so live paper trading sees exactly the backtest's values.
    """
    rows: list[dict[str, float | str | None]] = [{} for _ in range(len(candles))]
    if risk.trend_filter_period > 0:
        period = risk.trend_filter_period
        sma = candles["close"].rolling(period, min_periods=period).mean().to_numpy()
        for row, v in zip(rows, sma):
            row["trend_sma"] = float(v) if math.isfinite(v) else None
    if voting is not None and voting.regime_weights:
        labels = trend_labels(candles["close"], voting.regime_bars, voting.regime_slope_bars)
        for row, label in zip(rows, labels):
            row["regime"] = None if label == WARMUP else str(label)
    return rows


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
