"""Per-bar market facts for entry filters (causal; identical in backtests and live)."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from trading_lab.config import RiskConfig, VotingConfig
from trading_lab.core.symbols import timeframe_to_seconds

WARMUP = "warm-up"


def trend_labels(close: pd.Series, trend_bars: int, slope_bars: int) -> np.ndarray:
    """Per bar: "up" (close above a rising ``trend_bars`` SMA), "down" (below a falling one),
    "sideways", or ``WARMUP`` before the average exists. Uses bars up to each bar only."""
    sma = close.rolling(trend_bars, min_periods=trend_bars).mean()
    rising, falling = sma > sma.shift(slope_bars), sma < sma.shift(slope_bars)
    return np.where(sma.isna(), WARMUP,
                    np.where((close > sma) & rising, "up", np.where((close < sma) & falling, "down", "sideways")))


def _filter_ratio(risk: RiskConfig, timeframe: str) -> int:
    if not risk.trend_filter_timeframe:
        return 1
    return timeframe_to_seconds(risk.trend_filter_timeframe) // timeframe_to_seconds(timeframe)


def trend_filter_history(risk: RiskConfig, timeframe: str) -> int:
    """Bars of ``timeframe`` the trend filter needs before the first decision (0 when off)."""
    if risk.trend_filter_period <= 0:
        return 0
    return (risk.trend_filter_period + 1) * _filter_ratio(risk, timeframe)


def trend_filter_label(risk: RiskConfig) -> str:
    """How the average is named in decision reasons, e.g. "200-bar" or "100 x 1d"."""
    if not risk.trend_filter_timeframe:
        return f"{risk.trend_filter_period}-bar"
    return f"{risk.trend_filter_period} x {risk.trend_filter_timeframe}"


def higher_timeframe_sma(candles: pd.DataFrame, timeframe: str, filter_timeframe: str, period: int) -> np.ndarray:
    """Per bar: the average of the last ``period`` completed ``filter_timeframe`` closes (NaN if too few).

    A period's close is the close of its last bar (periods are aligned to UTC
    midnight, like exchange candles). A period counts once a bar closing at or
    after its end has closed, so the period still in progress never does.
    Periods are complete whatever the data window's start, so live runs see the
    backtest's values once the window holds ``period`` complete periods.
    """
    if candles.empty:
        return np.array([], dtype="float64")
    length = pd.Timedelta(seconds=timeframe_to_seconds(filter_timeframe))
    buckets = candles.index.floor(length)
    closes = candles["close"].groupby(buckets).last()
    values = closes.to_numpy(dtype="float64")
    sma = np.full(len(values), np.nan)
    if len(values) >= period:  # each window summed on its own: exact whatever the window's start
        sma[period - 1:] = np.lib.stride_tricks.sliding_window_view(values, period).mean(axis=1)
    ends = (closes.index + length).as_unit("ns").asi8
    bar_closes = (candles.index + pd.Timedelta(seconds=timeframe_to_seconds(timeframe))).as_unit("ns").asi8
    latest = np.searchsorted(ends, bar_closes, side="right") - 1
    return np.where(latest >= 0, sma[np.maximum(latest, 0)], np.nan)


def filter_columns(
    candles: pd.DataFrame, risk: RiskConfig, voting: VotingConfig | None = None, timeframe: str = "1h",
) -> list[dict[str, float | str | None]]:
    """Per-bar market facts, each from candles up to and including the bar:

    * ``trend_sma``: simple average of the last ``trend_filter_period`` closes
      (trend filter), or of that many completed ``trend_filter_timeframe``
      closes when one is set (``timeframe`` is the candles' own);
    * ``regime``: the trend label for regime-dependent strategy weights (``voting.regime_weights``).

    Simple (not exponential) averages do not depend on where the data window
    starts, so live paper trading sees exactly the backtest's values.
    """
    rows: list[dict[str, float | str | None]] = [{} for _ in range(len(candles))]
    if risk.trend_filter_period > 0:
        period = risk.trend_filter_period
        if risk.trend_filter_timeframe and risk.trend_filter_timeframe != timeframe:
            sma = higher_timeframe_sma(candles, timeframe, risk.trend_filter_timeframe, period)
        else:
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
