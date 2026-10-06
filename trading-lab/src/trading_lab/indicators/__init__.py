"""Technical indicators as pure functions of a price series.

Every function is causal: the value at bar ``t`` depends only on bars ``<= t``.
Values are NaN until enough history exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_period(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, ``alpha = 2 / (span + 1)``, seeded with the first value.

    Values are NaN for the first ``span - 1`` bars.
    """
    _check_period("span", span)
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing.

    The first value (at bar ``period``) uses simple averages of the first
    ``period`` changes. After that:
    ``avg = (prev_avg * (period - 1) + current) / period``.
    If there are no losses, RSI is 100 (or 50 when there is no movement at all).
    """
    _check_period("period", period)
    values = close.to_numpy(dtype="float64")
    out = np.full(values.size, np.nan)
    if values.size <= period:
        return pd.Series(out, index=close.index, name="rsi")

    delta = np.diff(values)
    gains = np.clip(delta, 0.0, None)
    losses = np.clip(-delta, 0.0, None)

    def value(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0.0:
            return 100.0 if avg_gain > 0.0 else 50.0
        return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = value(avg_gain, avg_loss)
    for i in range(period, delta.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = value(avg_gain, avg_loss)
    return pd.Series(out, index=close.index, name="rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line (fast EMA - slow EMA), signal line (EMA of MACD) and histogram."""
    _check_period("fast", fast)
    _check_period("slow", slow)
    _check_period("signal", signal)
    if fast >= slow:
        raise ValueError(f"fast period ({fast}) must be shorter than slow period ({slow})")
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": line, "signal": signal_line, "hist": line - signal_line}, index=close.index
    )


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands using the population standard deviation (Bollinger's definition).

    Also returns ``percent_b = (close - lower) / (upper - lower)``, which is NaN
    when the bands have zero width, and ``bandwidth = (upper - lower) / middle``.
    """
    _check_period("period", period)
    if not np.isfinite(num_std) or num_std <= 0:
        raise ValueError(f"num_std must be positive, got {num_std!r}")
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    percent_b = ((close - lower) / width).where(width > 0)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "percent_b": percent_b,
            "bandwidth": width / middle,
        },
        index=close.index,
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range with Wilder's smoothing.

    True range = max(high - low, |high - previous close|, |low - previous close|).
    The first value (at bar ``period - 1``) is the simple mean of the first
    ``period`` true ranges; after that ``atr = (prev * (period - 1) + tr) / period``.
    """
    _check_period("period", period)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=True)
    values = true_range.to_numpy(dtype="float64")
    out = np.full(values.size, np.nan)
    if values.size >= period:
        current = values[:period].mean()
        out[period - 1] = current
        for i in range(period, values.size):
            current = (current * (period - 1) + values[i]) / period
            out[i] = current
    return pd.Series(out, index=close.index, name="atr")


__all__ = ["atr", "bollinger_bands", "ema", "macd", "rsi"]
