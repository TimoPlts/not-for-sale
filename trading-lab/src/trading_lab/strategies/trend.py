"""Trend-following strategies: moving-average crossover and Donchian channel breakout.

The built-in RSI and Bollinger strategies buy weakness; these two follow
strength, which suits trending markets and (with ``risk.allow_short``) both
directions. Both are opt-in: enable them with a ``[strategies.ma_cross]`` or
``[strategies.donchian]`` table.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from trading_lab.core.models import Direction
from trading_lab.indicators import atr, ema
from trading_lab.strategies.base import IndicatorStrategy, finite_or_none, scaled_confidence
from trading_lab.strategies.registry import register_strategy
from trading_lab.strategies.validation import check_int


@register_strategy
class MovingAverageCrossStrategy(IndicatorStrategy):
    """Fast versus slow moving average of the close (``average`` = "sma" or "ema").

    * ``signal_on = "cross"`` (default): BUY on the bar where the fast average
      crosses above the slow one, SELL where it crosses below, HOLD otherwise.
    * ``signal_on = "state"``: BUY on every bar the fast average is above the
      slow one and SELL on every bar it is below (always in the market's
      direction; the engine ignores repeats while a position is open).

    Confidence measures the move against its usual size: for a cross, the
    one-bar change of the gap (fast - slow); for state, the gap itself; both
    divided by the gap's standard deviation over ``normalization_window`` bars.
    """

    name = "ma_cross"

    def __init__(
        self, fast: int = 20, slow: int = 50, average: str = "sma", signal_on: str = "cross",
        normalization_window: int = 50,
    ) -> None:
        self.fast = check_int("fast", fast)
        self.slow = check_int("slow", slow, minimum=2)
        if self.fast >= self.slow:
            raise ValueError("fast must be shorter than slow")
        if average not in ("sma", "ema"):
            raise ValueError(f"average must be 'sma' or 'ema', got {average!r}")
        if signal_on not in ("cross", "state"):
            raise ValueError(f"signal_on must be 'cross' or 'state', got {signal_on!r}")
        self.average, self.signal_on = average, signal_on
        self.normalization_window = check_int("normalization_window", normalization_window, minimum=2)

    @property
    def warmup_bars(self) -> int:
        return self.slow + 1  # the slow average plus one bar to see a cross

    @property
    def params(self) -> dict[str, Any]:
        return {"fast": self.fast, "slow": self.slow, "average": self.average, "signal_on": self.signal_on,
                "normalization_window": self.normalization_window}

    def _ma(self, close: pd.Series, n: int) -> pd.Series:
        return ema(close, n) if self.average == "ema" else close.rolling(n, min_periods=n).mean()

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        close = candles["close"]
        frame = pd.DataFrame({"fast_ma": self._ma(close, self.fast), "slow_ma": self._ma(close, self.slow)})
        frame["gap"] = frame["fast_ma"] - frame["slow_ma"]
        frame["prev_gap"] = frame["gap"].shift(1)
        frame["gap_std"] = frame["gap"].rolling(self.normalization_window, min_periods=2).std(ddof=0)
        return frame

    def _decide(self, row: Mapping[str, Any]) -> tuple[Direction, float, dict[str, Any]]:
        gap, prev, scale = float(row["gap"]), float(row["prev_gap"]), float(row["gap_std"])
        meta: dict[str, Any] = {"fast_ma": finite_or_none(row["fast_ma"]), "slow_ma": finite_or_none(row["slow_ma"]),
                                "gap": finite_or_none(gap), "crossover": None}
        if math.isnan(gap) or math.isnan(prev):
            return Direction.HOLD, 0.0, meta
        if self.signal_on == "state":
            strength = abs(gap) / scale if scale > 0 else 0.0
            if gap > 0:
                return Direction.BUY, scaled_confidence(strength), meta
            if gap < 0:
                return Direction.SELL, scaled_confidence(strength), meta
            return Direction.HOLD, 0.0, meta
        strength = abs(gap - prev) / scale if scale > 0 else 0.0
        if prev <= 0.0 < gap:
            meta["crossover"] = "bullish"
            return Direction.BUY, scaled_confidence(strength), meta
        if prev >= 0.0 > gap:
            meta["crossover"] = "bearish"
            return Direction.SELL, scaled_confidence(strength), meta
        return Direction.HOLD, 0.0, meta


@register_strategy
class DonchianBreakoutStrategy(IndicatorStrategy):
    """Channel breakout ("turtle" style), on channels of the bars *before* the current one.

    * BUY when the close breaks above the highest high of the previous
      ``entry_period`` bars, SELL when it breaks below the lowest low.
    * With ``exit_period`` > 0, a close below the lowest low of the previous
      ``exit_period`` bars (but not the entry channel) is a weaker SELL
      (confidence 0.5) that exits longs early; the mirror case is a weaker BUY.
      With ``risk.allow_short`` such signals can also open positions.

    Confidence grows with the breakout distance measured in ATRs
    (``atr_period``): 0.5 at the channel, 1.0 one ATR beyond it.
    """

    name = "donchian"

    def __init__(self, entry_period: int = 20, exit_period: int = 10, atr_period: int = 14) -> None:
        self.entry_period = check_int("entry_period", entry_period, minimum=2)
        self.exit_period = check_int("exit_period", exit_period, minimum=0)
        if self.exit_period >= self.entry_period:
            raise ValueError("exit_period must be shorter than entry_period (or 0 to disable)")
        self.atr_period = check_int("atr_period", atr_period, minimum=2)

    @property
    def warmup_bars(self) -> int:
        return max(self.entry_period, self.atr_period) + 1

    @property
    def params(self) -> dict[str, Any]:
        return {"entry_period": self.entry_period, "exit_period": self.exit_period, "atr_period": self.atr_period}

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        high, low = candles["high"].shift(1), candles["low"].shift(1)  # channels exclude the current bar
        frame = pd.DataFrame({
            "close": candles["close"],
            "upper": high.rolling(self.entry_period, min_periods=self.entry_period).max(),
            "lower": low.rolling(self.entry_period, min_periods=self.entry_period).min(),
            "atr": atr(candles["high"], candles["low"], candles["close"], self.atr_period),
        })
        if self.exit_period:
            frame["exit_high"] = high.rolling(self.exit_period, min_periods=self.exit_period).max()
            frame["exit_low"] = low.rolling(self.exit_period, min_periods=self.exit_period).min()
        return frame

    def _decide(self, row: Mapping[str, Any]) -> tuple[Direction, float, dict[str, Any]]:
        close, upper, lower, band = (float(row[k]) for k in ("close", "upper", "lower", "atr"))
        meta: dict[str, Any] = {"upper": finite_or_none(upper), "lower": finite_or_none(lower),
                                "atr": finite_or_none(band), "breakout": None}
        if math.isnan(upper) or math.isnan(lower):
            return Direction.HOLD, 0.0, meta

        def strength(distance: float) -> float:
            return distance / band if band > 0 else 0.0

        if close > upper:
            meta["breakout"] = "up"
            return Direction.BUY, scaled_confidence(strength(close - upper)), meta
        if close < lower:
            meta["breakout"] = "down"
            return Direction.SELL, scaled_confidence(strength(lower - close)), meta
        if self.exit_period:
            exit_low, exit_high = float(row["exit_low"]), float(row["exit_high"])
            if close < exit_low:
                meta["breakout"] = "exit_down"
                return Direction.SELL, 0.5, meta
            if close > exit_high:
                meta["breakout"] = "exit_up"
                return Direction.BUY, 0.5, meta
        return Direction.HOLD, 0.0, meta
