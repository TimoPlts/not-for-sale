"""Trend-following strategies: moving-average crossover, Donchian channel breakout and
time-series momentum.

The built-in RSI and Bollinger strategies buy weakness; these follow
strength, which suits trending markets and (with ``risk.allow_short``) both
directions. All are opt-in: enable them with a ``[strategies.ma_cross]``,
``[strategies.donchian]`` or ``[strategies.tsmom]`` table.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
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


@register_strategy
class TimeSeriesMomentumStrategy(IndicatorStrategy):
    """Time-series momentum: is the price higher than it was N bars ago, over several horizons?

    For each of the ``lookbacks`` (in bars) the strategy looks at the return
    over that horizon. The score is (horizons up - horizons down) / horizons,
    from -1 to 1. BUY when the score is at least ``threshold``, SELL when it
    is at most ``-threshold``, HOLD in between (so one horizon turning does
    not flip a position: with three horizons and the default threshold, two
    must agree).

    Every horizon is long, so it trades rarely: suited to daily bars
    (``market.timeframe = "1d"``, the defaults are about one, three and six
    months). On hourly bars use horizons in the hundreds.

    Confidence grows with the size of the agreeing moves measured in their
    usual size: each return divided by the volatility of one-bar log returns
    (``vol_window`` bars) times the square root of its horizon. 0.5 when the
    moves are tiny, 1.0 at two standard deviations.
    """

    name = "tsmom"

    def __init__(self, lookbacks: Any = (20, 60, 120), threshold: float = 0.3, vol_window: int = 20) -> None:
        if not isinstance(lookbacks, (list, tuple)) or not lookbacks:
            raise ValueError("lookbacks must be a non-empty list of bar counts")
        checked = [check_int("lookbacks", n) for n in lookbacks]
        if len(set(checked)) != len(checked):
            raise ValueError("lookbacks contains duplicates")
        if max(checked) > 5000:
            raise ValueError("lookbacks must be at most 5000 bars")
        self.lookbacks = tuple(sorted(checked))
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
        self.threshold = float(threshold)
        self.vol_window = check_int("vol_window", vol_window, minimum=2)

    @property
    def warmup_bars(self) -> int:
        return max(max(self.lookbacks), self.vol_window) + 1

    @property
    def history_bars(self) -> int:
        return self.warmup_bars + 1  # plain returns and a rolling deviation: nothing to converge

    @property
    def params(self) -> dict[str, Any]:
        return {"lookbacks": list(self.lookbacks), "threshold": self.threshold, "vol_window": self.vol_window}

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        close = candles["close"]
        frame = pd.DataFrame({"vol": np.log(close).diff().rolling(self.vol_window, min_periods=self.vol_window)
                             .std(ddof=0)})
        for n in self.lookbacks:
            frame[f"ret_{n}"] = close / close.shift(n) - 1.0
        return frame

    def _decide(self, row: Mapping[str, Any]) -> tuple[Direction, float, dict[str, Any]]:
        returns = {n: float(row[f"ret_{n}"]) for n in self.lookbacks}
        vol = float(row["vol"])
        meta: dict[str, Any] = {"returns": {str(n): finite_or_none(r) for n, r in returns.items()},
                                "vol": finite_or_none(vol), "score": None}
        if any(math.isnan(r) for r in returns.values()):
            return Direction.HOLD, 0.0, meta
        score = (sum(r > 0 for r in returns.values()) - sum(r < 0 for r in returns.values())) / len(returns)
        meta["score"] = score
        if abs(score) < self.threshold:
            return Direction.HOLD, 0.0, meta
        sign = 1.0 if score > 0 else -1.0
        if vol > 0 and math.isfinite(vol):
            z = [sign * math.log1p(r) / (vol * math.sqrt(n)) for n, r in returns.items() if sign * r > 0]
            strength = sum(z) / len(z) / 2.0
        else:
            strength = 0.0
        return (Direction.BUY if sign > 0 else Direction.SELL), scaled_confidence(strength), meta
