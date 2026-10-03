"""MACD signal-line crossover strategy."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from trading_lab.core.models import Direction
from trading_lab.indicators import macd
from trading_lab.strategies.base import Strategy, finite_or_none, scaled_confidence
from trading_lab.strategies.registry import register_strategy
from trading_lab.strategies.validation import check_int


@register_strategy
class MacdStrategy(Strategy):
    """BUY on the bar where the MACD histogram crosses above zero (bullish
    crossover), SELL where it crosses below zero, and HOLD otherwise.

    Confidence measures how decisive the crossover is: the one-bar change in
    the histogram divided by the histogram's standard deviation over the last
    ``normalization_window`` bars. It maps to 0.5 when the change is small and
    to 1.0 at one standard deviation or more.
    """

    name = "macd"

    def __init__(
        self, fast: int = 12, slow: int = 26, signal: int = 9, normalization_window: int = 50
    ) -> None:
        self.fast = check_int("fast", fast)
        self.slow = check_int("slow", slow, minimum=2)
        self.signal = check_int("signal", signal)
        self.normalization_window = check_int("normalization_window", normalization_window, minimum=2)
        if self.fast >= self.slow:
            raise ValueError("fast must be shorter than slow")

    @property
    def warmup_bars(self) -> int:
        # The first histogram value appears at bar slow + signal - 2, and a
        # crossover needs one more bar.
        return self.slow + self.signal

    @property
    def params(self) -> dict[str, Any]:
        return {
            "fast": self.fast,
            "slow": self.slow,
            "signal": self.signal,
            "normalization_window": self.normalization_window,
        }

    def _evaluate(self, candles: pd.DataFrame) -> tuple[Direction, float, dict[str, Any]]:
        frame = macd(candles["close"], self.fast, self.slow, self.signal)
        last, prev = frame.iloc[-1], frame.iloc[-2]
        hist, prev_hist = float(last["hist"]), float(prev["hist"])
        meta: dict[str, Any] = {
            "macd": finite_or_none(last["macd"]),
            "signal": finite_or_none(last["signal"]),
            "hist": finite_or_none(hist),
            "prev_hist": finite_or_none(prev_hist),
            "crossover": None,
        }
        if math.isnan(hist) or math.isnan(prev_hist):
            return Direction.HOLD, 0.0, meta

        history = frame["hist"].dropna().iloc[-self.normalization_window :]
        scale = float(history.std(ddof=0)) if len(history) > 1 else 0.0
        strength = abs(hist - prev_hist) / scale if scale > 0 else 0.0
        meta["hist_std"] = scale

        if prev_hist <= 0.0 < hist:
            meta["crossover"] = "bullish"
            return Direction.BUY, scaled_confidence(strength), meta
        if prev_hist >= 0.0 > hist:
            meta["crossover"] = "bearish"
            return Direction.SELL, scaled_confidence(strength), meta
        return Direction.HOLD, 0.0, meta
