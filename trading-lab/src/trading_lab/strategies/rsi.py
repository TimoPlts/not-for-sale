"""RSI overbought/oversold strategy."""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from trading_lab.core.models import Direction
from trading_lab.indicators import rsi
from trading_lab.strategies.base import IndicatorStrategy, finite_or_none, scaled_confidence
from trading_lab.strategies.registry import register_strategy
from trading_lab.strategies.validation import check_int, check_range


@register_strategy
class RsiStrategy(IndicatorStrategy):
    """BUY while RSI < ``oversold``, SELL while RSI > ``overbought``, otherwise HOLD.

    Confidence goes from 0.5 at the threshold to 1.0 at RSI 0 (BUY) or 100 (SELL).
    """

    name = "rsi"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        self.period = check_int("period", period, minimum=2)
        self.oversold = check_range("oversold", oversold, 0.0, 100.0)
        self.overbought = check_range("overbought", overbought, 0.0, 100.0)
        if not self.oversold < self.overbought:
            raise ValueError("oversold must be below overbought")

    @property
    def warmup_bars(self) -> int:
        return self.period + 1

    @property
    def params(self) -> dict[str, Any]:
        return {"period": self.period, "oversold": self.oversold, "overbought": self.overbought}

    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        return rsi(candles["close"], self.period).to_frame("rsi")

    def _decide(self, row: Mapping[str, Any]) -> tuple[Direction, float, dict[str, Any]]:
        value = float(row["rsi"])
        meta = {"rsi": finite_or_none(value)}
        if math.isnan(value):
            return Direction.HOLD, 0.0, meta
        if value < self.oversold:
            strength = (self.oversold - value) / self.oversold
            return Direction.BUY, scaled_confidence(strength), meta
        if value > self.overbought:
            strength = (value - self.overbought) / (100.0 - self.overbought)
            return Direction.SELL, scaled_confidence(strength), meta
        return Direction.HOLD, 0.0, meta
