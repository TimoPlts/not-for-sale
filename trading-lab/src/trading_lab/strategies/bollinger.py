"""Bollinger Band mean-reversion strategy."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from trading_lab.core.models import Direction
from trading_lab.indicators import bollinger_bands
from trading_lab.strategies.base import Strategy, finite_or_none, scaled_confidence
from trading_lab.strategies.registry import register_strategy
from trading_lab.strategies.validation import check_int, check_range


@register_strategy
class BollingerMeanReversionStrategy(Strategy):
    """Mean reversion using %B, the close's position within the bands
    (0 = lower band, 0.5 = middle, 1 = upper).

    * BUY while the close is below the lower band (``%B < 0``). Confidence
      reaches 1.0 half a band-width below the lower band.
    * SELL while ``%B >= exit_percent_b``. The default 1.0 means the upper
      band; 0.5 exits at the middle band. Confidence reaches 1.0 half a
      band-width beyond the exit level.
    * HOLD otherwise, including when the bands have zero width.
    """

    name = "bollinger"

    def __init__(self, period: int = 20, num_std: float = 2.0, exit_percent_b: float = 1.0) -> None:
        self.period = check_int("period", period, minimum=2)
        self.num_std = check_range("num_std", num_std, 0.0, 10.0)
        self.exit_percent_b = check_range("exit_percent_b", exit_percent_b, 0.0, 2.0)

    @property
    def warmup_bars(self) -> int:
        return self.period

    @property
    def params(self) -> dict[str, Any]:
        return {"period": self.period, "num_std": self.num_std, "exit_percent_b": self.exit_percent_b}

    def _evaluate(self, candles: pd.DataFrame) -> tuple[Direction, float, dict[str, Any]]:
        close = candles["close"]
        bands = bollinger_bands(close, self.period, self.num_std).iloc[-1]
        percent_b = float(bands["percent_b"])
        meta = {
            "close": float(close.iloc[-1]),
            "upper": finite_or_none(bands["upper"]),
            "middle": finite_or_none(bands["middle"]),
            "lower": finite_or_none(bands["lower"]),
            "percent_b": finite_or_none(percent_b),
        }
        if math.isnan(percent_b):
            return Direction.HOLD, 0.0, meta
        if percent_b < 0.0:
            return Direction.BUY, scaled_confidence(-percent_b / 0.5), meta
        if percent_b >= self.exit_percent_b:
            strength = (percent_b - self.exit_percent_b) / 0.5
            return Direction.SELL, scaled_confidence(strength), meta
        return Direction.HOLD, 0.0, meta
