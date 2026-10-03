"""Strategy interface.

A strategy is a pure function of the candles it is given. It sees no portfolio
state and returns one standardized ``Signal`` for the **last** candle. Callers
pass only candles up to the decision bar, so a strategy cannot look ahead.

Confidence convention for the built-in strategies: a BUY or SELL starts at
0.5 when its trigger condition is just met and rises toward 1.0 as the
condition becomes more extreme. HOLD always has confidence 0.

To add a strategy (or an AI agent acting as one), subclass ``Strategy``,
implement ``warmup_bars`` and ``_evaluate``, and decorate the class with
``@register_strategy`` so it can be enabled from the config.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import pandas as pd

from trading_lab.core.models import Direction, Signal


def scaled_confidence(strength: float) -> float:
    """Map trigger strength (0 = barely triggered, >= 1 = extreme) to [0.5, 1.0]."""
    if not math.isfinite(strength):
        return 0.5
    return 0.5 + 0.5 * min(max(strength, 0.0), 1.0)


class Strategy(ABC):
    name: ClassVar[str]

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        """Minimum number of candles needed before a non-HOLD signal is possible."""

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Parameters, recorded in every signal's metadata for reproducibility."""

    @abstractmethod
    def _evaluate(self, candles: pd.DataFrame) -> tuple[Direction, float, dict[str, Any]]:
        """Return ``(direction, confidence, metadata)`` for the last candle."""

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        if candles.empty:
            raise ValueError(f"{self.name}: no candles supplied for {symbol}")
        timestamp = candles.index[-1].to_pydatetime()
        base_meta: dict[str, Any] = {"params": self.params, "bars": len(candles)}
        if len(candles) < self.warmup_bars:
            return Signal.hold(
                self.name,
                symbol,
                timestamp,
                {**base_meta, "reason": "warmup", "required_bars": self.warmup_bars},
            )
        direction, confidence, metadata = self._evaluate(candles)
        if direction is Direction.HOLD:
            confidence = 0.0
        return Signal(self.name, symbol, direction, confidence, timestamp, {**base_meta, **metadata})


def finite_or_none(value: float) -> float | None:
    """JSON-friendly float: NaN/inf become None."""
    value = float(value)
    return value if math.isfinite(value) else None
