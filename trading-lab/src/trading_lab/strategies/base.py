"""Strategy interface.

A strategy is a pure function of the candles it is given. It sees no portfolio
state and returns one standardized ``Signal`` for the **last** candle. Callers
pass only candles up to the decision bar, so a strategy cannot look ahead.

There are two ways to get signals:
  * ``generate_signal(symbol, candles)`` gives the signal for the last candle
    (live trading)
  * ``generate_signals(symbol, candles)`` gives one signal per candle
    (backtesting). Element ``i`` must equal
    ``generate_signal(symbol, candles.iloc[: i + 1])``.

Confidence convention for the built-in strategies: a BUY or SELL starts at
0.5 when its trigger condition is just met and rises toward 1.0 as the
condition becomes more extreme. HOLD always has confidence 0.

To add a strategy (or an AI agent acting as one), subclass ``Strategy``,
implement ``warmup_bars``, ``params`` and ``_evaluate``, and decorate the
class with ``@register_strategy`` so it can be enabled from the config.
Indicator-based strategies should subclass ``IndicatorStrategy`` instead,
which gives them a fast vectorised ``generate_signals``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping

import pandas as pd

from trading_lab.core.models import Direction, Signal


def scaled_confidence(strength: float) -> float:
    """Map trigger strength (0 = barely triggered, >= 1 = extreme) to [0.5, 1.0]."""
    if not math.isfinite(strength):
        return 0.5
    return 0.5 + 0.5 * min(max(strength, 0.0), 1.0)


def finite_or_none(value: Any) -> float | None:
    """JSON-friendly float: NaN/inf/None become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


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

    @property
    def history_bars(self) -> int:
        """Candles of history to load before the first decision.

        Generous enough that exponentially smoothed indicators (EMA, Wilder)
        converge regardless of where the data starts.
        """
        return max(300, 5 * self.warmup_bars)

    @abstractmethod
    def _evaluate(self, candles: pd.DataFrame) -> tuple[Direction, float, dict[str, Any]]:
        """Return ``(direction, confidence, metadata)`` for the last candle."""

    # ------------------------------------------------------------------ helpers
    def _warmup_signal(self, symbol: str, candles_seen: int, timestamp: Any) -> Signal:
        return Signal.hold(
            self.name,
            symbol,
            timestamp,
            {
                "params": self.params,
                "bars": candles_seen,
                "reason": "warmup",
                "required_bars": self.warmup_bars,
            },
        )

    def _make_signal(
        self,
        symbol: str,
        timestamp: Any,
        candles_seen: int,
        direction: Direction,
        confidence: float,
        metadata: Mapping[str, Any],
    ) -> Signal:
        if direction is Direction.HOLD:
            confidence = 0.0
        meta = {"params": self.params, "bars": candles_seen, **metadata}
        return Signal(self.name, symbol, direction, confidence, timestamp, meta)

    # --------------------------------------------------------------------- API
    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        if candles.empty:
            raise ValueError(f"{self.name}: no candles supplied for {symbol}")
        timestamp = candles.index[-1].to_pydatetime()
        if len(candles) < self.warmup_bars:
            return self._warmup_signal(symbol, len(candles), timestamp)
        direction, confidence, metadata = self._evaluate(candles)
        return self._make_signal(symbol, timestamp, len(candles), direction, confidence, metadata)

    def generate_signals(self, symbol: str, candles: pd.DataFrame) -> list[Signal]:
        """One signal per candle. The default re-evaluates an expanding window
        (correct but O(n²)). ``IndicatorStrategy`` overrides it with an O(n) version."""
        return [
            self.generate_signal(symbol, candles.iloc[: i + 1]) for i in range(len(candles))
        ]


class IndicatorStrategy(Strategy):
    """A strategy defined by causal indicator columns plus a per-bar decision rule.

    ``indicators`` must be causal: row ``t`` depends only on candles ``<= t``.
    Then computing the indicators once for the whole history gives exactly the
    same per-bar decisions as recomputing them bar by bar.
    """

    @abstractmethod
    def indicators(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Indicator columns aligned to ``candles.index``."""

    @abstractmethod
    def _decide(self, row: Mapping[str, Any]) -> tuple[Direction, float, dict[str, Any]]:
        """Decision for one bar from that bar's indicator row."""

    def _evaluate(self, candles: pd.DataFrame) -> tuple[Direction, float, dict[str, Any]]:
        return self._decide(self.indicators(candles).iloc[-1].to_dict())

    def generate_signals(self, symbol: str, candles: pd.DataFrame) -> list[Signal]:
        if candles.empty:
            return []
        rows = self.indicators(candles).to_dict("records")
        signals = []
        for i, (ts, row) in enumerate(zip(candles.index, rows)):
            timestamp = ts.to_pydatetime()
            if i + 1 < self.warmup_bars:
                signals.append(self._warmup_signal(symbol, i + 1, timestamp))
                continue
            direction, confidence, metadata = self._decide(row)
            signals.append(
                self._make_signal(symbol, timestamp, i + 1, direction, confidence, metadata)
            )
        return signals
