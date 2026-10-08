"""Strategies on market context: futures positioning (funding) and sentiment (Fear & Greed).

These are the desk's "whale" and "shill" seats for large coins. Both are
contrarian by default, opt-in (``[strategies.funding]``,
``[strategies.sentiment]``) and fail safe: missing, stale or failing data
gives HOLD with the reason in the signal, never a trade.

* ``funding`` averages the last ``average`` funding rates known at the bar's
  close. It votes SELL when longs are crowded (average >= ``high``, e.g.
  0.05% per 8 hours) and BUY when shorts are (average <= ``low``).
  ``mode = "follow"`` reverses it.
* ``sentiment`` reads the Fear & Greed index (0-100) known at the close. It
  votes BUY in fear (<= ``fear``) and SELL in greed (>= ``greed``).
  ``mode = "follow"`` reverses it.

Confidence starts at 0.5 at the threshold and reaches 1.0 at ``extreme``
times the threshold distance. The data comes from a ``data.context.ContextFeed``
that the engine attaches (``attach_context_feeds``), from the same source as
the candles: real public data for exchanges, synthetic data for the
synthetic provider.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Sequence

import numpy as np
import pandas as pd

from trading_lab.core.models import Direction, Signal
from trading_lab.data.context import FUNDING, SENTIMENT, ContextFeed, FeedError, value_asof
from trading_lab.strategies.base import Strategy, finite_or_none, scaled_confidence
from trading_lab.strategies.registry import register_strategy
from trading_lab.strategies.validation import check_int, check_range

MODES = ("contrarian", "follow")


class ContextStrategy(Strategy):
    """A strategy voting on a context feed's value known at each bar's close."""

    feed_kind: ClassVar[str]

    def __init__(self, mode: str, max_age_hours: float) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.max_age_hours = check_range("max_age_hours", max_age_hours, 0.0, 24.0 * 365)
        self.feed: ContextFeed | None = None

    def attach_feed(self, feed: ContextFeed) -> None:
        self.feed = feed

    @property
    def warmup_bars(self) -> int:
        return 1

    @property
    def history_bars(self) -> int:
        return 1

    def _values(self, symbol: str, candles: pd.DataFrame) -> pd.Series:
        """The feed's values from a little before the first candle to the last bar's close."""
        raise NotImplementedError

    @staticmethod
    def _closes(candles: pd.DataFrame) -> pd.DatetimeIndex:
        """Each bar's close time (open + timeframe; the timeframe is read from the index)."""
        index = pd.DatetimeIndex(candles.index)
        step = index[1] - index[0] if len(index) > 1 else pd.Timedelta(hours=1)
        return index + step

    def _vote(self, value: float) -> tuple[Direction, float, dict[str, Any]]:
        raise NotImplementedError

    def _evaluate(self, candles: pd.DataFrame) -> tuple[Direction, float, dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError  # generate_signals is the only path

    def _signals(self, symbol: str, candles: pd.DataFrame, positions: list[int]) -> list[Signal]:
        """Signals for the bars at ``positions``, each from the value known at that bar's close."""
        stamps = [candles.index[i].to_pydatetime() for i in positions]
        if self.feed is None:
            return [self._make_signal(symbol, ts, i + 1, Direction.HOLD, 0.0, {"reason": "no data feed", "source": None})
                    for i, ts in zip(positions, stamps)]
        try:
            series = self._values(symbol, candles)
        except FeedError as exc:
            return [self._make_signal(symbol, ts, i + 1, Direction.HOLD, 0.0,
                                      {"reason": "data error", "error": str(exc), "source": self.feed.name})
                    for i, ts in zip(positions, stamps)]
        closes = self._closes(candles)[positions]
        values, known = value_asof(series, closes)
        max_age = np.timedelta64(int(self.max_age_hours * 3600), "s")
        signals = []
        for k, (i, ts) in enumerate(zip(positions, stamps)):
            value, at = values[k], known[k]
            meta: dict[str, Any] = {"source": self.feed.name, self.feed_kind: finite_or_none(value),
                                    "known_at": None if np.isnat(at) else pd.Timestamp(at, tz="UTC").isoformat()}
            if np.isnat(at) or not math.isfinite(value):
                signals.append(self._make_signal(symbol, ts, i + 1, Direction.HOLD, 0.0, {**meta, "reason": "no data"}))
                continue
            if closes[k].to_datetime64() - at > max_age:
                signals.append(self._make_signal(symbol, ts, i + 1, Direction.HOLD, 0.0, {**meta, "reason": "stale"}))
                continue
            direction, confidence, extra = self._vote(float(value))
            if self.mode == "follow" and direction is not Direction.HOLD:
                direction = Direction.SELL if direction is Direction.BUY else Direction.BUY
            signals.append(self._make_signal(symbol, ts, i + 1, direction, confidence, {**meta, **extra}))
        return signals

    def generate_signals(self, symbol: str, candles: pd.DataFrame) -> list[Signal]:
        return [] if candles.empty else self._signals(symbol, candles, list(range(len(candles))))

    def generate_signals_at(self, symbol: str, candles: pd.DataFrame, positions: Sequence[int]) -> list[Signal]:
        return self._signals(symbol, candles, list(positions)) if len(positions) else []

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        if candles.empty:
            raise ValueError(f"{self.name}: no candles supplied for {symbol}")
        return self._signals(symbol, candles, [len(candles) - 1])[0]


@register_strategy
class FundingStrategy(ContextStrategy):
    """Perpetual futures funding: crowded longs -> SELL, crowded shorts -> BUY (contrarian)."""

    name = "funding"
    feed_kind = FUNDING

    def __init__(self, high: float = 0.0005, low: float = -0.0001, average: int = 3, extreme: float = 3.0,
                 mode: str = "contrarian", max_age_hours: float = 24.0) -> None:
        super().__init__(mode, max_age_hours)
        self.high = check_range("high", high, -0.05, 0.05)
        self.low = check_range("low", low, -0.05, 0.05)
        if not self.low < self.high:
            raise ValueError(f"low ({low}) must be below high ({high})")
        self.average = check_int("average", average)
        self.extreme = check_range("extreme", extreme, 1.0, 100.0, inclusive=True)

    @property
    def params(self) -> dict[str, Any]:
        return {"high": self.high, "low": self.low, "average": self.average, "extreme": self.extreme,
                "mode": self.mode, "max_age_hours": self.max_age_hours}

    def _values(self, symbol: str, candles: pd.DataFrame) -> pd.Series:
        assert self.feed is not None
        closes = self._closes(candles)
        since = closes[0] - pd.Timedelta(hours=8 * (self.average + 2)) - pd.Timedelta(hours=self.max_age_hours)
        raw = self.feed.series(symbol, since.to_pydatetime(), closes[-1].to_pydatetime())
        return raw.rolling(self.average, min_periods=self.average).mean().dropna()

    def _vote(self, value: float) -> tuple[Direction, float, dict[str, Any]]:
        span = max(abs(self.high), abs(self.low), 1e-6)
        if value >= self.high:
            return Direction.SELL, scaled_confidence((value - self.high) / ((self.extreme - 1) * span or 1)), \
                {"positioning": "crowded longs"}
        if value <= self.low:
            return Direction.BUY, scaled_confidence((self.low - value) / ((self.extreme - 1) * span or 1)), \
                {"positioning": "crowded shorts"}
        return Direction.HOLD, 0.0, {"positioning": "neutral"}


@register_strategy
class SentimentStrategy(ContextStrategy):
    """Fear & Greed: extreme fear -> BUY, extreme greed -> SELL (contrarian)."""

    name = "sentiment"
    feed_kind = SENTIMENT

    def __init__(self, fear: float = 25.0, greed: float = 75.0, mode: str = "contrarian",
                 max_age_hours: float = 72.0) -> None:
        super().__init__(mode, max_age_hours)
        self.fear = check_range("fear", fear, 0.0, 100.0)
        self.greed = check_range("greed", greed, 0.0, 100.0)
        if not self.fear < self.greed:
            raise ValueError(f"fear ({fear}) must be below greed ({greed})")

    @property
    def params(self) -> dict[str, Any]:
        return {"fear": self.fear, "greed": self.greed, "mode": self.mode, "max_age_hours": self.max_age_hours}

    def _values(self, symbol: str, candles: pd.DataFrame) -> pd.Series:
        assert self.feed is not None
        closes = self._closes(candles)
        since = closes[0] - pd.Timedelta(hours=self.max_age_hours) - pd.Timedelta(days=2)
        return self.feed.series(symbol, since.to_pydatetime(), closes[-1].to_pydatetime())

    def _vote(self, value: float) -> tuple[Direction, float, dict[str, Any]]:
        if value <= self.fear:
            return Direction.BUY, scaled_confidence((self.fear - value) / max(self.fear, 1.0)), {"mood": "fear"}
        if value >= self.greed:
            return Direction.SELL, scaled_confidence((value - self.greed) / max(100.0 - self.greed, 1.0)), \
                {"mood": "greed"}
        return Direction.HOLD, 0.0, {"mood": "neutral"}
