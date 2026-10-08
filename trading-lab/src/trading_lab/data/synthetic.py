"""Deterministic, offline market data for tests and experiments.

``SyntheticProvider`` generates a seeded random walk on a fixed time grid that
starts at ``anchor``. A candle at a given timestamp is therefore identical no
matter which window is requested, the same guarantee a real exchange gives.
"""

from __future__ import annotations

import zlib
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from trading_lab.data.base import (
    MarketDataProvider,
    empty_ohlcv,
    normalize_ohlcv,
    slice_ohlcv,
    timeframe_delta,
    to_utc_timestamp,
)

DEFAULT_START_PRICES: Mapping[str, float] = {
    "BTC/USDT": 30_000.0,
    "ETH/USDT": 2_000.0,
    "SOL/USDT": 50.0,
    "DOGE/USDT": 0.1,
}
DEFAULT_ANCHOR = datetime(2020, 1, 1, tzinfo=timezone.utc)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SyntheticProvider(MarketDataProvider):
    """Seeded geometric random walk with plausible OHLCV structure.

    With ``until=None`` it behaves like a live feed: it returns candles that
    have closed by ``clock()``, which makes it usable for offline paper trading.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        volatility: float = 0.005,
        drift: float = 0.0,
        start_prices: Mapping[str, float] = DEFAULT_START_PRICES,
        anchor: datetime = DEFAULT_ANCHOR,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if volatility < 0:
            raise ValueError("volatility must be >= 0")
        self._seed = seed
        self._volatility = volatility
        self._drift = drift
        self._start_prices = dict(start_prices)
        self._anchor = to_utc_timestamp(anchor)
        self._clock = clock

    @property
    def name(self) -> str:
        return f"synthetic-{self._seed}"

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        if symbol not in self._start_prices:
            raise ValueError(f"no start price configured for {symbol}")
        step = pd.Timedelta(timeframe_delta(timeframe))
        if until is None:
            # Only closed candles: stop at the start of the candle forming now.
            now = to_utc_timestamp(self._clock())
            until = self._anchor + ((now - self._anchor) // step) * step
        end = to_utc_timestamp(until)
        if end <= self._anchor:
            return empty_ohlcv()
        n = int(np.ceil((end - self._anchor) / step))
        rng = np.random.default_rng([self._seed, zlib.crc32(f"{symbol}|{timeframe}".encode())])

        # One row of draws per candle, so candle i is independent of how many follow.
        z = rng.standard_normal((n, 4))
        returns = self._drift + self._volatility * z[:, 0]
        close = self._start_prices[symbol] * np.exp(np.cumsum(returns))
        open_ = np.concatenate(([self._start_prices[symbol]], close[:-1]))
        high = np.maximum(open_, close) * (1 + np.abs(z[:, 1]) * self._volatility * 0.5)
        low = np.minimum(open_, close) * (1 - np.abs(z[:, 2]) * self._volatility * 0.5)
        volume = np.exp(10.0 + 0.5 * z[:, 3])

        index = pd.date_range(self._anchor, periods=n, freq=step, name="timestamp")
        frame = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
        return normalize_ohlcv(slice_ohlcv(frame, since, until))

    def context_feed(self, kind: str, timeframe: str) -> Any:
        """Synthetic funding and sentiment derived from these prices (see ``data.context``)."""
        from trading_lab.data.context import FUNDING, SENTIMENT, SyntheticFundingFeed, SyntheticSentimentFeed

        if kind == FUNDING:
            return SyntheticFundingFeed(self, timeframe)
        if kind == SENTIMENT:
            return SyntheticSentimentFeed(self, timeframe)
        return None

    def current_open(self, symbol: str, timeframe: str, bar_open: datetime) -> float | None:
        start = to_utc_timestamp(bar_open)
        if start > to_utc_timestamp(self._clock()):
            return None
        step = timeframe_delta(timeframe)
        frame = self.fetch_ohlcv(symbol, timeframe, start.to_pydatetime(), (start + step).to_pydatetime())
        return float(frame["open"].iloc[0]) if not frame.empty and frame.index[0] == start else None


def candles_from_closes(
    closes: Sequence[float],
    *,
    start: datetime = DEFAULT_ANCHOR,
    timeframe: str = "1h",
    spread: float = 0.001,
) -> pd.DataFrame:
    """Build a valid OHLCV frame from a close series (open = previous close).

    Handy for crafting exact scenarios in strategy tests.
    """
    close = np.asarray(closes, dtype="float64")
    if close.ndim != 1 or close.size == 0:
        raise ValueError("closes must be a non-empty 1-D sequence")
    open_ = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    index = pd.date_range(
        to_utc_timestamp(start), periods=close.size, freq=timeframe_delta(timeframe), name="timestamp"
    )
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": np.full(close.size, 1.0)},
        index=index,
    )
    return normalize_ohlcv(frame)
