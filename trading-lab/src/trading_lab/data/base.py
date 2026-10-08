"""Market data interface and the canonical OHLCV frame format.

Canonical OHLCV frame:
  * index: ``DatetimeIndex`` named ``timestamp``, UTC, the candle **open** time,
    strictly increasing and unique
  * columns: ``open, high, low, close, volume`` as float64
  * only **closed** candles (a candle still forming must never be returned)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from typing import Any

import numpy as np
import pandas as pd

from trading_lab.core.errors import DataError
from trading_lab.core.symbols import timeframe_to_seconds

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class MarketDataProvider(ABC):
    """Source of historical, closed OHLCV candles. Implementations must be read-only."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier of the data source (e.g. the CCXT exchange id)."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        """Closed candles with open time in ``[since, until)`` (``until=None`` means now)."""

    def current_open(self, symbol: str, timeframe: str, bar_open: datetime) -> float | None:
        """Open price of the candle that starts at ``bar_open``, once that candle has started.

        A candle's open never changes after it starts, so live paper trading
        can fill orders scheduled for this bar right away. Returns None when
        the source cannot provide it; callers then wait for the bar to close.
        """
        return None

    def context_feed(self, kind: str, timeframe: str) -> Any:
        """A ``data.context.ContextFeed`` of ``kind`` ("funding" or "sentiment") for this source, or None."""
        return None


def empty_ohlcv() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp").as_unit("ns")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS}, index=index)


def to_utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware, got naive {value!r}")
    return ts.tz_convert("UTC")


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a candle frame and return it in canonical form.

    Sorts by time and drops duplicate timestamps (keeping the last). Raises
    ``DataError`` on missing columns, NaNs, non-positive prices, negative
    volume or inconsistent high/low values.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        raise DataError(f"OHLCV frame is missing columns {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataError("OHLCV frame must be indexed by a DatetimeIndex")
    if frame.index.tz is None:
        raise DataError("OHLCV index must be timezone-aware (UTC)")

    out = frame.loc[:, list(OHLCV_COLUMNS)].astype("float64")
    out.index = out.index.tz_convert("UTC").as_unit("ns")
    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()

    if out.isna().to_numpy().any():
        raise DataError("OHLCV frame contains NaN values")
    prices = out[["open", "high", "low", "close"]].to_numpy()
    if (prices <= 0).any() or not np.isfinite(prices).all():
        raise DataError("OHLCV prices must be positive and finite")
    if (out["volume"] < 0).any():
        raise DataError("OHLCV volume must be non-negative")
    body_high = out[["open", "close"]].max(axis=1)
    body_low = out[["open", "close"]].min(axis=1)
    if (out["high"] < body_high).any() or (out["low"] > body_low).any():
        raise DataError("OHLCV high/low inconsistent with open/close")
    return out


def find_gaps(frame: pd.DataFrame, timeframe: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Missing candle ranges as ``(first_missing, last_missing)`` open times."""
    if len(frame) < 2:
        return []
    step = pd.Timedelta(seconds=timeframe_to_seconds(timeframe))
    deltas = frame.index.to_series().diff().iloc[1:]
    gaps = []
    for ts, delta in deltas[deltas > step].items():
        gaps.append((ts - delta + step, ts - step))
    return gaps


def slice_ohlcv(
    frame: pd.DataFrame, since: datetime | pd.Timestamp, until: datetime | pd.Timestamp | None
) -> pd.DataFrame:
    """Rows with open time in ``[since, until)``."""
    mask = frame.index >= to_utc_timestamp(since)
    if until is not None:
        mask &= frame.index < to_utc_timestamp(until)
    return frame.loc[mask]


def timeframe_delta(timeframe: str) -> timedelta:
    return timedelta(seconds=timeframe_to_seconds(timeframe))
