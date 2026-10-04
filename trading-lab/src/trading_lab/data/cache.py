"""On-disk CSV cache wrapped around any ``MarketDataProvider``.

Only closed candles are ever cached, and closed candles never change, so the
cache can be extended incrementally. Cached data makes backtests reproducible
and offline.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from trading_lab.data.base import (
    MarketDataProvider,
    empty_ohlcv,
    find_gaps,
    normalize_ohlcv,
    slice_ohlcv,
    timeframe_delta,
    to_utc_timestamp,
)


class CachedProvider(MarketDataProvider):
    """Serves candles from ``<cache_dir>/<exchange>/<BASE-QUOTE>_<tf>.csv``.

    It fetches from the wrapped provider only what is missing: either the
    tail after the last cached candle, or the whole requested range when the
    cache does not cover its start or has gaps inside it.
    """

    def __init__(self, inner: MarketDataProvider, cache_dir: str | Path) -> None:
        self._inner = inner
        self._cache_dir = Path(cache_dir)

    @property
    def name(self) -> str:
        return self._inner.name

    def current_open(self, symbol: str, timeframe: str, bar_open: datetime) -> float | None:
        return self._inner.current_open(symbol, timeframe, bar_open)

    def cache_path(self, symbol: str, timeframe: str) -> Path:
        return self._cache_dir / self.name / f"{symbol.replace('/', '-')}_{timeframe}.csv"

    def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return empty_ohlcv()
        frame = pd.read_csv(path, index_col="timestamp")
        frame.index = pd.to_datetime(frame.index, utc=True)
        return normalize_ohlcv(frame)

    def _save(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".csv.tmp")
        frame.to_csv(tmp, date_format="%Y-%m-%dT%H:%M:%S%z")
        tmp.replace(path)  # atomic on the same filesystem

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        path = self.cache_path(symbol, timeframe)
        cached = self._load(path)
        start = to_utc_timestamp(since)
        step = timeframe_delta(timeframe)

        covered = slice_ohlcv(cached, start, until)
        starts_covered = not cached.empty and cached.index[0] <= start
        if starts_covered and not find_gaps(covered, timeframe):
            fetch_from = max(start, cached.index[-1] + step)
        else:
            fetch_from = start

        needs_fetch = until is None or fetch_from < to_utc_timestamp(until)
        if needs_fetch:
            fresh = self._inner.fetch_ohlcv(symbol, timeframe, fetch_from.to_pydatetime(), until)
            if not fresh.empty:
                merged = fresh if cached.empty else normalize_ohlcv(pd.concat([cached, fresh]))
                self._save(path, merged)
                cached = merged
        return slice_ohlcv(cached, start, until)
