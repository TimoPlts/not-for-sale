"""Market context beyond prices: futures funding rates and market sentiment.

These feed the ``funding`` (positioning, the desk's "whale" seat) and
``sentiment`` (the "shill" seat) strategies. A ``ContextFeed`` returns values
indexed by the moment each one **became known**, so a strategy deciding at a
bar's close only ever sees values published by then (no look-ahead):

* **funding**: the perpetual futures funding rate per funding period
  (usually 8 hours). Positive means longs pay shorts (crowded longs). It is
  known at its funding time.
* **sentiment**: the Crypto Fear & Greed index (0 = extreme fear,
  100 = extreme greed), one value per day. To be safe it counts as known one
  day after the day it describes.

Real feeds read public data only:

* funding comes through CCXT from the futures market of the configured
  exchange (``binance`` -> ``binanceusdm``), with a client built without
  credentials like the candle provider; only ``fetch_funding_rate_history``
  is called;
* the index is one public JSON request to ``api.alternative.me``.

Synthetic feeds, derived from a ``SyntheticProvider``'s prices, keep offline
backtests and tests working: funding rises after rallies (crowded longs) and
sentiment follows the 30-day trend. Both are deterministic.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from trading_lab.core.errors import DataError
from trading_lab.data.base import to_utc_timestamp

FUNDING = "funding"
SENTIMENT = "sentiment"
KINDS = (FUNDING, SENTIMENT)

# Spot exchange id -> the CCXT id of its USD-margined perpetual futures market.
DERIVATIVES = {"binance": "binanceusdm", "binanceus": "binanceusdm", "bybit": "bybit", "okx": "okx",
               "kucoin": "kucoinfutures", "kraken": "krakenfutures", "gate": "gate", "bitget": "bitget",
               "mexc": "mexc"}
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=0&format=json"
SENTIMENT_LAG = pd.Timedelta(days=1)


class FeedError(DataError):
    """A context feed could not deliver data."""


class ContextFeed(ABC):
    kind: str

    @property
    @abstractmethod
    def name(self) -> str:
        """Where the data comes from."""

    @abstractmethod
    def series(self, symbol: str, since: datetime, until: datetime) -> pd.Series:
        """Values indexed by the UTC time each became known, within ``[since, until]``."""


def _window(values: pd.Series, since: datetime, until: datetime) -> pd.Series:
    start, end = to_utc_timestamp(since), to_utc_timestamp(until)
    return values[(values.index >= start) & (values.index <= end)]


def _noise(period: np.ndarray, code: int) -> np.ndarray:
    """Deterministic pseudo-random numbers in [-1, 1) for integer periods (the same period, the same value)."""
    x = np.sin(period.astype("float64") * 12.9898 + code * 78.233) * 43758.5453
    return (x - np.floor(x)) * 2.0 - 1.0


def _code(text: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % 10_007


def _closes_by_close_time(provider: Any, symbol: str, timeframe: str, since: pd.Timestamp,
                          until: pd.Timestamp) -> pd.Series:
    """Closes indexed by the time each candle closed (open time + timeframe): what was known when."""
    from trading_lab.data.base import timeframe_delta

    step = pd.Timedelta(timeframe_delta(timeframe))
    candles = provider.fetch_ohlcv(symbol, timeframe, since.to_pydatetime(), until.to_pydatetime())
    closes = candles["close"].copy()
    closes.index = closes.index + step
    return closes[closes.index <= until]


class SyntheticFundingFeed(ContextFeed):
    """Funding every 8 hours from the synthetic prices: higher after rallies, lower after sell-offs.

    Each rate uses only candles closed by its funding time, in the run's timeframe.
    """

    kind = FUNDING

    def __init__(self, provider: Any, timeframe: str = "1h") -> None:
        self._provider, self._timeframe = provider, timeframe

    @property
    def name(self) -> str:
        return f"{self._provider.name}-funding"

    def series(self, symbol: str, since: datetime, until: datetime) -> pd.Series:
        start = to_utc_timestamp(since).floor("8h")
        end = to_utc_timestamp(until)
        times = pd.date_range(start, end, freq="8h", tz="UTC")
        if not len(times):
            return pd.Series(dtype="float64", name=FUNDING)
        closes = _closes_by_close_time(self._provider, symbol, self._timeframe, start - pd.Timedelta(days=3), end)
        if closes.empty:
            return pd.Series(dtype="float64", name=FUNDING)
        now = closes.reindex(times, method="ffill").to_numpy()
        before = closes.reindex(times - pd.Timedelta(hours=24), method="ffill").to_numpy()
        ret = np.nan_to_num(now / before - 1.0)
        periods = (times.as_unit("ns").asi8 // pd.Timedelta(hours=8).value).astype("int64")
        rate = 0.0001 + 0.004 * ret + 0.00008 * _noise(periods, _code(symbol))
        return _window(pd.Series(np.clip(rate, -0.003, 0.003), index=times, name=FUNDING), since, until)


class SyntheticSentimentFeed(ContextFeed):
    """A daily 0-100 index following the 30-day trend of BTC/USDT in the synthetic prices.

    The value for a day uses only candles closed by the end of that day, and counts as known a day later.
    """

    kind = SENTIMENT

    def __init__(self, provider: Any, timeframe: str = "1h", reference: str = "BTC/USDT") -> None:
        self._provider, self._timeframe, self._reference = provider, timeframe, reference

    @property
    def name(self) -> str:
        return f"{self._provider.name}-sentiment"

    def series(self, symbol: str, since: datetime, until: datetime) -> pd.Series:
        first = (to_utc_timestamp(since) - SENTIMENT_LAG).floor("D")
        end = to_utc_timestamp(until)
        days = pd.date_range(first, (end - SENTIMENT_LAG).floor("D"), freq="D", tz="UTC")
        if not len(days):
            return pd.Series(dtype="float64", name=SENTIMENT)
        day_ends = days + pd.Timedelta(days=1)
        closes = _closes_by_close_time(self._provider, self._reference, self._timeframe,
                                       first - pd.Timedelta(days=35), day_ends[-1])
        if closes.empty:
            return pd.Series(dtype="float64", name=SENTIMENT)
        now = closes.reindex(day_ends, method="ffill").to_numpy()
        before = closes.reindex(day_ends - pd.Timedelta(days=30), method="ffill").to_numpy()
        ret = np.nan_to_num(now / before - 1.0)
        value = 50 + 45 * np.tanh(ret / 0.15) + 8 * _noise(days.as_unit("ns").asi8 // 86_400_000_000_000, _code("fng"))
        known = days + SENTIMENT_LAG
        return _window(pd.Series(np.clip(np.round(value), 0, 100), index=known, name=SENTIMENT), since, until)


def perpetual_symbol(symbol: str) -> str:
    """``BTC/USDT`` -> ``BTC/USDT:USDT`` (CCXT's unified linear perpetual symbol)."""
    return symbol if ":" in symbol else f"{symbol}:{symbol.split('/')[1]}"


class CcxtFundingFeed(ContextFeed):
    """Public funding-rate history through CCXT (no credentials; read-only)."""

    kind = FUNDING

    def __init__(self, exchange_id: str, *, client: Any | None = None, page_limit: int = 1000,
                 max_retries: int = 3, retry_delay: float = 1.0, sleep: Callable[[float], None] = time.sleep) -> None:
        from trading_lab.data.ccxt_provider import CcxtPublicProvider

        self._exchange_id = exchange_id
        self._client = client if client is not None else CcxtPublicProvider._build_public_client(exchange_id)
        CcxtPublicProvider._assert_public_only(self._client)
        self._page_limit = page_limit
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._sleep = sleep
        self._memo: dict[str, pd.Series] = {}

    @property
    def name(self) -> str:
        return f"{self._exchange_id}-funding"

    def _page(self, symbol: str, since_ms: int) -> list[dict[str, Any]]:
        for attempt in range(self._max_retries + 1):
            try:
                return list(self._client.fetch_funding_rate_history(symbol, since=since_ms, limit=self._page_limit))
            except Exception as exc:  # network or exchange errors: retry, then give up
                if attempt == self._max_retries:
                    raise FeedError(f"funding rates for {symbol} from {self._exchange_id}: {exc}") from None
                self._sleep(self._retry_delay * (2 ** attempt))
        return []

    def _fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        perp = perpetual_symbol(symbol)
        rows: dict[int, float] = {}
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        for _ in range(10_000):
            batch = [r for r in self._page(perp, cursor) if r.get("timestamp") is not None]
            if not batch:
                break
            for r in batch:
                if r.get("fundingRate") is not None:
                    rows[int(r["timestamp"])] = float(r["fundingRate"])
            last = max(int(r["timestamp"]) for r in batch)
            if last <= cursor or last >= end_ms:
                break
            cursor = last + 1
        index = pd.to_datetime(sorted(rows), unit="ms", utc=True)
        return pd.Series([rows[k] for k in sorted(rows)], index=index, name=FUNDING, dtype="float64")

    def series(self, symbol: str, since: datetime, until: datetime) -> pd.Series:
        start, end = to_utc_timestamp(since), to_utc_timestamp(until)
        cached = self._memo.get(symbol)
        if cached is None or cached.empty or start < cached.index[0] - pd.Timedelta(hours=8):
            cached = self._fetch(symbol, start, end)
        elif end > cached.index[-1] + pd.Timedelta(hours=8):  # only the new tail
            tail = self._fetch(symbol, cached.index[-1] + pd.Timedelta(milliseconds=1), end)
            cached = pd.concat([cached, tail[tail.index > cached.index[-1]]])
        self._memo[symbol] = cached
        return _window(cached, since, until)


def _http_json(url: str, timeout: float = 15.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "trading-lab (paper trading research)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # public data, no credentials
        return json.loads(response.read().decode("utf-8"))


class FearGreedFeed(ContextFeed):
    """The Crypto Fear & Greed index from alternative.me (public; one request covers the whole history)."""

    kind = SENTIMENT

    def __init__(self, *, fetch: Callable[[str], Any] = _http_json, url: str = FEAR_GREED_URL,
                 refresh: timedelta = timedelta(hours=1), clock: Callable[[], datetime] | None = None) -> None:
        self._fetch, self._url, self._refresh = fetch, url, refresh
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._values: pd.Series | None = None
        self._fetched_at: datetime | None = None

    @property
    def name(self) -> str:
        return "alternative.me-fear-greed"

    def _load(self) -> pd.Series:
        try:
            payload = self._fetch(self._url)
            rows = {int(item["timestamp"]): float(item["value"]) for item in payload["data"]}
        except Exception as exc:
            raise FeedError(f"Fear & Greed index: {exc}") from None
        days = pd.to_datetime(sorted(rows), unit="s", utc=True)
        return pd.Series([rows[k] for k in sorted(rows)], index=days + SENTIMENT_LAG, name=SENTIMENT)

    def series(self, symbol: str, since: datetime, until: datetime) -> pd.Series:
        now = self._clock()
        stale = (self._values is None or self._fetched_at is None
                 or (now - self._fetched_at >= self._refresh and to_utc_timestamp(until) > self._values.index[-1]))
        if stale:
            self._values, self._fetched_at = self._load(), now
        assert self._values is not None
        return _window(self._values, since, until)


class CachedContextFeed(ContextFeed):
    """A feed's values kept in ``<cache_dir>/<feed>/<symbol>.csv``, fetching only what is missing.

    * Requests already covered by the file are served from it; otherwise only
      the missing tail (or the whole range, if it starts before the file) is
      fetched and merged in.
    * Within one process a range is not asked for again, so a value that is
      not published yet does not cause a request on every bar.
    * If the source fails, what the file has is served instead (the strategies'
      ``max_age_hours`` still turns old data into HOLD); with no file, the
      error is raised.

    Sentiment is the same for every symbol and is stored once.
    """

    def __init__(self, inner: ContextFeed, cache_dir: str | Any, *,
                 clock: Callable[[], datetime] | None = None) -> None:
        from pathlib import Path

        self._inner = inner
        self.kind = inner.kind
        self._dir = Path(cache_dir) / inner.name
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._asked: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        self._slack = pd.Timedelta(hours=8) if self.kind == FUNDING else pd.Timedelta(days=1) + SENTIMENT_LAG

    @property
    def name(self) -> str:
        return self._inner.name

    def path(self, symbol: str) -> Any:
        key = "all" if self.kind == SENTIMENT else symbol.replace("/", "-")
        return self._dir / f"{key}.csv"

    def _load(self, symbol: str) -> pd.Series:
        path = self.path(symbol)
        if not path.exists():
            return pd.Series(dtype="float64", name=self.kind, index=pd.DatetimeIndex([], tz="UTC"))
        frame = pd.read_csv(path, index_col="known_at", float_precision="round_trip")
        frame.index = pd.to_datetime(frame.index, utc=True)
        return frame[self.kind].astype("float64").sort_index()

    def _save(self, symbol: str, values: pd.Series) -> None:
        path = self.path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".csv.tmp")
        values.rename(self.kind).rename_axis("known_at").to_frame().to_csv(
            tmp, date_format="%Y-%m-%dT%H:%M:%S%z", float_format="%.17g")  # exact round trip
        tmp.replace(path)

    def series(self, symbol: str, since: datetime, until: datetime) -> pd.Series:
        start, end = to_utc_timestamp(since), to_utc_timestamp(until)
        key = "all" if self.kind == SENTIMENT else symbol
        cached = self._load(symbol)
        asked = self._asked.get(key)
        if asked is not None and asked[0] <= start and end <= asked[1]:
            return _window(cached, since, until)
        has_start = not cached.empty and cached.index[0] <= start + self._slack
        has_end = not cached.empty and cached.index[-1] >= min(end, to_utc_timestamp(self._clock())) - self._slack
        if not (has_start and has_end):
            fetch_from = cached.index[-1] if has_start else start
            try:
                fresh = self._inner.series(symbol, fetch_from.to_pydatetime(), until)
            except FeedError:
                if cached.empty:
                    raise
                return _window(cached, since, until)  # the source is down: serve what is stored
            if not fresh.empty:
                merged = pd.concat([cached, fresh])
                cached = merged[~merged.index.duplicated(keep="last")].sort_index()
                self._save(symbol, cached)
        lo, hi = asked if asked is not None else (start, end)
        self._asked[key] = (min(lo, start), max(hi, end))
        return _window(cached, since, until)


def value_asof(series: pd.Series, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """For each time, the latest value known at it and when it became known (NaN / NaT if none)."""
    if series.empty:
        return np.full(len(times), math.nan), np.full(len(times), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    known = pd.DatetimeIndex(series.index).as_unit("ns").asi8  # one unit on both sides (us vs ns)
    pos = np.searchsorted(known, pd.DatetimeIndex(times).as_unit("ns").asi8, side="right") - 1
    values = np.where(pos >= 0, series.to_numpy()[np.clip(pos, 0, None)], math.nan)
    stamps = np.where(pos >= 0, known[np.clip(pos, 0, None)], np.iinfo("int64").min).astype("datetime64[ns]")
    return values, stamps
