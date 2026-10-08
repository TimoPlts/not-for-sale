"""Public OHLCV market data through CCXT.

SAFETY: the CCXT client is created **without credentials**, and only the
public ``fetch_ohlcv`` endpoint is ever called. The constructor refuses any
client that has credentials configured.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

import ccxt
import pandas as pd

from trading_lab.core.errors import DataError
from trading_lab.core.symbols import SUPPORTED_SYMBOLS, timeframe_to_seconds
from trading_lab.data.base import (
    OHLCV_COLUMNS,
    MarketDataProvider,
    empty_ohlcv,
    normalize_ohlcv,
    to_utc_timestamp,
)

# CCXT attributes that hold credentials; all must be empty.
_CREDENTIAL_ATTRS = ("apiKey", "secret", "password", "uid", "login", "privateKey", "walletAddress", "token")
_MAX_PAGES = 100_000


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CcxtPublicProvider(MarketDataProvider):
    """Paginated, rate-limited, retrying reader of public candles.

    Only closed candles are returned: a candle is closed once
    ``open_time + timeframe <= now``.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        *,
        client: Any | None = None,
        page_limit: int = 1000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        clock: Callable[[], datetime] = _utcnow,
        sleep: Callable[[float], None] = time.sleep,
        funding_exchange: str = "",
    ) -> None:
        if page_limit < 1:
            raise ValueError("page_limit must be >= 1")
        self._exchange_id = exchange_id
        self._client = client if client is not None else self._build_public_client(exchange_id)
        self._assert_public_only(self._client)
        self._page_limit = page_limit
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._clock = clock
        self._sleep = sleep
        self._funding_exchange = funding_exchange
        self._feeds: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._exchange_id

    def context_feed(self, kind: str, timeframe: str) -> Any:
        """Real funding rates (this exchange's perpetual futures market, public) and the Fear & Greed index."""
        from trading_lab.data.context import DERIVATIVES, FUNDING, SENTIMENT, CcxtFundingFeed, FearGreedFeed

        if kind not in self._feeds:
            if kind == FUNDING:
                exchange = self._funding_exchange or DERIVATIVES.get(self._exchange_id, self._exchange_id)
                self._feeds[kind] = CcxtFundingFeed(exchange, sleep=self._sleep)
            elif kind == SENTIMENT:
                self._feeds[kind] = FearGreedFeed()
            else:
                return None
        return self._feeds[kind]

    @staticmethod
    def _build_public_client(exchange_id: str) -> Any:
        if exchange_id not in ccxt.exchanges:
            raise DataError(f"unknown CCXT exchange id {exchange_id!r}")
        exchange_cls = getattr(ccxt, exchange_id)
        # Deliberately no credentials: public market data only.
        return exchange_cls({"enableRateLimit": True})

    @staticmethod
    def _assert_public_only(client: Any) -> None:
        configured = [attr for attr in _CREDENTIAL_ATTRS if getattr(client, attr, None)]
        if configured:
            raise DataError(
                f"refusing to use an exchange client with credentials set ({configured}); "
                "trading-lab only uses public market data"
            )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
    ) -> pd.DataFrame:
        if symbol not in SUPPORTED_SYMBOLS:
            raise DataError(f"unsupported symbol {symbol!r}; supported: {list(SUPPORTED_SYMBOLS)}")
        tf_ms = timeframe_to_seconds(timeframe) * 1000
        start = to_utc_timestamp(since)
        now = to_utc_timestamp(self._clock())
        end = now if until is None else min(to_utc_timestamp(until), now)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        rows: list[list[Any]] = []
        cursor = start_ms
        for _ in range(_MAX_PAGES):
            if cursor >= end_ms:
                break
            batch = self._fetch_page(symbol, timeframe, cursor)
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][0]) + tf_ms
            if next_cursor <= cursor:  # exchange ignored `since`; avoid looping forever
                break
            cursor = next_cursor
        else:
            raise DataError(f"pagination did not terminate for {symbol} {timeframe}")

        frame = self._to_frame(rows)
        if frame.empty:
            return frame
        open_ms = frame.index.as_unit("ms").asi8
        closed = open_ms + tf_ms <= int(now.timestamp() * 1000)
        in_range = (open_ms >= start_ms) & (open_ms < end_ms)
        return frame.loc[closed & in_range]

    def current_open(self, symbol: str, timeframe: str, bar_open: datetime) -> float | None:
        bar_ms = int(to_utc_timestamp(bar_open).timestamp() * 1000)
        if bar_ms > int(to_utc_timestamp(self._clock()).timestamp() * 1000):
            return None  # the candle has not started yet
        for row in self._fetch_page(symbol, timeframe, bar_ms, limit=1):
            if int(row[0]) == bar_ms and row[1] is not None:
                return float(row[1])
        return None

    def _fetch_page(
        self, symbol: str, timeframe: str, since_ms: int, limit: int | None = None
    ) -> list[list[Any]]:
        for attempt in range(self._max_retries + 1):
            try:
                return self._client.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since_ms, limit=limit or self._page_limit
                )
            except ccxt.NetworkError as exc:  # timeouts, rate limits, exchange unavailable
                if attempt >= self._max_retries:
                    raise DataError(
                        f"network error fetching {symbol} {timeframe} after "
                        f"{self._max_retries + 1} attempts: {exc}"
                    ) from exc
                self._sleep(self._retry_delay * (2**attempt))
            except ccxt.BaseError as exc:
                raise DataError(f"exchange error fetching {symbol} {timeframe}: {exc}") from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _to_frame(rows: list[list[Any]]) -> pd.DataFrame:
        if not rows:
            return empty_ohlcv()
        raw = pd.DataFrame([r[:6] for r in rows], columns=["timestamp", *OHLCV_COLUMNS])
        raw["volume"] = raw["volume"].fillna(0.0)  # some exchanges omit volume
        raw.index = pd.to_datetime(raw.pop("timestamp").astype("int64"), unit="ms", utc=True)
        return normalize_ohlcv(raw)
