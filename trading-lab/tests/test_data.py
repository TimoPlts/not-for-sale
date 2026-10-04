"""Market data tests: validation, CCXT public provider (with fake clients), cache, synthetic data."""

from datetime import datetime, timedelta, timezone

import ccxt
import numpy as np
import pandas as pd
import pytest

from trading_lab.config import AppConfig
from trading_lab.core.errors import DataError
from trading_lab.data import (
    CachedProvider,
    CcxtPublicProvider,
    MarketDataProvider,
    SyntheticProvider,
    build_provider,
    candles_from_closes,
    find_gaps,
    normalize_ohlcv,
)

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
HOUR_MS = 3_600_000
START_MS = int(START.timestamp() * 1000)


def rows(n, start_ms=START_MS, step_ms=HOUR_MS):
    return [[start_ms + i * step_ms, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0] for i in range(n)]


class FakeExchange:
    """Implements only the public endpoint; anything else would raise AttributeError."""

    def __init__(self, data, fail_times=0, ignore_since=False):
        self.data = data
        self.fail_times = fail_times
        self.ignore_since = ignore_since
        self.calls = []

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls.append((symbol, timeframe, since, limit))
        if self.fail_times:
            self.fail_times -= 1
            raise ccxt.NetworkError("temporary outage")
        if self.ignore_since:
            return self.data[-limit:]
        return [r for r in self.data if r[0] >= since][:limit]


def fixed_clock(dt):
    return lambda: dt


# ------------------------------------------------------------------ validation
def test_normalize_sorts_dedupes_and_types():
    frame = candles_from_closes([1.0, 2.0, 3.0])
    shuffled = pd.concat([frame.iloc[[2, 0, 1]], frame.iloc[[1]]])
    out = normalize_ohlcv(shuffled)
    assert list(out.index) == list(frame.index)
    assert (out.dtypes == "float64").all()
    assert str(out.index.tz) == "UTC" and out.index.name == "timestamp"


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda f: f.drop(columns="volume"), "missing columns"),
        (lambda f: f.assign(close=np.nan), "NaN"),
        (lambda f: f.assign(low=-1.0), "positive"),
        (lambda f: f.assign(high=f["low"] * 0.5), "high/low"),
        (lambda f: f.assign(volume=-1.0), "volume"),
        (lambda f: f.tz_localize(None), "timezone-aware"),
    ],
)
def test_normalize_rejects_bad_data(mutate, match):
    with pytest.raises(DataError, match=match):
        normalize_ohlcv(mutate(candles_from_closes([1.0, 2.0, 3.0])))


def test_find_gaps():
    frame = candles_from_closes(range(1, 11))
    holed = frame.drop(frame.index[[3, 4, 7]])
    gaps = find_gaps(holed, "1h")
    assert gaps == [(frame.index[3], frame.index[4]), (frame.index[7], frame.index[7])]
    assert find_gaps(frame, "1h") == []


# ---------------------------------------------------------------- CCXT provider
def test_ccxt_provider_paginates_and_returns_canonical_frame():
    client = FakeExchange(rows(25))
    provider = CcxtPublicProvider(
        "fake", client=client, page_limit=10, clock=fixed_clock(START + timedelta(days=2))
    )
    frame = provider.fetch_ohlcv("BTC/USDT", "1h", START, START + timedelta(hours=25))
    assert len(frame) == 25
    assert [c[2] for c in client.calls] == [START_MS, START_MS + 10 * HOUR_MS, START_MS + 20 * HOUR_MS]
    assert frame.index[0] == pd.Timestamp(START)
    assert frame["close"].iloc[-1] == 124.5
    assert provider.name == "fake"


def test_ccxt_provider_respects_until_and_drops_forming_candle():
    client = FakeExchange(rows(30))
    now = START + timedelta(hours=20, minutes=30)  # candle 20 is still forming
    provider = CcxtPublicProvider("fake", client=client, clock=fixed_clock(now))
    frame = provider.fetch_ohlcv("BTC/USDT", "1h", START)
    assert len(frame) == 20
    assert frame.index[-1] == pd.Timestamp(START + timedelta(hours=19))
    bounded = provider.fetch_ohlcv("BTC/USDT", "1h", START + timedelta(hours=5), START + timedelta(hours=8))
    assert list(bounded.index.hour) == [5, 6, 7]


def test_ccxt_provider_retries_network_errors_with_backoff():
    sleeps = []
    client = FakeExchange(rows(5), fail_times=2)
    provider = CcxtPublicProvider(
        "fake", client=client, clock=fixed_clock(START + timedelta(days=1)), sleep=sleeps.append
    )
    assert len(provider.fetch_ohlcv("BTC/USDT", "1h", START)) == 5
    assert sleeps == [1.0, 2.0]


def test_ccxt_provider_gives_up_after_max_retries():
    client = FakeExchange(rows(5), fail_times=10)
    provider = CcxtPublicProvider(
        "fake", client=client, max_retries=2, clock=fixed_clock(START + timedelta(days=1)),
        sleep=lambda s: None,
    )
    with pytest.raises(DataError, match="after 3 attempts"):
        provider.fetch_ohlcv("BTC/USDT", "1h", START)


def test_ccxt_provider_terminates_when_exchange_ignores_since():
    client = FakeExchange(rows(5), ignore_since=True)
    provider = CcxtPublicProvider("fake", client=client, clock=fixed_clock(START + timedelta(days=30)))
    frame = provider.fetch_ohlcv("BTC/USDT", "1h", START + timedelta(hours=2))
    assert list(frame.index.hour) == [2, 3, 4]
    assert len(client.calls) <= 2


def test_ccxt_provider_rejects_unsupported_symbol():
    provider = CcxtPublicProvider("fake", client=FakeExchange([]))
    with pytest.raises(DataError, match="unsupported symbol"):
        provider.fetch_ohlcv("XRP/USDT", "1h", START)


def test_ccxt_provider_refuses_clients_with_credentials():
    client = FakeExchange([])
    client.apiKey = "should-never-be-here"
    with pytest.raises(DataError, match="credentials"):
        CcxtPublicProvider("fake", client=client)


def test_real_ccxt_client_is_built_without_credentials():
    """Constructing the client makes no network calls."""
    provider = CcxtPublicProvider("binance")
    client = provider._client
    assert isinstance(client, ccxt.Exchange)
    assert client.enableRateLimit is True
    assert not client.apiKey and not client.secret
    with pytest.raises(DataError, match="unknown CCXT exchange"):
        CcxtPublicProvider("not_an_exchange")


def test_build_provider_from_config(tmp_path):
    cfg = AppConfig.from_mapping({"data": {"cache_dir": str(tmp_path)}})
    provider = build_provider(cfg)
    assert isinstance(provider, CachedProvider) and provider.name == "binance"
    no_cache = build_provider(AppConfig.from_mapping({"data": {"use_cache": False}}))
    assert isinstance(no_cache, CcxtPublicProvider)


# ------------------------------------------------------------------------ cache
class CountingProvider(MarketDataProvider):
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    @property
    def name(self):
        return "counting"

    def fetch_ohlcv(self, symbol, timeframe, since, until=None):
        self.calls.append((since, until))
        return self.inner.fetch_ohlcv(symbol, timeframe, since, until)


def test_cache_serves_repeat_requests_from_disk(tmp_path):
    inner = CountingProvider(SyntheticProvider(seed=1))
    cached = CachedProvider(inner, tmp_path)
    until = START + timedelta(days=3)
    first = cached.fetch_ohlcv("BTC/USDT", "1h", START, until)
    second = cached.fetch_ohlcv("BTC/USDT", "1h", START, until)
    assert len(inner.calls) == 1
    assert cached.cache_path("BTC/USDT", "1h").exists()
    pd.testing.assert_frame_equal(first, second, check_freq=False)  # exact float round-trip
    # A fresh instance (new process) reads the same data from disk.
    third = CachedProvider(inner, tmp_path).fetch_ohlcv("BTC/USDT", "1h", START, until)
    assert len(inner.calls) == 1
    pd.testing.assert_frame_equal(first, third, check_freq=False)


def test_cache_fetches_only_the_missing_tail(tmp_path):
    inner = CountingProvider(SyntheticProvider(seed=1))
    cached = CachedProvider(inner, tmp_path)
    cached.fetch_ohlcv("BTC/USDT", "1h", START, START + timedelta(days=1))
    longer = cached.fetch_ohlcv("BTC/USDT", "1h", START, START + timedelta(days=2))
    assert inner.calls[-1][0] == START + timedelta(days=1)  # only the new day was requested
    direct = SyntheticProvider(seed=1).fetch_ohlcv("BTC/USDT", "1h", START, START + timedelta(days=2))
    pd.testing.assert_frame_equal(longer, direct, check_freq=False)


def test_cache_refetches_when_start_not_covered(tmp_path):
    inner = CountingProvider(SyntheticProvider(seed=1))
    cached = CachedProvider(inner, tmp_path)
    cached.fetch_ohlcv("BTC/USDT", "1h", START + timedelta(days=1), START + timedelta(days=2))
    out = cached.fetch_ohlcv("BTC/USDT", "1h", START, START + timedelta(days=2))
    assert inner.calls[-1][0] == START
    assert len(out) == 48 and not find_gaps(out, "1h")


# -------------------------------------------------------------------- synthetic
def test_synthetic_is_deterministic_and_window_independent():
    a = SyntheticProvider(seed=5).fetch_ohlcv("SOL/USDT", "1h", START, START + timedelta(days=5))
    b = SyntheticProvider(seed=5).fetch_ohlcv(
        "SOL/USDT", "1h", START + timedelta(days=2), START + timedelta(days=3)
    )
    pd.testing.assert_frame_equal(a.loc[b.index], b, check_freq=False)
    c = SyntheticProvider(seed=6).fetch_ohlcv("SOL/USDT", "1h", START, START + timedelta(days=5))
    assert not np.allclose(a["close"], c["close"])
    assert len(a) == 120 and not find_gaps(a, "1h")


def test_synthetic_without_until_returns_closed_candles_only():
    now = START + timedelta(hours=5, minutes=20)
    provider = SyntheticProvider(seed=1, clock=lambda: now)
    frame = provider.fetch_ohlcv("BTC/USDT", "1h", START)
    assert frame.index[-1] == pd.Timestamp(START + timedelta(hours=4))  # 05:00 still forming
    # The forming candle's open is already known; a future candle's is not.
    full = provider.fetch_ohlcv("BTC/USDT", "1h", START, START + timedelta(hours=6))
    assert provider.current_open("BTC/USDT", "1h", START + timedelta(hours=5)) == full["open"].iloc[5]
    assert provider.current_open("BTC/USDT", "1h", START + timedelta(hours=6)) is None


def test_ccxt_current_open_reads_forming_candle():
    client = FakeExchange(rows(10))
    now = START + timedelta(hours=7, minutes=5)
    provider = CcxtPublicProvider("fake", client=client, clock=fixed_clock(now))
    assert provider.current_open("BTC/USDT", "1h", START + timedelta(hours=7)) == 107.0
    assert client.calls[-1][3] == 1  # a single-candle request
    assert provider.current_open("BTC/USDT", "1h", START + timedelta(hours=8)) is None


def test_candles_from_closes():
    frame = candles_from_closes([10.0, 11.0, 9.0], timeframe="4h")
    assert list(frame["open"]) == [10.0, 10.0, 11.0]
    assert (frame.index[1] - frame.index[0]) == pd.Timedelta(hours=4)
    assert (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
