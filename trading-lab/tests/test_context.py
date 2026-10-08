"""Stage 29B: market context (funding rates, Fear & Greed) and the funding/sentiment strategies."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from test_live import ANCHOR, START, Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError, DataError
from trading_lab.core.models import Direction
from trading_lab.data import CachedProvider, CcxtPublicProvider, SyntheticProvider, candles_from_closes
from trading_lab.data.base import MarketDataProvider
from trading_lab.data.context import (
    SENTIMENT_LAG,
    CcxtFundingFeed,
    ContextFeed,
    FearGreedFeed,
    FeedError,
    SyntheticFundingFeed,
    SyntheticSentimentFeed,
    perpetual_symbol,
    value_asof,
)
from trading_lab.live import LivePaperTrader
from trading_lab.research.sweep import MemoizedProvider
from trading_lab.storage import SQLiteStore
from trading_lab.strategies.context import FundingStrategy, SentimentStrategy

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 2, 1, tzinfo=UTC)


class FakeFeed(ContextFeed):
    def __init__(self, kind, values, fail=False):
        self.kind, self._values, self._fail, self.calls = kind, values, fail, 0

    @property
    def name(self):
        return "fake"

    def series(self, symbol, since, until):
        self.calls += 1
        if self._fail:
            raise FeedError("down")
        s = self._values
        return s[(s.index >= pd.Timestamp(since)) & (s.index <= pd.Timestamp(until))]


def candles(n=48, start=T0):
    return candles_from_closes(np.linspace(100, 110, n), start=start)


# ------------------------------------------------------------------ helpers
def test_value_asof_by_hand_and_across_units():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(
        ["2024-02-01 00:00", "2024-02-01 08:00", "2024-02-01 16:00"], tz="UTC").as_unit("us"))
    times = pd.DatetimeIndex(["2024-01-31 23:00", "2024-02-01 00:00", "2024-02-01 07:59", "2024-02-02 00:00"],
                             tz="UTC").as_unit("ns")
    values, known = value_asof(s, times)
    assert np.isnan(values[0]) and np.isnat(known[0])
    assert list(values[1:]) == [1.0, 1.0, 3.0]  # a value counts from the moment it is known
    assert pd.Timestamp(known[3]) == pd.Timestamp("2024-02-01 16:00")
    empty_v, empty_k = value_asof(pd.Series(dtype="float64"), times)
    assert np.isnan(empty_v).all() and np.isnat(empty_k).all()
    assert perpetual_symbol("BTC/USDT") == "BTC/USDT:USDT" and perpetual_symbol("ETH/USDT:USDT") == "ETH/USDT:USDT"


# ----------------------------------------------------------- synthetic feeds
class Tampered(MarketDataProvider):
    """The same synthetic prices, but every candle closing after ``cut`` is doubled."""

    def __init__(self, inner, cut):
        self._inner, self._cut = inner, pd.Timestamp(cut)

    @property
    def name(self):
        return self._inner.name

    def fetch_ohlcv(self, symbol, timeframe, since, until=None):
        frame = self._inner.fetch_ohlcv(symbol, timeframe, since, until).copy()
        step = frame.index[1] - frame.index[0] if len(frame) > 1 else pd.Timedelta(hours=1)
        late = frame.index + step > self._cut
        frame.loc[late, ["open", "high", "low", "close"]] *= 2
        return frame


@pytest.mark.parametrize("feed_cls", [SyntheticFundingFeed, SyntheticSentimentFeed])
def test_synthetic_feeds_are_deterministic_and_causal(feed_cls):
    provider = SyntheticProvider(seed=3)
    feed = feed_cls(provider, "1h")
    long = feed.series("BTC/USDT", T0, T0 + timedelta(days=30))
    short = feed.series("BTC/USDT", T0 + timedelta(days=5), T0 + timedelta(days=12))
    assert len(short) > 3 and long.loc[short.index].tolist() == short.tolist()  # the window does not matter
    assert feed_cls(SyntheticProvider(seed=3), "1h").series("BTC/USDT", T0, T0 + timedelta(days=30)).equals(long)
    cut = T0 + timedelta(days=15, hours=3)
    tampered = feed_cls(Tampered(provider, cut), "1h").series("BTC/USDT", T0, T0 + timedelta(days=30))
    known = long.index <= cut
    assert tampered[known].tolist() == long[known].tolist()  # nothing after a value's time changes it
    assert tampered[~known].tolist() != long[~known].tolist()


def test_synthetic_values_are_plausible():
    provider = SyntheticProvider(seed=5, volatility=0.01)
    funding = SyntheticFundingFeed(provider, "1h").series("ETH/USDT", T0, T0 + timedelta(days=40))
    assert (funding.index.hour % 8 == 0).all() and funding.abs().max() <= 0.003
    closes = provider.fetch_ohlcv("ETH/USDT", "1h", T0 - timedelta(days=2), T0 + timedelta(days=41))["close"]
    closes.index = closes.index + H
    ret = (closes.reindex(funding.index, method="ffill").to_numpy()
           / closes.reindex(funding.index - 24 * H, method="ffill").to_numpy() - 1)
    assert np.corrcoef(ret, funding.to_numpy())[0, 1] > 0.9  # crowded longs after rallies
    sentiment = SyntheticSentimentFeed(provider, "4h").series("SOL/USDT", T0, T0 + timedelta(days=60))
    assert sentiment.between(0, 100).all() and (sentiment.index.hour == 0).all()


# --------------------------------------------------------------- strategies
FUNDING = pd.Series([0.0001, 0.0006, 0.0007, 0.0008, -0.0002, -0.0003, -0.0004, 0.0001],
                    index=pd.date_range(T0 - 16 * H, periods=8, freq="8h", tz="UTC"))


def test_funding_votes_by_hand():
    s = FundingStrategy(high=0.0005, low=-0.0001, average=3, max_age_hours=12)
    s.attach_feed(FakeFeed("funding", FUNDING))
    sigs = s.generate_signals("BTC/USDT", candles(48))
    by_time = {sig.timestamp: sig for sig in sigs}
    # averages known at each close: 08:00 -> (6+7+8)/3 bp above high -> SELL; 32:00 -> negative -> BUY
    sell = by_time[T0 + 8 * H]  # its close at 09:00 sees the 08:00 print
    assert sell.direction is Direction.SELL and sell.metadata["positioning"] == "crowded longs"
    assert sell.metadata["funding"] == pytest.approx(0.0007)
    buy = by_time[T0 + 32 * H]
    assert buy.direction is Direction.BUY and buy.metadata["funding"] == pytest.approx(-0.0003)
    early = by_time[T0 + 6 * H]  # closes at 07:00: only the 00:00 print is known, average (1+6+7)/3 bp
    assert early.direction is Direction.HOLD and early.metadata["funding"] == pytest.approx(0.0014 / 3)
    assert by_time[T0 + 7 * H].direction is Direction.SELL  # closes at 08:00, when the 08:00 print arrives
    assert by_time[T0 + 47 * H].metadata["known_at"].startswith("2024-02-02T16")  # the last print, 8h old
    late = {sig.timestamp: sig for sig in s.generate_signals("BTC/USDT", candles(60))}
    assert late[T0 + 55 * H].metadata["reason"] == "stale"  # 16h after the last print, max_age 12h
    follow = FundingStrategy(mode="follow")
    follow.attach_feed(FakeFeed("funding", FUNDING))
    assert follow.generate_signals("BTC/USDT", candles(48))[8].direction is Direction.BUY
    for i in range(0, 48, 5):  # live and backtest give the same signal for every bar
        assert s.generate_signal("BTC/USDT", candles(48).iloc[: i + 1]) == sigs[i]
    assert s.generate_signals_at("BTC/USDT", candles(48), [3, 9]) == [sigs[3], sigs[9]]


def test_sentiment_votes_and_failures():
    days = pd.date_range(T0 - timedelta(days=2), periods=5, freq="D", tz="UTC") + SENTIMENT_LAG
    s = SentimentStrategy(fear=25, greed=75, max_age_hours=30)
    s.attach_feed(FakeFeed("sentiment", pd.Series([20.0, 50.0, 80.0, 10.0, 60.0], index=days)))
    sigs = {sig.timestamp: sig for sig in s.generate_signals("BTC/USDT", candles(24 * 4))}
    # known: T0-1d 20, T0 50, T0+1d 80, T0+2d 10, T0+3d 60 (each value a day after the day it describes)
    assert sigs[T0 + 2 * H].direction is Direction.HOLD and sigs[T0 + 2 * H].metadata["sentiment"] == 50.0
    assert sigs[T0 + 23 * H].metadata["sentiment"] == 80.0  # its close at T0+1d is when 80 becomes known
    assert sigs[T0 + 24 * H].direction is Direction.SELL and sigs[T0 + 24 * H].metadata["mood"] == "greed"
    assert sigs[T0 + 48 * H].direction is Direction.BUY and sigs[T0 + 48 * H].metadata["sentiment"] == 10.0
    assert sigs[T0 + 80 * H].direction is Direction.HOLD and sigs[T0 + 80 * H].metadata["mood"] == "neutral"
    late = SentimentStrategy(max_age_hours=1)
    late.attach_feed(FakeFeed("sentiment", pd.Series([10.0], index=days[:1])))
    assert late.generate_signals("BTC/USDT", candles(10))[-1].metadata["reason"] == "stale"
    broken = SentimentStrategy()
    broken.attach_feed(FakeFeed("sentiment", None, fail=True))
    sig = broken.generate_signals("BTC/USDT", candles(5))[0]
    assert sig.direction is Direction.HOLD and sig.metadata["reason"] == "data error" and "down" in sig.metadata["error"]
    unattached = SentimentStrategy().generate_signals("BTC/USDT", candles(3))
    assert all(x.direction is Direction.HOLD and x.metadata["reason"] == "no data feed" for x in unattached)
    nothing = SentimentStrategy()
    nothing.attach_feed(FakeFeed("sentiment", pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))))
    assert nothing.generate_signals("BTC/USDT", candles(3))[0].metadata["reason"] == "no data"


@pytest.mark.parametrize("bad", [{"high": 0.0001, "low": 0.0002}, {"average": 0}, {"mode": "sideways"},
                                 {"max_age_hours": 0}])
def test_funding_parameters(bad):
    with pytest.raises(ValueError):
        FundingStrategy(**bad)


def test_sentiment_parameters():
    with pytest.raises(ValueError):
        SentimentStrategy(fear=80, greed=20)


# ------------------------------------------------------------------ engines
CFG = AppConfig.from_mapping({
    "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
    "strategies": {"rsi": {}, "macd": {}, "bollinger": {},
                   "funding": {"weight": 1.0, "high": 0.00012, "low": -0.0001},
                   "sentiment": {"weight": 1.0, "fear": 35, "greed": 70}},
})


def test_backtests_use_the_feeds_and_live_matches():
    bars = 24 * 10
    bt = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    votes = [s for s in bt.signals if s.strategy in ("funding", "sentiment") and s.direction is not Direction.HOLD]
    assert {s.strategy for s in votes} == {"funding", "sentiment"}
    assert bt.trades
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(bars // 2):
            trader.run_cycle()
            clock.now += H
        trader.stop()
        resumed = LivePaperTrader.resume(store, trader.run_id, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock),
                                         clock=clock)
        for _ in range(bars - bars // 2):
            resumed.run_cycle()
            clock.now += H
        live = [fill_key(f)[1:] for f in resumed.portfolio.fills if f.timestamp < START + bars * H]
    assert live == [fill_key(f)[1:] for f in bt.fills]


def test_a_source_without_context_data_is_refused():
    class Bare(MarketDataProvider):
        name = "bare"

        def fetch_ohlcv(self, symbol, timeframe, since, until=None):
            return SyntheticProvider(seed=1).fetch_ohlcv(symbol, timeframe, since, until)

    with pytest.raises(ConfigError, match="needs funding data"):
        BacktestEngine(CFG, Bare())


def test_wrappers_pass_the_feeds_through(tmp_path):
    inner = SyntheticProvider(seed=2)
    assert CachedProvider(inner, tmp_path).context_feed("funding", "1h").name == "synthetic-2-funding"
    memo = MemoizedProvider(inner)
    assert memo.context_feed("sentiment", "1h") is memo.context_feed("sentiment", "1h")  # shared between backtests
    assert inner.context_feed("nope", "1h") is None


# -------------------------------------------------------------- real feeds
class FakeClient:
    def __init__(self, rows, fail_times=0):
        self.rows, self.fail_times, self.calls = rows, fail_times, []

    def fetch_funding_rate_history(self, symbol, since=None, limit=None):
        self.calls.append((symbol, since, limit))
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("timeout")
        return [r for r in self.rows if r["timestamp"] >= since][:limit]


def funding_rows(n):
    base = int(T0.timestamp() * 1000)
    return [{"timestamp": base + i * 8 * 3600_000, "fundingRate": 0.0001 * (i % 5)} for i in range(n)]


def test_ccxt_funding_feed():
    client = FakeClient(funding_rows(30))
    feed = CcxtFundingFeed("binanceusdm", client=client, page_limit=7, sleep=lambda s: None)
    s = feed.series("BTC/USDT", T0, T0 + timedelta(days=5))
    assert len(s) == 16 and s.index[0] == pd.Timestamp(T0) and s.iloc[1] == pytest.approx(0.0001)
    assert all(call[0] == "BTC/USDT:USDT" and call[2] == 7 for call in client.calls)  # paginated, perpetual symbol
    before = len(client.calls)
    feed.series("BTC/USDT", T0 + timedelta(days=1), T0 + timedelta(days=4))  # covered: no new request
    assert len(client.calls) == before
    later = feed.series("BTC/USDT", T0, T0 + timedelta(days=9))
    assert len(later) == 28 and client.calls[before][1] > int(T0.timestamp() * 1000)  # only the tail
    flaky = CcxtFundingFeed("x", client=FakeClient(funding_rows(3), fail_times=2), sleep=lambda s: None)
    assert len(flaky.series("BTC/USDT", T0, T0 + timedelta(days=1))) == 3  # retried
    dead = CcxtFundingFeed("x", client=FakeClient([], fail_times=9), max_retries=1, sleep=lambda s: None)
    with pytest.raises(FeedError, match="funding rates for BTC/USDT:USDT from x"):
        dead.series("BTC/USDT", T0, T0 + timedelta(days=1))


def test_the_funding_client_must_be_public():
    client = FakeClient([])
    client.apiKey = "not-allowed"
    with pytest.raises(DataError, match="credentials"):
        CcxtFundingFeed("binanceusdm", client=client)


def test_fear_greed_feed():
    calls = []
    days = [int((T0 + timedelta(days=i)).timestamp()) for i in range(3)]

    def fetch(url):
        calls.append(url)
        return {"data": [{"value": str(40 + i), "timestamp": str(ts)} for i, ts in enumerate(days)]}

    now = [T0 + timedelta(days=3)]
    feed = FearGreedFeed(fetch=fetch, clock=lambda: now[0])
    s = feed.series("BTC/USDT", T0, T0 + timedelta(days=4))
    assert list(s) == [40.0, 41.0, 42.0] and s.index[0] == pd.Timestamp(T0) + SENTIMENT_LAG  # known a day later
    feed.series("BTC/USDT", T0, T0 + timedelta(days=4))
    assert len(calls) == 1  # cached for an hour
    now[0] += timedelta(hours=2)
    feed.series("BTC/USDT", T0, T0 + timedelta(days=4))
    assert len(calls) == 2 and "api.alternative.me" in calls[0]
    with pytest.raises(FeedError, match="Fear & Greed"):
        FearGreedFeed(fetch=lambda url: {"oops": 1}).series("BTC/USDT", T0, T0 + timedelta(days=1))


def test_the_exchange_provider_offers_both_feeds():
    provider = CcxtPublicProvider("binance", client=FakeClient([]))
    assert provider.context_feed("funding", "1h").name == "binanceusdm-funding"
    assert provider.context_feed("sentiment", "1h").name == "alternative.me-fear-greed"
    assert provider.context_feed("funding", "1h") is provider.context_feed("funding", "4h")  # one per kind
    okx = CcxtPublicProvider("binance", client=FakeClient([]), funding_exchange="okx")
    assert okx.context_feed("funding", "1h").name == "okx-funding"
    config = AppConfig.from_mapping({"data": {"funding_exchange": "bybit"}})
    assert config.data.funding_exchange == "bybit"
    with pytest.raises(Exception):
        AppConfig.from_mapping({"data": {"funding_exchange": 5}})
