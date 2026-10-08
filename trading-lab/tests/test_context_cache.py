"""Stage 30B: context feeds on disk (CachedContextFeed) and trading-lab prefetch."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trading_lab.cli import main
from trading_lab.data import CachedProvider, SyntheticProvider
from trading_lab.data.context import CachedContextFeed, ContextFeed, FeedError

UTC = timezone.utc
T0 = datetime(2024, 2, 1, tzinfo=UTC)
NOW = T0 + timedelta(days=60)


class CountingFeed(ContextFeed):
    """Funding every 8 hours with an exact-looking but awkward float, counting requests."""

    def __init__(self, kind="funding", fail=False):
        self.kind, self.fail, self.calls = kind, fail, []

    @property
    def name(self):
        return f"counting-{self.kind}"

    def series(self, symbol, since, until):
        self.calls.append((symbol, pd.Timestamp(since), pd.Timestamp(until)))
        if self.fail:
            raise FeedError("source down")
        freq = "8h" if self.kind == "funding" else "D"
        index = pd.date_range(pd.Timestamp(since).ceil(freq), pd.Timestamp(until), freq=freq, tz="UTC")
        return pd.Series([0.1 / 3 + i * 1e-7 for i in range(len(index))], index=index, name=self.kind)


def cached(tmp_path, inner):
    return CachedContextFeed(inner, tmp_path, clock=lambda: NOW)


def test_fetches_only_what_is_missing(tmp_path):
    inner = CountingFeed()
    feed = cached(tmp_path, inner)
    first = feed.series("BTC/USDT", T0, T0 + timedelta(days=10))
    assert len(inner.calls) == 1 and len(first) == 31
    assert feed.series("BTC/USDT", T0 + timedelta(days=2), T0 + timedelta(days=5)).equals(
        first[(first.index >= T0 + timedelta(days=2)) & (first.index <= T0 + timedelta(days=5))])
    assert len(inner.calls) == 1  # covered: no request
    again = cached(tmp_path, CountingFeed())  # a new process
    assert again.series("BTC/USDT", T0, T0 + timedelta(days=10)).equals(first)  # exact round trip from disk
    assert again._inner.calls == []
    feed.series("BTC/USDT", T0, T0 + timedelta(days=20))
    assert inner.calls[-1][1] == first.index[-1]  # only the tail is fetched
    feed.series("BTC/USDT", T0 - timedelta(days=5), T0 + timedelta(days=20))
    assert inner.calls[-1][1] == pd.Timestamp(T0 - timedelta(days=5))  # an earlier start refetches the range
    whole = feed.series("BTC/USDT", T0 - timedelta(days=5), T0 + timedelta(days=20))
    assert whole.index.is_unique and whole.index.is_monotonic_increasing
    assert feed.path("BTC/USDT").name == "BTC-USDT.csv" and feed.path("ETH/USDT").name == "ETH-USDT.csv"


def test_values_not_yet_published_are_not_asked_for_on_every_bar(tmp_path):
    inner = CountingFeed()
    feed = CachedContextFeed(inner, tmp_path, clock=lambda: T0 + timedelta(days=3, hours=1))
    feed.series("BTC/USDT", T0, T0 + timedelta(days=3, hours=1))
    for hour in range(2, 6):  # later bars of a live run, inside the range already asked for
        feed.series("BTC/USDT", T0, T0 + timedelta(days=3, hours=hour - 1))
    assert len(inner.calls) == 1


def test_a_failing_source_falls_back_to_the_file(tmp_path):
    cached(tmp_path, CountingFeed()).series("BTC/USDT", T0, T0 + timedelta(days=5))
    down = cached(tmp_path, CountingFeed(fail=True))
    served = down.series("BTC/USDT", T0, T0 + timedelta(days=30))
    assert len(served) == 16 and len(down._inner.calls) == 1  # what the file has
    with pytest.raises(FeedError):
        cached(tmp_path, CountingFeed(fail=True)).series("ETH/USDT", T0, T0 + timedelta(days=5))  # no file


def test_sentiment_is_stored_once(tmp_path):
    feed = cached(tmp_path, CountingFeed("sentiment"))
    a = feed.series("BTC/USDT", T0, T0 + timedelta(days=10))
    assert feed.path("BTC/USDT") == feed.path("SOL/USDT") and feed.path("SOL/USDT").name == "all.csv"
    assert feed.series("SOL/USDT", T0, T0 + timedelta(days=10)).equals(a) and len(feed._inner.calls) == 1


def test_the_cached_provider_wraps_its_feeds(tmp_path):
    provider = CachedProvider(SyntheticProvider(seed=2), tmp_path)
    feed = provider.context_feed("funding", "1h")
    assert isinstance(feed, CachedContextFeed) and feed is provider.context_feed("funding", "1h")
    assert feed.name == "synthetic-2-funding" and provider.context_feed("nope", "1h") is None
    values = feed.series("BTC/USDT", T0, T0 + timedelta(days=3))
    assert feed.path("BTC/USDT").exists()
    direct = SyntheticProvider(seed=2).context_feed("funding", "1h").series("BTC/USDT", T0, T0 + timedelta(days=3))
    assert (values.to_numpy() == direct.to_numpy()).all()


def test_prefetch(tmp_path, capsys):
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[market]\nsymbols = ["BTC/USDT", "ETH/USDT"]\n[data]\ncache_dir = "{tmp_path / "cache"}"\n')
    assert main(["--config", str(cfg), "prefetch", "--synthetic", "4", "--start", "2024-02-01",
                 "--end", "2024-02-11"]) == 0
    out = capsys.readouterr().out
    assert "candles BTC/USDT" in out and "funding ETH/USDT" in out and "sentiment" in out and "Done." in out
    assert (tmp_path / "cache" / "synthetic-4" / "BTC-USDT_1h.csv").exists()
    assert (tmp_path / "cache" / "context" / "synthetic-4-funding" / "ETH-USDT.csv").exists()
    assert (tmp_path / "cache" / "context" / "synthetic-4-sentiment" / "all.csv").exists()
    assert main(["--config", str(cfg), "prefetch", "--synthetic", "4", "--days", "5", "--no-context"]) == 0
    assert "funding" not in capsys.readouterr().out
    off = tmp_path / "off.toml"
    off.write_text("[data]\nuse_cache = false\n")
    assert main(["--config", str(off), "prefetch", "--synthetic", "4"]) == 1
