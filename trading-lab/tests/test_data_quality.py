"""Stage 14B: the market-data quality report."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import trading_lab.cli as cli
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.errors import DataError
from trading_lab.data import MarketDataProvider, SyntheticProvider, candles_from_closes
from trading_lab.data.quality import (
    QualityRules,
    check_candles,
    check_market_data,
    check_stored_bars,
    format_quality,
)
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 2, 1, tzinfo=UTC)


def walk(n=300, vol=0.005, seed=1):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))


def frame(closes=None):
    return candles_from_closes(walk() if closes is None else closes, start=T0)


def check(candles, n=300, **kwargs):
    return check_candles("BTC/USDT", candles, "1h", T0, T0 + n * H, **kwargs)


class Fixed(MarketDataProvider):
    def __init__(self, data=None, error=None):
        self.data, self.error = data, error

    @property
    def name(self):
        return "fixed"

    def fetch_ohlcv(self, symbol, timeframe, since, until=None):
        if self.error is not None:
            raise self.error
        return self.data


def test_clean_data_is_ok():
    r = check(frame())
    assert r.ok and r.expected == r.bars == 300 and r.missing == 0
    assert r.errors == r.warnings == [] and r.jump_threshold >= 0.05
    reports = check_market_data(SyntheticProvider(seed=3), ["BTC/USDT", "ETH/USDT"], "1h", T0, T0 + 500 * H)
    assert all(r.ok and r.bars == 500 for r in reports), format_quality(reports)
    assert format_quality(reports).endswith("OK: no problems found")


def test_gaps_and_missing_edges():
    data = frame()
    data = data.drop(data.index[[0, 1, 50, 51, 52, 120, 299]])
    r = check(data)
    assert r.missing_head == 2 and r.missing_tail == 1
    assert r.gaps == [(T0 + 50 * H, T0 + 52 * H, 3), (T0 + 120 * H, T0 + 120 * H, 1)]
    assert r.missing == 7 and r.bars == 293 and not r.ok and not r.errors
    assert "2 gap(s), 4 candle(s) missing" in r.warnings
    text = format_quality([r])
    assert "gap: 2024-02-03 02:00 -> 2024-02-03 04:00 (3 candle(s))" in text
    assert text.splitlines()[-1].startswith("WARNINGS")


def test_zero_volume_and_flat_candles():
    data = frame()
    data.iloc[10, data.columns.get_loc("volume")] = 0.0
    price = data["close"].iloc[20]
    data.iloc[20, :4] = price
    data.iloc[21, data.columns.get_loc("open")] = price  # keep the next open continuous
    r = check(data)
    assert r.zero_volume == [data.index[10]] and r.flat == [data.index[20]]
    assert r.jumps == [] and r.open_gaps == []


def test_extreme_moves_adapt_to_the_symbol():
    closes = walk()
    closes[100:] *= 1.3  # one candle about +30%
    r = check(frame(closes))
    assert [ts for ts, _ in r.jumps] == [T0 + 100 * H]
    assert r.jumps[0][1] == pytest.approx(closes[100] / closes[99] - 1, rel=1e-9)
    assert r.open_gaps == []  # candles_from_closes opens at the previous close

    volatile = check(frame(walk(vol=0.04, seed=2)))  # 4% candles are normal here
    assert volatile.jumps == [] and volatile.jump_threshold > 0.3
    calm = check(frame(walk(vol=0.0001)))
    assert calm.jump_threshold == pytest.approx(0.05)  # the floor
    strict = check(frame(walk(vol=0.0001)), rules=QualityRules(jump_floor=0.001, jump_sigmas=3))
    assert strict.jump_threshold < 0.05


def test_opens_far_from_the_previous_close():
    data = frame()
    data.iloc[150, data.columns.get_loc("open")] *= 1.2
    data.iloc[150, data.columns.get_loc("high")] = data.iloc[150][["open", "close", "high"]].max()
    data.iloc[200, data.columns.get_loc("open")] *= 1.2  # right after a gap: explained by the gap
    data.iloc[200, data.columns.get_loc("high")] = data.iloc[200][["open", "close", "high"]].max()
    data = data.drop(data.index[199])
    r = check(data)
    assert [ts for ts, _ in r.open_gaps] == [T0 + 150 * H]
    assert "open vs previous close: 2024-02-07 06:00 +20.0%" in format_quality([r])


def test_stale_data_when_checking_up_to_now():
    data = frame()  # last candle opens at T0 + 299h
    just_closed = T0 + 300 * H + timedelta(minutes=1)
    assert check_candles("X", data, "1h", T0, None, now=just_closed).ok
    publishing = check_candles("X", data, "1h", T0, None, now=just_closed + H)  # newest not out yet
    assert publishing.ok and publishing.missing_tail == 1
    stale = check_candles("X", data, "1h", T0, None, now=just_closed + 5 * H)
    assert stale.stale and stale.missing_tail == 5 and not stale.warnings
    assert stale.errors == ["stale: the latest 5 closed candle(s) are missing (last 2024-02-13 11:00)"]
    assert format_quality([stale]).splitlines()[-1].startswith("ERRORS")
    # A fixed past period is never stale; its end is clipped to what has closed.
    past = check_candles("X", data, "1h", T0, T0 + 400 * H, now=just_closed)
    assert past.ok and past.expected == 300


def test_failures_and_empty_data_are_errors():
    reports = check_market_data(Fixed(error=DataError("exchange down")), ["BTC/USDT"], "1h", T0, T0 + 10 * H)
    assert reports[0].errors == ["DataError: exchange down"] and reports[0].warnings == []
    empty = check(frame().iloc[:0])
    assert empty.errors == ["no candles returned"] and empty.missing_tail == 300
    with pytest.raises(ValueError, match="current time"):
        check_market_data(Fixed(frame()), ["BTC/USDT"], "1h", T0)


@pytest.mark.parametrize("rules", [{"jump_floor": 0}, {"jump_sigmas": -1}, {"stale_grace_bars": -1},
                                   {"jump_floor": float("nan")}])
def test_rules_are_validated(rules):
    with pytest.raises(ValueError):
        QualityRules(**rules)


def test_stored_bars_of_a_run(tmp_path):
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
    with SQLiteStore(tmp_path / "r.db") as store:
        run = BacktestEngine(cfg, SyntheticProvider(seed=4), store=store).run(T0, T0 + 100 * H)
        reports = check_stored_bars(store, run.run_id)
        assert [r.symbol for r in reports] == ["BTC/USDT", "ETH/USDT"]
        assert all(r.ok and r.bars == r.expected == 100 for r in reports), format_quality(reports)
        with pytest.raises(ValueError, match="unknown run"):
            check_stored_bars(store, "nope")


def test_cli(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "c.db")
    args = ["--db", db, "data-check", "--symbols", "BTC/USDT", "ETH/USDT"]
    assert main([*args, "--synthetic", "3", "--days", "3"]) == 0  # up to now: synthetic data is never stale
    assert "OK: no problems found" in capsys.readouterr().out
    assert main([*args, "--synthetic", "3", "--start", "2024-01-01", "--end", "2024-01-15"]) == 0
    assert main([*args, "--synthetic", "3", "--start", "2024-01-15", "--end", "2024-01-01"]) == 1

    gappy = frame().drop(frame().index[50])
    monkeypatch.setattr(cli, "_provider", lambda cfg, seed: Fixed(gappy))
    period = ["--start", "2024-02-01", "--end", "2024-02-13"]
    assert main([*args, *period]) == 0  # warnings only
    assert "1 gap(s)" in capsys.readouterr().out
    assert main([*args, *period, "--strict"]) == 1
    assert main([*args, *period, "--jump-floor", "0"]) == 1  # invalid rule
    monkeypatch.setattr(cli, "_provider", lambda cfg, seed: Fixed(error=DataError("down")))
    assert main([*args, *period]) == 1

    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    with SQLiteStore(db) as store:
        run_id = BacktestEngine(cfg, SyntheticProvider(seed=4), store=store).run(T0, T0 + 50 * H).run_id
    assert main(["--db", db, "data-check", "--run", run_id]) == 0
    assert f"stored by run {run_id}" in capsys.readouterr().out
    assert main(["--db", db, "data-check", "--run", "nope"]) == 1
    assert main(["--db", str(tmp_path / "none.db"), "data-check", "--run", run_id]) == 1
