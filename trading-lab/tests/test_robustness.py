"""Stage 13C: bootstrap robustness ranges."""

import hashlib
from datetime import datetime, timezone

import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.html_report import build_html_report
from trading_lab.research import bootstrap, format_robustness, robustness_for_run
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START, END = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 3, 1, tzinfo=UTC)


def curve(n=200, step=0.001, start=10_000.0):
    values = [start]
    for i in range(n):
        values.append(values[-1] * (1 + step * (1 if i % 3 else -1)))
    return values


def test_deterministic_and_ordered():
    pnls = [50, -30, 20, -10, 80, -60, 15, 5] * 5
    a = bootstrap(pnls, curve(), initial_cash=10_000, timeframe="1h", samples=2000, seed=3)
    b = bootstrap(pnls, curve(), initial_cash=10_000, timeframe="1h", samples=2000, seed=3)
    c = bootstrap(pnls, curve(), initial_cash=10_000, timeframe="1h", samples=2000, seed=4)
    assert a == b and a != c
    for r in (a.trade_total_return, a.bar_total_return, a.sharpe_range):
        assert r.low <= r.median <= r.high
    assert a.trades == 40 and a.bars == 200 and a.block_bars == 14


def test_known_cases():
    winners = bootstrap([10.0] * 40, curve(), initial_cash=1000, timeframe="1h", samples=500)
    assert winners.prob_loss == 0 and winners.trade_total_return.low == pytest.approx(0.4)
    coin = bootstrap([10.0, -10.0] * 100, curve(), initial_cash=1000, timeframe="1h", samples=4000)
    assert 0.4 < coin.prob_loss < 0.55 and coin.trade_total_return.low < 0 < coin.trade_total_return.high
    assert "includes both gains and losses" in " ".join(coin.warnings)
    flat_growth = [10_000 * 1.001 ** i for i in range(101)]  # identical returns: no uncertainty at all
    steady = bootstrap([1.0] * 40, flat_growth, initial_cash=10_000, timeframe="1h", samples=500)
    assert steady.bar_total_return.low == pytest.approx(steady.bar_total_return.high)
    assert steady.bar_total_return.median == pytest.approx(steady.total_return)


def test_warnings_and_edge_cases():
    few = bootstrap([5.0, -3.0], curve(20), initial_cash=1000, timeframe="1h", samples=200)
    assert any("too few to judge" in w for w in few.warnings)
    none = bootstrap([], curve(5), initial_cash=1000, timeframe="1h", samples=200)
    assert none.trade_total_return is None and none.prob_loss is None and none.bar_total_return is None
    assert "no per-bar bootstrap" in " ".join(none.warnings)
    with pytest.raises(ValueError):
        bootstrap([1.0], curve(), initial_cash=1, timeframe="1h", samples=10)
    assert "probability of a loss n/a" in format_robustness(none)


def test_stored_run_cli_and_report(tmp_path, capsys):
    db = tmp_path / "h.db"
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
    with SQLiteStore(db) as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=2), store=store).run(START, END)
        robust = robustness_for_run(store, result.run_id, samples=1000)
    assert robust.trades == len(result.trades) and robust.total_return == pytest.approx(result.metrics.total_return)
    assert robust.sharpe == pytest.approx(result.metrics.sharpe_ratio)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert main(["--db", str(db), "robustness", "--samples", "500"]) == 0
    out = capsys.readouterr().out
    assert result.run_id in out and "Trade bootstrap" in out and "Block bootstrap" in out
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    page = build_html_report(str(db))
    assert "<h2>Robustness</h2>" in page and "P(loss)" in page
