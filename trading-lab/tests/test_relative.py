"""Stage 13D: performance relative to buy & hold (beta, alpha, correlation, information ratio)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from test_live import ANCHOR, START as LIVE_START, Clock

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.dashboard import DashboardData
from trading_lab.data import SyntheticProvider
from trading_lab.html_report import build_html_report
from trading_lab.live import LivePaperTrader
from trading_lab.metrics import periods_per_year, relative_metrics
from trading_lab.research import run_experiment
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START, END = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)
BENCH_RETURNS = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.0, 0.01, -0.015]


def grow(returns, start=1000.0):
    out = [start]
    for r in returns:
        out.append(out[-1] * (1 + r))
    return out


def test_definitions_on_hand_made_curves():
    bench = grow(BENCH_RETURNS)
    same = relative_metrics(bench, bench, "1h")
    assert same.beta == pytest.approx(1) and same.alpha_annualized == pytest.approx(0, abs=1e-12)
    assert same.correlation == pytest.approx(1) and same.excess_return == 0
    assert same.tracking_error == 0 and same.information_ratio is None  # no active risk: undefined
    half = relative_metrics(grow([r / 2 for r in BENCH_RETURNS]), bench, "1h")
    assert half.beta == pytest.approx(0.5) and half.correlation == pytest.approx(1)
    assert half.alpha_annualized == pytest.approx(0, abs=1e-9)
    extra = relative_metrics(grow([r + 0.001 for r in BENCH_RETURNS]), bench, "1h")
    assert extra.beta == pytest.approx(1) and extra.alpha_annualized == pytest.approx(0.001 * periods_per_year("1h"))
    cash = relative_metrics([1000.0] * 11, bench, "1h")
    assert cash.beta == pytest.approx(0) and cash.correlation is None and cash.excess_return < 0
    assert relative_metrics([1, 2], [1, 2], "1h") is None and relative_metrics([1, 2, 3], [1, 2], "1h") is None


def test_backtests_store_and_print_it(tmp_path, capsys):
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
    db = tmp_path / "h.db"
    with SQLiteStore(db) as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store).run(START, END)
        stored = store.load_metrics(result.run_id)["relative"]
    rel = result.relative
    assert rel is not None and stored == json.loads(json.dumps(rel.to_dict()))
    assert rel.excess_return == pytest.approx(result.metrics.total_return - result.benchmark.total_return)
    assert 0 <= rel.beta < 1.5 and -1 <= rel.correlation <= 1
    assert main(["--db", str(db), "report", result.run_id]) == 0
    assert "Relative to buy & hold: excess return" in capsys.readouterr().out
    with DashboardData(db) as data:
        assert data.research(result.run_id)["relative"]["beta"] == pytest.approx(rel.beta)
    page = build_html_report(str(db))
    assert "Excess vs buy &amp; hold" in page and ">Beta<" in page


def test_paper_runs_get_it_from_stored_bars(tmp_path):
    clock = Clock(LIVE_START + timedelta(hours=1, minutes=1))
    db = tmp_path / "p.db"
    with SQLiteStore(db) as store:
        trader = LivePaperTrader(AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}}),
                                 SyntheticProvider(seed=1, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(30):
            trader.run_cycle()
            clock.now += timedelta(hours=1)
    with DashboardData(db) as data:
        rel = data.research(trader.run_id)["relative"]
    assert rel is not None and rel["bars"] == 30


def test_experiments_include_it():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    rows = run_experiment(cfg, SyntheticProvider(seed=3), START, END, ["baseline"])
    assert rows[0].summary()["relative"]["beta"] == pytest.approx(rows[0].relative.beta)
