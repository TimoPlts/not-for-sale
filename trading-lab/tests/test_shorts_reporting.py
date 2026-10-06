"""Stage 16C: shorts in attribution, reports, exports, the dashboard and summaries."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from test_live import ANCHOR, START, Clock
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig, VotingConfig
from trading_lab.core.models import SHORT, ClosedTrade, Decision, DecisionAction, Direction, Signal
from trading_lab.dashboard import DashboardData
from trading_lab.data import SyntheticProvider
from trading_lab.export import export_run
from trading_lab.html_report import build_html_report
from trading_lab.live import LivePaperTrader
from trading_lab.research import attribute
from trading_lab.storage import SQLiteStore
from trading_lab.summary import build_summary, format_summary

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}, "risk": {"allow_short": True}})


def t(i):
    return T0 + i * H


def test_a_sell_vote_agrees_with_a_short_entry():
    agent = {"params": {"agent": "a"}}
    signals = [
        Signal("a", "BTC/USDT", Direction.SELL, 0.8, t(0), agent),
        Signal("b", "BTC/USDT", Direction.BUY, 0.3, t(0), agent),
        Signal("ensemble", "BTC/USDT", Direction.SELL, 0.25, t(0), {"votes": [
            {"strategy": "a", "direction": "sell", "confidence": 0.8, "weight": 1.0},
            {"strategy": "b", "direction": "buy", "confidence": 0.3, "weight": 1.0}]}),
    ]
    decisions = [Decision(t(0), "BTC/USDT", DecisionAction.ENTER_SIGNAL, "short entry scheduled for next bar open")]
    trades = [ClosedTrade("BTC/USDT", 1.0, 100, 90, 101, 111, 10, t(1), t(3), "c", side=SHORT)]
    closes = {"BTC/USDT": pd.Series([100.0, 99, 95, 90], index=pd.DatetimeIndex([t(i) for i in range(4)]))}
    result = attribute(signals, decisions, trades, closes, voting=VotingConfig(), initial_cash=10_000, horizon=2)
    a, b = result["a"], result["b"]
    assert (a.trades_agreed, a.trades_disagreed, a.pnl_agreed) == (1, 0, 10)
    assert a.trades_pivotal == 1  # without its SELL the ensemble would have said BUY, not SELL
    assert (b.trades_agreed, b.trades_disagreed, b.trades_pivotal) == (0, 1, 0)


@pytest.fixture(scope="module")
def short_run(tmp_path_factory):
    db = tmp_path_factory.mktemp("shorts") / "s.db"
    with SQLiteStore(db) as store:
        result = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR), store=store).run(
            START, START + 200 * H)
    assert any(tr.side == SHORT for tr in result.trades)
    return db, result


def test_cli_report_and_backtest_show_the_side(short_run, capsys, tmp_path):
    db, result = short_run
    assert main(["--db", str(db), "report", result.run_id, "--limit", "200"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "qty=" in ln]
    assert any(" short qty=" in ln for ln in lines) and any(" long  qty=" in ln for ln in lines)
    assert main(["--db", str(tmp_path / "b.db"), "backtest", "--synthetic", "11", "--days", "8",
                 "--symbols", "BTC/USDT"]) == 0
    assert "shorts=" not in capsys.readouterr().out  # long-only: unchanged output


def test_export_html_and_dashboard_show_the_side(short_run, tmp_path):
    db, result = short_run
    with SQLiteStore(db, readonly=True) as store:
        export_run(store, result.run_id, tmp_path / "x")
    trades = pd.read_csv(tmp_path / "x" / "trades.csv")
    assert list(trades.columns)[-1] == "side" and set(trades["side"]) == {"long", "short"}
    assert (trades["side"] == "short").sum() == sum(tr.side == SHORT for tr in result.trades)
    page = build_html_report(str(db), result.run_id)
    assert "<th>Side</th>" in page and "<td>short</td>" in page
    with DashboardData(db) as data:
        assert {r["side"] for r in data.recent_trades(result.run_id, limit=500)} == {"long", "short"}


def test_open_shorts_on_the_dashboard_and_in_summaries(tmp_path):
    db = tmp_path / "p.db"
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(db) as store:
        trader = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(200):
            trader.run_cycle()
            clock.now += H
            if any(p.is_short for p in trader.portfolio.positions.values()) and any(
                    tr.side == SHORT for tr in trader.portfolio.closed_trades):
                break
        run_id = trader.run_id
        shorts = {s: p for s, p in trader.portfolio.positions.items() if p.is_short}
        assert shorts
    with DashboardData(db) as data:
        positions = {p["symbol"]: p for p in data.open_positions(run_id)}
        overview = data.overview(run_id)
    for sym, pos in shorts.items():
        shown = positions[sym]
        assert shown["side"] == "short" and shown["stop_price"] > shown["current_price"] * 0.9
        assert shown["value"] == pytest.approx(pos.quantity * shown["current_price"])
    gross = sum(p["value"] for p in positions.values())
    assert overview["exposure_pct"] == pytest.approx(gross / overview["equity"])
    with SQLiteStore(db, readonly=True) as store:
        text = format_summary(build_summary(store, run_id, hours=1000))
    assert "short)" in text
