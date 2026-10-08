"""Stage 25B: maximum adverse and favourable excursions (MAE/MFE) of closed trades."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.models import ClosedTrade
from trading_lab.data import SyntheticProvider
from trading_lab.presets import PRESETS
from trading_lab.research.trades import (
    TradeRow,
    analyze_trades,
    excursion,
    excursion_summary,
    format_trades,
    group_stats,
)
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    db = tmp_path_factory.mktemp("exc") / "h.db"
    with SQLiteStore(db) as store:
        longs = BacktestEngine(PRESETS["trend"].config(BASE), SyntheticProvider(seed=2), store=store).run(
            START, START + timedelta(days=30))
        shorts = BacktestEngine(PRESETS["trend-shorts"].config(BASE), SyntheticProvider(seed=2), store=store).run(
            START, START + timedelta(days=30))
        atr = BacktestEngine(BASE.with_overrides({"risk": {"stop_mode": "atr"}}), SyntheticProvider(seed=2),
                             store=store).run(START, START + timedelta(days=10))
    return db, longs, shorts, atr


def test_excursion_by_hand():
    assert excursion("long", 100.0, 101.0, [98.0, 95.0], [103.0, 104.0]) == pytest.approx((-0.05, 0.04))
    assert excursion("short", 100.0, 99.0, [97.0], [103.0, 106.0]) == pytest.approx((-0.06, 0.03))
    assert excursion("long", 100.0, 102.0, [100.5], [102.5]) == pytest.approx((0.0, 0.025))  # never below
    assert excursion("long", 100.0, 96.0, [], []) == pytest.approx((-0.04, 0.0))  # exit in the entry bar
    assert excursion("short", 100.0, 103.0, [], []) == pytest.approx((-0.03, 0.0))


def test_summary_by_hand():
    t0 = datetime(2024, 1, 1, tzinfo=UTC)

    def row(pnl, mae, mfe):
        return TradeRow("BTC/USDT", "long", t0, t0 + timedelta(hours=3), pnl, pnl / 1000, 3, "signal",
                        100.0, 100.0, mae, mfe)

    winners = [row(10, -i / 1000, 0.03) for i in range(10)]  # dips 0.0% .. 0.9%
    losers = [row(-10, -0.02, 0.015), row(-10, -0.02, 0.005), row(-10, -0.03, 0.0)]
    x = excursion_summary(winners + losers, 0.05)
    assert x.winners_mae_p90 == pytest.approx(pd.Series([-i / 1000 for i in range(10)]).quantile(0.10))
    assert x.losers == 3 and x.losers_in_profit == 1 and x.stop_loss_pct == 0.05
    mean_mae = (sum(i / 1000 for i in range(10)) + 0.07) / 13
    mean_mfe = (0.3 + 0.02) / 13
    assert x.e_ratio == pytest.approx(mean_mfe / mean_mae)
    g = group_stats("all", winners + losers)
    assert g.avg_mae == pytest.approx(-mean_mae) and g.avg_mfe == pytest.approx(mean_mfe)
    assert excursion_summary([TradeRow("X", "long", t0, t0, 1.0, 0.0, 1, "signal")], None) is None
    assert group_stats("x", [TradeRow("X", "long", t0, t0, 1.0, 0.0, 1, "signal")]).avg_mae is None


def independent(result, trade):
    """Recompute one trade's excursions straight from the backtest's bars."""
    lows = [b.low for ts, sym, b in result.bars if sym == trade.symbol and trade.opened_at <= ts < trade.closed_at]
    highs = [b.high for ts, sym, b in result.bars if sym == trade.symbol and trade.opened_at <= ts < trade.closed_at]
    worst, best = min(lows + [trade.exit_price]), max(highs + [trade.exit_price])
    if trade.side == "short":
        return min(0, 1 - best / trade.entry_price), max(0, 1 - worst / trade.entry_price)
    return min(0, worst / trade.entry_price - 1), max(0, best / trade.entry_price - 1)


def test_backtests(runs):
    db, longs, shorts, atr = runs
    with SQLiteStore(db, readonly=True) as store:
        a = analyze_trades(store, longs.run_id)
        s = analyze_trades(store, shorts.run_id)
        k = analyze_trades(store, atr.run_id)
    for analysis, result in ((a, longs), (s, shorts)):
        assert len(analysis.rows) == len(result.trades) > 5
        for row, trade in zip(analysis.rows, result.trades):
            assert (row.mae, row.mfe) == pytest.approx(independent(result, trade))
            assert row.mae <= 0 <= row.mfe
            move = (trade.exit_price / trade.entry_price - 1) * (-1 if trade.side == "short" else 1)
            assert row.mae - 1e-12 <= move <= row.mfe + 1e-12  # the exit is inside the excursion range
    assert {r.side for r in s.rows} == {"long", "short"}
    assert a.excursions.stop_loss_pct == PRESETS["trend"].config(BASE).risk.stop_loss_pct
    assert k.excursions.trades == len(atr.trades) > 0 and k.excursions.stop_loss_pct is None  # ATR stops vary
    text = format_trades(a)
    assert "90% of winning trades went at most" in text and "the stop-loss is 5.0%" in text
    assert "e-ratio" in text and " mae " in text


def test_runs_without_candles(tmp_path):
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    with SQLiteStore(tmp_path / "old.db") as store:
        store.create_run("old", kind="backtest", timeframe="1h", symbols=["BTC/USDT"], exchange="test",
                         config=BASE.to_mapping(), config_fingerprint="f")
        store.add_closed_trades("old", [ClosedTrade("BTC/USDT", 1.0, 100.0, 101.0, 100.0, 101.0, 1.0, t0,
                                                    t0 + timedelta(hours=2))])
        a = analyze_trades(store, "old")
    assert a.rows[0].mae is None and a.excursions is None and a.total.avg_mae is None
    assert "Excursions: n/a (no stored candles for this run)" in format_trades(a)


def test_cli(runs, tmp_path, capsys):
    db, longs, _, _ = runs
    csv = tmp_path / "t.csv"
    assert main(["--db", str(db), "trades", longs.run_id, "--json", "--csv", str(csv)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["excursions"]["trades"] == len(longs.trades)
    assert all(t["mae"] <= 0 <= t["mfe"] for t in data["trades"])
    frame = pd.read_csv(csv)
    assert {"mae", "mfe"} <= set(frame.columns) and frame["mae"].max() <= 0


def test_sentences_from_object_or_dict(runs):
    from trading_lab.research.trades import excursion_sentences

    db, longs, _, _ = runs
    with SQLiteStore(db, readonly=True) as store:
        a = analyze_trades(store, longs.run_id)
    sentences = excursion_sentences(a.excursions)
    assert sentences == excursion_sentences(a.excursions.to_dict()) and len(sentences) == 3
    assert excursion_sentences(None) == []


def test_html_report_and_snapshot(runs, tmp_path):
    from trading_lab.dashboard import DashboardData

    db, longs, _, _ = runs
    with DashboardData(db) as data:
        breakdown = data.trade_breakdown(longs.run_id)
        a = analyze_trades(data.store, longs.run_id)
    assert breakdown["excursions"] == pytest.approx(a.excursions.to_dict())
    assert all(g["avg_mae"] <= 0 <= g["avg_mfe"] for g in breakdown["groups"]["exit"])
    page = tmp_path / "r.html"
    assert main(["--db", str(db), "report", longs.run_id, "--html", str(page)]) == 0
    html = page.read_text()
    assert "<th>MAE</th>" in html and "<th>MFE</th>" in html
    assert "90% of winning trades went at most" in html and "the stop-loss is 5.0%" in html


def test_dashboard_shows_them(runs, monkeypatch):
    pytest.importorskip("streamlit")
    from test_dashboard_app import render

    db, _, _, atr = runs  # the latest run is the ATR backtest
    at = render(monkeypatch, db)
    captions = " ".join(c.value for c in at.caption)
    assert "90% of winning trades went at most" in captions and "e-ratio" in captions
    assert "the stop-loss is" not in captions  # ATR stops differ per trade
    table = next(df.value for df in at.dataframe if "avg_mae" in df.value.columns)
    assert (table["avg_mae"] <= 0).all() and (table["avg_mfe"] >= 0).all()
