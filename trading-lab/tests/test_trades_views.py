"""Stage 24B: the trade breakdown in the dashboard, the HTML report, the snapshot and dashboard-data."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.dashboard import DashboardData
from trading_lab.data import SyntheticProvider
from trading_lab.presets import PRESETS
from trading_lab.research.trades import analyze_trades
from trading_lab.storage import SQLiteStore

START = datetime(2024, 2, 1, tzinfo=timezone.utc)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


@pytest.fixture(scope="module")
def runs_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("views") / "h.db"
    with SQLiteStore(db) as store:
        quiet = BacktestEngine(BASE.with_overrides({"voting": {"min_agreeing": 3}}), SyntheticProvider(seed=2),
                               store=store).run(START, START + timedelta(days=1))
        trend = BacktestEngine(PRESETS["trend"].config(BASE), SyntheticProvider(seed=2), store=store).run(
            START, START + timedelta(days=30))
    return db, trend.run_id, quiet.run_id


def test_data_and_snapshot(runs_db):
    db, trend, quiet = runs_db
    with DashboardData(db) as data:
        b = data.trade_breakdown(trend)
        expected = analyze_trades(data.store, trend, groupings=("exit", "side", "holding", "symbol"))
        assert set(b["groups"]) == {"exit", "side", "holding", "symbol"} and "trades" not in b
        assert b["total"]["trades"] == len(expected.rows)
        assert [g["name"] for g in b["groups"]["exit"]] == [g.name for g in expected.groups["exit"]]
        assert b["observations"] == expected.observations
        assert data.trade_breakdown(quiet) is None
        snap = data.snapshot(trend)
        assert snap["trade_breakdown"] == b
        json.dumps(snap)


def test_html_report(runs_db, tmp_path):
    db, trend, quiet = runs_db
    page = tmp_path / "r.html"
    assert main(["--db", str(db), "report", trend, "--html", str(page)]) == 0
    html = page.read_text()
    assert "<h2>Where the money comes from</h2>" in html and "Holding time" in html and "time stop" in html
    assert main(["--db", str(db), "report", quiet, "--html", str(page)]) == 0
    assert "Where the money comes from" not in page.read_text()


def test_dashboard_data_text(runs_db, capsys):
    db, trend, _ = runs_db
    assert main(["--db", str(db), "dashboard-data", trend]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("Trades by exit: "))
    assert "signal" in line and "time stop" in line


def test_dashboard(runs_db, monkeypatch):
    pytest.importorskip("streamlit")
    from test_dashboard_app import render

    db, trend, _ = runs_db
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    at = render(monkeypatch, db)  # the latest run is the trend backtest
    assert "Research" in [h.value for h in at.subheader]
    markdown = [m.value for m in at.markdown]
    assert "Trades by exit type" in markdown and "Trades by holding time" in markdown
    exit_table = next(df.value for df in at.dataframe if "avg_bars" in df.value.columns and
                      "time stop" in set(df.value["name"]))
    with SQLiteStore(db, readonly=True) as store:
        assert exit_table["trades"].sum() == analyze_trades(store, trend).total.trades
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
