"""Stage 19C: market regimes in the HTML report, the dashboard data and the dashboard."""

from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.dashboard import DashboardData
from trading_lab.data import SyntheticProvider
from trading_lab.html_report import build_html_report
from trading_lab.research import regimes_for_run
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
START = datetime(2024, 2, 1, tzinfo=UTC)
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


def run(tmp_path, bars):
    db = tmp_path / f"r{bars}.db"
    with SQLiteStore(db) as store:
        run_id = BacktestEngine(CFG, SyntheticProvider(seed=7), store=store).run(START, START + bars * H).run_id
    return db, run_id


def test_long_runs_show_regimes(tmp_path):
    db, run_id = run(tmp_path, 400)
    with DashboardData(db) as data:
        regimes = data.regimes(run_id)
        snap = data.snapshot(run_id)
        with SQLiteStore(db, readonly=True) as store:
            direct = regimes_for_run(store, run_id).to_dict()
    assert regimes["combined"] == direct["combined"]
    assert snap["regimes"]["total_bars"] == 400 and snap["regimes"]["by_trend"]
    page = build_html_report(str(db), run_id)
    assert "<h2>Market regimes</h2>" in page and "<td>up / calm</td>" in page
    assert "warm-up bars are in no regime" in page


def test_short_runs_leave_the_section_out(tmp_path):
    db, run_id = run(tmp_path, 40)  # shorter than the 50-bar trend warm-up
    with DashboardData(db) as data:
        assert data.regimes(run_id) is None and data.snapshot(run_id)["regimes"] is None
    assert "Market regimes" not in build_html_report(str(db), run_id)


def test_the_dashboard_shows_the_table(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from conftest import SRC_DIR

    db, _ = run(tmp_path, 400)
    monkeypatch.setenv("TRADING_LAB_DB", str(db))
    at = AppTest.from_file(str(SRC_DIR / "dashboard" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert any(m.value.startswith("Market regimes") for m in at.markdown)
