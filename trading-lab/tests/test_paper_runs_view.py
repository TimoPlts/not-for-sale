"""Stage 22C: every paper run at a glance (dashboard, dashboard-data and the snapshot)."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from test_live import ANCHOR, START, Clock
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.dashboard import DashboardData
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.presets import PRESETS
from trading_lab.storage import SQLiteStore

H = timedelta(hours=1)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


@pytest.fixture(scope="module")
def runs_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("runs") / "h.db"
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(db) as store:
        BacktestEngine(BASE, SyntheticProvider(seed=3), store=store).run(
            datetime(2024, 2, 1, tzinfo=timezone.utc), datetime(2024, 2, 3, tzinfo=timezone.utc))
        traders = [LivePaperTrader(cfg, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock,
                                   run_id=name) for name, cfg in (("trend", PRESETS["trend"].config(BASE)),
                                                                  ("default", BASE), ("old", BASE))]
        for _ in range(30):
            for trader in traders:
                trader.run_cycle()
            clock.now += H
        traders[2].stop()
        LivePaperTrader(BASE, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock,
                        run_id="fresh")  # created, no cycle yet
    return db, clock.now - H


def test_paper_runs_rows(runs_db):
    db, now = runs_db
    with DashboardData(db) as data:
        rows = data.paper_runs(now=now)
        assert [r["run_id"] for r in rows] == ["default", "fresh", "trend", "old"]  # running first, by id
        by_id = {r["run_id"]: r for r in rows}
        for run_id in ("default", "trend", "old"):
            o = data.overview(run_id)
            r = by_id[run_id]
            assert r["equity"] == pytest.approx(o["equity"]) and r["total_return"] == pytest.approx(o["total_return"])
            assert r["max_drawdown"] == pytest.approx(o["max_drawdown"])
            assert r["open_positions"] == o["open_positions"] and r["last_bar"] == o["last_bar"]
        assert by_id["default"]["check"] == by_id["trend"]["check"] == "OK"
        assert by_id["old"]["status"] == "stopped" and by_id["old"]["check"] is None  # not checked
        assert by_id["fresh"]["equity"] is None and by_id["fresh"]["last_bar"] is None
        later = {r["run_id"]: r for r in data.paper_runs(now=now + 10 * H)}
        assert later["default"]["check"].startswith("stalled: 10 closed candle(s)") and later["default"]["behind"] == 10
        snap = data.snapshot("default")
        assert [r["run_id"] for r in snap["paper_runs"]] == ["default", "fresh", "trend", "old"]
        json.dumps(snap)  # JSON-safe


def test_dashboard_data_text(runs_db, capsys):
    db, _ = runs_db
    assert main(["--db", str(db), "dashboard-data", "default"]) == 0
    out = capsys.readouterr().out
    assert "Paper runs:" in out and "watchdog: not running" in out and "no bars yet" in out
    assert next(line for line in out.splitlines() if line.strip().startswith("old")).split()[1] == "stopped"


def test_single_run_has_no_overview(tmp_path, capsys):
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(tmp_path / "one.db") as store:
        LivePaperTrader(BASE, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock).run_cycle()
    assert main(["--db", str(tmp_path / "one.db"), "dashboard-data"]) == 0
    assert "Paper runs:" not in capsys.readouterr().out


def test_dashboard_section(monkeypatch, runs_db):
    pytest.importorskip("streamlit")
    from test_dashboard_app import render

    db, _ = runs_db
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    at = render(monkeypatch, db)
    headers = [h.value for h in at.subheader]
    assert headers[0] == "Paper runs" and headers[1] == "Portfolio"
    table = at.dataframe[0].value
    assert list(table["Run"]) == ["default", "fresh", "trend", "old"]
    assert list(table["Status"]) == ["running", "running", "running", "stopped"]
    assert table["Watchdog"].iloc[3] == "not running" and table["Equity"].iloc[1] == "n/a"
    assert "live-compare" in " ".join(c.value for c in at.caption)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before  # read-only
