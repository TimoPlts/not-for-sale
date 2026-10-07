"""Stage 23B: the weekly digest of every paper run."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from test_alerts import SECRET_URL, Capture
from test_live import ANCHOR, START, Clock
from trading_lab import digest as digest_module
from trading_lab.alerts import URL_ENV
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.digest import build_digest, format_digest
from trading_lab.live import LivePaperTrader
from trading_lab.presets import PRESETS
from trading_lab.research import live_compare
from trading_lab.storage import SQLiteStore
from trading_lab.summary import build_summary

H = timedelta(hours=1)
DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


def trader(store, clock, run_id, cfg=BASE):
    return LivePaperTrader(cfg, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock,
                           run_id=run_id)


@pytest.fixture(scope="module")
def week_db(tmp_path_factory):
    db = tmp_path_factory.mktemp("digest") / "h.db"
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(db) as store:
        old = trader(store, clock, "old")  # stops before the digest period
        for _ in range(24):
            old.run_cycle()
            clock.now += H
        old.stop()
        default, trend = trader(store, clock, "default"), trader(store, clock, "trend", PRESETS["trend"].config(BASE))
        late = None
        for i in range(24 * 9):
            if i == 24 * 6:
                late = trader(store, clock, "late")  # starts and stops inside the period
            for t in (default, trend, late):
                if t is not None:
                    t.run_cycle()
            if i == 24 * 8:
                late.stop()
                late = None
            clock.now += H
    return db, clock.now - H


def test_digest_contents(week_db):
    db, now = week_db
    with SQLiteStore(db, readonly=True) as store:
        d = build_digest(store, days=7, now=now)
        assert [s.run_id for s in d.runs] == ["default", "late", "trend"]  # "old" stopped before the period
        assert d.checks == {"default": "OK", "late": None, "trend": "OK"} and d.ok
        for s in d.runs:
            expected = build_summary(store, s.run_id, hours=7 * 24, now=now)
            assert s.equity_change_pct == pytest.approx(expected.equity_change_pct)
            assert len(s.trades) == len(expected.trades)
        assert len(d.comparisons) == 1
        assert d.comparisons[0].verdict == live_compare(store, "default", "trend").verdict
        later = build_digest(store, days=7, now=now + 10 * H)
        assert later.checks["default"].startswith("stalled") and not later.ok
    text = format_digest(d)
    assert "default (running)" in text and "late (stopped)" in text and "old" not in text
    assert "watchdog: OK" in text and "watchdog: not running" in text
    assert "default vs trend [" in text and "Paper trading only" in text
    data = d.to_dict()
    assert [r["run_id"] for r in data["runs"]] == ["default", "late", "trend"]
    assert data["runs"][0]["period_return"] == pytest.approx(d.runs[0].equity_change_pct)
    json.dumps(data, default=str)


def test_pairs_are_capped(week_db, monkeypatch):
    db, now = week_db
    monkeypatch.setattr(digest_module, "MAX_PAIRS", 0)
    with SQLiteStore(db, readonly=True) as store:
        d = build_digest(store, days=7, now=now)
    assert d.comparisons == [] and d.skipped_pairs == 1
    with SQLiteStore(db, readonly=True) as store:
        with pytest.raises(ValueError):
            build_digest(store, days=0, now=now)


def test_no_paper_runs(tmp_path):
    with SQLiteStore(tmp_path / "e.db") as store:
        d = build_digest(store)
    assert d.runs == [] and "No paper run traded in this period." in format_digest(d)


def test_cli_and_alert(week_db, tmp_path, monkeypatch, capsys):
    db, _ = week_db
    transport = Capture()
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
    monkeypatch.setenv(URL_ENV, SECRET_URL)
    cfg = tmp_path / "c.toml"
    cfg.write_text('[alerts]\nenabled = true\nformat = "json"\nmin_level = "critical"\n')
    assert main(["--config", str(cfg), "--db", str(db), "digest", "--alert"]) == 0  # sent despite min_level
    out = capsys.readouterr().out
    assert "default (running)" in out and SECRET_URL not in out
    sent = json.loads(transport.requests[0]["body"])
    assert sent["title"] == "Paper trading digest (7 days)" and sent["body"].startswith("Paper trading, ")
    assert "\ndefault (running)" in sent["body"]
    assert main(["--db", str(db), "digest", "--days", "30", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {r["run_id"] for r in data["runs"]} == {"default", "trend"}  # real clock: the others ended long ago
    assert main(["--db", str(tmp_path / "none.db"), "digest"]) == 1
    assert main(["--db", str(db), "digest", "--days", "0"]) == 1


def test_systemd_units():
    service = (DEPLOY / "trading-lab-digest.service").read_text()
    assert "digest --days 7 --alert" in service and "ReadOnlyPaths=/opt/trading-lab/trading-lab" in service
    assert "Type=oneshot" in service and "QWEN_API_KEY" not in service
    timer = (DEPLOY / "trading-lab-digest.timer").read_text()
    assert "OnCalendar=Mon *-*-* 07:52:00 UTC" in timer and "Persistent=true" in timer
