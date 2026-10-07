"""Stage 18B: the paper-run watchdog (``trading-lab status``)."""

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from test_alerts import SECRET_URL, Capture
from test_failure_recovery import FlakyMarket
from test_live import ANCHOR, START, Clock
from trading_lab.alerts import URL_ENV
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.status import check_status, format_status, latest_paper_run
from trading_lab.storage import SQLiteStore

H = timedelta(hours=1)
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "systemd"


def paper(store, cycles, clock=None, market=None, run_id=None):
    clock = clock or Clock(START + H + timedelta(minutes=1))
    trader = LivePaperTrader(CFG, market or SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store,
                             clock=clock, run_id=run_id)
    for _ in range(cycles):
        trader.run_cycle()
        clock.now += H
    clock.now -= H  # back to the moment of the last cycle
    return trader, clock


def test_a_live_run_keeping_up_is_ok(tmp_path):
    with SQLiteStore(tmp_path / "s.db") as store:
        trader, clock = paper(store, 10)
        s = check_status(store, trader.run_id, now=clock.now)
        assert s.ok and s.behind == 0 and s.status == "running" and s.equity > 0
        assert s.last_processed == START + 9 * H and s.consecutive_errors == 0
        late = check_status(store, trader.run_id, now=clock.now + H + timedelta(minutes=30))
        assert late.ok and late.behind == 1  # one candle closed and not handled yet: normal
        text = format_status(s)
        assert text.endswith("OK") and "0 closed candle(s) waiting" in text


def test_a_stalled_run_is_reported(tmp_path):
    with SQLiteStore(tmp_path / "s.db") as store:
        trader, clock = paper(store, 10)
        s = check_status(store, trader.run_id, now=clock.now + 5 * H)
        assert not s.ok and s.behind == 5 and s.problems[0].startswith("stalled: 5 closed candle(s)")
        assert check_status(store, trader.run_id, now=clock.now + 5 * H, max_behind=5).ok
        assert format_status(s).endswith("NOT OK")


def test_stopped_runs(tmp_path):
    with SQLiteStore(tmp_path / "s.db") as store:
        trader, clock = paper(store, 3)
        trader.stop()
        s = check_status(store, trader.run_id, now=clock.now + 10 * H)
        assert s.problems == ["the run is 'stopped', not running"]  # not "stalled": it was stopped
        relaxed = check_status(store, trader.run_id, now=clock.now + 10 * H, expect_running=False)
        assert relaxed.ok and relaxed.notes == ["the run is 'stopped', not running"]


def test_failing_cycles(tmp_path):
    with SQLiteStore(tmp_path / "s.db") as store:
        clock = Clock(START + H + timedelta(minutes=1))
        market = FlakyMarket(0, seed=11, anchor=ANCHOR, clock=clock)
        trader, clock = paper(store, 3, clock, market)
        market.failures = 10
        for n in range(1, 4):
            trader.run_cycle()
            s = check_status(store, trader.run_id, now=clock.now)
            assert s.consecutive_errors == n and "exchange unreachable" in s.last_error
            assert s.ok == (n < 3)
        assert s.problems == ["3 cycles failed in a row: market data unavailable: exchange unreachable"]


def test_which_run_and_what_kind(tmp_path):
    with SQLiteStore(tmp_path / "s.db") as store:
        assert latest_paper_run(store) is None
        first, _ = paper(store, 2, run_id="first")
        second, _ = paper(store, 2, run_id="second")
        second.stop()
        assert latest_paper_run(store) == "first"  # the running one, even if older
        first.stop()
        assert latest_paper_run(store) == "second"
        bt = BacktestEngine(CFG, SyntheticProvider(seed=1), store=store).run(START, START + 24 * H)
        with pytest.raises(ValueError, match="checks paper runs"):
            check_status(store, bt.run_id)
        with pytest.raises(ValueError, match="unknown run"):
            check_status(store, "nope")


def test_cli_and_alert(tmp_path, monkeypatch, capsys):
    db = tmp_path / "s.db"
    with SQLiteStore(db) as store:
        trader, _ = paper(store, 4)  # its last candle is long closed by the real clock: stalled
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    assert main(["--db", str(db), "status"]) == 1
    assert "PROBLEM: stalled" in capsys.readouterr().out
    assert main(["--db", str(db), "status", trader.run_id, "--json", "--max-behind", "1000000"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True

    transport = Capture()
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
    monkeypatch.setenv(URL_ENV, SECRET_URL)
    config = tmp_path / "c.toml"
    config.write_text('[alerts]\nenabled = true\nformat = "json"\n')
    assert main(["--config", str(config), "--db", str(db), "status", "--alert"]) == 1
    sent = json.loads(transport.requests[0]["body"])
    assert sent["level"] == "critical" and sent["run_id"] == trader.run_id and "stalled" in sent["body"]
    assert "secret" not in capsys.readouterr().out
    assert main(["--db", str(db), "status", "--alert"]) == 1  # alerts disabled: says so, still exit 1
    assert "alerts are disabled" in capsys.readouterr().err
    assert main(["--db", str(tmp_path / "none.db"), "status"]) == 1
    assert hashlib.sha256(db.read_bytes()).hexdigest() == digest  # read-only


def test_watchdog_templates():
    service = (DEPLOY / "trading-lab-watchdog.service").read_text()
    timer = (DEPLOY / "trading-lab-watchdog.timer").read_text()
    assert "Type=oneshot" in service and "status vm-paper-1 --alert" in service
    assert "ReadOnlyPaths=/opt/trading-lab/trading-lab" in service and "ReadWritePaths" not in service
    assert "NoNewPrivileges=true" in service and "QWEN_API_KEY" not in service
    assert "OnUnitActiveSec=15min" in timer and "WantedBy=timers.target" in timer
