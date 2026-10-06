"""Stage 13A: ``trading-lab doctor`` readiness checks (read-only, no secrets shown)."""

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from test_smoke import SmokeTransport
from test_specialists import ENV as QWEN_ENV
from trading_lab.alerts import URL_ENV
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.errors import DataError
from trading_lab.data import SyntheticProvider
from trading_lab.doctor import FAIL, OK, SKIP, WARN, format_checks, run_checks
from trading_lab.storage import SCHEMA_VERSION, SQLiteStore

UTC = timezone.utc
NOW = datetime(2024, 3, 1, 12, 5, tzinfo=UTC)
AGENT_CFG = "[strategies.rsi]\n[strategies.qwen_trend]\nweight = 1.0\n"


def by_name(checks):
    return {c.name: c for c in checks}


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in (*QWEN_ENV, URL_ENV):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def write(path, text):
    path.write_text(text)
    return str(path)


def test_fresh_install_is_ready(workdir):
    checks = run_checks(None, db_path=str(workdir / "data" / "h.db"))
    c = by_name(checks)
    assert c["python"].status == OK and c["package pandas"].status == OK and c["config"].status == OK
    assert c["model"].status == SKIP and c["alerts"].status == SKIP
    assert c["database"].status == OK and "will be created" in c["database"].detail
    assert not any(x.status == FAIL for x in checks)
    assert format_checks(checks).splitlines()[-1].startswith("ready")
    assert list(workdir.iterdir()) == []  # nothing was created


def test_agents_need_their_environment_and_secrets_stay_hidden(workdir):
    cfg = write(workdir / "c.toml", AGENT_CFG)
    c = by_name(run_checks(cfg, db_path=str(workdir / "h.db"), env={}))
    assert c["model"].status == FAIL
    assert all(f"{v} missing" in c["model"].detail for v in QWEN_ENV)
    checks = run_checks(cfg, db_path=str(workdir / "h.db"), env=QWEN_ENV)
    c = by_name(checks)
    assert c["model"].status == OK and "QWEN_API_KEY set" in c["model"].detail
    assert QWEN_ENV["QWEN_API_KEY"] not in format_checks(checks)
    assert c["answer cache"].status == OK and "created on the first answer" in c["answer cache"].detail


def test_replay_needs_only_the_model_name_but_recorded_answers(workdir):
    cfg = write(workdir / "c.toml", AGENT_CFG + '[agents]\nmode = "replay"\n')
    c = by_name(run_checks(cfg, db_path=str(workdir / "h.db"), env={"QWEN_MODEL": "m"}))
    assert c["model"].status == OK and "QWEN_API_KEY" not in c["model"].detail
    assert c["answer cache"].status == FAIL and "replay mode needs recorded answers" in c["answer cache"].detail


def test_alerts_need_their_url(workdir):
    cfg = write(workdir / "c.toml", '[alerts]\nenabled = true\nformat = "slack"\n')
    assert by_name(run_checks(cfg, env={}))["alerts"].status == FAIL
    secret = "https://hooks.example.test/T000/B000/very-secret"
    checks = run_checks(cfg, env={URL_ENV: secret})
    assert by_name(checks)["alerts"].status == OK and secret not in format_checks(checks)


def test_invalid_config_stops_early(workdir):
    checks = run_checks(write(workdir / "c.toml", "[risk]\nstop_loss_pct = 7\n"))
    assert checks[-1].name == "config" and checks[-1].status == FAIL and "stop_loss_pct" in checks[-1].detail


def test_existing_database(workdir):
    db = workdir / "h.db"
    with SQLiteStore(db) as store:
        cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
        BacktestEngine(cfg, SyntheticProvider(seed=1), store=store).run(NOW - timedelta(days=2), NOW)
        store.create_run("vm-paper-1", kind="paper", timeframe="1h", symbols=["BTC/USDT"], exchange="x",
                         config=cfg.to_dict(), config_fingerprint=cfg.fingerprint())
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    c = by_name(run_checks(None, db_path=str(db)))
    assert c["database"].status == OK and "2 run(s)" in c["database"].detail
    assert "running paper run(s): vm-paper-1" in c["database"].detail
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    assert by_name(run_checks(None, db_path=str(db)))["database"].status == FAIL


def test_online_checks(workdir, monkeypatch):
    fresh = run_checks(None, online=True, now=NOW,
                       market_factory=lambda cfg: SyntheticProvider(seed=1, clock=lambda: NOW))
    assert by_name(fresh)["market data"].status == OK and by_name(fresh)["model call"].status == SKIP
    stale = run_checks(None, online=True, now=NOW + timedelta(hours=5),
                       market_factory=lambda cfg: SyntheticProvider(seed=1, clock=lambda: NOW))
    assert by_name(stale)["market data"].status == WARN and "stale" in by_name(stale)["market data"].detail

    class Down(SyntheticProvider):
        def fetch_ohlcv(self, *args, **kwargs):
            raise DataError("exchange unreachable")

    down = run_checks(None, online=True, now=NOW, market_factory=lambda cfg: Down(seed=1))
    assert by_name(down)["market data"].status == FAIL and "unreachable" in by_name(down)["market data"].detail

    for var, value in QWEN_ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", SmokeTransport().post)
    cfg = write(workdir / "c.toml", AGENT_CFG)
    checks = run_checks(cfg, online=True, now=NOW,
                        market_factory=lambda c: SyntheticProvider(seed=1, clock=lambda: NOW))
    assert by_name(checks)["model call"].status == OK and "qwen-test answered" in by_name(checks)["model call"].detail


def test_cli_exit_codes(workdir, capsys):
    assert main(["--db", str(workdir / "h.db"), "doctor"]) == 0
    assert "ready" in capsys.readouterr().out
    write(workdir / "c.toml", AGENT_CFG)
    assert main(["--config", str(workdir / "c.toml"), "doctor"]) == 1
    assert "NOT READY" in capsys.readouterr().out
