"""Stage 11C: alerts (webhook notifications) and resuming runs saved by older versions."""

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from test_failure_recovery import FlakyMarket, Seconds, Switchable
from test_live import ANCHOR, START, Clock
from test_specialists import ENV as QWEN_ENV
from trading_lab.alerts import URL_ENV, Alert, AlertManager, MemoryNotifier, WebhookNotifier, build_alerts
from trading_lab.cli import main
from trading_lab.config import AlertsConfig, AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.llm import QwenProvider, TransportTimeout
from trading_lab.storage import SQLiteStore

H = timedelta(hours=1)
SECRET_URL = "https://ntfy.example.test/very-secret-topic-123"


class Capture:
    def __init__(self, status=200, error=None):
        self.requests, self.status, self.error = [], status, error

    def post(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "body": body})
        if self.error:
            raise self.error
        return self.status, b"ok"


# --------------------------------------------------------------- notifier
@pytest.mark.parametrize("fmt, check", [
    ("ntfy", lambda r: r["headers"]["Priority"] == "urgent" and r["body"] == "kill switch body".encode()
     and r["headers"]["Title"] == "trading-lab: Kill switch tripped"),
    ("slack", lambda r: json.loads(r["body"])["text"].startswith("[!!] *trading-lab: Kill switch tripped*")),
    ("discord", lambda r: "**trading-lab: Kill switch tripped**" in json.loads(r["body"])["content"]),
    ("json", lambda r: json.loads(r["body"])["level"] == "critical" and json.loads(r["body"])["run_id"] == "pp-1"),
])
def test_webhook_formats(fmt, check):
    transport = Capture()
    WebhookNotifier(SECRET_URL, fmt, transport=transport).send(
        Alert("critical", "Kill switch tripped", "kill switch body", "pp-1"))
    assert transport.requests[0]["url"] == SECRET_URL and check(transport.requests[0])


def test_webhook_errors_never_reveal_the_url(caplog):
    for transport in (Capture(status=500), Capture(error=TransportTimeout("slow"))):
        notifier = WebhookNotifier(SECRET_URL, transport=transport)
        with pytest.raises(ConnectionError) as exc:
            notifier.send(Alert("warning", "x"))
        assert "ntfy.example.test" in str(exc.value) and "secret" not in str(exc.value)
    manager = AlertManager(WebhookNotifier(SECRET_URL, transport=Capture(status=403)))
    caplog.set_level(logging.WARNING)
    assert manager.emit("critical", "x") is False and manager.failed == 1
    assert "secret" not in caplog.text and "secret" not in repr(manager.notifier)


def test_configuration_and_environment():
    with pytest.raises(ConfigError, match="http"):
        WebhookNotifier("ftp://x/y")
    with pytest.raises(ConfigError, match="format"):
        WebhookNotifier(SECRET_URL, "telegram")
    with pytest.raises(ConfigError, match=URL_ENV):
        WebhookNotifier.from_env("ntfy", env={})
    assert build_alerts(AppConfig()) is None  # disabled by default
    enabled = AppConfig.from_mapping({"alerts": {"enabled": True}})
    with pytest.raises(ConfigError, match=URL_ENV):
        build_alerts(enabled, env={})
    assert isinstance(build_alerts(enabled, env={URL_ENV: SECRET_URL}), AlertManager)
    for bad in ({"url": SECRET_URL}, {"webhook": "x"}):
        with pytest.raises(ConfigError, match="unknown key"):
            AppConfig.from_mapping({"alerts": bad})
    for bad in ({"format": "sms"}, {"min_level": "debug"}, {"outage_after_cycles": 0}):
        with pytest.raises(ConfigError):
            AlertsConfig(**bad)


def test_manager_levels_repeats_and_failures():
    clock, sink = Seconds(), MemoryNotifier()
    manager = AlertManager(sink, min_level="warning", repeat_after_seconds=600, clock=clock)
    assert manager.emit("info", "entry") is False
    assert manager.emit("info", "summary", force=True) is True
    assert manager.emit("warning", "outage", key="outage") is True
    assert manager.emit("warning", "outage", key="outage") is False  # repeated too soon
    clock.t += 601
    assert manager.emit("warning", "outage", key="outage") is True
    manager.clear("outage")
    assert manager.emit("critical", "outage", key="outage") is True
    assert [a.title for a in sink.sent] == ["summary", "outage", "outage", "outage"]

    class Broken:
        def send(self, alert):
            raise RuntimeError("boom")

    broken = AlertManager(Broken())
    assert broken.emit("critical", "x") is False and broken.failed == 1


# ------------------------------------------------------------ live trader
def trader(store, clock, market=None, *, alerts_cfg=None, risk=None, sink=None, min_level="info", **kwargs):
    cfg = AppConfig.from_mapping({
        "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
        "risk": risk or {},
        "alerts": {"enabled": True, **(alerts_cfg or {})},
    })
    manager = AlertManager(sink or MemoryNotifier(), min_level=min_level)
    market = market or SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock)
    return LivePaperTrader(cfg, market, store, clock=clock, alerts=manager, **kwargs)


def drive(t, clock, n):
    for _ in range(n):
        t.run_cycle()
        clock.now += H


def titles(t):
    return [a.title for a in t.alerts.notifier.sent]


def test_breaker_trips_and_trades_are_reported():
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        t = trader(store, clock, risk={"max_drawdown_pct": 0.002})
        drive(t, clock, 60)
    sent = t.alerts.notifier.sent
    kill = [a for a in sent if a.title == "Kill switch tripped"]
    assert len(kill) == 1 and kill[0].level == "critical" and "New entries are blocked" in kill[0].body
    assert any(a.title.startswith("enter ") for a in sent) and all(a.run_id == t.run_id for a in sent)


def test_info_alerts_are_filtered_at_warning_level():
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        t = trader(store, clock, min_level="warning", alerts_cfg={"daily_summary": False})
        drive(t, clock, 30)
    assert titles(t) == []


def test_outage_alert_once_then_recovery():
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        market = FlakyMarket(5, seed=11, anchor=ANCHOR, clock=clock)
        t = trader(store, clock, market, alerts_cfg={"outage_after_cycles": 3, "daily_summary": False},
                   min_level="warning")
        for _ in range(7):
            t.run_cycle()
    assert titles(t) == ["3 trading cycles failed in a row", "Trading cycles processed again"]


def test_model_pause_and_recovery_alerts(tmp_path):
    clock, model_clock = Clock(START + H + timedelta(minutes=1)), Seconds()
    endpoint = Switchable()
    endpoint.down = True
    qwen = QwenProvider(env=QWEN_ENV, transport=endpoint, sleep=lambda s: None, clock=model_clock,
                        max_retries=0, failure_threshold=2, failure_cooldown_seconds=60)
    cfg = AppConfig.from_mapping({
        "market": {"symbols": ["BTC/USDT"]},
        "strategies": {"rsi": {}, "qwen_momentum": {"weight": 1.0, "decision_interval": 1, "lookback": 10}},
        "agents": {"mode": "live", "cache_path": str(tmp_path / "unused.db")},  # every answer is a real call
        "alerts": {"enabled": True, "daily_summary": False},
    })
    sink = MemoryNotifier()
    with SQLiteStore(":memory:") as store:
        t = LivePaperTrader(cfg, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock,
                            llm_provider=qwen, alerts=AlertManager(sink, min_level="warning"))
        drive(t, clock, 5)
        endpoint.down = False
        model_clock.t += 61
        drive(t, clock, 2)
    assert [a.title for a in sink.sent] == ["Model calls paused", "Model answers again"]
    assert "ProviderUnavailableError" in sink.sent[0].body


def test_daily_summary_once_per_day_and_across_resume(tmp_path):
    clock = Clock(datetime(2024, 3, 1, 20, 1, tzinfo=timezone.utc))
    with SQLiteStore(tmp_path / "h.db") as store:
        t = trader(store, clock, min_level="critical")
        drive(t, clock, 6)  # crosses midnight UTC once
        assert titles(t) == ["Daily summary"]
        body = t.alerts.notifier.sent[0].body
        assert body.startswith(f"**trading-lab paper run {t.run_id}**") and "Equity" in body
        t.stop()
        resumed = LivePaperTrader.resume(store, t.run_id, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock),
                                         clock=clock, alerts=AlertManager(MemoryNotifier(), min_level="critical"))
        drive(resumed, clock, 3)  # still the same day: no second summary
        assert resumed.alerts.notifier.sent == []
        assert store.load_state(t.run_id)["alerts"]["last_summary_day"] == "2024-03-02"


def test_failing_alerts_never_affect_trading(tmp_path):
    class Broken:
        def send(self, alert):
            raise ConnectionError("webhook down")

    results = []
    for sink in (Broken(), MemoryNotifier()):
        clock = Clock(START + H + timedelta(minutes=1))
        with SQLiteStore(":memory:") as store:
            t = trader(store, clock, sink=sink)
            drive(t, clock, 40)
            results.append([(f.timestamp, f.symbol, f.side, f.quantity) for f in t.portfolio.fills])
    assert results[0] == results[1] and results[0]


# ------------------------------------------------------------------- CLI
def test_alert_test_command(monkeypatch, capsys):
    transport = Capture()
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
    monkeypatch.setenv(URL_ENV, SECRET_URL)
    assert main(["alert-test", "--format", "slack"]) == 0
    out = capsys.readouterr().out
    assert "ntfy.example.test" in out and "secret" not in out and out.rstrip().endswith("OK")
    assert "Test alert" in json.loads(transport.requests[0]["body"])["text"]
    monkeypatch.delenv(URL_ENV)
    assert main(["alert-test"]) == 1 and URL_ENV in capsys.readouterr().err


def test_paper_command_reports_start_and_stop(tmp_path, monkeypatch, capsys):
    transport = Capture()
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
    monkeypatch.setenv(URL_ENV, SECRET_URL)
    cfg = tmp_path / "c.toml"
    cfg.write_text('[market]\nsymbols = ["BTC/USDT"]\n[alerts]\nenabled = true\nmin_level = "info"\nformat = "json"\n')
    assert main(["--config", str(cfg), "--db", str(tmp_path / "h.db"), "paper", "--synthetic", "1", "--once",
                 "--run-id", "alerted"]) == 0
    sent = [json.loads(r["body"])["title"] for r in transport.requests]
    assert sent[0] == "Paper run alerted running" and sent[-1] == "Paper run alerted stopped"
    monkeypatch.delenv(URL_ENV)
    assert main(["--config", str(cfg), "--db", str(tmp_path / "h.db"), "paper", "--synthetic", "1", "--once"]) == 1
    assert URL_ENV in capsys.readouterr().err  # fails before any trading starts


# ------------------------------------------------- older runs still resume
def test_runs_saved_by_older_versions_can_be_resumed(tmp_path):
    clock = Clock(START + H + timedelta(minutes=1))
    db = tmp_path / "h.db"
    with SQLiteStore(db) as store:
        cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
        t = LivePaperTrader(cfg, SyntheticProvider(seed=1, anchor=ANCHOR, clock=clock), store, clock=clock)
        drive(t, clock, 5)
        t.stop()
        old = cfg.to_dict()  # what an older version stored: without settings added since
        del old["alerts"]
        for key in ("trailing_stop_pct", "trailing_activation_pct", "take_profit_pct"):
            del old["risk"][key]
        for key in ("provider", "request_timeout_seconds", "max_retries", "retry_backoff_seconds", "temperature",
                    "max_output_tokens", "failure_threshold", "failure_cooldown_seconds"):
            del old["agents"][key]
        old_fingerprint = "0" * 64  # whatever the older version computed
        store._conn.execute("UPDATE runs SET config_json = ?, config_fingerprint = ? WHERE run_id = ?",
                            (json.dumps(old), old_fingerprint, t.run_id))
        store._conn.commit()
        resumed = LivePaperTrader.resume(store, t.run_id, SyntheticProvider(seed=1, anchor=ANCHOR, clock=clock),
                                         clock=clock)
        assert resumed.run_cycle().error is None
        changed = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}, "risk": {"stop_loss_pct": 0.1}})
        with pytest.raises(ValueError, match="config differs"):
            LivePaperTrader(changed, SyntheticProvider(seed=1, anchor=ANCHOR, clock=clock), store, clock=clock,
                            run_id=t.run_id)
