"""Stage 21B: Telegram alerts (fake API; the bot token never leaks)."""

import json
import logging

import pytest

from test_alerts import Capture
from trading_lab.alerts import (
    TELEGRAM_CHAT_ENV,
    TELEGRAM_TOKEN_ENV,
    Alert,
    AlertManager,
    MultiNotifier,
    TelegramNotifier,
    WebhookNotifier,
    build_notifier,
)
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.doctor import OK, format_checks, run_checks
from trading_lab.llm import TransportTimeout

TOKEN = "123456:SECRET-bot-token-abc"
ENV = {TELEGRAM_TOKEN_ENV: TOKEN, TELEGRAM_CHAT_ENV: "-1001234567890"}


def test_a_message_to_the_chat():
    transport = Capture()
    n = TelegramNotifier.from_env(ENV, transport=transport)
    n.send(Alert("critical", "Kill switch tripped", "Equity fell 25%.", run_id="vm-paper-1"))
    request = transport.requests[0]
    assert request["url"] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    body = json.loads(request["body"])
    assert body["chat_id"] == "-1001234567890" and body["disable_web_page_preview"] is True
    assert body["text"] == "[!!] trading-lab: Kill switch tripped\nEquity fell 25%.\nRun: vm-paper-1"
    n.send(Alert("info", "Daily summary", "x" * 10_000))
    assert len(json.loads(transport.requests[1]["body"])["text"]) == 4000  # within Telegram's limit


@pytest.mark.parametrize("env, message", [
    ({}, TELEGRAM_TOKEN_ENV),
    ({TELEGRAM_TOKEN_ENV: TOKEN}, TELEGRAM_CHAT_ENV),
    ({**ENV, TELEGRAM_TOKEN_ENV: "abc/../x"}, "bot token"),
    ({**ENV, TELEGRAM_CHAT_ENV: "12 34"}, "chat id"),
])
def test_settings_are_validated(env, message):
    with pytest.raises(ConfigError, match=message):
        TelegramNotifier.from_env(env)


def test_the_token_never_leaks(caplog):
    n = TelegramNotifier.from_env(ENV, transport=Capture(status=401))
    assert TOKEN not in repr(n)
    with pytest.raises(ConnectionError) as info:
        n.send(Alert("warning", "x"))
    assert "HTTP 401" in str(info.value) and TELEGRAM_TOKEN_ENV in str(info.value) and TOKEN not in str(info.value)
    down = TelegramNotifier.from_env(ENV, transport=Capture(error=TransportTimeout(f"timeout for {TOKEN}")))
    with caplog.at_level(logging.DEBUG):
        assert AlertManager(down, min_level="info").emit("warning", "x") is False  # never raises
    assert TOKEN not in caplog.text
    with pytest.raises(ConnectionError, match="chat"):
        TelegramNotifier.from_env(ENV, transport=Capture(status=400)).send(Alert("info", "x"))


def test_channels():
    cfg = AppConfig.from_mapping({"alerts": {"enabled": True, "channels": ["webhook", "telegram"]}})
    notifier = build_notifier(cfg, {**ENV, "TRADING_LAB_ALERT_URL": "https://ntfy.example.test/t"})
    assert isinstance(notifier, MultiNotifier)
    assert [type(n) for n in notifier.notifiers] == [WebhookNotifier, TelegramNotifier]
    only = AppConfig.from_mapping({"alerts": {"enabled": True, "channels": ["telegram"]}})
    assert isinstance(build_notifier(only, ENV), TelegramNotifier)


def test_cli_and_doctor(tmp_path, monkeypatch, capsys):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[alerts]\nenabled = true\nchannels = ["telegram"]\n')
    checks = {c.name: c for c in run_checks(str(cfg_file), env=ENV)}
    assert checks["alerts"].status == OK and "telegram" in checks["alerts"].detail
    assert TOKEN not in format_checks(list(checks.values()))
    transport = Capture()
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    assert main(["--config", str(cfg_file), "alert-test"]) == 0
    out = capsys.readouterr().out
    assert "Sending a test Telegram message" in out and out.rstrip().endswith("OK") and TOKEN not in out
    assert "Test alert" in json.loads(transport.requests[0]["body"])["text"]
