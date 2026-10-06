"""Stage 15B: e-mail alerts over SMTP (a fake SMTP server; nothing is sent)."""

import logging
import smtplib

import pytest

from trading_lab.alerts import (
    EMAIL_FROM_ENV,
    EMAIL_TO_ENV,
    SMTP_HOST_ENV,
    SMTP_PASS_ENV,
    SMTP_PORT_ENV,
    SMTP_SECURITY_ENV,
    SMTP_USER_ENV,
    URL_ENV,
    Alert,
    AlertManager,
    EmailNotifier,
    MemoryNotifier,
    MultiNotifier,
    WebhookNotifier,
    build_alerts,
    build_notifier,
)
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.doctor import FAIL, OK, format_checks, run_checks

LOGIN = "smtp-login-value-4711"
ENV = {SMTP_HOST_ENV: "smtp.example.test", SMTP_USER_ENV: "bot@example.test", SMTP_PASS_ENV: LOGIN,
       EMAIL_TO_ENV: "me@example.test, partner@example.test"}
EMAIL_CFG = AppConfig.from_mapping({"alerts": {"enabled": True, "channels": ["email"]}})


class FakeSMTP:
    instances = []
    fail_with = None

    def __init__(self, host, port, timeout=None, context=None):
        self.host, self.port, self.timeout, self.context = host, port, timeout, context
        self.calls, self.sent = [], []
        FakeSMTP.instances.append(self)
        if isinstance(FakeSMTP.fail_with, OSError):
            raise FakeSMTP.fail_with

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.calls.append("quit")

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, pw):
        self.calls.append(("login", user))
        if isinstance(FakeSMTP.fail_with, smtplib.SMTPException):
            raise FakeSMTP.fail_with

    def send_message(self, msg):
        self.calls.append("send")
        self.sent.append(msg)


@pytest.fixture(autouse=True)
def fake_smtp(monkeypatch):
    FakeSMTP.instances, FakeSMTP.fail_with = [], None
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    yield FakeSMTP


def test_an_alert_becomes_an_encrypted_email():
    n = EmailNotifier.from_env(ENV)
    assert (n.host, n.port, n.security, n.sender) == ("smtp.example.test", 587, "starttls", "bot@example.test")
    n.send(Alert("warning", "Kill switch", "Equity fell 20%.", run_id="vm-paper-1"))
    smtp = FakeSMTP.instances[0]
    assert smtp.calls == ["starttls", ("login", "bot@example.test"), "send", "quit"]  # encrypted before login
    msg = smtp.sent[0]
    assert msg["Subject"] == "[trading-lab] WARNING: Kill switch"
    assert msg["To"] == "me@example.test, partner@example.test" and msg["From"] == "bot@example.test"
    body = msg.get_content()
    assert "Equity fell 20%." in body and "Run: vm-paper-1" in body and "no real order" in body


def test_ssl_and_local_relay():
    n = EmailNotifier.from_env({**ENV, SMTP_SECURITY_ENV: "ssl"})
    n.send(Alert("info", "Hello"))
    smtp = FakeSMTP.instances[-1]
    assert smtp.port == 465 and smtp.context is not None and "starttls" not in smtp.calls
    relay = EmailNotifier.from_env({SMTP_HOST_ENV: "localhost", SMTP_SECURITY_ENV: "none",
                                    EMAIL_TO_ENV: "me@example.test", EMAIL_FROM_ENV: "bot@localhost.test"})
    relay.send(Alert("info", "Hello"))
    assert FakeSMTP.instances[-1].port == 25 and FakeSMTP.instances[-1].calls == ["send", "quit"]


@pytest.mark.parametrize("env, message", [
    ({**ENV, SMTP_SECURITY_ENV: "none"}, "never sent unencrypted"),  # a login without encryption
    ({SMTP_HOST_ENV: "smtp.example.test", SMTP_SECURITY_ENV: "none", EMAIL_TO_ENV: "a@b.test",
      EMAIL_FROM_ENV: "c@d.test"}, "local relay"),
    ({**ENV, SMTP_SECURITY_ENV: "tls13"}, SMTP_SECURITY_ENV),
    ({k: v for k, v in ENV.items() if k != SMTP_HOST_ENV}, SMTP_HOST_ENV),
    ({k: v for k, v in ENV.items() if k != EMAIL_TO_ENV}, EMAIL_TO_ENV),
    ({k: v for k, v in ENV.items() if k != SMTP_PASS_ENV}, SMTP_PASS_ENV),
    ({**ENV, SMTP_PORT_ENV: "smtp"}, SMTP_PORT_ENV),
    ({**ENV, SMTP_PORT_ENV: "70000"}, SMTP_PORT_ENV),
    ({**ENV, EMAIL_TO_ENV: "me@example.test\nBcc: x@evil.test"}, EMAIL_TO_ENV),
    ({**ENV, SMTP_USER_ENV: "bot"}, EMAIL_FROM_ENV),  # no sender address to use
])
def test_settings_are_validated(env, message):
    with pytest.raises(ConfigError, match=message):
        EmailNotifier.from_env(env)


def test_the_login_is_never_shown(caplog):
    n = EmailNotifier.from_env(ENV)
    assert LOGIN not in repr(n)
    FakeSMTP.fail_with = smtplib.SMTPAuthenticationError(535, b"authentication failed")
    with pytest.raises(ConnectionError) as info:
        n.send(Alert("critical", "Crash"))
    assert "SMTPAuthenticationError" in str(info.value) and LOGIN not in str(info.value)
    manager = AlertManager(n, min_level="info")
    with caplog.at_level(logging.DEBUG):
        assert manager.emit("critical", "Crash") is False and manager.failed == 1  # never raises
    assert LOGIN not in caplog.text


def test_an_unreachable_server_never_raises_from_the_manager():
    FakeSMTP.fail_with = ConnectionRefusedError("refused")
    manager = AlertManager(EmailNotifier.from_env(ENV), min_level="info")
    assert manager.emit("warning", "Outage") is False and manager.failed == 1


def test_titles_cannot_inject_headers():
    msg = EmailNotifier.from_env(ENV).message(Alert("info", "Hello\r\nBcc: x@evil.test"))
    assert msg["Subject"] == "[trading-lab] INFO: Hello Bcc: x@evil.test" and msg["Bcc"] is None


def test_several_channels():
    class Broken:
        def send(self, alert):
            raise ConnectionError("down")

    memory = MemoryNotifier()
    MultiNotifier([Broken(), memory]).send(Alert("info", "One works"))
    assert [a.title for a in memory.sent] == ["One works"]
    with pytest.raises(ConnectionError, match="down; down"):
        MultiNotifier([Broken(), Broken()]).send(Alert("info", "None works"))

    both = AppConfig.from_mapping({"alerts": {"enabled": True, "channels": ["webhook", "email"]}})
    notifier = build_notifier(both, {**ENV, URL_ENV: "https://ntfy.example.test/topic"})
    assert isinstance(notifier, MultiNotifier)
    assert [type(n) for n in notifier.notifiers] == [WebhookNotifier, EmailNotifier]
    with pytest.raises(ConfigError, match=URL_ENV):
        build_notifier(both, ENV)


def test_configuration():
    assert AppConfig().alerts.channels == ("webhook",)  # unchanged default
    manager = build_alerts(EMAIL_CFG, ENV)
    assert isinstance(manager.notifier, EmailNotifier)
    for bad in ([], ["sms"], ["email", "email"]):
        with pytest.raises(ConfigError, match="alerts.channels"):
            AppConfig.from_mapping({"alerts": {"channels": bad}})
    fields = set(type(AppConfig().alerts).__dataclass_fields__)
    assert not {f for f in fields if any(w in f for w in ("password", "user", "host", "smtp", "login", "url"))}


def test_doctor_and_alert_test(tmp_path, monkeypatch, capsys):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[alerts]\nenabled = true\nchannels = ["email"]\n')
    checks = {c.name: c for c in run_checks(str(cfg_file), env=ENV)}
    assert checks["alerts"].status == OK
    assert "email to 2 recipient(s) via smtp.example.test:587" in checks["alerts"].detail
    assert LOGIN not in format_checks(list(checks.values()))
    assert {c.name: c for c in run_checks(str(cfg_file), env={})}["alerts"].status == FAIL

    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    assert main(["--config", str(cfg_file), "alert-test"]) == 0
    out = capsys.readouterr().out
    assert "Sending a test e-mail to 2 recipient(s) via smtp.example.test:587" in out and LOGIN not in out
    assert FakeSMTP.instances[-1].sent[0]["Subject"] == "[trading-lab] INFO: Test alert"
    FakeSMTP.fail_with = smtplib.SMTPAuthenticationError(535, b"no")
    assert main(["--config", str(cfg_file), "alert-test", "--channel", "email"]) == 1
    assert LOGIN not in capsys.readouterr().err
