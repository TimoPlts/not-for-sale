"""Alerts: push notifications about a live paper run.

Two channels, chosen with ``[alerts] channels``:

* **webhook** (ntfy, Slack, Discord or any JSON endpoint): the URL comes from
  the ``TRADING_LAB_ALERT_URL`` environment variable only. Such URLs usually
  contain a secret token, so the URL is never stored or logged; messages
  only name its host.
* **email** over SMTP: every setting comes from ``TRADING_LAB_SMTP_*`` and
  ``TRADING_LAB_ALERT_EMAIL_*`` environment variables. The connection is
  encrypted (STARTTLS or SSL); without encryption only a local relay with no
  login is allowed, so a login is never sent in clear text. The login is
  never stored, logged or shown.

Alerts are best effort. A notification that cannot be delivered is logged
and dropped: it never delays, changes or stops trading. Alerts only report
what already happened in the simulation; nothing here can act on it.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import time
import urllib.parse
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate
from typing import Callable, Mapping, Protocol, Sequence

from trading_lab.config import ALERT_FORMATS
from trading_lab.core.errors import ConfigError
from trading_lab.llm.openai_compat import HttpTransport, TransportConnectionError, TransportTimeout, UrllibTransport

logger = logging.getLogger(__name__)

URL_ENV = "TRADING_LAB_ALERT_URL"
SMTP_HOST_ENV = "TRADING_LAB_SMTP_HOST"
SMTP_PORT_ENV = "TRADING_LAB_SMTP_PORT"
SMTP_SECURITY_ENV = "TRADING_LAB_SMTP_SECURITY"  # starttls (default) | ssl | none (local relay only)
SMTP_USER_ENV = "TRADING_LAB_SMTP_USER"
SMTP_PASS_ENV = "TRADING_LAB_SMTP_PASSWORD"
EMAIL_FROM_ENV = "TRADING_LAB_ALERT_EMAIL_FROM"
EMAIL_TO_ENV = "TRADING_LAB_ALERT_EMAIL_TO"  # comma-separated
SMTP_SECURITY = ("starttls", "ssl", "none")
_DEFAULT_PORT = {"starttls": 587, "ssl": 465, "none": 25}
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
LEVELS = {"info": 0, "warning": 1, "critical": 2}
FORMATS = ALERT_FORMATS
_NTFY_PRIORITY = {"info": "default", "warning": "high", "critical": "urgent"}
_ICON = {"info": "i", "warning": "!", "critical": "!!"}


@dataclass(frozen=True, slots=True)
class Alert:
    level: str
    title: str
    body: str = ""
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"alert level must be one of {list(LEVELS)}")


class Notifier(Protocol):
    def send(self, alert: Alert) -> None: ...


class MemoryNotifier:
    """Keeps alerts in a list (tests, dry runs)."""

    def __init__(self) -> None:
        self.sent: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.sent.append(alert)


class WebhookNotifier:
    def __init__(
        self, url: str, fmt: str = "ntfy", *, timeout: float = 10.0, transport: HttpTransport | None = None
    ) -> None:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConfigError(f"{URL_ENV} must be an http(s) URL")
        if fmt not in FORMATS:
            raise ConfigError(f"alerts.format must be one of {FORMATS}, got {fmt!r}")
        self._url = url
        self.host = parts.hostname
        self.fmt = fmt
        self.timeout = timeout
        self._transport = transport or UrllibTransport()

    @classmethod
    def from_env(cls, fmt: str, env: Mapping[str, str] | None = None, **kwargs: object) -> WebhookNotifier:
        url = (os.environ if env is None else env).get(URL_ENV, "").strip()
        if not url:
            raise ConfigError(f"alerts are enabled but {URL_ENV} is not set (put it in the service environment file)")
        return cls(url, fmt, **kwargs)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"WebhookNotifier(format={self.fmt!r}, host={self.host!r})"

    def _request(self, alert: Alert) -> tuple[dict[str, str], bytes]:
        title = f"trading-lab: {alert.title}"
        text = f"{title}\n{alert.body}".strip()
        if self.fmt == "ntfy":
            headers = {
                "Title": title.encode("ascii", "replace").decode("ascii"),
                "Priority": _NTFY_PRIORITY[alert.level],
                "Tags": f"chart_with_upwards_trend,{alert.level}",
                "Markdown": "yes",
                "Content-Type": "text/plain; charset=utf-8",
            }
            return headers, (alert.body or alert.title).encode("utf-8")
        if self.fmt == "slack":
            payload: dict[str, object] = {"text": f"[{_ICON[alert.level]}] *{title}*\n{alert.body}".strip()}
        elif self.fmt == "discord":
            payload = {"content": f"[{_ICON[alert.level]}] **{title}**\n{alert.body}".strip()[:1990]}
        else:
            payload = {"level": alert.level, "title": alert.title, "body": alert.body, "run_id": alert.run_id,
                       "text": text}
        return {"Content-Type": "application/json"}, json.dumps(payload).encode("utf-8")

    def send(self, alert: Alert) -> None:
        headers, body = self._request(alert)
        try:
            status, _ = self._transport.post(self._url, headers, body, self.timeout)
        except (TransportTimeout, TransportConnectionError) as exc:
            raise ConnectionError(f"alert webhook at {self.host} unreachable ({type(exc).__name__})") from None
        if not 200 <= status < 300:
            raise ConnectionError(f"alert webhook at {self.host} answered HTTP {status}")


def _address(value: str, what: str) -> str:
    value = value.strip()
    if not value or "@" not in value or any(c in value for c in "\r\n,;<> "):
        raise ConfigError(f"{what} must be a plain e-mail address, got {value!r}")
    return value


class EmailNotifier:
    """Sends each alert as a plain-text e-mail over SMTP (STARTTLS, SSL, or a local relay)."""

    def __init__(
        self,
        host: str,
        recipients: Sequence[str],
        *,
        sender: str | None = None,
        port: int | None = None,
        security: str = "starttls",
        login: tuple[str, str] | None = None,
        timeout: float = 20.0,
        smtp_class: Callable[..., smtplib.SMTP] | None = None,
        smtp_ssl_class: Callable[..., smtplib.SMTP] | None = None,
    ) -> None:
        host = host.strip()
        if not host or any(c.isspace() for c in host):
            raise ConfigError(f"{SMTP_HOST_ENV} must be a host name")
        if security not in SMTP_SECURITY:
            raise ConfigError(f"{SMTP_SECURITY_ENV} must be one of {SMTP_SECURITY}, got {security!r}")
        if security == "none" and (host not in _LOCAL_HOSTS or login is not None):
            raise ConfigError(f"{SMTP_SECURITY_ENV}=none is only allowed for a local relay ({', '.join(_LOCAL_HOSTS)}) "
                              "without a login: a login is never sent unencrypted")
        port = _DEFAULT_PORT[security] if port is None else port
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ConfigError(f"{SMTP_PORT_ENV} must be a port number, got {port!r}")
        self.recipients = tuple(_address(r, EMAIL_TO_ENV) for r in recipients)
        if not self.recipients:
            raise ConfigError(f"{EMAIL_TO_ENV} must name at least one recipient")
        if sender is None and login is not None and "@" in login[0]:
            sender = login[0]
        if sender is None:
            raise ConfigError(f"set {EMAIL_FROM_ENV} (the sender address)")
        self.sender = _address(sender, EMAIL_FROM_ENV)
        self.host, self.port, self.security, self.timeout = host, port, security, timeout
        self._login = login
        self._smtp_class = smtp_class or smtplib.SMTP
        self._smtp_ssl_class = smtp_ssl_class or smtplib.SMTP_SSL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kwargs: object) -> EmailNotifier:
        env = os.environ if env is None else env
        get = lambda name: env.get(name, "").strip()  # noqa: E731
        missing = [name for name in (SMTP_HOST_ENV, EMAIL_TO_ENV) if not get(name)]
        if missing:
            raise ConfigError(f"e-mail alerts are enabled but {', '.join(missing)} is not set "
                              "(put it in the service environment file)")
        user, pw = get(SMTP_USER_ENV), env.get(SMTP_PASS_ENV, "")
        if bool(user) != bool(pw):
            raise ConfigError(f"set both {SMTP_USER_ENV} and {SMTP_PASS_ENV}, or neither (a local relay)")
        port = None
        if get(SMTP_PORT_ENV):
            try:
                port = int(get(SMTP_PORT_ENV))
            except ValueError:
                raise ConfigError(f"{SMTP_PORT_ENV} must be a port number") from None
        return cls(
            get(SMTP_HOST_ENV),
            [r for r in get(EMAIL_TO_ENV).split(",") if r.strip()],
            sender=get(EMAIL_FROM_ENV) or None,
            port=port,
            security=get(SMTP_SECURITY_ENV) or "starttls",
            login=(user, pw) if user else None,
            **kwargs,  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:  # never shows the login
        return (f"EmailNotifier(host={self.host!r}, port={self.port}, security={self.security!r}, "
                f"recipients={len(self.recipients)})")

    def message(self, alert: Alert) -> EmailMessage:
        msg = EmailMessage()
        title = " ".join(alert.title.split())  # one line: no header injection
        msg["Subject"] = f"[trading-lab] {alert.level.upper()}: {title}"
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg["Date"] = formatdate(usegmt=True)
        lines = [alert.body or alert.title, "", f"Level: {alert.level}"]
        if alert.run_id:
            lines.append(f"Run: {alert.run_id}")
        lines += ["", "Sent by trading-lab. Simulated paper trading only: no real order is ever placed."]
        msg.set_content("\n".join(lines))
        return msg

    def send(self, alert: Alert) -> None:
        msg = self.message(alert)
        context = ssl.create_default_context()
        try:
            if self.security == "ssl":
                smtp = self._smtp_ssl_class(self.host, self.port, timeout=self.timeout, context=context)
            else:
                smtp = self._smtp_class(self.host, self.port, timeout=self.timeout)
            with smtp:
                if self.security == "starttls":
                    smtp.starttls(context=context)
                if self._login is not None:
                    smtp.login(*self._login)
                smtp.send_message(msg)
        except (smtplib.SMTPException, OSError) as exc:
            raise ConnectionError(f"alert e-mail via {self.host}:{self.port} failed ({type(exc).__name__})") from None


class MultiNotifier:
    """Sends to every channel. Fails only when no channel delivered."""

    def __init__(self, notifiers: Sequence[Notifier]) -> None:
        if not notifiers:
            raise ValueError("give at least one notifier")
        self.notifiers = tuple(notifiers)

    def __repr__(self) -> str:
        return f"MultiNotifier({', '.join(repr(n) for n in self.notifiers)})"

    def send(self, alert: Alert) -> None:
        errors = []
        for notifier in self.notifiers:
            try:
                notifier.send(alert)
            except Exception as exc:
                errors.append(f"{exc}")
                logger.warning("alert %r not delivered by one channel: %s", alert.title, exc)
        if len(errors) == len(self.notifiers):
            raise ConnectionError("; ".join(errors))


def build_notifier(config: object, env: Mapping[str, str] | None = None,
                   channels: Sequence[str] | None = None) -> Notifier:
    """The notifier for the configured (or given) channels; fails fast on missing settings."""
    alerts = config.alerts  # type: ignore[attr-defined]
    notifiers: list[Notifier] = []
    for channel in channels or alerts.channels:
        if channel == "webhook":
            notifiers.append(WebhookNotifier.from_env(alerts.format, env))
        elif channel == "email":
            notifiers.append(EmailNotifier.from_env(env))
        else:
            raise ConfigError(f"unknown alert channel {channel!r}")
    return notifiers[0] if len(notifiers) == 1 else MultiNotifier(notifiers)


class AlertManager:
    """Filters by level, suppresses repeats of the same alert, and never raises."""

    def __init__(
        self,
        notifier: Notifier,
        *,
        min_level: str = "warning",
        repeat_after_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_level not in LEVELS:
            raise ConfigError(f"alerts.min_level must be one of {list(LEVELS)}")
        self.notifier = notifier
        self.min_level = min_level
        self.repeat_after_seconds = repeat_after_seconds
        self._clock = clock
        self._last: dict[str, float] = {}
        self.delivered = 0
        self.failed = 0

    def emit(self, level: str, title: str, body: str = "", *, key: str | None = None,
             run_id: str | None = None, force: bool = False) -> bool:
        """Send unless below ``min_level`` (``force`` overrides) or repeated within the window."""
        if not force and LEVELS[level] < LEVELS[self.min_level]:
            return False
        now = self._clock()
        if key is not None and key in self._last and now - self._last[key] < self.repeat_after_seconds:
            return False
        try:
            self.notifier.send(Alert(level, title, body, run_id))
        except Exception as exc:  # alerts must never interfere with trading
            self.failed += 1
            logger.warning("alert %r not delivered: %s", title, exc)
            return False
        self.delivered += 1
        if key is not None:
            self._last[key] = now
        return True

    def clear(self, key: str) -> None:
        self._last.pop(key, None)


def build_alerts(config: object, env: Mapping[str, str] | None = None) -> AlertManager | None:
    """The configured ``AlertManager``, or None when alerts are disabled."""
    alerts = config.alerts  # type: ignore[attr-defined]
    if not alerts.enabled:
        return None
    notifier = build_notifier(config, env)
    return AlertManager(notifier, min_level=alerts.min_level, repeat_after_seconds=alerts.repeat_after_minutes * 60)
