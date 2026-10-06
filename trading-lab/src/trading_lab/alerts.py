"""Alerts: push notifications about a live paper run.

A webhook URL (ntfy, Slack, Discord or any JSON endpoint) comes from the
``TRADING_LAB_ALERT_URL`` environment variable only. Such URLs usually
contain a secret token, so the URL is never stored or logged; messages only
name its host.

Alerts are best effort. A notification that cannot be delivered is logged
and dropped: it never delays, changes or stops trading. Alerts only report
what already happened in the simulation; nothing here can act on it.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from trading_lab.config import ALERT_FORMATS
from trading_lab.core.errors import ConfigError
from trading_lab.llm.openai_compat import HttpTransport, TransportConnectionError, TransportTimeout, UrllibTransport

logger = logging.getLogger(__name__)

URL_ENV = "TRADING_LAB_ALERT_URL"
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
    notifier = WebhookNotifier.from_env(alerts.format, env)
    return AlertManager(notifier, min_level=alerts.min_level, repeat_after_seconds=alerts.repeat_after_minutes * 60)
