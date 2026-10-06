"""Client for model endpoints with an OpenAI-compatible ``/chat/completions`` API.

Many hosted and self-hosted models (Qwen via DashScope, vLLM, Ollama, LM
Studio, ...) expose this API. A concrete provider only names its environment
variable prefix: ``QwenProvider`` reads ``QWEN_API_URL``, ``QWEN_API_KEY`` and
``QWEN_MODEL``.

* ``<PREFIX>_API_URL`` is the base URL (``https://host/v1``) or the full
  ``.../chat/completions`` URL.
* Requests time out, and timeouts, network errors, HTTP 429 and 5xx are
  retried with exponential backoff. Other HTTP errors (bad token, bad model
  name, ...) fail at once with a clear message.
* The token is sent only in the ``Authorization`` header of requests to the
  configured URL. It never appears in errors, logs or ``repr()``.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, ClassVar, Mapping, Protocol

from trading_lab.llm.base import (
    Completion,
    LLMProvider,
    ProviderConfigError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_EXCERPT_CHARS = 200


# ------------------------------------------------------------------ transport
class TransportTimeout(Exception):
    """No complete answer within the timeout."""


class TransportConnectionError(Exception):
    """The endpoint could not be reached (DNS, refused, reset, ...)."""


class HttpTransport(Protocol):
    def post(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        """POST ``body`` and return ``(status, response body)`` for any HTTP status."""
        ...


class UrllibTransport:
    """Standard-library HTTP transport (honours the usual proxy environment variables)."""

    def post(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (scheme checked)
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            try:
                payload = exc.read()
            except (OSError, http.client.HTTPException):
                payload = b""
            return int(exc.code), payload
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TransportTimeout(str(exc.reason)) from None
            raise TransportConnectionError(str(exc.reason)) from None
        except (TimeoutError, socket.timeout) as exc:
            raise TransportTimeout(str(exc)) from None
        except (OSError, http.client.HTTPException) as exc:
            raise TransportConnectionError(f"{type(exc).__name__}: {exc}") from None


# ------------------------------------------------------------------- provider
def chat_completions_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    return url if url.endswith("/chat/completions") else f"{url}/chat/completions"


def _safe_url(url: str) -> str:
    """URL without user info, query string or fragment (for messages)."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urllib.parse.urlunsplit((parts.scheme, host, parts.path, "", ""))


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class OpenAICompatibleProvider(LLMProvider):
    """Subclasses set ``name`` and ``env_prefix``."""

    name: ClassVar[str] = "openai_compatible"
    env_prefix: ClassVar[str]

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        env: Mapping[str, str] | None = None,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0 or retry_backoff_seconds < 0:
            raise ValueError("max_retries and retry_backoff_seconds must be >= 0")
        source = os.environ if env is None else env
        self._url = source.get(self.env_var("API_URL"), "").strip()
        self._token = source.get(self.env_var("API_KEY"), "").strip()
        self._model = source.get(self.env_var("MODEL"), "").strip()
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self._transport: HttpTransport = transport or UrllibTransport()
        self._sleep = sleep
        self._clock = clock

    @classmethod
    def env_var(cls, suffix: str) -> str:
        return f"{cls.env_prefix}_{suffix}"

    @property
    def model(self) -> str:
        return self._model

    @property
    def endpoint(self) -> str:
        """The chat-completions URL without credentials or query string."""
        return _safe_url(chat_completions_url(self._url)) if self._url else ""

    @property
    def cache_params(self) -> dict[str, Any]:
        return {
            **super().cache_params,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

    def __repr__(self) -> str:
        token = "set" if self._token else "missing"
        return f"{type(self).__name__}(model={self._model!r}, endpoint={self.endpoint!r}, token={token})"

    # ------------------------------------------------------------- validation
    def check_ready(self, *, need_credentials: bool = True) -> None:
        required = [("MODEL", self._model)]
        if need_credentials:
            required += [("API_URL", self._url), ("API_KEY", self._token)]
        missing = [self.env_var(suffix) for suffix, value in required if not value]
        if missing:
            why = "to call the model" if need_credentials else "to look up cached answers (replay mode)"
            raise ProviderConfigError(
                f"{self.name} provider: environment variable(s) {', '.join(missing)} not set; "
                f"they are needed {why}. Export them in your shell or service environment file; "
                "never put them in the config."
            )
        if need_credentials:
            scheme = urllib.parse.urlsplit(self._url).scheme.lower()
            if scheme not in ("http", "https"):
                raise ProviderConfigError(
                    f"{self.env_var('API_URL')} must be an http(s) URL, got scheme {scheme!r}"
                )

    # ---------------------------------------------------------------- request
    def _redact(self, text: str) -> str:
        if self._token:
            text = text.replace(self._token, "***")
        return text

    def _http_error(self, status: int, raw: bytes) -> str:
        excerpt = self._redact(raw[:_EXCERPT_CHARS].decode("utf-8", "replace")).strip()
        hint = ""
        if status in (401, 403):
            hint = f" (check {self.env_var('API_KEY')})"
        elif status == 404:
            hint = f" (check {self.env_var('API_URL')} and {self.env_var('MODEL')})"
        return f"{self.name}: HTTP {status} from {self.endpoint}{hint}: {excerpt or '<empty body>'}"

    def chat(self, system_prompt: str, user_prompt: str) -> Completion:
        self.check_ready(need_credentials=True)
        url = chat_completions_url(self._url)
        body = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._token}",
        }
        total = self.max_retries + 1
        started = self._clock()
        last: ProviderError | None = None
        for attempt in range(1, total + 1):
            try:
                status, raw = self._transport.post(url, headers, body, self.timeout_seconds)
            except TransportTimeout:
                last = ProviderTimeoutError(
                    f"{self.name}: no answer from {self.endpoint} within {self.timeout_seconds:g}s",
                    attempts=attempt,
                )
            except TransportConnectionError as exc:
                last = ProviderError(
                    f"{self.name}: cannot reach {self.endpoint}: {self._redact(str(exc))}",
                    attempts=attempt,
                )
            else:
                if 200 <= status < 300:
                    return self._parse(raw, attempts=attempt, latency=self._clock() - started)
                if status not in RETRYABLE_STATUS:
                    raise ProviderError(self._http_error(status, raw), attempts=attempt)
                last = ProviderError(self._http_error(status, raw), attempts=attempt)
            if attempt < total:
                delay = self.retry_backoff_seconds * 2 ** (attempt - 1)
                logger.warning("%s (attempt %d/%d); retrying in %.1fs", last, attempt, total, delay)
                self._sleep(delay)
        assert last is not None
        raise type(last)(f"{last} (gave up after {total} attempt(s))", attempts=total)

    def _parse(self, raw: bytes, *, attempts: int, latency: float) -> Completion:
        def fail(reason: str) -> ProviderResponseError:
            return ProviderResponseError(f"{self.name}: unusable response from {self.endpoint}: {reason}",
                                         attempts=attempts)

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            excerpt = self._redact(raw[:_EXCERPT_CHARS].decode("utf-8", "replace"))
            raise fail(f"body is not JSON: {excerpt!r}") from None
        if not isinstance(data, dict):
            raise fail("body is not a JSON object")
        if "error" in data:
            raise fail(f"endpoint reported an error: {self._redact(json.dumps(data['error'])[:_EXCERPT_CHARS])}")
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise fail("no choices[0].message.content") from None
        if not isinstance(content, str):
            raise fail("message content is not text")
        text = _THINK_BLOCK.sub("", content).strip()  # reasoning models (e.g. Qwen3) think aloud first
        if not text:
            raise fail("empty answer")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        finish = choice.get("finish_reason") if isinstance(choice, dict) else None
        return Completion(
            text=text,
            model=str(data.get("model") or self._model),
            latency_seconds=max(latency, 0.0),
            attempts=attempts,
            input_tokens=_int_or_none(usage.get("prompt_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            finish_reason=finish if isinstance(finish, str) else None,
        )
