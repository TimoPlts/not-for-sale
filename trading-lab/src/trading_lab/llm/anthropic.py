"""Anthropic (Claude), through the Messages API.

Environment variables (never config, never committed):

* ``ANTHROPIC_API_KEY``: the API key, sent only in the ``x-api-key`` header
* ``ANTHROPIC_MODEL``: the model name
* ``ANTHROPIC_API_URL``: optional; defaults to ``https://api.anthropic.com``.
  A base URL gets ``/v1/messages`` appended; a full ``.../messages`` URL is
  used as is.

Retries, the circuit breaker, redaction and usage accounting are the same as
for the OpenAI-compatible providers. HTTP 529 (overloaded) is retried too.
"""

from __future__ import annotations

import json
from typing import Any

from trading_lab.llm.openai_compat import RETRYABLE_STATUS, OpenAICompatibleProvider

API_VERSION = "2023-06-01"


def messages_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/messages"):
        return url
    return f"{url}/messages" if url.endswith("/v1") else f"{url}/v1/messages"


class AnthropicProvider(OpenAICompatibleProvider):
    name = "anthropic"
    env_prefix = "ANTHROPIC"
    default_url = "https://api.anthropic.com"
    retryable_status = RETRYABLE_STATUS | {529}

    def _request_url(self) -> str:
        return messages_url(self._url)

    def _request(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, str], bytes]:
        body = json.dumps({
            "model": self._model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "anthropic-version": API_VERSION,
            "x-api-key": self._token,
        }
        return headers, body

    def _content(self, data: dict[str, Any]) -> tuple[str, Any, Any, Any]:
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ValueError("no content blocks")
        texts = [b.get("text") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        if not texts or not all(isinstance(t, str) for t in texts):
            raise ValueError("no text content block")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return "".join(texts), data.get("stop_reason"), usage.get("input_tokens"), usage.get("output_tokens")
