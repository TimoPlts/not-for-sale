"""Build the LLM provider named in ``[agents] provider``."""

from __future__ import annotations

from typing import Any

from trading_lab.config import AgentsConfig
from trading_lab.llm.base import LLMProvider, ProviderConfigError
from trading_lab.llm.qwen import QwenProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    QwenProvider.name: QwenProvider,
}


def build_llm_provider(config: AgentsConfig, **overrides: Any) -> LLMProvider:
    """Provider with the configured timeout, retries and sampling settings.

    ``overrides`` (``env``, ``transport``, ``sleep``, ``clock``) exist for tests.
    """
    try:
        cls = PROVIDERS[config.provider]
    except KeyError:
        raise ProviderConfigError(
            f"unknown LLM provider {config.provider!r}; available: {sorted(PROVIDERS)}"
        ) from None
    return cls(
        timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        **overrides,
    )
