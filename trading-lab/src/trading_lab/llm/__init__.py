"""Model providers for LLM-backed agents (provider-agnostic; Qwen is the first).

Providers only turn prompts into text. They cannot see the portfolio and
cannot place orders; see ``trading_lab.agents`` for how answers become signals.
"""

from trading_lab.llm.base import (
    Completion,
    LLMProvider,
    ProviderConfigError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from trading_lab.llm.factory import PROVIDERS, build_llm_provider
from trading_lab.llm.openai_compat import (
    HttpTransport,
    OpenAICompatibleProvider,
    TransportConnectionError,
    TransportTimeout,
    UrllibTransport,
    chat_completions_url,
)
from trading_lab.llm.qwen import QwenProvider

__all__ = [
    "PROVIDERS",
    "Completion",
    "HttpTransport",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "ProviderConfigError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "QwenProvider",
    "TransportConnectionError",
    "TransportTimeout",
    "UrllibTransport",
    "build_llm_provider",
    "chat_completions_url",
]
