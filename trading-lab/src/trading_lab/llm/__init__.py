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
    ProviderUnavailableError,
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
from trading_lab.llm.usage import (
    CallRecord,
    UsageStats,
    UsageTracker,
    estimate_tokens,
    format_usage,
    total_usage,
    usage_from_signals,
)

__all__ = [
    "PROVIDERS",
    "CallRecord",
    "UsageStats",
    "UsageTracker",
    "estimate_tokens",
    "format_usage",
    "total_usage",
    "usage_from_signals",
    "Completion",
    "HttpTransport",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "ProviderConfigError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "QwenProvider",
    "TransportConnectionError",
    "TransportTimeout",
    "UrllibTransport",
    "build_llm_provider",
    "chat_completions_url",
]
