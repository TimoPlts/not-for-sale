"""Provider-agnostic interface to text-generation models (LLMs).

An ``LLMProvider`` turns a system prompt and a user prompt into text. That
is all it does: it knows nothing about markets, orders or portfolios. Agents
(``trading_lab.agents``) turn the text into a validated signal, and the
ensemble, the circuit breakers, the risk manager and the paper executor sit
between any model and any simulated trade.

Credentials for a model endpoint come only from environment variables. They
are never part of the configuration, never stored and never printed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from trading_lab.core.errors import ConfigError, TradingLabError


class ProviderConfigError(ConfigError):
    """A provider is not usable as configured (e.g. missing environment variables)."""


class ProviderError(TradingLabError):
    """A request to the model failed. ``attempts`` counts the requests made."""

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts


class ProviderTimeoutError(ProviderError):
    """The model did not answer within the timeout (after all retries)."""


class ProviderResponseError(ProviderError):
    """The endpoint answered, but not with a usable completion."""


@dataclass(frozen=True, slots=True)
class Completion:
    """One model answer plus the facts needed for usage accounting."""

    text: str
    model: str
    latency_seconds: float  # wall time of the whole call, retries included
    attempts: int = 1  # HTTP requests made (1 = no retry)
    input_tokens: int | None = None  # as reported by the endpoint, if it does
    output_tokens: int | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Base class for model providers (Qwen today; OpenAI, Anthropic, Gemini later)."""

    name: ClassVar[str]

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier sent to the endpoint ("" if not configured)."""

    @property
    def cache_params(self) -> dict[str, Any]:
        """Everything that changes the answers; part of every agent cache key."""
        return {"provider": self.name, "model": self.model}

    def check_ready(self, *, need_credentials: bool = True) -> None:
        """Raise ``ProviderConfigError`` if the provider cannot be used.

        With ``need_credentials=False`` (replay mode) only what is needed to
        look answers up in the cache is checked; the model is never called.
        """

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> Completion:
        """Ask the model once (with the provider's own timeout and retries)."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """The ``complete(system, user) -> text`` function used by ``LLMAgent``."""
        return self.chat(system_prompt, user_prompt).text
