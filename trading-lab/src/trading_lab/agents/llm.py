"""Agents answered by the LLM provider configured in ``[agents]``.

``LLMProviderStrategy`` is the base for every model-backed agent strategy.
The strategy is built from its ``[strategies.<name>]`` table like any other;
``strategy_factory.configure_agents`` then attaches the one shared provider
(e.g. Qwen) built from ``[agents]`` and the environment.

The model only ever sees a market context and only ever returns text. The
text is strictly parsed into a direction, a confidence and a rationale and
becomes a signal; anything else (an error, a timeout, malformed JSON) becomes
a HOLD signal. The ensemble, the circuit breakers, the risk manager and the
paper executor always sit between the model and any simulated trade.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, ClassVar, Mapping, Sequence

from trading_lab.agents.base import Agent, AgentResponse, AgentResponseError
from trading_lab.agents.context import MarketContext
from trading_lab.agents.parsing import RESPONSE_INSTRUCTIONS, parse_agent_json
from trading_lab.agents.strategy import AgentStrategy
from trading_lab.llm.base import LLMProvider, ProviderConfigError, ProviderError
from trading_lab.llm.usage import CallRecord, estimate_tokens
from trading_lab.strategies.registry import register_strategy


class ProviderAgent(Agent):
    """An agent that asks an ``LLMProvider`` with a fixed system prompt."""

    version = "1"

    def __init__(
        self,
        name: str,
        system_prompt: str = RESPONSE_INSTRUCTIONS,
        *,
        labels: Mapping[str, Sequence[str]] | None = None,
        version: str | None = None,
    ) -> None:
        self.name = name  # type: ignore[misc]  # one agent identity per strategy
        if version is not None:
            self.version = version  # type: ignore[misc]
        self.system_prompt = system_prompt
        self.labels = {k: tuple(v) for k, v in (labels or {}).items()}
        self.provider: LLMProvider | None = None
        self.last_call: CallRecord | None = None

    @property
    def params(self) -> dict[str, Any]:
        # Provider, model, sampling settings and prompt are all part of the cache key.
        prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:12]
        provider = self.provider.cache_params if self.provider is not None else {"provider": None}
        return {**provider, "prompt": prompt_hash}

    def user_prompt(self, context: MarketContext) -> str:
        return context.to_prompt()

    def decide(self, context: MarketContext) -> AgentResponse:
        provider = self.provider
        if provider is None:
            raise ProviderConfigError(f"{self.name}: no LLM provider attached")
        system, user = self.system_prompt, self.user_prompt(context)
        input_chars = len(system) + len(user)
        started = time.monotonic()
        try:
            completion = provider.chat(system, user)
        except ProviderError as exc:
            self._record(provider, CallRecord(
                provider.name, provider.model, False, time.monotonic() - started, exc.attempts,
                input_chars, 0, estimate_tokens(input_chars), 0, True, type(exc).__name__,
            ), valid=False)
            raise
        record = CallRecord(
            provider.name, completion.model, True, completion.latency_seconds, completion.attempts,
            input_chars, len(completion.text),
            completion.input_tokens if completion.input_tokens is not None else estimate_tokens(input_chars),
            completion.output_tokens if completion.output_tokens is not None else estimate_tokens(len(completion.text)),
            completion.input_tokens is None or completion.output_tokens is None,
        )
        try:
            response = parse_agent_json(completion.text, self.labels)
        except AgentResponseError as exc:
            self._record(provider, record, valid=False)
            if completion.finish_reason == "length":
                raise AgentResponseError(
                    f"{exc} (the answer was cut off; raise [agents] max_output_tokens)"
                ) from None
            raise
        self._record(provider, record, valid=True)
        return response

    def _record(self, provider: LLMProvider, record: CallRecord, *, valid: bool) -> None:
        self.last_call = record
        provider.usage.record_call(self.name, record, valid=valid)

    def pop_last_call(self) -> CallRecord | None:
        record, self.last_call = self.last_call, None
        return record


class LLMProviderStrategy(AgentStrategy):
    """Base for strategies whose agent is the configured LLM provider."""

    system_prompt: ClassVar[str] = RESPONSE_INSTRUCTIONS

    def build_agent(self, **params: Any) -> Agent:
        if params:
            raise ValueError(f"unknown parameter(s) {sorted(params)}")
        return ProviderAgent(self.name, self.system_prompt)

    def attach_provider(self, provider: LLMProvider) -> None:
        assert isinstance(self.agent, ProviderAgent)
        self.agent.provider = provider

    @property
    def provider(self) -> LLMProvider | None:
        return self.agent.provider if isinstance(self.agent, ProviderAgent) else None

    def _call_meta(self) -> dict[str, Any]:
        record = self.agent.pop_last_call() if isinstance(self.agent, ProviderAgent) else None
        return {"llm": record.to_json()} if record is not None else {}

    def _on_cache(self, hit: bool) -> None:
        if self.provider is not None:
            self.provider.usage.record_cache(self.name, hit)


@register_strategy
class LLMAnalystStrategy(LLMProviderStrategy):
    """General-purpose model analyst. Config: ``[strategies.llm_analyst]`` with
    optional ``lookback`` and ``decision_interval``; the provider comes from ``[agents]``."""

    name = "llm_analyst"
