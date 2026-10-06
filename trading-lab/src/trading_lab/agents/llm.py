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
from typing import Any, ClassVar, Mapping, Sequence

from trading_lab.agents.base import Agent, AgentResponse, AgentResponseError
from trading_lab.agents.context import MarketContext
from trading_lab.agents.parsing import RESPONSE_INSTRUCTIONS, parse_agent_json
from trading_lab.agents.strategy import AgentStrategy
from trading_lab.llm.base import LLMProvider, ProviderConfigError
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

    @property
    def params(self) -> dict[str, Any]:
        # Provider, model, sampling settings and prompt are all part of the cache key.
        prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:12]
        provider = self.provider.cache_params if self.provider is not None else {"provider": None}
        return {**provider, "prompt": prompt_hash}

    def user_prompt(self, context: MarketContext) -> str:
        return context.to_prompt()

    def decide(self, context: MarketContext) -> AgentResponse:
        if self.provider is None:
            raise ProviderConfigError(f"{self.name}: no LLM provider attached")
        completion = self.provider.chat(self.system_prompt, self.user_prompt(context))
        try:
            return parse_agent_json(completion.text, self.labels)
        except AgentResponseError as exc:
            if completion.finish_reason == "length":
                raise AgentResponseError(
                    f"{exc} (the answer was cut off; raise [agents] max_output_tokens)"
                ) from None
            raise


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


@register_strategy
class LLMAnalystStrategy(LLMProviderStrategy):
    """General-purpose model analyst. Config: ``[strategies.llm_analyst]`` with
    optional ``lookback`` and ``decision_interval``; the provider comes from ``[agents]``."""

    name = "llm_analyst"
