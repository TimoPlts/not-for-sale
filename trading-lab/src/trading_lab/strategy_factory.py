"""Build the configured strategies, including AI-agent strategies, ready to use."""

from __future__ import annotations

from typing import Sequence

import trading_lab.agents  # noqa: F401  (registers agent strategies)
from trading_lab.agents import AgentStrategy, LLMProviderStrategy, SQLiteResponseCache
from trading_lab.config import AppConfig
from trading_lab.llm import LLMProvider, build_llm_provider
from trading_lab.strategies import Strategy, build_strategies


def configure_agents(
    strategies: Sequence[Strategy], config: AppConfig, *, llm_provider: LLMProvider | None = None
) -> None:
    """Give agent strategies the configured mode, timeframe and shared answer cache.

    LLM-backed strategies also share one provider, built from ``[agents]`` and
    the environment unless ``llm_provider`` is given. Missing environment
    variables fail here, before any run starts. In replay mode the model is
    never called, so only the model name (part of the cache key) is required.
    """
    agent_strategies = [s for s in strategies if isinstance(s, AgentStrategy)]
    if not agent_strategies:
        return
    llm_strategies = [s for s in agent_strategies if isinstance(s, LLMProviderStrategy)]
    if llm_strategies:
        provider = llm_provider if llm_provider is not None else build_llm_provider(config.agents)
        provider.check_ready(need_credentials=config.agents.mode != "replay")
        for strategy in llm_strategies:
            strategy.attach_provider(provider)
    cache = SQLiteResponseCache(config.agents.cache_path)
    for strategy in agent_strategies:
        strategy.configure(mode=config.agents.mode, cache=cache, timeframe=config.market.timeframe)


def strategies_for(config: AppConfig, *, llm_provider: LLMProvider | None = None) -> list[Strategy]:
    strategies = build_strategies(config.enabled_strategies)
    configure_agents(strategies, config, llm_provider=llm_provider)
    return strategies
