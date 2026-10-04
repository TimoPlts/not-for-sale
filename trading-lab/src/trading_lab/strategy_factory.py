"""Build the configured strategies, including AI-agent strategies, ready to use."""

from __future__ import annotations

from typing import Sequence

import trading_lab.agents  # noqa: F401  (registers agent strategies)
from trading_lab.agents import AgentStrategy, SQLiteResponseCache
from trading_lab.config import AppConfig
from trading_lab.strategies import Strategy, build_strategies


def configure_agents(strategies: Sequence[Strategy], config: AppConfig) -> None:
    """Give agent strategies the configured mode, timeframe and shared answer cache."""
    agent_strategies = [s for s in strategies if isinstance(s, AgentStrategy)]
    if not agent_strategies:
        return
    cache = SQLiteResponseCache(config.agents.cache_path)
    for strategy in agent_strategies:
        strategy.configure(mode=config.agents.mode, cache=cache, timeframe=config.market.timeframe)


def strategies_for(config: AppConfig) -> list[Strategy]:
    strategies = build_strategies(config.enabled_strategies)
    configure_agents(strategies, config)
    return strategies
