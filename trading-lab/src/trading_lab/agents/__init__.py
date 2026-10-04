"""AI agents as signal sources (see ``agents.strategy`` for how they plug in).

Importing this package registers the example agent strategies.
"""

from trading_lab.agents.base import Agent, AgentResponse, AgentResponseError
from trading_lab.agents.cache import MemoryResponseCache, SQLiteResponseCache, cache_key
from trading_lab.agents.context import MarketContext, build_context, indicator_frame
from trading_lab.agents.examples import (
    LLMAgent,
    LLMAgentStrategy,
    TrendAnalystStrategy,
    TrendFollowingAnalyst,
)
from trading_lab.agents.parsing import RESPONSE_INSTRUCTIONS, parse_agent_json
from trading_lab.agents.strategy import AgentStrategy

__all__ = [
    "RESPONSE_INSTRUCTIONS",
    "Agent",
    "AgentResponse",
    "AgentResponseError",
    "AgentStrategy",
    "LLMAgent",
    "LLMAgentStrategy",
    "MarketContext",
    "MemoryResponseCache",
    "SQLiteResponseCache",
    "TrendAnalystStrategy",
    "TrendFollowingAnalyst",
    "build_context",
    "cache_key",
    "indicator_frame",
    "parse_agent_json",
]
