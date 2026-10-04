"""Example agents.

* ``TrendFollowingAnalyst``: an offline, deterministic stand-in for an AI
  analyst. It reads the context and answers with a direction, a confidence
  and a plain-English rationale. It is registered as the ``trend_analyst``
  strategy and can be enabled from the config.
* ``LLMAgent``: wraps any text-completion function, for example a call to an
  LLM API. It renders the context into a prompt and strictly parses the JSON
  answer. No LLM is called by this package; you supply the function.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from trading_lab.agents.base import Agent, AgentResponse
from trading_lab.agents.context import MarketContext
from trading_lab.agents.parsing import RESPONSE_INSTRUCTIONS, parse_agent_json
from trading_lab.agents.strategy import AgentStrategy
from trading_lab.core.models import Direction
from trading_lab.strategies.registry import register_strategy


class TrendFollowingAnalyst(Agent):
    name = "trend_follower"
    version = "1"

    def __init__(self, overbought: float = 75.0, min_trend: float = 0.002) -> None:
        if not 50.0 < overbought < 100.0:
            raise ValueError("overbought must be between 50 and 100")
        if min_trend <= 0:
            raise ValueError("min_trend must be positive")
        self.overbought = float(overbought)
        self.min_trend = float(min_trend)

    @property
    def params(self) -> dict[str, Any]:
        return {"overbought": self.overbought, "min_trend": self.min_trend}

    def decide(self, context: MarketContext) -> AgentResponse:
        ind = context.indicators
        sma20, sma50, rsi_value = ind.get("sma_20"), ind.get("sma_50"), ind.get("rsi_14")
        if sma20 is None or sma50 is None or rsi_value is None:
            return AgentResponse(Direction.HOLD, 0.0, "Not enough history to judge the trend.")
        close = context.last_close
        trend = sma20 / sma50 - 1.0
        strength = min(abs(trend) / (5 * self.min_trend), 1.0)

        if rsi_value > self.overbought:
            return AgentResponse(
                Direction.SELL,
                0.5 + 0.5 * min((rsi_value - self.overbought) / (100 - self.overbought), 1.0),
                f"RSI {rsi_value:.1f} is overbought (> {self.overbought:.0f}); taking profits is prudent.",
            )
        if trend > self.min_trend and close > sma20:
            return AgentResponse(
                Direction.BUY,
                0.5 + 0.5 * strength,
                f"Uptrend: the 20-bar average is {trend:+.2%} versus the 50-bar average "
                f"and price ({close:.6g}) is above it.",
            )
        if trend < -self.min_trend and close < sma20:
            return AgentResponse(
                Direction.SELL,
                0.5 + 0.5 * strength,
                f"Downtrend: the 20-bar average is {trend:+.2%} versus the 50-bar average "
                f"and price ({close:.6g}) is below it.",
            )
        return AgentResponse(Direction.HOLD, 0.0, f"No clear trend ({trend:+.2%}); staying out.")


@register_strategy
class TrendAnalystStrategy(AgentStrategy):
    """Config: ``[strategies.trend_analyst]`` with optional ``lookback``,
    ``decision_interval``, ``overbought`` and ``min_trend``."""

    name = "trend_analyst"

    def build_agent(self, overbought: float = 75.0, min_trend: float = 0.002) -> Agent:
        return TrendFollowingAnalyst(overbought, min_trend)


class LLMAgent(Agent):
    """Agent backed by a text-completion function ``complete(system, user) -> text``.

    Example wiring (pseudo-code; needs an LLM API key, never an exchange key)::

        def complete(system, user):
            reply = client.messages.create(model=..., system=system,
                                           messages=[{"role": "user", "content": user}], ...)
            return reply.content[0].text

        strategy = LLMAgentStrategy(complete=complete, model="...")
    """

    name = "llm"
    version = "1"

    def __init__(
        self,
        complete: Callable[[str, str], str],
        model: str = "unspecified",
        system_prompt: str = RESPONSE_INSTRUCTIONS,
    ) -> None:
        if not callable(complete):
            raise ValueError("complete must be a callable (system, user) -> text")
        self._complete = complete
        self.model = model
        self.system_prompt = system_prompt

    @property
    def params(self) -> dict[str, Any]:
        # The prompt is part of the cache key, so editing it invalidates old answers.
        prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()[:12]
        return {"model": self.model, "prompt": prompt_hash}

    def decide(self, context: MarketContext) -> AgentResponse:
        return parse_agent_json(self._complete(self.system_prompt, context.to_prompt()))


class LLMAgentStrategy(AgentStrategy):
    """Not registered by default because it needs a ``complete`` function from code."""

    name = "llm_agent"

    def build_agent(self, complete: Callable[[str, str], str], **kwargs: Any) -> Agent:
        return LLMAgent(complete, **kwargs)
