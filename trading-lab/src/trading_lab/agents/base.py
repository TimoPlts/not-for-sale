"""Agent interface.

An agent is a decision maker that looks at a ``MarketContext`` and answers
with a direction, a confidence and a rationale. Agents are wrapped by
``AgentStrategy`` and only ever produce *signals*: they vote in the ensemble
like any other strategy, and the risk manager and paper executor sit between
them and any (simulated) trade.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from trading_lab.agents.context import MarketContext
from trading_lab.core.models import Direction


class AgentResponseError(ValueError):
    """An agent's answer could not be understood or is invalid."""


@dataclass(frozen=True, slots=True)
class AgentResponse:
    direction: Direction
    confidence: float
    rationale: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "direction", Direction(self.direction))
        except ValueError:
            raise AgentResponseError(
                f"direction must be one of buy/sell/hold, got {self.direction!r}"
            ) from None
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise AgentResponseError(f"confidence must be a number in [0, 1], got {self.confidence!r}")
        object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.rationale, str):
            raise AgentResponseError("rationale must be a string")

    def to_json(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> AgentResponse:
        return cls(data["direction"], data["confidence"], data.get("rationale", ""), data.get("extra", {}))


class Agent(ABC):
    """Base class for agents. Bump ``version`` whenever the behaviour changes,
    so cached answers from the old behaviour are not reused."""

    name: ClassVar[str]
    version: ClassVar[str] = "1"

    @property
    def params(self) -> dict[str, Any]:
        """Settings that influence answers; part of the cache key."""
        return {}

    @abstractmethod
    def decide(self, context: MarketContext) -> AgentResponse:
        """Answer for one decision. May be slow, costly or non-deterministic."""
