"""``AgentStrategy``: the adapter that lets an agent vote like any strategy.

For every decision bar it builds a ``MarketContext`` from candles up to that
bar, gets an answer (from the cache, or by asking the agent, depending on
the mode), and turns it into a standard ``Signal``. The rationale, the cache
status and the context hash go into the signal metadata, and from there into
the SQLite history.

Failures never crash a backtest or a live run. An exception or an invalid
answer from the agent becomes a HOLD signal with the error in its metadata.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from trading_lab.agents.base import Agent, AgentResponse, AgentResponseError
from trading_lab.agents.cache import MemoryResponseCache, ResponseCache, cache_key
from trading_lab.agents.context import build_context, indicator_frame
from trading_lab.config import AGENT_MODES
from trading_lab.core.models import Direction, Signal
from trading_lab.core.symbols import timeframe_to_seconds
from trading_lab.strategies.base import Strategy
from trading_lab.strategies.validation import check_int

# Indicators in the context need this many bars (SMA 50 is the longest).
_INDICATOR_WARMUP = 50


class AgentStrategy(Strategy):
    """Subclasses set ``name`` and implement ``build_agent``."""

    name: ClassVar[str]

    def __init__(self, lookback: int = 30, decision_interval: int = 1, timeframe: str = "1h", **agent_params: Any) -> None:
        self.lookback = check_int("lookback", lookback, minimum=1)
        self.decision_interval = check_int("decision_interval", decision_interval, minimum=1)
        timeframe_to_seconds(timeframe)  # validate
        self.timeframe = timeframe
        self.agent = self.build_agent(**agent_params)
        self._agent_params = dict(agent_params)
        self.mode = "record"
        self.cache: ResponseCache = MemoryResponseCache()
        self.calls = 0  # how many times the agent was actually asked

    def build_agent(self, **params: Any) -> Agent:
        raise NotImplementedError

    def configure(self, *, mode: str, cache: ResponseCache, timeframe: str | None = None) -> None:
        if mode not in AGENT_MODES:
            raise ValueError(f"mode must be one of {AGENT_MODES}")
        self.mode = mode
        self.cache = cache
        if timeframe is not None:
            timeframe_to_seconds(timeframe)
            self.timeframe = timeframe

    @property
    def warmup_bars(self) -> int:
        return max(self.lookback, _INDICATOR_WARMUP)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "decision_interval": self.decision_interval,
            "agent": self.agent.name,
            "agent_version": self.agent.version,
            **self.agent.params,
        }

    # ------------------------------------------------------------- decisions
    def _is_decision_bar(self, ts: pd.Timestamp) -> bool:
        # Bar number since the epoch, so the schedule is the same in any data window.
        bar_number = int(ts.timestamp()) // timeframe_to_seconds(self.timeframe)
        return bar_number % self.decision_interval == 0

    def _answer(self, symbol: str, context) -> tuple[AgentResponse | None, dict[str, Any]]:  # type: ignore[no-untyped-def]
        context_hash = context.fingerprint()
        key = cache_key(
            self.agent.name, self.agent.version, self.agent.params, symbol, self.timeframe,
            context.bar_time, context_hash,
        )
        meta: dict[str, Any] = {"context_hash": context_hash[:16]}
        if self.mode != "live":
            cached = self.cache.get(key)
            if cached is not None:
                return cached, {**meta, "cache": "hit"}
            if self.mode == "replay":
                return None, {**meta, "cache": "miss", "error": "not in cache (replay mode)"}
        self.calls += 1
        try:
            response = self.agent.decide(context)
            if not isinstance(response, AgentResponse):
                raise AgentResponseError(f"agent returned {type(response).__name__}, not AgentResponse")
        except Exception as exc:  # an agent must never crash the simulation
            return None, {**meta, "cache": "none", "error": f"{type(exc).__name__}: {exc}"}
        if self.mode == "record":
            self.cache.put(key, response, meta={
                "agent": self.agent.name, "version": self.agent.version, "symbol": symbol,
                "timeframe": self.timeframe, "bar_time": context.bar_time, "context_hash": context_hash,
            })
            return response, {**meta, "cache": "stored"}
        return response, {**meta, "cache": "none"}

    def _signal_at(
        self, symbol: str, candles: pd.DataFrame, indicators: pd.DataFrame, i: int
    ) -> Signal:
        ts = candles.index[i]
        timestamp = ts.to_pydatetime()
        if i + 1 < self.warmup_bars:
            return self._warmup_signal(symbol, i + 1, timestamp)
        if not self._is_decision_bar(ts):
            return self._make_signal(
                symbol, timestamp, i + 1, Direction.HOLD, 0.0, {"reason": "not a decision bar"}
            )
        context = build_context(symbol, self.timeframe, candles, indicators, i, self.lookback)
        response, meta = self._answer(symbol, context)
        if response is None:
            return self._make_signal(symbol, timestamp, i + 1, Direction.HOLD, 0.0, meta)
        meta["rationale"] = response.rationale
        return self._make_signal(
            symbol, timestamp, i + 1, response.direction, response.confidence, meta
        )

    def _evaluate(self, candles: pd.DataFrame):  # type: ignore[no-untyped-def]
        raise NotImplementedError  # generate_signal/generate_signals are overridden

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        if candles.empty:
            raise ValueError(f"{self.name}: no candles supplied for {symbol}")
        return self._signal_at(symbol, candles, indicator_frame(candles), len(candles) - 1)

    def generate_signals(self, symbol: str, candles: pd.DataFrame) -> list[Signal]:
        indicators = indicator_frame(candles)
        return [self._signal_at(symbol, candles, indicators, i) for i in range(len(candles))]
