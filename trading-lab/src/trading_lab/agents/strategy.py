"""``AgentStrategy``: the adapter that lets an agent vote like any strategy.

For every decision bar it builds a ``MarketContext`` from candles up to that
bar, gets an answer (from the cache, or by asking the agent, depending on
the mode), and turns it into a standard ``Signal``. The rationale, any extra
labels (e.g. ``regime``), the cache status and the context hash go into the
signal metadata, and from there into the SQLite history.

Signal metadata written for every decision bar:

* ``cache``: ``hit`` (answer reused), ``stored`` (asked and cached, record
  mode), ``miss`` (not cached and no usable answer: replay mode, or the call
  failed in record mode) or ``none`` (live mode, no cache involved)
* ``called``: present and true when the agent itself was asked
* ``rationale`` and the agent's labels, or ``error`` when no answer is usable

With ``portfolio_context = true`` the context also contains the simulated
portfolio state (position, exposure, breakers, recent stop-outs). Engines
then evaluate the agent bar by bar inside the trading session. Such contexts
depend on the trading path, so cached answers are shared less between
experiments than for market-only agents.

Failures never crash a backtest or a live run. An exception or an invalid
answer from the agent becomes a HOLD signal with the error in its metadata.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Sequence

import pandas as pd

from trading_lab.agents.base import Agent, AgentResponse, AgentResponseError
from trading_lab.agents.cache import MemoryResponseCache, ResponseCache, cache_key
from trading_lab.agents.context import SIGNIFICANT_DIGITS, MarketContext, build_context, indicator_frame
from trading_lab.config import AGENT_MODES
from trading_lab.core.models import Direction, Signal
from trading_lab.core.symbols import timeframe_to_seconds
from trading_lab.strategies.base import Strategy
from trading_lab.strategies.validation import check_int

# Indicators in the context need this many bars (SMA 50 is the longest).
_INDICATOR_WARMUP = 50


class AgentStrategy(Strategy):
    """Subclasses set ``name`` and implement ``build_agent``.

    They may also override ``context_frame`` (the causal per-bar features shown
    to the agent), ``indicator_warmup`` and ``context_digits``.
    """

    name: ClassVar[str]
    indicator_warmup: ClassVar[int] = _INDICATOR_WARMUP
    context_digits: ClassVar[int] = SIGNIFICANT_DIGITS
    default_portfolio_context: ClassVar[bool] = False

    def __init__(
        self,
        lookback: int = 30,
        decision_interval: int = 1,
        timeframe: str = "1h",
        portfolio_context: bool | None = None,
        **agent_params: Any,
    ) -> None:
        self.lookback = check_int("lookback", lookback, minimum=1)
        self.decision_interval = check_int("decision_interval", decision_interval, minimum=1)
        timeframe_to_seconds(timeframe)  # validate
        self.timeframe = timeframe
        if portfolio_context is None:
            portfolio_context = self.default_portfolio_context
        if not isinstance(portfolio_context, bool):
            raise ValueError("portfolio_context must be true or false")
        self.portfolio_context = portfolio_context
        self.agent = self.build_agent(**agent_params)
        self._agent_params = dict(agent_params)
        self.mode = "record"
        self.cache: ResponseCache = MemoryResponseCache()
        self.calls = 0  # how many times the agent was actually asked
        self._frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

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
        return max(self.lookback, self.indicator_warmup)

    @property
    def uses_portfolio(self) -> bool:
        return self.portfolio_context

    @property
    def params(self) -> dict[str, Any]:
        params = {
            "lookback": self.lookback,
            "decision_interval": self.decision_interval,
            "agent": self.agent.name,
            "agent_version": self.agent.version,
            **self.agent.params,
        }
        if self.portfolio_context:
            params["portfolio_context"] = True
        return params

    # --------------------------------------------------------------- features
    def context_frame(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Causal per-bar features shown to the agent (row ``t`` uses candles ``<= t``)."""
        return indicator_frame(candles)

    def _frame_for(self, symbol: str, candles: pd.DataFrame) -> pd.DataFrame:
        # Engines ask bar by bar with the same candle frame; compute features once per frame.
        cached = self._frames.get(symbol)
        if cached is not None and cached[0] is candles:
            return cached[1]
        frame = self.context_frame(candles)
        self._frames[symbol] = (candles, frame)
        return frame

    # ------------------------------------------------------------- decisions
    def _is_decision_bar(self, ts: pd.Timestamp) -> bool:
        # Bar number since the epoch, so the schedule is the same in any data window.
        bar_number = int(ts.timestamp()) // timeframe_to_seconds(self.timeframe)
        return bar_number % self.decision_interval == 0

    def _answer(self, symbol: str, context: MarketContext) -> tuple[AgentResponse | None, dict[str, Any]]:
        context_hash = context.fingerprint()
        key = cache_key(
            self.agent.name, self.agent.version, self.agent.params, symbol, self.timeframe,
            context.bar_time, context_hash,
        )
        meta: dict[str, Any] = {"context_hash": context_hash[:16]}
        if self.mode != "live":
            try:
                cached = self.cache.get(key)
            except Exception as exc:  # a broken cache must not crash the run
                cached = None
                meta["cache_error"] = f"{type(exc).__name__}: {exc}"
            self._on_cache(cached is not None)
            if cached is not None:
                return cached, {**meta, "cache": "hit"}
            if self.mode == "replay":
                return None, {**meta, "cache": "miss", "error": "not in cache (replay mode)"}
        no_answer = "miss" if self.mode == "record" else "none"
        self.calls += 1
        meta["called"] = True
        try:
            response = self.agent.decide(context)
            if not isinstance(response, AgentResponse):
                raise AgentResponseError(f"agent returned {type(response).__name__}, not AgentResponse")
        except Exception as exc:  # an agent must never crash the simulation
            return None, {**meta, **self._call_meta(), "cache": no_answer,
                          "error": f"{type(exc).__name__}: {exc}"}
        meta.update(self._call_meta())
        if self.mode == "record":
            try:
                self.cache.put(key, response, meta={
                    "agent": self.agent.name, "version": self.agent.version, "symbol": symbol,
                    "timeframe": self.timeframe, "bar_time": context.bar_time, "context_hash": context_hash,
                })
            except Exception as exc:
                return response, {**meta, "cache": "miss", "cache_error": f"{type(exc).__name__}: {exc}"}
            return response, {**meta, "cache": "stored"}
        return response, {**meta, "cache": "none"}

    def _call_meta(self) -> dict[str, Any]:
        """Extra facts about the last call (e.g. model usage); see ``ProviderAgent``."""
        return {}

    def _on_cache(self, hit: bool) -> None:
        """Called after every cache lookup (for usage accounting)."""

    def _signal_at(
        self,
        symbol: str,
        candles: pd.DataFrame,
        indicators: pd.DataFrame,
        i: int,
        portfolio: Mapping[str, Any] | None = None,
    ) -> Signal:
        ts = candles.index[i]
        timestamp = ts.to_pydatetime()
        if i + 1 < self.warmup_bars:
            return self._warmup_signal(symbol, i + 1, timestamp)
        if not self._is_decision_bar(ts):
            return self._make_signal(
                symbol, timestamp, i + 1, Direction.HOLD, 0.0, {"reason": "not a decision bar"}
            )
        view = dict(portfolio) if self.portfolio_context and portfolio is not None else None
        context = build_context(
            symbol, self.timeframe, candles, indicators, i, self.lookback, view, self.context_digits
        )
        response, meta = self._answer(symbol, context)
        if response is None:
            return self._make_signal(symbol, timestamp, i + 1, Direction.HOLD, 0.0, meta)
        meta["rationale"] = response.rationale
        for label, value in response.extra.items():
            meta.setdefault(label, value)
        return self._make_signal(
            symbol, timestamp, i + 1, response.direction, response.confidence, meta
        )

    def _evaluate(self, candles: pd.DataFrame):  # type: ignore[no-untyped-def]
        raise NotImplementedError  # generate_signal/generate_signals are overridden

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal:
        if candles.empty:
            raise ValueError(f"{self.name}: no candles supplied for {symbol}")
        return self._signal_at(symbol, candles, self.context_frame(candles), len(candles) - 1)

    def generate_signals(self, symbol: str, candles: pd.DataFrame) -> list[Signal]:
        return self.generate_signals_at(symbol, candles, range(len(candles)))

    def generate_signals_at(
        self, symbol: str, candles: pd.DataFrame, positions: Sequence[int]
    ) -> list[Signal]:
        frame = self._frame_for(symbol, candles)
        return [self._signal_at(symbol, candles, frame, i) for i in positions]

    def signal_at(
        self, symbol: str, candles: pd.DataFrame, position: int, portfolio: Mapping[str, Any] | None = None
    ) -> Signal:
        return self._signal_at(symbol, candles, self._frame_for(symbol, candles), position, portfolio)
