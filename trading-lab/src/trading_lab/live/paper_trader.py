"""Live paper trading: real-time public market data, simulated execution.

SAFETY: no orders are ever sent anywhere. The trader reads public candles,
runs the same ``TradingSession`` as the backtester, and records simulated
fills in SQLite.

How a cycle works (``run_cycle``):
  1. Fetch the latest **closed** candles for every symbol.
  2. Feed every bar that closed since the last cycle to the session, in time
     order: ``open_bar`` (fills orders scheduled for that bar at its open),
     then ``close_bar`` (stop-losses, mark to market, new signals). This is
     exactly the backtest model, so after downtime the trader catches up
     bar by bar instead of skipping anything.
  3. If orders were scheduled at the close that was just processed, fill them
     right away at the open of the candle that has just started. The provider
     supplies that open price, which never changes once a candle starts. This
     is the same price a backtest would use, without waiting a whole bar.
  4. Save all new records **and** the trader state (last processed bar,
     scheduled orders, last prices, order counter) in one SQLite transaction.

Resuming (``LivePaperTrader.resume``) loads the run's stored config, rebuilds
the portfolio by replaying its stored fills, restores the saved state and
carries on.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

import pandas as pd

from trading_lab.backtest.engine import build_voting
from trading_lab.config import AppConfig
from trading_lab.core.errors import DataError
from trading_lab.core.models import Decision, DecisionAction, Fill
from trading_lab.core.timeutils import ensure_utc
from trading_lab.data.base import MarketDataProvider, timeframe_delta
from trading_lab.engine import Bar, Intent, TradingSession
from trading_lab.engine.session import RestingLimit
from trading_lab.execution.costs import market_stats_frame, next_bar_stats, stats_series
from trading_lab.llm import LLMProvider
from trading_lab.portfolio import Portfolio
from trading_lab.risk.breakers import BreakerState
from trading_lab.reporting import run_metrics
from trading_lab.storage import SQLiteStore
from trading_lab.strategies import Strategy
from trading_lab.strategy_factory import strategies_for

RUN_KIND = "paper"
_GRACE = timedelta(seconds=5)  # wait a little after a candle closes before fetching it


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class CycleReport:
    checked_at: datetime
    new_bars: int
    last_bar: datetime | None
    fills: tuple[Fill, ...]
    decisions: tuple[Decision, ...]  # everything except HOLD
    equity: float | None
    error: str | None = None
    warning: str | None = None


class LivePaperTrader:
    def __init__(
        self,
        config: AppConfig,
        provider: MarketDataProvider,
        store: SQLiteStore,
        *,
        strategies: Sequence[Strategy] | None = None,
        clock: Callable[[], datetime] = _utcnow,
        run_id: str | None = None,
        notes: str = "",
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._store = store
        self._clock = clock
        self._strategies = (
            list(strategies) if strategies is not None
            else strategies_for(config, llm_provider=llm_provider)
        )
        self._voting = build_voting(config, self._strategies)
        self._step = timeframe_delta(config.market.timeframe)
        self._history = max(s.history_bars for s in self._strategies)

        existing = store.get_run(run_id) if run_id else None
        if existing is not None:
            self._restore(existing)
        else:
            self.run_id = run_id or f"pp-{uuid.uuid4().hex[:12]}"
            store.create_run(
                self.run_id,
                kind=RUN_KIND,
                timeframe=config.market.timeframe,
                symbols=config.market.symbols,
                exchange=provider.name,
                config=config.to_dict(),
                config_fingerprint=config.fingerprint(),
                period_start=ensure_utc(clock()),
                notes=notes,
            )
            self._session = TradingSession(config, self._voting, id_prefix="pp")
            self._last_processed: pd.Timestamp | None = None
            self._persisted_trades = 0

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def resume(
        cls,
        store: SQLiteStore,
        run_id: str,
        provider: MarketDataProvider,
        **kwargs: Any,
    ) -> LivePaperTrader:
        """Continue a stored paper run with exactly the config it was started with."""
        run = store.get_run(run_id)
        if run is None:
            raise ValueError(f"unknown run id {run_id!r}")
        if run["kind"] != RUN_KIND:
            raise ValueError(f"run {run_id} is a {run['kind']} run, not a paper run")
        return cls(AppConfig.from_dict(run["config"]), provider, store, run_id=run_id, **kwargs)

    def _restore(self, run: dict[str, Any]) -> None:
        self.run_id = run["run_id"]
        if run["config_fingerprint"] != self._config.fingerprint():
            raise ValueError(
                "config differs from the one this run was started with; "
                "use LivePaperTrader.resume() to load the stored config"
            )
        cfg = self._config
        portfolio = Portfolio(cfg.portfolio.initial_cash, cfg.portfolio.quote_currency)
        for fill in self._store.load_fill_objects(self.run_id):  # event-sourced rebuild
            portfolio.apply_fill(fill)
        state = self._store.load_state(self.run_id) or {}
        self._session = TradingSession(
            cfg,
            self._voting,
            portfolio=portfolio,
            id_prefix="pp",
            order_sequence=state.get("order_sequence", self._store.count("orders", self.run_id)),
            pending={s: Intent.from_json(v) for s, v in state.get("pending", {}).items()},
            last_close=state.get("last_close", {}),
            breaker_state=BreakerState.from_json(state["breakers"]) if "breakers" in state else None,
            resting={s: RestingLimit.from_json(v) for s, v in state.get("resting", {}).items()},
            stop_events=[(datetime.fromisoformat(t), sym) for t, sym in state.get("stop_events", [])],
        )
        last = state.get("last_processed")
        self._last_processed = pd.Timestamp(last) if last else None
        self._persisted_trades = self._store.count("closed_trades", self.run_id)
        if self._persisted_trades != len(portfolio.closed_trades):
            raise RuntimeError("stored trades do not match the replayed fills; database is inconsistent")
        self._store.set_run_status(self.run_id, "running")

    def stop(self) -> None:
        """Save final metrics and mark the run as stopped (it can be resumed later)."""
        metrics = run_metrics(self._store, self.run_id)
        with self._store.atomic():
            if metrics is not None:
                self._store.save_metrics(self.run_id, metrics.to_dict())
            self._store.finish_run(self.run_id, "stopped")

    # ------------------------------------------------------------------ state
    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def portfolio(self) -> Portfolio:
        return self._session.portfolio

    @property
    def last_processed(self) -> datetime | None:
        return None if self._last_processed is None else self._last_processed.to_pydatetime()

    @property
    def pending_symbols(self) -> tuple[str, ...]:
        return tuple(self._session.pending)

    def _state(self) -> dict[str, Any]:
        return {
            "last_processed": None if self._last_processed is None else self._last_processed.isoformat(),
            "pending": {s: intent.to_json() for s, intent in self._session.pending.items()},
            "last_close": dict(self._session.last_close),
            "order_sequence": self._session.executor.sequence,
            "breakers": self._session.breakers.state.to_json(),
            "resting": {s: order.to_json() for s, order in self._session.resting.items()},
            "stop_events": [[t.isoformat(), sym] for t, sym in self._session.stop_events],
        }

    # ------------------------------------------------------------------ cycle
    def run_cycle(self) -> CycleReport:
        cfg = self._config
        tf = cfg.market.timeframe
        now = ensure_utc(self._clock())
        since = now - (self._history + 2) * self._step
        if self._last_processed is not None:
            since = min(since, self._last_processed.to_pydatetime() - self._history * self._step)

        try:
            candles = {sym: self._provider.fetch_ohlcv(sym, tf, since) for sym in cfg.market.symbols}
        except DataError as exc:
            return CycleReport(now, 0, self.last_processed, (), (), self._equity(), error=str(exc))

        timeline = sorted(set().union(*(frame.index for frame in candles.values())))
        if not timeline:
            return CycleReport(now, 0, None, (), (), self._equity(), error="no closed candles available")
        if self._last_processed is None:
            new_bars = timeline[-1:]  # start trading from the most recent closed candle
        else:
            new_bars = [t for t in timeline if t > self._last_processed]

        warning = None
        if self._last_processed is not None and new_bars and new_bars[0] > self._last_processed + self._step:
            warning = f"no candles between {self._last_processed} and {new_bars[0]} (exchange gap?)"

        fills_before = len(self.portfolio.fills)
        if new_bars:
            position = {sym: {ts: i for i, ts in enumerate(candles[sym].index)} for sym in candles}
            # Strategies are evaluated for the new bars only: an AI agent is never asked
            # again about bars processed in earlier cycles (or before a resume).
            # Portfolio-aware strategies are evaluated inside the session, bar by bar.
            new_positions = {
                sym: [i for t in new_bars if (i := position[sym].get(t)) is not None]
                for sym in cfg.market.symbols
            }
            precomputed = {
                sym: [
                    None if s.uses_portfolio
                    else dict(zip(new_positions[sym], s.generate_signals_at(sym, candles[sym], new_positions[sym])))
                    for s in self._strategies
                ]
                for sym in cfg.market.symbols
            }

            def sources(sym: str, i: int) -> list:  # type: ignore[type-arg]
                out = []
                for strategy, ready in zip(self._strategies, precomputed[sym]):
                    if ready is None:
                        out.append(lambda view, s=strategy: s.signal_at(sym, candles[sym], i, view))
                    else:
                        out.append(ready[i])
                return out

            lookback = cfg.execution.volume_lookback
            market_stats = {
                sym: stats_series(market_stats_frame(frame, lookback)) for sym, frame in candles.items()
            }
            columns = ("open", "high", "low", "close", "volume")
            for t in new_bars:
                ts = t.to_pydatetime()
                idx = {sym: i for sym in cfg.market.symbols if (i := position[sym].get(t)) is not None}
                bars = {
                    sym: Bar(*(float(candles[sym][c].iloc[i]) for c in columns)) for sym, i in idx.items()
                }
                stats = {sym: market_stats[sym][i] for sym, i in idx.items()}
                self._session.open_bar(ts, {sym: b.open for sym, b in bars.items()}, stats)
                self._session.close_bar(ts, bars, {sym: sources(sym, i) for sym, i in idx.items()}, stats)
                self._last_processed = t

        # Fill freshly scheduled orders at the open of the candle that just started.
        # Sizing marks every position at that same open, as the backtest does, so
        # the open is needed for each scheduled symbol and each open position.
        # If one is missing, wait for the bar to close instead (same result,
        # just later).
        if self._session.pending and self._last_processed is not None:
            next_bar = self._last_processed + self._step
            needed = set(self._session.pending) | set(self.portfolio.positions)
            opens: dict[str, float] = {}
            for sym in cfg.market.symbols:
                try:
                    price = self._provider.current_open(sym, tf, next_bar.to_pydatetime())
                except DataError:
                    price = None
                if price is not None:
                    opens[sym] = price
            if needed <= set(opens):
                lookback = cfg.execution.volume_lookback
                stats = {sym: next_bar_stats(candles[sym], lookback) for sym in opens}
                self._session.open_bar(next_bar.to_pydatetime(), opens, stats)

        records = self._session.drain()
        new_trades = self.portfolio.closed_trades[self._persisted_trades :]
        with self._store.atomic():
            self._store.add_signals(self.run_id, records.signals)
            self._store.add_decisions(self.run_id, records.decisions)
            self._store.add_execution_reports(self.run_id, records.reports)
            self._store.add_snapshots(self.run_id, records.snapshots)
            self._store.add_closed_trades(self.run_id, new_trades)
            self._store.add_bars(self.run_id, records.bars)
            self._store.save_state(self.run_id, self._state())
        self._persisted_trades += len(new_trades)

        return CycleReport(
            checked_at=now,
            new_bars=len(new_bars),
            last_bar=self.last_processed,
            fills=self.portfolio.fills[fills_before:],
            decisions=tuple(d for d in records.decisions if d.action is not DecisionAction.HOLD),
            equity=self._equity(),
            warning=warning,
        )

    def _equity(self) -> float | None:
        try:
            return self.portfolio.equity(self._session.last_close)
        except Exception:
            return None

    # ------------------------------------------------------------------- loop
    def seconds_until_next_check(self, poll_seconds: float) -> float:
        """Sleep until the next candle should have closed, or ``poll_seconds`` while waiting for one."""
        if self._last_processed is None:
            return poll_seconds
        next_close = (self._last_processed + 2 * self._step).to_pydatetime() + _GRACE
        wait = (next_close - ensure_utc(self._clock())).total_seconds()
        return wait if wait > 0 else poll_seconds

    def run_forever(
        self,
        *,
        poll_seconds: float = 30.0,
        max_cycles: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        on_cycle: Callable[[CycleReport], None] | None = None,
        on_wait: Callable[[float], None] | None = None,
    ) -> int:
        """Run cycles until ``max_cycles`` or Ctrl+C. Always leaves the run resumable."""
        cycles = 0
        try:
            while max_cycles is None or cycles < max_cycles:
                report = self.run_cycle()
                cycles += 1
                if on_cycle is not None:
                    on_cycle(report)
                if max_cycles is not None and cycles >= max_cycles:
                    break
                wait = poll_seconds if report.error else self.seconds_until_next_check(poll_seconds)
                if on_wait is not None:
                    on_wait(wait)
                sleep(wait)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        return cycles
