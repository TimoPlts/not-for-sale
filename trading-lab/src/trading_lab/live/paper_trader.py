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

import sqlite3
import threading
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
from trading_lab.engine.filters import correlation_lookup, filter_columns
from trading_lab.engine.session import RestingLimit
from trading_lab.execution.costs import market_stats_frame, next_bar_stats, stats_series
from trading_lab.alerts import AlertManager
from trading_lab.llm import LLMProvider
from trading_lab.portfolio import Portfolio
from trading_lab.risk.breakers import BreakerState
from trading_lab.reporting import run_metrics
from trading_lab.storage import SQLiteStore
from trading_lab.strategies import Strategy
from trading_lab.strategy_factory import attach_context_feeds, strategies_for

RUN_KIND = "paper"
_GRACE = timedelta(seconds=5)  # wait a little after a candle closes before fetching it
MAX_ERROR_BACKOFF_SECONDS = 900.0  # longest wait between retries during an outage


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
    agent_errors: int = 0  # agent decisions in this cycle without a usable answer (voted HOLD)


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
        alerts: AlertManager | None = None,
    ) -> None:
        self._config = config
        self.alerts = alerts
        self._outage_alerted = False
        self._model_paused = False
        self._last_summary_day: str | None = None
        self._provider = provider
        self._store = store
        self._clock = clock
        self._strategies = (
            list(strategies) if strategies is not None
            else strategies_for(config, llm_provider=llm_provider)
        )
        attach_context_feeds(self._strategies, provider, config.market.timeframe)
        self._voting = build_voting(config, self._strategies)
        self._step = timeframe_delta(config.market.timeframe)
        voting = config.voting
        self._history = max(max(s.history_bars for s in self._strategies), config.risk.trend_filter_period + 1,
                            config.risk.correlation_lookback + 2 if config.risk.max_correlated_positions else 0,
                            voting.regime_bars + voting.regime_slope_bars + 1 if voting.regime_weights else 0)

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
        self.consecutive_errors = 0
        self._last_error: str | None = None

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
        # Compare by value: settings added in later versions take their (behaviour-preserving)
        # defaults, so runs started with older code can still be resumed.
        if AppConfig.from_dict(run["config"]) != self._config:
            raise ValueError(
                "config differs from the one this run was started with; "
                "use LivePaperTrader.resume() to load the stored config"
            )
        cfg = self._config
        portfolio = Portfolio(cfg.portfolio.initial_cash, cfg.portfolio.quote_currency,
                              allow_short=cfg.risk.allow_short)
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
            trailing=state.get("trailing", {}),
            risk_states={s: (datetime.fromisoformat(t), v) for s, (t, v) in state.get("risk_states", {}).items()},
        )
        last = state.get("last_processed")
        self._last_processed = pd.Timestamp(last) if last else None
        self._last_summary_day = (state.get("alerts") or {}).get("last_summary_day")
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
            "trailing": {s: dict(v) for s, v in self._session.trailing.items()},
            "risk_states": {s: [t.isoformat(), v] for s, (t, v) in self._session.risk_states.items()},
            "alerts": {"last_summary_day": self._last_summary_day},
            "health": {
                "last_cycle_at": ensure_utc(self._clock()).isoformat(),
                "consecutive_errors": self.consecutive_errors,
                "last_error": self._last_error,
            },
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
        except (DataError, OSError) as exc:  # exchange or network outage: nothing processed, retry later
            return self._failed_cycle(now, f"market data unavailable: {exc}")

        timeline = sorted(set().union(*(frame.index for frame in candles.values())))
        if not timeline:
            return self._failed_cycle(now, "no closed candles available")
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
                sym: stats_series(market_stats_frame(frame, lookback, cfg.risk.atr_period))
                for sym, frame in candles.items()
            }
            filters = {sym: filter_columns(frame, cfg.risk, cfg.voting) for sym, frame in candles.items()}
            correlations = correlation_lookup(candles, cfg.risk)
            columns = ("open", "high", "low", "close", "volume")
            for t in new_bars:
                ts = t.to_pydatetime()
                idx = {sym: i for sym in cfg.market.symbols if (i := position[sym].get(t)) is not None}
                bars = {
                    sym: Bar(*(float(candles[sym][c].iloc[i]) for c in columns)) for sym, i in idx.items()
                }
                stats = {sym: market_stats[sym][i] for sym, i in idx.items()}
                self._session.open_bar(ts, {sym: b.open for sym, b in bars.items()}, stats)
                self._session.close_bar(ts, bars, {sym: sources(sym, i) for sym, i in idx.items()}, stats,
                                        {sym: {**filters[sym][i], "correlations": correlations[sym].get(t, {})}
                                         for sym, i in idx.items()})
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
                except (DataError, OSError):
                    price = None
                if price is not None:
                    opens[sym] = price
            if needed <= set(opens):
                lookback = cfg.execution.volume_lookback
                stats = {sym: next_bar_stats(candles[sym], lookback, cfg.risk.atr_period) for sym in opens}
                self._session.open_bar(next_bar.to_pydatetime(), opens, stats)

        records = self._session.drain()
        new_trades = self.portfolio.closed_trades[self._persisted_trades :]
        summary_day_before = self._last_summary_day
        summary_due = self._summary_due()
        errors_before = (self.consecutive_errors, self._last_error)
        self.consecutive_errors, self._last_error = 0, None
        try:
            with self._store.atomic():
                self._store.add_signals(self.run_id, records.signals)
                self._store.add_decisions(self.run_id, records.decisions)
                self._store.add_execution_reports(self.run_id, records.reports)
                self._store.add_snapshots(self.run_id, records.snapshots)
                self._store.add_closed_trades(self.run_id, new_trades)
                self._store.add_bars(self.run_id, records.bars)
                self._store.save_state(self.run_id, self._state())
        except sqlite3.OperationalError as exc:
            # Nothing of this cycle was saved (one transaction), but the in-memory
            # session already moved on. Never continue from a state the database does
            # not have: reload it, so the same bars are processed again next cycle.
            self.consecutive_errors, self._last_error = errors_before
            self._last_summary_day = summary_day_before
            self._reload()
            self._alert("warning", "Database error: cycle rolled back", f"{exc}. The state was reloaded from "
                        "the database and the bars will be processed again.", key="db_error")
            return self._failed_cycle(now, f"database error, state reloaded and bars will be retried: {exc}",
                                      save_health=False)
        except BaseException:
            self.consecutive_errors, self._last_error = errors_before
            self._last_summary_day = summary_day_before
            raise
        self._persisted_trades += len(new_trades)
        self._after_cycle(records, new_trades, summary_due)

        agent_errors = sum(
            1 for sig in records.signals
            if "error" in sig.metadata and "agent" in (sig.metadata.get("params") or {})
        )
        return CycleReport(
            checked_at=now,
            new_bars=len(new_bars),
            last_bar=self.last_processed,
            fills=self.portfolio.fills[fills_before:],
            decisions=tuple(d for d in records.decisions if d.action is not DecisionAction.HOLD),
            equity=self._equity(),
            warning=warning,
            agent_errors=agent_errors,
        )

    # ----------------------------------------------------------------- alerts
    def _alert(self, level: str, title: str, body: str = "", *, key: str | None = None,
               force: bool = False) -> None:
        if self.alerts is not None:
            self.alerts.emit(level, title, body, key=key, run_id=self.run_id, force=force)

    def _summary_due(self) -> bool:
        """True once per UTC day: the first cycle whose last bar falls on a new day."""
        if self._last_processed is None or self.alerts is None or not self._config.alerts.daily_summary:
            return False
        day = (self._last_processed + self._step).date().isoformat()  # the day the last bar closed in
        if self._last_summary_day is None:
            self._last_summary_day = day  # first day of the run: nothing to summarise yet
            return False
        if day > self._last_summary_day:
            self._last_summary_day = day
            return True
        return False

    def _after_cycle(self, records: Any, new_trades: Sequence[Any], summary_due: bool) -> None:
        if self.alerts is None:
            return
        if self._outage_alerted:
            self._outage_alerted = False
            self._alert("info", "Trading cycles processed again", "The outage is over; missed candles were "
                        "processed in order.", force=True)
            self.alerts.clear("outage")
        for d in records.decisions:
            if d.action is DecisionAction.CIRCUIT_BREAKER:
                critical = d.reason.startswith("max drawdown")
                self._alert("critical" if critical else "warning",
                            "Kill switch tripped" if critical else "Daily loss limit hit",
                            f"{d.reason}. New entries are blocked; exits still work.")
            elif d.action in (DecisionAction.ENTER, DecisionAction.EXIT, DecisionAction.STOP_LOSS,
                              DecisionAction.TAKE_PROFIT):
                self._alert("info", f"{d.action.value.replace('_', ' ')} {d.symbol}", d.reason)
        for trade in new_trades:
            self._alert("info", f"Trade closed {trade.symbol} {trade.pnl:+,.2f}",
                        f"return {trade.return_pct:+.2%}, held {trade.opened_at:%m-%d %H:%M} -> "
                        f"{trade.closed_at:%m-%d %H:%M}")
        agent_signals = [s for s in records.signals if "agent" in (s.metadata.get("params") or {})
                         and "called" in s.metadata]
        paused = [s for s in agent_signals if str(s.metadata.get("error", "")).startswith("ProviderUnavailableError")]
        answered = [s for s in agent_signals if "error" not in s.metadata]
        if paused and not self._model_paused:
            self._model_paused = True
            self._alert("warning", "Model calls paused", f"{paused[0].metadata['error']}. Agents vote HOLD "
                        "until the endpoint answers again.", key="model_paused")
        elif self._model_paused and answered:
            self._model_paused = False
            self.alerts.clear("model_paused")
            self._alert("info", "Model answers again", "The agents are voting normally again.", force=True)
        if summary_due:
            from trading_lab.summary import build_summary, format_summary

            try:
                text = format_summary(build_summary(self._store, self.run_id, hours=24))
            except Exception as exc:  # a summary must never break the trader
                text = f"(summary unavailable: {type(exc).__name__}: {exc})"
            self._alert("info", "Daily summary", text, force=True)

    def _failed_cycle(self, now: datetime, error: str, *, save_health: bool = True) -> CycleReport:
        """A cycle that processed nothing; the trader state is unchanged and is retried later."""
        self.consecutive_errors += 1
        self._last_error = error
        if self.consecutive_errors == self._config.alerts.outage_after_cycles:
            self._outage_alerted = True
            self._alert("warning", f"{self.consecutive_errors} trading cycles failed in a row",
                        f"{error}. Nothing is traded meanwhile; retrying with back-off.", key="outage")
        if save_health:
            try:
                self._store.save_state(self.run_id, self._state())
            except sqlite3.OperationalError:
                pass  # health is best effort; the trading state itself did not change
        return CycleReport(now, 0, self.last_processed, (), (), self._equity(), error=error)

    def _reload(self) -> None:
        """Rebuild the trader from the database (after a failed save)."""
        run = self._store.get_run(self.run_id)
        if run is None:
            raise RuntimeError(f"run {self.run_id} disappeared from the database")
        self._restore(run)

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
        stop_event: threading.Event | None = None,
    ) -> int:
        """Run cycles until ``max_cycles``, Ctrl+C or ``stop_event``. Always leaves the run resumable.

        Setting ``stop_event`` (e.g. from a SIGTERM handler) lets the current
        cycle finish and save, ends any wait at once, and stops the run cleanly.
        """
        cycles = 0
        try:
            while (max_cycles is None or cycles < max_cycles) and not (stop_event and stop_event.is_set()):
                report = self.run_cycle()
                cycles += 1
                if on_cycle is not None:
                    on_cycle(report)
                if max_cycles is not None and cycles >= max_cycles:
                    break
                if report.error:  # outage: back off exponentially, up to 15 minutes
                    wait = min(poll_seconds * 2 ** (self.consecutive_errors - 1), MAX_ERROR_BACKOFF_SECONDS)
                    wait = max(wait, poll_seconds)
                else:
                    wait = self.seconds_until_next_check(poll_seconds)
                if stop_event is not None and stop_event.is_set():
                    break
                if on_wait is not None:
                    on_wait(wait)
                if stop_event is not None:
                    stop_event.wait(wait)
                else:
                    sleep(wait)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        return cycles
