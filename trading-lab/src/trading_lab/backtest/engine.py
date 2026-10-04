"""Event-driven, bar-by-bar backtesting engine.

Timing model (no look-ahead):
  * Bars are keyed by their **open** time. "Bar t" covers ``[t, t + timeframe)``.
  * Signals for bar t are computed at its **close**, from candles up to and
    including t.
  * Orders resulting from those signals fill at the **open of the symbol's
    next bar**, priced through the cost model (slippage and fees). Exits fill
    before entries, so the cash they free up is available. Entries are sized
    by the risk manager at fill time, with the open price, and processed in
    descending signal confidence.
  * Stop-losses: if a bar's low reaches a position's stop, the position exits
    at the stop price, or at the open when the bar gapped below the stop.
    A position entered at a bar's open can be stopped out on that same bar.
  * The equity snapshot for bar t marks positions at bar t's close.

Every signal (each strategy's and the ensemble's), every decision (HOLDs
included), every order and every fill is recorded. When a ``SQLiteStore`` is
given, all of it is persisted under a run id together with the config and
its fingerprint.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from trading_lab.config import AppConfig
from trading_lab.core.errors import DataError
from trading_lab.core.models import (
    ClosedTrade,
    Decision,
    DecisionAction,
    Direction,
    ExecutionReport,
    Fill,
    PortfolioSnapshot,
    Signal,
)
from trading_lab.core.timeutils import ensure_utc
from trading_lab.data.base import MarketDataProvider, timeframe_delta
from trading_lab.ensemble import VotingEngine
from trading_lab.execution import CostModel, PaperExecutor
from trading_lab.metrics import PerformanceMetrics, compute_metrics
from trading_lab.portfolio import Portfolio
from trading_lab.risk import RiskManager
from trading_lab.storage import SQLiteStore
from trading_lab.strategies import Strategy, build_strategies

SNAPSHOT_COLUMNS = (
    "cash",
    "positions_value",
    "equity",
    "realized_pnl",
    "unrealized_pnl",
    "fees_paid",
    "open_positions",
)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    run_id: str | None
    start: datetime
    end: datetime
    config_fingerprint: str
    metrics: PerformanceMetrics
    equity_curve: pd.DataFrame
    snapshots: tuple[PortfolioSnapshot, ...]
    trades: tuple[ClosedTrade, ...]
    fills: tuple[Fill, ...]
    reports: tuple[ExecutionReport, ...]
    decisions: tuple[Decision, ...]
    signals: tuple[Signal, ...]

    def actions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.decisions:
            counts[d.action.value] = counts.get(d.action.value, 0) + 1
        return counts


@dataclass(slots=True)
class _Intent:
    action: DecisionAction  # ENTER_SIGNAL or EXIT_SIGNAL
    signal: Signal


@dataclass(frozen=True, slots=True)
class _Series:
    """Numpy views of one symbol's candles for fast per-bar access."""

    index: pd.DatetimeIndex
    position: Mapping[pd.Timestamp, int]
    open: np.ndarray
    low: np.ndarray
    close: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> _Series:
        return cls(
            index=frame.index,
            position={ts: i for i, ts in enumerate(frame.index)},
            open=frame["open"].to_numpy(),
            low=frame["low"].to_numpy(),
            close=frame["close"].to_numpy(),
        )


class BacktestEngine:
    def __init__(
        self,
        config: AppConfig,
        provider: MarketDataProvider,
        *,
        strategies: Sequence[Strategy] | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._store = store
        self._strategy_override = strategies is not None
        self._strategies = (
            list(strategies) if strategies is not None else build_strategies(config.enabled_strategies)
        )
        if not self._strategies:
            raise ValueError("at least one strategy is required")
        names = [s.name for s in self._strategies]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate strategy names: {names}")
        configured = {s.name: s.weight for s in config.strategies}
        self._voting = VotingEngine({n: configured.get(n, 1.0) for n in names}, config.voting)

    # ------------------------------------------------------------------ data
    def _load(self, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        cfg = self._config
        step = timeframe_delta(cfg.market.timeframe)
        history = max(s.history_bars for s in self._strategies)
        data_start = start - history * step
        candles = {}
        for symbol in cfg.market.symbols:
            frame = self._provider.fetch_ohlcv(symbol, cfg.market.timeframe, data_start, end)
            if frame.empty or frame.index[-1] < pd.Timestamp(start):
                raise DataError(f"no {cfg.market.timeframe} candles for {symbol} in the backtest period")
            candles[symbol] = frame
        return candles

    # ------------------------------------------------------------------- run
    def run(
        self, start: datetime, end: datetime, *, run_id: str | None = None, notes: str = ""
    ) -> BacktestResult:
        start, end = ensure_utc(start, "start"), ensure_utc(end, "end")
        if end <= start:
            raise ValueError("end must be after start")
        cfg = self._config
        if self._store is not None:
            run_id = run_id or f"bt-{uuid.uuid4().hex[:12]}"
            config_dict = cfg.to_dict()
            if self._strategy_override:
                config_dict["strategy_override"] = [
                    {"name": s.name, "params": s.params} for s in self._strategies
                ]
            self._store.create_run(
                run_id,
                kind="backtest",
                timeframe=cfg.market.timeframe,
                symbols=cfg.market.symbols,
                exchange=self._provider.name,
                config=config_dict,
                config_fingerprint=cfg.fingerprint(),
                period_start=start,
                period_end=end,
                notes=notes,
            )
        try:
            result = self._simulate(start, end, run_id)
            if self._store is not None and run_id is not None:
                self._persist(run_id, result)
                self._store.finish_run(run_id, "completed")
            return result
        except Exception as exc:
            if self._store is not None and run_id is not None:
                self._store.finish_run(run_id, "failed", error=f"{type(exc).__name__}: {exc}")
            raise

    def _persist(self, run_id: str, result: BacktestResult) -> None:
        store = self._store
        assert store is not None
        store.add_signals(run_id, result.signals)
        store.add_decisions(run_id, result.decisions)
        store.add_execution_reports(run_id, result.reports)
        store.add_snapshots(run_id, result.snapshots)
        store.add_closed_trades(run_id, result.trades)
        store.save_metrics(run_id, result.metrics.to_dict())

    def _simulate(self, start: datetime, end: datetime, run_id: str | None) -> BacktestResult:
        cfg = self._config
        symbols = list(cfg.market.symbols)
        order = {s: i for i, s in enumerate(symbols)}
        candles = self._load(start, end)
        series = {sym: _Series.from_frame(frame) for sym, frame in candles.items()}

        # Signals are computed in one vectorised pass per strategy. This is
        # equivalent to bar-by-bar evaluation because indicators are causal
        # (enforced by tests).
        strategy_signals = {
            sym: [s.generate_signals(sym, candles[sym]) for s in self._strategies]
            for sym in symbols
        }

        start_ts = pd.Timestamp(start)
        timeline = sorted(
            set().union(*(frame.index[frame.index >= start_ts] for frame in candles.values()))
        )

        costs = CostModel.from_config(cfg.execution)
        portfolio = Portfolio(cfg.portfolio.initial_cash, cfg.portfolio.quote_currency)
        executor = PaperExecutor(portfolio, costs, min_notional=cfg.execution.min_notional, id_prefix="bt")
        risk = RiskManager(cfg.risk, costs, min_notional=cfg.execution.min_notional)

        decisions: list[Decision] = []
        reports: list[ExecutionReport] = []
        signals_out: list[Signal] = []
        snapshots: list[PortfolioSnapshot] = []
        in_market: list[bool] = []
        pending: dict[str, _Intent] = {}

        # Last close before the period starts, so every symbol always has a mark.
        last_close: dict[str, float] = {}
        for sym, s in series.items():
            before = np.searchsorted(s.index, start_ts) - 1
            if before >= 0:
                last_close[sym] = float(s.close[before])

        def exit_position(
            sym: str, ref: float, ts: datetime, action: DecisionAction, reason: str,
            signal: Signal | None = None,
        ) -> None:
            decision = risk.evaluate_exit(sym, portfolio, reason)
            sig_dir = signal.direction if signal else None
            sig_conf = signal.confidence if signal else None
            if not decision.approved:
                decisions.append(Decision(ts, sym, DecisionAction.REJECTED, decision.reason, sig_dir, sig_conf))
                return
            report = executor.submit(decision.to_order(ts, reason=action.value), ref)
            reports.append(report)
            decisions.append(
                Decision(
                    ts, sym, action if report.filled else DecisionAction.REJECTED,
                    reason if report.filled else report.reason, sig_dir, sig_conf,
                    quantity=decision.quantity, reference_price=ref, order_id=report.order_id,
                )
            )

        for t in timeline:
            ts = t.to_pydatetime()
            bar = {sym: i for sym in symbols if (i := series[sym].position.get(t)) is not None}
            opens = {sym: float(series[sym].open[i]) for sym, i in bar.items()}
            marks = {**last_close, **opens}

            # 1. Scheduled exits fill at this bar's open.
            for sym in symbols:
                intent = pending.get(sym)
                if intent and intent.action is DecisionAction.EXIT_SIGNAL and sym in bar:
                    del pending[sym]
                    exit_position(sym, opens[sym], ts, DecisionAction.EXIT, "signal exit", intent.signal)

            # 2. Scheduled entries, most confident first.
            entry_syms = sorted(
                (s for s, it in pending.items() if it.action is DecisionAction.ENTER_SIGNAL and s in bar),
                key=lambda s: (-pending[s].signal.confidence, order[s]),
            )
            for sym in entry_syms:
                intent = pending.pop(sym)
                sig = intent.signal
                decision = risk.evaluate_entry(sym, opens[sym], portfolio, marks)
                if not decision.approved:
                    decisions.append(
                        Decision(ts, sym, DecisionAction.REJECTED, decision.reason, sig.direction,
                                 sig.confidence, reference_price=opens[sym], details=decision.sizing)
                    )
                    continue
                report = executor.submit(decision.to_order(ts, reason="enter"), opens[sym])
                reports.append(report)
                decisions.append(
                    Decision(
                        ts, sym, DecisionAction.ENTER if report.filled else DecisionAction.REJECTED,
                        decision.reason if report.filled else report.reason,
                        sig.direction, sig.confidence, quantity=decision.quantity,
                        reference_price=opens[sym], stop_price=decision.stop_price,
                        order_id=report.order_id, details=decision.sizing,
                    )
                )

            # 3. Stop-losses within this bar.
            for sym, i in bar.items():
                position = portfolio.position(sym)
                if position is not None and risk.stop_triggered(position, float(series[sym].low[i])):
                    stop = position.stop_price
                    assert stop is not None
                    fill_ref = min(opens[sym], stop)  # a gap below the stop fills at the open
                    exit_position(
                        sym, fill_ref, ts, DecisionAction.STOP_LOSS,
                        f"stop {stop:.8g} hit (bar low {series[sym].low[i]:.8g})",
                    )

            # 4. Mark to market at the close.
            for sym, i in bar.items():
                last_close[sym] = float(series[sym].close[i])
            snapshots.append(portfolio.snapshot(last_close, ts))
            in_market.append(bool(portfolio.positions))

            # 5. Signals at the close schedule orders for the next bar.
            for sym, i in bar.items():
                strat_sigs = [per_bar[i] for per_bar in strategy_signals[sym]]
                ensemble = self._voting.combine(strat_sigs)
                signals_out.extend(strat_sigs)
                signals_out.append(ensemble)
                has_position = portfolio.position(sym) is not None
                d, c = ensemble.direction, ensemble.confidence
                if d is Direction.BUY:
                    if has_position and not cfg.risk.allow_pyramiding:
                        decisions.append(Decision(ts, sym, DecisionAction.IGNORED,
                                                  "BUY signal but position already open", d, c))
                    else:
                        pending[sym] = _Intent(DecisionAction.ENTER_SIGNAL, ensemble)
                        decisions.append(Decision(ts, sym, DecisionAction.ENTER_SIGNAL,
                                                  "entry scheduled for next bar open", d, c))
                elif d is Direction.SELL:
                    if has_position:
                        pending[sym] = _Intent(DecisionAction.EXIT_SIGNAL, ensemble)
                        decisions.append(Decision(ts, sym, DecisionAction.EXIT_SIGNAL,
                                                  "exit scheduled for next bar open", d, c))
                    else:
                        decisions.append(Decision(ts, sym, DecisionAction.IGNORED,
                                                  "SELL signal but no position (long-only)", d, c))
                else:
                    decisions.append(Decision(ts, sym, DecisionAction.HOLD, "ensemble HOLD", d, c))

        if not timeline:
            raise DataError("no candles inside the backtest period")
        last_ts = timeline[-1].to_pydatetime()
        for sym, intent in sorted(pending.items(), key=lambda kv: order[kv[0]]):
            decisions.append(
                Decision(last_ts, sym, DecisionAction.EXPIRED, "data ended before the order could fill",
                         intent.signal.direction, intent.signal.confidence)
            )
        if cfg.backtest.liquidate_at_end and portfolio.positions:
            for sym in sorted(portfolio.positions, key=order.__getitem__):
                exit_position(sym, last_close[sym], last_ts, DecisionAction.LIQUIDATE, "end of backtest")
            snapshots[-1] = portfolio.snapshot(last_close, last_ts)
            in_market[-1] = bool(portfolio.positions)

        equity_curve = pd.DataFrame(
            [{c: getattr(s, c) for c in SNAPSHOT_COLUMNS} for s in snapshots],
            index=pd.DatetimeIndex([s.timestamp for s in snapshots], name="timestamp"),
        )
        metrics = compute_metrics(
            [portfolio.initial_cash, *equity_curve["equity"].tolist()],
            portfolio.closed_trades,
            cfg.market.timeframe,
            total_fees=portfolio.fees_paid,
            in_market=in_market,
        )
        return BacktestResult(
            run_id=run_id,
            start=start,
            end=end,
            config_fingerprint=cfg.fingerprint(),
            metrics=metrics,
            equity_curve=equity_curve,
            snapshots=tuple(snapshots),
            trades=portfolio.closed_trades,
            fills=portfolio.fills,
            reports=tuple(reports),
            decisions=tuple(decisions),
            signals=tuple(signals_out),
        )
