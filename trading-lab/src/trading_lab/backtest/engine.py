"""Event-driven, bar-by-bar backtesting engine.

Timing model (no look-ahead), implemented by ``engine.TradingSession``:
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
from typing import Sequence

import numpy as np
import pandas as pd

from trading_lab.config import AppConfig
from trading_lab.core.errors import DataError
from trading_lab.core.models import (
    ClosedTrade,
    Decision,
    ExecutionReport,
    Fill,
    PortfolioSnapshot,
    Signal,
)
from trading_lab.core.timeutils import ensure_utc
from trading_lab.data.base import MarketDataProvider, timeframe_delta
from trading_lab.engine import Bar, TradingSession
from trading_lab.ensemble import VotingEngine
from trading_lab.execution import CostModel
from trading_lab.metrics import PerformanceMetrics, compute_metrics
from trading_lab.metrics.benchmark import buy_and_hold_equity
from trading_lab.storage import SQLiteStore
from trading_lab.strategies import Strategy
from trading_lab.strategy_factory import strategies_for

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
    benchmark: PerformanceMetrics | None = None  # equal-weight buy and hold, same costs
    benchmark_curve: pd.Series | None = None

    def stored_metrics(self) -> dict:
        data = self.metrics.to_dict()
        if self.benchmark is not None:
            data["benchmark"] = self.benchmark.to_dict()
        return data

    def actions(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.decisions:
            counts[d.action.value] = counts.get(d.action.value, 0) + 1
        return counts


def equity_frame(snapshots: Sequence[PortfolioSnapshot]) -> pd.DataFrame:
    return pd.DataFrame(
        [{c: getattr(s, c) for c in SNAPSHOT_COLUMNS} for s in snapshots],
        index=pd.DatetimeIndex([s.timestamp for s in snapshots], name="timestamp"),
    )


def build_voting(config: AppConfig, strategies: Sequence[Strategy]) -> VotingEngine:
    """Voting weights from the config, defaulting to 1.0 for strategies not in it."""
    names = [s.name for s in strategies]
    if not names:
        raise ValueError("at least one strategy is required")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate strategy names: {names}")
    configured = {s.name: s.weight for s in config.strategies}
    return VotingEngine({n: configured.get(n, 1.0) for n in names}, config.voting)


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
            list(strategies) if strategies is not None else strategies_for(config)
        )
        self._voting = build_voting(config, self._strategies)

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
        with store.atomic():
            store.add_signals(run_id, result.signals)
            store.add_decisions(run_id, result.decisions)
            store.add_execution_reports(run_id, result.reports)
            store.add_snapshots(run_id, result.snapshots)
            store.add_closed_trades(run_id, result.trades)
            store.save_metrics(run_id, result.stored_metrics())

    def _simulate(self, start: datetime, end: datetime, run_id: str | None) -> BacktestResult:
        cfg = self._config
        symbols = list(cfg.market.symbols)
        candles = self._load(start, end)

        # Signals are computed in one vectorised pass per strategy. This is
        # equivalent to bar-by-bar evaluation because indicators are causal
        # (enforced by tests).
        strategy_signals = {
            sym: [s.generate_signals(sym, candles[sym]) for s in self._strategies]
            for sym in symbols
        }
        position = {sym: {ts: i for i, ts in enumerate(candles[sym].index)} for sym in symbols}
        ohlc = {
            sym: (frame["open"].to_numpy(), frame["low"].to_numpy(), frame["close"].to_numpy())
            for sym, frame in candles.items()
        }

        start_ts = pd.Timestamp(start)
        timeline = sorted(
            set().union(*(frame.index[frame.index >= start_ts] for frame in candles.values()))
        )
        if not timeline:
            raise DataError("no candles inside the backtest period")

        # Last close before the period starts, so every symbol always has a mark.
        last_close: dict[str, float] = {}
        for sym, frame in candles.items():
            before = np.searchsorted(frame.index, start_ts) - 1
            if before >= 0:
                last_close[sym] = float(frame["close"].iloc[before])

        session = TradingSession(cfg, self._voting, id_prefix="bt", last_close=last_close)
        for t in timeline:
            ts = t.to_pydatetime()
            idx = {sym: i for sym in symbols if (i := position[sym].get(t)) is not None}
            bars = {
                sym: Bar(float(ohlc[sym][0][i]), float(ohlc[sym][1][i]), float(ohlc[sym][2][i]))
                for sym, i in idx.items()
            }
            session.open_bar(ts, {sym: bar.open for sym, bar in bars.items()})
            session.close_bar(
                ts, bars, {sym: [per_bar[i] for per_bar in strategy_signals[sym]] for sym, i in idx.items()}
            )

        last_ts = timeline[-1].to_pydatetime()
        session.expire_pending(last_ts)
        if cfg.backtest.liquidate_at_end:
            session.liquidate(last_ts)

        records = session.records
        portfolio = session.portfolio
        equity_curve = equity_frame(records.snapshots)
        metrics = compute_metrics(
            [portfolio.initial_cash, *equity_curve["equity"].tolist()],
            portfolio.closed_trades,
            cfg.market.timeframe,
            total_fees=portfolio.fees_paid,
            in_market=records.in_market,
        )
        benchmark_curve = buy_and_hold_equity(
            candles, timeline, portfolio.initial_cash, CostModel.from_config(cfg.execution)
        )
        benchmark = compute_metrics(
            [portfolio.initial_cash, *benchmark_curve.tolist()],
            [],
            cfg.market.timeframe,
            in_market=[True] * len(benchmark_curve),
        )
        return BacktestResult(
            run_id=run_id,
            start=start,
            end=end,
            config_fingerprint=cfg.fingerprint(),
            metrics=metrics,
            equity_curve=equity_curve,
            snapshots=tuple(records.snapshots),
            trades=portfolio.closed_trades,
            fills=portfolio.fills,
            reports=tuple(records.reports),
            decisions=tuple(records.decisions),
            signals=tuple(records.signals),
            benchmark=benchmark,
            benchmark_curve=benchmark_curve,
        )
