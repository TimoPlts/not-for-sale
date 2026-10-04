"""The per-bar trading logic shared by backtesting and live paper trading.

``TradingSession`` owns the simulated portfolio, the paper executor, the risk
manager and the scheduled orders. Callers feed it bars in time order:

    session.open_bar(t, opens)                     # fill orders scheduled for bar t
    session.close_bar(t, bars, strategy_signals)   # stops, mark to market, new signals

Because the backtester and the live paper trader both drive this one object,
a live paper run follows exactly the same rules as a backtest over the same
period:

  * Signals are computed at a bar's close. The resulting orders fill at the
    open of that symbol's next bar, through the cost model.
  * Exits fill before entries. Entries are sized at fill time and processed
    in descending signal confidence.
  * A stop-loss triggers when a bar's low reaches the stop. It fills at the
    stop price, or at the open if the bar gapped below it.

``open_bar`` only touches scheduled orders, so calling it again for the same
bar is harmless. Every output (signals, decisions, execution reports,
snapshots) is appended to lists that callers can read or ``drain()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from trading_lab.config import AppConfig
from trading_lab.data.base import timeframe_delta
from trading_lab.core.models import (
    Decision,
    DecisionAction,
    Direction,
    ExecutionReport,
    PortfolioSnapshot,
    Signal,
)
from trading_lab.ensemble import VotingEngine
from trading_lab.execution import CostModel, PaperExecutor
from trading_lab.portfolio import Portfolio
from trading_lab.risk import RiskManager
from trading_lab.risk.breakers import BreakerState, CircuitBreakers


PORTFOLIO = "PORTFOLIO"  # symbol used for portfolio-level decisions


@dataclass(frozen=True, slots=True)
class Bar:
    open: float
    low: float
    close: float


@dataclass(frozen=True, slots=True)
class Intent:
    """An order scheduled at a bar's close, to fill at the next bar's open."""

    action: DecisionAction  # ENTER_SIGNAL or EXIT_SIGNAL
    signal: Signal

    def to_json(self) -> dict[str, Any]:
        s = self.signal
        return {
            "action": self.action.value,
            "signal": {
                "strategy": s.strategy,
                "symbol": s.symbol,
                "direction": s.direction.value,
                "confidence": s.confidence,
                "timestamp": s.timestamp.isoformat(),
                "metadata": _plain(s.metadata),
            },
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Intent:
        sig = data["signal"]
        return cls(
            DecisionAction(data["action"]),
            Signal(
                sig["strategy"],
                sig["symbol"],
                Direction(sig["direction"]),
                sig["confidence"],
                datetime.fromisoformat(sig["timestamp"]),
                sig.get("metadata", {}),
            ),
        )


def _plain(value: Any) -> Any:
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass(slots=True)
class SessionRecords:
    signals: list[Signal] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    reports: list[ExecutionReport] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)
    in_market: list[bool] = field(default_factory=list)


class TradingSession:
    def __init__(
        self,
        config: AppConfig,
        voting: VotingEngine,
        *,
        portfolio: Portfolio | None = None,
        id_prefix: str = "bt",
        order_sequence: int = 0,
        pending: Mapping[str, Intent] | None = None,
        last_close: Mapping[str, float] | None = None,
        breaker_state: BreakerState | None = None,
    ) -> None:
        self.config = config
        self.voting = voting
        costs = CostModel.from_config(config.execution)
        self.portfolio = portfolio or Portfolio(
            config.portfolio.initial_cash, config.portfolio.quote_currency
        )
        self.executor = PaperExecutor(
            self.portfolio,
            costs,
            min_notional=config.execution.min_notional,
            id_prefix=id_prefix,
            start_sequence=order_sequence,
        )
        self.risk = RiskManager(config.risk, costs, min_notional=config.execution.min_notional)
        self.breakers = CircuitBreakers(
            config.risk,
            timeframe_delta(config.market.timeframe),
            self.portfolio.initial_cash,
            breaker_state,
        )
        self.pending: dict[str, Intent] = dict(pending or {})
        self.last_close: dict[str, float] = dict(last_close or {})
        self.records = SessionRecords()
        self._order = {s: i for i, s in enumerate(config.market.symbols)}

    def drain(self) -> SessionRecords:
        """Return the records accumulated so far and start a fresh batch."""
        records, self.records = self.records, SessionRecords()
        return records

    # ------------------------------------------------------------------ steps
    def open_bar(self, ts: datetime, opens: Mapping[str, float]) -> None:
        """Fill orders scheduled for this bar at its open price (exits first)."""
        marks = {**self.last_close, **opens}
        for sym in sorted(self.pending, key=self._order.__getitem__):
            intent = self.pending[sym]
            if intent.action is DecisionAction.EXIT_SIGNAL and sym in opens:
                del self.pending[sym]
                self._exit(sym, opens[sym], ts, DecisionAction.EXIT, "signal exit", intent.signal)

        entry_syms = sorted(
            (s for s, it in self.pending.items()
             if it.action is DecisionAction.ENTER_SIGNAL and s in opens),
            key=lambda s: (-self.pending[s].signal.confidence, self._order[s]),
        )
        for sym in entry_syms:
            sig = self.pending.pop(sym).signal
            blocked = self.breakers.entry_block_reason(sym, ts)
            if blocked is not None:
                self.records.decisions.append(
                    Decision(ts, sym, DecisionAction.REJECTED, blocked, sig.direction,
                             sig.confidence, reference_price=opens[sym])
                )
                continue
            decision = self.risk.evaluate_entry(sym, opens[sym], self.portfolio, marks)
            if not decision.approved:
                self.records.decisions.append(
                    Decision(ts, sym, DecisionAction.REJECTED, decision.reason, sig.direction,
                             sig.confidence, reference_price=opens[sym], details=decision.sizing)
                )
                continue
            report = self.executor.submit(decision.to_order(ts, reason="enter"), opens[sym])
            self.records.reports.append(report)
            self.records.decisions.append(
                Decision(
                    ts, sym, DecisionAction.ENTER if report.filled else DecisionAction.REJECTED,
                    decision.reason if report.filled else report.reason,
                    sig.direction, sig.confidence, quantity=decision.quantity,
                    reference_price=opens[sym], stop_price=decision.stop_price,
                    order_id=report.order_id, details=decision.sizing,
                )
            )

    def close_bar(
        self,
        ts: datetime,
        bars: Mapping[str, Bar],
        strategy_signals: Mapping[str, Sequence[Signal]],
    ) -> None:
        """Check stops, mark to market at the close, and schedule new orders."""
        for sym in sorted(bars, key=self._order.__getitem__):
            bar = bars[sym]
            position = self.portfolio.position(sym)
            if position is not None and self.risk.stop_triggered(position, bar.low):
                stop = position.stop_price
                assert stop is not None
                self._exit(
                    sym, min(bar.open, stop), ts, DecisionAction.STOP_LOSS,
                    f"stop {stop:.8g} hit (bar low {bar.low:.8g})",
                )
                self.breakers.on_stop_loss(sym, ts)

        for sym, bar in bars.items():
            self.last_close[sym] = bar.close
        snapshot = self.portfolio.snapshot(self.last_close, ts)
        self.records.snapshots.append(snapshot)
        self.records.in_market.append(bool(self.portfolio.positions))

        was_halted = self.breakers.halted
        for event in self.breakers.on_bar_close(ts, snapshot.equity):
            self.records.decisions.append(
                Decision(ts, PORTFOLIO, DecisionAction.CIRCUIT_BREAKER, event)
            )
        if self.breakers.halted and not was_halted and self.config.risk.flatten_on_halt:
            self._flatten(ts)

        for sym in sorted(bars, key=self._order.__getitem__):
            strat_sigs = list(strategy_signals[sym])
            ensemble = self.voting.combine(strat_sigs)
            self.records.signals.extend(strat_sigs)
            self.records.signals.append(ensemble)
            self._schedule(sym, ts, ensemble)

    def expire_pending(self, ts: datetime) -> None:
        for sym in sorted(self.pending, key=self._order.__getitem__):
            intent = self.pending.pop(sym)
            self.records.decisions.append(
                Decision(ts, sym, DecisionAction.EXPIRED, "data ended before the order could fill",
                         intent.signal.direction, intent.signal.confidence)
            )

    def liquidate(self, ts: datetime) -> None:
        """Close every position at its last close and refresh the latest snapshot."""
        if not self.portfolio.positions:
            return
        for sym in sorted(self.portfolio.positions, key=self._order.__getitem__):
            self._exit(sym, self.last_close[sym], ts, DecisionAction.LIQUIDATE, "end of backtest")
        snapshot = self.portfolio.snapshot(self.last_close, ts)
        if self.records.snapshots and self.records.snapshots[-1].timestamp == snapshot.timestamp:
            self.records.snapshots[-1] = snapshot
            self.records.in_market[-1] = bool(self.portfolio.positions)
        else:
            self.records.snapshots.append(snapshot)
            self.records.in_market.append(bool(self.portfolio.positions))

    # --------------------------------------------------------------- helpers
    def _flatten(self, ts: datetime) -> None:
        """Schedule exits for every position at the next open (kill switch)."""
        for sym in sorted(self.portfolio.positions, key=self._order.__getitem__):
            signal = Signal("risk_manager", sym, Direction.SELL, 1.0, ts, {"reason": "kill switch"})
            self.pending[sym] = Intent(DecisionAction.EXIT_SIGNAL, signal)
            self.records.decisions.append(
                Decision(ts, sym, DecisionAction.EXIT_SIGNAL, "kill switch: closing position",
                         Direction.SELL, 1.0)
            )

    def _schedule(self, sym: str, ts: datetime, ensemble: Signal) -> None:
        has_position = self.portfolio.position(sym) is not None
        d, c = ensemble.direction, ensemble.confidence
        decisions = self.records.decisions
        existing = self.pending.get(sym)
        if existing is not None and existing.signal.strategy == "risk_manager":
            return  # a kill-switch exit is scheduled; strategies cannot override it
        if d is Direction.BUY:
            if has_position and not self.config.risk.allow_pyramiding:
                decisions.append(Decision(ts, sym, DecisionAction.IGNORED,
                                          "BUY signal but position already open", d, c))
            else:
                self.pending[sym] = Intent(DecisionAction.ENTER_SIGNAL, ensemble)
                decisions.append(Decision(ts, sym, DecisionAction.ENTER_SIGNAL,
                                          "entry scheduled for next bar open", d, c))
        elif d is Direction.SELL:
            if has_position:
                self.pending[sym] = Intent(DecisionAction.EXIT_SIGNAL, ensemble)
                decisions.append(Decision(ts, sym, DecisionAction.EXIT_SIGNAL,
                                          "exit scheduled for next bar open", d, c))
            else:
                decisions.append(Decision(ts, sym, DecisionAction.IGNORED,
                                          "SELL signal but no position (long-only)", d, c))
        else:
            decisions.append(Decision(ts, sym, DecisionAction.HOLD, "ensemble HOLD", d, c))

    def _exit(
        self, sym: str, ref: float, ts: datetime, action: DecisionAction, reason: str,
        signal: Signal | None = None,
    ) -> None:
        decision = self.risk.evaluate_exit(sym, self.portfolio, reason)
        sig_dir = signal.direction if signal else None
        sig_conf = signal.confidence if signal else None
        if not decision.approved:
            self.records.decisions.append(
                Decision(ts, sym, DecisionAction.REJECTED, decision.reason, sig_dir, sig_conf)
            )
            return
        report = self.executor.submit(decision.to_order(ts, reason=action.value), ref)
        self.records.reports.append(report)
        self.records.decisions.append(
            Decision(
                ts, sym, action if report.filled else DecisionAction.REJECTED,
                reason if report.filled else report.reason, sig_dir, sig_conf,
                quantity=decision.quantity, reference_price=ref, order_id=report.order_id,
            )
        )
