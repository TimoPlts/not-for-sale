"""The per-bar trading logic shared by backtesting and live paper trading.

``TradingSession`` owns the simulated portfolio, the paper executor, the risk
manager, the circuit breakers and all working orders. Callers feed it bars in
time order:

    session.open_bar(t, opens, stats)                    # orders scheduled for bar t
    session.close_bar(t, bars, strategy_signals, stats)  # limit fills, stops, marks, signals

Because the backtester and the live paper trader both drive this one object,
a live paper run follows exactly the same rules as a backtest over the same
period:

  * Signals are computed at a bar's close. The resulting orders are handled
    at the open of that symbol's next bar.
  * Exits are market orders and fill before entries. Entries are sized at
    that open and processed in descending signal confidence.
  * Market entries fill at the open through the cost model, which can include
    volume-aware impact based on ``MarketStats`` from earlier bars.
  * Limit entries (``entry_order_type = "limit"``) rest at
    ``open × (1 − limit_offset_bps)``:
      - They fill only when a bar trades *through* the limit (low < limit), at
        the limit price with the maker fee and no slippage.
      - Each bar's fill is capped at ``max_participation_pct`` of that bar's
        volume, which produces partial fills.
      - Whatever is unfilled after ``limit_ttl_bars`` bars expires.
      - An exit signal or the kill switch cancels a resting order.
  * A stop-loss triggers when a bar's low reaches the stop. It fills at the
    stop price, or at the open if the bar gapped below it.
  * Optional take-profit (``risk.take_profit_pct``): when a bar's high reaches
    average cost x (1 + pct), the position exits at that price (or at the open
    if the bar gapped above it). If the stop and the target are both reached in
    one bar, the stop is assumed to come first (the conservative choice).
  * Optional trailing stop (``risk.trailing_stop_pct``): after each bar
    closes, the stop is raised to ``highest high since entry x (1 - pct)``
    (once the high is ``trailing_activation_pct`` above the average cost).
    Stops only move up, and a raised stop applies from the next bar on.
  * The stop-loss cooldown only follows stop exits that lost money.

``open_bar`` only touches scheduled orders, so calling it again for the same
bar is harmless. Every output (signals, decisions, execution reports,
snapshots) is appended to lists that callers can read or ``drain()``.

Signals passed to ``close_bar`` are either ready ``Signal`` objects or
callables ``f(portfolio_view) -> Signal`` for strategies that look at the
portfolio (``Strategy.uses_portfolio``). Callables are evaluated after stops,
marks and circuit breakers for the bar, with a read-only, JSON-safe view of
the simulated portfolio (``portfolio_view``). They can only return a signal;
the voting engine, breakers, risk manager and executor decide the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence, Union

from trading_lab.config import AppConfig
from trading_lab.core.models import (
    Decision,
    DecisionAction,
    Direction,
    ExecutionReport,
    Order,
    OrderType,
    PortfolioSnapshot,
    Side,
    Signal,
)
from trading_lab.data.base import timeframe_delta
from trading_lab.ensemble import VotingEngine
from trading_lab.execution import CostModel, PaperExecutor
from trading_lab.execution.costs import MarketStats
from trading_lab.portfolio import Portfolio
from trading_lab.risk import RiskManager
from trading_lab.risk.breakers import BreakerState, CircuitBreakers

PORTFOLIO = "PORTFOLIO"  # symbol used for portfolio-level decisions
_DUST = 1e-12
RECENT_STOP_BARS = 24  # window for "recent stop-outs" in the portfolio view

# A ready signal, or a function of the portfolio view that returns one.
SignalSource = Union[Signal, Callable[[Mapping[str, Any]], Signal]]


@dataclass(frozen=True, slots=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0  # base-currency volume traded during the bar


def _signal_to_json(s: Signal) -> dict[str, Any]:
    return {
        "strategy": s.strategy,
        "symbol": s.symbol,
        "direction": s.direction.value,
        "confidence": s.confidence,
        "timestamp": s.timestamp.isoformat(),
        "metadata": _plain(s.metadata),
    }


def _signal_from_json(sig: Mapping[str, Any]) -> Signal:
    return Signal(
        sig["strategy"], sig["symbol"], Direction(sig["direction"]), sig["confidence"],
        datetime.fromisoformat(sig["timestamp"]), sig.get("metadata", {}),
    )


@dataclass(frozen=True, slots=True)
class Intent:
    """An order scheduled at a bar's close, to be handled at the next bar's open."""

    action: DecisionAction  # ENTER_SIGNAL or EXIT_SIGNAL
    signal: Signal

    def to_json(self) -> dict[str, Any]:
        return {"action": self.action.value, "signal": _signal_to_json(self.signal)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Intent:
        return cls(DecisionAction(data["action"]), _signal_from_json(data["signal"]))


@dataclass(slots=True)
class RestingLimit:
    """A working limit buy order."""

    symbol: str
    limit_price: float
    quantity: float  # originally requested
    remaining: float
    stop_price: float | None
    placed_at: datetime
    bars_left: int
    signal: Signal

    def to_json(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "limit_price": self.limit_price, "quantity": self.quantity,
            "remaining": self.remaining, "stop_price": self.stop_price,
            "placed_at": self.placed_at.isoformat(), "bars_left": self.bars_left,
            "signal": _signal_to_json(self.signal),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RestingLimit:
        return cls(
            data["symbol"], data["limit_price"], data["quantity"], data["remaining"],
            data["stop_price"], datetime.fromisoformat(data["placed_at"]), data["bars_left"],
            _signal_from_json(data["signal"]),
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
    bars: list[tuple[datetime, str, Bar]] = field(default_factory=list)  # every closed bar seen


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
        resting: Mapping[str, RestingLimit] | None = None,
        stop_events: Sequence[tuple[datetime, str]] | None = None,
        trailing: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.config = config
        self.voting = voting
        ex = config.execution
        self.costs = CostModel.from_config(ex)
        self.portfolio = portfolio or Portfolio(
            config.portfolio.initial_cash, config.portfolio.quote_currency
        )
        self.executor = PaperExecutor(
            self.portfolio, self.costs, min_notional=ex.min_notional,
            id_prefix=id_prefix, start_sequence=order_sequence,
        )
        self.risk = RiskManager(
            config.risk, self.costs, min_notional=ex.min_notional,
            max_participation_pct=ex.max_participation_pct,
        )
        self.breakers = CircuitBreakers(
            config.risk, timeframe_delta(config.market.timeframe),
            self.portfolio.initial_cash, breaker_state,
        )
        self.pending: dict[str, Intent] = dict(pending or {})
        self.resting: dict[str, RestingLimit] = dict(resting or {})
        self.last_close: dict[str, float] = dict(last_close or {})
        self.stop_events: list[tuple[datetime, str]] = list(stop_events or [])
        # Per open position: {"high": highest high since entry, "stop": raised stop (if any)}.
        self.trailing: dict[str, dict[str, float]] = {k: dict(v) for k, v in (trailing or {}).items()}
        for sym, info in self.trailing.items():  # re-apply raised stops (positions are rebuilt from fills)
            if "stop" in info and self.portfolio.position(sym) is not None:
                self.portfolio.set_stop(sym, info["stop"])
        self._bar = timeframe_delta(config.market.timeframe)
        self.records = SessionRecords()
        self._order = {s: i for i, s in enumerate(config.market.symbols)}

    def drain(self) -> SessionRecords:
        """Return the records accumulated so far and start a fresh batch."""
        records, self.records = self.records, SessionRecords()
        return records

    def _sorted(self, symbols) -> list[str]:  # type: ignore[no-untyped-def]
        return sorted(symbols, key=self._order.__getitem__)

    # ------------------------------------------------------------------ steps
    def open_bar(
        self,
        ts: datetime,
        opens: Mapping[str, float],
        stats: Mapping[str, MarketStats | None] | None = None,
    ) -> None:
        """Handle orders scheduled for this bar at its open (exits first)."""
        stats = stats or {}
        marks = {**self.last_close, **opens}
        for sym in self._sorted(self.pending):
            intent = self.pending[sym]
            if intent.action is DecisionAction.EXIT_SIGNAL and sym in opens:
                del self.pending[sym]
                self._exit(sym, opens[sym], ts, DecisionAction.EXIT, "signal exit", intent.signal,
                           stats.get(sym))

        entry_syms = sorted(
            (s for s, it in self.pending.items()
             if it.action is DecisionAction.ENTER_SIGNAL and s in opens),
            key=lambda s: (-self.pending[s].signal.confidence, self._order[s]),
        )
        for sym in entry_syms:
            sig = self.pending.pop(sym).signal
            blocked = self.breakers.entry_block_reason(sym, ts)
            if blocked is not None:
                self._decide(ts, sym, DecisionAction.REJECTED, blocked, sig, reference_price=opens[sym])
                continue
            if self.config.execution.entry_order_type == "limit":
                self._place_limit(sym, opens[sym], ts, sig, marks, stats.get(sym))
            else:
                self._market_entry(sym, opens[sym], ts, sig, marks, stats.get(sym))

    def close_bar(
        self,
        ts: datetime,
        bars: Mapping[str, Bar],
        strategy_signals: Mapping[str, Sequence[SignalSource]],
        stats: Mapping[str, MarketStats | None] | None = None,
    ) -> None:
        """Limit fills, stop-losses, mark to market, circuit breakers, new signals."""
        stats = stats or {}
        for sym in self._sorted(s for s in self.resting if s in bars):
            self._match_limit(sym, bars[sym], ts)

        risk_cfg = self.config.risk
        for sym in self._sorted(bars):
            bar = bars[sym]
            position = self.portfolio.position(sym)
            if position is not None and self.risk.stop_triggered(position, bar.low):
                stop = position.stop_price
                assert stop is not None
                kind = "trailing stop" if "stop" in self.trailing.get(sym, {}) else "stop"
                self._cancel_resting(sym, ts, "cancelled: stop-loss hit")
                trades_before = len(self.portfolio.closed_trades)
                self._exit(
                    sym, min(bar.open, stop), ts, DecisionAction.STOP_LOSS,
                    f"{kind} {stop:.8g} hit (bar low {bar.low:.8g})", None, stats.get(sym),
                )
                closed = self.portfolio.closed_trades[trades_before:]
                if not closed or closed[-1].pnl < 0:  # no cooldown after a profitable trailing exit
                    self.breakers.on_stop_loss(sym, ts)
                self.stop_events.append((ts, sym))
            elif position is not None and risk_cfg.take_profit_pct > 0:
                target = position.avg_entry_price * (1.0 + risk_cfg.take_profit_pct)
                if bar.high >= target:
                    self._cancel_resting(sym, ts, "cancelled: take-profit hit")
                    self._exit(
                        sym, max(bar.open, target), ts, DecisionAction.TAKE_PROFIT,
                        f"take-profit {target:.8g} reached (bar high {bar.high:.8g})", None, stats.get(sym),
                    )
        self._update_trailing(bars)
        horizon = ts - RECENT_STOP_BARS * self._bar
        self.stop_events = [(t, s) for t, s in self.stop_events if t > horizon]

        for sym in self._sorted(bars):
            self.last_close[sym] = bars[sym].close
            self.records.bars.append((ts, sym, bars[sym]))
        snapshot = self.portfolio.snapshot(self.last_close, ts)
        self.records.snapshots.append(snapshot)
        self.records.in_market.append(bool(self.portfolio.positions))

        was_halted = self.breakers.halted
        for event in self.breakers.on_bar_close(ts, snapshot.equity):
            self.records.decisions.append(Decision(ts, PORTFOLIO, DecisionAction.CIRCUIT_BREAKER, event))
        if self.breakers.halted and not was_halted:
            for sym in self._sorted(self.resting):
                self._cancel_resting(sym, ts, "cancelled by kill switch")
            if self.config.risk.flatten_on_halt:
                self._flatten(ts)

        for sym in self._sorted(bars):
            sources = strategy_signals[sym]
            view = (
                self.portfolio_view(sym, ts) if any(not isinstance(s, Signal) for s in sources) else None
            )
            strat_sigs = [s if isinstance(s, Signal) else s(view) for s in sources]  # type: ignore[arg-type]
            ensemble = self.voting.combine(strat_sigs)
            self.records.signals.extend(strat_sigs)
            self.records.signals.append(ensemble)
            self._schedule(sym, ts, ensemble)

    def _update_trailing(self, bars: Mapping[str, Bar]) -> None:
        """Track the highest high of each open position and raise trailing stops (from the next bar)."""
        for sym in list(self.trailing):
            if self.portfolio.position(sym) is None:
                del self.trailing[sym]
        cfg = self.config.risk
        for sym in self._sorted(bars):
            position = self.portfolio.position(sym)
            if position is None:
                continue
            info = self.trailing.setdefault(sym, {"high": bars[sym].high})
            info["high"] = max(info["high"], bars[sym].high)
            if cfg.trailing_stop_pct <= 0:
                continue
            if info["high"] < position.avg_entry_price * (1.0 + cfg.trailing_activation_pct):
                continue
            candidate = info["high"] * (1.0 - cfg.trailing_stop_pct)
            if position.stop_price is None or candidate > position.stop_price:
                self.portfolio.set_stop(sym, candidate)
                info["stop"] = candidate

    def portfolio_view(self, sym: str, ts: datetime) -> dict[str, Any]:
        """Read-only, rounded summary of the simulated portfolio at the close of bar ``ts``.

        Values are rounded coarsely so that agents asked about similar
        situations get identical contexts (and cache keys).
        """
        snapshot = self.portfolio.snapshot(self.last_close, ts)
        equity = snapshot.equity
        state = self.breakers.state
        position = self.portfolio.position(sym)
        view: dict[str, Any] = {"position": "long" if position is not None else "flat"}
        if position is not None:
            price = self.last_close.get(sym, position.avg_entry_price)
            view["position_return_pct"] = round(position.unrealized_pnl(price) / position.cost_basis * 100, 1)
            view["position_bars_held"] = max(0, int((ts - position.opened_at) / self._bar) + 1)
            if position.stop_price is not None and price > 0:
                view["stop_distance_pct"] = round((price - position.stop_price) / price * 100, 1)
        blocked = self.breakers.entry_block_reason(sym, ts + self._bar)
        view.update({
            "exposure_pct": round(snapshot.positions_value / equity * 100) if equity > 0 else 0,
            "open_positions": snapshot.open_positions,
            "max_open_positions": self.config.risk.max_open_positions,
            "drawdown_pct": round(max(0.0, 1.0 - equity / state.peak_equity) * 100, 1)
            if state.peak_equity > 0 else 0.0,
            "daily_pnl_pct": round((equity / state.day_start_equity - 1.0) * 100, 1)
            if state.day_start_equity > 0 else 0.0,
            f"stop_outs_last_{RECENT_STOP_BARS}_bars": len(self.stop_events),
            f"symbol_stop_outs_last_{RECENT_STOP_BARS}_bars": sum(s == sym for _, s in self.stop_events),
            "kill_switch_active": self.breakers.halted,
            "new_entries_blocked": blocked is not None,
        })
        if blocked is not None:
            view["block_reason"] = (
                "daily loss limit" if blocked.startswith("circuit breaker: daily")
                else "kill switch" if blocked.startswith("circuit breaker")
                else "stop-loss cooldown"
            )
        return view

    def expire_pending(self, ts: datetime) -> None:
        for sym in self._sorted(self.pending):
            intent = self.pending.pop(sym)
            self._decide(ts, sym, DecisionAction.EXPIRED, "data ended before the order could fill",
                         intent.signal)
        for sym in self._sorted(self.resting):
            self._cancel_resting(sym, ts, "data ended with the limit order still working")

    def liquidate(self, ts: datetime) -> None:
        """Close every position at its last close and refresh the latest snapshot."""
        if not self.portfolio.positions:
            return
        for sym in self._sorted(self.portfolio.positions):
            self._exit(sym, self.last_close[sym], ts, DecisionAction.LIQUIDATE, "end of backtest")
        snapshot = self.portfolio.snapshot(self.last_close, ts)
        if self.records.snapshots and self.records.snapshots[-1].timestamp == snapshot.timestamp:
            self.records.snapshots[-1] = snapshot
            self.records.in_market[-1] = bool(self.portfolio.positions)
        else:
            self.records.snapshots.append(snapshot)
            self.records.in_market.append(bool(self.portfolio.positions))

    # ---------------------------------------------------------------- entries
    def _market_entry(
        self, sym: str, open_price: float, ts: datetime, sig: Signal,
        marks: Mapping[str, float], stats: MarketStats | None,
    ) -> None:
        decision = self.risk.evaluate_entry(sym, open_price, self.portfolio, marks, stats)
        if not decision.approved:
            self._decide(ts, sym, DecisionAction.REJECTED, decision.reason, sig,
                         reference_price=open_price, details=decision.sizing)
            return
        quantity = decision.quantity
        # Market impact grows with size; shrink the order if impact makes it unaffordable.
        fill = self.costs.fill_price(Side.BUY, open_price, quantity, stats)
        affordable = self.costs.fee_model.max_notional(self.portfolio.cash) * (1 - 1e-9) / fill
        quantity = min(quantity, affordable)
        stop = self.risk.stop_price_for(fill)
        order = Order(sym, Side.BUY, quantity, ts, stop_price=stop, reason="enter")
        report = self.executor.submit(order, open_price, stats)
        self.records.reports.append(report)
        details = dict(decision.sizing)
        if report.fill is not None and stats is not None:
            details["impact_bps"] = (report.fill.fill_price / open_price - 1) * 1e4
        self._decide(
            ts, sym, DecisionAction.ENTER if report.filled else DecisionAction.REJECTED,
            decision.reason if report.filled else report.reason, sig, quantity=quantity,
            reference_price=open_price, stop_price=stop, order_id=report.order_id, details=details,
        )

    def _place_limit(
        self, sym: str, open_price: float, ts: datetime, sig: Signal,
        marks: Mapping[str, float], stats: MarketStats | None,
    ) -> None:
        ex = self.config.execution
        if sym in self.resting:
            self._decide(ts, sym, DecisionAction.IGNORED, "limit order already working", sig)
            return
        limit = open_price * (1.0 - ex.limit_offset_bps / 10_000.0)
        decision = self.risk.evaluate_entry(sym, limit, self.portfolio, marks, stats)
        if not decision.approved:
            self._decide(ts, sym, DecisionAction.REJECTED, decision.reason, sig,
                         reference_price=limit, details=decision.sizing)
            return
        stop = self.risk.stop_price_for(limit)
        self.resting[sym] = RestingLimit(
            sym, limit, decision.quantity, decision.quantity, stop, ts, ex.limit_ttl_bars, sig
        )
        self._decide(
            ts, sym, DecisionAction.ORDER_PLACED,
            f"limit buy {decision.quantity:.8g} @ {limit:.8g} for {ex.limit_ttl_bars} bar(s)",
            sig, quantity=decision.quantity, reference_price=limit, stop_price=stop,
            details=decision.sizing,
        )

    def _match_limit(self, sym: str, bar: Bar, ts: datetime) -> None:
        order = self.resting[sym]
        ex = self.config.execution
        if bar.low < order.limit_price:  # traded through the limit: we were filled
            quantity = order.remaining
            if ex.max_participation_pct > 0:
                quantity = min(quantity, ex.max_participation_pct * bar.volume)
            quantity = min(quantity, self.costs.max_limit_buy_quantity(self.portfolio.cash, order.limit_price))
            if quantity * order.limit_price >= max(ex.min_notional, _DUST):
                limit_order = Order(
                    sym, Side.BUY, quantity, ts, order_type=OrderType.LIMIT, stop_price=order.stop_price,
                    reason="enter_limit", limit_price=order.limit_price,
                )
                report = self.executor.submit(limit_order, order.limit_price)
                self.records.reports.append(report)
                if report.filled:
                    order.remaining -= quantity
                    partial = order.remaining > _DUST * max(1.0, order.quantity)
                    self._decide(
                        ts, sym, DecisionAction.ENTER,
                        f"limit filled {quantity:.8g} @ {order.limit_price:.8g}"
                        + (f" (partial, {order.remaining:.8g} left)" if partial else ""),
                        order.signal, quantity=quantity, reference_price=order.limit_price,
                        stop_price=order.stop_price, order_id=report.order_id,
                        details={"partial": partial, "requested": order.quantity},
                    )
                else:
                    self._cancel_resting(sym, ts, f"limit fill rejected: {report.reason}")
                    return
            elif order.remaining * order.limit_price < ex.min_notional or self.portfolio.cash <= 0:
                self._cancel_resting(sym, ts, "remaining limit order below minimum or unaffordable")
                return
        order.bars_left -= 1
        if order.remaining <= _DUST * max(1.0, order.quantity):
            del self.resting[sym]
        elif order.bars_left <= 0:
            self._cancel_resting(sym, ts, "limit order expired")

    def _cancel_resting(self, sym: str, ts: datetime, reason: str) -> None:
        order = self.resting.pop(sym, None)
        if order is None:
            return
        filled = order.quantity - order.remaining
        self._decide(
            ts, sym, DecisionAction.EXPIRED,
            f"{reason}; filled {filled:.8g} of {order.quantity:.8g}", order.signal,
            quantity=order.remaining, reference_price=order.limit_price,
        )

    # ---------------------------------------------------------------- helpers
    def _decide(
        self, ts: datetime, sym: str, action: DecisionAction, reason: str, signal: Signal | None,
        **kwargs: Any,
    ) -> None:
        self.records.decisions.append(
            Decision(ts, sym, action, reason,
                     signal.direction if signal else None, signal.confidence if signal else None,
                     **kwargs)
        )

    def _flatten(self, ts: datetime) -> None:
        """Schedule exits for every position at the next open (kill switch)."""
        for sym in self._sorted(self.portfolio.positions):
            signal = Signal("risk_manager", sym, Direction.SELL, 1.0, ts, {"reason": "kill switch"})
            self.pending[sym] = Intent(DecisionAction.EXIT_SIGNAL, signal)
            self._decide(ts, sym, DecisionAction.EXIT_SIGNAL, "kill switch: closing position", signal)

    def _schedule(self, sym: str, ts: datetime, ensemble: Signal) -> None:
        existing = self.pending.get(sym)
        if existing is not None and existing.signal.strategy == "risk_manager":
            return  # a kill-switch exit is scheduled; strategies cannot override it
        has_position = self.portfolio.position(sym) is not None
        d = ensemble.direction
        if d is Direction.BUY:
            if has_position and not self.config.risk.allow_pyramiding:
                self._decide(ts, sym, DecisionAction.IGNORED, "BUY signal but position already open", ensemble)
            elif sym in self.resting:
                self._decide(ts, sym, DecisionAction.IGNORED, "BUY signal but limit order already working", ensemble)
            else:
                self.pending[sym] = Intent(DecisionAction.ENTER_SIGNAL, ensemble)
                self._decide(ts, sym, DecisionAction.ENTER_SIGNAL, "entry scheduled for next bar open", ensemble)
        elif d is Direction.SELL:
            self._cancel_resting(sym, ts, "cancelled by SELL signal")
            if has_position:
                self.pending[sym] = Intent(DecisionAction.EXIT_SIGNAL, ensemble)
                self._decide(ts, sym, DecisionAction.EXIT_SIGNAL, "exit scheduled for next bar open", ensemble)
            else:
                self._decide(ts, sym, DecisionAction.IGNORED, "SELL signal but no position (long-only)", ensemble)
        else:
            self._decide(ts, sym, DecisionAction.HOLD, "ensemble HOLD", ensemble)

    def _exit(
        self, sym: str, ref: float, ts: datetime, action: DecisionAction, reason: str,
        signal: Signal | None = None, stats: MarketStats | None = None,
    ) -> None:
        decision = self.risk.evaluate_exit(sym, self.portfolio, reason)
        if not decision.approved:
            self._decide(ts, sym, DecisionAction.REJECTED, decision.reason, signal)
            return
        report = self.executor.submit(decision.to_order(ts, reason=action.value), ref, stats)
        self.records.reports.append(report)
        self._decide(
            ts, sym, action if report.filled else DecisionAction.REJECTED,
            reason if report.filled else report.reason, signal,
            quantity=decision.quantity, reference_price=ref, order_id=report.order_id,
        )
