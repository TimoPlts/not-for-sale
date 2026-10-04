"""Immutable domain models.

These are the plain-data objects that every module exchanges. They validate
themselves on construction so invalid state can never travel through the system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from trading_lab.core.timeutils import ensure_utc


class Side(StrEnum):
    """Side of an order or fill."""

    BUY = "buy"
    SELL = "sell"


class Direction(StrEnum):
    """What a strategy (or ensemble) recommends for a symbol.

    In the long-only V1: BUY = open/increase a long, SELL = reduce/close a long,
    HOLD = do nothing.
    """

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderType(StrEnum):
    MARKET = "market"


class OrderStatus(StrEnum):
    FILLED = "filled"
    REJECTED = "rejected"


class DecisionAction(StrEnum):
    """What the trading loop decided for a symbol at a point in time."""

    HOLD = "hold"  # ensemble said HOLD
    ENTER_SIGNAL = "enter_signal"  # BUY accepted; entry order scheduled for the next bar
    EXIT_SIGNAL = "exit_signal"  # SELL accepted; exit order scheduled for the next bar
    IGNORED = "ignored"  # signal not actionable (e.g. SELL without a position)
    ENTER = "enter"  # entry filled
    EXIT = "exit"  # signal exit filled
    STOP_LOSS = "stop_loss"  # stop-loss exit filled
    LIQUIDATE = "liquidate"  # position closed at the end of a backtest
    REJECTED = "rejected"  # risk manager or executor refused the order
    EXPIRED = "expired"  # scheduled order never executed (data ended)


def _check_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


def _check_non_negative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
    return float(value)


def _check_symbol(symbol: str) -> None:
    if not isinstance(symbol, str) or "/" not in symbol:
        raise ValueError(f"symbol must look like 'BASE/QUOTE', got {symbol!r}")


@dataclass(frozen=True, slots=True)
class Signal:
    """Standardized output of every strategy, AI agent and the voting engine.

    ``confidence`` is in [0, 1]. ``metadata`` carries free-form, JSON-serialisable
    context (indicator values, reasoning, ...) and is made read-only.
    """

    strategy: str
    symbol: str
    direction: Direction
    confidence: float
    timestamp: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy:
            raise ValueError("signal strategy name must be non-empty")
        _check_symbol(self.symbol)
        object.__setattr__(self, "direction", Direction(self.direction))
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a number")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence!r}")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def hold(
        cls,
        strategy: str,
        symbol: str,
        timestamp: datetime,
        metadata: Mapping[str, Any] | None = None,
    ) -> Signal:
        return cls(strategy, symbol, Direction.HOLD, 0.0, timestamp, metadata or {})


@dataclass(frozen=True, slots=True)
class Order:
    """A request to trade. In V1 only simulated market orders exist."""

    symbol: str
    side: Side
    quantity: float
    timestamp: datetime
    order_type: OrderType = OrderType.MARKET
    stop_price: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        _check_symbol(self.symbol)
        object.__setattr__(self, "side", Side(self.side))
        object.__setattr__(self, "order_type", OrderType(self.order_type))
        object.__setattr__(self, "quantity", _check_positive("quantity", self.quantity))
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", _check_positive("stop_price", self.stop_price))


@dataclass(frozen=True, slots=True)
class Fill:
    """A simulated execution. ``fee`` is in quote currency."""

    order_id: str
    symbol: str
    side: Side
    quantity: float
    reference_price: float
    fill_price: float
    fee: float
    timestamp: datetime
    stop_price: float | None = None

    def __post_init__(self) -> None:
        _check_symbol(self.symbol)
        object.__setattr__(self, "side", Side(self.side))
        object.__setattr__(self, "quantity", _check_positive("quantity", self.quantity))
        object.__setattr__(
            self, "reference_price", _check_positive("reference_price", self.reference_price)
        )
        object.__setattr__(self, "fill_price", _check_positive("fill_price", self.fill_price))
        object.__setattr__(self, "fee", _check_non_negative("fee", self.fee))
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if self.stop_price is not None:
            object.__setattr__(self, "stop_price", _check_positive("stop_price", self.stop_price))

    @property
    def notional(self) -> float:
        """Traded value at the fill price, excluding fees."""
        return self.quantity * self.fill_price

    @property
    def slippage_cost(self) -> float:
        """Quote currency lost to slippage versus the reference price (always >= 0)."""
        return abs(self.fill_price - self.reference_price) * self.quantity

    @property
    def cash_delta(self) -> float:
        """Change in cash caused by this fill."""
        if self.side is Side.BUY:
            return -(self.notional + self.fee)
        return self.notional - self.fee


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Outcome of submitting an order: either a fill or a rejection reason."""

    order_id: str
    order: Order
    status: OrderStatus
    fill: Fill | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if (self.status is OrderStatus.FILLED) != (self.fill is not None):
            raise ValueError("a FILLED report must carry a fill and a REJECTED one must not")

    @property
    def filled(self) -> bool:
        return self.status is OrderStatus.FILLED


@dataclass(frozen=True, slots=True)
class Position:
    """An open long position.

    ``cost_basis`` is the total quote currency paid for the current quantity,
    *including* buy fees and slippage.
    """

    symbol: str
    quantity: float
    cost_basis: float
    opened_at: datetime
    stop_price: float | None = None

    @property
    def avg_entry_price(self) -> float:
        """Average cost per unit, fees included."""
        return self.cost_basis / self.quantity

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        """Mark-to-market PnL (exit fees/slippage are not deducted)."""
        return self.market_value(price) - self.cost_basis


@dataclass(frozen=True, slots=True)
class ClosedTrade:
    """A realized (partial or full) exit of a long position."""

    symbol: str
    quantity: float
    entry_price: float  # average cost per unit incl. buy fees
    exit_price: float  # fill price of the sell
    cost_basis: float  # basis released by this exit
    proceeds: float  # sell notional minus sell fee
    pnl: float
    opened_at: datetime
    closed_at: datetime
    exit_order_id: str = ""

    @property
    def return_pct(self) -> float:
        return self.pnl / self.cost_basis if self.cost_basis else 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Point-in-time valuation of a portfolio."""

    timestamp: datetime
    cash: float
    positions_value: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    open_positions: int


@dataclass(frozen=True, slots=True)
class Decision:
    """One entry in the decision audit trail."""

    timestamp: datetime
    symbol: str
    action: DecisionAction
    reason: str = ""
    signal_direction: Direction | None = None
    signal_confidence: float | None = None
    quantity: float | None = None
    reference_price: float | None = None
    stop_price: float | None = None
    order_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        object.__setattr__(self, "action", DecisionAction(self.action))
        if self.signal_direction is not None:
            object.__setattr__(self, "signal_direction", Direction(self.signal_direction))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
