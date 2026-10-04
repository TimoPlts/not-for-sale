"""Simulated portfolio accounting (long-only, single quote currency).

Accounting conventions:
  * Buy fees and slippage are capitalised into a position's ``cost_basis``.
  * Selling releases cost basis pro rata. Realized PnL is
    ``(sell notional - sell fee) - released basis``, so it includes the fees
    on both sides.
  * Invariant: ``equity == initial_cash + realized_pnl + unrealized_pnl``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from trading_lab.core.errors import InsufficientFundsError, MissingPriceError, PositionError
from trading_lab.core.models import ClosedTrade, Fill, PortfolioSnapshot, Position, Side
from trading_lab.core.timeutils import ensure_utc

# Absolute tolerance (quote currency) for float rounding in cash checks.
CASH_TOLERANCE = 1e-9
# Relative tolerance for treating two quantities as equal.
QUANTITY_RTOL = 1e-9


def quantities_match(a: float, b: float) -> bool:
    """True if ``a`` and ``b`` are equal up to float dust."""
    return math.isclose(a, b, rel_tol=QUANTITY_RTOL, abs_tol=1e-12)


class Portfolio:
    """Cash, open positions and trade history for one simulated account."""

    def __init__(self, initial_cash: float, quote_currency: str = "USDT") -> None:
        if (
            isinstance(initial_cash, bool)
            or not isinstance(initial_cash, (int, float))
            or not math.isfinite(initial_cash)
            or initial_cash <= 0
        ):
            raise ValueError(f"initial_cash must be a positive finite number, got {initial_cash!r}")
        self._quote_currency = quote_currency
        self._initial_cash = float(initial_cash)
        self._cash = float(initial_cash)
        self._positions: dict[str, Position] = {}
        self._realized_pnl = 0.0
        self._fees_paid = 0.0
        self._fills: list[Fill] = []
        self._closed_trades: list[ClosedTrade] = []

    # ------------------------------------------------------------------ state
    @property
    def quote_currency(self) -> str:
        return self._quote_currency

    @property
    def initial_cash(self) -> float:
        return self._initial_cash

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def fees_paid(self) -> float:
        return self._fees_paid

    @property
    def positions(self) -> Mapping[str, Position]:
        return MappingProxyType(self._positions)

    @property
    def fills(self) -> tuple[Fill, ...]:
        return tuple(self._fills)

    @property
    def closed_trades(self) -> tuple[ClosedTrade, ...]:
        return tuple(self._closed_trades)

    def position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    # --------------------------------------------------------------- updates
    def apply_fill(self, fill: Fill) -> None:
        """Apply a fill. Raises without mutating anything if the fill is invalid."""
        if fill.side is Side.BUY:
            self._apply_buy(fill)
        else:
            self._apply_sell(fill)
        self._fees_paid += fill.fee
        self._fills.append(fill)

    def _apply_buy(self, fill: Fill) -> None:
        cost = fill.notional + fill.fee
        if cost > self._cash + CASH_TOLERANCE:
            raise InsufficientFundsError(
                f"buy of {fill.symbol} costs {cost:.6f} but cash is {self._cash:.6f}"
            )
        self._cash -= cost
        existing = self._positions.get(fill.symbol)
        if existing is None:
            self._positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=fill.quantity,
                cost_basis=cost,
                opened_at=fill.timestamp,
                stop_price=fill.stop_price,
            )
        else:
            self._positions[fill.symbol] = replace(
                existing,
                quantity=existing.quantity + fill.quantity,
                cost_basis=existing.cost_basis + cost,
                stop_price=fill.stop_price if fill.stop_price is not None else existing.stop_price,
            )

    def _apply_sell(self, fill: Fill) -> None:
        position = self._positions.get(fill.symbol)
        if position is None:
            raise PositionError(f"long-only: cannot sell {fill.symbol} without an open position")
        closes = quantities_match(fill.quantity, position.quantity)
        if fill.quantity > position.quantity and not closes:
            raise PositionError(
                f"long-only: cannot sell {fill.quantity} {fill.symbol}, "
                f"only {position.quantity} held"
            )

        released = (
            position.cost_basis
            if closes
            else position.cost_basis * (fill.quantity / position.quantity)
        )
        proceeds = fill.notional - fill.fee
        pnl = proceeds - released

        self._cash += proceeds
        self._realized_pnl += pnl
        self._closed_trades.append(
            ClosedTrade(
                symbol=fill.symbol,
                quantity=fill.quantity,
                entry_price=position.avg_entry_price,
                exit_price=fill.fill_price,
                cost_basis=released,
                proceeds=proceeds,
                pnl=pnl,
                opened_at=position.opened_at,
                closed_at=fill.timestamp,
                exit_order_id=fill.order_id,
            )
        )
        if closes:
            del self._positions[fill.symbol]
        else:
            self._positions[fill.symbol] = replace(
                position,
                quantity=position.quantity - fill.quantity,
                cost_basis=position.cost_basis - released,
            )

    # -------------------------------------------------------------- valuation
    def _price_for(self, symbol: str, prices: Mapping[str, float]) -> float:
        try:
            price = prices[symbol]
        except KeyError:
            raise MissingPriceError(f"no mark price provided for open position {symbol}") from None
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"invalid mark price for {symbol}: {price!r}")
        return float(price)

    def positions_value(self, prices: Mapping[str, float]) -> float:
        """Mark-to-market value of all open positions. Every open symbol needs a price."""
        return sum(
            pos.market_value(self._price_for(sym, prices)) for sym, pos in self._positions.items()
        )

    def unrealized_pnl(self, prices: Mapping[str, float]) -> float:
        return sum(
            pos.unrealized_pnl(self._price_for(sym, prices)) for sym, pos in self._positions.items()
        )

    def equity(self, prices: Mapping[str, float]) -> float:
        return self._cash + self.positions_value(prices)

    def snapshot(self, prices: Mapping[str, float], timestamp: datetime) -> PortfolioSnapshot:
        positions_value = self.positions_value(prices)
        return PortfolioSnapshot(
            timestamp=ensure_utc(timestamp),
            cash=self._cash,
            positions_value=positions_value,
            equity=self._cash + positions_value,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=self.unrealized_pnl(prices),
            fees_paid=self._fees_paid,
            open_positions=len(self._positions),
        )
