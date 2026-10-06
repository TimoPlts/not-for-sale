"""Simulated portfolio accounting (single quote currency; long-only unless shorts are allowed).

Accounting conventions:
  * Buy fees and slippage are capitalised into a position's ``cost_basis``.
  * Selling releases cost basis pro rata. Realized PnL is
    ``(sell notional - sell fee) - released basis``, so it includes the fees
    on both sides.
  * With ``allow_short=True``, a SELL without a long position opens (or adds
    to) a short, and a BUY against a short covers it. A short is fully
    collateralised: opening it moves the entry notional plus the fee from
    cash into the position, and covering returns
    ``2 x entry notional - cover notional - cover fee`` (the collateral plus
    the gain, or minus the loss). No fill ever turns a long into a short or
    back: a position has to be closed first.
  * Invariant: ``equity == initial_cash + realized_pnl + unrealized_pnl``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from trading_lab.core.errors import InsufficientFundsError, MissingPriceError, PositionError
from trading_lab.core.models import SHORT, ClosedTrade, Fill, PortfolioSnapshot, Position, Side
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

    def __init__(self, initial_cash: float, quote_currency: str = "USDT", *, allow_short: bool = False) -> None:
        if (
            isinstance(initial_cash, bool)
            or not isinstance(initial_cash, (int, float))
            or not math.isfinite(initial_cash)
            or initial_cash <= 0
        ):
            raise ValueError(f"initial_cash must be a positive finite number, got {initial_cash!r}")
        self._quote_currency = quote_currency
        self._allow_short = bool(allow_short)
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
    def allow_short(self) -> bool:
        return self._allow_short

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
        position = self._positions.get(fill.symbol)
        if fill.side is Side.BUY:
            if position is not None and position.is_short:
                self._cover(fill, position)
            else:
                self._apply_buy(fill)
        elif self._allow_short and (position is None or position.is_short):
            self._open_short(fill, position)
        else:
            self._apply_sell(fill)
        self._fees_paid += fill.fee
        self._fills.append(fill)

    def set_stop(self, symbol: str, stop_price: float) -> None:
        """Move an open position's stop (e.g. a trailing stop). No cash or fill is involved."""
        position = self._positions.get(symbol)
        if position is None:
            raise PositionError(f"no open position in {symbol}")
        if not (isinstance(stop_price, (int, float)) and stop_price > 0):
            raise ValueError(f"stop price must be positive, got {stop_price!r}")
        self._positions[symbol] = replace(position, stop_price=float(stop_price))

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

    def _open_short(self, fill: Fill, existing: Position | None) -> None:
        cost = fill.notional + fill.fee  # collateral plus fee
        if cost > self._cash + CASH_TOLERANCE:
            raise InsufficientFundsError(
                f"short of {fill.symbol} needs {cost:.6f} collateral and fee but cash is {self._cash:.6f}"
            )
        self._cash -= cost
        if existing is None:
            self._positions[fill.symbol] = Position(
                symbol=fill.symbol, quantity=fill.quantity, cost_basis=cost, opened_at=fill.timestamp,
                stop_price=fill.stop_price, side=SHORT, entry_notional=fill.notional,
            )
        else:
            self._positions[fill.symbol] = replace(
                existing,
                quantity=existing.quantity + fill.quantity,
                cost_basis=existing.cost_basis + cost,
                entry_notional=existing.entry_notional + fill.notional,
                stop_price=fill.stop_price if fill.stop_price is not None else existing.stop_price,
            )

    def _cover(self, fill: Fill, position: Position) -> None:
        closes = quantities_match(fill.quantity, position.quantity)
        if fill.quantity > position.quantity and not closes:
            raise PositionError(
                f"cannot buy {fill.quantity} {fill.symbol} against a short of {position.quantity}: "
                "a fill never turns a short into a long"
            )
        share = 1.0 if closes else fill.quantity / position.quantity
        released = position.cost_basis * share
        released_notional = position.entry_notional * share
        proceeds = 2.0 * released_notional - fill.notional - fill.fee
        pnl = proceeds - released
        self._cash += proceeds  # may be negative after a loss larger than the collateral
        self._realized_pnl += pnl
        self._closed_trades.append(
            ClosedTrade(
                symbol=fill.symbol, quantity=fill.quantity, entry_price=position.avg_entry_price,
                exit_price=fill.fill_price, cost_basis=released, proceeds=proceeds, pnl=pnl,
                opened_at=position.opened_at, closed_at=fill.timestamp, exit_order_id=fill.order_id,
                side=SHORT,
            )
        )
        if closes:
            del self._positions[fill.symbol]
        else:
            self._positions[fill.symbol] = replace(
                position,
                quantity=position.quantity - fill.quantity,
                cost_basis=position.cost_basis - released,
                entry_notional=position.entry_notional - released_notional,
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

    def gross_exposure(self, prices: Mapping[str, float]) -> float:
        """Sum of |quantity x price| over open positions, long and short."""
        return sum(
            pos.exposure(self._price_for(sym, prices)) for sym, pos in self._positions.items()
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
