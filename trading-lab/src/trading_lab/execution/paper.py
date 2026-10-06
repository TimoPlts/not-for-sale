"""Paper (simulated) market-order execution.

SAFETY: this module never contacts an exchange. Orders are filled against a
caller-supplied reference price using the configured cost model, and the
result is applied to an in-memory ``Portfolio``.
"""

from __future__ import annotations

import math
from datetime import datetime

from trading_lab.core.models import (
    ExecutionReport,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from trading_lab.execution.costs import CostModel, MarketStats
from trading_lab.portfolio.portfolio import CASH_TOLERANCE, Portfolio, quantities_match


class PaperExecutor:
    """Fills orders immediately.

    * Market orders fill at ``reference_price`` adjusted by the slippage model
      (which may use the order size and ``MarketStats``), plus the taker fee.
    * Limit orders fill at their ``limit_price`` with the maker fee and no
      slippage. Deciding *whether* and *how much* a limit order fills is the
      caller's job (see ``engine.TradingSession``).

    Rules enforced (rejections leave the portfolio untouched):
      * long-only (unless the portfolio allows shorts): no sells without a
        position, and no selling more than is held
      * with shorts allowed, a sell without a long position opens or adds to a
        short, and a buy against a short covers it (never more than is short)
      * buys and short entries must be fully affordable (notional plus fee <= cash;
        a short's notional is its collateral)
      * notional must be at least ``min_notional``, except for an exit that
        closes the whole position (so dust can always be exited)
      * covering a short also pays the borrow fee: ``borrow_bps_per_day`` of
        the entry notional per day held, included in the cover's fee

    Order IDs are sequential (``paper-000001``, ...) so runs are reproducible.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        cost_model: CostModel,
        *,
        min_notional: float = 10.0,
        id_prefix: str = "paper",
        start_sequence: int = 0,
        borrow_bps_per_day: float = 0.0,
    ) -> None:
        if not math.isfinite(min_notional) or min_notional < 0:
            raise ValueError(f"min_notional must be >= 0, got {min_notional!r}")
        if not math.isfinite(borrow_bps_per_day) or borrow_bps_per_day < 0:
            raise ValueError(f"borrow_bps_per_day must be >= 0, got {borrow_bps_per_day!r}")
        self.borrow_bps_per_day = float(borrow_bps_per_day)
        self._portfolio = portfolio
        self._costs = cost_model
        self._min_notional = float(min_notional)
        self._id_prefix = id_prefix
        self._sequence = start_sequence  # continue numbering when resuming a run

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def cost_model(self) -> CostModel:
        return self._costs

    @property
    def sequence(self) -> int:
        """Number of orders submitted so far (including rejected ones)."""
        return self._sequence

    def borrow_fee(self, position: Position, quantity: float, at: datetime) -> float:
        """Borrow cost of covering ``quantity`` of a short at ``at`` (0 for longs)."""
        if not position.is_short or self.borrow_bps_per_day <= 0:
            return 0.0
        days = max(0.0, (at - position.opened_at).total_seconds() / 86_400.0)
        share = min(1.0, quantity / position.quantity)
        return position.entry_notional * share * self.borrow_bps_per_day / 10_000.0 * days

    def submit(
        self, order: Order, reference_price: float, stats: MarketStats | None = None
    ) -> ExecutionReport:
        self._sequence += 1
        order_id = f"{self._id_prefix}-{self._sequence:06d}"

        def reject(reason: str) -> ExecutionReport:
            return ExecutionReport(order_id, order, OrderStatus.REJECTED, reason=reason)

        is_limit = order.order_type is OrderType.LIMIT
        if is_limit:
            reference_price = order.limit_price  # type: ignore[assignment]
        if (
            isinstance(reference_price, bool)
            or not isinstance(reference_price, (int, float))
            or not math.isfinite(reference_price)
            or reference_price <= 0
        ):
            return reject(f"invalid reference price {reference_price!r}")

        quantity = order.quantity
        closes_position = False
        position = self._portfolio.position(order.symbol)
        covers = order.side is Side.BUY and position is not None and position.is_short
        if order.side is Side.SELL and (position is None or position.is_short):
            if not self._portfolio.allow_short:
                return reject("long-only: no open position to sell")
        elif order.side is Side.SELL or covers:
            assert position is not None
            if quantity > position.quantity:
                if not quantities_match(quantity, position.quantity):
                    what = "cover quantity" if covers else "long-only: sell quantity"
                    return reject(f"{what} {quantity} exceeds position {position.quantity}")
                quantity = position.quantity  # absorb float dust
            closes_position = quantities_match(quantity, position.quantity)

        if is_limit:
            fill_price = float(reference_price)
        else:
            fill_price = self._costs.fill_price(order.side, reference_price, quantity, stats)
        notional = quantity * fill_price
        if notional < self._min_notional and not closes_position:
            return reject(f"notional {notional:.4f} below minimum {self._min_notional:.4f}")

        fee = self._costs.maker_fee(notional) if is_limit else self._costs.fee(notional)
        if covers:
            assert position is not None
            fee += self.borrow_fee(position, quantity, order.timestamp)
        elif (order.side is Side.BUY or position is None or position.is_short) \
                and notional + fee > self._portfolio.cash + CASH_TOLERANCE:
            return reject(
                f"insufficient cash: need {notional + fee:.4f}, have {self._portfolio.cash:.4f}"
            )

        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            reference_price=float(reference_price),
            fill_price=fill_price,
            fee=fee,
            timestamp=order.timestamp,
            stop_price=order.stop_price,
        )
        self._portfolio.apply_fill(fill)
        return ExecutionReport(order_id, order, OrderStatus.FILLED, fill=fill)
