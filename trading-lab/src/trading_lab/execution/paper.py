"""Paper (simulated) market-order execution.

SAFETY: this module never contacts an exchange. Orders are filled against a
caller-supplied reference price using the configured cost model, and the
result is applied to an in-memory ``Portfolio``.
"""

from __future__ import annotations

import math

from trading_lab.core.models import (
    ExecutionReport,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
)
from trading_lab.execution.costs import CostModel
from trading_lab.portfolio.portfolio import CASH_TOLERANCE, Portfolio, quantities_match


class PaperExecutor:
    """Fills market orders immediately at ``reference_price`` adjusted for slippage, plus fees.

    Rules enforced (rejections leave the portfolio untouched):
      * long-only: no sells without a position, and no selling more than is held
      * buys must be fully affordable (notional plus fee <= cash)
      * notional must be at least ``min_notional``, except for a sell that
        closes the whole position (so dust can always be exited)

    Order IDs are sequential (``paper-000001``, ...) so runs are reproducible.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        cost_model: CostModel,
        *,
        min_notional: float = 10.0,
        id_prefix: str = "paper",
    ) -> None:
        if not math.isfinite(min_notional) or min_notional < 0:
            raise ValueError(f"min_notional must be >= 0, got {min_notional!r}")
        self._portfolio = portfolio
        self._costs = cost_model
        self._min_notional = float(min_notional)
        self._id_prefix = id_prefix
        self._sequence = 0

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def cost_model(self) -> CostModel:
        return self._costs

    def submit(self, order: Order, reference_price: float) -> ExecutionReport:
        self._sequence += 1
        order_id = f"{self._id_prefix}-{self._sequence:06d}"

        def reject(reason: str) -> ExecutionReport:
            return ExecutionReport(order_id, order, OrderStatus.REJECTED, reason=reason)

        if order.order_type is not OrderType.MARKET:
            return reject(f"unsupported order type {order.order_type}")
        if (
            isinstance(reference_price, bool)
            or not isinstance(reference_price, (int, float))
            or not math.isfinite(reference_price)
            or reference_price <= 0
        ):
            return reject(f"invalid reference price {reference_price!r}")

        quantity = order.quantity
        closes_position = False
        if order.side is Side.SELL:
            position = self._portfolio.position(order.symbol)
            if position is None:
                return reject("long-only: no open position to sell")
            if quantity > position.quantity:
                if not quantities_match(quantity, position.quantity):
                    return reject(
                        f"long-only: sell quantity {quantity} exceeds position {position.quantity}"
                    )
                quantity = position.quantity  # absorb float dust
            closes_position = quantities_match(quantity, position.quantity)

        fill_price = self._costs.fill_price(order.side, reference_price)
        notional = quantity * fill_price
        if notional < self._min_notional and not closes_position:
            return reject(f"notional {notional:.4f} below minimum {self._min_notional:.4f}")

        fee = self._costs.fee(notional)
        if order.side is Side.BUY and notional + fee > self._portfolio.cash + CASH_TOLERANCE:
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
