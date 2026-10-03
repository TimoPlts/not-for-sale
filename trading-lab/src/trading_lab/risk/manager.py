"""Risk management: position sizing, exposure limits and stop-loss checks.

The risk manager never places orders. It turns an intent ("enter BTC/USDT at
~price") into an approved quantity and stop price, or a rejection with a
reason, and both are meant to be logged as part of the decision trail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from trading_lab.config import RiskConfig
from trading_lab.core.models import Order, Position, Side
from trading_lab.execution.costs import CostModel
from trading_lab.portfolio.portfolio import Portfolio


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Result of a risk evaluation.

    ``sizing`` records the candidate quantities from each limit, plus the
    binding one, so every sizing decision can be audited.
    """

    approved: bool
    symbol: str
    side: Side
    quantity: float
    reason: str
    reference_price: float | None = None
    stop_price: float | None = None
    sizing: Mapping[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sizing", MappingProxyType(dict(self.sizing)))
        if self.approved and not (math.isfinite(self.quantity) and self.quantity > 0):
            raise ValueError("an approved decision must have a positive quantity")

    def to_order(self, timestamp: datetime, reason: str = "") -> Order:
        if not self.approved:
            raise ValueError(f"cannot create an order from a rejected decision: {self.reason}")
        return Order(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            timestamp=timestamp,
            stop_price=self.stop_price,
            reason=reason or self.reason,
        )


class RiskManager:
    """Applies the configured limits to every proposed trade.

    Entry size is the minimum of:
      * **risk per trade**: the quantity at which hitting the stop loses
        ``equity * risk_per_trade_pct``, counting entry and exit fees and slippage
      * **max position size**: position value <= ``equity * max_position_pct``
      * **max total exposure**: all positions <= ``equity * max_total_exposure_pct``
      * **available cash**: notional plus fee must be affordable

    Entries are also rejected when the open-position limit is reached, when
    pyramiding is disabled and a position already exists, or when the result
    is below ``min_notional``.
    """

    def __init__(
        self, config: RiskConfig, cost_model: CostModel, *, min_notional: float = 0.0
    ) -> None:
        self._config = config
        self._costs = cost_model
        self._min_notional = float(min_notional)

    @property
    def config(self) -> RiskConfig:
        return self._config

    def stop_price_for(self, entry_fill_price: float) -> float:
        return entry_fill_price * (1.0 - self._config.stop_loss_pct)

    def evaluate_entry(
        self,
        symbol: str,
        reference_price: float,
        portfolio: Portfolio,
        prices: Mapping[str, float],
    ) -> RiskDecision:
        """Size a new long entry. ``prices`` must contain marks for all open positions."""
        cfg = self._config

        def reject(reason: str, **sizing: float | str) -> RiskDecision:
            return RiskDecision(
                False, symbol, Side.BUY, 0.0, reason, reference_price=reference_price, sizing=sizing
            )

        if not (math.isfinite(reference_price) and reference_price > 0):
            return reject(f"invalid reference price {reference_price!r}")

        existing = portfolio.position(symbol)
        if existing is not None and not cfg.allow_pyramiding:
            return reject("position already open and pyramiding is disabled")
        if existing is None and len(portfolio.positions) >= cfg.max_open_positions:
            return reject(f"max open positions reached ({cfg.max_open_positions})")

        marks = {**prices, symbol: reference_price}
        equity = portfolio.equity(marks)
        if equity <= 0:
            return reject("non-positive equity")

        fill_price = self._costs.fill_price(Side.BUY, reference_price)
        stop_price = self.stop_price_for(fill_price)
        loss_per_unit = self._costs.entry_cost_per_unit(
            reference_price
        ) - self._costs.exit_proceeds_per_unit(stop_price)

        existing_value = existing.market_value(reference_price) if existing else 0.0
        total_value = portfolio.positions_value(marks)

        candidates: dict[str, float] = {
            "risk_per_trade": equity * cfg.risk_per_trade_pct / loss_per_unit,
            "max_position_size": (equity * cfg.max_position_pct - existing_value) / fill_price,
            "max_total_exposure": (equity * cfg.max_total_exposure_pct - total_value) / fill_price,
            "available_cash": self._costs.max_buy_quantity(portfolio.cash, reference_price),
        }
        binding = min(candidates, key=candidates.__getitem__)
        quantity = max(candidates[binding], 0.0)
        sizing: dict[str, float | str] = {
            **candidates,
            "binding_limit": binding,
            "equity": equity,
            "loss_per_unit_at_stop": loss_per_unit,
        }

        notional = quantity * fill_price
        if quantity <= 0 or notional < max(self._min_notional, 1e-12):
            return reject(
                f"size {notional:.4f} below minimum notional {self._min_notional:.4f} "
                f"(limited by {binding})",
                **sizing,
            )
        return RiskDecision(
            True,
            symbol,
            Side.BUY,
            quantity,
            f"entry approved (limited by {binding})",
            reference_price=reference_price,
            stop_price=stop_price,
            sizing=sizing,
        )

    def evaluate_exit(self, symbol: str, portfolio: Portfolio, reason: str = "exit") -> RiskDecision:
        """Close the full position. Exits are always allowed when a position exists."""
        position = portfolio.position(symbol)
        if position is None:
            return RiskDecision(False, symbol, Side.SELL, 0.0, "long-only: no open position to exit")
        return RiskDecision(True, symbol, Side.SELL, position.quantity, reason)

    @staticmethod
    def stop_triggered(position: Position, low_price: float) -> bool:
        """True if the bar's low reached the position's stop price."""
        return position.stop_price is not None and low_price <= position.stop_price
