"""Execution engine interface."""

from typing import Protocol

from trading_lab.core.models import ExecutionReport, Order


class ExecutionEngine(Protocol):
    """Anything that turns an order into a fill or a rejection.

    ``reference_price`` is the market price the order executes against (for
    example the next bar's open in a backtest). The engine applies its own
    cost model on top of it.
    """

    def submit(self, order: Order, reference_price: float) -> ExecutionReport: ...
