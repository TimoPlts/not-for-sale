"""Simulated order execution. Nothing in this package talks to an exchange."""

from trading_lab.execution.base import ExecutionEngine
from trading_lab.execution.costs import CostModel, FixedBpsSlippage, PercentageFeeModel
from trading_lab.execution.paper import PaperExecutor

__all__ = [
    "CostModel",
    "ExecutionEngine",
    "FixedBpsSlippage",
    "PaperExecutor",
    "PercentageFeeModel",
]
