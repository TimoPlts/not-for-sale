"""Trading cost models: fees and slippage.

The models are small Protocols so that richer models (tiered fees,
volume-aware or volatility-aware slippage) can be swapped in later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from trading_lab.core.models import Side

if TYPE_CHECKING:
    from trading_lab.config import ExecutionConfig

# Sizing leaves this relative margin so float rounding can never overdraw cash.
_CASH_SAFETY_MARGIN = 1e-9


class FeeModel(Protocol):
    def fee(self, notional: float) -> float:
        """Fee in quote currency for a fill of the given notional."""
        ...

    def max_notional(self, budget: float) -> float:
        """Largest notional such that ``notional + fee(notional) <= budget``."""
        ...


class SlippageModel(Protocol):
    def fill_price(self, side: Side, reference_price: float) -> float:
        """Price actually obtained when trading ``side`` against ``reference_price``."""
        ...


@dataclass(frozen=True, slots=True)
class PercentageFeeModel:
    """Fee proportional to notional (e.g. 0.001 = 0.1% taker fee)."""

    rate: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate) or not 0.0 <= self.rate < 1.0:
            raise ValueError(f"fee rate must be in [0, 1), got {self.rate!r}")

    def fee(self, notional: float) -> float:
        return notional * self.rate

    def max_notional(self, budget: float) -> float:
        return max(budget, 0.0) / (1.0 + self.rate)


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage:
    """Constant adverse slippage in basis points: buys pay more, sells receive less."""

    bps: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.bps) or not 0.0 <= self.bps < 10_000.0:
            raise ValueError(f"slippage bps must be in [0, 10000), got {self.bps!r}")

    def fill_price(self, side: Side, reference_price: float) -> float:
        factor = self.bps / 10_000.0
        if Side(side) is Side.BUY:  # Side() also validates plain strings like "buy"
            return reference_price * (1.0 + factor)
        return reference_price * (1.0 - factor)


@dataclass(frozen=True, slots=True)
class CostModel:
    """Fee and slippage combined: the single source of truth for trading costs.

    The executor and the risk manager share one instance, so position sizing
    accounts for exactly the costs that execution will charge.
    """

    fee_model: FeeModel
    slippage_model: SlippageModel

    @classmethod
    def from_config(cls, config: ExecutionConfig) -> CostModel:
        return cls(PercentageFeeModel(config.fee_rate), FixedBpsSlippage(config.slippage_bps))

    def fill_price(self, side: Side, reference_price: float) -> float:
        return self.slippage_model.fill_price(side, reference_price)

    def fee(self, notional: float) -> float:
        return self.fee_model.fee(notional)

    def max_buy_quantity(self, cash: float, reference_price: float) -> float:
        """Largest quantity whose fill notional plus fee fits within ``cash``."""
        if cash <= 0:
            return 0.0
        price = self.fill_price(Side.BUY, reference_price)
        return self.fee_model.max_notional(cash) * (1.0 - _CASH_SAFETY_MARGIN) / price

    def entry_cost_per_unit(self, reference_price: float) -> float:
        """Cash spent per unit bought (slippage and fee included).

        Assumes fees are proportional to notional, which holds for the V1 fee model.
        """
        price = self.fill_price(Side.BUY, reference_price)
        return price + self.fee(price)

    def exit_proceeds_per_unit(self, reference_price: float) -> float:
        """Cash received per unit sold (slippage and fee deducted)."""
        price = self.fill_price(Side.SELL, reference_price)
        return price - self.fee(price)
