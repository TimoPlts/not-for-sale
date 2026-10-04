"""Trading cost models: fees and slippage.

The models are small Protocols, so richer models can be swapped in.

* ``FixedBpsSlippage``: constant adverse slippage.
* ``VolumeImpactSlippage``: constant slippage plus square-root market impact,
  ``impact = coefficient × volatility × sqrt(order value / average bar value
  traded)``, using ``MarketStats`` from bars *before* the fill. Small orders in
  deep markets pay almost nothing extra; large orders in thin markets pay a
  lot.

Limit-order fills pay the maker fee and no slippage. They fill at their own
limit price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd

from trading_lab.core.models import Side

if TYPE_CHECKING:
    from trading_lab.config import ExecutionConfig

# Sizing leaves this relative margin so float rounding can never overdraw cash.
_CASH_SAFETY_MARGIN = 1e-9


@dataclass(frozen=True, slots=True)
class MarketStats:
    """Liquidity statistics known *before* a fill: from the preceding bars only."""

    volatility: float  # std of per-bar log returns
    avg_quote_volume: float  # average traded value per bar (quote currency)


def market_stats_frame(candles: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Per-bar ``MarketStats`` columns where row ``i`` uses bars ``i-lookback .. i-1``.

    The shift by one bar keeps a fill at bar i's open from seeing bar i itself.
    """
    log_returns = np.log(candles["close"]).diff()
    quote_volume = candles["close"] * candles["volume"]
    return pd.DataFrame(
        {
            "volatility": log_returns.rolling(lookback, min_periods=lookback).std(ddof=0).shift(1),
            "avg_quote_volume": quote_volume.rolling(lookback, min_periods=lookback).mean().shift(1),
        },
        index=candles.index,
    )


def stats_series(frame: pd.DataFrame) -> list[MarketStats | None]:
    """``market_stats_frame`` rows as objects (None where history is too short)."""
    return [
        MarketStats(float(v), float(q)) if math.isfinite(v) and math.isfinite(q) and q > 0 else None
        for v, q in zip(frame["volatility"].to_numpy(), frame["avg_quote_volume"].to_numpy())
    ]


def next_bar_stats(candles: pd.DataFrame, lookback: int) -> MarketStats | None:
    """Stats for the bar right after the last candle (for fills at the next open)."""
    if len(candles) < lookback + 1:
        return None
    tail = candles.iloc[-(lookback + 1):]
    log_returns = np.diff(np.log(tail["close"].to_numpy()))
    quote_volume = (tail["close"] * tail["volume"]).to_numpy()[1:]
    vol, qv = float(np.std(log_returns)), float(np.mean(quote_volume))
    return MarketStats(vol, qv) if math.isfinite(vol) and qv > 0 else None


def stats_at(frame: pd.DataFrame, position: int) -> MarketStats | None:
    vol, qv = frame["volatility"].iloc[position], frame["avg_quote_volume"].iloc[position]
    if not (math.isfinite(vol) and math.isfinite(qv)) or qv <= 0:
        return None
    return MarketStats(float(vol), float(qv))


class FeeModel(Protocol):
    def fee(self, notional: float) -> float:
        """Fee in quote currency for a fill of the given notional."""
        ...

    def max_notional(self, budget: float) -> float:
        """Largest notional such that ``notional + fee(notional) <= budget``."""
        ...


class SlippageModel(Protocol):
    def fill_price(
        self,
        side: Side,
        reference_price: float,
        quantity: float | None = None,
        stats: MarketStats | None = None,
    ) -> float:
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


def _adverse(side: Side, reference_price: float, fraction: float) -> float:
    if Side(side) is Side.BUY:  # Side() also validates plain strings like "buy"
        return reference_price * (1.0 + fraction)
    return reference_price * (1.0 - fraction)


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage:
    """Constant adverse slippage in basis points: buys pay more, sells receive less."""

    bps: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.bps) or not 0.0 <= self.bps < 10_000.0:
            raise ValueError(f"slippage bps must be in [0, 10000), got {self.bps!r}")

    def fill_price(
        self,
        side: Side,
        reference_price: float,
        quantity: float | None = None,
        stats: MarketStats | None = None,
    ) -> float:
        return _adverse(side, reference_price, self.bps / 10_000.0)


@dataclass(frozen=True, slots=True)
class VolumeImpactSlippage:
    """Fixed slippage plus square-root market impact (capped at 50%)."""

    base_bps: float
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.base_bps) or not 0.0 <= self.base_bps < 10_000.0:
            raise ValueError(f"base slippage bps must be in [0, 10000), got {self.base_bps!r}")
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise ValueError("impact coefficient must be >= 0")

    def impact(self, reference_price: float, quantity: float | None, stats: MarketStats | None) -> float:
        if quantity is None or stats is None or stats.avg_quote_volume <= 0:
            return 0.0
        participation = quantity * reference_price / stats.avg_quote_volume
        return min(self.coefficient * stats.volatility * math.sqrt(participation), 0.5)

    def fill_price(
        self,
        side: Side,
        reference_price: float,
        quantity: float | None = None,
        stats: MarketStats | None = None,
    ) -> float:
        fraction = self.base_bps / 10_000.0 + self.impact(reference_price, quantity, stats)
        return _adverse(side, reference_price, fraction)


@dataclass(frozen=True, slots=True)
class CostModel:
    """Fees and slippage combined: the single source of truth for trading costs.

    The executor and the risk manager share one instance, so position sizing
    accounts for exactly the costs that execution will charge.
    """

    fee_model: FeeModel
    slippage_model: SlippageModel
    maker_fee_model: FeeModel | None = None  # limit fills; defaults to fee_model

    @classmethod
    def from_config(cls, config: ExecutionConfig) -> CostModel:
        slippage: SlippageModel
        if config.slippage_model == "volume":
            slippage = VolumeImpactSlippage(config.slippage_bps, config.impact_coefficient)
        else:
            slippage = FixedBpsSlippage(config.slippage_bps)
        return cls(
            PercentageFeeModel(config.fee_rate), slippage, PercentageFeeModel(config.maker_fee_rate)
        )

    def fill_price(
        self,
        side: Side,
        reference_price: float,
        quantity: float | None = None,
        stats: MarketStats | None = None,
    ) -> float:
        return self.slippage_model.fill_price(side, reference_price, quantity, stats)

    def fee(self, notional: float) -> float:
        return self.fee_model.fee(notional)

    def maker_fee(self, notional: float) -> float:
        return (self.maker_fee_model or self.fee_model).fee(notional)

    def max_buy_quantity(self, cash: float, reference_price: float) -> float:
        """Largest quantity whose fill notional plus fee fits within ``cash`` (base slippage)."""
        if cash <= 0:
            return 0.0
        price = self.fill_price(Side.BUY, reference_price)
        return self.fee_model.max_notional(cash) * (1.0 - _CASH_SAFETY_MARGIN) / price

    def max_limit_buy_quantity(self, cash: float, limit_price: float) -> float:
        """Largest quantity a limit buy can fill within ``cash`` (maker fee, no slippage)."""
        if cash <= 0:
            return 0.0
        fees = self.maker_fee_model or self.fee_model
        return fees.max_notional(cash) * (1.0 - _CASH_SAFETY_MARGIN) / limit_price

    def entry_cost_per_unit(self, reference_price: float) -> float:
        """Cash spent per unit bought (base slippage and fee included).

        Assumes fees are proportional to notional, which holds for the V1 fee model.
        """
        price = self.fill_price(Side.BUY, reference_price)
        return price + self.fee(price)

    def exit_proceeds_per_unit(self, reference_price: float) -> float:
        """Cash received per unit sold (base slippage and fee deducted)."""
        price = self.fill_price(Side.SELL, reference_price)
        return price - self.fee(price)
