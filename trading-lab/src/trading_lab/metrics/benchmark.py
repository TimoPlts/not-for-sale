"""Buy-and-hold benchmark.

The initial cash is split equally across the symbols. Each slice is invested
at the symbol's first open inside the period, paying the same fee and
slippage as the strategy, and held to the end. Positions are marked at bar
closes, exactly like the strategy's equity curve.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from trading_lab.execution import CostModel


def buy_and_hold_equity(
    candles: Mapping[str, pd.DataFrame],
    timeline: Sequence[pd.Timestamp],
    initial_cash: float,
    costs: CostModel,
) -> pd.Series:
    if not timeline:
        raise ValueError("timeline is empty")
    index = pd.DatetimeIndex(timeline)
    budget = initial_cash / len(candles)
    equity = pd.Series(0.0, index=index)
    for frame in candles.values():
        period = frame.loc[frame.index >= index[0]]
        if period.empty:
            equity += budget  # never tradable in the period: stays cash
            continue
        first_open = float(period["open"].iloc[0])
        quantity = costs.max_buy_quantity(budget, first_open)
        fill = costs.fill_price("buy", first_open)
        spent = quantity * fill + costs.fee(quantity * fill)
        closes = period["close"].reindex(index).ffill()
        started = index >= period.index[0]
        value = (budget - spent) + quantity * closes
        equity += value.where(started, budget).fillna(budget)
    return equity.rename("buy_and_hold")
