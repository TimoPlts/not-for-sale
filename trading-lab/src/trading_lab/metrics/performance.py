"""Performance metrics.

Definitions (all fractions, not percentages):
  * total_return: ``final_equity / initial_equity - 1``
  * annualized_return: compound annual growth over the period covered
  * max_drawdown: largest peak-to-trough fall of the equity curve, as a
    positive fraction of the peak
  * sharpe_ratio: mean / sample std of per-bar returns × sqrt(bars per year),
    with risk-free rate 0. Crypto trades 365 days a year. None when undefined.
  * sortino_ratio: like Sharpe but divided by downside deviation
  * win_rate: share of closed trades with pnl > 0
  * profit_factor: gross profit / gross loss (inf if there are no losing
    trades, None if there are no trades or only break-even ones)
  * exposure: share of bars that ended with at least one open position
  * calmar_ratio: annualized return / max drawdown (None when undefined)
  * max_drawdown_bars: the longest stretch of bars spent below an earlier
    equity peak ("time under water")

``monthly_returns`` turns an equity curve into a calendar table of returns.

The equity curve is expected to start with the initial equity, i.e. the
value before the first bar.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trading_lab.core.models import ClosedTrade
from trading_lab.core.symbols import timeframe_to_seconds

_SECONDS_PER_YEAR = 365 * 24 * 60 * 60


def periods_per_year(timeframe: str) -> float:
    return _SECONDS_PER_YEAR / timeframe_to_seconds(timeframe)


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    initial_equity: float
    final_equity: float
    total_return: float
    annualized_return: float | None
    max_drawdown: float
    volatility_annualized: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    num_bars: int
    num_trades: int
    win_rate: float | None
    profit_factor: float | None
    avg_trade_return: float | None
    avg_win: float | None
    avg_loss: float | None
    best_trade: float | None
    worst_trade: float | None
    total_fees: float
    exposure: float | None
    calmar_ratio: float | None = None
    max_drawdown_bars: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict: NaN → None, inf → "inf"."""
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, float) and math.isnan(value):
                value = None
            elif isinstance(value, float) and math.isinf(value):
                value = "inf" if value > 0 else "-inf"
            out[key] = value
        return out

    def format_table(self) -> str:
        def pct(v: float | None) -> str:
            return "n/a" if v is None else f"{v:+.2%}"

        def num(v: float | None, fmt: str = ".2f") -> str:
            if v is None:
                return "n/a"
            return "inf" if math.isinf(v) else format(v, fmt)

        rows = [
            ("Initial equity", f"{self.initial_equity:,.2f}"),
            ("Final equity", f"{self.final_equity:,.2f}"),
            ("Total return", pct(self.total_return)),
            ("Annualized return", pct(self.annualized_return)),
            ("Max drawdown", "n/a" if self.max_drawdown is None else f"{-self.max_drawdown:.2%}"),
            ("Longest drawdown", f"{self.max_drawdown_bars} bars"),
            ("Calmar ratio", num(self.calmar_ratio)),
            ("Sharpe ratio", num(self.sharpe_ratio)),
            ("Sortino ratio", num(self.sortino_ratio)),
            ("Volatility (ann.)", "n/a" if self.volatility_annualized is None
             else f"{self.volatility_annualized:.2%}"),
            ("Trades", str(self.num_trades)),
            ("Win rate", "n/a" if self.win_rate is None else f"{self.win_rate:.1%}"),
            ("Profit factor", num(self.profit_factor)),
            ("Avg trade return", pct(self.avg_trade_return)),
            ("Avg win / loss", f"{num(self.avg_win)} / {num(self.avg_loss)} USDT"),
            ("Best / worst trade", f"{num(self.best_trade)} / {num(self.worst_trade)} USDT"),
            ("Fees paid", f"{self.total_fees:,.2f}"),
            ("Exposure", "n/a" if self.exposure is None else f"{self.exposure:.1%}"),
            ("Bars", str(self.num_bars)),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)


def _max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    return float(-drawdowns.min()) if drawdowns.size else 0.0


def _longest_drawdown(equity: np.ndarray) -> int:
    """Longest run of consecutive points strictly below the running peak."""
    below = equity < np.maximum.accumulate(equity)
    longest = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def monthly_returns(equity: pd.Series, initial: float) -> pd.DataFrame:
    """Calendar table of returns: one row per year, columns 1-12 and "year".

    ``equity`` is indexed by bar time (UTC). A month's return compares its last
    equity with the previous month's last equity (or ``initial`` for the
    first month); months without bars are NaN. "year" compounds the months.
    """
    if equity.empty:
        return pd.DataFrame(columns=[*range(1, 13), "year"], dtype="float64")
    index = pd.DatetimeIndex(equity.index)
    month_end = equity.groupby([index.year, index.month]).last()
    previous = month_end.shift(1)
    previous.iloc[0] = initial
    table = (month_end / previous - 1.0).unstack()
    table = table.reindex(columns=range(1, 13))
    table["year"] = (1.0 + table[list(range(1, 13))].fillna(0.0)).prod(axis=1) - 1.0
    table.index.name = "year"
    return table


def compute_metrics(
    equity: pd.Series | Sequence[float],
    trades: Sequence[ClosedTrade],
    timeframe: str,
    *,
    total_fees: float = 0.0,
    in_market: Sequence[bool] | None = None,
) -> PerformanceMetrics:
    """Compute metrics. ``equity[0]`` must be the initial equity (before bar 1)."""
    values = np.asarray(equity, dtype="float64")
    if values.size == 0:
        raise ValueError("equity curve is empty")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("equity values must be positive and finite")

    initial, final = float(values[0]), float(values[-1])
    num_bars = values.size - 1
    ppy = periods_per_year(timeframe)
    returns = values[1:] / values[:-1] - 1.0

    annualized: float | None = None
    if num_bars > 0:
        # Log space: annualising a short period can overflow a float.
        exponent = math.log(final / initial) * ppy / num_bars
        annualized = math.inf if exponent > 700 else math.expm1(exponent)

    volatility = sharpe = sortino = None
    if returns.size >= 2:
        std = float(np.std(returns, ddof=1))
        mean = float(np.mean(returns))
        if std > 0:
            volatility = std * math.sqrt(ppy)
            sharpe = mean / std * math.sqrt(ppy)
        downside = np.minimum(returns, 0.0)
        downside_dev = math.sqrt(float(np.mean(downside**2)))
        if downside_dev > 0:
            sortino = mean / downside_dev * math.sqrt(ppy)

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit, gross_loss = sum(wins), -sum(losses)
    if gross_loss > 0:
        profit_factor: float | None = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = None

    exposure = None
    if in_market is not None and len(in_market) > 0:
        exposure = float(np.mean(np.asarray(in_market, dtype=bool)))
    max_dd = _max_drawdown(values)
    calmar = annualized / max_dd if annualized is not None and math.isfinite(annualized) and max_dd > 0 else None

    return PerformanceMetrics(
        initial_equity=initial,
        final_equity=final,
        total_return=final / initial - 1.0,
        annualized_return=annualized,
        max_drawdown=max_dd,
        volatility_annualized=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        num_bars=num_bars,
        num_trades=len(pnls),
        win_rate=len(wins) / len(pnls) if pnls else None,
        profit_factor=profit_factor,
        avg_trade_return=float(np.mean([t.return_pct for t in trades])) if trades else None,
        avg_win=float(np.mean(wins)) if wins else None,
        avg_loss=float(np.mean(losses)) if losses else None,
        best_trade=max(pnls) if pnls else None,
        worst_trade=min(pnls) if pnls else None,
        total_fees=float(total_fees),
        exposure=exposure,
        calmar_ratio=calmar,
        max_drawdown_bars=_longest_drawdown(values),
    )
