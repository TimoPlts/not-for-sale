"""How much of a result could be luck? Bootstrap ranges for a finished run.

Two resamplings, both seeded so the same run always gives the same numbers:

* **Trade bootstrap:** draw the run's closed trades (their PnL in quote
  currency) with replacement, as many as the run had, many times. The total
  of each draw, divided by the initial cash, gives a distribution of total
  return. ``prob_loss`` is the share of draws that lost money. It treats
  trades as independent, which they are not quite, so read it as a rough
  guide.
* **Block bootstrap of per-bar returns:** cut the equity curve's per-bar
  returns into blocks of about sqrt(n) bars and draw blocks with
  replacement, which keeps streaks (volatility clusters) intact. It gives
  ranges for total return and the annualised Sharpe ratio.

Wide ranges, a range that includes losses, or few trades all mean the same
thing: the run alone does not show that the strategy works. See
docs/EXPERIMENT_PROTOCOL.md for how to test that properly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from trading_lab.metrics.performance import periods_per_year

MIN_TRADES = 30


@dataclass(frozen=True, slots=True)
class Range:
    low: float  # 5th percentile
    median: float
    high: float  # 95th percentile

    @classmethod
    def of(cls, values: np.ndarray) -> Range:
        lo, med, hi = np.percentile(values, [5, 50, 95])
        return cls(float(lo), float(med), float(hi))


@dataclass(frozen=True, slots=True)
class Robustness:
    samples: int
    seed: int
    trades: int
    total_return: float  # actual
    trade_total_return: Range | None  # trade bootstrap
    prob_loss: float | None
    bars: int
    block_bars: int
    bar_total_return: Range | None  # block bootstrap
    sharpe: float | None  # actual
    sharpe_range: Range | None
    warnings: tuple[str, ...]


def _sharpe(returns: np.ndarray, per_year: float) -> float:
    sd = returns.std(ddof=1) if len(returns) > 1 else 0.0
    return float(returns.mean() / sd * math.sqrt(per_year)) if sd > 0 else float("nan")


def bootstrap(
    trade_pnls: Sequence[float],
    equity: Sequence[float],
    *,
    initial_cash: float,
    timeframe: str,
    samples: int = 5000,
    seed: int = 7,
) -> Robustness:
    """``equity`` starts with the initial equity, like ``compute_metrics`` expects."""
    if samples < 100:
        raise ValueError("use at least 100 samples")
    rng = np.random.default_rng(seed)
    pnls = np.asarray(trade_pnls, dtype="float64")
    curve = np.asarray(equity, dtype="float64")
    warnings = []
    total = float(curve[-1] / curve[0] - 1.0) if len(curve) > 1 else 0.0

    trade_range = prob_loss = None
    if len(pnls):
        draws = rng.choice(pnls, size=(samples, len(pnls)), replace=True).sum(axis=1) / initial_cash
        trade_range = Range.of(draws)
        prob_loss = float((draws < 0).mean())
    if len(pnls) < MIN_TRADES:
        warnings.append(f"only {len(pnls)} closed trade(s): too few to judge (want at least {MIN_TRADES})")

    returns = curve[1:] / curve[:-1] - 1.0 if len(curve) > 1 else np.array([])
    n = len(returns)
    block = max(1, int(round(math.sqrt(n)))) if n else 0
    bar_range = sharpe_range = None
    per_year = periods_per_year(timeframe)
    sharpe = _sharpe(returns, per_year) if n > 1 else float("nan")
    if n >= 10:
        starts = rng.integers(0, n - block + 1, size=(samples, math.ceil(n / block)))
        idx = (starts[:, :, None] + np.arange(block)).reshape(samples, -1)[:, :n]
        resampled = returns[idx]
        bar_range = Range.of(np.prod(1.0 + resampled, axis=1) - 1.0)
        sd = resampled.std(axis=1, ddof=1)
        sharpes = np.where(sd > 0, resampled.mean(axis=1) / np.where(sd > 0, sd, 1) * math.sqrt(per_year), np.nan)
        finite = sharpes[np.isfinite(sharpes)]
        sharpe_range = Range.of(finite) if len(finite) else None
    else:
        warnings.append("fewer than 10 bars: no per-bar bootstrap")
    if trade_range is not None and trade_range.low < 0 < trade_range.high:
        warnings.append("the trade-bootstrap range includes both gains and losses")
    return Robustness(samples, seed, len(pnls), total, trade_range, prob_loss, n, block, bar_range,
                      None if math.isnan(sharpe) else sharpe, sharpe_range, tuple(warnings))


def robustness_for_run(store: object, run_id: str, *, samples: int = 5000, seed: int = 7) -> Robustness:
    from trading_lab.config import AppConfig

    run = store.get_run(run_id)  # type: ignore[attr-defined]
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    cfg = AppConfig.from_dict(run["config"])
    curve = store.load_equity_curve(run_id)  # type: ignore[attr-defined]
    equity = [cfg.portfolio.initial_cash, *curve["equity"].tolist()]
    pnls = [t.pnl for t in store.load_closed_trades(run_id)]  # type: ignore[attr-defined]
    return bootstrap(pnls, equity, initial_cash=cfg.portfolio.initial_cash, timeframe=run["timeframe"],
                     samples=samples, seed=seed)


def _r(r: Range | None, pct: bool = True) -> str:
    if r is None:
        return "n/a"
    f = (lambda v: f"{v:+.2%}") if pct else (lambda v: f"{v:+.2f}")
    return f"{f(r.low)} .. {f(r.median)} .. {f(r.high)}"


def format_robustness(r: Robustness) -> str:
    lines = [
        f"Robustness ({r.samples} bootstrap samples, seed {r.seed}; ranges are 5th .. median .. 95th percentile)",
        f"  Actual total return: {r.total_return:+.2%}   actual Sharpe: "
        f"{'n/a' if r.sharpe is None else f'{r.sharpe:+.2f}'}",
        f"  Trade bootstrap ({r.trades} trades): total return {_r(r.trade_total_return)}; "
        f"probability of a loss {'n/a' if r.prob_loss is None else f'{r.prob_loss:.0%}'}",
        f"  Block bootstrap ({r.bars} bars, blocks of {r.block_bars}): total return {_r(r.bar_total_return)}; "
        f"Sharpe {_r(r.sharpe_range, pct=False)}",
    ]
    lines += [f"  ! {w}" for w in r.warnings]
    return "\n".join(lines)
