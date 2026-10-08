"""What should you be ready for next? Forward risk from a run's trades.

``outlook_for_run`` resamples a finished run's closed trades into many
possible futures of ``horizon`` trades (seeded, so the same run always gives
the same numbers). Each trade's return is its PnL relative to the equity
just before it closed, and each path compounds them in sequence. Across the
paths it reports:

* the **max drawdown** to be ready for: the median, and the bad case (95th
  percentile);
* the chance of a drawdown of at least 10%, 20% and 30% (``levels``);
* the final return range and the chance of ending with a loss;
* the **longest losing streak**: the median and the bad case.

Caveats, all of which make the real numbers worse rather than better:

* trades are drawn independently, so losing streaks that cluster in bad
  markets are under-represented;
* drawdowns are measured trade by trade, not at every bar, so the dips
  inside open trades are not seen;
* the run's own trades are the whole universe: a market unlike the tested
  period is not in it.

So read the bad case as a floor for what to prepare for, not a worst case.
Read-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trading_lab.config import AppConfig
from trading_lab.research.robustness import MIN_TRADES

LEVELS = (0.10, 0.20, 0.30)


@dataclass(frozen=True, slots=True)
class Outlook:
    run_id: str
    trades: int  # in the run
    horizon: int  # trades per simulated future
    samples: int
    seed: int
    actual_drawdown: float  # the run's own trade-by-trade max drawdown (positive fraction)
    drawdown_median: float
    drawdown_bad: float  # 95th percentile
    prob_drawdown: tuple[tuple[float, float], ...]  # (level, probability)
    return_low: float  # 5th percentile
    return_median: float
    return_high: float  # 95th percentile
    prob_loss: float
    streak_median: int
    streak_bad: int  # 95th percentile
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "trades": self.trades, "horizon": self.horizon, "samples": self.samples,
                "seed": self.seed, "actual_drawdown": self.actual_drawdown, "drawdown_median": self.drawdown_median,
                "drawdown_bad": self.drawdown_bad,
                "prob_drawdown": [{"level": lv, "probability": p} for lv, p in self.prob_drawdown],
                "return_low": self.return_low, "return_median": self.return_median, "return_high": self.return_high,
                "prob_loss": self.prob_loss, "streak_median": self.streak_median, "streak_bad": self.streak_bad,
                "warnings": list(self.warnings)}


def max_drawdown(returns: np.ndarray) -> np.ndarray:
    """Max drawdown (positive fraction) of each row of compounded per-trade returns, starting from 1."""
    equity = np.cumprod(1.0 + returns, axis=-1)
    peaks = np.maximum.accumulate(np.maximum(equity, 1.0), axis=-1)
    return np.max(1.0 - equity / peaks, axis=-1)


def longest_losing_streak(returns: np.ndarray) -> np.ndarray:
    """The longest run of consecutive losing trades in each row."""
    losing = np.atleast_2d(returns < 0)
    best = np.zeros(losing.shape[0], dtype=int)
    current = np.zeros(losing.shape[0], dtype=int)
    for column in losing.T:
        current = np.where(column, current + 1, 0)
        best = np.maximum(best, current)
    return best


def simulate(run_id: str, returns: Sequence[float], *, horizon: int | None = None, samples: int = 5000,
             seed: int = 7, levels: Sequence[float] = LEVELS) -> Outlook | None:
    """Outlook from per-trade returns (fractions of equity, in the order they happened); None without trades."""
    values = np.asarray(returns, dtype="float64")
    if values.size == 0:
        return None
    if samples < 100:
        raise ValueError("use at least 100 samples")
    horizon = values.size if horizon is None else int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be at least 1 trade")
    rng = np.random.default_rng([seed, values.size, horizon])
    paths = values[rng.integers(0, values.size, size=(samples, horizon))]
    drawdowns = max_drawdown(paths)
    finals = np.prod(1.0 + paths, axis=1) - 1.0
    streaks = longest_losing_streak(paths)
    warnings = []
    if values.size < MIN_TRADES:
        warnings.append(f"only {values.size} trades: the outlook rests on very few examples")
    if horizon > 3 * values.size:
        warnings.append(f"the horizon ({horizon} trades) is far beyond the {values.size} the run made")
    return Outlook(
        run_id=run_id, trades=int(values.size), horizon=horizon, samples=samples, seed=seed,
        actual_drawdown=float(max_drawdown(values)), drawdown_median=float(np.median(drawdowns)),
        drawdown_bad=float(np.percentile(drawdowns, 95)),
        prob_drawdown=tuple((float(lv), float(np.mean(drawdowns >= lv))) for lv in levels),
        return_low=float(np.percentile(finals, 5)), return_median=float(np.median(finals)),
        return_high=float(np.percentile(finals, 95)), prob_loss=float(np.mean(finals < 0)),
        streak_median=int(np.median(streaks)), streak_bad=int(np.percentile(streaks, 95)),
        warnings=tuple(warnings),
    )


def trade_returns(store: Any, run_id: str) -> list[float]:
    """Each closed trade's PnL relative to the equity just before it closed, in order."""
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    initial = AppConfig.from_dict(run["config"]).portfolio.initial_cash
    curve = store.load_equity_curve(run_id)
    trades = sorted(store.load_closed_trades(run_id), key=lambda t: t.closed_at)
    out = []
    for t in trades:
        before = curve[curve.index < pd.Timestamp(t.closed_at)]["equity"] if not curve.empty else curve
        equity = float(before.iloc[-1]) if len(before) else initial
        out.append(t.pnl / equity if equity > 0 else 0.0)
    return out


def outlook_for_run(store: Any, run_id: str, *, horizon: int | None = None, samples: int = 5000,
                    seed: int = 7) -> Outlook | None:
    return simulate(run_id, trade_returns(store, run_id), horizon=horizon, samples=samples, seed=seed)


def format_outlook(o: Outlook | None, run_id: str = "") -> str:
    if o is None:
        return f"Run {run_id}: no closed trades, so no outlook."
    lines = [f"Run {o.run_id}: the next {o.horizon} trades, resampled {o.samples:,} times from the run's {o.trades} "
             "trades",
             f"  max drawdown       median {o.drawdown_median:.1%}, bad case (1 in 20) {o.drawdown_bad:.1%}"
             f"   (the run itself: {o.actual_drawdown:.1%})",
             "  chance of a drawdown of at least "
             + ", ".join(f"{lv:.0%}: {p:.0%}" for lv, p in o.prob_drawdown),
             f"  return             {o.return_low:+.1%} .. median {o.return_median:+.1%} .. {o.return_high:+.1%}"
             f"   (chance of a loss {o.prob_loss:.0%})",
             f"  losing streak      median {o.streak_median} trades, bad case {o.streak_bad} in a row"]
    lines += [f"  warning: {w}" for w in o.warnings]
    lines.append("Trades are drawn independently and drawdowns are measured trade by trade, so real streaks and "
                 "dips can be worse: treat the bad case as a floor to prepare for, not a worst case.")
    return "\n".join(lines)
