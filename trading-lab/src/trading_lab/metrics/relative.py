"""Performance relative to a benchmark (equal-weight buy & hold by default).

From the per-bar returns of the strategy (s) and the benchmark (b), aligned on
the same bars:

* beta = cov(s, b) / var(b): how much of the benchmark's moves the strategy carries
* alpha = (mean(s) - beta x mean(b)) x bars per year: annualised return not explained by beta
* correlation of s and b
* tracking error = std(s - b) x sqrt(bars per year)
* information ratio = mean(s - b) x bars per year / tracking error
* excess return = total return of the strategy - total return of the benchmark

Values that are undefined (e.g. beta when the benchmark never moves) are None.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from trading_lab.metrics.performance import periods_per_year


@dataclass(frozen=True, slots=True)
class RelativeMetrics:
    excess_return: float
    beta: float | None
    alpha_annualized: float | None
    correlation: float | None
    tracking_error: float | None
    information_ratio: float | None
    bars: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_line(self) -> str:
        def f(v: float | None, fmt: str) -> str:
            return "n/a" if v is None else format(v, fmt)

        return (f"Relative to buy & hold: excess return {self.excess_return:+.2%}, alpha {f(self.alpha_annualized, '+.2%')}/yr, "
                f"beta {f(self.beta, '.2f')}, correlation {f(self.correlation, '.2f')}, "
                f"information ratio {f(self.information_ratio, '+.2f')}")


def relative_metrics(strategy_equity: Sequence[float], benchmark_equity: Sequence[float],
                     timeframe: str) -> RelativeMetrics | None:
    """Both curves start with the initial equity and cover the same bars."""
    s = np.asarray(strategy_equity, dtype="float64")
    b = np.asarray(benchmark_equity, dtype="float64")
    if len(s) != len(b) or len(s) < 3 or s[0] <= 0 or b[0] <= 0:
        return None
    rs, rb = s[1:] / s[:-1] - 1.0, b[1:] / b[:-1] - 1.0
    ppy = periods_per_year(timeframe)
    var_b = float(np.var(rb, ddof=1))
    sd_s = float(np.std(rs, ddof=1))
    beta = float(np.cov(rs, rb, ddof=1)[0, 1] / var_b) if var_b > 0 else None
    alpha = (float(rs.mean()) - beta * float(rb.mean())) * ppy if beta is not None else None
    corr = float(np.corrcoef(rs, rb)[0, 1]) if var_b > 0 and sd_s > 0 else None
    active = rs - rb
    te = float(np.std(active, ddof=1)) * math.sqrt(ppy)
    ir = float(active.mean()) * ppy / te if te > 1e-12 else None
    return RelativeMetrics(
        excess_return=float(s[-1] / s[0] - b[-1] / b[0]),
        beta=beta, alpha_annualized=alpha, correlation=corr,
        tracking_error=te if te > 1e-12 else 0.0, information_ratio=ir, bars=len(rs),
    )
