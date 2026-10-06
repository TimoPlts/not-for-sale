"""Out-of-sample comparison of experiment variants against the baseline.

Used by ``trading-lab experiment --walkforward`` and by the experiment
protocol (docs/EXPERIMENT_PROTOCOL.md). For each variant it summarises the
walk-forward out-of-sample folds and compares them fold by fold with the
baseline's folds (same test windows):

* ``wins``: folds where the variant's out-of-sample ``metric`` beats the
  baseline's (ties and undefined values are left out of ``compared``)
* ``sign_test_p``: one-sided sign test, P(at least ``wins`` wins out of
  ``compared`` if variant and baseline were equally good). A small p-value
  is necessary but not sufficient evidence; see the protocol.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from trading_lab.research.sweep import metric_value


def sign_test_p(wins: int, compared: int) -> float | None:
    """One-sided P(X >= wins) for X ~ Binomial(compared, 0.5)."""
    if compared <= 0:
        return None
    return sum(math.comb(compared, k) for k in range(wins, compared + 1)) / 2 ** compared


def _mean(values: Sequence[float | None]) -> float | None:
    defined = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(defined) / len(defined) if defined else None


@dataclass(frozen=True, slots=True)
class VariantSummary:
    variant: str
    folds: int
    oos_return: float  # compounded over the out-of-sample windows
    benchmark_return: float | None
    worst_fold_drawdown: float  # largest out-of-sample max drawdown of any fold
    mean_sharpe: float | None
    mean_profit_factor: float | None
    total_trades: int
    mean_exposure: float | None
    compared: int = 0  # folds compared with the baseline (both defined, not tied)
    wins: int = 0
    sign_test_p: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(rows: Sequence[Any], *, metric: str = "sharpe_ratio", baseline: str = "baseline") -> list[VariantSummary]:
    """Summaries for walk-forward ``ExperimentRow``s, in the given order."""
    by_name = {r.variant: r for r in rows if r.walkforward is not None}
    base = by_name.get(baseline)
    out = []
    for row in rows:
        wf = row.walkforward
        if wf is None:
            continue
        oos = [f.out_of_sample for f in wf.folds]
        compared = wins = 0
        if base is not None and row.variant != baseline:
            for mine, theirs in zip(wf.folds, base.walkforward.folds):
                a = metric_value(mine.out_of_sample, metric)
                b = metric_value(theirs.out_of_sample, metric)
                if a is None or b is None or a == b:
                    continue
                compared += 1
                wins += int(a > b)
        pf = [m.profit_factor if m.profit_factor is None or math.isfinite(m.profit_factor) else None for m in oos]
        out.append(VariantSummary(
            variant=row.variant,
            folds=len(wf.folds),
            oos_return=wf.out_of_sample_return,
            benchmark_return=wf.benchmark_return,
            worst_fold_drawdown=max((m.max_drawdown for m in oos), default=0.0),
            mean_sharpe=_mean([m.sharpe_ratio for m in oos]),
            mean_profit_factor=_mean(pf),
            total_trades=sum(m.num_trades for m in oos),
            mean_exposure=_mean([m.exposure for m in oos]),
            compared=compared,
            wins=wins,
            sign_test_p=sign_test_p(wins, compared) if compared else None,
        ))
    return out
