"""Walk-forward evaluation: an honest check for overfitting.

The period is split into consecutive folds. For each fold:

  1. **Train:** sweep the grid on the training window and pick the best
     parameters by ``metric``. This is the in-sample result.
  2. **Test:** backtest those parameters on the window right after it, which
     was not used for the choice. This is the out-of-sample result.

If out-of-sample results are much worse than in-sample ones, the parameter
choice is fitting noise. Each test window starts from fresh initial cash.
The combined out-of-sample return compounds the per-window returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data.base import MarketDataProvider
from trading_lab.llm import LLMProvider
from trading_lab.metrics import PerformanceMetrics
from trading_lab.research.sweep import MemoizedProvider, apply_params, metric_value, run_sweep


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict[str, Any]
    in_sample: PerformanceMetrics
    out_of_sample: PerformanceMetrics
    benchmark: PerformanceMetrics | None


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    metric: str
    folds: tuple[WalkForwardFold, ...]

    @property
    def out_of_sample_return(self) -> float:
        total = 1.0
        for fold in self.folds:
            total *= 1.0 + fold.out_of_sample.total_return
        return total - 1.0

    @property
    def benchmark_return(self) -> float | None:
        if any(f.benchmark is None for f in self.folds):
            return None
        total = 1.0
        for fold in self.folds:
            total *= 1.0 + fold.benchmark.total_return  # type: ignore[union-attr]
        return total - 1.0

    def mean_metric(self, which: str) -> float | None:
        values = [
            metric_value(getattr(f, which), self.metric) for f in self.folds
        ]
        defined = [v for v in values if v is not None]
        return sum(defined) / len(defined) if defined else None


def make_folds(
    start: datetime, end: datetime, train: timedelta, test: timedelta, step: timedelta | None = None
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    step = step or test
    if train <= timedelta(0) or test <= timedelta(0) or step <= timedelta(0):
        raise ConfigError("train, test and step lengths must be positive")
    folds = []
    cursor = start
    while cursor + train + test <= end:
        folds.append((cursor, cursor + train, cursor + train, cursor + train + test))
        cursor += step
    if not folds:
        raise ConfigError("period too short for one train + test window")
    return folds


def walk_forward(
    base_config: AppConfig,
    provider: MarketDataProvider,
    start: datetime,
    end: datetime,
    grid: Mapping[str, Sequence[Any]],
    *,
    train: timedelta,
    test: timedelta,
    step: timedelta | None = None,
    metric: str = "sharpe_ratio",
    progress: Callable[[str], None] | None = None,
    llm_provider: LLMProvider | None = None,
) -> WalkForwardResult:
    memo = provider if isinstance(provider, MemoizedProvider) else MemoizedProvider(provider)
    folds = []
    for n, (tr_start, tr_end, te_start, te_end) in enumerate(make_folds(start, end, train, test, step), 1):
        if progress is not None:
            progress(f"fold {n}: train {tr_start:%Y-%m-%d} -> {tr_end:%Y-%m-%d}, "
                     f"test {te_start:%Y-%m-%d} -> {te_end:%Y-%m-%d}")
        best = run_sweep(base_config, memo, tr_start, tr_end, grid, metric=metric, llm_provider=llm_provider)[0]
        test_run = BacktestEngine(
            apply_params(base_config, best.params), memo, llm_provider=llm_provider
        ).run(te_start, te_end)
        folds.append(
            WalkForwardFold(tr_start, tr_end, te_start, te_end, best.params, best.metrics,
                            test_run.metrics, test_run.benchmark)
        )
    return WalkForwardResult(metric, tuple(folds))
