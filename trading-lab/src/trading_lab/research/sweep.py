"""Parameter sweeps: run one backtest per combination of parameter values.

Parameters are addressed with dotted config keys:

    {"strategies.rsi.period": [7, 14, 21], "voting.min_agreeing": [1, 2]}

Market data is fetched once and reused by every run. Each run is an ordinary,
deterministic backtest, so any row of a sweep can be reproduced exactly from
its parameters.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data.base import MarketDataProvider
from trading_lab.llm import LLMProvider
from trading_lab.metrics import PerformanceMetrics
from trading_lab.storage import SQLiteStore

HIGHER_IS_BETTER = {
    "total_return", "annualized_return", "sharpe_ratio", "sortino_ratio", "win_rate",
    "profit_factor", "avg_trade_return", "final_equity",
}
LOWER_IS_BETTER = {"max_drawdown", "volatility_annualized", "total_fees"}


class MemoizedProvider(MarketDataProvider):
    """Remembers every response so repeated backtests do not refetch data."""

    def __init__(self, inner: MarketDataProvider) -> None:
        self._inner = inner
        self._memo: dict[tuple[Any, ...], pd.DataFrame] = {}
        self.fetches = 0

    @property
    def name(self) -> str:
        return self._inner.name

    def fetch_ohlcv(self, symbol, timeframe, since, until=None):  # type: ignore[no-untyped-def]
        key = (symbol, timeframe, since, until)
        if key not in self._memo:
            self.fetches += 1
            self._memo[key] = self._inner.fetch_ohlcv(symbol, timeframe, since, until)
        return self._memo[key]


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product in a deterministic order (keys sorted, values as given)."""
    if not grid:
        return [{}]
    keys = sorted(grid)
    for key in keys:
        if isinstance(grid[key], (str, bytes)) or not isinstance(grid[key], Sequence) or not grid[key]:
            raise ConfigError(f"grid values for {key!r} must be a non-empty list")
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def apply_params(config: AppConfig, params: Mapping[str, Any]) -> AppConfig:
    """Config with dotted keys set, e.g. ``strategies.rsi.period`` or ``risk.stop_loss_pct``.

    Other keys in the same table are kept (a deep merge), and the result is
    validated like a loaded config.
    """
    data = config.to_mapping()
    for dotted, value in params.items():
        parts = dotted.split(".")
        if len(parts) < 2 or not all(parts):
            raise ConfigError(f"parameter {dotted!r} must look like 'section.key'")
        node: Any = data
        for part in parts[:-1]:
            if part not in node:
                if node is data.get("strategies"):
                    node[part] = {}
                else:
                    raise ConfigError(f"unknown config path {dotted!r}")
            node = node[part]
            if not isinstance(node, dict):
                raise ConfigError(f"config path {dotted!r} does not lead to a table")
        node[parts[-1]] = value
    return AppConfig.from_mapping(data)


def metric_value(metrics: PerformanceMetrics, name: str) -> float | None:
    if not hasattr(metrics, name):
        raise ConfigError(f"unknown metric {name!r}")
    value = getattr(metrics, name)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def rank_key(metric: str) -> Callable[[PerformanceMetrics], tuple[int, float]]:
    """Sort key putting the best result first and undefined values last."""
    sign = 1.0 if metric in LOWER_IS_BETTER else -1.0

    def key(metrics: PerformanceMetrics) -> tuple[int, float]:
        value = metric_value(metrics, metric)
        return (1, 0.0) if value is None else (0, sign * value)

    return key


@dataclass(frozen=True, slots=True)
class SweepResult:
    params: dict[str, Any]
    metrics: PerformanceMetrics
    benchmark: PerformanceMetrics | None
    config_fingerprint: str
    run_id: str | None = None


def run_sweep(
    base_config: AppConfig,
    provider: MarketDataProvider,
    start: datetime,
    end: datetime,
    grid: Mapping[str, Sequence[Any]],
    *,
    metric: str = "sharpe_ratio",
    store: SQLiteStore | None = None,
    label: str = "sweep",
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    llm_provider: LLMProvider | None = None,
) -> list[SweepResult]:
    """Backtest every grid combination; results sorted best-first by ``metric``.

    AI agents in the grid share ``llm_provider`` (default: one built from the
    config) and the configured answer cache, so in record mode a market-only
    agent is asked about each bar once, however many combinations use it.
    """
    rank_key(metric)  # validate the metric name early
    combos = expand_grid(grid)
    configs = [apply_params(base_config, params) for params in combos]  # fail fast on bad keys
    memo = provider if isinstance(provider, MemoizedProvider) else MemoizedProvider(provider)
    results = []
    for n, (params, cfg) in enumerate(zip(combos, configs), start=1):
        if progress is not None:
            progress(n, len(combos), params)
        result = BacktestEngine(cfg, memo, store=store, llm_provider=llm_provider).run(
            start, end, notes=f"{label} {params}" if store is not None else ""
        )
        results.append(SweepResult(params, result.metrics, result.benchmark, cfg.fingerprint(), result.run_id))
    key = rank_key(metric)
    return sorted(results, key=lambda r: key(r.metrics))


def params_key(params: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((k, repr(v)) for k, v in params.items()))


def stability_scores(
    results: Sequence[SweepResult], grid: Mapping[str, Sequence[Any]], metric: str
) -> dict[tuple[tuple[str, str], ...], tuple[float | None, int]]:
    """Per grid point: the mean ``metric`` of the point and its grid neighbours, and how many were averaged.

    Neighbours differ in exactly one parameter, by one position in that parameter's list of values
    (in the order given). A lone peak among poor neighbours scores low; a broad plateau scores high.
    """
    by_key = {params_key(r.params): r for r in results}
    out: dict[tuple[tuple[str, str], ...], tuple[float | None, int]] = {}
    for r in results:
        values = []
        points = [r.params]
        for name, options in grid.items():
            position = next((i for i, v in enumerate(options) if repr(v) == repr(r.params.get(name))), None)
            if position is None:
                continue
            for j in (position - 1, position + 1):
                if 0 <= j < len(options):
                    points.append({**r.params, name: options[j]})
        for point in points:
            other = by_key.get(params_key(point))
            if other is not None and (v := metric_value(other.metrics, metric)) is not None:
                values.append(v)
        out[params_key(r.params)] = (sum(values) / len(values) if values else None, len(values))
    return out


def rank_by_stability(
    results: Sequence[SweepResult], grid: Mapping[str, Sequence[Any]], metric: str
) -> list[SweepResult]:
    """``results`` ordered by their stability score (best first, undefined last)."""
    scores = stability_scores(results, grid, metric)
    sign = 1.0 if metric in LOWER_IS_BETTER else -1.0

    def key(r: SweepResult) -> tuple[int, float]:
        score = scores[params_key(r.params)][0]
        return (1, 0.0) if score is None else (0, sign * score)

    return sorted(results, key=key)
