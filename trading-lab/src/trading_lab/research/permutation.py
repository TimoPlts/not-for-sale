"""Could a market without any pattern have produced this result? A permutation test.

``permutation_test`` backtests a config on the real candles, then many times on
*shuffled* versions of the same period:

* each candle is described relative to the previous close (log open, high,
  low and close ratios, plus its volume);
* the candles of the test period are put in a random order, the **same**
  order for every symbol, so the links between coins survive;
* prices are rebuilt from the last real close before the period, and the
  warm-up history before the period stays real.

Shuffling keeps each candle's shape, the volatility, the overall drift (the
same candles, so the same total move) and the cross-coin correlation, but
destroys any order a strategy could exploit: trends, momentum, mean
reversion. So the shuffled runs show what the strategy earns from luck and
from simply being in the market. The **p-value** is the share of shuffled
runs at least as good as the real one (counting the real run itself, so it
is never 0): small means the result is unlikely to be luck on this data.

It is still one period, in-sample: a small p-value is necessary, not
sufficient. AI agents would be asked about every shuffled market (many model
calls), so they are refused unless ``allow_agents`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data.base import MarketDataProvider, to_utc_timestamp
from trading_lab.llm import LLMProvider
from trading_lab.metrics import PerformanceMetrics
from trading_lab.research.sweep import LOWER_IS_BETTER, MemoizedProvider, metric_value


def permute_candles(frame: pd.DataFrame, start: datetime, seed: int) -> pd.DataFrame:
    """``frame`` with the candles from ``start`` on reordered (see the module docstring)."""
    if frame.empty:
        return frame
    first = int(frame.index.searchsorted(to_utc_timestamp(start)))
    n = len(frame) - first
    if n < 2:
        return frame
    prev_close = frame["close"].shift(1)
    anchor = float(frame["close"].iloc[first - 1]) if first > 0 else float(frame["open"].iloc[0])
    if first == 0:
        prev_close.iloc[0] = anchor
    period = frame.iloc[first:]
    base = prev_close.iloc[first:].to_numpy(dtype="float64")
    rel = {k: np.log(period[k].to_numpy(dtype="float64") / base) for k in ("open", "high", "low", "close")}
    order = np.random.default_rng([seed, n]).permutation(n)  # same length -> same order for every symbol
    closes = anchor * np.exp(np.cumsum(rel["close"][order]))
    prevs = np.concatenate(([anchor], closes[:-1]))
    shuffled = pd.DataFrame({
        "open": prevs * np.exp(rel["open"][order]),
        "high": prevs * np.exp(rel["high"][order]),
        "low": prevs * np.exp(rel["low"][order]),
        "close": closes,
        "volume": period["volume"].to_numpy(dtype="float64")[order],
    }, index=period.index)
    return pd.concat([frame.iloc[:first], shuffled])


class PermutedProvider(MarketDataProvider):
    """Serves ``inner``'s candles with the period from ``start`` shuffled (read-only, deterministic)."""

    def __init__(self, inner: MarketDataProvider, start: datetime, seed: int) -> None:
        self._inner, self._start, self._seed = inner, start, seed

    @property
    def name(self) -> str:
        return f"{self._inner.name}-permuted-{self._seed}"

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: datetime, until: datetime | None = None
                    ) -> pd.DataFrame:
        return permute_candles(self._inner.fetch_ohlcv(symbol, timeframe, since, until), self._start, self._seed)


@dataclass(frozen=True, slots=True)
class PermutationResult:
    metric: str
    real: PerformanceMetrics
    permuted: tuple[float | None, ...]  # the metric of each shuffled run

    @property
    def real_value(self) -> float | None:
        return metric_value(self.real, self.metric)

    @property
    def defined(self) -> list[float]:
        return [v for v in self.permuted if v is not None]

    @property
    def at_least_as_good(self) -> int:
        real = self.real_value
        if real is None:
            return 0
        lower = self.metric in LOWER_IS_BETTER
        return sum(1 for v in self.defined if (v <= real if lower else v >= real))

    @property
    def p_value(self) -> float | None:
        if self.real_value is None or not self.defined:
            return None
        return (1 + self.at_least_as_good) / (1 + len(self.defined))

    def percentiles(self) -> tuple[float, float, float] | None:
        if not self.defined:
            return None
        p5, p50, p95 = np.percentile(self.defined, [5, 50, 95])
        return float(p5), float(p50), float(p95)

    @property
    def verdict(self) -> str:
        p = self.p_value
        if p is None:
            return "not measurable (undefined metric)"
        if p < 0.05:
            return (f"unlikely to be luck: only {p:.1%} of shuffled markets did as well "
                    "(still one period, in-sample: confirm it out of sample)")
        if p < 0.2:
            return f"weak evidence: {p:.0%} of shuffled markets did as well"
        return f"indistinguishable from luck: {p:.0%} of shuffled markets did as well"

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "real": self.real_value, "permuted": list(self.permuted),
                "p_value": self.p_value, "percentiles_5_50_95": self.percentiles(), "verdict": self.verdict}


def permutation_test(
    config: AppConfig,
    provider: MarketDataProvider,
    start: datetime,
    end: datetime,
    *,
    permutations: int = 100,
    metric: str = "total_return",
    seed: int = 0,
    llm_provider: LLMProvider | None = None,
    allow_agents: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> PermutationResult:
    from trading_lab.strategy_factory import needs_llm

    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
        raise ConfigError("permutations must be a positive integer")
    if metric not in PerformanceMetrics.__dataclass_fields__:
        raise ConfigError(f"unknown metric {metric!r}")
    if needs_llm(config) and not allow_agents:
        raise ConfigError("AI agents would be asked about every shuffled market (many model calls); "
                          "set their weights to 0 for this test, or allow it explicitly")
    memo = provider if isinstance(provider, MemoizedProvider) else MemoizedProvider(provider)
    real = BacktestEngine(config, memo, llm_provider=llm_provider).run(start, end).metrics
    values = []
    for i in range(permutations):
        if progress is not None:
            progress(i + 1, permutations)
        shuffled = PermutedProvider(memo, start, seed + i)
        values.append(metric_value(BacktestEngine(config, shuffled, llm_provider=llm_provider)
                                   .run(start, end).metrics, metric))
    return PermutationResult(metric, real, tuple(values))


def format_permutation(r: PermutationResult) -> str:
    pct = r.metric in ("total_return", "annualized_return", "max_drawdown", "win_rate", "exposure")

    def fmt(v: float | None) -> str:
        return "n/a" if v is None else (f"{v:+.2%}" if pct else f"{v:.2f}")

    lines = [f"Real market: {r.metric} {fmt(r.real_value)}"]
    pcts = r.percentiles()
    if pcts is not None:
        lines.append(f"Shuffled markets ({len(r.defined)}): 5% {fmt(pcts[0])} | median {fmt(pcts[1])} | "
                     f"95% {fmt(pcts[2])}")
        lines.append(f"At least as good as the real result: {r.at_least_as_good} of {len(r.defined)} "
                     f"(p = {r.p_value:.3f})" if r.p_value is not None else "")
    lines.append(f"Verdict: {r.verdict}.")
    return "\n".join(line for line in lines if line)
