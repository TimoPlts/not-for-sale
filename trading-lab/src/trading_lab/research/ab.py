"""Is config B better than config A? An A/B comparison over independent windows.

``ab_test`` splits ``[start, end)`` into ``windows`` equal, consecutive,
non-overlapping windows and runs a fresh backtest of each config in every
window (flat, with the initial cash). So each window is a separate,
out-of-sample-like comparison of the two configs on identical data.

* ``b_wins`` counts the windows where B's ``metric`` is better (for
  ``max_drawdown`` lower is better). Ties and undefined values are not
  compared.
* ``p_b_better`` is the one-sided sign test P(at least ``b_wins`` wins out of
  ``compared`` if both were equally good). ``p_a_better`` is the mirror. With
  few windows no result can be significant: 6 windows need 6 out of 6 for
  p < 0.05.
* Both configs must trade the same symbols and timeframe with the same
  initial cash; everything else may differ, and ``config_diff`` lists it.

Data and cached agent answers are shared between all runs. Nothing is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data.base import MarketDataProvider
from trading_lab.llm import LLMProvider
from trading_lab.metrics import PerformanceMetrics
from trading_lab.research.protocol import sign_test_p
from trading_lab.research.sweep import LOWER_IS_BETTER, MemoizedProvider, metric_value


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: value}


def config_diff(a: AppConfig, b: AppConfig) -> list[tuple[str, Any, Any]]:
    """``(dotted key, value in A, value in B)`` for every setting that differs (None = absent)."""
    fa, fb = _flatten(a.to_mapping()), _flatten(b.to_mapping())
    return [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]


@dataclass(frozen=True, slots=True)
class ABWindow:
    start: datetime
    end: datetime
    a: PerformanceMetrics
    b: PerformanceMetrics
    benchmark: PerformanceMetrics | None


@dataclass(frozen=True, slots=True)
class ABResult:
    metric: str
    windows: tuple[ABWindow, ...]
    diff: tuple[tuple[str, Any, Any], ...]

    def _values(self, w: ABWindow) -> tuple[float | None, float | None]:
        return metric_value(w.a, self.metric), metric_value(w.b, self.metric)

    def _better(self, x: float, y: float) -> bool:
        return x < y if self.metric in LOWER_IS_BETTER else x > y

    @property
    def compared(self) -> int:
        return sum(1 for w in self.windows if None not in (v := self._values(w)) and v[0] != v[1])

    @property
    def b_wins(self) -> int:
        return sum(1 for w in self.windows
                   if None not in (v := self._values(w)) and v[0] != v[1] and self._better(v[1], v[0]))

    @property
    def a_wins(self) -> int:
        return self.compared - self.b_wins

    @property
    def p_b_better(self) -> float | None:
        return sign_test_p(self.b_wins, self.compared)

    @property
    def p_a_better(self) -> float | None:
        return sign_test_p(self.a_wins, self.compared)

    @staticmethod
    def _compound(metrics: list[PerformanceMetrics]) -> float:
        total = 1.0
        for m in metrics:
            total *= 1.0 + m.total_return
        return total - 1.0

    @property
    def a_return(self) -> float:
        return self._compound([w.a for w in self.windows])

    @property
    def b_return(self) -> float:
        return self._compound([w.b for w in self.windows])

    @property
    def verdict(self) -> str:
        if self.compared == 0:
            undefined = any(None in self._values(w) for w in self.windows)
            return "no window could be compared" if undefined else "no difference: A and B tie in every window"
        for name, wins, p in (("B", self.b_wins, self.p_b_better), ("A", self.a_wins, self.p_a_better)):
            if p is not None and p < 0.05:
                return f"{name} is better in {wins} of {self.compared} windows (sign test p = {p:.3f})"
        leader = "B" if self.b_wins > self.a_wins else "A" if self.a_wins > self.b_wins else None
        if leader is None:
            return f"no difference: each wins {self.b_wins} of {self.compared} windows"
        wins, p = (self.b_wins, self.p_b_better) if leader == "B" else (self.a_wins, self.p_a_better)
        return (f"{leader} leads {wins} of {self.compared} windows, but that could easily be chance "
                f"(p = {p:.2f}); use more windows or a longer period")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric, "compared": self.compared, "a_wins": self.a_wins, "b_wins": self.b_wins,
            "p_a_better": self.p_a_better, "p_b_better": self.p_b_better, "a_return": self.a_return,
            "b_return": self.b_return, "verdict": self.verdict,
            "diff": [{"key": k, "a": a, "b": b} for k, a, b in self.diff],
            "windows": [{"start": w.start.isoformat(), "end": w.end.isoformat(), "a": w.a.to_dict(),
                         "b": w.b.to_dict(), "benchmark": None if w.benchmark is None else w.benchmark.to_dict()}
                        for w in self.windows],
        }


def ab_windows(start: datetime, end: datetime, windows: int) -> list[tuple[datetime, datetime]]:
    if isinstance(windows, bool) or not isinstance(windows, int) or windows < 2:
        raise ConfigError("use at least 2 windows")
    if end <= start:
        raise ConfigError("the period must end after it starts")
    step = (end - start) / windows
    return [(start + i * step, start + (i + 1) * step) for i in range(windows)]


def ab_test(
    config_a: AppConfig,
    config_b: AppConfig,
    provider: MarketDataProvider,
    start: datetime,
    end: datetime,
    *,
    windows: int = 6,
    metric: str = "total_return",
    llm_provider: LLMProvider | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ABResult:
    for what, a, b in (("symbols", config_a.market.symbols, config_b.market.symbols),
                       ("timeframe", config_a.market.timeframe, config_b.market.timeframe),
                       ("initial cash", config_a.portfolio.initial_cash, config_b.portfolio.initial_cash)):
        if a != b:
            raise ConfigError(f"A and B must use the same {what} for a fair comparison ({a!r} vs {b!r})")
    if metric not in PerformanceMetrics.__dataclass_fields__:
        raise ConfigError(f"unknown metric {metric!r}")
    spans = ab_windows(start, end, windows)
    memo = provider if isinstance(provider, MemoizedProvider) else MemoizedProvider(provider)
    out = []
    for i, (ws, we) in enumerate(spans, 1):
        if progress is not None:
            progress(i, len(spans))
        a = BacktestEngine(config_a, memo, llm_provider=llm_provider).run(ws, we)
        b = BacktestEngine(config_b, memo, llm_provider=llm_provider).run(ws, we)
        out.append(ABWindow(ws, we, a.metrics, b.metrics, a.benchmark))
    return ABResult(metric, tuple(out), tuple(config_diff(config_a, config_b)))


def format_ab(r: ABResult, labels: tuple[str, str] = ("A", "B")) -> str:
    la, lb = labels
    lines = [f"A = {la}", f"B = {lb}"]
    if r.diff:
        lines.append("Settings that differ (A -> B):")
        lines += [f"  {k}: {a!r} -> {b!r}" for k, a, b in r.diff[:25]]
        if len(r.diff) > 25:
            lines.append(f"  ... and {len(r.diff) - 25} more")
    else:
        lines.append("The two configs are identical: expect a tie.")
    pct = r.metric in ("total_return", "annualized_return", "max_drawdown", "win_rate", "exposure")

    def fmt(v: float | None) -> str:
        if v is None:
            return "n/a"
        return f"{v:+.2%}" if pct else f"{v:.2f}"

    lines.append(f"\n{'window':<23} {'A':>9} {'B':>9}  better   ({r.metric})")
    for w in r.windows:
        va, vb = r._values(w)
        better = "-" if va is None or vb is None or va == vb else ("B" if r._better(vb, va) else "A")
        lines.append(f"{w.start:%Y-%m-%d} -> {w.end:%Y-%m-%d}  {fmt(va):>9} {fmt(vb):>9}  {better:^6}")
    lines.append(f"\nReturn over all windows: A {r.a_return:+.2%}, B {r.b_return:+.2%} "
                 "(each window starts flat with the initial cash)")
    lines.append(f"Verdict: {r.verdict}.")
    return "\n".join(lines)
