"""How much do trading costs decide the result? The same backtest at scaled costs.

``cost_sensitivity`` backtests one config over one period several times. Each
run multiplies every trading cost by a factor (0 = free trading, 1 = as
configured, 2 = twice as expensive, ...):

* the taker and maker fee rates;
* the base slippage (``slippage_bps``) and the volume impact coefficient;
* the short borrow fee.

Everything else is identical, including the data, the signals and the cached
agent answers. So the differences between rows come from costs alone. That
includes *which* trades happen: a costlier entry can shrink or reject an
order.

The **break-even multiplier** is where the total return crosses zero,
interpolated linearly between the two runs around it. A strategy whose
break-even is close to 1 earns little more than it pays; one that loses
money even at 0 has no edge before costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.data.base import MarketDataProvider
from trading_lab.llm import LLMProvider
from trading_lab.metrics import PerformanceMetrics
from trading_lab.research.sweep import MemoizedProvider, apply_params

DEFAULT_MULTIPLIERS: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0)
_SCALED = ("fee_rate", "maker_fee_rate", "slippage_bps", "impact_coefficient", "short_borrow_bps_per_day")


@dataclass(frozen=True, slots=True)
class CostRow:
    multiplier: float
    costs: dict[str, float]  # the scaled [execution] values
    metrics: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class CostSensitivity:
    rows: tuple[CostRow, ...]  # by increasing multiplier

    def row(self, multiplier: float) -> CostRow | None:
        return next((r for r in self.rows if r.multiplier == multiplier), None)

    @property
    def break_even(self) -> float | None:
        """Multiplier where the return crosses zero (None if it never does within the rows)."""
        for a, b in zip(self.rows, self.rows[1:]):
            ra, rb = a.metrics.total_return, b.metrics.total_return
            if ra > 0 >= rb:
                return a.multiplier + (b.multiplier - a.multiplier) * ra / (ra - rb)
        return None

    @property
    def verdict(self) -> str:
        first, last = self.rows[0], self.rows[-1]
        if first.metrics.total_return <= 0:
            return (f"loses money even at {first.multiplier:g}x costs "
                    f"({first.metrics.total_return:+.2%}): no edge before costs in this period")
        if last.metrics.total_return > 0:
            return f"still profitable at {last.multiplier:g}x costs ({last.metrics.total_return:+.2%})"
        be = self.break_even
        assert be is not None
        margin = "thin: costs only a little above today's would erase it" if be < 1.5 else "some room for costs"
        return f"break-even at about {be:.2f}x the configured costs ({margin})"

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [{"multiplier": r.multiplier, "costs": r.costs, "metrics": r.metrics.to_dict()}
                         for r in self.rows],
                "break_even": self.break_even, "verdict": self.verdict}


def scaled_config(config: AppConfig, multiplier: float) -> tuple[AppConfig, dict[str, float]]:
    if not multiplier >= 0:
        raise ValueError(f"cost multipliers must be >= 0, got {multiplier!r}")
    ex = config.execution
    costs = {name: float(getattr(ex, name)) * multiplier for name in _SCALED}
    return apply_params(config, {f"execution.{k}": v for k, v in costs.items()}), costs


def cost_sensitivity(
    config: AppConfig,
    provider: MarketDataProvider,
    start: datetime,
    end: datetime,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    *,
    llm_provider: LLMProvider | None = None,
    progress: Callable[[float], None] | None = None,
) -> CostSensitivity:
    values = sorted(set(float(m) for m in multipliers))
    if len(values) < 2:
        raise ValueError("give at least two different cost multipliers")
    configs = [(m, *scaled_config(config, m)) for m in values]  # validate all before running any
    memo = provider if isinstance(provider, MemoizedProvider) else MemoizedProvider(provider)
    rows = []
    for multiplier, cfg, costs in configs:
        if progress is not None:
            progress(multiplier)
        result = BacktestEngine(cfg, memo, llm_provider=llm_provider).run(start, end)
        rows.append(CostRow(multiplier, costs, result.metrics))
    return CostSensitivity(tuple(rows))


def format_costs(c: CostSensitivity) -> str:
    lines = [f"{'costs':>6} {'fee':>7} {'slip bps':>8} {'return':>9} {'sharpe':>7} {'trades':>6} {'fees paid':>10}"]
    for r in c.rows:
        m = r.metrics
        sharpe = "n/a" if m.sharpe_ratio is None else f"{m.sharpe_ratio:.2f}"
        lines.append(f"{r.multiplier:>5g}x {r.costs['fee_rate']:>7.3%} {r.costs['slippage_bps']:>8.1f} "
                     f"{m.total_return:>+9.2%} {sharpe:>7} {m.num_trades:>6} {m.total_fees:>10,.2f}")
    base, free = c.row(1.0), c.row(0.0)
    if base is not None and free is not None:
        drag = free.metrics.total_return - base.metrics.total_return
        lines.append(f"Costs as configured take {drag * 100:.2f} percentage points of return off the cost-free result "
                     f"({base.metrics.total_fees:,.2f} in fees; slippage comes on top).")
    lines.append(f"Verdict: {c.verdict}.")
    return "\n".join(lines)
