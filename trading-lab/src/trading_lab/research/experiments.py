"""Baseline versus AI experiments: the same period, costs and risk rules, different voters.

A *variant* says which voters take part. It only changes strategy weights
in the config; everything else (period, fees, slippage, liquidity, breakers,
voting thresholds) is identical across variants:

=================  ===================================================
``baseline``       the deterministic strategies only (RSI, MACD, Bollinger)
``trend``          baseline + ``qwen_trend``
``momentum``       baseline + ``qwen_momentum``
``risk``           baseline + ``qwen_risk``
``trend_momentum`` baseline + ``qwen_trend`` + ``qwen_momentum``
``all_agents``     baseline + all three agents
``ai_only``        the three agents only (deterministic strategies at weight 0)
=================  ===================================================

An agent switched on keeps its configured weight, or gets 1.0 when the
config has it at 0. A switched-off voter gets weight 0 and is not built at
all, so it is never called and casts no vote.

All variants share the market data, one LLM provider and the agent answer
cache. In record mode a market-only agent (``qwen_trend``, ``qwen_momentum``
by default) is asked about each bar once and every variant reuses that
answer, so differences between variants come from the ensemble, not from
the model answering differently. ``qwen_risk`` sees the portfolio by
default, so it is asked again wherever a variant's trades differ.
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
from trading_lab.research.sweep import MemoizedProvider, apply_params
from trading_lab.research.walkforward import WalkForwardResult, walk_forward
from trading_lab.storage import SQLiteStore

AGENT_STRATEGIES: tuple[str, ...] = ("qwen_trend", "qwen_momentum", "qwen_risk")

VARIANTS: dict[str, tuple[str, tuple[str, ...], bool]] = {
    # name: (description, agents switched on, deterministic strategies on?)
    "baseline": ("RSI + MACD + Bollinger", (), True),
    "trend": ("baseline + Qwen Trend", ("qwen_trend",), True),
    "momentum": ("baseline + Qwen Momentum", ("qwen_momentum",), True),
    "risk": ("baseline + Qwen Risk", ("qwen_risk",), True),
    "trend_momentum": ("baseline + Qwen Trend + Momentum", ("qwen_trend", "qwen_momentum"), True),
    "all_agents": ("baseline + all 3 Qwen agents", AGENT_STRATEGIES, True),
    "ai_only": ("the 3 Qwen agents only", AGENT_STRATEGIES, False),
}
DEFAULT_VARIANTS: tuple[str, ...] = ("baseline", "trend", "trend_momentum", "all_agents")


def variant_overrides(config: AppConfig, variant: str) -> dict[str, Any]:
    """Dotted-key overrides (as for ``apply_params``) that turn ``config`` into ``variant``."""
    if variant not in VARIANTS:
        raise ConfigError(f"unknown variant {variant!r}; available: {list(VARIANTS)}")
    _, agents_on, baseline_on = VARIANTS[variant]
    specs = {s.name: s for s in config.strategies}
    overrides: dict[str, Any] = {}
    for name in AGENT_STRATEGIES:
        spec = specs.get(name)
        on = name in agents_on
        weight = (spec.weight if spec is not None and spec.weight > 0 else 1.0) if on else 0.0
        overrides[f"strategies.{name}.weight"] = weight
        if on:
            overrides[f"strategies.{name}.enabled"] = True
    if not baseline_on:
        for spec in config.strategies:
            if spec.name not in AGENT_STRATEGIES and spec.enabled and spec.weight > 0:
                overrides[f"strategies.{spec.name}.weight"] = 0.0
    return overrides


def variant_config(config: AppConfig, variant: str) -> AppConfig:
    return apply_params(config, variant_overrides(config, variant))


@dataclass(frozen=True, slots=True)
class ExperimentRow:
    variant: str
    description: str
    config_fingerprint: str
    metrics: PerformanceMetrics | None = None  # backtest mode
    benchmark: PerformanceMetrics | None = None
    walkforward: WalkForwardResult | None = None  # walk-forward mode
    run_id: str | None = None
    signals: tuple[Any, ...] = ()  # kept for usage accounting / attribution

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"variant": self.variant, "description": self.description,
                               "config_fingerprint": self.config_fingerprint, "run_id": self.run_id}
        if self.metrics is not None:
            out["metrics"] = self.metrics.to_dict()
        if self.benchmark is not None:
            out["benchmark"] = self.benchmark.to_dict()
        if self.walkforward is not None:
            wf = self.walkforward
            out["walkforward"] = {
                "metric": wf.metric,
                "out_of_sample_return": wf.out_of_sample_return,
                "benchmark_return": wf.benchmark_return,
                "mean_in_sample": wf.mean_metric("in_sample"),
                "mean_out_of_sample": wf.mean_metric("out_of_sample"),
                "folds": [
                    {"test_start": f.test_start, "test_end": f.test_end, "best_params": f.best_params,
                     "in_sample": f.in_sample.to_dict(), "out_of_sample": f.out_of_sample.to_dict(),
                     "benchmark": None if f.benchmark is None else f.benchmark.to_dict()}
                    for f in wf.folds
                ],
            }
        return out


def run_experiment(
    base_config: AppConfig,
    provider: MarketDataProvider,
    start: datetime,
    end: datetime,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    *,
    walkforward: bool = False,
    grid: Mapping[str, Sequence[Any]] | None = None,
    train: timedelta = timedelta(days=90),
    test: timedelta = timedelta(days=30),
    metric: str = "sharpe_ratio",
    store: SQLiteStore | None = None,
    llm_provider: LLMProvider | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[ExperimentRow]:
    """Run every variant over the same period, in the order given.

    Backtest mode runs one backtest per variant (``grid`` is not used).
    Walk-forward mode runs ``walk_forward`` per variant with ``grid`` (which
    may be empty): parameters are chosen in-sample and judged out-of-sample.
    """
    if not variants:
        raise ConfigError("give at least one variant")
    configs = {v: variant_config(base_config, v) for v in variants}  # fail fast on bad names/configs
    memo = provider if isinstance(provider, MemoizedProvider) else MemoizedProvider(provider)
    rows = []
    for variant in variants:
        cfg = configs[variant]
        description = VARIANTS[variant][0]
        if progress is not None:
            progress(f"{variant}: {description}")
        if walkforward:
            result = walk_forward(cfg, memo, start, end, dict(grid or {}), train=train, test=test,
                                  metric=metric, llm_provider=llm_provider)
            rows.append(ExperimentRow(variant, description, cfg.fingerprint(), walkforward=result))
        else:
            backtest = BacktestEngine(cfg, memo, store=store, llm_provider=llm_provider).run(
                start, end, notes=f"experiment {variant}" if store is not None else ""
            )
            rows.append(ExperimentRow(variant, description, cfg.fingerprint(), backtest.metrics,
                                      backtest.benchmark, run_id=backtest.run_id, signals=backtest.signals))
    return rows
