"""Is this strategy any good? One command that runs every check and gives one verdict.

``checkup`` backtests a config over a period (in memory, nothing is stored)
and runs the research tools on it:

==========================  ===================================================
Enough trades               at least 30 closed trades (fewer: results are noise)
Edge before costs           profitable with free trading (``costs`` at 0x)
Survives costs              profitable as configured, with room (break-even >= 1.5x)
Not luck                    the permutation test against shuffled markets
Robust to resampling        bootstrap probability of a loss
Sharpe is real              probabilistic Sharpe ratio: P(true Sharpe > 0)
Beats buy & hold            excess return over equal-weight buy & hold
Drawdown                    worst peak-to-trough loss
Works in several regimes    profitable in more than one kind of market
Beats your other trials     deflated Sharpe against the trial log (only when it
                            holds earlier trials on overlapping data)
==========================  ===================================================

Each check passes, warns or fails, with the number behind it and what to do
next. The overall verdict is the worst check. It is one period, in-sample:
even a full pass needs confirming out of sample (``walkforward``, ``ab``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.data.base import MarketDataProvider
from trading_lab.llm import LLMProvider
from trading_lab.metrics import PerformanceMetrics
from trading_lab.research.costs import CostSensitivity, cost_sensitivity
from trading_lab.research.permutation import PermutationResult, permutation_test
from trading_lab.research.regimes import RegimeReport, regimes_for_run
from trading_lab.research.robustness import Robustness, robustness_for_run
from trading_lab.research.sweep import MemoizedProvider

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "n/a"
_RANK = {PASS: 0, NA: 0, WARN: 1, FAIL: 2}


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: str
    detail: str
    advice: str = ""


@dataclass
class Checkup:
    start: datetime
    end: datetime
    metrics: PerformanceMetrics
    benchmark: PerformanceMetrics | None
    excess_return: float | None
    costs: CostSensitivity
    permutation: PermutationResult | None
    robustness: Robustness
    regimes: RegimeReport
    checks: list[Check] = field(default_factory=list)
    trials: int = 0  # logged trials on overlapping data, this one included (0 = no log)
    deflated_sharpe: float | None = None  # against those trials

    @property
    def overall(self) -> str:
        worst = max((_RANK[c.status] for c in self.checks), default=0)
        return {0: PASS, 1: WARN, 2: FAIL}[worst]

    @property
    def verdict(self) -> str:
        failed = [c.name for c in self.checks if c.status == FAIL]
        warned = [c.name for c in self.checks if c.status == WARN]
        if failed:
            return f"not convincing: failed {', '.join(failed)}"
        if warned:
            return f"promising but not proven: check {', '.join(warned)}"
        return "passes every check on this period; confirm it out of sample before relying on it"

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(), "end": self.end.isoformat(), "overall": self.overall,
            "verdict": self.verdict, "metrics": self.metrics.to_dict(),
            "benchmark": None if self.benchmark is None else self.benchmark.to_dict(),
            "excess_return": self.excess_return, "costs": self.costs.to_dict(),
            "permutation": None if self.permutation is None else self.permutation.to_dict(),
            "prob_loss": self.robustness.prob_loss, "regimes": self.regimes.to_dict(),
            "trials": self.trials, "deflated_sharpe": self.deflated_sharpe,
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail, "advice": c.advice}
                       for c in self.checks],
        }


def _grade(value: float | None, good: Callable[[float], bool], ok: Callable[[float], bool]) -> str:
    if value is None:
        return NA
    return PASS if good(value) else WARN if ok(value) else FAIL


def evaluate(c: Checkup) -> list[Check]:
    m = c.metrics
    lost = m.total_return <= 0
    checks = [Check(
        "Enough trades", _grade(m.num_trades, lambda v: v >= 30, lambda v: v >= 10),
        f"{m.num_trades} closed trades",
        "use a longer period or more symbols: with few trades every other number is mostly noise",
    )]
    free, base = c.costs.row(0.0), c.costs.row(1.0)
    if free is not None:
        r0 = free.metrics.total_return
        checks.append(Check("Edge before costs", PASS if r0 > 0 else FAIL, f"{r0:+.2%} with free trading",
                            "the signals themselves lose money here: change the strategy, not the costs"))
    if base is not None:
        r1, be = base.metrics.total_return, c.costs.break_even
        status = FAIL if r1 <= 0 else WARN if be is not None and be < 1.5 else PASS
        detail = f"{r1:+.2%} as configured" + (f", break-even at {be:.2f}x costs" if be is not None else "")
        checks.append(Check("Survives costs", status, detail,
                            "trade less often, use limit entries or a longer timeframe to cut costs"))
    if c.permutation is None:
        checks.append(Check("Not luck", NA, "skipped (AI agents would be asked about every shuffled market)"))
    else:
        p = c.permutation.p_value
        checks.append(Check(
            "Not luck", _grade(p, lambda v: v < 0.05, lambda v: v < 0.2),
            "n/a" if p is None else f"p = {p:.3f} ({c.permutation.at_least_as_good} of "
                                    f"{len(c.permutation.defined)} shuffled markets did as well)",
            "it lost money, so there is no result to tell apart from luck" if lost
            else "the result is within what shuffled markets produce: it may be luck",
        ))
    pl = c.robustness.prob_loss
    checks.append(Check(
        "Robust to resampling", _grade(pl, lambda v: v < 0.1, lambda v: v < 0.3),
        "n/a" if pl is None else f"{pl:.0%} chance of a loss when the trades are resampled",
        "it lost money; resampling the trades only confirms that" if lost
        else "the result hangs on a few trades: look for a more consistent edge",
    ))
    psr = m.probabilistic_sharpe
    checks.append(Check(
        "Sharpe is real", _grade(psr, lambda v: v >= 0.95, lambda v: v >= 0.8),
        "n/a" if psr is None else f"{psr:.0%} chance the true Sharpe ratio is above 0 (sample length, skew, fat tails)",
        "it lost money, so its Sharpe ratio is not above 0" if lost
        else "the Sharpe ratio could still be 0: use a longer period or more symbols for more evidence",
    ))
    if c.excess_return is not None:
        checks.append(Check(
            "Beats buy & hold", PASS if c.excess_return > 0 else WARN,
            f"{c.excess_return:+.2%} versus equal-weight buy & hold",
            "simply holding the coins did better; that is fine only if the risk was much lower",
        ))
    dd = m.max_drawdown
    checks.append(Check("Drawdown", _grade(dd, lambda v: v <= 0.20, lambda v: v <= 0.35),
                        f"worst drawdown {-dd:.1%}", "reduce position sizes or add stops and filters"))
    combined = [r for r in c.regimes.combined if r.bars >= max(10, 0.05 * max(c.regimes.total_bars, 1))]
    if combined:
        positive = sum(1 for r in combined if r.strategy_return > 0)
        status = PASS if positive * 2 >= len(combined) else WARN if positive else FAIL
        checks.append(Check("Works in several regimes", status,
                            f"profitable in {positive} of {len(combined)} market regimes",
                            "see `trading-lab regimes`: consider a trend filter or allow_short for the weak ones"))
    else:
        checks.append(Check("Works in several regimes", NA, "the period is too short to classify regimes"))
    if c.trials >= 2:
        dsr = c.deflated_sharpe
        checks.append(Check(
            "Beats your other trials", _grade(dsr, lambda v: v >= 0.95, lambda v: v >= 0.5),
            "n/a" if dsr is None else f"deflated Sharpe {dsr:.0%} against {c.trials} logged trials on overlapping "
                                      "data",
            "it lost money, so it cannot beat the luck bar of the other trials" if lost
            else "after this many tries, a result this good is within what the luckiest of them would show: confirm "
                 "it on data you have not tried configs on",
        ))
    return checks


def checkup(
    config: AppConfig,
    provider: MarketDataProvider,
    start: datetime,
    end: datetime,
    *,
    permutations: int = 50,
    multipliers: Sequence[float] = (0.0, 1.0, 2.0),
    llm_provider: LLMProvider | None = None,
    allow_agents: bool = False,
    progress: Callable[[str], None] | None = None,
    trials: Any = None,
) -> Checkup:
    """``trials``: a ``research.trials.TrialSummary`` of earlier trials on overlapping data, if logged."""
    from trading_lab.storage import SQLiteStore
    from trading_lab.strategy_factory import needs_llm

    say = progress or (lambda message: None)
    memo = provider if isinstance(provider, MemoizedProvider) else MemoizedProvider(provider)
    say("backtest")
    with SQLiteStore(":memory:") as store:  # the research tools read a stored run; nothing touches disk
        result = BacktestEngine(config, memo, store=store, llm_provider=llm_provider).run(start, end)
        robust = robustness_for_run(store, result.run_id, samples=2000)
        regimes = regimes_for_run(store, result.run_id)
    say("costs")
    costs = cost_sensitivity(config, memo, start, end, multipliers, llm_provider=llm_provider)
    perm = None
    if allow_agents or not needs_llm(config):
        say("permutation test")
        perm = permutation_test(config, memo, start, end, permutations=permutations, llm_provider=llm_provider,
                                allow_agents=allow_agents)
    excess = None if result.relative is None else result.relative.excess_return
    report = Checkup(start, end, result.metrics, result.benchmark, excess, costs, perm, robust, regimes)
    if trials is not None and trials.count:
        from trading_lab.research.trials import deflated_against_log, trial, trial_key

        equity = [config.portfolio.initial_cash, *result.equity_curve["equity"].tolist()]
        returns = [b / a - 1.0 for a, b in zip(equity, equity[1:])]
        this = trial("checkup", config, start, end, result.metrics)
        report.trials = trials.count + (trial_key(this) not in {trial_key(r) for r in trials.unique})
        report.deflated_sharpe = deflated_against_log(trials, returns, include=this)
    report.checks = evaluate(report)
    return report


_MARK = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", NA: " -- "}


def format_checkup(c: Checkup) -> str:
    m = c.metrics
    lines = [f"Checkup {c.start:%Y-%m-%d} -> {c.end:%Y-%m-%d}: return {m.total_return:+.2%}, "
             f"max drawdown {-m.max_drawdown:.1%}, {m.num_trades} trades"
             + ("" if c.benchmark is None else f" (buy & hold {c.benchmark.total_return:+.2%})"), ""]
    width = max(len(ch.name) for ch in c.checks)
    for ch in c.checks:
        lines.append(f"  [{_MARK[ch.status]}] {ch.name:<{width}}  {ch.detail}")
    advice = [ch for ch in c.checks if ch.status in (FAIL, WARN) and ch.advice]
    if advice:
        lines.append("\nNext steps:")
        lines += [f"  - {ch.name}: {ch.advice}" for ch in advice]
    lines.append(f"\nVerdict: {c.verdict}.")
    return "\n".join(lines)


def checkup_html(c: Checkup, title: str = "Strategy checkup") -> str:
    """A self-contained page in the style of the run report."""
    from trading_lab.html_report import _CSS, _e, _table, _td

    colour = {PASS: "pos", WARN: "", FAIL: "neg", NA: "muted"}
    rows = [[_td(ch.name), _td(_MARK[ch.status].strip() or "n/a", colour[ch.status]), _td(ch.detail),
             _td(ch.advice if ch.status in (FAIL, WARN) else "")] for ch in c.checks]
    m = c.metrics
    bench = "" if c.benchmark is None else f" · buy & hold {c.benchmark.total_return:+.2%}"
    return "\n".join([
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_e(title)}</title><style>{_CSS}"
        ".checks th,.checks td{text-align:left;white-space:normal;vertical-align:top}</style></head>",
        '<body class="viz-root"><main>',
        f"<h1>{_e(title)}</h1>",
        f'<div class="secondary">{c.start:%Y-%m-%d} → {c.end:%Y-%m-%d} · return {m.total_return:+.2%} · '
        f"max drawdown {-m.max_drawdown:.1%} · {m.num_trades} trades{_e(bench)}</div>",
        f'<h2>Verdict: {_e(c.overall.upper())}</h2><div class="card"><p>{_e(c.verdict)}.</p></div>',
        '<h2>Checks</h2><div class="card checks">' + _table(["Check", "Result", "Detail", "Next step"], rows)
        + "</div>",
        '<p class="muted">One period, in-sample. Confirm any pass out of sample (walkforward, ab).</p>',
        "</main></body></html>",
    ])
