"""Should this config get a paper run? Every research check, one recommendation.

``validate`` runs, on the same period and data:

1. **checkup**: the nine-plus checks (costs, luck, resampling, Sharpe,
   drawdown, regimes, trial log);
2. **A/B against a baseline** (the config you run now): independent windows
   with a sign test. The baseline is aligned to the candidate's symbols,
   timeframe and cash, so only the strategy settings differ;
3. **outlook and sizing**: the bad-case drawdown and losing streak to
   expect, and the risk per trade for a drawdown budget;
4. **where the money comes from**: the most and least profitable exit types.

Then one recommendation, deliberately strict:

* **not ready**: the checkup fails, or the baseline is clearly better
  (sign test p < 0.05 for the baseline);
* **strong candidate**: the checkup passes and the candidate beats the
  baseline clearly (p < 0.05);
* **paper-trade it next to the baseline**: anything else. A backtest that
  is promising but not proven needs live evidence (``live-compare``), not
  more tuning.

It is one period, in-sample for whatever was tuned on it. A paper run is
the next step for any candidate, never real money. Nothing is stored
(except trial-log rows when ``storage.record_trials`` is on, written by the
CLI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.data.base import MarketDataProvider
from trading_lab.llm import LLMProvider
from trading_lab.research.ab import ABResult, ab_test
from trading_lab.research.checkup import FAIL, PASS, Checkup, checkup
from trading_lab.research.outlook import Outlook, outlook_for_run
from trading_lab.research.sizing import SizingResult, size_for_drawdown
from trading_lab.research.sweep import MemoizedProvider
from trading_lab.research.trades import TradeAnalysis, analyze_trades

NOT_READY, PAPER, STRONG = "not ready", "paper-trade it next to the baseline", "strong candidate"


@dataclass
class Validation:
    label: str
    baseline_label: str
    start: datetime
    end: datetime
    checkup: Checkup
    ab: ABResult | None
    outlook: Outlook | None
    sizing: SizingResult | None
    trades: TradeAnalysis | None
    aligned: list[str] = field(default_factory=list)  # baseline settings aligned to the candidate
    recommendation: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "baseline": self.baseline_label, "start": self.start.isoformat(),
                "end": self.end.isoformat(), "recommendation": self.recommendation, "reasons": list(self.reasons),
                "aligned": list(self.aligned), "checkup": self.checkup.to_dict(),
                "ab": None if self.ab is None else self.ab.to_dict(),
                "outlook": None if self.outlook is None else self.outlook.to_dict(),
                "sizing": None if self.sizing is None else self.sizing.to_dict(),
                "trades": None if self.trades is None else {
                    k: v for k, v in self.trades.to_dict().items() if k != "trades"}}


def align(baseline: AppConfig, candidate: AppConfig) -> tuple[AppConfig, list[str]]:
    """The baseline with the candidate's symbols, timeframe and cash (an A/B test needs them equal)."""
    changes = []
    overrides: dict[str, dict[str, Any]] = {}
    if baseline.market.symbols != candidate.market.symbols:
        overrides.setdefault("market", {})["symbols"] = list(candidate.market.symbols)
        changes.append("symbols")
    if baseline.market.timeframe != candidate.market.timeframe:
        overrides.setdefault("market", {})["timeframe"] = candidate.market.timeframe
        changes.append("timeframe")
    if baseline.portfolio.initial_cash != candidate.portfolio.initial_cash:
        overrides.setdefault("portfolio", {})["initial_cash"] = candidate.portfolio.initial_cash
        changes.append("initial cash")
    return (baseline.with_overrides(overrides) if overrides else baseline), changes


def recommend(c: Checkup, ab: ABResult | None) -> tuple[str, list[str]]:
    reasons = []
    failed = [ch.name for ch in c.checks if ch.status == FAIL]
    if failed:
        reasons.append(f"the checkup failed: {', '.join(failed)}")
    baseline_better = ab is not None and ab.p_a_better is not None and ab.p_a_better < 0.05
    candidate_better = ab is not None and ab.p_b_better is not None and ab.p_b_better < 0.05
    if baseline_better:
        reasons.append(f"the baseline was better in {ab.a_wins} of {ab.compared} windows (p = {ab.p_a_better:.3f})")
    if failed or baseline_better:
        return NOT_READY, reasons
    if c.overall == PASS and candidate_better:
        assert ab is not None
        return STRONG, [f"every check passed and it beat the baseline in {ab.b_wins} of {ab.compared} windows "
                        f"(p = {ab.p_b_better:.3f})"]
    warned = [ch.name for ch in c.checks if ch.status != PASS and ch.status != FAIL and ch.status != "n/a"]
    if warned:
        reasons.append(f"the checkup warns about: {', '.join(warned)}")
    if ab is not None:
        reasons.append(f"against the baseline: {ab.verdict}")
    return PAPER, reasons or ["nothing failed, but nothing is proven either"]


def validate(config: AppConfig, baseline: AppConfig, provider: MarketDataProvider, start: datetime, end: datetime,
             *, label: str = "candidate", baseline_label: str = "baseline", windows: int = 6,
             permutations: int = 50, max_drawdown: float = 0.20, llm_provider: LLMProvider | None = None,
             allow_agents: bool = False, trials: Any = None,
             progress: Callable[[str], None] | None = None) -> Validation:
    from trading_lab.storage import SQLiteStore

    say = progress or (lambda message: None)
    memo = provider if isinstance(provider, MemoizedProvider) else MemoizedProvider(provider)
    check = checkup(config, memo, start, end, permutations=permutations, llm_provider=llm_provider,
                    allow_agents=allow_agents, progress=say, trials=trials)
    aligned_baseline, aligned = align(baseline, config)
    ab = None
    if aligned_baseline.fingerprint() != config.fingerprint():
        say("A/B against the baseline")
        ab = ab_test(aligned_baseline, config, memo, start, end, windows=windows, llm_provider=llm_provider)
    say("outlook and sizing")
    with SQLiteStore(":memory:") as store:
        run = BacktestEngine(config, memo, store=store, llm_provider=llm_provider).run(start, end)
        outlook = outlook_for_run(store, run.run_id, samples=2000)
        sizing = size_for_drawdown(store, run.run_id, max_drawdown, samples=1000) if outlook is not None else None
        trades = analyze_trades(store, run.run_id, groupings=("exit",)) if run.trades else None
    out = Validation(label, baseline_label, start, end, check, ab, outlook, sizing, trades, aligned)
    out.recommendation, out.reasons = recommend(check, ab)
    return out


def format_validation(v: Validation) -> str:
    c, m = v.checkup, v.checkup.metrics
    lines = [f"Validation of {v.label} ({v.start:%Y-%m-%d} -> {v.end:%Y-%m-%d})",
             f"  backtest: return {m.total_return:+.2%}, max drawdown {-m.max_drawdown:.1%}, {m.num_trades} trades, "
             f"Sharpe {'n/a' if m.sharpe_ratio is None else f'{m.sharpe_ratio:.2f}'}"
             + ("" if c.benchmark is None else f" (buy & hold {c.benchmark.total_return:+.2%})"),
             f"  checkup: {c.overall.upper()} ({c.verdict})"]
    for ch in c.checks:
        if ch.status != PASS:
            lines.append(f"    [{ch.status.upper():<4}] {ch.name}: {ch.detail}")
    if v.ab is None:
        lines.append(f"  against {v.baseline_label}: identical configs, no A/B test")
    else:
        lines.append(f"  against {v.baseline_label} (A = baseline, B = candidate; {v.ab.compared} windows "
                     f"compared): {v.ab.verdict}; "
                     f"return {v.ab.b_return:+.2%} vs {v.ab.a_return:+.2%}"
                     + (f" (baseline aligned: {', '.join(v.aligned)})" if v.aligned else ""))
    if v.outlook is not None:
        o = v.outlook
        lines.append(f"  outlook ({o.horizon} trades): bad-case drawdown {o.drawdown_bad:.1%}, "
                     f"bad-case losing streak {o.streak_bad}, chance of a loss {o.prob_loss:.0%}")
    if v.sizing is not None:
        s = v.sizing
        lines.append(f"  sizing for a {s.target:.0%} bad-case drawdown: "
                     + ("not reachable within the limits" if s.scale is None
                        else f"risk_per_trade_pct {s.suggested_risk_pct:.4f} ({s.scale:.2f}x the current)"))
    if v.trades is not None and v.trades.groups.get("exit"):
        groups = v.trades.groups["exit"]
        best, worst = max(groups, key=lambda g: g.pnl), min(groups, key=lambda g: g.pnl)
        lines.append(f"  exits: best {best.name} ({best.pnl:+,.2f} over {best.trades}), "
                     f"worst {worst.name} ({worst.pnl:+,.2f} over {worst.trades})")
    lines.append(f"Recommendation: {v.recommendation.upper()}")
    lines += [f"  - {r}" for r in v.reasons]
    lines.append("One period, in-sample for anything tuned on it. The next step for any candidate is a paper run "
                 "next to the baseline (trading-lab-paper@NAME, then live-compare), never real money.")
    return "\n".join(lines)


def validation_html(v: Validation) -> str:
    """The checkup page plus the A/B, outlook and recommendation, self-contained."""
    from trading_lab.html_report import _e
    from trading_lab.research.checkup import checkup_html

    page = checkup_html(v.checkup, title=f"Validation of {v.label}")
    summary = "".join(f"<p>{_e(line.strip())}</p>" for line in format_validation(v).splitlines()[1:])
    block = (f'<h2>Recommendation: {_e(v.recommendation.upper())}</h2><div class="card">{summary}</div>')
    return page.replace("<h2>Checks</h2>", block + "<h2>Checks</h2>", 1)
