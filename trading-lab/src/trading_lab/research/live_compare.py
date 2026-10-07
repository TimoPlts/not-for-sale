"""Which of two paper runs is doing better? A comparison over the time they ran together.

``live_compare`` reads two stored runs (usually paper runs trading side by
side, see ``trading-lab-paper@NAME``) and compares them only over their
**overlap**: from the later of their first bars to the earlier of their last
bars. Each run is measured from its equity just before the overlap (or its
initial cash when it started there), so a run that started earlier gets no
head start.

* **metrics** over the overlap: return, drawdown, Sharpe, the trades closed
  in it, fees and time in the market (``compute_metrics``, as everywhere);
* **daily returns** (UTC days, last equity of each day): ``a_wins`` and
  ``b_wins`` count the days each run did better, ties (e.g. both flat) are
  not compared, and the one-sided sign test gives ``p_a_better`` and
  ``p_b_better``. Days are not fully independent (positions span days), so
  the p-value is a rough guide;
* the **settings that differ** between the two configs.

Until ``min_days`` days have been compared the verdict is "too early to
tell". Read-only: it never changes either run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from trading_lab.config import AppConfig
from trading_lab.metrics import PerformanceMetrics, compute_metrics
from trading_lab.research.ab import config_diff
from trading_lab.research.protocol import sign_test_p


@dataclass(frozen=True, slots=True)
class DayResult:
    day: date
    a: float
    b: float

    @property
    def better(self) -> str | None:
        return None if self.a == self.b else "A" if self.a > self.b else "B"


@dataclass
class LiveComparison:
    run_a: dict[str, Any]
    run_b: dict[str, Any]
    start: datetime | None = None  # the overlap (bar times, UTC)
    end: datetime | None = None
    a: PerformanceMetrics | None = None  # over the overlap
    b: PerformanceMetrics | None = None
    days: tuple[DayResult, ...] = ()
    diff: tuple[tuple[str, Any, Any], ...] = ()
    notes: list[str] = field(default_factory=list)
    min_days: int = 14

    @property
    def a_wins(self) -> int:
        return sum(1 for d in self.days if d.better == "A")

    @property
    def b_wins(self) -> int:
        return sum(1 for d in self.days if d.better == "B")

    @property
    def compared(self) -> int:
        return self.a_wins + self.b_wins

    @property
    def p_a_better(self) -> float | None:
        return sign_test_p(self.a_wins, self.compared)

    @property
    def p_b_better(self) -> float | None:
        return sign_test_p(self.b_wins, self.compared)

    @property
    def verdict(self) -> str:
        if self.start is None:
            return "not comparable: the runs never traded at the same time"
        if self.compared < self.min_days:
            return (f"too early to tell: {self.compared} day(s) compared so far; "
                    f"wait for at least {self.min_days}")
        for name, wins, p in (("A", self.a_wins, self.p_a_better), ("B", self.b_wins, self.p_b_better)):
            if p is not None and p < 0.05:
                return f"{name} had the better day {wins} of {self.compared} times (sign test p = {p:.3f})"
        if self.a_wins == self.b_wins:
            return f"no difference: each had the better day {self.a_wins} of {self.compared} times"
        leader, wins, p = (("A", self.a_wins, self.p_a_better) if self.a_wins > self.b_wins
                           else ("B", self.b_wins, self.p_b_better))
        return (f"{leader} leads on {wins} of {self.compared} days, but that could easily be chance "
                f"(p = {p:.2f}); let both runs continue")

    def to_dict(self) -> dict[str, Any]:
        def run(r: dict[str, Any]) -> dict[str, Any]:
            return {k: r.get(k) for k in ("run_id", "kind", "status", "timeframe")}

        return {
            "run_a": run(self.run_a), "run_b": run(self.run_b),
            "start": None if self.start is None else self.start.isoformat(),
            "end": None if self.end is None else self.end.isoformat(),
            "a": None if self.a is None else self.a.to_dict(), "b": None if self.b is None else self.b.to_dict(),
            "compared": self.compared, "a_wins": self.a_wins, "b_wins": self.b_wins,
            "p_a_better": self.p_a_better, "p_b_better": self.p_b_better, "min_days": self.min_days,
            "verdict": self.verdict, "notes": list(self.notes),
            "diff": [{"key": k, "a": a, "b": b} for k, a, b in self.diff],
            "days": [{"day": d.day.isoformat(), "a": d.a, "b": d.b, "better": d.better} for d in self.days],
        }


@dataclass(frozen=True, slots=True)
class _Side:
    metrics: PerformanceMetrics
    daily: pd.Series  # daily return, indexed by UTC date


def _side(store: Any, run: dict[str, Any], curve: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> _Side:
    config = AppConfig.from_dict(run["config"])
    before = curve[curve.index < start]
    base_equity = float(before["equity"].iloc[-1]) if not before.empty else config.portfolio.initial_cash
    base_fees = float(before["fees_paid"].iloc[-1]) if not before.empty else 0.0
    part = curve[(curve.index >= start) & (curve.index <= end)]
    trades = [t for t in store.load_closed_trades(run["run_id"])
              if start <= pd.Timestamp(t.closed_at) <= end]
    metrics = compute_metrics([base_equity, *part["equity"].tolist()], trades, run["timeframe"],
                              total_fees=float(part["fees_paid"].iloc[-1]) - base_fees,
                              in_market=(part["open_positions"] > 0).tolist())
    equity = part["equity"]
    day_end = equity.groupby(pd.DatetimeIndex(equity.index).date).last()
    previous = day_end.shift(1)
    previous.iloc[0] = base_equity
    return _Side(metrics, day_end / previous - 1.0)


def live_compare(store: Any, run_a: str, run_b: str, *, min_days: int = 14) -> LiveComparison:
    if run_a == run_b:
        raise ValueError("compare two different runs")
    if isinstance(min_days, bool) or not isinstance(min_days, int) or min_days < 1:
        raise ValueError("min_days must be a positive integer")
    runs = []
    for run_id in (run_a, run_b):
        run = store.get_run(run_id)
        if run is None:
            raise ValueError(f"unknown run id {run_id!r}")
        runs.append(run)
    ra, rb = runs
    out = LiveComparison(ra, rb, min_days=min_days,
                         diff=tuple(config_diff(AppConfig.from_dict(ra["config"]), AppConfig.from_dict(rb["config"]))))
    for what in ("kind", "timeframe"):
        if ra[what] != rb[what]:
            out.notes.append(f"different {what}: {ra[what]} vs {rb[what]}")
    curves = [store.load_equity_curve(r["run_id"]) for r in runs]
    if any(c.empty for c in curves):
        out.notes.append("a run has no equity curve yet")
        return out
    start = max(c.index[0] for c in curves)
    end = min(c.index[-1] for c in curves)
    if start > end:
        return out
    a, b = (_side(store, r, c, start, end) for r, c in zip(runs, curves))
    out.start, out.end = start.to_pydatetime(), end.to_pydatetime()
    out.a, out.b = a.metrics, b.metrics
    both = a.daily.index.intersection(b.daily.index)
    out.days = tuple(DayResult(d, float(a.daily[d]), float(b.daily[d])) for d in sorted(both))
    return out


def format_live_compare(c: LiveComparison, *, last_days: int = 10) -> str:
    def head(label: str, r: dict[str, Any]) -> str:
        return f"{label} = {r['run_id']} ({r['kind']}, {r['status']}, {r['timeframe']})"

    lines = [head("A", c.run_a), head("B", c.run_b)]
    lines += [f"Note: {n}" for n in c.notes]
    if c.start is None or c.a is None or c.b is None:
        lines.append(f"Verdict: {c.verdict}.")
        return "\n".join(lines)
    span = (c.end - c.start).total_seconds() / 86400
    lines.append(f"Together: {c.start:%Y-%m-%d %H:%M} -> {c.end:%Y-%m-%d %H:%M} UTC ({span:.1f} days); "
                 "each run measured from its equity at the start of that period")
    if c.diff:
        lines.append("Settings that differ (A -> B):")
        lines += [f"  {k}: {a!r} -> {b!r}" for k, a, b in c.diff[:25]]
        if len(c.diff) > 25:
            lines.append(f"  ... and {len(c.diff) - 25} more")
    else:
        lines.append("The two configs are identical.")

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:+.2%}"

    def num(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    def share(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0%}"

    rows = [("Return", lambda m: pct(m.total_return)), ("Max drawdown", lambda m: f"{-m.max_drawdown:.2%}"),
            ("Sharpe", lambda m: num(m.sharpe_ratio)),
            ("Trades closed", lambda m: f"{m.num_trades}" + ("" if m.win_rate is None else f" ({m.win_rate:.0%} won)")),
            ("Fees", lambda m: f"{m.total_fees:,.2f}"), ("Exposure", lambda m: share(m.exposure))]
    lines.append(f"\n{'':<14} {'A':>14} {'B':>14}")
    lines += [f"{label:<14} {get(c.a):>14} {get(c.b):>14}" for label, get in rows]
    if c.days:
        lines.append(f"\n{'day':<11} {'A':>8} {'B':>8}  better")
        for d in c.days[-last_days:]:
            lines.append(f"{d.day:%Y-%m-%d}  {d.a:>+8.2%} {d.b:>+8.2%}  {d.better or '-':^6}")
        if len(c.days) > last_days:
            lines.append(f"(last {last_days} of {len(c.days)} days; --json lists them all)")
    ties = len(c.days) - c.compared
    lines.append(f"\nBetter day: A {c.a_wins}, B {c.b_wins}" + (f", {ties} tied" if ties else "")
                 + " (UTC days; days are not fully independent, so treat the p-value as a rough guide)")
    lines.append(f"Verdict: {c.verdict}.")
    return "\n".join(lines)
