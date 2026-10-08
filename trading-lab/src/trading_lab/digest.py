"""The week in paper trading: one message about every paper run (``trading-lab digest``).

For each paper run that is running, or that had bars in the period (e.g.
stopped during the week):

* the return over the period next to the market's (``summary.build_summary``);
* the worst drawdown in the period, the trades closed in it, the equity,
  the return since the start and the open positions;
* the watchdog check for running runs (``status.check_status``).

With several running runs it adds the ``live_compare`` verdict for every pair
(over all the time they ran together, not only this period).

Read-only. A weekly systemd timer can send it through the configured alert
channels (``deploy/systemd/trading-lab-digest.*``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from trading_lab.research.live_compare import LiveComparison, live_compare
from trading_lab.status import check_status
from trading_lab.summary import RunSummary, build_summary

MAX_PAIRS = 10


@dataclass
class Digest:
    start: datetime
    end: datetime
    runs: list[RunSummary] = field(default_factory=list)
    checks: dict[str, str | None] = field(default_factory=dict)  # "OK", the problems, or None (not running)
    comparisons: list[LiveComparison] = field(default_factory=list)
    skipped_pairs: int = 0
    seats: dict[str, list[Any]] = field(default_factory=dict)  # running run -> research.desk.Seat list

    @property
    def ok(self) -> bool:
        return all(check in (None, "OK") for check in self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(), "end": self.end.isoformat(), "ok": self.ok,
            "runs": [{
                "run_id": s.run_id, "status": s.status, "period_return": s.equity_change_pct,
                "market_return": s.market_change_pct, "max_drawdown": s.max_drawdown_window,
                "trades": len(s.trades), "wins": sum(1 for t in s.trades if t["pnl"] > 0),
                "realized_pnl": s.realized_pnl, "equity": s.equity, "total_return": s.total_return,
                "open_positions": s.open_positions[0]["count"] if s.open_positions else 0,
                "model_calls": s.usage.get("calls", 0) if s.usage else 0, "check": self.checks.get(s.run_id),
            } for s in self.runs],
            "comparisons": [{"run_a": c.run_a["run_id"], "run_b": c.run_b["run_id"], "compared": c.compared,
                             "a_wins": c.a_wins, "b_wins": c.b_wins, "verdict": c.verdict,
                             "a_return": None if c.a is None else c.a.total_return,
                             "b_return": None if c.b is None else c.b.total_return} for c in self.comparisons],
            "skipped_pairs": self.skipped_pairs,
            "seats": {run: [s.to_dict() for s in seats] for run, seats in self.seats.items()},
        }


def build_digest(store: Any, *, days: float = 7.0, now: datetime | None = None, max_behind: int = 2,
                 seats: bool = True) -> Digest:
    if days <= 0:
        raise ValueError("days must be positive")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out = Digest(now - timedelta(days=days), now)
    running = []
    for r in sorted(store.list_runs(10_000), key=lambda r: str(r["run_id"])):
        if r["kind"] != "paper":
            continue
        run_id = str(r["run_id"])
        summary = build_summary(store, run_id, hours=days * 24, now=now)
        if r["status"] != "running" and summary.window_end is None:
            continue  # stopped before the period
        out.runs.append(summary)
        if r["status"] == "running":
            running.append(run_id)
            status = check_status(store, run_id, now=now, max_behind=max_behind)
            out.checks[run_id] = "OK" if status.ok else "; ".join(status.problems)
            if seats:
                from trading_lab.research.desk import seat_review

                out.seats[run_id] = seat_review(store, run_id)
        else:
            out.checks[run_id] = None
    pairs = list(itertools.combinations(running, 2))
    out.comparisons = [live_compare(store, a, b) for a, b in pairs[:MAX_PAIRS]]
    out.skipped_pairs = max(0, len(pairs) - MAX_PAIRS)
    return out


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def format_digest(d: Digest) -> str:
    days = (d.end - d.start).total_seconds() / 86400
    lines = [f"Paper trading, {d.start:%Y-%m-%d %H:%M} -> {d.end:%Y-%m-%d %H:%M} UTC ({days:g} days)"]
    if not d.runs:
        lines.append("No paper run traded in this period.")
        return "\n".join(lines)
    for s in d.runs:
        wins = sum(1 for t in s.trades if t["pnl"] > 0)
        check = d.checks.get(s.run_id)
        lines.append(f"\n{s.run_id} ({s.status})")
        if s.equity is None:
            lines.append("  no bars yet")
        else:
            opened = s.open_positions[0]["count"] if s.open_positions else 0
            lines.append(f"  period {_pct(s.equity_change_pct)} (market {_pct(s.market_change_pct)}), "
                         f"worst drawdown {_pct(s.max_drawdown_window)}")
            lines.append(f"  {len(s.trades)} trade(s) closed, {wins} won, PnL {s.realized_pnl:+,.2f}; "
                         f"equity {s.equity:,.2f} ({_pct(s.total_return)} since the start), {opened} open")
        if s.usage and s.usage.get("calls"):
            lines.append(f"  model calls: {s.usage['calls']}")
        lines.append(f"  watchdog: {'not running' if check is None else check}")
    if d.comparisons:
        lines.append("\nComparisons (over all the time both ran):")
        for c in d.comparisons:
            returns = "" if c.a is None or c.b is None else f" [{_pct(c.a.total_return)} vs {_pct(c.b.total_return)}]"
            lines.append(f"  {c.run_a['run_id']} vs {c.run_b['run_id']}{returns}: {c.verdict}")
        if d.skipped_pairs:
            lines.append(f"  ... and {d.skipped_pairs} more pair(s); use trading-lab live-compare")
    if any(d.seats.values()):
        from trading_lab.research.desk import format_seats

        lines.append("\nWhich seats earned their place:")
        lines += [format_seats(run, seats) for run, seats in d.seats.items() if seats]
    lines.append("\nPaper trading only: simulated fills, no real orders.")
    return "\n".join(lines)
