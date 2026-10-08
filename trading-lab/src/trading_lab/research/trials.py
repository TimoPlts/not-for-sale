"""How many configurations have you tried on this data? A log of research trials.

Trying many settings and keeping the best one is the most common way to
fool yourself: with enough tries, one looks good by luck alone. The deflated
Sharpe ratio (``metrics.sharpe``) corrects for that, but only if it knows
how many trials there were, including the ones in earlier sessions.

With ``[storage] record_trials = true``, every ``backtest``, ``sweep`` (each
combination), ``ab`` (each config in each window), ``checkup`` and
``permutation-test`` (the real-market run) appends one row per evaluated
config and period to the ``trials`` table: the command, the config
fingerprint, the timeframe, symbols and period, the return, the Sharpe ratio
and the number of bars. Walk-forward test windows are out of sample by
design and are not logged.

``trial_summary`` counts the trials whose period overlaps a given one on
the same timeframe: any overlap counts, which errs on the strict side. The
same config re-run on the same period is one trial, however often it was
logged. It also gives the Sharpe ratio the best of them would reach by luck
alone.
``deflated_against_log`` is the deflated Sharpe ratio of a new result
against those trials. Recording is off by default and never affects
results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from trading_lab.config import AppConfig
from trading_lab.metrics import PerformanceMetrics, periods_per_year
from trading_lab.metrics.sharpe import deflated_sharpe, expected_max_sharpe


def trial(command: str, config: AppConfig, start: datetime, end: datetime, metrics: PerformanceMetrics,
          label: str = "") -> dict[str, Any]:
    """One row for the log."""
    return {"command": command, "label": label, "config_fingerprint": config.fingerprint(),
            "timeframe": config.market.timeframe, "symbols": list(config.market.symbols),
            "period_start": start, "period_end": end, "total_return": metrics.total_return,
            "sharpe_ratio": metrics.sharpe_ratio, "num_bars": metrics.num_bars}


def record_trials(config: AppConfig, rows: Sequence[Mapping[str, Any]]) -> int:
    """Append ``rows`` to the log when ``storage.record_trials`` is on; returns how many were written."""
    if not config.storage.record_trials or not rows:
        return 0
    from trading_lab.storage import SQLiteStore

    with SQLiteStore(config.storage.db_path) as store:
        return store.add_trials(rows)


def per_bar_sharpe(row: Mapping[str, Any]) -> float | None:
    sharpe = row.get("sharpe_ratio")
    if sharpe is None or not math.isfinite(sharpe):
        return None
    return float(sharpe) / math.sqrt(periods_per_year(row["timeframe"]))


@dataclass
class TrialSummary:
    timeframe: str
    start: datetime
    end: datetime
    trials: list[dict[str, Any]] = field(default_factory=list)

    @property
    def unique(self) -> list[dict[str, Any]]:
        """One row per (config, period): re-running the same config on the same data is not a new trial."""
        seen: dict[tuple[str, datetime, datetime], dict[str, Any]] = {}
        for t in self.trials:
            seen.setdefault(trial_key(t), t)
        return list(seen.values())

    @property
    def count(self) -> int:
        return len(self.unique)

    @property
    def configs(self) -> int:
        return len({t["config_fingerprint"] for t in self.trials})

    @property
    def sharpes(self) -> list[float]:
        return [s for t in self.unique if (s := per_bar_sharpe(t)) is not None]

    @property
    def variance(self) -> float:
        values = self.sharpes
        return float(np.var(values, ddof=1)) if len(values) >= 2 else 0.0

    @property
    def luck_bar(self) -> float | None:
        """The annualised Sharpe ratio the best of these trials would reach by luck alone."""
        if self.count < 2:
            return None
        return expected_max_sharpe(self.count, self.variance) * math.sqrt(periods_per_year(self.timeframe))

    @property
    def best(self) -> dict[str, Any] | None:
        defined = [t for t in self.trials if per_bar_sharpe(t) is not None]
        return max(defined, key=lambda t: t["sharpe_ratio"]) if defined else None

    def by_command(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.trials:
            out[t["command"]] = out.get(t["command"], 0) + 1
        return dict(sorted(out.items()))

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        return {"timeframe": self.timeframe, "start": self.start.isoformat(), "end": self.end.isoformat(),
                "count": self.count, "configs": self.configs, "by_command": self.by_command(),
                "luck_bar_sharpe": self.luck_bar,
                "best": None if best is None else {k: (v.isoformat() if isinstance(v, datetime) else v)
                                                    for k, v in best.items()}}


def trial_key(row: Mapping[str, Any]) -> tuple[str, datetime, datetime]:
    return str(row["config_fingerprint"]), row["period_start"], row["period_end"]


def trial_summary(store: Any, timeframe: str, start: datetime, end: datetime) -> TrialSummary:
    return TrialSummary(timeframe, start, end, store.load_trials(timeframe=timeframe, start=start, end=end))


def deflated_against_log(summary: TrialSummary, returns: Sequence[float], *,
                         include: Mapping[str, Any] | None = None) -> float | None:
    """DSR of ``returns`` against the logged trials (plus ``include``, the new result, unless already logged)."""
    rows = summary.unique
    if include is not None and trial_key(include) not in {trial_key(r) for r in rows}:
        rows = [*rows, include]
    return deflated_sharpe(returns, [per_bar_sharpe(r) for r in rows])


def format_trial_summary(s: TrialSummary, *, recent: int = 10) -> str:
    repeats = len(s.trials) - s.count
    lines = [f"Trials on {s.timeframe} data overlapping {s.start:%Y-%m-%d} -> {s.end:%Y-%m-%d}: {s.count} "
             f"({s.configs} distinct configs" + (f"; {repeats} repeated run(s) not counted again" if repeats else "")
             + ")"]
    if not s.count:
        lines.append("None recorded. Set [storage] record_trials = true to log backtest, sweep, ab, checkup and "
                     "permutation-test results.")
        return "\n".join(lines)
    lines.append("  by command: " + ", ".join(f"{k} {v}" for k, v in s.by_command().items()))
    best = s.best
    if best is not None:
        label = f" {best['label']}" if best["label"] else ""
        lines.append(f"  best Sharpe {best['sharpe_ratio']:.2f}: {best['command']}{label}, "
                     f"{best['period_start']:%Y-%m-%d} -> {best['period_end']:%Y-%m-%d}, "
                     f"return {best['total_return']:+.2%}")
    if s.luck_bar is not None:
        lines.append(f"  the best of {s.count} trials would reach a Sharpe of about {s.luck_bar:.2f} by luck alone "
                     "(given how much their Sharpe ratios vary); a new result must clear that bar")
    lines.append(f"\n{'recorded':<17} {'command':<17} {'period':<23} {'return':>8} {'sharpe':>7}  config / label")
    for t in s.trials[-recent:]:
        sharpe = "n/a" if t["sharpe_ratio"] is None else f"{t['sharpe_ratio']:.2f}"
        lines.append(f"{t['created_at']:%Y-%m-%d %H:%M}  {t['command']:<17} "
                     f"{t['period_start']:%Y-%m-%d} -> {t['period_end']:%Y-%m-%d}  {t['total_return']:>+8.2%} "
                     f"{sharpe:>7}  {t['config_fingerprint'][:10]} {t['label']}".rstrip())
    if s.count > recent:
        lines.append(f"(last {recent} of {s.count}; --json lists them all)")
    return "\n".join(lines)
