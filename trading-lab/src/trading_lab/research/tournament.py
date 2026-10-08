"""Which config should get the next paper run? A tournament of candidates.

``tournament`` validates every candidate (``research.validate``: checkup, A/B
against the baseline, outlook and sizing) on the same period and data, and
ranks them:

1. by recommendation: strong candidate, then paper-trade it next to the
   baseline, then not ready;
2. then by the A/B record against the baseline (windows won);
3. then by the backtest's Sharpe ratio.

Trying several candidates and keeping the best one makes the best one look
better than it is. So each candidate also gets its **deflated Sharpe ratio
against the whole field**: the chance its true Sharpe beats what the luckiest
of all the candidates (and the baseline) would show by chance
(``metrics.sharpe``). A winner with a low deflated Sharpe is probably the
luckiest, not the best. Candidates may use different timeframes (the
``swing`` preset trades daily bars): every Sharpe ratio of the field is
converted to the candidate's bar length first (square-root-of-time).

The result names one candidate for a paper run next to the baseline, or
none when every candidate is "not ready". As always: one period, a paper
run before anything else, never real money.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.symbols import timeframe_to_seconds
from trading_lab.data.base import MarketDataProvider
from trading_lab.llm import LLMProvider
from trading_lab.metrics.sharpe import deflated_sharpe, per_bar_sharpe
from trading_lab.research.sweep import MemoizedProvider
from trading_lab.research.validate import NOT_READY, PAPER, STRONG, Validation, align, validate

RANK = {STRONG: 0, PAPER: 1, NOT_READY: 2}


def rescale_sharpe(sharpe: float | None, timeframe: str, to: str) -> float | None:
    """A per-bar Sharpe ratio on ``timeframe`` bars expressed per ``to`` bar (square-root-of-time rule).

    Candidates on different timeframes (an hourly and a daily one) have
    per-bar Sharpe ratios in different units; deflation needs them in one.
    """
    if sharpe is None or timeframe == to:
        return sharpe
    return sharpe * math.sqrt(timeframe_to_seconds(to) / timeframe_to_seconds(timeframe))


@dataclass
class Entry:
    name: str
    validation: Validation
    deflated: float | None = None

    @property
    def ab_record(self) -> tuple[int, int]:
        ab = self.validation.ab
        return (0, 0) if ab is None else (ab.b_wins, ab.compared)

    def sort_key(self) -> tuple[Any, ...]:
        wins, compared = self.ab_record
        sharpe = self.validation.checkup.metrics.sharpe_ratio
        return (RANK[self.validation.recommendation], -(wins / compared if compared else 0.0),
                -(sharpe if sharpe is not None else -1e9), self.name)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "deflated_sharpe": self.deflated, **self.validation.to_dict()}


@dataclass
class Tournament:
    baseline_label: str
    start: datetime
    end: datetime
    entries: list[Entry] = field(default_factory=list)  # ranked, best first

    @property
    def winner(self) -> Entry | None:
        best = self.entries[0] if self.entries else None
        return None if best is None or best.validation.recommendation == NOT_READY else best

    def to_dict(self) -> dict[str, Any]:
        return {"baseline": self.baseline_label, "start": self.start.isoformat(), "end": self.end.isoformat(),
                "winner": None if self.winner is None else self.winner.name,
                "entries": [e.to_dict() for e in self.entries]}


def tournament(candidates: Mapping[str, AppConfig], baseline: AppConfig, provider: MarketDataProvider,
               start: datetime, end: datetime, *, baseline_label: str = "baseline", windows: int = 6,
               permutations: int = 20, max_drawdown: float = 0.20, llm_provider: LLMProvider | None = None,
               trials: Callable[[AppConfig], Any] | None = None,
               progress: Callable[[str], None] | None = None) -> Tournament:
    if not candidates:
        raise ValueError("no candidates")
    say = progress or (lambda message: None)
    memo = provider if isinstance(provider, MemoizedProvider) else MemoizedProvider(provider)
    out = Tournament(baseline_label, start, end)
    for i, (name, config) in enumerate(candidates.items(), 1):
        say(f"[{i}/{len(candidates)}] {name}")
        result = validate(config, baseline, memo, start, end, label=name, baseline_label=baseline_label,
                          windows=windows, permutations=permutations, max_drawdown=max_drawdown,
                          llm_provider=llm_provider, trials=None if trials is None else trials(config))
        out.entries.append(Entry(name, result))
    # deflate every candidate against the whole field (the baseline, aligned to the first candidate, included)
    first = next(iter(candidates.values()))
    aligned, _ = align(baseline, first)
    base_run = BacktestEngine(aligned, memo, llm_provider=llm_provider).run(start, end)
    equity = [aligned.portfolio.initial_cash, *base_run.equity_curve["equity"].tolist()]
    field = [(per_bar_sharpe([b / a - 1.0 for a, b in zip(equity, equity[1:])]), aligned.market.timeframe)]
    field += [(per_bar_sharpe(e.validation.returns), candidates[e.name].market.timeframe) for e in out.entries]
    for entry in out.entries:
        if entry.validation.returns:
            timeframe = candidates[entry.name].market.timeframe
            entry.deflated = deflated_sharpe(entry.validation.returns,
                                             [rescale_sharpe(s, tf, timeframe) for s, tf in field])
    out.entries.sort(key=Entry.sort_key)
    return out


def format_tournament(t: Tournament) -> str:
    lines = [f"Tournament: {len(t.entries)} candidate(s) against {t.baseline_label} "
             f"({t.start:%Y-%m-%d} -> {t.end:%Y-%m-%d})",
             f"{'#':>2}  {'candidate':<18} {'recommendation':<36} {'checkup':<7} {'vs baseline':>11} "
             f"{'return':>8} {'max dd':>7} {'sharpe':>6} {'deflated':>8}"]
    for rank, e in enumerate(t.entries, 1):
        v, m = e.validation, e.validation.checkup.metrics
        wins, compared = e.ab_record
        record = "same config" if v.ab is None else f"{wins}/{compared} won"
        sharpe = "n/a" if m.sharpe_ratio is None else f"{m.sharpe_ratio:.2f}"
        deflated = "n/a" if e.deflated is None else f"{e.deflated:.0%}"
        lines.append(f"{rank:>2}  {e.name:<18} {v.recommendation:<36} {v.checkup.overall.upper():<7} {record:>11} "
                     f"{m.total_return:>+8.2%} {-m.max_drawdown:>7.1%} {sharpe:>6} {deflated:>8}")
    winner = t.winner
    if winner is None:
        lines.append("\nNo candidate is ready: keep the baseline, and change the strategy rather than its "
                     "parameters (see each candidate's checkup with trading-lab validate).")
    else:
        lines.append(f"\nNext paper run: {winner.name} ({winner.validation.recommendation}). "
                     f"Set it up with: trading-lab paper-plan {winner.name}")
        if winner.deflated is not None and winner.deflated < 0.5 and len(t.entries) > 1:
            lines.append(f"Caution: its deflated Sharpe against the field is {winner.deflated:.0%}, so it may simply "
                         f"be the luckiest of {len(t.entries)} candidates. Let the paper run decide.")
        for reason in winner.validation.reasons:
            lines.append(f"  - {reason}")
    lines.append("One period. The winner earns a paper run next to the baseline (live-compare), never real money.")
    return "\n".join(lines)


def tournament_html(t: Tournament) -> str:
    from trading_lab.html_report import _CSS, _e, _table, _td

    rows = []
    for rank, e in enumerate(t.entries, 1):
        v, m = e.validation, e.validation.checkup.metrics
        wins, compared = e.ab_record
        rows.append([_td(rank), _td(e.name), _td(v.recommendation), _td(v.checkup.overall.upper()),
                     _td("same config" if v.ab is None else f"{wins}/{compared}"), _td(f"{m.total_return:+.2%}"),
                     _td(f"{-m.max_drawdown:.1%}"),
                     _td("n/a" if m.sharpe_ratio is None else f"{m.sharpe_ratio:.2f}"),
                     _td("n/a" if e.deflated is None else f"{e.deflated:.0%}")])
    summary = "".join(f"<p>{_e(line)}</p>" for line in format_tournament(t).splitlines()[len(t.entries) + 2:] if line)
    return "\n".join([
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Tournament</title><style>{_CSS}</style></head>", '<body class="viz-root"><main>',
        f"<h1>Tournament against {_e(t.baseline_label)}</h1>",
        f'<div class="secondary">{t.start:%Y-%m-%d} → {t.end:%Y-%m-%d}</div>',
        '<h2>Ranking</h2><div class="card">' + _table(
            ["#", "Candidate", "Recommendation", "Checkup", "Vs baseline", "Return", "Max dd", "Sharpe",
             "Deflated"], rows) + "</div>",
        f'<h2>Next step</h2><div class="card">{summary}</div>',
        "</main></body></html>",
    ])
