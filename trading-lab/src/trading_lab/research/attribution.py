"""Agent (and strategy) performance attribution: does each voter add value?

Definitions. All of them are computed from the run's own records, so the
same run always gives the same numbers.

* **Vote:** a signal on a decision bar. Warm-up bars and bars skipped by
  ``decision_interval`` are not votes. A vote with no usable answer (model
  error, malformed answer, replay cache miss) is counted under ``errors``
  and not as BUY/SELL/HOLD.
* **Average confidence:** mean confidence of BUY and SELL votes (HOLD always
  has confidence 0).
* **Forward return (horizon N):** ``close[t + N] / close[t] − 1`` for the
  symbol, where ``t`` is the vote's bar and ``t + N`` is N bars later. Votes
  too close to the end of the data are not measurable.
* **Directional correctness:** share of measurable BUY/SELL votes where the
  price then moved the voted way (up after BUY, down after SELL). This uses
  raw prices: no fees, no position sizing.
* **Average outcome after BUY / SELL:** mean forward return after BUY votes
  and after SELL votes (for SELL, a negative number means the vote was right).
* **Calibration:** directional votes grouped by confidence bucket, each with
  its correctness and its mean *signed* forward return (positive = right).
* **Trade attribution:** every closed trade is linked to its entry signal,
  the last ensemble ENTER_SIGNAL for that symbol before the trade opened. At
  that bar the voter:
    - **agreed** (voted BUY for a long, SELL for a short), **disagreed** (the
      opposite) or abstained;
    - **influenced** the trade when it agreed or disagreed;
    - was **pivotal** when removing its vote would have turned the ensemble's
      entry vote (BUY, or SELL for a short) into no entry.
  ``pnl_agreed`` / ``pnl_disagreed`` sum those trades' realised PnL (fees
  included), also shown as a percentage of the initial cash.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from trading_lab.config import AppConfig, VotingConfig
from trading_lab.core.models import SHORT, ClosedTrade, Decision, DecisionAction, Signal
from trading_lab.ensemble.voting import ENSEMBLE_NAME

CONFIDENCE_BUCKETS: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-9)
_NOT_A_VOTE = {"warmup", "not a decision bar"}


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    low: float
    high: float
    votes: int
    measured: int
    correct: int
    mean_signed_return: float | None

    @property
    def hit_rate(self) -> float | None:
        return self.correct / self.measured if self.measured else None


@dataclass(frozen=True, slots=True)
class Attribution:
    strategy: str
    is_agent: bool
    horizon_bars: int
    votes: int = 0
    buy: int = 0
    sell: int = 0
    hold: int = 0
    errors: int = 0
    avg_confidence: float | None = None
    measured: int = 0
    correct: int = 0
    avg_return_after_buy: float | None = None
    avg_return_after_sell: float | None = None
    trades: int = 0
    trades_influenced: int = 0
    trades_agreed: int = 0
    trades_disagreed: int = 0
    trades_pivotal: int = 0
    pnl_agreed: float = 0.0
    pnl_disagreed: float = 0.0
    pnl_agreed_pct: float = 0.0  # of the initial cash
    pnl_disagreed_pct: float = 0.0
    avg_trade_return_agreed: float | None = None
    avg_trade_return_disagreed: float | None = None
    calibration: tuple[CalibrationBucket, ...] = field(default_factory=tuple)

    @property
    def directional_correctness(self) -> float | None:
        return self.correct / self.measured if self.measured else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["directional_correctness"] = self.directional_correctness
        data["calibration"] = [{**asdict(b), "hit_rate": b.hit_rate} for b in self.calibration]
        return data


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _counterfactual_buy(votes: Sequence[Mapping[str, Any]], without: str, cfg: VotingConfig) -> bool:
    """Would the ensemble still say BUY without ``without``'s vote?"""
    return _counterfactual(votes, without, cfg) == "buy"


def _counterfactual(votes: Sequence[Mapping[str, Any]], without: str, cfg: VotingConfig) -> str:
    """The ensemble's direction without ``without``'s vote (same rules as ``VotingEngine``)."""
    rest = [v for v in votes if v["strategy"] != without]
    total = sum(float(v["weight"]) for v in rest)
    if total <= 0:
        return "hold"
    buy = sum(float(v["weight"]) * float(v["confidence"]) for v in rest if v["direction"] == "buy") / total
    sell = sum(float(v["weight"]) * float(v["confidence"]) for v in rest if v["direction"] == "sell") / total
    n_buy = sum(v["direction"] == "buy" for v in rest)
    n_sell = sum(v["direction"] == "sell" for v in rest)
    if buy - sell >= cfg.buy_threshold and n_buy >= cfg.min_agreeing:
        return "buy"
    if sell - buy >= cfg.sell_threshold and n_sell >= cfg.min_agreeing:
        return "sell"
    return "hold"


def _entry_bars(decisions: Iterable[Decision], trades: Sequence[ClosedTrade]) -> list[datetime | None]:
    by_symbol: dict[str, list[datetime]] = {}
    for d in decisions:
        if d.action is DecisionAction.ENTER_SIGNAL:
            by_symbol.setdefault(d.symbol, []).append(d.timestamp)
    for times in by_symbol.values():
        times.sort()
    out: list[datetime | None] = []
    for trade in trades:
        times = by_symbol.get(trade.symbol, [])
        k = bisect.bisect_left(times, trade.opened_at)
        out.append(times[k - 1] if k > 0 else None)
    return out


def attribute(
    signals: Iterable[Signal],
    decisions: Iterable[Decision],
    trades: Sequence[ClosedTrade],
    closes: Mapping[str, pd.Series],
    *,
    voting: VotingConfig,
    initial_cash: float,
    horizon: int = 4,
    strategies: Sequence[str] | None = None,
) -> dict[str, Attribution]:
    """Attribution per strategy (agents and, if listed, deterministic strategies)."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    signals = list(signals)
    ensemble: dict[tuple[str, datetime], Signal] = {}
    per_strategy: dict[str, list[Signal]] = {}
    for s in signals:
        if s.strategy == ENSEMBLE_NAME:
            ensemble[(s.symbol, s.timestamp)] = s
        else:
            per_strategy.setdefault(s.strategy, []).append(s)
    names = list(strategies) if strategies is not None else sorted(per_strategy)

    # Forward returns per (symbol, bar).
    forward: dict[tuple[str, datetime], float] = {}
    for sym, series in closes.items():
        series = series.sort_index()
        values = series.to_numpy(dtype="float64")
        for k, ts in enumerate(series.index):
            if k + horizon < len(values) and values[k] > 0:
                forward[(sym, ts.to_pydatetime())] = float(values[k + horizon] / values[k] - 1.0)

    entry_bars = _entry_bars(decisions, trades)
    out: dict[str, Attribution] = {}
    for name in names:
        sigs = per_strategy.get(name, [])
        votes = [s for s in sigs if s.metadata.get("reason") not in _NOT_A_VOTE]
        answered = [s for s in votes if "error" not in s.metadata]
        directional = [s for s in answered if s.direction.value != "hold"]
        is_agent = any("agent" in (s.metadata.get("params") or {}) for s in sigs)

        buckets = []
        for low, high in zip(CONFIDENCE_BUCKETS, CONFIDENCE_BUCKETS[1:]):
            members = [s for s in directional if low <= s.confidence < high]
            signed = []
            for s in members:
                r = forward.get((s.symbol, s.timestamp))
                if r is not None:
                    signed.append(r if s.direction.value == "buy" else -r)
            buckets.append(CalibrationBucket(
                low, min(high, 1.0), len(members), len(signed), int(sum(r > 0 for r in signed)), _mean(signed),
            ))
        after_buy = [r for s in directional if s.direction.value == "buy"
                     if (r := forward.get((s.symbol, s.timestamp))) is not None]
        after_sell = [r for s in directional if s.direction.value == "sell"
                      if (r := forward.get((s.symbol, s.timestamp))) is not None]

        by_bar = {(s.symbol, s.timestamp): s for s in answered}
        agreed: list[ClosedTrade] = []
        disagreed: list[ClosedTrade] = []
        pivotal = 0
        for trade, bar in zip(trades, entry_bars):
            if bar is None:
                continue
            vote = by_bar.get((trade.symbol, bar))
            if vote is None or vote.direction.value == "hold":
                continue
            entry = "sell" if trade.side == SHORT else "buy"  # the vote that opened this trade
            (agreed if vote.direction.value == entry else disagreed).append(trade)
            ens = ensemble.get((trade.symbol, bar))
            if (vote.direction.value == entry and ens is not None
                    and _counterfactual(ens.metadata.get("votes", []), name, voting) != entry):
                pivotal += 1

        def pnl(ts: Sequence[ClosedTrade]) -> float:
            return float(sum(t.pnl for t in ts))

        out[name] = Attribution(
            strategy=name,
            is_agent=is_agent,
            horizon_bars=horizon,
            votes=len(votes),
            buy=sum(s.direction.value == "buy" for s in answered),
            sell=sum(s.direction.value == "sell" for s in answered),
            hold=sum(s.direction.value == "hold" for s in answered),
            errors=len(votes) - len(answered),
            avg_confidence=_mean([s.confidence for s in directional]),
            measured=sum(b.measured for b in buckets),
            correct=sum(b.correct for b in buckets),
            avg_return_after_buy=_mean(after_buy),
            avg_return_after_sell=_mean(after_sell),
            trades=len(trades),
            trades_influenced=len(agreed) + len(disagreed),
            trades_agreed=len(agreed),
            trades_disagreed=len(disagreed),
            trades_pivotal=pivotal,
            pnl_agreed=pnl(agreed),
            pnl_disagreed=pnl(disagreed),
            pnl_agreed_pct=pnl(agreed) / initial_cash,
            pnl_disagreed_pct=pnl(disagreed) / initial_cash,
            avg_trade_return_agreed=_mean([t.return_pct for t in agreed]),
            avg_trade_return_disagreed=_mean([t.return_pct for t in disagreed]),
            calibration=tuple(buckets),
        )
    return out


def attribute_result(result: Any, config: AppConfig, *, horizon: int = 4) -> dict[str, Attribution]:
    """Attribution for an in-memory ``BacktestResult``."""
    return attribute(
        result.signals, result.decisions, list(result.trades), result.closes(),
        voting=config.voting, initial_cash=config.portfolio.initial_cash, horizon=horizon,
    )


def attribute_run(store: Any, run_id: str, *, horizon: int = 4) -> dict[str, Attribution]:
    """Attribution for a stored run (forward returns need runs saved with schema v3 or later)."""
    from trading_lab.core.models import Direction

    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    config = AppConfig.from_dict(run["config"])
    rows = store.load_signals(run_id)
    signals = [
        Signal(r.strategy, r.symbol, Direction(r.direction), float(r.confidence),
               r.timestamp.to_pydatetime(), json.loads(r.metadata_json))
        for r in rows.itertuples(index=False)
    ]
    frame = store.load_decisions(run_id)
    decisions = [
        Decision(r.timestamp.to_pydatetime(), r.symbol, DecisionAction(r.action), r.reason)
        for r in frame.itertuples(index=False)
        if r.action == DecisionAction.ENTER_SIGNAL.value
    ]
    return attribute(
        signals, decisions, store.load_closed_trades(run_id), store.load_closes(run_id),
        voting=config.voting, initial_cash=config.portfolio.initial_cash, horizon=horizon,
    )


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.1%}"


def format_attribution(a: Attribution) -> str:
    lines = [
        f"Agent: {a.strategy}" if a.is_agent else f"Strategy: {a.strategy}",
        f"  Votes: {a.votes}   BUY: {a.buy}   SELL: {a.sell}   HOLD: {a.hold}"
        + (f"   errors: {a.errors}" if a.errors else ""),
        f"  Avg confidence (BUY/SELL): {'n/a' if a.avg_confidence is None else f'{a.avg_confidence:.2f}'}",
        f"  Directional correctness ({a.horizon_bars}-bar horizon): {_pct(a.directional_correctness)}"
        f" of {a.measured} measurable votes",
        f"  Avg outcome after BUY: {_pct(a.avg_return_after_buy, True)}   "
        f"after SELL: {_pct(a.avg_return_after_sell, True)}",
        f"  Trades influenced: {a.trades_influenced} of {a.trades} (agreed {a.trades_agreed}, "
        f"disagreed {a.trades_disagreed}, pivotal {a.trades_pivotal})",
        f"  PnL when agreed: {a.pnl_agreed_pct:+.2%} ({a.pnl_agreed:+,.2f} USDT)   "
        f"when disagreed: {a.pnl_disagreed_pct:+.2%} ({a.pnl_disagreed:+,.2f} USDT)",
    ]
    rows = [b for b in a.calibration if b.votes]
    if rows:
        lines.append("  Calibration:  confidence   votes  correct  mean signed return")
        for b in rows:
            lines.append(f"                {b.low:.1f}-{b.high:.1f}   {b.votes:>6}  {_pct(b.hit_rate):>7}  "
                         f"{_pct(b.mean_signed_return, True):>10}")
    return "\n".join(lines)
