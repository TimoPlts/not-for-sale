"""The desk funnel: every setup from first vote to closed trade, with a lead ID.

A trading desk passes each idea through stages: someone finds it, the vote
confirms it, risk clears it, it is executed, and it is closed. ``desk_funnel``
rebuilds that funnel for a stored run (backtest or paper) from its audit
trail, without changing anything:

1. **scanned**: every symbol-bar the strategies evaluated;
2. **lead**: a strategy or agent voted to enter (BUY, or SELL when shorts
   are allowed) while the symbol had no position. Each lead gets an ID
   (``L-0001``, in time order), so its whole thread can be followed;
3. **confirmed**: the vote passed (the ensemble signalled the entry);
4. **cleared**: it passed the entry filters at the close, then the circuit
   breakers and the risk manager at the next open (and, for limit entries,
   the order started working);
5. **executed**: the entry filled;
6. **closed**: the position was closed, with its PnL and exit type (see
   ``research.trades``). Executed leads that are still open are counted
   as open.

For confirmed leads that went no further it counts why: the entry filter,
the breaker or the risk limit that stopped them, or an order that expired.
Read-only.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from trading_lab.config import AppConfig
from trading_lab.core.models import DecisionAction
from trading_lab.ensemble.voting import ENSEMBLE_NAME
from trading_lab.research.trades import analyze_trades

STAGES = ("scanned", "leads", "confirmed", "cleared", "executed", "closed")
_SIGNAL_STAGE = {DecisionAction.HOLD.value, DecisionAction.IGNORED.value, DecisionAction.ENTER_SIGNAL.value,
                 DecisionAction.EXIT_SIGNAL.value}
_EXITS = {DecisionAction.EXIT.value, DecisionAction.STOP_LOSS.value, DecisionAction.TAKE_PROFIT.value,
          DecisionAction.LIQUIDATE.value}
_NOT_A_VOTE = ("warmup", "not a decision bar")


@dataclass
class Lead:
    lead_id: str
    timestamp: datetime
    symbol: str
    side: str  # long | short
    voters: list[str]
    stage: str = "leads"  # the furthest stage reached
    outcome: str = "not confirmed"
    order_id: str | None = None
    filled_at: datetime | None = None
    pnl: float | None = None
    exit_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"lead_id": self.lead_id, "timestamp": self.timestamp.isoformat(), "symbol": self.symbol,
                "side": self.side, "voters": list(self.voters), "stage": self.stage, "outcome": self.outcome,
                "order_id": self.order_id, "filled_at": None if self.filled_at is None else self.filled_at.isoformat(),
                "pnl": self.pnl, "exit_reason": self.exit_reason}


@dataclass
class Funnel:
    run_id: str
    start: datetime | None  # the window (None: the whole run)
    end: datetime | None
    scanned: int = 0
    leads: list[Lead] = field(default_factory=list)
    killed: Counter = field(default_factory=Counter)  # why confirmed leads went no further

    def count(self, stage: str) -> int:
        if stage == "scanned":
            return self.scanned
        order = STAGES.index(stage)
        return sum(1 for lead in self.leads if STAGES.index(lead.stage) >= order)

    @property
    def open(self) -> int:
        return sum(1 for lead in self.leads if lead.stage == "executed")

    @property
    def closed(self) -> list[Lead]:
        return [lead for lead in self.leads if lead.stage == "closed"]

    def to_dict(self) -> dict[str, Any]:
        closed = self.closed
        return {"run_id": self.run_id, "start": None if self.start is None else self.start.isoformat(),
                "end": None if self.end is None else self.end.isoformat(),
                "funnel": {stage: self.count(stage) for stage in STAGES}, "open": self.open,
                "wins": sum(1 for lead in closed if (lead.pnl or 0) > 0),
                "pnl": sum(lead.pnl or 0.0 for lead in closed), "killed": dict(self.killed.most_common()),
                "leads": [lead.to_dict() for lead in self.leads]}


def _clean(text: str) -> str:
    """A reason without its details (numbers, dates, parentheses), so the same cause counts together."""
    for cut in (" (", ";", " until ", " hit on "):
        text = text.split(cut)[0]
    words = []
    for word in text.split():
        if any(c.isdigit() for c in word):
            break
        words.append(word)
    return " ".join(words).rstrip(":,")


def _kill_reason(action: str, reason: str) -> str:
    """A short, countable reason a confirmed lead went no further."""
    if "blocked by " in reason:
        return "filter: " + _clean(reason.split("blocked by ", 1)[1].split(":")[0])
    if reason.startswith("circuit breaker"):
        return "circuit breaker: " + _clean(reason.split(":", 1)[1].strip())
    if "limit order already working" in reason:
        return "a limit order was already working"
    if "below minimum notional" in reason:
        return "risk: size too small"
    if action == DecisionAction.EXPIRED.value:
        return "expired: " + _clean(reason)
    return ("risk: " if action == DecisionAction.REJECTED.value else "") + _clean(reason)


def desk_funnel(store: Any, run_id: str, *, hours: float | None = None) -> Funnel:
    """The funnel of ``run_id``; with ``hours``, only leads in the last ``hours`` of the run."""
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    allow_short = AppConfig.from_dict(run["config"]).risk.allow_short
    decisions = store.load_decisions(run_id, include_holds=True)
    signals = store.load_signals(run_id)
    end = None if decisions.empty else decisions["timestamp"].max().to_pydatetime()
    start = None if hours is None or end is None else end - timedelta(hours=hours)
    out = Funnel(run_id, start, end)
    if decisions.empty:
        return out

    votes: dict[tuple[pd.Timestamp, str], dict[str, list[str]]] = {}
    for s in signals.itertuples(index=False):
        if s.strategy == ENSEMBLE_NAME or s.direction == "hold":
            continue
        meta = json.loads(s.metadata_json or "{}")
        if meta.get("reason") in _NOT_A_VOTE:
            continue
        votes.setdefault((s.timestamp, s.symbol), {}).setdefault(s.direction, []).append(s.strategy)

    trades = {(r.symbol, pd.Timestamp(r.opened_at)): r for r in analyze_trades(store, run_id, groupings=()).rows}
    positions: dict[str, str] = {}
    waiting: dict[str, Lead] = {}  # confirmed leads waiting for their next-open outcome, by symbol
    executed: dict[str, Lead] = {}  # filled leads by symbol, waiting for their exit
    counter = 0
    for d in decisions.itertuples(index=False):
        ts, sym, action, reason = d.timestamp, d.symbol, d.action, d.reason
        in_window = start is None or ts >= pd.Timestamp(start)
        lead = waiting.get(sym)
        if lead is not None and action in (DecisionAction.ENTER.value, DecisionAction.REJECTED.value,
                                           DecisionAction.EXPIRED.value, DecisionAction.ORDER_PLACED.value) \
                and ts > pd.Timestamp(lead.timestamp):
            if action == DecisionAction.ORDER_PLACED.value:
                lead.stage, lead.outcome = "cleared", "limit order working"
                continue
            del waiting[sym]
            if action == DecisionAction.ENTER.value:
                lead.stage, lead.outcome = "executed", "open"
                lead.order_id, lead.filled_at = str(d.order_id), ts.to_pydatetime()
                executed[sym] = lead
            else:
                lead.outcome = _kill_reason(action, reason)
                out.killed[lead.outcome] += 1
        if action == DecisionAction.ENTER.value:
            positions[sym] = "short" if d.signal_direction == "sell" else "long"
        elif action in _EXITS:
            positions.pop(sym, None)
            filled = executed.pop(sym, None)
            if filled is not None and filled.filled_at is not None:
                trade = trades.get((sym, pd.Timestamp(filled.filled_at)))
                filled.stage, filled.outcome = "closed", "closed"
                if trade is not None:
                    filled.pnl, filled.exit_reason = trade.pnl, trade.exit_reason
        if action not in _SIGNAL_STAGE:
            continue
        if in_window:
            out.scanned += 1
        if sym in positions:
            continue
        cast = votes.get((ts, sym), {})
        sides = [("long", cast.get("buy", []))] + ([("short", cast.get("sell", []))] if allow_short else [])
        confirmed_side = ("long" if d.signal_direction == "buy" else "short" if d.signal_direction == "sell"
                          and allow_short else None)
        side, voters = max(sides, key=lambda kv: (kv[0] == confirmed_side, len(kv[1])))
        if not voters and confirmed_side is None:
            continue
        counter += 1
        lead = Lead(f"L-{counter:04d}", ts.to_pydatetime(), sym, side, list(voters))
        if confirmed_side is not None:
            lead.stage = "confirmed"
            if action == DecisionAction.ENTER_SIGNAL.value:
                lead.stage, lead.outcome = "cleared", "scheduled for the next open"
                waiting[sym] = lead
            else:
                lead.outcome = _kill_reason(action, reason)
                if in_window:
                    out.killed[lead.outcome] += 1
        if in_window:
            out.leads.append(lead)
    return out


@dataclass(frozen=True, slots=True)
class Seat:
    """How one voter (strategy or agent) has done on the desk (see ``research.attribution``)."""

    strategy: str
    calls: int  # measurable BUY/SELL votes
    correct: float | None  # share of those the price then followed
    agreed: int  # closed trades it voted for
    pnl_agreed: float
    disagreed: int
    pnl_disagreed: float
    pivotal: int  # trades that would not have opened without it
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "calls": self.calls, "correct": self.correct, "agreed": self.agreed,
                "pnl_agreed": self.pnl_agreed, "disagreed": self.disagreed, "pnl_disagreed": self.pnl_disagreed,
                "pivotal": self.pivotal, "verdict": self.verdict}


MIN_CALLS = 30


def seat_verdict(calls: int, correct: float | None, agreed: int, pnl_agreed: float, pnl_disagreed: float) -> str:
    """Cautious on purpose: fewer than MIN_CALLS measurable calls is "too early"."""
    if calls < MIN_CALLS or correct is None:
        return "too early"
    if correct >= 0.52 and pnl_agreed >= pnl_disagreed:
        return "earning its seat"
    if correct < 0.48 or (agreed >= 5 and pnl_agreed < min(pnl_disagreed, 0.0)):
        return "not earning its seat"
    return "unclear"


def seat_review(store: Any, run_id: str, *, horizon: int = 4) -> list[Seat]:
    """Every voter of a run, best first (by PnL of the trades it voted for)."""
    from trading_lab.research.attribution import attribute_run

    seats = []
    for name, a in attribute_run(store, run_id, horizon=horizon).items():
        if not a.votes:
            continue
        seats.append(Seat(name, a.measured, a.directional_correctness, a.trades_agreed, a.pnl_agreed,
                          a.trades_disagreed, a.pnl_disagreed, a.trades_pivotal,
                          seat_verdict(a.measured, a.directional_correctness, a.trades_agreed, a.pnl_agreed,
                                       a.pnl_disagreed)))
    return sorted(seats, key=lambda s: (-s.pnl_agreed, s.strategy))


def format_seats(run_id: str, seats: list[Seat]) -> str:
    if not seats:
        return f"Seats of {run_id}: no votes yet."
    lines = [f"Seats of {run_id} (whole run; calls measured {MIN_CALLS}+ before any verdict)",
             f"  {'voter':<15} {'calls':>6} {'right':>6} {'for: trades':>11} {'pnl':>10} {'against':>8} {'pnl':>10} "
             f"{'pivotal':>7}  verdict"]
    for s in seats:
        right = "n/a" if s.correct is None else f"{s.correct:.0%}"
        lines.append(f"  {s.strategy:<15} {s.calls:>6} {right:>6} {s.agreed:>11} {s.pnl_agreed:>+10,.2f} "
                     f"{s.disagreed:>8} {s.pnl_disagreed:>+10,.2f} {s.pivotal:>7}  {s.verdict}")
    return "\n".join(lines)


def format_funnel(f: Funnel, *, recent: int = 10) -> str:
    span = ("the whole run" if f.start is None
            else f"{f.start:%Y-%m-%d %H:%M} -> {f.end:%Y-%m-%d %H:%M} UTC")
    lines = [f"Desk funnel for {f.run_id} ({span})"]
    labels = {"scanned": "symbol-bars evaluated", "leads": "a strategy voted to enter while flat",
              "confirmed": "the vote passed", "cleared": "passed filters, breakers and risk",
              "executed": "filled", "closed": "position closed"}
    previous = None
    for stage in STAGES:
        n = f.count(stage)
        rate = "" if previous in (None, 0) or stage == "leads" else f"  ({n / previous:.0%})"
        lines.append(f"  {stage:<10} {n:>7,}  {labels[stage]}{rate}")
        previous = n
    closed = f.closed
    if closed:
        wins = sum(1 for lead in closed if (lead.pnl or 0) > 0)
        lines.append(f"  closed: {wins} won, {len(closed) - wins} lost, PnL {sum(l.pnl or 0 for l in closed):+,.2f}; "
                     f"{f.open} still open")
    if f.killed:
        lines.append("Why confirmed setups went no further:")
        lines += [f"  {n:>5}  {reason}" for reason, n in f.killed.most_common(8)]
    shown = [lead for lead in f.leads if lead.stage != "leads"][-recent:]
    if shown:
        lines.append(f"Latest confirmed leads ({len(shown)}):")
        for lead in shown:
            result = (f"closed by {lead.exit_reason}, PnL {lead.pnl:+,.2f}" if lead.stage == "closed"
                      and lead.pnl is not None else lead.outcome)
            lines.append(f"  {lead.lead_id} {lead.timestamp:%Y-%m-%d %H:%M} {lead.symbol:<10} {lead.side:<5} "
                         f"votes: {', '.join(lead.voters) or '-'} -> {result}")
    return "\n".join(lines)
