"""Run summaries: what happened in a run over the last N hours (``trading-lab summary``).

Built only from reads of the SQLite history, so it works on a database that
a live paper run is writing, on a read-only copy, and for backtests. The
daily alert (``trading_lab.alerts``) sends the Markdown version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from trading_lab.config import AppConfig
from trading_lab.core.models import DecisionAction
from trading_lab.ensemble.voting import ENSEMBLE_NAME
from trading_lab.llm.usage import total_usage, usage_from_signals
from trading_lab.storage import SQLiteStore

_NOT_A_VOTE = ("warmup", "not a decision bar")


@dataclass
class RunSummary:
    run_id: str
    kind: str
    status: str
    window_start: datetime
    window_end: datetime | None  # last bar in the window
    initial_cash: float
    equity: float | None = None
    equity_change: float | None = None  # over the window
    equity_change_pct: float | None = None
    total_return: float | None = None  # since the run started
    drawdown: float | None = None  # now, from the running peak
    max_drawdown_window: float | None = None
    market_change_pct: float | None = None  # equal-weight average close-to-close move of the symbols
    trades: list[dict[str, Any]] = field(default_factory=list)  # closed in the window
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    actions: dict[str, int] = field(default_factory=dict)
    breaker_events: list[str] = field(default_factory=list)
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] | None = None

    @property
    def realized_pnl(self) -> float:
        return float(sum(t["pnl"] for t in self.trades))


def build_summary(store: SQLiteStore, run_id: str, *, hours: float = 24.0, now: datetime | None = None) -> RunSummary:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    config = AppConfig.from_dict(run["config"])
    curve = store.load_equity_curve(run_id)
    if now is None:
        now = curve.index[-1].to_pydatetime() if not curve.empty else datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    out = RunSummary(run_id, run["kind"], run["status"], start, None, config.portfolio.initial_cash)
    state = store.load_state(run_id) or {}
    out.health = state.get("health")

    if not curve.empty:
        window = curve[curve.index >= start]
        before = curve[curve.index < start]
        base = float(before["equity"].iloc[-1]) if not before.empty else config.portfolio.initial_cash
        equity = float(curve["equity"].iloc[-1])
        peak = max(config.portfolio.initial_cash, float(curve["equity"].max()))
        out.equity = equity
        out.equity_change = equity - base
        out.equity_change_pct = equity / base - 1.0 if base > 0 else None
        out.total_return = equity / config.portfolio.initial_cash - 1.0
        out.drawdown = equity / peak - 1.0
        if not window.empty:
            out.window_end = window.index[-1].to_pydatetime()
            running = curve["equity"].cummax().clip(lower=config.portfolio.initial_cash)
            out.max_drawdown_window = float((window["equity"] / running[window.index] - 1.0).min())
        last = curve.iloc[-1]
        out.open_positions = [{"count": int(last["open_positions"]), "value": float(last["positions_value"]),
                               "unrealized_pnl": float(last["unrealized_pnl"])}]

    bars = store.load_bars(run_id)
    if not bars.empty:
        moves = []
        for _, group in bars.groupby("symbol"):
            group = group.sort_values("timestamp")
            inside = group[group["timestamp"] >= start]
            prior = group[group["timestamp"] < start]
            if inside.empty:
                continue
            first = float(prior["close"].iloc[-1]) if not prior.empty else float(inside["open"].iloc[0])
            moves.append(float(inside["close"].iloc[-1]) / first - 1.0)
        out.market_change_pct = sum(moves) / len(moves) if moves else None

    out.trades = [
        {"symbol": t.symbol, "pnl": t.pnl, "return_pct": t.return_pct, "opened_at": t.opened_at,
         "closed_at": t.closed_at}
        for t in store.load_closed_trades(run_id) if t.closed_at >= start
    ]

    decisions = store.load_decisions(run_id, include_holds=False)
    decisions = decisions[decisions["timestamp"] >= start]
    counts = decisions["action"].value_counts().to_dict()
    out.actions = {str(k): int(v) for k, v in sorted(counts.items())}
    out.breaker_events = [str(r) for r in decisions.loc[
        decisions["action"] == DecisionAction.CIRCUIT_BREAKER.value, "reason"]]

    rows = store.load_signals(run_id, since=start)
    pairs = []
    for r in rows.itertuples(index=False):
        if r.strategy == ENSEMBLE_NAME:
            continue
        meta = json.loads(r.metadata_json)
        pairs.append((r.strategy, meta))
        if "agent" not in (meta.get("params") or {}) or meta.get("reason") in _NOT_A_VOTE:
            continue
        agent = out.agents.setdefault(r.strategy, {"buy": 0, "sell": 0, "hold": 0, "errors": 0, "latest": None})
        if "error" in meta:
            agent["errors"] += 1
        else:
            agent[r.direction] += 1
            label = meta.get("regime") or meta.get("momentum_state") or meta.get("risk_state")
            agent["latest"] = {"symbol": r.symbol, "timestamp": r.timestamp.to_pydatetime(),
                               "direction": r.direction, "confidence": float(r.confidence),
                               "label": label, "rationale": meta.get("rationale", "")}
    per_agent = usage_from_signals(pairs)
    if per_agent:
        out.usage = total_usage(per_agent).to_dict()
    return out


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def format_summary(s: RunSummary) -> str:
    """Markdown summary (also readable as plain text)."""
    end = s.window_end.strftime("%Y-%m-%d %H:%M") if s.window_end else "no bars"
    lines = [f"**trading-lab {s.kind} run {s.run_id}** ({s.status}) | {s.window_start:%Y-%m-%d %H:%M} -> {end} UTC"]
    if s.equity is None:
        lines.append("No equity recorded yet.")
        return "\n".join(lines)
    lines += [
        f"- Equity **{s.equity:,.2f}** ({s.equity_change:+,.2f}, {_pct(s.equity_change_pct)} in the window; "
        f"{_pct(s.total_return)} since start)",
        f"- Drawdown now {_pct(s.drawdown)}, worst in the window {_pct(s.max_drawdown_window)}; "
        f"market (equal-weight symbols) {_pct(s.market_change_pct)}",
    ]
    if s.trades:
        wins = sum(t["pnl"] > 0 for t in s.trades)
        best = max(s.trades, key=lambda t: t["pnl"])
        worst = min(s.trades, key=lambda t: t["pnl"])
        lines.append(f"- Closed trades: {len(s.trades)} ({wins} won), realized {s.realized_pnl:+,.2f}; best "
                     f"{best['symbol']} {best['pnl']:+,.2f}, worst {worst['symbol']} {worst['pnl']:+,.2f}")
    else:
        lines.append("- Closed trades: none")
    pos = s.open_positions[0] if s.open_positions else None
    if pos and pos["count"]:
        lines.append(f"- Open positions: {pos['count']} worth {pos['value']:,.2f} "
                     f"(unrealized {pos['unrealized_pnl']:+,.2f})")
    if s.actions:
        lines.append("- Decisions: " + ", ".join(f"{k} {v}" for k, v in s.actions.items()))
    for event in s.breaker_events:
        lines.append(f"- **Circuit breaker:** {event}")
    for name, a in sorted(s.agents.items()):
        line = f"- {name}: BUY {a['buy']} / SELL {a['sell']} / HOLD {a['hold']}"
        if a["errors"]:
            line += f" / no answer {a['errors']}"
        latest = a["latest"]
        if latest:
            label = f" {latest['label']}" if latest["label"] else ""
            line += (f"; latest {latest['symbol']} {latest['direction'].upper()} {latest['confidence']:.2f}{label}: "
                     f"{latest['rationale'][:160]}")
        lines.append(line)
    if s.usage:
        u = s.usage
        avg = "n/a" if u["avg_latency_seconds"] is None else f"{u['avg_latency_seconds']:.1f}s"
        lines.append(f"- Model: {u['calls']} calls, {u['cache_hits']} cache hits, {u['failures']} failures, "
                     f"avg latency {avg}")
    if s.health and s.health.get("consecutive_errors"):
        lines.append(f"- **Health:** {s.health['consecutive_errors']} failed cycle(s) in a row: "
                     f"{s.health.get('last_error')}")
    return "\n".join(lines)
