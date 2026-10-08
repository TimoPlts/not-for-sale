"""Where does the money come from? A run's closed trades, grouped.

``analyze_trades`` reads a stored run (backtest or paper) and groups its
closed trades by:

* **exit**: why the position was closed:
  * stop-loss or trailing stop;
  * take-profit;
  * time stop;
  * kill switch;
  * signal (the vote turned);
  * end of backtest (still open when the backtest ended).

  It is read from the decision that filled the exit order, and for exits
  scheduled a bar earlier from the decision that scheduled them;
* **symbol** and **side** (long or short);
* **holding** time in bars (1-2, 3-6, 7-24, 25-72, more than 72);
* the **weekday** and, on request and below daily candles, the **hour** (UTC) of the entry.

Each group has its number of trades, wins, PnL, average return, average
holding time, best and worst trade and profit factor. A few plain-language
observations point at the biggest effects, such as a run that depends on a
handful of trades.

**Excursions** (from the run's stored candles, schema v3 and later):

* MAE, the maximum adverse excursion: how far the price went against the
  trade while it was open (0 or negative);
* MFE, the maximum favourable excursion: how far it went in the trade's
  favour (0 or positive).

Both are fractions of the entry price, measured over the lows and highs of
the candles from the entry bar up to (not including) the exit bar, plus the
exit price. Excursions inside the exit bar before a stop or target fill are
not seen. The entry price is the break-even one (entry fees included), as
in the PnL. The summary reports:

* how far 90% of the winning trades dipped at most, next to the configured
  stop-loss;
* how many losing trades were at least 1% in profit at some point;
* the e-ratio (average MFE over average |MAE|; above 1, trades move further
  for you than against you).

Read-only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Sequence

import pandas as pd

from trading_lab.config import AppConfig
from trading_lab.core.models import DecisionAction
from trading_lab.data.base import timeframe_delta

GROUPINGS = ("exit", "symbol", "side", "holding", "weekday", "hour")
DEFAULT_GROUPINGS = ("exit", "symbol", "side", "holding", "weekday")  # "hour" on request: 24 noisy rows
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
HOLDING = ((2, "1-2 bars"), (6, "3-6 bars"), (24, "7-24 bars"), (72, "25-72 bars"), (math.inf, "over 72 bars"))


@dataclass(frozen=True, slots=True)
class TradeRow:
    symbol: str
    side: str
    opened_at: datetime
    closed_at: datetime
    pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str
    entry_price: float = 0.0
    exit_price: float = 0.0
    mae: float | None = None  # maximum adverse excursion (<= 0), a fraction of the entry price
    mfe: float | None = None  # maximum favourable excursion (>= 0)


@dataclass(frozen=True, slots=True)
class GroupStats:
    name: str
    trades: int
    wins: int
    pnl: float
    avg_return: float
    avg_bars: float
    best: float
    worst: float
    gross_profit: float
    gross_loss: float
    avg_mae: float | None = None
    avg_mfe: float | None = None

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def profit_factor(self) -> float | None:
        if self.gross_loss > 0:
            return self.gross_profit / self.gross_loss
        return math.inf if self.gross_profit > 0 else None

    def to_dict(self) -> dict[str, Any]:
        pf = self.profit_factor
        return {"name": self.name, "trades": self.trades, "wins": self.wins, "win_rate": self.win_rate,
                "pnl": self.pnl, "avg_return": self.avg_return, "avg_bars": self.avg_bars, "best": self.best,
                "worst": self.worst, "profit_factor": "inf" if pf == math.inf else pf,
                "avg_mae": self.avg_mae, "avg_mfe": self.avg_mfe}


@dataclass(frozen=True, slots=True)
class Excursions:
    trades: int  # trades with excursions
    winners_mae_p90: float | None  # 90% of winners dipped at most this far (a negative fraction)
    losers_in_profit: int  # losing trades that were at least ``profit_threshold`` up at some point
    losers: int
    profit_threshold: float
    e_ratio: float | None  # mean MFE / mean |MAE|
    stop_loss_pct: float | None  # the configured fixed stop-loss, for comparison

    def to_dict(self) -> dict[str, Any]:
        return {"trades": self.trades, "winners_mae_p90": self.winners_mae_p90,
                "losers_in_profit": self.losers_in_profit, "losers": self.losers,
                "profit_threshold": self.profit_threshold, "e_ratio": self.e_ratio,
                "stop_loss_pct": self.stop_loss_pct}


@dataclass
class TradeAnalysis:
    run_id: str
    timeframe: str
    rows: list[TradeRow] = field(default_factory=list)
    groups: dict[str, list[GroupStats]] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    excursions: Excursions | None = None

    @property
    def total(self) -> GroupStats | None:
        return group_stats("all", self.rows) if self.rows else None

    def to_dict(self) -> dict[str, Any]:
        total = self.total
        return {"run_id": self.run_id, "timeframe": self.timeframe,
                "total": None if total is None else total.to_dict(),
                "groups": {k: [g.to_dict() for g in v] for k, v in self.groups.items()},
                "observations": list(self.observations),
                "excursions": None if self.excursions is None else self.excursions.to_dict(),
                "trades": [{"symbol": r.symbol, "side": r.side, "opened_at": r.opened_at.isoformat(),
                            "closed_at": r.closed_at.isoformat(), "pnl": r.pnl, "return_pct": r.return_pct,
                            "bars_held": r.bars_held, "exit_reason": r.exit_reason, "mae": r.mae, "mfe": r.mfe}
                           for r in self.rows]}


def group_stats(name: str, rows: Sequence[TradeRow]) -> GroupStats:
    pnls = [r.pnl for r in rows]
    known = [r for r in rows if r.mae is not None and r.mfe is not None]
    return GroupStats(
        name=name, trades=len(rows), wins=sum(1 for p in pnls if p > 0), pnl=float(sum(pnls)),
        avg_return=float(sum(r.return_pct for r in rows) / len(rows)),
        avg_bars=float(sum(r.bars_held for r in rows) / len(rows)), best=max(pnls), worst=min(pnls),
        gross_profit=float(sum(p for p in pnls if p > 0)), gross_loss=float(-sum(p for p in pnls if p < 0)),
        avg_mae=float(sum(r.mae for r in known) / len(known)) if known else None,  # type: ignore[misc]
        avg_mfe=float(sum(r.mfe for r in known) / len(known)) if known else None,  # type: ignore[misc]
    )


def excursion(side: str, entry: float, exit_price: float, lows: Sequence[float],
              highs: Sequence[float]) -> tuple[float, float]:
    """``(mae, mfe)`` of one trade as fractions of ``entry`` (see the module docstring)."""
    worst_low = min([*lows, exit_price])
    best_high = max([*highs, exit_price])
    if side == "short":
        return min(0.0, 1.0 - best_high / entry), max(0.0, 1.0 - worst_low / entry)
    return min(0.0, worst_low / entry - 1.0), max(0.0, best_high / entry - 1.0)


def excursion_summary(rows: Sequence[TradeRow], stop_loss_pct: float | None,
                      profit_threshold: float = 0.01) -> Excursions | None:
    known = [r for r in rows if r.mae is not None and r.mfe is not None]
    if not known:
        return None
    winners = sorted(r.mae for r in known if r.pnl > 0)  # type: ignore[type-var]
    losers = [r for r in known if r.pnl < 0]
    p90 = None
    if winners:  # the dip that 90% of winners stayed within (MAE is negative: take the 10th percentile)
        p90 = float(pd.Series(winners).quantile(0.10))
    mean_mae = sum(abs(r.mae) for r in known) / len(known)  # type: ignore[arg-type]
    mean_mfe = sum(r.mfe for r in known) / len(known)  # type: ignore[misc]
    return Excursions(
        trades=len(known), winners_mae_p90=p90,
        losers_in_profit=sum(1 for r in losers if r.mfe >= profit_threshold),  # type: ignore[operator]
        losers=len(losers), profit_threshold=profit_threshold,
        e_ratio=mean_mfe / mean_mae if mean_mae > 0 else None, stop_loss_pct=stop_loss_pct,
    )


def exit_reason(action: str, reason: str, scheduled_by: str | None) -> str:
    """A short exit category from the filling decision (and the one that scheduled it, for next-open exits)."""
    if action == DecisionAction.STOP_LOSS.value:
        return "trailing stop" if reason.startswith("trailing stop") else "stop-loss"
    if action == DecisionAction.TAKE_PROFIT.value:
        return "take-profit"
    if action == DecisionAction.LIQUIDATE.value:
        return "end of backtest"
    why = scheduled_by or reason
    if reason.startswith("time stop") or why.startswith("time stop"):
        return "time stop"
    if why.startswith("kill switch"):
        return "kill switch"
    return "signal"


def _exit_reasons(decisions: pd.DataFrame) -> dict[str, str]:
    """``order_id -> exit category`` for every filled exit order."""
    if decisions.empty:
        return {}
    out: dict[str, str] = {}
    scheduled: dict[str, str] = {}  # the latest exit_signal reason per symbol
    for d in decisions.itertuples(index=False):
        if d.action == DecisionAction.EXIT_SIGNAL.value:
            scheduled[d.symbol] = d.reason
        elif d.order_id and d.action in (DecisionAction.EXIT.value, DecisionAction.STOP_LOSS.value,
                                         DecisionAction.TAKE_PROFIT.value, DecisionAction.LIQUIDATE.value):
            out[str(d.order_id)] = exit_reason(d.action, d.reason,
                                               scheduled.pop(d.symbol, None) if d.action == "exit" else None)
    return out


def _bucket(grouping: str, row: TradeRow) -> str:
    if grouping == "exit":
        return row.exit_reason
    if grouping == "symbol":
        return row.symbol
    if grouping == "side":
        return row.side
    if grouping == "holding":
        return next(label for limit, label in HOLDING if row.bars_held <= limit)
    if grouping == "weekday":
        return WEEKDAYS[row.opened_at.weekday()]
    if grouping == "hour":
        return f"{row.opened_at.hour:02d}:00"
    raise ValueError(f"unknown grouping {grouping!r}; choose from {', '.join(GROUPINGS)}")


_ORDER: dict[str, Callable[[str], Any]] = {
    "holding": lambda name: [label for _, label in HOLDING].index(name),
    "weekday": lambda name: WEEKDAYS.index(name),
    "hour": lambda name: name,
}


def analyze_trades(store: Any, run_id: str, *, groupings: Sequence[str] = DEFAULT_GROUPINGS) -> TradeAnalysis:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    for g in groupings:
        if g not in GROUPINGS:
            raise ValueError(f"unknown grouping {g!r}; choose from {', '.join(GROUPINGS)}")
    step = timeframe_delta(run["timeframe"])
    reasons = _exit_reasons(store.load_decisions(run_id, include_holds=False))
    bars = store.load_bars(run_id)
    by_symbol = {str(s): g.set_index("timestamp") for s, g in bars.groupby("symbol")} if not bars.empty else {}
    out = TradeAnalysis(run_id, run["timeframe"])
    for t in store.load_closed_trades(run_id):
        held = max(1, int(round((pd.Timestamp(t.closed_at) - pd.Timestamp(t.opened_at)) / step)))
        mae = mfe = None
        candles = by_symbol.get(t.symbol)
        if candles is not None:
            inside = candles[(candles.index >= pd.Timestamp(t.opened_at)) & (candles.index < pd.Timestamp(t.closed_at))]
            mae, mfe = excursion(t.side, float(t.entry_price), float(t.exit_price), inside["low"].tolist(),
                                 inside["high"].tolist())
        out.rows.append(TradeRow(t.symbol, t.side, t.opened_at, t.closed_at, float(t.pnl), float(t.return_pct),
                                 held, reasons.get(t.exit_order_id, "signal"), float(t.entry_price),
                                 float(t.exit_price), mae, mfe))
    intraday = step < pd.Timedelta(days=1)
    for g in groupings:
        if g == "hour" and not intraday:
            continue
        buckets: dict[str, list[TradeRow]] = defaultdict(list)
        for row in out.rows:
            buckets[_bucket(g, row)].append(row)
        stats = [group_stats(name, rows) for name, rows in buckets.items()]
        order = _ORDER.get(g)
        out.groups[g] = sorted(stats, key=lambda s: order(s.name)) if order else sorted(stats, key=lambda s: s.pnl)
    risk = AppConfig.from_dict(run["config"]).risk
    stop = risk.stop_loss_pct if risk.stop_mode == "percent" else None  # ATR stops differ per trade
    out.excursions = excursion_summary(out.rows, stop)
    out.observations = observations(out)
    return out


def observations(a: TradeAnalysis) -> list[str]:
    rows = a.rows
    if len(rows) < 5:
        return ["too few closed trades to say much"] if rows else []
    notes = []
    total = sum(r.pnl for r in rows)
    best = sorted((r.pnl for r in rows), reverse=True)
    top = max(1, min(3, len(rows) // 10))
    if total > 0 and total - sum(best[:top]) <= 0:
        notes.append(f"without its best {top} trade(s) the run would have lost money: the result rests on a few "
                     "outliers")
    for grouping, what in (("exit", "exit type"), ("symbol", "symbol"), ("side", "side")):
        groups = a.groups.get(grouping, [])
        if len(groups) < 2:
            continue
        worst = min(groups, key=lambda g: g.pnl)
        if worst.pnl < 0 and worst.trades >= 3:
            notes.append(f"the costliest {what} is {worst.name}: {worst.trades} trades, PnL {worst.pnl:+,.2f}")
    holding = a.groups.get("holding", [])
    if len(holding) >= 2 and holding[-1].trades >= 3:
        shorter = holding[:-1]
        short_pnl = sum(g.pnl for g in shorter)
        if short_pnl < 0 < holding[-1].pnl and sum(g.trades for g in shorter) >= 3:
            notes.append(f"trades held {holding[-1].name} make money ({holding[-1].pnl:+,.2f}) while shorter ones "
                         f"lose ({short_pnl:+,.2f}); stops and exits close losers sooner, so this is partly by design")
    return notes


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def excursion_sentences(x: Excursions | dict[str, Any] | None) -> list[str]:
    """The excursion summary in plain sentences (from an ``Excursions`` or its ``to_dict()``)."""
    if x is None:
        return []
    d = x.to_dict() if isinstance(x, Excursions) else x
    out = []
    if d["winners_mae_p90"] is not None:
        stop = "" if d["stop_loss_pct"] is None else f"; the stop-loss is {d['stop_loss_pct']:.1%}"
        out.append(f"90% of winning trades went at most {abs(d['winners_mae_p90']):.2%} against the entry{stop}")
    if d["losers"]:
        out.append(f"{d['losers_in_profit']} of {d['losers']} losing trades were at least "
                   f"{d['profit_threshold']:.0%} in profit at some point")
    if d["e_ratio"] is not None:
        out.append(f"e-ratio {d['e_ratio']:.2f} (average mfe / average |mae|; above 1, trades move further "
                   "for you than against you)")
    return out


def _format_excursions(x: Excursions | None) -> list[str]:
    if x is None:
        return ["", "Excursions: n/a (no stored candles for this run)"]
    return ["", f"Excursions (mae: worst move against the entry while open, mfe: best move for it; "
                f"{x.trades} trades)"] + [f"  {s}" for s in excursion_sentences(x)]


def format_trades(a: TradeAnalysis) -> str:
    total = a.total
    if total is None:
        return f"Run {a.run_id}: no closed trades."
    lines = [f"Run {a.run_id}: {total.trades} closed trades, {total.win_rate:.0%} won, PnL {total.pnl:+,.2f}, "
             f"average {total.avg_bars:.1f} bars held ({a.timeframe} candles)"]
    head = (f"  {'':<16} {'trades':>6} {'won':>5} {'pnl':>11} {'avg ret':>8} {'avg bars':>8} {'best':>10} "
            f"{'worst':>10} {'pf':>5} {'mae':>7} {'mfe':>7}")
    titles = {"exit": "By exit", "symbol": "By symbol", "side": "By side", "holding": "By holding time",
              "weekday": "By entry weekday (UTC)", "hour": "By entry hour (UTC)"}
    for grouping, groups in a.groups.items():
        lines += ["", titles[grouping], head]
        for g in groups:
            pf = g.profit_factor
            pf_text = "n/a" if pf is None else "inf" if pf == math.inf else f"{pf:.2f}"
            lines.append(f"  {g.name:<16} {g.trades:>6} {g.win_rate:>5.0%} {g.pnl:>+11,.2f} {g.avg_return:>+8.2%} "
                         f"{g.avg_bars:>8.1f} {g.best:>+10,.2f} {g.worst:>+10,.2f} {pf_text:>5} "
                         f"{_pct(g.avg_mae):>7} {_pct(g.avg_mfe):>7}")
    lines += _format_excursions(a.excursions)
    if a.observations:
        lines += ["", "Observations:"] + [f"  - {o}" for o in a.observations]
    lines.append("\nGroups with few trades are noise; compare configs with ab or walkforward before acting on them.")
    return "\n".join(lines)
