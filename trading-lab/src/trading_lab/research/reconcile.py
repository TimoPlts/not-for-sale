"""Reconcile a live paper run with a backtest of the same bars.

A live paper run follows exactly the backtest's rules, so a backtest over the
bars the run processed, with the run's stored config, must give the same
fills and decisions. ``reconcile`` checks that:

* **Market data:** the candles stored by the run are compared with the
  candles the data source returns now. Exchanges occasionally revise
  candles, which would explain differences.
* **Fills and decisions:** a backtest over the run's bars (in memory,
  agents in ``replay`` mode, so the model is never called and the recorded
  answers are reused) is compared with what the run recorded. Fills the
  live run made at the open of the candle after its last processed bar are
  left out, because a backtest of those bars cannot have them yet.

Any difference points to a data revision, a missing cached answer (counted
as ``replay_misses``), or a bug. Nothing is written to the run database.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.models import DecisionAction
from trading_lab.data.base import MarketDataProvider, timeframe_delta
from trading_lab.llm import LLMProvider

_PRICE_COLUMNS = ("open", "high", "low", "close")


def _num(x: float) -> str:
    return f"{x:.10g}"


@dataclass
class Reconciliation:
    run_id: str
    start: datetime | None
    end: datetime | None
    bars_compared: int = 0
    bars_missing: int = 0  # stored bars the data source no longer returns
    bars_revised: int = 0
    max_price_diff: float = 0.0  # largest relative OHLC difference of a revised bar
    fills_matched: int = 0
    fills_live_only: list[str] = field(default_factory=list)
    fills_backtest_only: list[str] = field(default_factory=list)
    decisions_matched: int = 0
    decisions_live_only: list[str] = field(default_factory=list)
    decisions_backtest_only: list[str] = field(default_factory=list)
    replay_misses: int = 0
    note: str = ""

    @property
    def ok(self) -> bool:
        return not (self.fills_live_only or self.fills_backtest_only or self.decisions_live_only
                    or self.decisions_backtest_only or self.bars_revised or self.bars_missing)


def _fill_key(f: Any) -> str:
    side = f.side.value if hasattr(f.side, "value") else f.side
    return f"{f.timestamp:%Y-%m-%d %H:%M} {f.symbol} {side} qty={_num(f.quantity)} @ {_num(f.fill_price)}"


def _decision_key(ts: Any, symbol: str, action: str) -> str:
    return f"{pd.Timestamp(ts):%Y-%m-%d %H:%M} {symbol} {action}"


def _diff(live: Counter, other: Counter) -> tuple[int, list[str], list[str]]:
    matched = sum((live & other).values())
    return matched, sorted((live - other).elements()), sorted((other - live).elements())


def reconcile(
    store: Any, run_id: str, market: MarketDataProvider, *, llm_provider: LLMProvider | None = None
) -> Reconciliation:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    if run["kind"] != "paper":
        raise ValueError(f"{run_id} is a {run['kind']} run; only paper runs can be reconciled")
    config = AppConfig.from_dict(run["config"])
    bars = store.load_bars(run_id)
    if bars.empty:
        return Reconciliation(run_id, None, None, note="the run has no stored bars yet")
    step = timeframe_delta(config.market.timeframe)
    start = bars["timestamp"].min().to_pydatetime()
    end = bars["timestamp"].max().to_pydatetime() + step
    out = Reconciliation(run_id, start, end)

    for symbol, group in bars.groupby("symbol"):
        fresh = market.fetch_ohlcv(str(symbol), config.market.timeframe, start, end)
        stored = group.set_index("timestamp")
        out.bars_compared += len(stored)
        common = stored.index.intersection(fresh.index)
        out.bars_missing += len(stored) - len(common)
        if len(common):
            a = stored.loc[common, list(_PRICE_COLUMNS)].to_numpy(dtype="float64")
            b = fresh.loc[common, list(_PRICE_COLUMNS)].to_numpy(dtype="float64")
            rel = abs(a - b) / abs(a).clip(min=1e-12)
            revised = (rel > 1e-9).any(axis=1)
            out.bars_revised += int(revised.sum())
            if revised.any():
                out.max_price_diff = max(out.max_price_diff, float(rel.max()))

    # A live run never liquidates at the end of its data, so neither does its backtest.
    replay = config.with_overrides({"agents": {"mode": "replay"}, "backtest": {"liquidate_at_end": False}})
    result = BacktestEngine(replay, market, llm_provider=llm_provider).run(start, end)
    out.replay_misses = sum(1 for s in result.signals
                            if s.metadata.get("cache") == "miss" and "replay" in str(s.metadata.get("error", "")))

    live_fills = Counter(_fill_key(f) for f in store.load_fill_objects(run_id) if f.timestamp < end)
    bt_fills = Counter(_fill_key(f) for f in result.fills)
    out.fills_matched, out.fills_live_only, out.fills_backtest_only = _diff(live_fills, bt_fills)

    frame = store.load_decisions(run_id, include_holds=False)
    frame = frame[frame["timestamp"] < pd.Timestamp(end)]
    live_decisions = Counter(_decision_key(r.timestamp, r.symbol, r.action) for r in frame.itertuples())
    # Data-ended bookkeeping exists only in backtests; drop it from the comparison.
    bt_decisions = Counter(
        _decision_key(d.timestamp, d.symbol, d.action.value) for d in result.decisions
        if d.action is not DecisionAction.HOLD
        and not (d.action is DecisionAction.EXPIRED and "data ended" in d.reason)
    )
    out.decisions_matched, out.decisions_live_only, out.decisions_backtest_only = _diff(live_decisions, bt_decisions)
    return out


def format_reconciliation(r: Reconciliation, limit: int = 10) -> str:
    if r.start is None:
        return f"Run {r.run_id}: {r.note}"
    lines = [f"Run {r.run_id}: {r.start:%Y-%m-%d %H:%M} -> {r.end:%Y-%m-%d %H:%M} UTC"]
    lines.append(f"  market data: {r.bars_compared} stored bars, {r.bars_revised} revised"
                 + (f" (max {r.max_price_diff:.4%})" if r.bars_revised else "")
                 + (f", {r.bars_missing} no longer returned" if r.bars_missing else ""))
    lines.append(f"  fills: {r.fills_matched} match, {len(r.fills_live_only)} only live, "
                 f"{len(r.fills_backtest_only)} only in the backtest")
    lines.append(f"  decisions: {r.decisions_matched} match, {len(r.decisions_live_only)} only live, "
                 f"{len(r.decisions_backtest_only)} only in the backtest")
    if r.replay_misses:
        lines.append(f"  ! {r.replay_misses} agent answer(s) missing from the cache (replayed as HOLD)")
    for title, items in (("only live", r.fills_live_only), ("only backtest", r.fills_backtest_only),
                         ("decision only live", r.decisions_live_only),
                         ("decision only backtest", r.decisions_backtest_only)):
        for item in items[:limit]:
            lines.append(f"    {title}: {item}")
    lines.append("OK: the live run matches its backtest" if r.ok else
                 "DIFFERENT: see above (data revisions, missing cached answers or a bug)")
    return "\n".join(lines)
