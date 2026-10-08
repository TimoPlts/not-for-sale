"""How big should positions be? Sizing to a drawdown you can live with.

``size_for_drawdown`` finds the scale of ``risk.risk_per_trade_pct`` at
which the bad-case drawdown of the run's outlook (``research.outlook``)
equals a target, such as 20%.

Scaling is done trade by trade, from the sizing that each entry recorded:

* every entry stored all its size limits: risk per trade, max position
  size, max total exposure, available cash and, when configured, liquidity
  and the volatility target;
* at scale s, an entry's size becomes ``min(s x risk-per-trade size, the
  smallest other limit)``, so scaling up stops helping where another limit
  takes over;
* the trade's return scales with its size, and fees scale with it too.

Entries without stored sizing (older runs) are scaled linearly. The search
uses the same random draws for every scale, so results move smoothly with
s. ``binding`` counts which limit set each entry's size: if
``risk_per_trade`` rarely did, changing it changes little.

It is still an estimate: equity, cash and concurrent positions would differ
at another size. Confirm the suggestion with a backtest before relying on
it. Read-only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trading_lab.config import AppConfig
from trading_lab.core.models import DecisionAction
from trading_lab.research.outlook import Outlook, simulate, trade_returns

LIMITS = ("risk_per_trade", "max_position_size", "max_total_exposure", "available_cash", "liquidity",
          "volatility_target")
SCALES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass(frozen=True, slots=True)
class ScaleRow:
    scale: float
    risk_per_trade_pct: float
    outlook: Outlook


@dataclass
class SizingResult:
    run_id: str
    target: float  # bad-case drawdown to aim for
    current_risk_pct: float
    scale: float | None  # None: the target cannot be reached within the scale range
    rows: list[ScaleRow] = field(default_factory=list)  # the standard scales, plus the chosen one
    binding: dict[str, int] = field(default_factory=dict)  # entries per limit that set their size
    unmatched: int = 0  # trades without stored sizing (scaled linearly)
    max_effective_scale: float = 1.0  # the average size gain at the largest scale tried
    headroom: float | None = None  # median scale at which another limit takes over an entry's size
    warnings: list[str] = field(default_factory=list)

    @property
    def suggested_risk_pct(self) -> float | None:
        return None if self.scale is None else self.current_risk_pct * self.scale

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "target": self.target, "current_risk_pct": self.current_risk_pct,
                "scale": self.scale, "suggested_risk_pct": self.suggested_risk_pct, "binding": dict(self.binding),
                "unmatched": self.unmatched, "headroom": self.headroom, "warnings": list(self.warnings),
                "rows": [{"scale": r.scale, "risk_per_trade_pct": r.risk_per_trade_pct, **r.outlook.to_dict()}
                         for r in self.rows]}


def entry_sizing(store: Any, run_id: str) -> dict[tuple[str, pd.Timestamp], dict[str, Any]]:
    """``(symbol, entry time) -> the sizing details`` of every filled entry."""
    decisions = store.load_decisions(run_id, include_holds=False)
    out: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    for d in decisions[decisions["action"] == DecisionAction.ENTER.value].itertuples(index=False):
        details = json.loads(d.details_json or "{}")
        if "risk_per_trade" in details:
            out[(d.symbol, pd.Timestamp(d.timestamp))] = details
    return out


def headroom(sizing: dict[str, Any]) -> float:
    """The scale of risk per trade at which another limit would set this entry's size."""
    risk = float(sizing["risk_per_trade"])
    others = [float(sizing[k]) for k in LIMITS[1:] if isinstance(sizing.get(k), (int, float))]
    return min(others) / risk if others and risk > 0 else math.inf


def size_factor(sizing: dict[str, Any] | None, scale: float) -> float:
    """How much an entry's size changes when ``risk_per_trade_pct`` is multiplied by ``scale``."""
    if sizing is None:
        return scale
    risk = float(sizing["risk_per_trade"])
    others = [float(sizing[k]) for k in LIMITS[1:] if isinstance(sizing.get(k), (int, float))]
    other = min(others) if others else math.inf
    base = min(risk, other)
    if base <= 0:
        return 1.0
    return min(scale * risk, other) / base


def size_for_drawdown(store: Any, run_id: str, target: float = 0.20, *, horizon: int | None = None,
                      samples: int = 2000, seed: int = 7, scales: Sequence[float] = SCALES,
                      low: float = 0.05, high: float = 10.0) -> SizingResult | None:
    if not 0 < target < 1:
        raise ValueError("the target drawdown must be between 0 and 1, e.g. 0.2 for 20%")
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    config = AppConfig.from_dict(run["config"])
    returns = np.asarray(trade_returns(store, run_id), dtype="float64")
    if returns.size == 0:
        return None
    trades = sorted(store.load_closed_trades(run_id), key=lambda t: t.closed_at)
    sizing = entry_sizing(store, run_id)
    matched = [sizing.get((t.symbol, pd.Timestamp(t.opened_at))) for t in trades]
    out = SizingResult(run_id, target, config.risk.risk_per_trade_pct, None,
                       unmatched=sum(1 for m in matched if m is None))
    for details in sizing.values():
        limit = str(details.get("binding_limit", "unknown"))
        out.binding[limit] = out.binding.get(limit, 0) + 1

    def at(scale: float) -> Outlook:
        factors = np.array([size_factor(m, scale) for m in matched])
        result = simulate(run_id, returns * factors, horizon=horizon, samples=samples, seed=seed)
        assert result is not None
        return result

    def bad(scale: float) -> float:
        return at(scale).drawdown_bad

    if bad(low) > target:
        out.warnings.append(f"even at {low:g}x the bad-case drawdown is above {target:.0%}")
    elif bad(high) < target:
        out.warnings.append(f"even at {high:g}x the bad-case drawdown stays below {target:.0%}: other limits cap "
                            "the size, so raise max_position_pct or the other limits instead")
    else:
        lo, hi = low, high
        for _ in range(40):  # bisection on a monotone curve (same draws for every scale)
            mid = math.sqrt(lo * hi)
            if bad(mid) > target:
                hi = mid
            else:
                lo = mid
            if hi / lo < 1.001:
                break
        out.scale = lo
    out.max_effective_scale = float(np.mean([size_factor(m, high) for m in matched]))
    rooms = [headroom(m) for m in matched if m is not None]
    if rooms:
        out.headroom = float(np.median(rooms))
    chosen = [] if out.scale is None else [out.scale]
    for s in sorted({*scales, *chosen}):
        out.rows.append(ScaleRow(s, config.risk.risk_per_trade_pct * s, at(s)))
    if returns.sum() <= 0:
        out.warnings.append("the run lost money: a smaller size only loses more slowly")
    entries = sum(out.binding.values())
    if entries and out.binding.get("risk_per_trade", 0) < entries / 2:
        out.warnings.append(f"risk_per_trade set the size of only {out.binding.get('risk_per_trade', 0)} of "
                            f"{entries} entries; scaling it has a limited effect")
    if out.unmatched:
        out.warnings.append(f"{out.unmatched} trade(s) had no stored sizing and were scaled linearly")
    return out


def format_sizing(r: SizingResult | None, run_id: str = "") -> str:
    if r is None:
        return f"Run {run_id}: no closed trades, so nothing to size."
    lines = [f"Run {r.run_id}: sizing for a bad-case (1 in 20) drawdown of {r.target:.0%} "
             f"(risk.risk_per_trade_pct is {r.current_risk_pct:.2%})"]
    if r.binding:
        total = sum(r.binding.values())
        lines.append("  size set by: " + ", ".join(f"{k} {v / total:.0%}" for k, v in
                                                   sorted(r.binding.items(), key=lambda kv: -kv[1])))
    if r.headroom is not None and math.isfinite(r.headroom):
        lines.append(f"  beyond about {r.headroom:.2f}x, another limit (e.g. max_position_pct) sets most entries' "
                     "size, so larger scales change little")
    lines.append(f"\n{'scale':>6} {'risk/trade':>10} {'dd median':>10} {'dd bad':>8} {'return med':>11} "
                 f"{'P(loss)':>8} {'streak bad':>11}")
    for row in r.rows:
        o = row.outlook
        mark = "  <- target" if r.scale is not None and row.scale == r.scale else ""
        lines.append(f"{row.scale:>5.2f}x {row.risk_per_trade_pct:>10.2%} {o.drawdown_median:>10.1%} "
                     f"{o.drawdown_bad:>8.1%} {o.return_median:>+11.1%} {o.prob_loss:>8.0%} {o.streak_bad:>11}{mark}")
    if r.scale is not None:
        lines.append(f"\nSuggested: risk_per_trade_pct = {r.suggested_risk_pct:.4f} ({r.scale:.2f}x the current "
                     "setting). Confirm with a backtest before using it:")
        lines.append(f"  [risk]\n  risk_per_trade_pct = {r.suggested_risk_pct:.4f}")
    lines += [f"warning: {w}" for w in r.warnings]
    lines.append("An estimate from the run's own trades, scaled by each entry's recorded size limits: equity, cash "
                 "and overlapping positions would differ at another size.")
    return "\n".join(lines)
