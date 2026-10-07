"""Where does a strategy make or lose money? Performance by market regime.

Every bar of a stored run gets a market regime, read from an equal-weight
index of the run's symbols (the average close-to-close return of the symbols
each bar, compounded):

* **trend**:
  * ``up`` when the index is above its ``trend_bars`` simple moving average
    and that average is higher than ``slope_bars`` bars ago;
  * ``down`` when it is below a falling average;
  * ``sideways`` otherwise.
  Only bars up to the current one are used. The first ``trend_bars`` bars
  of the run are ``warm-up``, because stored runs keep only their own
  period's candles.
* **volatility**: ``volatile`` when the rolling standard deviation of index
  returns over ``vol_bars`` bars is above its median over the run, ``calm``
  otherwise. This is a description after the fact, not a trading signal, so
  the whole-run median is fine here.

For each regime this reports the bars and the share of time, the strategy's
compounded return in those bars (from the equity curve) next to the
market's, the time in the market, and the trades opened in that regime (count,
wins, PnL). Read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TRENDS = ("up", "sideways", "down")
VOLS = ("calm", "volatile")
WARMUP = "warm-up"


@dataclass(frozen=True, slots=True)
class RegimeStats:
    name: str
    bars: int
    time_share: float
    strategy_return: float
    market_return: float
    in_market: float  # share of these bars with an open position
    trades: int = 0
    wins: int = 0
    pnl: float = 0.0

    @property
    def edge(self) -> float:
        return self.strategy_return - self.market_return


@dataclass
class RegimeReport:
    run_id: str
    trend_bars: int
    vol_bars: int
    total_bars: int
    warmup_bars: int
    warmup_trades: int = 0  # trades opened during the warm-up (in no regime)
    by_trend: list[RegimeStats] = field(default_factory=list)
    by_vol: list[RegimeStats] = field(default_factory=list)
    combined: list[RegimeStats] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        def rows(items: list[RegimeStats]) -> list[dict[str, Any]]:
            return [{"name": s.name, "bars": s.bars, "time_share": s.time_share,
                     "strategy_return": s.strategy_return, "market_return": s.market_return,
                     "in_market": s.in_market, "trades": s.trades, "wins": s.wins, "pnl": s.pnl} for s in items]

        return {"run_id": self.run_id, "trend_bars": self.trend_bars, "vol_bars": self.vol_bars,
                "total_bars": self.total_bars, "warmup_bars": self.warmup_bars, "warmup_trades": self.warmup_trades,
                "by_trend": rows(self.by_trend),
                "by_volatility": rows(self.by_vol), "combined": rows(self.combined), "note": self.note}


def classify(index: pd.Series, *, trend_bars: int = 50, slope_bars: int = 10, vol_bars: int = 24) -> pd.DataFrame:
    """Per-bar ``trend`` and ``volatility`` labels for an index level series."""
    if trend_bars < 2 or slope_bars < 1 or vol_bars < 2:
        raise ValueError("trend_bars and vol_bars must be >= 2 and slope_bars >= 1")
    sma = index.rolling(trend_bars, min_periods=trend_bars).mean()
    rising, falling = sma > sma.shift(slope_bars), sma < sma.shift(slope_bars)
    trend = np.where(sma.isna(), WARMUP,
                     np.where((index > sma) & rising, "up", np.where((index < sma) & falling, "down", "sideways")))
    returns = np.log(index).diff()
    vol = returns.rolling(vol_bars, min_periods=vol_bars).std(ddof=0)
    median = vol.median()
    volatility = np.where(vol.isna(), WARMUP, np.where(vol > median, "volatile", "calm"))
    return pd.DataFrame({"trend": trend, "volatility": volatility}, index=index.index)


def market_index(closes: dict[str, pd.Series]) -> pd.Series:
    """Equal-weight index: the mean per-bar return of the symbols, compounded from 1.0."""
    frame = pd.DataFrame(closes).sort_index()
    returns = frame.pct_change().mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + returns).cumprod()


def _stats(name: str, mask: pd.Series, strat: pd.Series, market: pd.Series, in_market: pd.Series,
           trades: pd.DataFrame, total: int) -> RegimeStats:
    picked = trades[trades["entry_bar"].map(mask).fillna(False).astype(bool).to_numpy()]
    n = int(mask.sum())
    return RegimeStats(
        name=name, bars=n, time_share=n / total if total else 0.0,
        strategy_return=float((1 + strat[mask]).prod() - 1), market_return=float((1 + market[mask]).prod() - 1),
        in_market=float(in_market[mask].mean()) if n else 0.0,
        trades=len(picked), wins=int((picked["pnl"] > 0).sum()), pnl=float(picked["pnl"].sum()),
    )


def regimes_for_run(store: Any, run_id: str, *, trend_bars: int = 50, slope_bars: int = 10,
                    vol_bars: int = 24) -> RegimeReport:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    report = RegimeReport(run_id, trend_bars, vol_bars, 0, 0)
    curve = store.load_equity_curve(run_id)
    closes = store.load_closes(run_id)
    if curve.empty or not closes:
        report.note = "the run has no stored equity curve or candles"
        return report
    index = market_index(closes).reindex(curve.index).ffill()
    labels = classify(index, trend_bars=trend_bars, slope_bars=slope_bars, vol_bars=vol_bars)
    initial = float(run["config"].get("portfolio", {}).get("initial_cash", curve["equity"].iloc[0]))
    equity = curve["equity"]
    strat = equity / equity.shift(1).fillna(initial) - 1.0
    market = index / index.shift(1) - 1.0
    market.iloc[0] = 0.0
    in_market = (curve["open_positions"] > 0).astype(float)

    trades = pd.DataFrame([{"opened_at": t.opened_at, "pnl": t.pnl} for t in store.load_closed_trades(run_id)],
                          columns=["opened_at", "pnl"])
    if not trades.empty:  # the bar a trade was opened in (its fill is at that bar's open)
        opened = pd.to_datetime(trades["opened_at"], utc=True)
        positions = curve.index.searchsorted(opened, side="right") - 1
        trades["entry_bar"] = [curve.index[max(p, 0)] for p in positions]
    else:
        trades["entry_bar"] = pd.Series(dtype="datetime64[ns, UTC]")

    known = labels["trend"].ne(WARMUP) & labels["volatility"].ne(WARMUP)
    report.total_bars = len(curve)
    report.warmup_bars = int((~known).sum())
    report.warmup_trades = int((~trades["entry_bar"].map(known).fillna(False).astype(bool)).sum())
    total = int(known.sum())
    if total == 0:
        report.note = f"the run is shorter than the {trend_bars}-bar warm-up of the trend average"
        return report
    for trend in TRENDS:
        report.by_trend.append(_stats(trend, known & labels["trend"].eq(trend), strat, market, in_market,
                                      trades, total))
    for vol in VOLS:
        report.by_vol.append(_stats(vol, known & labels["volatility"].eq(vol), strat, market, in_market,
                                    trades, total))
    for trend in TRENDS:
        for vol in VOLS:
            mask = known & labels["trend"].eq(trend) & labels["volatility"].eq(vol)
            if mask.any():
                report.combined.append(_stats(f"{trend} / {vol}", mask, strat, market, in_market, trades, total))
    return report


def format_regimes(r: RegimeReport) -> str:
    if r.note and not r.by_trend:
        return f"Run {r.run_id}: {r.note}"
    head = (f"{'regime':<20} {'bars':>5} {'time':>5} {'strategy':>9} {'market':>8} {'in mkt':>6} "
            f"{'trades':>6} {'won':>4} {'pnl':>10}")
    lines = [f"Run {r.run_id}: {r.total_bars} bars, {r.warmup_bars} warm-up "
             f"(trend: {r.trend_bars}-bar average; volatility: {r.vol_bars}-bar returns)", head]

    def row(s: RegimeStats) -> str:
        return (f"{s.name:<20} {s.bars:>5} {s.time_share:>5.0%} {s.strategy_return:>+9.2%} "
                f"{s.market_return:>+8.2%} {s.in_market:>6.0%} {s.trades:>6} {s.wins:>4} {s.pnl:>+10,.2f}")

    for group in (r.by_trend, r.by_vol, r.combined):
        lines += [row(s) for s in group if s.bars] + [""]
    ranked = [s for s in r.combined if s.bars >= max(10, r.vol_bars)]
    if len(ranked) >= 2:
        best = max(ranked, key=lambda s: s.strategy_return)
        worst = min(ranked, key=lambda s: s.strategy_return)
        lines.append(f"Best: {best.name} ({best.strategy_return:+.2%} vs market {best.market_return:+.2%}); "
                     f"worst: {worst.name} ({worst.strategy_return:+.2%} vs market {worst.market_return:+.2%}).")
    lines.append("Returns compound only the bars of each regime; trades count in the regime they were opened in"
                 + (f" ({r.warmup_trades} opened during the warm-up are in no regime)." if r.warmup_trades else "."))
    return "\n".join(lines).rstrip()
