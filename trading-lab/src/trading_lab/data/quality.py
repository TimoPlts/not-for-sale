"""Market-data quality report: what a strategy would be fed, checked before it matters.

``check_candles`` looks at one symbol's candles over a period and reports:

* **errors** (the data cannot be trusted as is): the source failed or
  returned invalid candles, returned nothing, or, when checking up to now,
  the latest closed candles are missing (stale data);
* **warnings** (the data can be used, but look first): missing candles
  (gaps, or none at the start or end of the period), candles with zero
  volume or no price range at all, extreme moves and opens far from the
  previous close.

An *extreme move* is a close-to-close change larger than both
``jump_floor`` and ``jump_sigmas`` robust standard deviations of the
symbol's own log returns in the period (median absolute deviation × 1.4826),
so the threshold adapts to the symbol and timeframe. Opens are compared with
the previous close only for consecutive candles, because a gap in the data
explains a jump.

Everything is read-only: the data source is only asked for candles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from trading_lab.core.errors import DataError
from trading_lab.data.base import MarketDataProvider, find_gaps, timeframe_delta, to_utc_timestamp

_FMT = "%Y-%m-%d %H:%M"


@dataclass(frozen=True, slots=True)
class QualityRules:
    jump_floor: float = 0.05  # never call a move under 5% extreme
    jump_sigmas: float = 10.0
    stale_grace_bars: int = 1  # the newest candle may take a moment to be published

    def __post_init__(self) -> None:
        if not (self.jump_floor > 0 and math.isfinite(self.jump_floor)):
            raise ValueError("jump_floor must be positive")
        if not (self.jump_sigmas > 0 and math.isfinite(self.jump_sigmas)):
            raise ValueError("jump_sigmas must be positive")
        if isinstance(self.stale_grace_bars, bool) or not isinstance(self.stale_grace_bars, int) \
                or self.stale_grace_bars < 0:
            raise ValueError("stale_grace_bars must be a non-negative integer")


@dataclass
class SymbolQuality:
    symbol: str
    timeframe: str
    start: pd.Timestamp  # first expected candle
    end: pd.Timestamp  # exclusive: open time after the last expected candle
    expected: int
    bars: int = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)  # inside the data
    missing_head: int = 0  # expected candles before the first one returned
    missing_tail: int = 0  # expected candles after the last one returned
    up_to_now: bool = False  # checked up to the last closed candle (the newest may lag a little)
    stale: bool = False  # checked up to now and the latest closed candles are missing
    zero_volume: list[pd.Timestamp] = field(default_factory=list)
    flat: list[pd.Timestamp] = field(default_factory=list)  # high == low
    jumps: list[tuple[pd.Timestamp, float]] = field(default_factory=list)  # close-to-close
    open_gaps: list[tuple[pd.Timestamp, float]] = field(default_factory=list)  # open vs previous close
    jump_threshold: float | None = None
    error: str | None = None

    @property
    def missing(self) -> int:
        return self.missing_head + self.missing_tail + sum(n for _, _, n in self.gaps)

    @property
    def errors(self) -> list[str]:
        out = []
        if self.error:
            out.append(self.error)
        elif self.bars == 0 and self.expected > 0:
            out.append("no candles returned")
        if self.stale:
            out.append(f"stale: the latest {self.missing_tail} closed candle(s) are missing "
                       f"(last {self.last:{_FMT}})" if self.last is not None else "stale: no recent candles")
        return out

    @property
    def warnings(self) -> list[str]:
        if self.error or self.bars == 0:
            return []
        out = []
        if self.gaps:
            out.append(f"{len(self.gaps)} gap(s), {sum(n for _, _, n in self.gaps)} candle(s) missing")
        if self.missing_head:
            out.append(f"{self.missing_head} candle(s) missing at the start")
        if self.missing_tail and not self.up_to_now:
            out.append(f"{self.missing_tail} candle(s) missing at the end")
        if self.zero_volume:
            out.append(f"{len(self.zero_volume)} zero-volume candle(s)")
        if self.flat:
            out.append(f"{len(self.flat)} candle(s) without any price range")
        if self.jumps:
            out.append(f"{len(self.jumps)} extreme move(s)")
        if self.open_gaps:
            out.append(f"{len(self.open_gaps)} open(s) far from the previous close")
        return out

    @property
    def ok(self) -> bool:
        return not self.errors and not self.warnings


def expected_range(
    timeframe: str, start: datetime, end: datetime | None, now: datetime | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First expected candle and the exclusive end, clipped to the last closed candle at ``now``."""
    step = pd.Timedelta(timeframe_delta(timeframe))
    first = to_utc_timestamp(start).ceil(step)
    stop = None if end is None else to_utc_timestamp(end)
    if now is not None:
        closed_end = to_utc_timestamp(now).floor(step)  # the candle opened then is still forming
        stop = closed_end if stop is None else min(stop, closed_end)
    if stop is None:
        raise ValueError("give an end or the current time")
    return first, max(first, stop)


def check_candles(
    symbol: str,
    candles: pd.DataFrame,
    timeframe: str,
    start: datetime,
    end: datetime | None,
    *,
    now: datetime | None = None,
    rules: QualityRules = QualityRules(),
) -> SymbolQuality:
    """Quality of ``candles`` (canonical OHLCV frame) for ``[start, end)``.

    With ``end=None`` the period runs up to the last closed candle at ``now``
    and missing recent candles make the data stale.
    """
    step = pd.Timedelta(timeframe_delta(timeframe))
    first, stop = expected_range(timeframe, start, end, now)
    out = SymbolQuality(symbol, timeframe, first, stop, expected=int((stop - first) / step), up_to_now=end is None)
    frame = candles.loc[(candles.index >= first) & (candles.index < stop)]
    out.bars = len(frame)
    if frame.empty:
        out.missing_tail = out.expected
        out.stale = end is None and out.expected > rules.stale_grace_bars
        return out
    out.first, out.last = frame.index[0], frame.index[-1]
    out.missing_head = int((out.first - first) / step)
    out.missing_tail = int((stop - step - out.last) / step)
    out.stale = end is None and out.missing_tail > rules.stale_grace_bars
    out.gaps = [(a, b, int((b - a) / step) + 1) for a, b in find_gaps(frame, timeframe)]
    out.zero_volume = list(frame.index[frame["volume"].to_numpy() == 0])
    out.flat = list(frame.index[(frame["high"] == frame["low"]).to_numpy()])

    close = frame["close"].to_numpy(dtype="float64")
    if len(close) >= 3:
        log_ret = np.diff(np.log(close))
        sigma = 1.4826 * float(np.median(np.abs(log_ret - np.median(log_ret))))
        threshold = max(rules.jump_floor, math.expm1(rules.jump_sigmas * sigma))
        out.jump_threshold = threshold
        change = close[1:] / close[:-1] - 1
        out.jumps = [(ts, float(c)) for ts, c in zip(frame.index[1:], change) if abs(c) > threshold]
        consecutive = (frame.index[1:] - frame.index[:-1]) == step
        gap = frame["open"].to_numpy(dtype="float64")[1:] / close[:-1] - 1
        out.open_gaps = [(ts, float(g)) for ts, g, c in zip(frame.index[1:], gap, consecutive)
                         if c and abs(g) > threshold]
    return out


def check_market_data(
    provider: MarketDataProvider,
    symbols: Iterable[str],
    timeframe: str,
    start: datetime,
    end: datetime | None = None,
    *,
    now: datetime | None = None,
    rules: QualityRules = QualityRules(),
) -> list[SymbolQuality]:
    """Fetch each symbol's candles from ``provider`` and check them (``end=None``: up to now)."""
    if end is None and now is None:
        raise ValueError("checking up to now needs the current time")
    out = []
    for symbol in symbols:
        try:
            candles = provider.fetch_ohlcv(symbol, timeframe, start, end)
        except (DataError, OSError) as exc:
            first, stop = expected_range(timeframe, start, end, now)
            step = pd.Timedelta(timeframe_delta(timeframe))
            out.append(SymbolQuality(symbol, timeframe, first, stop, int((stop - first) / step),
                                     error=f"{type(exc).__name__}: {exc}"))
            continue
        out.append(check_candles(symbol, candles, timeframe, start, end, now=now, rules=rules))
    return out


def check_stored_bars(store: object, run_id: str, rules: QualityRules = QualityRules()) -> list[SymbolQuality]:
    """Check the candles a run stored (its own period; staleness is not checked)."""
    from trading_lab.config import AppConfig

    run = store.get_run(run_id)  # type: ignore[attr-defined]
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    timeframe = AppConfig.from_dict(run["config"]).market.timeframe
    bars = store.load_bars(run_id)  # type: ignore[attr-defined]
    if bars.empty:
        return []
    step = timeframe_delta(timeframe)
    start = bars["timestamp"].min().to_pydatetime()
    end = bars["timestamp"].max().to_pydatetime() + step
    out = []
    for symbol, group in bars.groupby("symbol"):
        frame = group.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype("float64")
        out.append(check_candles(str(symbol), frame, timeframe, start, end, rules=rules))
    return out


def format_quality(reports: list[SymbolQuality], limit: int = 5) -> str:
    if not reports:
        return "No candles to check."
    r0 = reports[0]
    lines = [f"Data check: {r0.timeframe} candles, {r0.start:{_FMT}} -> {r0.end:{_FMT}} UTC "
             f"({r0.expected} expected per symbol)"]
    width = max(len(r.symbol) for r in reports)
    for r in reports:
        problems = r.errors + r.warnings
        status = "OK" if not problems else ("ERROR" if r.errors else "WARN")
        lines.append(f"{r.symbol:<{width}}  {r.bars:>6}/{r.expected:<6} {status:<5} " + " | ".join(problems))
        details: list[str] = []
        details += [f"gap: {a:{_FMT}} -> {b:{_FMT}} ({n} candle(s))" for a, b, n in r.gaps]
        details += [f"extreme move: {ts:{_FMT}} {c:+.1%} (threshold {r.jump_threshold:.1%})" for ts, c in r.jumps]
        details += [f"open vs previous close: {ts:{_FMT}} {g:+.1%}" for ts, g in r.open_gaps]
        details += [f"zero volume: {ts:{_FMT}}" for ts in r.zero_volume]
        details += [f"no price range: {ts:{_FMT}}" for ts in r.flat]
        for item in details[:limit]:
            lines.append(f"{'':<{width}}    {item}")
        if len(details) > limit:
            lines.append(f"{'':<{width}}    ... and {len(details) - limit} more")
    errors = sum(1 for r in reports if r.errors)
    warned = sum(1 for r in reports if r.warnings and not r.errors)
    if errors:
        lines.append(f"ERRORS: {errors} symbol(s) have unusable or stale data")
    elif warned:
        lines.append(f"WARNINGS: {warned} symbol(s) need a look")
    else:
        lines.append("OK: no problems found")
    return "\n".join(lines)
