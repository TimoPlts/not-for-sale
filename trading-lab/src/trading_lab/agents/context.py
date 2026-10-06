"""The market context an agent receives.

The context is a plain, JSON-serialisable snapshot of what was knowable at a
bar's close: recent candles plus a few standard indicators. Values are
rounded to 6 significant digits, so the same market situation always produces
the same context, and therefore the same cache key, even when floats differ
in the last bits (e.g. live windows versus full backtest history).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.indicators import bollinger_bands, macd, rsi

SIGNIFICANT_DIGITS = 6


def _round(value: Any, digits: int = SIGNIFICANT_DIGITS) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return float(f"{value:.{digits}g}")


def indicator_frame(candles: pd.DataFrame) -> pd.DataFrame:
    """Causal per-bar indicators used to describe the market to an agent."""
    close = candles["close"]
    frame = pd.DataFrame(index=candles.index)
    frame["rsi_14"] = rsi(close, 14)
    frame["macd_hist"] = macd(close)["hist"]
    frame["bb_percent_b"] = bollinger_bands(close)["percent_b"]
    frame["sma_20"] = close.rolling(20, min_periods=20).mean()
    frame["sma_50"] = close.rolling(50, min_periods=50).mean()
    log_returns = np.log(close).diff()
    frame["volatility_20"] = log_returns.rolling(20, min_periods=20).std(ddof=0)
    return frame


@dataclass(frozen=True, slots=True)
class MarketContext:
    symbol: str
    timeframe: str
    bar_time: str  # ISO open time of the decision bar (decided at its close)
    candles: tuple[dict[str, float | str | None], ...]  # oldest first
    indicators: dict[str, float | None] = field(default_factory=dict)
    # Portfolio state at the decision (see ``TradingSession.portfolio_view``), or
    # None for market-only agents. Part of the fingerprint when present.
    portfolio: dict[str, Any] | None = None

    @property
    def last_close(self) -> float:
        return float(self.candles[-1]["close"])  # type: ignore[arg-type]

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar_time": self.bar_time,
            "candles": [dict(c) for c in self.candles],
            "indicators": dict(self.indicators),
        }
        if self.portfolio is not None:  # absent for market-only agents (keeps old cache keys)
            data["portfolio"] = json.loads(json.dumps(self.portfolio, sort_keys=True, default=str))
        return data

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_prompt(self) -> str:
        """Human/LLM-readable description of the context."""
        lines = [
            f"Market: {self.symbol} ({self.timeframe} candles). Decision at the close of the "
            f"candle opened {self.bar_time} UTC.",
            "",
            "Indicators at that close:",
        ]
        for key, value in self.indicators.items():
            lines.append(f"  {key}: {'n/a' if value is None else value}")
        if self.portfolio is not None:
            lines += ["", "Simulated portfolio at that close:"]
            for key, value in self.portfolio.items():
                lines.append(f"  {key}: {'n/a' if value is None else value}")
        lines += ["", "Recent candles (oldest first): time, open, high, low, close, volume"]
        for c in self.candles:
            lines.append(f"  {c['time']}, {c['open']}, {c['high']}, {c['low']}, {c['close']}, {c['volume']}")
        return "\n".join(lines)


def build_context(
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    indicators: pd.DataFrame,
    position: int,
    lookback: int,
    portfolio: dict[str, Any] | None = None,
    digits: int = SIGNIFICANT_DIGITS,
) -> MarketContext:
    """Context for the bar at ``position`` using only rows ``<= position``.

    ``digits`` is the number of significant digits kept for indicator values:
    fewer digits make the context (and so the cache key) more robust to tiny
    floating-point differences between data windows.
    """
    window = candles.iloc[max(0, position - lookback + 1) : position + 1]
    rows = tuple(
        {
            "time": ts.strftime("%Y-%m-%d %H:%M"),
            "open": _round(r.open),
            "high": _round(r.high),
            "low": _round(r.low),
            "close": _round(r.close),
            "volume": _round(r.volume),
        }
        for ts, r in zip(window.index, window.itertuples(index=False))
    )
    ind = indicators.iloc[position]
    return MarketContext(
        symbol=symbol,
        timeframe=timeframe,
        bar_time=candles.index[position].strftime("%Y-%m-%dT%H:%M"),
        candles=rows,
        indicators={k: _round(v, digits) for k, v in ind.items()},
        portfolio=portfolio,
    )
