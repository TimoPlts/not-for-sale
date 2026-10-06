"""Three specialist agents that share one LLM provider but play different roles.

* ``qwen_trend``: market direction and trend strength (EMA 20/50/200,
  returns, RSI, MACD, volume trend). Adds ``regime``.
* ``qwen_momentum``: whether momentum supports acting *now* (RSI and MACD
  changes, return acceleration, volume change, candle structure). Adds
  ``momentum_state``.
* ``qwen_risk``: whether conditions are too risky for new exposure (ATR,
  volatility, range expansion, drawdown, liquidity and, by default, the
  simulated portfolio: exposure, open positions, recent stop-outs, breakers).
  Adds ``risk_state``.

Each agent sees only its own causal features plus recent candles, answers in
strict JSON, and becomes one vote in the ensemble. A SELL is only an opinion
(the portfolio is long-only), and no agent can change portfolio state, size
an order or override a circuit breaker.

The names start with ``qwen_`` because Qwen is the configured provider; the
provider itself comes from ``[agents] provider``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from trading_lab.agents.base import Agent
from trading_lab.agents.llm import LLMProviderStrategy, ProviderAgent
from trading_lab.indicators import atr, ema, macd, rsi, windowed_ema
from trading_lab.strategies.registry import register_strategy

_COMMON_RULES = """\
You take part in a crypto PAPER-TRADING research simulation. You only give an opinion: you cannot \
place, size or cancel orders. An ensemble vote, circuit breakers, a risk manager and a simulated \
executor decide what happens, and they always have the final word.
The portfolio is long-only: "BUY" means open (or keep) a long position, "SELL" means close an \
existing long and never opens a short, "HOLD" means no change.
Base your answer only on the data given. Use "HOLD" with confidence 0 when the evidence is unclear.
Reply with exactly one JSON object and nothing else:
"""


def _system_prompt(role: str, task: str, label: str, values: tuple[str, ...]) -> str:
    choices = " | ".join(f'"{v}"' for v in values)
    schema = (
        '{"direction": "BUY" | "SELL" | "HOLD", "confidence": <number from 0 to 1>, '
        f'"rationale": "<one or two sentences citing the data>", "{label}": {choices}}}'
    )
    return f"You are the {role}.\n{task}\n\n{_COMMON_RULES}{schema}"


EMA_200_WINDOW = 1000


def _pct(series: pd.Series) -> pd.Series:
    return series * 100.0


def _returns_pct(close: pd.Series, bars: int) -> pd.Series:
    return _pct(close / close.shift(bars) - 1.0)


class SpecialistStrategy(LLMProviderStrategy):
    """A role-specific LLM agent: own prompt, own features, one extra label."""

    label: ClassVar[str]
    label_values: ClassVar[tuple[str, ...]]
    context_digits: ClassVar[int] = 5  # robust contexts: tiny float noise must not change cache keys
    agent_version: ClassVar[str] = "1"  # bump when features or the prompt layout change

    def build_agent(self, **params: Any) -> Agent:
        if params:
            raise ValueError(f"unknown parameter(s) {sorted(params)}")
        return ProviderAgent(
            self.name, self.system_prompt, labels={self.label: self.label_values}, version=self.agent_version
        )


@register_strategy
class QwenTrendStrategy(SpecialistStrategy):
    """Config: ``[strategies.qwen_trend]`` with ``lookback``, ``decision_interval`` and
    ``portfolio_context`` (default false: market-only, so answers are shared by experiments)."""

    name = "qwen_trend"
    label = "regime"
    label_values = ("bullish_trend", "bearish_trend", "sideways", "uncertain")
    indicator_warmup = 60  # EMA 50 plus its 10-bar slope
    system_prompt = _system_prompt(
        "Trend Agent",
        "Judge the market direction and the strength of the trend for this symbol. Prefer BUY only "
        "in a clear uptrend (price above rising averages, positive returns, supportive MACD), SELL "
        "when an uptrend has clearly broken down, and HOLD in sideways or unclear markets. The "
        "confidence should reflect how strong and consistent the trend is.",
        label, label_values,
    )

    @property
    def history_bars(self) -> int:
        return max(super().history_bars, EMA_200_WINDOW)

    def context_frame(self, candles: pd.DataFrame) -> pd.DataFrame:
        close, volume = candles["close"], candles["volume"]
        # EMA 200 over a fixed window: a plain one still differs in the 5th digit after
        # 1000 bars depending on where the data starts (live versus backtest windows).
        ema20, ema50, ema200 = ema(close, 20), ema(close, 50), windowed_ema(close, 200, EMA_200_WINDOW)
        m = macd(close)
        f = pd.DataFrame(index=candles.index)
        f["close"] = close
        f["ema_20"], f["ema_50"], f["ema_200"] = ema20, ema50, ema200
        f["close_vs_ema_20_pct"] = _pct(close / ema20 - 1)
        f["close_vs_ema_50_pct"] = _pct(close / ema50 - 1)
        f["close_vs_ema_200_pct"] = _pct(close / ema200 - 1)
        f["ema_20_vs_ema_50_pct"] = _pct(ema20 / ema50 - 1)
        f["ema_50_slope_10_bars_pct"] = _pct(ema50 / ema50.shift(10) - 1)
        for bars in (5, 20, 50):
            f[f"return_{bars}_bars_pct"] = _returns_pct(close, bars)
        f["rsi_14"] = rsi(close, 14)
        f["macd"], f["macd_signal"], f["macd_hist"] = m["macd"], m["signal"], m["hist"]
        f["volume_trend_10_vs_50"] = (
            volume.rolling(10, min_periods=10).mean() / volume.rolling(50, min_periods=50).mean()
        )
        return f


@register_strategy
class QwenMomentumStrategy(SpecialistStrategy):
    """Config: ``[strategies.qwen_momentum]`` with ``lookback``, ``decision_interval`` and
    ``portfolio_context`` (default false)."""

    name = "qwen_momentum"
    label = "momentum_state"
    label_values = ("strengthening", "weakening", "neutral")
    indicator_warmup = 40
    system_prompt = _system_prompt(
        "Momentum Agent",
        "Judge whether momentum supports entering or exiting right now. BUY when momentum is "
        "clearly strengthening upward (rising RSI and MACD histogram, accelerating returns, "
        "supportive volume), SELL when upward momentum is clearly fading or turning down, HOLD "
        "otherwise. Focus on the most recent bars, not the long-term trend.",
        label, label_values,
    )

    def context_frame(self, candles: pd.DataFrame) -> pd.DataFrame:
        o, h, lo, c, v = (candles[k] for k in ("open", "high", "low", "close", "volume"))
        r = rsi(c, 14)
        m = macd(c)
        ret3 = _returns_pct(c, 3)
        f = pd.DataFrame(index=candles.index)
        f["rsi_14"] = r
        f["rsi_change_3_bars"] = r - r.shift(3)
        f["macd"], f["macd_signal"], f["macd_hist"] = m["macd"], m["signal"], m["hist"]
        f["macd_hist_change_3_bars"] = m["hist"] - m["hist"].shift(3)
        for bars in (1, 3, 6, 12):
            f[f"return_{bars}_bars_pct"] = _returns_pct(c, bars)
        f["return_acceleration_pct"] = ret3 - ret3.shift(3)  # last 3 bars versus the 3 before
        avg_volume = v.rolling(20, min_periods=20).mean()
        f["volume_vs_avg_20"] = v / avg_volume
        f["volume_3_vs_avg_20"] = v.rolling(3, min_periods=3).mean() / avg_volume
        top, bottom = np.maximum(o, c), np.minimum(o, c)
        f["last_body_pct"] = _pct((c - o) / o)
        f["last_upper_wick_pct"] = _pct((h - top) / o)
        f["last_lower_wick_pct"] = _pct((bottom - lo) / o)
        f["last_close_location"] = ((c - lo) / (h - lo)).where(h > lo)  # 0 = at the low, 1 = at the high
        return f


@register_strategy
class QwenRiskStrategy(SpecialistStrategy):
    """Config: ``[strategies.qwen_risk]`` with ``lookback``, ``decision_interval`` and
    ``portfolio_context`` (default true: the agent sees exposure, stop-outs and breakers)."""

    name = "qwen_risk"
    label = "risk_state"
    label_values = ("low", "moderate", "high", "extreme")
    indicator_warmup = 100  # 100-bar volatility baseline
    default_portfolio_context = True
    system_prompt = _system_prompt(
        "Risk/Regime Agent",
        "Judge whether market and portfolio conditions are too risky for NEW exposure in this "
        "symbol. Consider volatility (ATR, volatility versus its baseline), range expansion, the "
        "recent drawdown from highs, liquidity and, when given, the simulated portfolio's exposure, "
        "open positions, recent stop-outs and circuit-breaker status. Answer SELL when risk is high "
        "or extreme (reduce exposure), BUY only when risk is low and conditions support new "
        "exposure, and HOLD otherwise. You cannot override circuit breakers; if new entries are "
        "blocked, say so in the rationale.",
        label, label_values,
    )

    def context_frame(self, candles: pd.DataFrame) -> pd.DataFrame:
        h, lo, c, v = (candles[k] for k in ("high", "low", "close", "volume"))
        log_returns = np.log(c).diff()
        vol20 = log_returns.rolling(20, min_periods=20).std(ddof=0)
        vol100 = log_returns.rolling(100, min_periods=100).std(ddof=0)
        bar_range = (h - lo) / c
        a = atr(h, lo, c, 14)
        f = pd.DataFrame(index=candles.index)
        f["atr_14"] = a
        f["atr_pct"] = _pct(a / c)
        f["volatility_20_pct"] = _pct(vol20)
        f["volatility_20_vs_100"] = vol20 / vol100
        f["range_pct"] = _pct(bar_range)
        f["range_vs_avg_20"] = bar_range / bar_range.rolling(20, min_periods=20).mean()
        f["drawdown_from_50_bar_high_pct"] = _pct(c / h.rolling(50, min_periods=50).max() - 1)
        f["return_24_bars_pct"] = _returns_pct(c, 24)
        quote_volume = c * v
        f["avg_quote_volume_20"] = quote_volume.rolling(20, min_periods=20).mean()
        f["volume_vs_avg_20"] = v / v.rolling(20, min_periods=20).mean()
        return f
