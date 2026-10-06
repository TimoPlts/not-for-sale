"""Connectivity and agent smoke tests (``trading-lab agent-test``).

* ``provider_smoke_test`` checks the environment, sends one tiny, harmless
  prompt to the model and validates the structured answer.
* ``agent_smoke_test`` asks one agent (e.g. ``qwen_trend``) about the latest
  closed candle of one symbol, exactly as a backtest would, in ``live`` mode
  with a throw-away in-memory cache.

Neither places, schedules or simulates a trade, writes to the run database
or the answer cache, or needs exchange credentials (market data comes from
public endpoints or the synthetic provider).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from trading_lab.agents import LLMProviderStrategy, MemoryResponseCache
from trading_lab.agents.parsing import parse_agent_json
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data.base import MarketDataProvider, timeframe_delta
from trading_lab.engine import TradingSession
from trading_lab.ensemble import VotingEngine
from trading_lab.llm import PROVIDERS, LLMProvider, OpenAICompatibleProvider, ProviderError, build_llm_provider
from trading_lab.strategies.registry import create_strategy, strategy_class

_HIDDEN_SUFFIXES = ("API_KEY",)  # values of these variables are never printed
SMOKE_SYSTEM = (
    "You are a connectivity check for a paper-trading research tool. "
    "Reply with exactly one JSON object and nothing else."
)
SMOKE_USER = 'Reply with exactly this JSON: {"direction": "HOLD", "confidence": 0, "rationale": "connectivity test"}'


@dataclass
class SmokeResult:
    ok: bool
    target: str
    lines: list[str] = field(default_factory=list)
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def environment_lines(provider: LLMProvider) -> list[str]:
    """Which variables are set; secrets are never shown."""
    if not isinstance(provider, OpenAICompatibleProvider):
        return []
    import os

    lines = []
    for suffix in ("API_URL", "API_KEY", "MODEL"):
        var = provider.env_var(suffix)
        present = bool(os.environ.get(var, "").strip())
        if not present:
            shown = "MISSING"
        elif suffix in _HIDDEN_SUFFIXES:
            shown = "set (hidden)"
        elif suffix == "API_URL":
            shown = f"set -> {provider.endpoint}"
        else:
            shown = f"set -> {provider.model}"
        lines.append(f"{var}: {shown}")
    return lines


def provider_smoke_test(config: AppConfig, provider: LLMProvider | None = None) -> SmokeResult:
    provider = provider or build_llm_provider(config.agents)
    result = SmokeResult(False, provider.name, environment_lines(provider))
    try:
        provider.check_ready(need_credentials=True)
        completion = provider.chat(SMOKE_SYSTEM, SMOKE_USER)
        answer = parse_agent_json(completion.text)
    except (ConfigError, ProviderError, ValueError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    tokens = (
        f"{completion.input_tokens} in / {completion.output_tokens} out (reported)"
        if completion.input_tokens is not None and completion.output_tokens is not None
        else "not reported by the endpoint"
    )
    result.ok = True
    result.details = {"model": completion.model, "latency_seconds": completion.latency_seconds,
                      "attempts": completion.attempts, "answer": answer.to_json()}
    result.lines += [
        f"model: {completion.model}",
        f"latency: {completion.latency_seconds:.2f}s ({completion.attempts} attempt(s))",
        f"tokens: {tokens}",
        f"structured answer: {answer.direction.value.upper()} confidence={answer.confidence:.2f} "
        f"rationale={answer.rationale!r}",
    ]
    return result


def agent_smoke_test(
    config: AppConfig,
    name: str,
    market: MarketDataProvider,
    symbol: str,
    *,
    provider: LLMProvider | None = None,
    now: datetime | None = None,
) -> SmokeResult:
    cls = strategy_class(name)
    if cls is None or not issubclass(cls, LLMProviderStrategy):
        raise ConfigError(f"{name!r} is not an LLM agent; choose a provider ({', '.join(sorted(PROVIDERS))}) "
                          f"or one of: {', '.join(_llm_strategy_names())}")
    specs = {s.name: s for s in config.strategies}
    params = dict(specs[name].params) if name in specs else {}
    params["decision_interval"] = 1  # always ask, whatever the bar
    strategy = create_strategy(name, params)
    assert isinstance(strategy, LLMProviderStrategy)
    provider = provider or build_llm_provider(config.agents)
    result = SmokeResult(False, name, environment_lines(provider))
    try:
        provider.check_ready(need_credentials=True)
    except ConfigError as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    strategy.attach_provider(provider)
    strategy.configure(mode="live", cache=MemoryResponseCache(), timeframe=config.market.timeframe)

    tf = config.market.timeframe
    now = now or datetime.now(timezone.utc)
    candles = market.fetch_ohlcv(symbol, tf, now - (strategy.history_bars + 2) * timeframe_delta(tf))
    if len(candles) < strategy.warmup_bars:
        result.error = f"only {len(candles)} closed candles for {symbol}; need {strategy.warmup_bars}"
        return result
    view = None
    if strategy.uses_portfolio:  # a fresh, flat simulated portfolio: nothing is traded
        session = TradingSession(config, VotingEngine({name: 1.0}))
        view = session.portfolio_view(symbol, candles.index[-1].to_pydatetime())
    signal = strategy.signal_at(symbol, candles, len(candles) - 1, view)
    meta = signal.metadata
    llm = meta.get("llm") or {}
    result.details = {"signal": {"direction": signal.direction.value, "confidence": signal.confidence,
                                 **{k: v for k, v in meta.items() if k != "params"}}}
    result.lines += [
        f"data: {market.name} {symbol} {tf}, decision at the close of the candle opened "
        f"{candles.index[-1]:%Y-%m-%d %H:%M} UTC" + (" (with a flat simulated portfolio)" if view else ""),
        f"model: {llm.get('model', provider.model)}",
    ]
    if llm:
        result.lines.append(f"latency: {llm['latency_seconds']:.2f}s ({llm['attempts']} attempt(s)), "
                            f"tokens {llm['input_tokens']} in / {llm['output_tokens']} out"
                            + (" (estimated)" if llm.get("tokens_estimated") else ""))
    if "error" in meta:
        result.error = str(meta["error"])
        return result
    label = getattr(strategy, "label", None)
    result.lines.append(
        f"answer: {signal.direction.value.upper()} confidence={signal.confidence:.2f}"
        + (f" {label}={meta.get(label)}" if label else "")
    )
    result.lines.append(f"rationale: {meta.get('rationale', '')}")
    result.ok = True
    return result


def _llm_strategy_names() -> list[str]:
    from trading_lab.strategies.registry import available_strategies

    return [n for n in available_strategies()
            if (c := strategy_class(n)) is not None and issubclass(c, LLMProviderStrategy)]


__all__ = ["SmokeResult", "agent_smoke_test", "provider_smoke_test"]
