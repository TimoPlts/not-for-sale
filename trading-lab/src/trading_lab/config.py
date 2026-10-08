"""Typed, validated configuration.

Defaults live in the dataclasses below. A TOML file can override them. Loading
is strict: unknown sections or keys raise ``ConfigError``, which catches typos
and makes sure no credential fields can sneak in.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

from trading_lab.core.errors import ConfigError
from trading_lab.core.symbols import SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES, split_symbol


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(obj: Any, name: str, *, low: float, high: float, low_inclusive: bool = True) -> None:
    """Validate ``obj.<name>`` lies in the range and normalise it to float."""
    value = getattr(obj, name)
    qualified = f"{type(obj).__name__}.{name}"
    _require(_is_number(value), f"{qualified} must be a finite number, got {value!r}")
    in_range = (value >= low if low_inclusive else value > low) and value <= high
    bracket = "[" if low_inclusive else "("
    _require(in_range, f"{qualified} must be in {bracket}{low}, {high}], got {value!r}")
    object.__setattr__(obj, name, float(value))


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    initial_cash: float = 10_000.0
    quote_currency: str = "USDT"

    def __post_init__(self) -> None:
        _number(self, "initial_cash", low=0.0, high=1e12, low_inclusive=False)
        _require(self.quote_currency == "USDT", "portfolio.quote_currency must be 'USDT' in V1")


@dataclass(frozen=True, slots=True)
class MarketConfig:
    exchange: str = "binance"
    symbols: tuple[str, ...] = SUPPORTED_SYMBOLS
    timeframe: str = "1h"

    def __post_init__(self) -> None:
        _require(
            isinstance(self.exchange, str) and self.exchange.isidentifier(),
            f"market.exchange must be a CCXT exchange id, got {self.exchange!r}",
        )
        _require(
            isinstance(self.symbols, (list, tuple)) and len(self.symbols) > 0,
            "market.symbols must be a non-empty list",
        )
        symbols = tuple(self.symbols)
        unsupported = [s for s in symbols if s not in SUPPORTED_SYMBOLS]
        _require(
            not unsupported,
            f"unsupported symbols {unsupported}; supported: {list(SUPPORTED_SYMBOLS)}",
        )
        _require(len(set(symbols)) == len(symbols), "market.symbols contains duplicates")
        object.__setattr__(self, "symbols", symbols)
        _require(
            self.timeframe in SUPPORTED_TIMEFRAMES,
            f"unsupported timeframe {self.timeframe!r}; supported: {list(SUPPORTED_TIMEFRAMES)}",
        )


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    fee_rate: float = 0.001  # taker fee (market orders)
    slippage_bps: float = 5.0  # base adverse slippage for market orders
    min_notional: float = 10.0
    # Market impact: "fixed" = slippage_bps only; "volume" = slippage_bps plus
    # impact_coefficient * volatility * sqrt(order value / average bar value traded).
    slippage_model: str = "fixed"
    impact_coefficient: float = 1.0
    volume_lookback: int = 20  # bars used for average volume and volatility
    # Liquidity: max share of average bar volume one order may take (0 disables).
    max_participation_pct: float = 0.0
    # Entries as "market" orders, or "limit" orders resting below the open.
    entry_order_type: str = "market"
    limit_offset_bps: float = 10.0  # buy limit = open * (1 - offset)
    limit_ttl_bars: int = 3  # unfilled remainder expires after this many bars
    maker_fee_rate: float = 0.001  # fee for limit-order fills
    short_borrow_bps_per_day: float = 2.0  # borrow cost of a short, charged when it is covered

    def __post_init__(self) -> None:
        _number(self, "fee_rate", low=0.0, high=0.05)
        _number(self, "slippage_bps", low=0.0, high=500.0)
        _number(self, "min_notional", low=0.0, high=1e9)
        _require(
            self.slippage_model in ("fixed", "volume"),
            f"execution.slippage_model must be 'fixed' or 'volume', got {self.slippage_model!r}",
        )
        _number(self, "impact_coefficient", low=0.0, high=100.0)
        _require(
            isinstance(self.volume_lookback, int)
            and not isinstance(self.volume_lookback, bool)
            and self.volume_lookback >= 2,
            f"execution.volume_lookback must be an integer >= 2, got {self.volume_lookback!r}",
        )
        _number(self, "max_participation_pct", low=0.0, high=1.0)
        _require(
            self.entry_order_type in ("market", "limit"),
            f"execution.entry_order_type must be 'market' or 'limit', got {self.entry_order_type!r}",
        )
        _number(self, "limit_offset_bps", low=0.0, high=2000.0)
        _require(
            isinstance(self.limit_ttl_bars, int)
            and not isinstance(self.limit_ttl_bars, bool)
            and self.limit_ttl_bars >= 1,
            f"execution.limit_ttl_bars must be an integer >= 1, got {self.limit_ttl_bars!r}",
        )
        _number(self, "maker_fee_rate", low=0.0, high=0.05)
        _number(self, "short_borrow_bps_per_day", low=0.0, high=1000.0)


RISK_STATES = ("low", "moderate", "high", "extreme")  # labels of the qwen_risk agent


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_position_pct: float = 0.25
    risk_per_trade_pct: float = 0.01
    stop_loss_pct: float = 0.05
    max_open_positions: int = 4
    max_total_exposure_pct: float = 1.0
    allow_pyramiding: bool = False
    # Simulated shorts: an ensemble SELL with no long open opens a fully collateralised short
    # (no leverage) and a BUY covers it. Off = long-only, as before.
    allow_short: bool = False
    # Circuit breakers (0 disables each one). They block new entries; exits stay allowed.
    max_drawdown_pct: float = 0.25  # kill switch: equity this far below its peak
    daily_loss_limit_pct: float = 0.05  # equity this far below the start of the UTC day
    stop_loss_cooldown_bars: int = 3  # no re-entry into a symbol for N bars after a stop-out
    flatten_on_halt: bool = False  # also close every position when the kill switch trips
    # Exits beyond the fixed stop (0 disables each one).
    trailing_stop_pct: float = 0.0  # stop follows the highest high at this distance (raised at bar closes)
    trailing_activation_pct: float = 0.0  # start trailing once the high is this far above the average cost
    take_profit_pct: float = 0.0  # exit when the high reaches average cost x (1 + this)
    max_holding_bars: int = 0  # time stop: exit at the next open once a position has been held this many bars
    # Initial stop: "percent" = stop_loss_pct below the entry fill; "atr" = atr_stop_multiple x ATR
    # below it (ATR of the bars before the fill), clamped to [atr_stop_min_pct, atr_stop_max_pct].
    # Risk-per-trade sizing uses that distance, so volatile coins get smaller positions.
    stop_mode: str = "percent"
    atr_period: int = 14
    atr_stop_multiple: float = 2.0
    atr_stop_min_pct: float = 0.005
    atr_stop_max_pct: float = 0.25
    # Volatility targeting (0 = off): a position may add at most this much annualised volatility,
    # as a fraction of equity (quantity <= equity x pct / (annual volatility x price)). Calm coins get
    # bigger positions, wild ones smaller. Volatility is that of the bars before the entry.
    position_volatility_pct: float = 0.0
    # Entry filters: they only block NEW entries (never force an exit, never override breakers).
    trend_filter_period: int = 0  # no new entry while the close is below its N-bar simple average (0 = off)
    block_entries_on_risk_states: tuple[str, ...] = ()  # e.g. ("extreme",) or ("high", "extreme")
    risk_state_max_age_bars: int = 8  # how long a reported risk_state stays in force
    # Correlation limit (0 = off): no new entry if this many open/pending positions already move with it.
    max_correlated_positions: int = 0
    correlation_threshold: float = 0.8  # correlation of per-bar log returns that counts as "moves together"
    correlation_lookback: int = 48  # bars in the rolling correlation (up to the signal bar)

    def __post_init__(self) -> None:
        _require(isinstance(self.allow_short, bool), f"risk.allow_short must be true or false, got {self.allow_short!r}")
        _require(
            isinstance(self.max_correlated_positions, int) and not isinstance(self.max_correlated_positions, bool)
            and self.max_correlated_positions >= 0,
            f"risk.max_correlated_positions must be an integer >= 0, got {self.max_correlated_positions!r}",
        )
        _number(self, "correlation_threshold", low=-1.0, high=1.0, low_inclusive=False)
        _require(
            isinstance(self.correlation_lookback, int) and not isinstance(self.correlation_lookback, bool)
            and 5 <= self.correlation_lookback <= 2000,
            f"risk.correlation_lookback must be an integer in [5, 2000], got {self.correlation_lookback!r}",
        )
        _require(
            isinstance(self.trend_filter_period, int) and not isinstance(self.trend_filter_period, bool)
            and 0 <= self.trend_filter_period <= 2000,
            f"risk.trend_filter_period must be an integer in [0, 2000], got {self.trend_filter_period!r}",
        )
        _require(isinstance(self.block_entries_on_risk_states, (list, tuple)),
                 "risk.block_entries_on_risk_states must be a list")
        states = tuple(self.block_entries_on_risk_states)
        bad = [x for x in states if x not in RISK_STATES]
        _require(not bad, f"risk.block_entries_on_risk_states: unknown state(s) {bad}; use {list(RISK_STATES)}")
        object.__setattr__(self, "block_entries_on_risk_states", states)
        _require(
            isinstance(self.risk_state_max_age_bars, int) and not isinstance(self.risk_state_max_age_bars, bool)
            and self.risk_state_max_age_bars >= 1,
            f"risk.risk_state_max_age_bars must be an integer >= 1, got {self.risk_state_max_age_bars!r}",
        )
        _require(self.stop_mode in ("percent", "atr"),
                 f"risk.stop_mode must be 'percent' or 'atr', got {self.stop_mode!r}")
        _require(
            isinstance(self.atr_period, int) and not isinstance(self.atr_period, bool) and 2 <= self.atr_period <= 500,
            f"risk.atr_period must be an integer in [2, 500], got {self.atr_period!r}",
        )
        _number(self, "atr_stop_multiple", low=0.0, high=50.0, low_inclusive=False)
        _number(self, "atr_stop_min_pct", low=0.0, high=0.99, low_inclusive=False)
        _number(self, "atr_stop_max_pct", low=0.0, high=0.99, low_inclusive=False)
        _number(self, "position_volatility_pct", low=0.0, high=10.0)
        _require(self.atr_stop_min_pct <= self.atr_stop_max_pct,
                 "risk.atr_stop_min_pct must not exceed risk.atr_stop_max_pct")
        _number(self, "trailing_stop_pct", low=0.0, high=0.99)
        _number(self, "trailing_activation_pct", low=0.0, high=10.0)
        _number(self, "take_profit_pct", low=0.0, high=100.0)
        _require(
            isinstance(self.max_holding_bars, int) and not isinstance(self.max_holding_bars, bool)
            and 0 <= self.max_holding_bars <= 100_000,
            f"risk.max_holding_bars must be an integer >= 0 (0 = off), got {self.max_holding_bars!r}",
        )
        _number(self, "max_drawdown_pct", low=0.0, high=0.99)
        _number(self, "daily_loss_limit_pct", low=0.0, high=0.99)
        _require(
            isinstance(self.stop_loss_cooldown_bars, int)
            and not isinstance(self.stop_loss_cooldown_bars, bool)
            and self.stop_loss_cooldown_bars >= 0,
            f"risk.stop_loss_cooldown_bars must be an integer >= 0, got {self.stop_loss_cooldown_bars!r}",
        )
        _require(isinstance(self.flatten_on_halt, bool), "risk.flatten_on_halt must be true or false")
        _number(self, "max_position_pct", low=0.0, high=1.0, low_inclusive=False)
        _number(self, "risk_per_trade_pct", low=0.0, high=1.0, low_inclusive=False)
        _number(self, "stop_loss_pct", low=0.0, high=0.99, low_inclusive=False)
        _number(self, "max_total_exposure_pct", low=0.0, high=1.0, low_inclusive=False)
        _require(
            isinstance(self.max_open_positions, int)
            and not isinstance(self.max_open_positions, bool)
            and self.max_open_positions >= 1,
            f"risk.max_open_positions must be an integer >= 1, got {self.max_open_positions!r}",
        )
        _require(
            isinstance(self.allow_pyramiding, bool), "risk.allow_pyramiding must be true or false"
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    cache_dir: str = "data/cache"  # relative paths resolve against the working directory
    use_cache: bool = True
    page_limit: int = 1000  # candles per public OHLCV request
    funding_exchange: str = ""  # CCXT id for funding rates ("" = the market exchange's futures, e.g. binanceusdm)

    def __post_init__(self) -> None:
        _require(
            isinstance(self.cache_dir, str) and self.cache_dir.strip() != "",
            "data.cache_dir must be a non-empty path",
        )
        _require(isinstance(self.use_cache, bool), "data.use_cache must be true or false")
        _require(
            isinstance(self.page_limit, int)
            and not isinstance(self.page_limit, bool)
            and 1 <= self.page_limit <= 5000,
            f"data.page_limit must be an integer in [1, 5000], got {self.page_limit!r}",
        )
        _require(isinstance(self.funding_exchange, str), "data.funding_exchange must be a string")


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """One strategy entry: ``[strategies.<name>]`` in TOML.

    ``enabled`` and ``weight`` are generic. Every other key is passed to the
    strategy's constructor as a parameter, and the strategy registry validates
    them when strategies are built.
    """

    name: str
    enabled: bool = True
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(
            isinstance(self.name, str) and self.name.isidentifier(),
            f"strategy name must be an identifier, got {self.name!r}",
        )
        _require(isinstance(self.enabled, bool), f"strategies.{self.name}.enabled must be bool")
        _number(self, "weight", low=0.0, high=1e6)
        _require(isinstance(self.params, dict), f"strategies.{self.name} params must be a table")
        object.__setattr__(self, "params", dict(self.params))

    @classmethod
    def from_table(cls, name: str, table: Mapping[str, Any]) -> StrategySpec:
        _require(isinstance(table, Mapping), f"[strategies.{name}] must be a table")
        params = {k: v for k, v in table.items() if k not in ("enabled", "weight")}
        return cls(
            name=name,
            enabled=table.get("enabled", True),
            weight=table.get("weight", 1.0),
            params=params,
        )


DEFAULT_STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec("rsi", params={"period": 14, "oversold": 30.0, "overbought": 70.0}),
    StrategySpec("macd", params={"fast": 12, "slow": 26, "signal": 9}),
    StrategySpec("bollinger", params={"period": 20, "num_std": 2.0}),
    # Qwen agents: weight 0 = off (never built, never called). Set a weight to let one vote.
    StrategySpec("qwen_trend", weight=0.0, params={"lookback": 40, "decision_interval": 4}),
    StrategySpec("qwen_momentum", weight=0.0, params={"lookback": 30, "decision_interval": 4}),
    StrategySpec("qwen_risk", weight=0.0, params={"lookback": 30, "decision_interval": 4}),
)


@dataclass(frozen=True, slots=True)
class VotingConfig:
    """How strategy signals are combined (see ``ensemble.voting``).

    ``regime_weights`` (empty = off) multiplies strategy weights by the symbol's
    current trend regime, e.g. ``{"up": {"ma_cross": 2.0}, "sideways": {"ma_cross": 0.5}}``.
    The regime is "up" (close above a rising ``regime_bars`` average), "down"
    (below a falling one) or "sideways", from candles up to the bar only.
    Strategies not listed keep their weight.

    ``min_confirms`` (0 = off) is the desk's confirmation gate: a new entry
    needs at least that many of the ``confirmers`` (strategy names) to confirm
    it. With ``confirm_mode = "agree"`` a confirmer confirms by voting the same
    way; with ``"not_against"`` by not voting the other way (it can veto).
    Exits never need confirmation.
    """

    buy_threshold: float = 0.15
    sell_threshold: float = 0.15
    min_agreeing: int = 1
    regime_weights: Any = ()  # normalised to ((regime, ((strategy, multiplier), ...)), ...)
    regime_bars: int = 50
    regime_slope_bars: int = 10
    confirmers: tuple[str, ...] = ()
    min_confirms: int = 0
    confirm_mode: str = "agree"

    def __post_init__(self) -> None:
        _number(self, "buy_threshold", low=0.0, high=1.0, low_inclusive=False)
        _number(self, "sell_threshold", low=0.0, high=1.0, low_inclusive=False)
        _require(
            isinstance(self.min_agreeing, int)
            and not isinstance(self.min_agreeing, bool)
            and self.min_agreeing >= 1,
            f"voting.min_agreeing must be an integer >= 1, got {self.min_agreeing!r}",
        )
        for name, low in (("regime_bars", 2), ("regime_slope_bars", 1)):
            value = getattr(self, name)
            _require(isinstance(value, int) and not isinstance(value, bool) and low <= value <= 5000,
                     f"voting.{name} must be an integer in [{low}, 5000], got {value!r}")
        object.__setattr__(self, "regime_weights", _normalise_regime_weights(self.regime_weights))
        _require(isinstance(self.confirmers, (list, tuple)) and all(isinstance(c, str) and c.isidentifier()
                                                                   for c in self.confirmers),
                 "voting.confirmers must be a list of strategy names")
        object.__setattr__(self, "confirmers", tuple(self.confirmers))
        _require(len(set(self.confirmers)) == len(self.confirmers), "voting.confirmers lists a strategy twice")
        _require(isinstance(self.min_confirms, int) and not isinstance(self.min_confirms, bool)
                 and 0 <= self.min_confirms <= len(self.confirmers),
                 f"voting.min_confirms must be an integer from 0 to the number of confirmers "
                 f"({len(self.confirmers)}), got {self.min_confirms!r}")
        _require(self.confirm_mode in ("agree", "not_against"),
                 f"voting.confirm_mode must be 'agree' or 'not_against', got {self.confirm_mode!r}")

    def regime_multipliers(self, regime: str | None) -> dict[str, float]:
        """Weight multipliers for a regime ({} when off, unknown or not configured)."""
        return dict(dict(self.regime_weights).get(regime, ())) if regime else {}


REGIMES = ("up", "sideways", "down")


def _pairs(value: Any, what: str) -> list[tuple[Any, Any]]:
    if isinstance(value, Mapping):
        return list(value.items())
    if isinstance(value, (list, tuple)) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in value):
        return [tuple(p) for p in value]
    raise ConfigError(f"{what} must be a table")


def _normalise_regime_weights(raw: Any) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    out = []
    for regime, table in _pairs(raw, "voting.regime_weights"):
        _require(regime in REGIMES, f"voting.regime_weights: unknown regime {regime!r}; use {list(REGIMES)}")
        weights = []
        for strategy, multiplier in _pairs(table, f"voting.regime_weights.{regime}"):
            _require(isinstance(strategy, str) and strategy, f"voting.regime_weights.{regime}: bad strategy name")
            _require(isinstance(multiplier, (int, float)) and not isinstance(multiplier, bool)
                     and math.isfinite(multiplier) and 0 <= multiplier <= 100,
                     f"voting.regime_weights.{regime}.{strategy} must be a number in [0, 100], got {multiplier!r}")
            weights.append((strategy, float(multiplier)))
        out.append((regime, tuple(sorted(weights))))
    return tuple(sorted(out))


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    liquidate_at_end: bool = False  # sell open positions at the final close?

    def __post_init__(self) -> None:
        _require(
            isinstance(self.liquidate_at_end, bool), "backtest.liquidate_at_end must be true or false"
        )


@dataclass(frozen=True, slots=True)
class StorageConfig:
    db_path: str = "data/trading_lab.db"  # SQLite file; ":memory:" for no persistence
    record_trials: bool = False  # log every backtest/sweep/ab/checkup result (see `trading-lab trials`)

    def __post_init__(self) -> None:
        _require(
            isinstance(self.db_path, str) and self.db_path.strip() != "",
            "storage.db_path must be a non-empty path",
        )
        _require(isinstance(self.record_trials, bool), "storage.record_trials must be true or false")


AGENT_MODES = ("live", "record", "replay")
LLM_PROVIDERS = ("qwen", "anthropic")  # keep in sync with trading_lab.llm.factory.PROVIDERS


@dataclass(frozen=True, slots=True)
class AgentsConfig:
    """How AI-agent strategies are called (see ``trading_lab.agents``).

    * ``record``: use a cached answer when there is one, otherwise ask the agent and cache it
    * ``replay``: only use cached answers (never call the agent) for exact reproducibility
    * ``live``: always ask the agent and cache nothing

    The remaining keys configure the model provider used by LLM-backed agents.
    Endpoint URL, model name and token are read from environment variables
    (e.g. ``QWEN_API_URL``, ``QWEN_MODEL``, ``QWEN_API_KEY``), never from here.
    """

    mode: str = "record"
    cache_path: str = "data/agent_cache.db"
    provider: str = "qwen"
    request_timeout_seconds: float = 30.0  # per HTTP request
    max_retries: int = 2  # extra attempts after a timeout, network error, 429 or 5xx
    retry_backoff_seconds: float = 1.0  # doubles after each failed attempt
    temperature: float = 0.0  # 0 = as deterministic as the model allows
    max_output_tokens: int = 512
    # Circuit breaker for the model endpoint: after this many failed calls in a row,
    # skip calls (agents vote HOLD at once) for the cooldown, then try again. 0 disables.
    failure_threshold: int = 5
    failure_cooldown_seconds: float = 300.0

    def __post_init__(self) -> None:
        _require(self.mode in AGENT_MODES, f"agents.mode must be one of {AGENT_MODES}, got {self.mode!r}")
        _require(
            isinstance(self.cache_path, str) and self.cache_path.strip() != "",
            "agents.cache_path must be a non-empty path",
        )
        _require(
            self.provider in LLM_PROVIDERS,
            f"agents.provider must be one of {LLM_PROVIDERS}, got {self.provider!r}",
        )
        _number(self, "request_timeout_seconds", low=0.0, high=600.0, low_inclusive=False)
        _require(
            isinstance(self.max_retries, int)
            and not isinstance(self.max_retries, bool)
            and 0 <= self.max_retries <= 10,
            f"agents.max_retries must be an integer in [0, 10], got {self.max_retries!r}",
        )
        _number(self, "retry_backoff_seconds", low=0.0, high=60.0)
        _number(self, "temperature", low=0.0, high=2.0)
        _require(
            isinstance(self.max_output_tokens, int)
            and not isinstance(self.max_output_tokens, bool)
            and 16 <= self.max_output_tokens <= 32768,
            f"agents.max_output_tokens must be an integer in [16, 32768], got {self.max_output_tokens!r}",
        )
        _require(
            isinstance(self.failure_threshold, int)
            and not isinstance(self.failure_threshold, bool)
            and 0 <= self.failure_threshold <= 1000,
            f"agents.failure_threshold must be an integer in [0, 1000], got {self.failure_threshold!r}",
        )
        _number(self, "failure_cooldown_seconds", low=0.0, high=86_400.0)


ALERT_LEVELS = ("info", "warning", "critical")
ALERT_FORMATS = ("ntfy", "slack", "discord", "json")
ALERT_CHANNELS = ("webhook", "email", "telegram")


@dataclass(frozen=True, slots=True)
class AlertsConfig:
    """Push notifications from live paper trading (see ``trading_lab.alerts``).

    The webhook URL is read from the ``TRADING_LAB_ALERT_URL`` environment
    variable only (it usually contains a secret token).
    """

    enabled: bool = False
    channels: tuple[str, ...] = ("webhook",)  # webhook, email and/or telegram (settings from the environment)
    format: str = "ntfy"  # ntfy | slack | discord | json (webhook only)
    min_level: str = "warning"  # info also reports every entry and exit
    daily_summary: bool = True  # a run summary after each UTC day
    outage_after_cycles: int = 3  # alert once this many cycles in a row could not be processed
    repeat_after_minutes: float = 60.0  # do not repeat the same alert sooner

    def __post_init__(self) -> None:
        _require(isinstance(self.enabled, bool), "alerts.enabled must be true or false")
        _require(
            isinstance(self.channels, (list, tuple)) and len(self.channels) > 0
            and all(c in ALERT_CHANNELS for c in self.channels) and len(set(self.channels)) == len(self.channels),
            f"alerts.channels must be a non-empty list of distinct values from {ALERT_CHANNELS}, "
            f"got {self.channels!r}",
        )
        object.__setattr__(self, "channels", tuple(self.channels))
        _require(self.format in ALERT_FORMATS, f"alerts.format must be one of {ALERT_FORMATS}, got {self.format!r}")
        _require(self.min_level in ALERT_LEVELS,
                 f"alerts.min_level must be one of {ALERT_LEVELS}, got {self.min_level!r}")
        _require(isinstance(self.daily_summary, bool), "alerts.daily_summary must be true or false")
        _require(
            isinstance(self.outage_after_cycles, int)
            and not isinstance(self.outage_after_cycles, bool)
            and self.outage_after_cycles >= 1,
            f"alerts.outage_after_cycles must be an integer >= 1, got {self.outage_after_cycles!r}",
        )
        _number(self, "repeat_after_minutes", low=0.0, high=10_080.0)


@dataclass(frozen=True, slots=True)
class AppConfig:
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategies: tuple[StrategySpec, ...] = DEFAULT_STRATEGIES
    voting: VotingConfig = field(default_factory=VotingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)

    def __post_init__(self) -> None:
        for symbol in self.market.symbols:
            _, quote = split_symbol(symbol)
            _require(
                quote == self.portfolio.quote_currency,
                f"symbol {symbol} is not quoted in {self.portfolio.quote_currency}",
            )
        object.__setattr__(self, "strategies", tuple(self.strategies))
        names = [s.name for s in self.strategies]
        _require(len(set(names)) == len(names), "duplicate strategy names")
        unknown = sorted({s for _, ws in self.voting.regime_weights for s, _ in ws} - set(names))
        _require(not unknown, f"voting.regime_weights names strategies that are not configured: {unknown}")
        voting = {s.name for s in self.strategies if s.enabled and s.weight > 0}
        silent = sorted(set(self.voting.confirmers) - voting)
        _require(not silent, f"voting.confirmers must be enabled strategies with a positive weight: {silent}")
        enabled = [s for s in self.strategies if s.enabled]
        _require(bool(enabled), "at least one strategy must be enabled")
        _require(
            sum(s.weight for s in enabled) > 0, "enabled strategies must have positive total weight"
        )

    @property
    def enabled_strategies(self) -> tuple[StrategySpec, ...]:
        """Strategies that take part: enabled and with a positive weight.

        A weight of 0 switches a strategy off completely: it is not built, not
        evaluated (an AI agent is never called) and casts no vote, so it cannot
        count towards ``voting.min_agreeing`` either.
        """
        return tuple(s for s in self.strategies if s.enabled and s.weight > 0)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AppConfig:
        unknown = set(data) - set(_SECTIONS) - {"strategies"}
        _require(not unknown, f"unknown config section(s): {sorted(unknown)}")
        sections: dict[str, Any] = {}
        for name, section_cls in _SECTIONS.items():
            raw = data.get(name, {})
            _require(isinstance(raw, Mapping), f"config section [{name}] must be a table")
            allowed = {f.name for f in fields(section_cls)}
            bad = set(raw) - allowed
            _require(not bad, f"unknown key(s) in [{name}]: {sorted(bad)}")
            sections[name] = section_cls(**raw)
        if "strategies" in data:
            raw_strategies = data["strategies"]
            _require(isinstance(raw_strategies, Mapping), "[strategies] must be a table of tables")
            sections["strategies"] = tuple(
                StrategySpec.from_table(name, table) for name, table in raw_strategies.items()
            )
        return cls(**sections)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["market"]["symbols"] = list(self.market.symbols)
        data["risk"]["block_entries_on_risk_states"] = list(self.risk.block_entries_on_risk_states)
        data["voting"]["regime_weights"] = {r: dict(w) for r, w in self.voting.regime_weights}
        data["voting"]["confirmers"] = list(self.voting.confirmers)
        data["strategies"] = [asdict(s) for s in self.strategies]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AppConfig:
        """Inverse of ``to_dict()``, e.g. to rebuild the config stored with a run."""
        data = dict(data)
        data.pop("strategy_override", None)
        strategies = data.get("strategies")
        if isinstance(strategies, list):
            data["strategies"] = {
                s["name"]: {"enabled": s["enabled"], "weight": s["weight"], **s["params"]}
                for s in strategies
            }
        return cls.from_mapping(data)

    def to_mapping(self) -> dict[str, Any]:
        """TOML-shaped mapping; ``AppConfig.from_mapping(cfg.to_mapping()) == cfg``."""
        data = self.to_dict()
        data["strategies"] = {
            s.name: {"enabled": s.enabled, "weight": s.weight, **s.params} for s in self.strategies
        }
        return data

    def with_overrides(self, overrides: Mapping[str, Mapping[str, Any]]) -> AppConfig:
        """Copy with selected keys replaced, e.g. ``{"market": {"timeframe": "15m"}}``.

        The result is validated like any loaded config.
        """
        data = self.to_mapping()
        for section, values in overrides.items():
            _require(isinstance(values, Mapping), f"override for [{section}] must be a table")
            target = data.setdefault(section, {})
            _require(isinstance(target, dict), f"cannot override [{section}]")
            target.update(values)
        return AppConfig.from_mapping(data)

    def fingerprint(self) -> str:
        """SHA-256 of the canonical config, recorded with every run for reproducibility."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_SECTIONS: dict[str, type] = {
    "portfolio": PortfolioConfig,
    "market": MarketConfig,
    "execution": ExecutionConfig,
    "risk": RiskConfig,
    "data": DataConfig,
    "voting": VotingConfig,
    "backtest": BacktestConfig,
    "storage": StorageConfig,
    "agents": AgentsConfig,
    "alerts": AlertsConfig,
}


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from a TOML file, or return the defaults if ``path`` is None."""
    if path is None:
        return AppConfig()
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return AppConfig.from_mapping(data)
