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


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_position_pct: float = 0.25
    risk_per_trade_pct: float = 0.01
    stop_loss_pct: float = 0.05
    max_open_positions: int = 4
    max_total_exposure_pct: float = 1.0
    allow_pyramiding: bool = False
    # Circuit breakers (0 disables each one). They block new entries; exits stay allowed.
    max_drawdown_pct: float = 0.25  # kill switch: equity this far below its peak
    daily_loss_limit_pct: float = 0.05  # equity this far below the start of the UTC day
    stop_loss_cooldown_bars: int = 3  # no re-entry into a symbol for N bars after a stop-out
    flatten_on_halt: bool = False  # also close every position when the kill switch trips

    def __post_init__(self) -> None:
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
)


@dataclass(frozen=True, slots=True)
class VotingConfig:
    """How strategy signals are combined (see ``ensemble.voting``)."""

    buy_threshold: float = 0.15
    sell_threshold: float = 0.15
    min_agreeing: int = 1

    def __post_init__(self) -> None:
        _number(self, "buy_threshold", low=0.0, high=1.0, low_inclusive=False)
        _number(self, "sell_threshold", low=0.0, high=1.0, low_inclusive=False)
        _require(
            isinstance(self.min_agreeing, int)
            and not isinstance(self.min_agreeing, bool)
            and self.min_agreeing >= 1,
            f"voting.min_agreeing must be an integer >= 1, got {self.min_agreeing!r}",
        )


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

    def __post_init__(self) -> None:
        _require(
            isinstance(self.db_path, str) and self.db_path.strip() != "",
            "storage.db_path must be a non-empty path",
        )


AGENT_MODES = ("live", "record", "replay")
LLM_PROVIDERS = ("qwen",)  # keep in sync with trading_lab.llm.factory.PROVIDERS


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
        enabled = [s for s in self.strategies if s.enabled]
        _require(bool(enabled), "at least one strategy must be enabled")
        _require(
            sum(s.weight for s in enabled) > 0, "enabled strategies must have positive total weight"
        )

    @property
    def enabled_strategies(self) -> tuple[StrategySpec, ...]:
        return tuple(s for s in self.strategies if s.enabled)

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
