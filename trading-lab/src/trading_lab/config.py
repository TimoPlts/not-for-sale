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
    fee_rate: float = 0.001
    slippage_bps: float = 5.0
    min_notional: float = 10.0

    def __post_init__(self) -> None:
        _number(self, "fee_rate", low=0.0, high=0.05)
        _number(self, "slippage_bps", low=0.0, high=500.0)
        _number(self, "min_notional", low=0.0, high=1e9)


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_position_pct: float = 0.25
    risk_per_trade_pct: float = 0.01
    stop_loss_pct: float = 0.05
    max_open_positions: int = 4
    max_total_exposure_pct: float = 1.0
    allow_pyramiding: bool = False

    def __post_init__(self) -> None:
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
class AppConfig:
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def __post_init__(self) -> None:
        for symbol in self.market.symbols:
            _, quote = split_symbol(symbol)
            _require(
                quote == self.portfolio.quote_currency,
                f"symbol {symbol} is not quoted in {self.portfolio.quote_currency}",
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AppConfig:
        unknown = set(data) - set(_SECTIONS)
        _require(not unknown, f"unknown config section(s): {sorted(unknown)}")
        sections: dict[str, Any] = {}
        for name, section_cls in _SECTIONS.items():
            raw = data.get(name, {})
            _require(isinstance(raw, Mapping), f"config section [{name}] must be a table")
            allowed = {f.name for f in fields(section_cls)}
            bad = set(raw) - allowed
            _require(not bad, f"unknown key(s) in [{name}]: {sorted(bad)}")
            sections[name] = section_cls(**raw)
        return cls(**sections)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["market"]["symbols"] = list(self.market.symbols)
        return data

    def fingerprint(self) -> str:
        """SHA-256 of the canonical config, recorded with every run for reproducibility."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_SECTIONS: dict[str, type] = {
    "portfolio": PortfolioConfig,
    "market": MarketConfig,
    "execution": ExecutionConfig,
    "risk": RiskConfig,
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
