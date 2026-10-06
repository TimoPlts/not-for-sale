"""Name → strategy class registry, so strategies can be enabled from config."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, TypeVar

from trading_lab.config import StrategySpec
from trading_lab.core.errors import ConfigError
from trading_lab.strategies.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}

S = TypeVar("S", bound=type[Strategy])


def register_strategy(cls: S) -> S:
    """Class decorator that makes a strategy available by its ``name``."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name.isidentifier():
        raise TypeError(f"{cls.__name__} must define a class attribute `name` (identifier)")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"strategy name {name!r} already registered by {existing.__name__}")
    _REGISTRY[name] = cls
    return cls


def available_strategies() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def strategy_class(name: str) -> type[Strategy] | None:
    return _REGISTRY.get(name)


def create_strategy(name: str, params: Mapping[str, Any] | None = None) -> Strategy:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ConfigError(
            f"unknown strategy {name!r}; available: {list(available_strategies())}"
        ) from None
    try:
        return cls(**dict(params or {}))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid parameters for strategy {name!r}: {exc}") from exc


def build_strategies(specs: Iterable[StrategySpec]) -> list[Strategy]:
    """Instantiate every enabled strategy spec, in order."""
    return [create_strategy(spec.name, spec.params) for spec in specs if spec.enabled]
