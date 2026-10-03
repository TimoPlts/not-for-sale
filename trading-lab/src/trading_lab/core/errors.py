"""Exception hierarchy for trading-lab."""


class TradingLabError(Exception):
    """Base class for all trading-lab errors."""


class ConfigError(TradingLabError):
    """Invalid or unknown configuration."""


class InsufficientFundsError(TradingLabError):
    """A fill would drive cash below zero."""


class PositionError(TradingLabError):
    """A fill violates position rules (e.g. selling more than held; long-only)."""


class MissingPriceError(TradingLabError, KeyError):
    """A mark price is required for a symbol but was not provided."""
