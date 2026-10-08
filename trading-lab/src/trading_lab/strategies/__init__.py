"""Trading strategies. Importing this package registers the built-in strategies."""

from trading_lab.strategies.base import IndicatorStrategy, Strategy, scaled_confidence
from trading_lab.strategies.registry import (
    available_strategies,
    build_strategies,
    create_strategy,
    register_strategy,
)

# Built-in strategies register themselves on import.
from trading_lab.strategies.bollinger import BollingerMeanReversionStrategy  # noqa: E402
from trading_lab.strategies.context import ContextStrategy, FundingStrategy, SentimentStrategy  # noqa: E402
from trading_lab.strategies.macd import MacdStrategy  # noqa: E402
from trading_lab.strategies.rsi import RsiStrategy  # noqa: E402
from trading_lab.strategies.trend import (  # noqa: E402
    DonchianBreakoutStrategy,
    MovingAverageCrossStrategy,
    TimeSeriesMomentumStrategy,
)

__all__ = [
    "BollingerMeanReversionStrategy",
    "ContextStrategy",
    "FundingStrategy",
    "SentimentStrategy",
    "DonchianBreakoutStrategy",
    "IndicatorStrategy",
    "MacdStrategy",
    "MovingAverageCrossStrategy",
    "RsiStrategy",
    "Strategy",
    "TimeSeriesMomentumStrategy",
    "available_strategies",
    "build_strategies",
    "create_strategy",
    "register_strategy",
    "scaled_confidence",
]
