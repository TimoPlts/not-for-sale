"""Market data providers (read-only, public data only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trading_lab.data.base import (
    OHLCV_COLUMNS,
    MarketDataProvider,
    empty_ohlcv,
    find_gaps,
    normalize_ohlcv,
    slice_ohlcv,
)
from trading_lab.data.cache import CachedProvider
from trading_lab.data.ccxt_provider import CcxtPublicProvider
from trading_lab.data.synthetic import SyntheticProvider, candles_from_closes

if TYPE_CHECKING:
    from trading_lab.config import AppConfig


def build_provider(config: AppConfig) -> MarketDataProvider:
    """Public CCXT provider for the configured exchange, optionally behind the CSV cache."""
    provider: MarketDataProvider = CcxtPublicProvider(
        config.market.exchange, page_limit=config.data.page_limit, funding_exchange=config.data.funding_exchange
    )
    if config.data.use_cache:
        provider = CachedProvider(provider, config.data.cache_dir)
    return provider


__all__ = [
    "OHLCV_COLUMNS",
    "CachedProvider",
    "CcxtPublicProvider",
    "MarketDataProvider",
    "SyntheticProvider",
    "build_provider",
    "candles_from_closes",
    "empty_ohlcv",
    "find_gaps",
    "normalize_ohlcv",
    "slice_ohlcv",
]
