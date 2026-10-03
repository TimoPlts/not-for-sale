"""Supported markets and candle timeframes."""

SUPPORTED_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT")

_TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 3 * 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "6h": 6 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
}

SUPPORTED_TIMEFRAMES: tuple[str, ...] = tuple(_TIMEFRAME_SECONDS)


def timeframe_to_seconds(timeframe: str) -> int:
    """Length of one candle in seconds, e.g. ``"1h" -> 3600``."""
    try:
        return _TIMEFRAME_SECONDS[timeframe]
    except KeyError:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; expected one of {SUPPORTED_TIMEFRAMES}"
        ) from None


def split_symbol(symbol: str) -> tuple[str, str]:
    """Split ``"BTC/USDT"`` into ``("BTC", "USDT")``."""
    base, sep, quote = symbol.partition("/")
    if not sep or not base or not quote:
        raise ValueError(f"malformed symbol {symbol!r}; expected 'BASE/QUOTE'")
    return base, quote
