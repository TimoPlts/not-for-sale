"""Guard rails: the codebase must never trade for real or handle private API keys."""

import re

from conftest import SRC_DIR

# Private/trading endpoints of CCXT (and exchanges generally) that must never be called.
FORBIDDEN_CALLS = re.compile(
    r"\.\s*(create_\w*order|edit_order|cancel_\w*order|withdraw|transfer|"
    r"fetch_balance|fetch_my_trades|fetch_\w*orders?|fetch_positions|set_leverage)\s*\(",
    re.IGNORECASE,
)
# Credential fields passed to exchange clients or read from config.
FORBIDDEN_CREDENTIALS = re.compile(
    r"""(apiKey|api_key|secret|password|privateKey|private_key)\s*["']?\s*[:=]""",
    re.IGNORECASE,
)


def _python_sources():
    files = sorted(SRC_DIR.rglob("*.py"))
    assert files, f"no sources found under {SRC_DIR}"
    return files


def test_no_private_exchange_calls():
    offenders = [
        f"{path.relative_to(SRC_DIR)}:{lineno}: {line.strip()}"
        for path in _python_sources()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if FORBIDDEN_CALLS.search(line)
    ]
    assert not offenders, "private/trading exchange calls found:\n" + "\n".join(offenders)


def test_no_credentials_in_source():
    offenders = [
        f"{path.relative_to(SRC_DIR)}:{lineno}: {line.strip()}"
        for path in _python_sources()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if FORBIDDEN_CREDENTIALS.search(line)
    ]
    assert not offenders, "credential fields found:\n" + "\n".join(offenders)


def test_scanner_actually_detects_violations():
    assert FORBIDDEN_CALLS.search("exchange.create_market_buy_order('BTC/USDT', 1)")
    assert FORBIDDEN_CALLS.search("ex.fetch_balance()")
    assert FORBIDDEN_CREDENTIALS.search("ccxt.binance({'apiKey': key})")
    assert not FORBIDDEN_CALLS.search("provider.fetch_ohlcv('BTC/USDT', '1h')")
