from datetime import datetime

import pytest

from conftest import ts
from trading_lab.core.models import Direction, ExecutionReport, Fill, Order, OrderStatus, Side, Signal


def test_signal_is_standardized_and_immutable():
    sig = Signal("rsi", "BTC/USDT", "buy", 0.7, ts(), {"rsi": 25.0})
    assert sig.direction is Direction.BUY
    assert sig.confidence == 0.7
    assert sig.metadata["rsi"] == 25.0
    with pytest.raises(TypeError):
        sig.metadata["rsi"] = 1.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        sig.confidence = 0.1  # type: ignore[misc]


def test_signal_metadata_is_copied_from_caller():
    meta = {"k": 1}
    sig = Signal("x", "BTC/USDT", Direction.HOLD, 0.0, ts(), meta)
    meta["k"] = 2
    assert sig.metadata["k"] == 1


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_signal_rejects_out_of_range_confidence(confidence):
    with pytest.raises(ValueError):
        Signal("rsi", "BTC/USDT", Direction.BUY, confidence, ts())


def test_signal_rejects_naive_timestamp_and_bad_direction():
    with pytest.raises(ValueError, match="timezone-aware"):
        Signal("rsi", "BTC/USDT", Direction.BUY, 0.5, datetime(2024, 1, 1))
    with pytest.raises(ValueError):
        Signal("rsi", "BTC/USDT", "short", 0.5, ts())


def test_hold_helper():
    sig = Signal.hold("macd", "ETH/USDT", ts(), {"why": "warmup"})
    assert sig.direction is Direction.HOLD and sig.confidence == 0.0


@pytest.mark.parametrize("qty", [0, -1, float("nan"), float("inf")])
def test_order_requires_positive_quantity(qty):
    with pytest.raises(ValueError):
        Order("BTC/USDT", Side.BUY, qty, ts())


def test_fill_derived_values():
    fill = Fill("o1", "BTC/USDT", Side.BUY, 2.0, 100.0, 100.1, 0.2002, ts())
    assert fill.notional == pytest.approx(200.2)
    assert fill.slippage_cost == pytest.approx(0.2)
    assert fill.cash_delta == pytest.approx(-(200.2 + 0.2002))


def test_execution_report_consistency():
    order = Order("BTC/USDT", Side.BUY, 1.0, ts())
    with pytest.raises(ValueError):
        ExecutionReport("o1", order, OrderStatus.FILLED)
