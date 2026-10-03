"""Paper execution tests: slippage, fees, rejections, determinism."""

import pytest

from conftest import buy, make_costs, sell, ts
from trading_lab.core.models import Order, OrderStatus, Side
from trading_lab.execution.costs import FixedBpsSlippage, PercentageFeeModel
from trading_lab.execution.paper import PaperExecutor
from trading_lab.portfolio.portfolio import Portfolio


def test_cost_models():
    slip = FixedBpsSlippage(10)
    assert slip.fill_price(Side.BUY, 100.0) == pytest.approx(100.1)
    assert slip.fill_price(Side.SELL, 100.0) == pytest.approx(99.9)
    fees = PercentageFeeModel(0.001)
    assert fees.fee(1000.0) == pytest.approx(1.0)
    n = fees.max_notional(1001.0)
    assert n + fees.fee(n) == pytest.approx(1001.0)
    with pytest.raises(ValueError):
        PercentageFeeModel(-0.1)
    with pytest.raises(ValueError):
        FixedBpsSlippage(float("nan"))


def test_buy_fill_applies_adverse_slippage_and_fee():
    p = Portfolio(10_000.0)
    ex = PaperExecutor(p, make_costs(fee_rate=0.001, slippage_bps=10))
    report = ex.submit(buy("BTC/USDT", 10, stop_price=95.0), reference_price=100.0)
    assert report.filled and report.status is OrderStatus.FILLED
    f = report.fill
    assert f.fill_price == pytest.approx(100.1)
    assert f.fee == pytest.approx(1.001)
    assert f.slippage_cost == pytest.approx(1.0)
    assert f.stop_price == 95.0
    assert p.cash == pytest.approx(10_000 - 1002.001)
    assert p.position("BTC/USDT").stop_price == 95.0


def test_sell_fill_applies_adverse_slippage_and_fee():
    p = Portfolio(10_000.0)
    ex = PaperExecutor(p, make_costs(fee_rate=0.001, slippage_bps=10))
    ex.submit(buy("BTC/USDT", 10), 100.0)
    report = ex.submit(sell("BTC/USDT", 10, hours=1), 110.0)
    assert report.fill.fill_price == pytest.approx(109.89)
    assert report.fill.fee == pytest.approx(1.0989)
    assert p.realized_pnl == pytest.approx(95.8001)
    assert p.cash == pytest.approx(10_095.8001)


def test_round_trip_at_flat_price_loses_exactly_costs():
    p = Portfolio(10_000.0)
    ex = PaperExecutor(p, make_costs(fee_rate=0.001, slippage_bps=5))
    b = ex.submit(buy("ETH/USDT", 2), 2000.0).fill
    s = ex.submit(sell("ETH/USDT", 2, hours=1), 2000.0).fill
    expected_loss = b.fee + s.fee + b.slippage_cost + s.slippage_cost
    assert p.realized_pnl == pytest.approx(-expected_loss)
    assert p.cash == pytest.approx(10_000 - expected_loss)


def test_zero_cost_model_is_frictionless():
    p = Portfolio(10_000.0)
    ex = PaperExecutor(p, make_costs(fee_rate=0.0, slippage_bps=0.0))
    ex.submit(buy("SOL/USDT", 10), 100.0)
    ex.submit(sell("SOL/USDT", 10), 100.0)
    assert p.cash == pytest.approx(10_000.0)
    assert p.fees_paid == 0.0


def test_reject_insufficient_cash_leaves_portfolio_untouched(executor, portfolio):
    report = executor.submit(buy("BTC/USDT", 1), 40_000.0)
    assert report.status is OrderStatus.REJECTED
    assert "insufficient cash" in report.reason
    assert report.fill is None
    assert portfolio.cash == 10_000.0 and not portfolio.positions and not portfolio.fills


def test_long_only_rejections(executor):
    r1 = executor.submit(sell("BTC/USDT", 1), 100.0)
    assert not r1.filled and "long-only" in r1.reason
    executor.submit(buy("BTC/USDT", 1), 100.0)
    r2 = executor.submit(sell("BTC/USDT", 2), 100.0)
    assert not r2.filled and "exceeds position" in r2.reason
    assert executor.portfolio.position("BTC/USDT").quantity == 1


def test_sell_slightly_over_position_due_to_float_dust_closes_it(executor):
    executor.submit(buy("BTC/USDT", 0.3), 100.0)
    report = executor.submit(sell("BTC/USDT", 0.1 + 0.2), 100.0)  # 0.30000000000000004
    assert report.filled
    assert report.fill.quantity == 0.3
    assert executor.portfolio.position("BTC/USDT") is None


def test_min_notional_rejects_small_orders_but_allows_full_exit(executor, portfolio):
    small = executor.submit(buy("DOGE/USDT", 50), 0.08)  # ~$4
    assert not small.filled and "below minimum" in small.reason
    executor.submit(buy("DOGE/USDT", 200), 0.08)  # ~$16
    partial = executor.submit(sell("DOGE/USDT", 100), 0.08)  # ~$8 partial
    assert not partial.filled
    full = executor.submit(sell("DOGE/USDT", 200), 0.08)  # ~$16 is fine anyway
    assert full.filled
    # Price collapses: the full exit is below min notional but still allowed.
    executor.submit(buy("DOGE/USDT", 200), 0.08)
    dust_exit = executor.submit(sell("DOGE/USDT", 200), 0.01)
    assert dust_exit.filled
    assert portfolio.position("DOGE/USDT") is None


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_reference_price_rejected(executor, price):
    report = executor.submit(buy("BTC/USDT", 1), price)
    assert not report.filled and "invalid reference price" in report.reason


def test_fill_timestamp_comes_from_order_not_wall_clock(executor):
    report = executor.submit(Order("BTC/USDT", Side.BUY, 0.01, ts(7)), 30_000.0)
    assert report.fill.timestamp == ts(7)


def _run_script():
    p = Portfolio(10_000.0)
    ex = PaperExecutor(p, make_costs())
    reports = [
        ex.submit(buy("BTC/USDT", 0.05, 0), 42_000.0),
        ex.submit(buy("ETH/USDT", 1.0, 1), 2_300.0),
        ex.submit(sell("BTC/USDT", 0.02, 2), 43_100.0),
        ex.submit(sell("SOL/USDT", 1.0, 3), 100.0),  # rejected
        ex.submit(sell("ETH/USDT", 1.0, 4), 2_250.0),
    ]
    return p, reports


def test_execution_is_deterministic_and_ids_are_sequential():
    p1, r1 = _run_script()
    p2, r2 = _run_script()
    assert [r.order_id for r in r1] == [f"paper-{i:06d}" for i in range(1, 6)]
    assert r1 == r2
    assert p1.fills == p2.fills
    assert p1.cash == p2.cash
    assert [r.status for r in r1].count(OrderStatus.REJECTED) == 1
