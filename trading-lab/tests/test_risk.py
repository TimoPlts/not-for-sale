"""Risk manager tests: sizing limits, rejections, stop handling."""

from dataclasses import replace

import pytest

from conftest import buy, make_costs, sell, ts
from trading_lab.config import RiskConfig
from trading_lab.core.models import Side
from trading_lab.execution.paper import PaperExecutor
from trading_lab.portfolio.portfolio import Portfolio
from trading_lab.risk.manager import RiskManager

DEFAULT_RISK = RiskConfig()


def setup(risk: RiskConfig = DEFAULT_RISK, cash: float = 10_000.0, fee=0.001, bps=5.0):
    costs = make_costs(fee, bps)
    p = Portfolio(cash)
    return p, PaperExecutor(p, costs, min_notional=10.0), RiskManager(risk, costs, min_notional=10.0)


def test_risk_per_trade_bounds_loss_at_stop_including_costs():
    p, ex, rm = setup()
    decision = rm.evaluate_entry("BTC/USDT", 100.0, p, {})
    assert decision.approved
    assert decision.sizing["binding_limit"] == "risk_per_trade"
    fill = ex.submit(decision.to_order(ts(0)), 100.0).fill
    assert decision.stop_price == pytest.approx(fill.fill_price * 0.95)
    # Exit exactly at the stop: total loss must equal 1% of starting equity.
    ex.submit(sell("BTC/USDT", fill.quantity, 1), decision.stop_price)
    assert p.realized_pnl == pytest.approx(-100.0, rel=1e-9)


def test_max_position_size_caps_entry():
    p, ex, rm = setup(replace(DEFAULT_RISK, stop_loss_pct=0.005))  # tight stop -> huge risk qty
    decision = rm.evaluate_entry("ETH/USDT", 2000.0, p, {})
    assert decision.approved
    assert decision.sizing["binding_limit"] == "max_position_size"
    fill = ex.submit(decision.to_order(ts()), 2000.0).fill
    assert fill.notional == pytest.approx(0.25 * 10_000)


def test_available_cash_caps_entry_and_order_is_affordable():
    risk = replace(
        DEFAULT_RISK, max_position_pct=1.0, risk_per_trade_pct=1.0, stop_loss_pct=0.5
    )
    p, ex, rm = setup(risk)
    decision = rm.evaluate_entry("SOL/USDT", 100.0, p, {})
    assert decision.sizing["binding_limit"] == "available_cash"
    report = ex.submit(decision.to_order(ts()), 100.0)
    assert report.filled
    assert 0.0 <= p.cash < 1e-3


def test_total_exposure_limit():
    risk = replace(DEFAULT_RISK, max_total_exposure_pct=0.5, stop_loss_pct=0.005)
    p, ex, rm = setup(risk)
    ex.submit(buy("BTC/USDT", 0.1), 40_000.0)  # ~$4,000 exposure
    prices = {"BTC/USDT": 40_000.0}
    equity_before = p.equity(prices)
    decision = rm.evaluate_entry("ETH/USDT", 2000.0, p, prices)
    assert decision.approved
    assert decision.sizing["binding_limit"] == "max_total_exposure"
    ex.submit(decision.to_order(ts(1)), 2000.0)
    # Exposure is capped in fill-notional terms against equity at decision time.
    assert p.positions_value({**prices, "ETH/USDT": 2000.0}) <= 0.5 * equity_before + 1e-6


def test_exposure_fully_used_rejects():
    risk = replace(DEFAULT_RISK, max_total_exposure_pct=0.3)
    p, ex, rm = setup(risk)
    ex.submit(buy("BTC/USDT", 0.075), 40_000.0)  # ~$3,000 = 30%
    decision = rm.evaluate_entry("ETH/USDT", 2000.0, p, {"BTC/USDT": 40_000.0})
    assert not decision.approved
    assert "below minimum notional" in decision.reason
    assert "max_total_exposure" in decision.reason


def test_max_open_positions():
    p, ex, rm = setup(replace(DEFAULT_RISK, max_open_positions=2))
    ex.submit(buy("BTC/USDT", 0.01), 40_000.0)
    ex.submit(buy("ETH/USDT", 0.2), 2_000.0)
    prices = {"BTC/USDT": 40_000.0, "ETH/USDT": 2_000.0}
    decision = rm.evaluate_entry("SOL/USDT", 100.0, p, prices)
    assert not decision.approved and "max open positions" in decision.reason


def test_pyramiding_disabled_by_default():
    p, ex, rm = setup()
    ex.submit(buy("BTC/USDT", 0.01), 40_000.0)
    decision = rm.evaluate_entry("BTC/USDT", 40_000.0, p, {"BTC/USDT": 40_000.0})
    assert not decision.approved and "pyramiding" in decision.reason


def test_pyramiding_respects_remaining_position_room():
    p, ex, rm = setup(replace(DEFAULT_RISK, allow_pyramiding=True, stop_loss_pct=0.005))
    ex.submit(buy("BTC/USDT", 0.05), 40_000.0)  # ~$2,000 of the $2,500 cap
    decision = rm.evaluate_entry("BTC/USDT", 40_000.0, p, {"BTC/USDT": 40_000.0})
    assert decision.approved
    assert decision.sizing["binding_limit"] == "max_position_size"
    ex.submit(decision.to_order(ts(1)), 40_000.0)
    value = p.position("BTC/USDT").market_value(40_000.0)
    assert value <= 0.25 * p.equity({"BTC/USDT": 40_000.0}) + 1.0


def test_tiny_account_rejected_by_min_notional():
    p, _, rm = setup(cash=50.0)
    decision = rm.evaluate_entry("BTC/USDT", 40_000.0, p, {})
    assert not decision.approved and "minimum notional" in decision.reason
    with pytest.raises(ValueError):
        decision.to_order(ts())


def test_invalid_price_rejected():
    p, _, rm = setup()
    assert not rm.evaluate_entry("BTC/USDT", float("nan"), p, {}).approved


def test_exit_decisions():
    p, ex, rm = setup()
    assert not rm.evaluate_exit("BTC/USDT", p).approved
    ex.submit(buy("BTC/USDT", 0.01), 40_000.0)
    decision = rm.evaluate_exit("BTC/USDT", p, reason="strategy sell")
    assert decision.approved and decision.side is Side.SELL
    assert decision.quantity == p.position("BTC/USDT").quantity
    assert ex.submit(decision.to_order(ts(1)), 41_000.0).filled
    assert p.position("BTC/USDT") is None


def test_stop_triggered_uses_position_stop():
    p, ex, rm = setup()
    decision = rm.evaluate_entry("BTC/USDT", 100.0, p, {})
    ex.submit(decision.to_order(ts()), 100.0)
    pos = p.position("BTC/USDT")
    assert pos.stop_price == pytest.approx(decision.stop_price)
    assert not rm.stop_triggered(pos, pos.stop_price + 0.01)
    assert rm.stop_triggered(pos, pos.stop_price)
    assert rm.stop_triggered(pos, pos.stop_price - 5)
