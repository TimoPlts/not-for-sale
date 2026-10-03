"""Portfolio accounting tests: cash, cost basis, realized/unrealized PnL, invariants."""

import random

import pytest

from conftest import ts
from trading_lab.core.errors import InsufficientFundsError, MissingPriceError, PositionError
from trading_lab.core.models import Fill, Side
from trading_lab.portfolio.portfolio import Portfolio


def fill(side, qty, price, fee, symbol="BTC/USDT", hours=0, stop=None, ref=None):
    return Fill("o", symbol, side, qty, ref or price, price, fee, ts(hours), stop_price=stop)


def assert_invariant(p: Portfolio, prices):
    assert p.equity(prices) == pytest.approx(
        p.initial_cash + p.realized_pnl + p.unrealized_pnl(prices), rel=1e-12, abs=1e-9
    )


def test_initial_state(portfolio):
    assert portfolio.cash == 10_000.0
    assert portfolio.equity({}) == 10_000.0
    assert portfolio.realized_pnl == 0.0
    assert portfolio.fees_paid == 0.0
    assert not portfolio.positions


@pytest.mark.parametrize("cash", [0, -1, float("nan"), float("inf")])
def test_initial_cash_must_be_positive(cash):
    with pytest.raises(ValueError):
        Portfolio(cash)


def test_buy_debits_notional_plus_fee_and_capitalises_fee(portfolio):
    portfolio.apply_fill(fill(Side.BUY, 10, 100.1, 1.001, stop=95.0))
    pos = portfolio.position("BTC/USDT")
    assert portfolio.cash == pytest.approx(10_000 - 1001 - 1.001)
    assert pos.quantity == 10
    assert pos.cost_basis == pytest.approx(1002.001)
    assert pos.avg_entry_price == pytest.approx(100.2001)
    assert pos.stop_price == 95.0
    assert portfolio.fees_paid == pytest.approx(1.001)
    # Marked at the fill price, unrealized PnL equals minus the buy fee.
    assert portfolio.unrealized_pnl({"BTC/USDT": 100.1}) == pytest.approx(-1.001)


def test_round_trip_realized_pnl_includes_both_fees(portfolio):
    portfolio.apply_fill(fill(Side.BUY, 10, 100.1, 1.001))
    portfolio.apply_fill(fill(Side.SELL, 10, 109.89, 1.0989, hours=5))
    # proceeds 1098.9 - 1.0989 = 1097.8011; basis 1002.001 -> pnl 95.8001
    assert portfolio.realized_pnl == pytest.approx(95.8001)
    assert portfolio.cash == pytest.approx(10_095.8001)
    assert portfolio.fees_paid == pytest.approx(1.001 + 1.0989)
    assert portfolio.position("BTC/USDT") is None
    (trade,) = portfolio.closed_trades
    assert trade.pnl == pytest.approx(95.8001)
    assert trade.is_win
    assert trade.return_pct == pytest.approx(95.8001 / 1002.001)
    assert trade.opened_at == ts(0) and trade.closed_at == ts(5)


def test_partial_sell_releases_basis_pro_rata(portfolio):
    portfolio.apply_fill(fill(Side.BUY, 4, 100.0, 0.4))  # basis 400.4
    portfolio.apply_fill(fill(Side.SELL, 1, 90.0, 0.09))
    pos = portfolio.position("BTC/USDT")
    assert pos.quantity == pytest.approx(3)
    assert pos.cost_basis == pytest.approx(400.4 * 3 / 4)
    assert portfolio.realized_pnl == pytest.approx((90 - 0.09) - 100.1)
    assert not portfolio.closed_trades[0].is_win
    # Average entry is unchanged by a partial exit.
    assert pos.avg_entry_price == pytest.approx(100.1)


def test_adding_to_position_averages_cost(portfolio):
    portfolio.apply_fill(fill(Side.BUY, 1, 100.0, 0.0))
    portfolio.apply_fill(fill(Side.BUY, 1, 200.0, 0.0, hours=1))
    pos = portfolio.position("BTC/USDT")
    assert pos.quantity == 2
    assert pos.avg_entry_price == pytest.approx(150.0)
    assert pos.opened_at == ts(0)


def test_overdraft_is_rejected_without_mutation(portfolio):
    with pytest.raises(InsufficientFundsError):
        portfolio.apply_fill(fill(Side.BUY, 100, 100.0, 10.0))
    assert portfolio.cash == 10_000.0
    assert not portfolio.positions
    assert not portfolio.fills


def test_long_only_sell_rules(portfolio):
    with pytest.raises(PositionError, match="long-only"):
        portfolio.apply_fill(fill(Side.SELL, 1, 100.0, 0.0))
    portfolio.apply_fill(fill(Side.BUY, 1, 100.0, 0.0))
    with pytest.raises(PositionError, match="long-only"):
        portfolio.apply_fill(fill(Side.SELL, 1.5, 100.0, 0.0))
    assert portfolio.position("BTC/USDT").quantity == 1


def test_sell_with_float_dust_closes_position(portfolio):
    portfolio.apply_fill(fill(Side.BUY, 0.1 + 0.2, 100.0, 0.0))
    portfolio.apply_fill(fill(Side.SELL, 0.3, 100.0, 0.0))
    assert portfolio.position("BTC/USDT") is None
    assert portfolio.realized_pnl == pytest.approx(0.0, abs=1e-9)


def test_missing_mark_price_raises(portfolio):
    portfolio.apply_fill(fill(Side.BUY, 1, 100.0, 0.0))
    with pytest.raises(MissingPriceError):
        portfolio.equity({})


def test_snapshot(portfolio):
    portfolio.apply_fill(fill(Side.BUY, 2, 100.0, 0.2))
    snap = portfolio.snapshot({"BTC/USDT": 110.0}, ts(1))
    assert snap.cash == pytest.approx(10_000 - 200.2)
    assert snap.positions_value == pytest.approx(220.0)
    assert snap.equity == pytest.approx(10_000 - 200.2 + 220)
    assert snap.unrealized_pnl == pytest.approx(220 - 200.2)
    assert snap.open_positions == 1


def test_accounting_invariant_holds_over_random_fill_sequence():
    """equity == initial + realized + unrealized after every fill (seeded, reproducible)."""
    rng = random.Random(42)
    p = Portfolio(10_000.0)
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT"]
    prices = {"BTC/USDT": 40_000.0, "ETH/USDT": 2_000.0, "SOL/USDT": 100.0, "DOGE/USDT": 0.08}
    for step in range(500):
        for s in symbols:
            prices[s] *= 1 + rng.uniform(-0.02, 0.02)
        sym = rng.choice(symbols)
        pos = p.position(sym)
        if pos is not None and rng.random() < 0.5:
            qty = pos.quantity * rng.choice([0.25, 0.5, 1.0])
            p.apply_fill(fill(Side.SELL, qty, prices[sym], qty * prices[sym] * 0.001, sym, step))
        else:
            budget = p.cash * rng.uniform(0.0, 0.3)
            if budget > 1:
                qty = budget / (prices[sym] * 1.001)
                p.apply_fill(fill(Side.BUY, qty, prices[sym], qty * prices[sym] * 0.001, sym, step))
        assert p.cash >= -1e-9
        assert_invariant(p, prices)
    assert p.fees_paid == pytest.approx(sum(f.fee for f in p.fills))
    assert p.realized_pnl == pytest.approx(sum(t.pnl for t in p.closed_trades))
