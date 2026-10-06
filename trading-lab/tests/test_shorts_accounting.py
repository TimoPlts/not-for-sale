"""Stage 16A: short-position accounting (portfolio, executor, storage). Off unless allowed."""

import random
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.core.errors import InsufficientFundsError, PositionError
from trading_lab.core.models import SHORT, ClosedTrade, Fill, Order, OrderType, Side
from trading_lab.execution import CostModel, PaperExecutor
from trading_lab.execution.costs import FixedBpsSlippage, PercentageFeeModel
from trading_lab.portfolio import Portfolio
from trading_lab.storage import SCHEMA_VERSION, SQLiteStore

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def fill(side, qty, price, fee=0.0, at=0, order_id="o", stop=None):
    return Fill(order_id, "BTC/USDT", side, qty, price, price, fee, T0 + timedelta(hours=at), stop_price=stop)


def check_invariant(p, price):
    marks = {s: price for s in p.positions}
    assert p.equity(marks) == pytest.approx(p.initial_cash + p.realized_pnl + p.unrealized_pnl(marks), abs=1e-9)


# ------------------------------------------------------------------ portfolio
def test_long_only_unless_allowed():
    p = Portfolio(10_000)
    with pytest.raises(PositionError, match="long-only"):
        p.apply_fill(fill(Side.SELL, 1, 100))
    assert not p.allow_short and p.cash == 10_000 and not p.positions


def test_a_short_round_trip():
    p = Portfolio(10_000, allow_short=True)
    p.apply_fill(fill(Side.SELL, 10, 100, fee=1.0, stop=110))
    pos = p.position("BTC/USDT")
    assert pos.is_short and pos.side == SHORT and pos.stop_price == 110
    assert (pos.quantity, pos.entry_notional, pos.cost_basis) == (10, 1000, 1001)
    assert p.cash == 10_000 - 1001  # collateral + fee leave cash: no leverage
    assert pos.avg_entry_price == pytest.approx(99.9)  # break-even after the entry fee
    assert pos.market_value(90) == pytest.approx(1100) and pos.unrealized_pnl(90) == pytest.approx(99)
    assert pos.exposure(90) == 900 and p.gross_exposure({"BTC/USDT": 90}) == 900
    check_invariant(p, 90)

    p.apply_fill(fill(Side.BUY, 10, 90, fee=0.9, at=5, order_id="c"))
    (trade,) = p.closed_trades
    assert trade.side == SHORT and trade.exit_order_id == "c"
    assert trade.pnl == pytest.approx(10 * (100 - 90) - 1.0 - 0.9)
    assert trade.proceeds == pytest.approx(2 * 1000 - 900 - 0.9) and trade.cost_basis == 1001
    assert trade.entry_price == pytest.approx(99.9) and trade.exit_price == 90
    assert trade.return_pct == pytest.approx(trade.pnl / 1001)
    assert p.cash == pytest.approx(10_000 + trade.pnl) and not p.positions
    assert p.fees_paid == pytest.approx(1.9)


def test_partial_covers_and_adding_to_a_short():
    p = Portfolio(10_000, allow_short=True)
    p.apply_fill(fill(Side.SELL, 4, 100, fee=0.4))
    p.apply_fill(fill(Side.SELL, 6, 110, fee=0.66, at=1))  # add: entry notional 400 + 660
    pos = p.position("BTC/USDT")
    assert pos.quantity == 10 and pos.entry_notional == pytest.approx(1060) and pos.opened_at == T0
    p.apply_fill(fill(Side.BUY, 5, 105, at=2))
    assert p.position("BTC/USDT").quantity == 5
    assert p.closed_trades[0].pnl == pytest.approx(0.5 * 1060 - 5 * 105 - 0.5 * 1.06)
    p.apply_fill(fill(Side.BUY, 5, 105, at=3))
    assert sum(t.pnl for t in p.closed_trades) == pytest.approx(1060 - 1050 - 1.06)
    assert not p.positions and p.cash == pytest.approx(10_000 + 1060 - 1050 - 1.06)


def test_losses_beyond_the_collateral():
    p = Portfolio(10_000, allow_short=True)
    p.apply_fill(fill(Side.SELL, 10, 100))
    check_invariant(p, 250)
    assert p.position("BTC/USDT").market_value(250) == -500  # owes more than it posted
    p.apply_fill(fill(Side.BUY, 10, 250, at=1))
    assert p.closed_trades[0].pnl == pytest.approx(-1500) and p.cash == pytest.approx(8_500)


def test_no_fill_flips_a_position():
    p = Portfolio(10_000, allow_short=True)
    p.apply_fill(fill(Side.SELL, 1, 100))
    with pytest.raises(PositionError, match="never turns a short into a long"):
        p.apply_fill(fill(Side.BUY, 2, 100))
    p.apply_fill(fill(Side.BUY, 1, 100))
    p.apply_fill(fill(Side.BUY, 1, 100))  # a long now
    with pytest.raises(PositionError, match="long-only: cannot sell"):
        p.apply_fill(fill(Side.SELL, 2, 100))
    assert p.position("BTC/USDT").quantity == 1 and not p.position("BTC/USDT").is_short


def test_shorts_need_collateral():
    p = Portfolio(1_000, allow_short=True)
    with pytest.raises(InsufficientFundsError, match="collateral"):
        p.apply_fill(fill(Side.SELL, 20, 100))
    assert p.cash == 1_000 and not p.positions and not p.fills


def test_replaying_fills_rebuilds_the_same_portfolio():
    rng = random.Random(4)
    p = Portfolio(50_000, allow_short=True)
    price, fills = 100.0, []
    for i in range(400):
        price *= 1 + rng.gauss(0, 0.02)
        pos = p.position("BTC/USDT")
        if pos is None:
            side = rng.choice([Side.BUY, Side.SELL])
            qty = rng.uniform(1, 50)
        else:
            side = Side.BUY if pos.is_short else Side.SELL
            qty = pos.quantity * rng.choice([0.3, 1.0]) if rng.random() < 0.5 else rng.uniform(1, 5)
            side = side if rng.random() < 0.6 else (Side.SELL if pos.is_short else Side.BUY)  # add to it
        f = fill(side, qty, price, fee=qty * price * 0.001, at=i, order_id=f"o{i}")
        try:
            p.apply_fill(f)
        except (PositionError, InsufficientFundsError):
            continue
        fills.append(f)
        check_invariant(p, price)
    assert any(t.side == SHORT for t in p.closed_trades) and any(t.side == "long" for t in p.closed_trades)
    again = Portfolio(50_000, allow_short=True)
    for f in fills:
        again.apply_fill(f)
    assert again.closed_trades == p.closed_trades and again.cash == p.cash and again.positions == p.positions


# ------------------------------------------------------------------- executor
def executor(cash=10_000, allow=True, borrow=0.0, slippage_bps=10.0, min_notional=10.0):
    costs = CostModel(PercentageFeeModel(0.001), FixedBpsSlippage(slippage_bps), PercentageFeeModel(0.0005))
    return PaperExecutor(Portfolio(cash, allow_short=allow), costs, min_notional=min_notional,
                         borrow_bps_per_day=borrow)


def order(side, qty, at=0, **kw):
    return Order("BTC/USDT", side, qty, T0 + timedelta(hours=at), **kw)


def test_executor_is_long_only_by_default():
    ex = executor(allow=False)
    report = ex.submit(order(Side.SELL, 1), 100)
    assert not report.filled and report.reason == "long-only: no open position to sell"


def test_executor_opens_and_covers_shorts_with_slippage_and_borrow():
    ex = executor(borrow=5.0)
    opened = ex.submit(order(Side.SELL, 10, stop_price=110), 100)
    assert opened.filled and opened.fill.fill_price == pytest.approx(99.9)  # a seller gets less
    pos = ex.portfolio.position("BTC/USDT")
    assert pos.is_short and pos.entry_notional == pytest.approx(999) and pos.stop_price == 110
    covered = ex.submit(order(Side.BUY, 10, at=48), 90)  # two days later
    assert covered.fill.fill_price == pytest.approx(90.09)  # a buyer pays more
    borrow = 999 * 5e-4 * 2
    assert covered.fill.fee == pytest.approx(900.9 * 0.001 + borrow)
    trade = ex.portfolio.closed_trades[0]
    assert trade.pnl == pytest.approx(999 - 900.9 - 0.999 - 0.9009 - borrow)
    assert ex.borrow_fee(pos, 10, pos.opened_at) == 0.0


def test_executor_limits_on_shorts():
    ex = executor(cash=500)
    assert "insufficient cash" in ex.submit(order(Side.SELL, 10), 100).reason  # 1000 collateral needed
    ex = executor()
    ex.submit(order(Side.SELL, 1), 100)
    assert "cover quantity 2.0 exceeds position 1.0" in ex.submit(order(Side.BUY, 2), 100).reason
    assert ex.submit(order(Side.BUY, 1 + 1e-13), 100).filled  # float dust is absorbed
    ex = executor(min_notional=50)
    ex.submit(order(Side.SELL, 1), 100)
    ex.submit(order(Side.BUY, 0.6), 100)
    assert ex.submit(order(Side.BUY, 0.4), 100).filled  # closing dust below the minimum is allowed
    limit = executor().submit(order(Side.SELL, 1, order_type=OrderType.LIMIT, limit_price=101), 100)
    assert limit.filled and limit.fill.fill_price == 101 and limit.fill.fee == pytest.approx(101 * 0.0005)
    with pytest.raises(ValueError):
        executor(borrow=-1)


# -------------------------------------------------------------------- storage
def test_trades_keep_their_side(tmp_path):
    trades = [ClosedTrade("BTC/USDT", 1, 100, 90, 101, 109, 8, T0, T0, "o1", side=SHORT),
              ClosedTrade("ETH/USDT", 1, 100, 110, 100, 110, 10, T0, T0, "o2")]
    with SQLiteStore(tmp_path / "s.db") as store:
        store.create_run("r", kind="backtest", timeframe="1h", symbols=["BTC/USDT"], exchange="x",
                         config={}, config_fingerprint="f")
        store.add_closed_trades("r", trades)
        assert store.load_closed_trades("r") == trades and store.schema_version == SCHEMA_VERSION == 4


def test_v3_databases_upgrade_and_old_trades_are_longs(tmp_path):
    path = tmp_path / "old.db"
    with SQLiteStore(path) as store:
        store.create_run("r", kind="backtest", timeframe="1h", symbols=["BTC/USDT"], exchange="x",
                         config={}, config_fingerprint="f")
        store.add_closed_trades("r", [ClosedTrade("BTC/USDT", 1, 100, 110, 100, 110, 10, T0, T0, "o")])
    conn = sqlite3.connect(path)
    conn.executescript("ALTER TABLE closed_trades DROP COLUMN side; PRAGMA user_version = 3;")
    conn.close()
    with SQLiteStore(path, readonly=True) as store:  # read-only: not upgraded, still readable
        assert store.schema_version == 3 and store.load_closed_trades("r")[0].side == "long"
    with SQLiteStore(path) as store:
        assert store.schema_version == 4 and store.load_closed_trades("r")[0].side == "long"
