"""Stage 8 tests: volume-aware slippage, liquidity cap, limit orders with partial fills."""

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from conftest import ts
from test_backtest import BUY, SELL, H, T0, config, run_scripted
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import DecisionAction, Order, OrderType, Side
from trading_lab.data import normalize_ohlcv
from trading_lab.execution import CostModel, PaperExecutor
from trading_lab.execution.costs import (
    FixedBpsSlippage,
    MarketStats,
    PercentageFeeModel,
    VolumeImpactSlippage,
    market_stats_frame,
    next_bar_stats,
)
from trading_lab.portfolio import Portfolio


def frame(rows, start=T0):
    """[(open, high, low, close, volume), ...] hourly bars."""
    index = pd.date_range(start, periods=len(rows), freq="1h", name="timestamp")
    return normalize_ohlcv(pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=index))


def actions(result, *kinds):
    return [d for d in result.decisions if d.action in kinds]


# -------------------------------------------------------------- cost models
def test_volume_impact_slippage_formula():
    model = VolumeImpactSlippage(base_bps=5.0, coefficient=1.0)
    stats = MarketStats(volatility=0.01, avg_quote_volume=1_000_000.0)
    # 10_000 USDT order = 1% of bar volume -> impact = 1.0 * 0.01 * sqrt(0.01) = 0.001
    buy = model.fill_price(Side.BUY, 100.0, quantity=100.0, stats=stats)
    assert buy == pytest.approx(100.0 * (1 + 0.0005 + 0.001))
    sell = model.fill_price(Side.SELL, 100.0, quantity=100.0, stats=stats)
    assert sell == pytest.approx(100.0 * (1 - 0.0015))
    assert model.fill_price(Side.BUY, 100.0) == pytest.approx(100.05)  # no stats: base only
    small = model.fill_price(Side.BUY, 100.0, quantity=1.0, stats=stats)
    assert small < buy  # impact grows with size
    huge = model.fill_price(Side.BUY, 100.0, quantity=1e12, stats=stats)
    assert huge == pytest.approx(100.0 * (1 + 0.0005 + 0.5))  # capped
    assert FixedBpsSlippage(5).fill_price(Side.BUY, 100.0, 1e9, stats) == pytest.approx(100.05)


def test_market_stats_use_only_earlier_bars():
    rng = np.random.default_rng(1)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 40)))
    rows = [(c, c * 1.01, c * 0.99, c, float(v)) for c, v in zip(closes, rng.uniform(10, 20, 40))]
    candles = frame(rows)
    stats = market_stats_frame(candles, 5)
    assert stats.iloc[:5].isna().all().all()
    # Changing bar 30 must not change the stats used for a fill at bar 30's open.
    altered = candles.copy()
    altered.iloc[30, altered.columns.get_loc("volume")] *= 100
    assert market_stats_frame(altered, 5).iloc[30].equals(stats.iloc[30])
    window = candles.iloc[25:30]
    assert stats["avg_quote_volume"].iloc[30] == pytest.approx((window["close"] * window["volume"]).mean())
    nxt = next_bar_stats(candles.iloc[:30], 5)
    assert nxt.avg_quote_volume == pytest.approx(stats["avg_quote_volume"].iloc[30])
    assert nxt.volatility == pytest.approx(stats["volatility"].iloc[30])


def test_executor_limit_orders_fill_at_limit_with_maker_fee():
    costs = CostModel(PercentageFeeModel(0.001), FixedBpsSlippage(5), PercentageFeeModel(0.0002))
    portfolio = Portfolio(10_000.0)
    ex = PaperExecutor(portfolio, costs, min_notional=1.0)
    order = Order("BTC/USDT", Side.BUY, 2.0, ts(), order_type=OrderType.LIMIT, limit_price=99.5)
    fill = ex.submit(order, reference_price=123.0).fill  # reference is ignored for limits
    assert fill.fill_price == 99.5 and fill.reference_price == 99.5
    assert fill.fee == pytest.approx(2 * 99.5 * 0.0002)
    with pytest.raises(ValueError):
        Order("BTC/USDT", Side.BUY, 1.0, ts(), order_type=OrderType.LIMIT)
    with pytest.raises(ValueError):
        Order("BTC/USDT", Side.BUY, 1.0, ts(), limit_price=99.0)


# ------------------------------------------------------------ in backtests
def wiggly(n, volume, start_price=100.0):
    """Bars alternating +-1% so that volatility is non-zero."""
    rows, price = [], start_price
    for i in range(n):
        close = price * (1.01 if i % 2 == 0 else 1 / 1.01)
        rows.append((price, max(price, close) * 1.001, min(price, close) * 0.999, close, volume))
        price = close
    return rows


def test_backtest_applies_volume_impact_on_market_entries():
    rows = wiggly(12, volume=50.0)
    candles = frame(rows)
    cfg = config(slip=5.0, execution={"slippage_model": "volume", "impact_coefficient": 2.0,
                                      "volume_lookback": 4})
    r = run_scripted({"BTC/USDT": candles}, {("BTC/USDT", 8): BUY}, cfg)
    fill = r.fills[0]
    stats = market_stats_frame(candles, 4).iloc[9]
    open_price = candles["open"].iloc[9]
    impact = 2.0 * stats["volatility"] * math.sqrt(fill.quantity * open_price / stats["avg_quote_volume"])
    assert fill.fill_price == pytest.approx(open_price * (1 + 0.0005 + impact), rel=1e-9)
    assert impact > 0.001  # thin market: a ~$2k order on ~$5k bars costs real money
    entry = actions(r, DecisionAction.ENTER)[0]
    assert entry.details["impact_bps"] == pytest.approx((0.0005 + impact) * 1e4, rel=1e-6)
    # The stop is placed relative to the actual (impacted) fill price.
    assert entry.stop_price == pytest.approx(fill.fill_price * 0.95)


def test_liquidity_cap_limits_order_size():
    candles = frame(wiggly(12, volume=50.0))
    cfg = config(execution={"max_participation_pct": 0.05, "volume_lookback": 4})
    r = run_scripted({"BTC/USDT": candles}, {("BTC/USDT", 8): BUY}, cfg)
    entry = actions(r, DecisionAction.ENTER)[0]
    assert entry.details["binding_limit"] == "liquidity"
    avg_quote = market_stats_frame(candles, 4)["avg_quote_volume"].iloc[9]
    assert r.fills[0].notional == pytest.approx(0.05 * avg_quote, rel=1e-9)


LIMIT = {"entry_order_type": "limit", "limit_offset_bps": 10.0, "limit_ttl_bars": 3, "maker_fee_rate": 0.0002}


def test_limit_entry_fills_at_limit_when_traded_through():
    rows = [(100, 101, 99.5, 100, 1e6)] * 2 + [(100, 101, 99.0, 100, 1e6)] * 3
    r = run_scripted({"BTC/USDT": frame(rows)}, {("BTC/USDT", 0): BUY}, config(fee=0.001, execution=LIMIT))
    placed = actions(r, DecisionAction.ORDER_PLACED)
    assert len(placed) == 1 and placed[0].reference_price == pytest.approx(99.9)
    (fill,) = r.fills
    assert fill.fill_price == pytest.approx(99.9) and fill.timestamp == T0 + H  # same bar: low 99.5 < 99.9
    assert fill.fee == pytest.approx(fill.notional * 0.0002)  # maker fee, not taker


def test_limit_order_needs_trade_through_and_expires():
    rows = [(100, 101, 99.95, 100, 1e6)] * 2 + [(100, 101, 99.9, 100, 1e6)] * 4  # touches 99.9, never below
    r = run_scripted({"BTC/USDT": frame(rows)}, {("BTC/USDT", 0): BUY}, config(execution=LIMIT))
    assert not r.fills
    (expired,) = actions(r, DecisionAction.EXPIRED)
    assert "expired" in expired.reason and expired.timestamp == T0 + 3 * H  # placed bar 1, ttl 3


def test_limit_partial_fills_capped_by_bar_volume():
    history = [(100, 101, 99.95, 100, 1000.0)] * 3          # deep market: sizing is not liquidity-bound
    thin = [(100, 101, 99.0, 100, 5.0)] * 4                 # trades through, but only 5 units per bar
    cfg = config(execution={**LIMIT, "max_participation_pct": 0.1, "volume_lookback": 2})
    r = run_scripted({"BTC/USDT": frame(history + thin)}, {("BTC/USDT", 2): BUY}, cfg)
    fills = r.fills
    assert len(fills) == 3 and all(f.quantity == pytest.approx(0.5) for f in fills)  # 10% of 5
    assert [f.timestamp for f in fills] == [T0 + 3 * H, T0 + 4 * H, T0 + 5 * H]
    entries = actions(r, DecisionAction.ENTER)
    assert all(e.details["partial"] for e in entries)
    (expired,) = actions(r, DecisionAction.EXPIRED)
    assert "filled 1.5 of" in expired.reason
    assert r.equity_curve["open_positions"].iloc[-1] == 1


def test_sell_signal_cancels_resting_limit():
    rows = [(100, 101, 99.95, 100, 1e6)] * 5
    script = {("BTC/USDT", 0): BUY, ("BTC/USDT", 1): SELL}
    r = run_scripted({"BTC/USDT": frame(rows)}, script, config(execution=LIMIT))
    (cancelled,) = actions(r, DecisionAction.EXPIRED)
    assert "cancelled by SELL signal" in cancelled.reason and cancelled.timestamp == T0 + H
    assert not r.fills


def test_default_config_keeps_market_orders_and_fixed_slippage():
    cfg = AppConfig()
    assert cfg.execution.entry_order_type == "market" and cfg.execution.slippage_model == "fixed"
    assert cfg.execution.max_participation_pct == 0.0


@pytest.mark.parametrize(
    "values",
    [{"slippage_model": "quadratic"}, {"entry_order_type": "stop"}, {"limit_ttl_bars": 0},
     {"volume_lookback": 1}, {"max_participation_pct": 1.5}, {"maker_fee_rate": -0.001}],
)
def test_invalid_execution_config(values):
    with pytest.raises(ConfigError):
        AppConfig.from_mapping({"execution": values})


def test_live_paper_with_limits_and_impact_matches_backtest_and_resumes(tmp_path):
    from test_live import ANCHOR, START, Clock, provider
    from trading_lab.backtest import BacktestEngine
    from trading_lab.data import SyntheticProvider
    from trading_lab.live import LivePaperTrader
    from trading_lab.storage import SQLiteStore

    cfg = AppConfig.from_mapping({
        "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
        "execution": {**LIMIT, "slippage_model": "volume", "max_participation_pct": 0.2},
    })
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(tmp_path / "l.db") as store:
        trader = LivePaperTrader(cfg, provider(clock), store, clock=clock)
        saw_resting = False
        for i in range(120):
            trader.run_cycle()
            clock.now += H
            if i == 60:
                trader.stop()
                resting = dict(trader._session.resting)
                trader = LivePaperTrader.resume(store, trader.run_id, provider(clock), clock=clock)
                assert {k: v.to_json() for k, v in trader._session.resting.items()} == {
                    k: v.to_json() for k, v in resting.items()
                }
            saw_resting |= bool(trader._session.resting)
        assert saw_resting
        live = [(f.timestamp, f.symbol, f.side, round(f.quantity, 6), round(f.fill_price, 4))
                for f in trader.portfolio.fills if f.timestamp < START + 120 * H]
    bt = BacktestEngine(cfg, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + 120 * H)
    expected = [(f.timestamp, f.symbol, f.side, round(f.quantity, 6), round(f.fill_price, 4)) for f in bt.fills]
    assert len(expected) > 5 and live == expected
