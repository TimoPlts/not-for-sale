"""Stage 12A: ATR-based stops and volatility-scaled position sizing."""

from datetime import datetime, timedelta, timezone

import pytest

from conftest import make_costs
from test_live import ANCHOR, START, Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, RiskConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import DecisionAction
from trading_lab.data import SyntheticProvider
from trading_lab.execution.costs import MarketStats, market_stats_frame, next_bar_stats, stats_series
from trading_lab.live import LivePaperTrader
from trading_lab.portfolio import Portfolio
from trading_lab.risk import RiskManager
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
ATR_RISK = {"stop_mode": "atr", "atr_period": 14, "atr_stop_multiple": 2.0, "max_position_pct": 1.0,
            "max_total_exposure_pct": 1.0, "risk_per_trade_pct": 0.01}


@pytest.fixture(scope="module")
def candles():
    return SyntheticProvider(seed=7).fetch_ohlcv("ETH/USDT", "1h", datetime(2024, 1, 1, tzinfo=UTC),
                                                 datetime(2024, 1, 20, tzinfo=UTC))


def test_atr_is_causal_and_identical_for_backtest_and_live(candles):
    frame = market_stats_frame(candles, 20, atr_period=14)
    series = stats_series(frame)
    for i in (40, 100, 300):
        live = next_bar_stats(candles.iloc[:i], 20, atr_period=14)  # what live trading knows before bar i
        assert live.atr == pytest.approx(series[i].atr, rel=1e-12)
        assert live.volatility == pytest.approx(series[i].volatility, rel=1e-9)
    changed = candles.copy()
    changed.iloc[200:, :] *= 1.5  # changing bar 200 onwards must not change the ATR known at bar 200
    assert stats_series(market_stats_frame(changed, 20))[200].atr == pytest.approx(series[200].atr)
    high, low, prev = candles["high"].iloc[99], candles["low"].iloc[99], candles["close"].iloc[98]
    true_range = max(high - low, abs(high - prev), abs(low - prev))
    assert true_range > 0 and series[100].atr > 0


def risk(**cfg):
    return RiskManager(RiskConfig(**cfg), make_costs(0.0, 0.0), min_notional=1.0)


def test_stop_distance_modes_clamping_and_fallback():
    stats = MarketStats(0.01, 1e9, atr=2.0)
    assert risk().stop_distance_pct(100.0, stats) == (0.05, "percent")
    atr = risk(stop_mode="atr")
    assert atr.stop_distance_pct(100.0, stats) == (pytest.approx(0.04), "atr")  # 2 x 2.0 / 100
    assert atr.stop_price_for(100.0, stats) == pytest.approx(96.0)
    assert atr.stop_distance_pct(100.0, MarketStats(0.01, 1e9, atr=0.01))[0] == 0.005  # clamped up
    assert atr.stop_distance_pct(100.0, MarketStats(0.01, 1e9, atr=50.0))[0] == 0.25  # clamped down
    assert atr.stop_distance_pct(100.0, None) == (0.05, "percent")  # no history yet
    assert atr.stop_distance_pct(100.0, MarketStats(0.01, 1e9)) == (0.05, "percent")


def test_volatile_coins_get_smaller_positions_for_the_same_risk():
    manager = risk(**ATR_RISK)
    portfolio = Portfolio(10_000.0)
    calm = manager.evaluate_entry("BTC/USDT", 100.0, portfolio, {}, MarketStats(0.01, 1e12, atr=1.0))
    wild = manager.evaluate_entry("SOL/USDT", 100.0, portfolio, {}, MarketStats(0.05, 1e12, atr=4.0))
    assert calm.approved and wild.approved
    assert calm.sizing["binding_limit"] == wild.sizing["binding_limit"] == "risk_per_trade"
    assert wild.quantity == pytest.approx(calm.quantity / 4)
    for decision in (calm, wild):  # the loss at the stop is 1% of equity either way
        assert decision.quantity * decision.sizing["loss_per_unit_at_stop"] == pytest.approx(100.0)
        assert decision.sizing["stop_basis"] == "atr"
    assert wild.stop_price == pytest.approx(92.0) and calm.stop_price == pytest.approx(98.0)


def test_backtest_uses_atr_stops():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}, "risk": ATR_RISK})
    result = BacktestEngine(cfg, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + 120 * H)
    entries = [d for d in result.decisions if d.action is DecisionAction.ENTER]
    assert entries and all(d.details["stop_basis"] == "atr" for d in entries)
    for d in entries:
        distance = d.details["stop_distance_pct"]
        assert 0.005 <= distance <= 0.25
        # stop = fill x (1 - distance); the fill is the open plus 5 bps of slippage
        assert d.stop_price == pytest.approx(d.reference_price * 1.0005 * (1 - distance), rel=1e-3)
    percent = BacktestEngine(cfg.with_overrides({"risk": {**ATR_RISK, "stop_mode": "percent"}}),
                             SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + 120 * H)
    assert [f.quantity for f in percent.fills] != [f.quantity for f in result.fills]


def test_live_matches_backtest_in_atr_mode():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}, "risk": ATR_RISK})
    bars = 120
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(cfg, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(bars):
            trader.run_cycle()
            clock.now += H
        live = [fill_key(f)[1:] for f in trader.portfolio.fills if f.timestamp < START + bars * H]
        live_stops = [f.stop_price for f in trader.portfolio.fills if f.timestamp < START + bars * H]
    bt = BacktestEngine(cfg, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    assert live == [fill_key(f)[1:] for f in bt.fills] and len(live) > 5
    assert live_stops == pytest.approx([f.stop_price for f in bt.fills])


def test_validation():
    for bad in ({"stop_mode": "chandelier"}, {"atr_period": 1}, {"atr_stop_multiple": 0},
                {"atr_stop_min_pct": 0.3, "atr_stop_max_pct": 0.2}):
        with pytest.raises(ConfigError):
            RiskConfig(**bad)
