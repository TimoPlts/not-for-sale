"""Stage 33: the trend filter on a longer timeframe (risk.trend_filter_timeframe)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from test_live import ANCHOR, START, Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, RiskConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data import SyntheticProvider, candles_from_closes
from trading_lab.engine.filters import filter_columns, higher_timeframe_sma, trend_filter_history, trend_filter_label
from trading_lab.live import LivePaperTrader
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
DAILY = {"market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
         "risk": {"trend_filter_period": 5, "trend_filter_timeframe": "1d"}}


def test_the_average_of_completed_days():
    closes = np.arange(1.0, 24 * 4 + 1)  # day d closes at 24 * (d + 1)
    candles = candles_from_closes(closes, start=T0)
    sma = higher_timeframe_sma(candles, "1h", "1d", 2)
    assert np.isnan(sma[:47]).all()  # fewer than two complete days
    assert sma[47] == (24 + 48) / 2  # the bar closing at midnight completes day 2
    assert (sma[47:71] == 36.0).all() and sma[71] == (48 + 72) / 2 and sma[-1] == (72 + 96) / 2
    four_hourly = higher_timeframe_sma(candles, "1h", "4h", 3)
    assert four_hourly[11] == (4 + 8 + 12) / 3 and four_hourly[12] == four_hourly[11]


def test_values_do_not_depend_on_where_the_window_starts():
    candles = SyntheticProvider(seed=3).fetch_ohlcv("BTC/USDT", "1h", T0, T0 + timedelta(days=12))
    full = higher_timeframe_sma(candles, "1h", "1d", 3)
    late = higher_timeframe_sma(candles.iloc[29:], "1h", "1d", 3)  # starts at 05:00 on day 2
    assert not np.isnan(full[-100:]).any()
    np.testing.assert_array_equal(full[-100:], late[-100:])


def test_a_missing_last_hour_counts_once_the_next_bar_closes():
    candles = candles_from_closes(np.arange(1.0, 24 * 3 + 1), start=T0).drop(T0 + 23 * H)
    sma = higher_timeframe_sma(candles, "1h", "1d", 1)
    assert np.isnan(sma[22])  # 22:00 closes, day 1 is not over
    assert sma[23] == 23.0  # the bar opening at 00:00 is the first to close after it: day 1 closed at 23
    assert higher_timeframe_sma(candles.iloc[:0], "1h", "1d", 1).size == 0


def test_filter_columns_and_history():
    candles = candles_from_closes(np.arange(1.0, 24 * 3 + 1), start=T0)
    risk = RiskConfig(trend_filter_period=2, trend_filter_timeframe="1d")
    rows = filter_columns(candles, risk, None, "1h")
    assert rows[46]["trend_sma"] is None and rows[47]["trend_sma"] == 36.0
    plain = filter_columns(candles, RiskConfig(trend_filter_period=2), None, "1h")
    assert plain[47]["trend_sma"] == 47.5  # the last two hours, as before
    same = filter_columns(candles, RiskConfig(trend_filter_period=2, trend_filter_timeframe="1h"), None, "1h")
    assert same == plain
    assert trend_filter_history(risk, "1h") == 72 and trend_filter_history(risk, "4h") == 18
    assert trend_filter_history(RiskConfig(trend_filter_period=200), "1h") == 201
    assert trend_filter_history(RiskConfig(), "1h") == 0
    assert trend_filter_label(risk) == "2 x 1d" and trend_filter_label(RiskConfig(trend_filter_period=9)) == "9-bar"


@pytest.mark.parametrize("risk, market", [
    ({"trend_filter_period": 5, "trend_filter_timeframe": "2d"}, {}),
    ({"trend_filter_period": 0, "trend_filter_timeframe": "1d"}, {}),
    ({"trend_filter_period": 5, "trend_filter_timeframe": "1h"}, {"timeframe": "4h"}),
    ({"trend_filter_period": 5, "trend_filter_timeframe": "6h"}, {"timeframe": "4h"}),  # not a multiple
])
def test_validation(risk, market):
    with pytest.raises(ConfigError):
        AppConfig.from_mapping({"market": market, "risk": risk})


def test_round_trip_and_old_configs():
    cfg = AppConfig.from_mapping(DAILY)
    assert AppConfig.from_dict(cfg.to_dict()) == cfg and cfg.risk.trend_filter_timeframe == "1d"
    old = AppConfig().to_dict()
    del old["risk"]["trend_filter_timeframe"]  # stored by an earlier version
    assert AppConfig.from_dict(old) == AppConfig()


def test_entries_are_blocked_against_the_daily_trend():
    cfg = AppConfig.from_mapping(DAILY)
    with SQLiteStore(":memory:") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=11, anchor=ANCHOR), store=store).run(
            START, START + 150 * H)
        decisions = store.load_decisions(result.run_id)
    reasons = decisions["reason"].dropna()
    assert reasons.str.contains("trend filter: close .* its 5 x 1d average", regex=True).any()
    assert (decisions["action"] == "enter_signal").any()


def test_live_matches_backtest_with_the_daily_trend_filter():
    cfg = AppConfig.from_mapping(DAILY)
    bars = 150
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(cfg, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(bars):
            trader.run_cycle()
            clock.now += H
        live = [fill_key(f)[1:] for f in trader.portfolio.fills if f.timestamp < START + bars * H]
    bt = BacktestEngine(cfg, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    assert live == [fill_key(f)[1:] for f in bt.fills] and len(live) > 3
    plain = BacktestEngine(AppConfig.from_mapping({"market": DAILY["market"]}),
                           SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    assert [fill_key(f)[1:] for f in plain.fills] != live  # the filter changed something
