"""Stage 17A: trend-following strategies (moving-average crossover, Donchian breakout)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.models import Direction
from trading_lab.data import SyntheticProvider, candles_from_closes
from trading_lab.strategies import DonchianBreakoutStrategy, MovingAverageCrossStrategy, create_strategy
from trading_lab.strategy_factory import strategies_for

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
STRATEGIES = [
    MovingAverageCrossStrategy(), MovingAverageCrossStrategy(fast=5, slow=12, average="ema", signal_on="state"),
    DonchianBreakoutStrategy(), DonchianBreakoutStrategy(entry_period=10, exit_period=0),
]


@pytest.fixture(scope="module")
def candles():
    return SyntheticProvider(seed=9).fetch_ohlcv("BTC/USDT", "1h", datetime(2024, 1, 1, tzinfo=UTC),
                                                 datetime(2024, 1, 20, tzinfo=UTC))


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: f"{s.name}-{list(s.params.values())}")
def test_vectorised_signals_equal_bar_by_bar_signals(strategy, candles):
    """No look-ahead: each bar's signal only depends on the candles up to it."""
    fast = strategy.generate_signals("BTC/USDT", candles)
    for i in range(0, len(candles), 7):
        assert fast[i] == strategy.generate_signal("BTC/USDT", candles.iloc[: i + 1])


def test_ma_cross_fires_exactly_on_crosses():
    closes = [100.0] * 60 + list(np.linspace(100, 130, 30)) + list(np.linspace(130, 90, 40))
    strategy = MovingAverageCrossStrategy(fast=5, slow=20)
    signals = strategy.generate_signals("BTC/USDT", candles_from_closes(closes))
    frame = strategy.indicators(candles_from_closes(closes))
    gap = frame["gap"].to_numpy()
    crosses = [i for i in range(1, len(closes)) if (gap[i - 1] <= 0 < gap[i]) or (gap[i - 1] >= 0 > gap[i])]
    fired = [i for i, s in enumerate(signals) if s.direction is not Direction.HOLD]
    assert fired == [i for i in crosses if i + 1 >= strategy.warmup_bars]
    directions = [signals[i].direction for i in fired]
    assert directions[0] is Direction.BUY and directions[-1] is Direction.SELL
    assert signals[fired[0]].metadata["crossover"] == "bullish"


def test_ma_cross_state_mode_follows_the_trend():
    closes = list(np.linspace(100, 150, 80)) + list(np.linspace(150, 100, 80))
    signals = MovingAverageCrossStrategy(fast=5, slow=20, signal_on="state").generate_signals(
        "BTC/USDT", candles_from_closes(closes))
    assert all(s.direction is Direction.BUY for s in signals[25:80])
    assert all(s.direction is Direction.SELL for s in signals[110:])


def test_donchian_breakouts_and_exits():
    flat = [100.0 + (i % 2) for i in range(30)]  # a 100-101 range
    up = flat + [103.0]
    sig = DonchianBreakoutStrategy().generate_signal("BTC/USDT", candles_from_closes(up, spread=0.0))
    assert sig.direction is Direction.BUY and sig.metadata["breakout"] == "up" and sig.confidence > 0.5
    down = flat + [98.0]
    assert DonchianBreakoutStrategy().generate_signal("BTC/USDT", candles_from_closes(down, spread=0.0)).direction \
        is Direction.SELL
    inside = DonchianBreakoutStrategy().generate_signal("BTC/USDT", candles_from_closes(flat + [100.5], spread=0.0))
    assert inside.direction is Direction.HOLD

    # A close below the 10-bar low but inside the 20-bar channel is a weak exit SELL.
    closes = [100.0 + (i % 2) for i in range(30)] + [99.0]
    closes[15] = 95.0  # inside the 20-bar channel, outside the 10-bar one
    candles = candles_from_closes(closes, spread=0.0)
    weak = DonchianBreakoutStrategy().generate_signal("BTC/USDT", candles)
    assert (weak.direction, weak.confidence, weak.metadata["breakout"]) == (Direction.SELL, 0.5, "exit_down")
    assert DonchianBreakoutStrategy(exit_period=0).generate_signal("BTC/USDT", candles).direction is Direction.HOLD


def test_the_channel_excludes_the_current_bar():
    closes = [100.0 + (i % 2) for i in range(30)] + [150.0]
    frame = DonchianBreakoutStrategy().indicators(candles_from_closes(closes, spread=0.0))
    assert frame["upper"].iloc[-1] == 101.0  # the breakout bar's own high is not part of its channel


@pytest.mark.parametrize("kwargs", [
    {"fast": 50, "slow": 20}, {"average": "wma"}, {"signal_on": "always"}, {"fast": 0},
])
def test_ma_cross_validation(kwargs):
    with pytest.raises(ValueError):
        MovingAverageCrossStrategy(**kwargs)


@pytest.mark.parametrize("kwargs", [{"entry_period": 1}, {"exit_period": 20}, {"exit_period": -1}, {"atr_period": 1}])
def test_donchian_validation(kwargs):
    with pytest.raises(ValueError):
        DonchianBreakoutStrategy(**kwargs)


def test_opt_in_from_the_config_and_tradable_both_ways():
    assert "donchian" not in [s.name for s in AppConfig().enabled_strategies]  # not on by default
    cfg = AppConfig.from_mapping({
        "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
        "risk": {"allow_short": True},
        "strategies": {"rsi": {"enabled": False}, "macd": {"enabled": False}, "bollinger": {"enabled": False},
                       "donchian": {"entry_period": 24, "exit_period": 12},
                       "ma_cross": {"fast": 12, "slow": 48, "average": "ema"}},
    })
    names = [s.name for s in strategies_for(cfg)]
    assert {"donchian", "ma_cross"} <= set(names) and "rsi" not in names
    assert create_strategy("ma_cross", {"fast": 12, "slow": 48, "average": "ema"}).params["average"] == "ema"
    result = BacktestEngine(cfg, SyntheticProvider(seed=2)).run(START, START + timedelta(days=30))
    assert {t.side for t in result.trades} == {"long", "short"}
    votes = {s.strategy for s in result.signals if s.direction is not Direction.HOLD}
    assert {"donchian", "ma_cross"} <= votes
    assert isinstance(result.equity_curve, pd.DataFrame)
