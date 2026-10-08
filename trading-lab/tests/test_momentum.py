"""Stage 32: time-series momentum (tsmom), the daily swing preset, and timeframe-aware tournament deflation."""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from test_live import Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import Direction
from trading_lab.data import SyntheticProvider, candles_from_closes
from trading_lab.live import LivePaperTrader
from trading_lab.presets import PRESETS
from trading_lab.research.tournament import rescale_sharpe, tournament
from trading_lab.storage import SQLiteStore
from trading_lab.strategies import TimeSeriesMomentumStrategy, create_strategy
from trading_lab.strategy_factory import strategies_for

UTC = timezone.utc
D = timedelta(days=1)
ANCHOR = datetime(2021, 1, 1, tzinfo=UTC)
START = datetime(2023, 1, 1, tzinfo=UTC)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


def tsmom(**kwargs):
    return TimeSeriesMomentumStrategy(**{"lookbacks": (5, 10, 20), "vol_window": 5, **kwargs})


@pytest.mark.parametrize("strategy", [tsmom(), tsmom(threshold=1.0), TimeSeriesMomentumStrategy()],
                         ids=lambda s: str(s.params))
def test_vectorised_signals_equal_bar_by_bar_signals(strategy):
    candles = SyntheticProvider(seed=4).fetch_ohlcv("BTC/USDT", "1h", datetime(2024, 1, 1, tzinfo=UTC),
                                                    datetime(2024, 1, 20, tzinfo=UTC))
    fast = strategy.generate_signals("BTC/USDT", candles)
    assert any(s.direction is not Direction.HOLD for s in fast)
    for i in range(0, len(candles), 7):
        assert fast[i] == strategy.generate_signal("BTC/USDT", candles.iloc[: i + 1])


def test_direction_follows_the_horizons():
    s = tsmom()
    up = s.generate_signal("BTC/USDT", candles_from_closes(np.linspace(100, 130, 40)))
    assert up.direction is Direction.BUY and up.metadata["score"] == 1.0 and 0.5 <= up.confidence <= 1.0
    down = s.generate_signal("BTC/USDT", candles_from_closes(np.linspace(130, 100, 40)))
    assert down.direction is Direction.SELL and down.metadata["score"] == -1.0
    # long horizons up, the last 5 bars down: 2 of 3 agree, enough for 0.3 but not for 1.0
    closes = np.concatenate([np.linspace(100, 130, 36), [129.5, 129.0, 128.5, 128.0]])
    mixed = candles_from_closes(closes)
    assert s.generate_signal("BTC/USDT", mixed).direction is Direction.BUY
    assert s.generate_signal("BTC/USDT", mixed).metadata["score"] == pytest.approx(1 / 3)
    assert tsmom(threshold=1.0).generate_signal("BTC/USDT", mixed).direction is Direction.HOLD
    warm = s.generate_signal("BTC/USDT", candles_from_closes(np.linspace(100, 130, 20)))
    assert warm.direction is Direction.HOLD and warm.metadata["reason"] == "warmup" and s.warmup_bars == 21
    assert s.history_bars == 22  # no smoothing to converge, unlike the 300-bar default


def test_confidence_grows_with_the_move_in_its_usual_size():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 40)
    small = 100 * np.exp(np.cumsum(noise + 0.002))
    large = 100 * np.exp(np.cumsum(noise + 0.02))
    s = tsmom()
    a, b = (s.generate_signal("BTC/USDT", candles_from_closes(c)) for c in (small, large))
    assert a.direction is Direction.BUY and b.direction is Direction.BUY
    assert 0.5 <= a.confidence < b.confidence == 1.0
    flat = s.generate_signal("BTC/USDT", candles_from_closes(np.concatenate([np.full(25, 100.0),
                                                                              np.full(15, 110.0)])))
    assert flat.direction is Direction.BUY and flat.confidence == 0.5  # no volatility to measure against
    assert flat.metadata["vol"] == 0.0 and set(flat.metadata["returns"]) == {"5", "10", "20"}


@pytest.mark.parametrize("kwargs", [{"lookbacks": []}, {"lookbacks": 20}, {"lookbacks": [5, 5]}, {"lookbacks": [0]},
                                    {"lookbacks": [6000]}, {"lookbacks": [5.5]}, {"threshold": 0},
                                    {"threshold": 1.5}, {"threshold": True}, {"vol_window": 1}])
def test_validation(kwargs):
    with pytest.raises(ConfigError):
        create_strategy("tsmom", kwargs)


def test_opt_in_and_parameters_are_recorded():
    assert "tsmom" not in [s.name for s in AppConfig().enabled_strategies]
    cfg = BASE.with_overrides({"strategies": {"tsmom": {"lookbacks": [60, 20]}}})
    strategy = [s for s in strategies_for(cfg) if s.name == "tsmom"][0]
    assert strategy.params == {"lookbacks": [20, 60], "threshold": 0.3, "vol_window": 20}


def test_the_swing_preset_trades_daily_bars_rarely():
    cfg = PRESETS["swing"].config(BASE)
    assert cfg.market.timeframe == "1d" and {s.name for s in cfg.enabled_strategies} == {"tsmom", "donchian",
                                                                                         "ma_cross"}
    result = BacktestEngine(cfg, SyntheticProvider(seed=1)).run(START, START + 365 * D)
    assert 0 < len(result.trades) <= 30  # a few trades a month at most, so fees matter less
    assert len(result.equity_curve) == 365
    assert "tsmom" in {s.strategy for s in result.signals if s.direction is not Direction.HOLD}


def test_swing_live_paper_trading_matches_backtest():
    cfg = PRESETS["swing"].config(BASE)
    bars = 120
    expected = BacktestEngine(cfg, SyntheticProvider(seed=5, anchor=ANCHOR)).run(START, START + bars * D)
    clock = Clock(START + D + timedelta(minutes=1))  # the day START has just closed
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(cfg, SyntheticProvider(seed=5, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(bars):
            trader.run_cycle()
            clock.now += D
        live = [fill_key(f) for f in trader.portfolio.fills if f.timestamp < START + bars * D]
    assert len(live) >= 2
    assert [k[1:] for k in live] == [k[1:] for k in map(fill_key, expected.fills)]


def test_sharpe_ratios_are_rescaled_between_timeframes():
    assert rescale_sharpe(0.01, "1h", "1d") == pytest.approx(0.01 * math.sqrt(24))
    assert rescale_sharpe(0.24, "1d", "1h") == pytest.approx(0.24 / math.sqrt(24))
    assert rescale_sharpe(0.3, "4h", "4h") == 0.3 and rescale_sharpe(None, "1h", "1d") is None


def test_a_tournament_mixes_timeframes():
    candidates = {"swing": PRESETS["swing"].config(BASE), "trend": PRESETS["trend"].config(BASE)}
    t = tournament(candidates, BASE, SyntheticProvider(seed=3), START, START + 60 * D, windows=2, permutations=2)
    assert {e.name for e in t.entries} == {"swing", "trend"}
    assert all(e.deflated is not None and 0 <= e.deflated <= 1 for e in t.entries)
    swing = next(e for e in t.entries if e.name == "swing")
    assert len(swing.validation.returns) == 60  # daily bars, not hourly
