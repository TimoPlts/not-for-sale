"""Strategy signal tests on crafted price series."""

from datetime import datetime, timezone

import numpy as np
import pytest

from trading_lab.config import AppConfig, StrategySpec
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import Direction, Signal
from trading_lab.data import SyntheticProvider, candles_from_closes
from trading_lab.indicators import macd
from trading_lab.strategies import (
    BollingerMeanReversionStrategy,
    DonchianBreakoutStrategy,
    MacdStrategy,
    MovingAverageCrossStrategy,
    RsiStrategy,
    Strategy,
    available_strategies,
    build_strategies,
    create_strategy,
    register_strategy,
)
from trading_lab.strategies import registry as registry_module

ALL_STRATEGIES = [RsiStrategy(), MacdStrategy(), BollingerMeanReversionStrategy(), MovingAverageCrossStrategy(),
                  DonchianBreakoutStrategy()]
IDS = [s.name for s in ALL_STRATEGIES]


@pytest.fixture(scope="module")
def synthetic_candles():
    return SyntheticProvider(seed=3).fetch_ohlcv(
        "ETH/USDT",
        "1h",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 15, tzinfo=timezone.utc),
    )


# --------------------------------------------------------------- shared contract
@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=IDS)
def test_warmup_returns_hold(strategy, synthetic_candles):
    sig = strategy.generate_signal("ETH/USDT", synthetic_candles.iloc[: strategy.warmup_bars - 1])
    assert sig.direction is Direction.HOLD and sig.confidence == 0.0
    assert sig.metadata["reason"] == "warmup"
    assert sig.metadata["required_bars"] == strategy.warmup_bars


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=IDS)
def test_signals_are_standardized(strategy, synthetic_candles):
    seen = set()
    for t in range(strategy.warmup_bars, len(synthetic_candles)):
        window = synthetic_candles.iloc[: t + 1]
        sig = strategy.generate_signal("ETH/USDT", window)
        assert isinstance(sig, Signal)
        assert sig.strategy == strategy.name and sig.symbol == "ETH/USDT"
        assert sig.timestamp == window.index[-1].to_pydatetime()
        assert sig.metadata["params"] == strategy.params
        if sig.direction is Direction.HOLD:
            assert sig.confidence == 0.0
        else:
            assert 0.5 <= sig.confidence <= 1.0
        seen.add(sig.direction)
    assert {Direction.BUY, Direction.SELL} <= seen  # every strategy fires both ways


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=IDS)
def test_signals_are_deterministic(strategy, synthetic_candles):
    a = strategy.generate_signal("ETH/USDT", synthetic_candles)
    b = strategy.generate_signal("ETH/USDT", synthetic_candles.copy())
    assert a == b


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=IDS)
def test_empty_candles_rejected(strategy, synthetic_candles):
    with pytest.raises(ValueError):
        strategy.generate_signal("ETH/USDT", synthetic_candles.iloc[:0])


# --------------------------------------------------------------------------- RSI
def test_rsi_buy_on_persistent_decline():
    candles = candles_from_closes(100 * 0.99 ** np.arange(30))
    sig = RsiStrategy().generate_signal("BTC/USDT", candles)
    assert sig.direction is Direction.BUY
    assert sig.confidence == pytest.approx(1.0)  # RSI 0 is the extreme
    assert sig.metadata["rsi"] == pytest.approx(0.0)


def test_rsi_sell_on_persistent_rally():
    candles = candles_from_closes(100 * 1.01 ** np.arange(30))
    sig = RsiStrategy().generate_signal("BTC/USDT", candles)
    assert sig.direction is Direction.SELL and sig.confidence == pytest.approx(1.0)


def test_rsi_confidence_scales_with_extremity():
    # Down 2, up 0.5 repeatedly gives RSI ~20: oversold but not extreme.
    closes = [100.0]
    for i in range(40):
        closes.append(closes[-1] + (0.5 if i % 3 == 2 else -1.0))
    sig = RsiStrategy().generate_signal("BTC/USDT", candles_from_closes(closes))
    assert sig.direction is Direction.BUY
    rsi_value = sig.metadata["rsi"]
    assert 0 < rsi_value < 30
    assert sig.confidence == pytest.approx(0.5 + 0.5 * (30 - rsi_value) / 30)


def test_rsi_hold_in_neutral_zone():
    closes = 100 + np.tile([1.0, -1.0], 20)
    sig = RsiStrategy().generate_signal("BTC/USDT", candles_from_closes(closes))
    assert sig.direction is Direction.HOLD
    assert 30 <= sig.metadata["rsi"] <= 70


# -------------------------------------------------------------------------- MACD
def test_macd_signals_exactly_on_histogram_crossovers():
    closes = 100 + 10 * np.sin(2 * np.pi * np.arange(200) / 50)
    candles = candles_from_closes(closes)
    hist = macd(candles["close"])["hist"].to_numpy()
    strategy = MacdStrategy()
    bullish = bearish = 0
    for t in range(strategy.warmup_bars, len(candles)):
        sig = strategy.generate_signal("SOL/USDT", candles.iloc[: t + 1])
        if hist[t - 1] <= 0 < hist[t]:
            assert sig.direction is Direction.BUY and sig.metadata["crossover"] == "bullish"
            bullish += 1
        elif hist[t - 1] >= 0 > hist[t]:
            assert sig.direction is Direction.SELL and sig.metadata["crossover"] == "bearish"
            bearish += 1
        else:
            assert sig.direction is Direction.HOLD
    assert bullish >= 2 and bearish >= 2


# ---------------------------------------------------------------------- Bollinger
def _oscillation(n=40):
    return list(100 + 0.5 * np.sin(np.arange(n)))


def test_bollinger_buy_below_lower_band():
    sig = BollingerMeanReversionStrategy().generate_signal(
        "DOGE/USDT", candles_from_closes(_oscillation() + [98.0])
    )
    assert sig.direction is Direction.BUY
    assert sig.metadata["percent_b"] < 0
    assert sig.metadata["close"] < sig.metadata["lower"]


def test_bollinger_sell_above_exit_level():
    sig = BollingerMeanReversionStrategy().generate_signal(
        "DOGE/USDT", candles_from_closes(_oscillation() + [102.0])
    )
    assert sig.direction is Direction.SELL
    assert sig.metadata["percent_b"] >= 1.0


def test_bollinger_middle_band_exit_variant():
    candles = candles_from_closes(_oscillation() + [100.3])
    default = BollingerMeanReversionStrategy().generate_signal("DOGE/USDT", candles)
    mid_exit = BollingerMeanReversionStrategy(exit_percent_b=0.5).generate_signal(
        "DOGE/USDT", candles
    )
    assert 0.5 <= default.metadata["percent_b"] < 1.0
    assert default.direction is Direction.HOLD
    assert mid_exit.direction is Direction.SELL


def test_bollinger_holds_when_bands_have_zero_width():
    sig = BollingerMeanReversionStrategy().generate_signal(
        "DOGE/USDT", candles_from_closes([1.0] * 30)
    )
    assert sig.direction is Direction.HOLD and sig.metadata["percent_b"] is None


# ------------------------------------------------------------ params & registry
@pytest.mark.parametrize(
    "factory",
    [
        lambda: RsiStrategy(oversold=80, overbought=70),
        lambda: RsiStrategy(period=1),
        lambda: RsiStrategy(period=14.5),
        lambda: MacdStrategy(fast=26, slow=12),
        lambda: BollingerMeanReversionStrategy(period=1),
        lambda: BollingerMeanReversionStrategy(num_std=-1),
    ],
)
def test_invalid_parameters_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_build_strategies_from_default_config():
    strategies = build_strategies(AppConfig().enabled_strategies)
    assert [s.name for s in strategies] == ["rsi", "macd", "bollinger"]
    assert set(available_strategies()) >= {"rsi", "macd", "bollinger"}


def test_disabled_strategies_are_skipped():
    specs = [StrategySpec("rsi"), StrategySpec("macd", enabled=False)]
    assert [s.name for s in build_strategies(specs)] == ["rsi"]


def test_registry_errors():
    with pytest.raises(ConfigError, match="unknown strategy"):
        create_strategy("does_not_exist")
    with pytest.raises(ConfigError, match="invalid parameters"):
        create_strategy("rsi", {"period": 14.5})
    with pytest.raises(ConfigError, match="invalid parameters"):
        create_strategy("rsi", {"lookback": 3})


def test_custom_strategy_can_be_registered():
    @register_strategy
    class AlwaysBuy(Strategy):
        name = "always_buy_test"
        warmup_bars = 1
        params = {}

        def _evaluate(self, candles):
            return Direction.BUY, 0.75, {"note": "test"}

    try:
        (strategy,) = build_strategies([StrategySpec("always_buy_test")])
        sig = strategy.generate_signal("BTC/USDT", candles_from_closes([1.0, 2.0]))
        assert sig.direction is Direction.BUY and sig.confidence == 0.75
        with pytest.raises(ValueError, match="already registered"):

            @register_strategy
            class Clash(AlwaysBuy):
                pass

    finally:
        registry_module._REGISTRY.pop("always_buy_test", None)
