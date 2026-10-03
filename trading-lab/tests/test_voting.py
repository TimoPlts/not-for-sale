"""Voting engine tests."""

import pytest

from conftest import ts
from trading_lab.config import StrategySpec, VotingConfig
from trading_lab.core.models import Direction, Signal
from trading_lab.ensemble import ENSEMBLE_NAME, VotingEngine

EQUAL = {"rsi": 1.0, "macd": 1.0, "bollinger": 1.0}


def sig(strategy, direction, confidence=0.0, symbol="BTC/USDT", hours=0):
    if direction == "hold":
        confidence = 0.0
    return Signal(strategy, symbol, direction, confidence, ts(hours))


def votes(**kwargs):
    return [sig(name, *spec) if isinstance(spec, tuple) else sig(name, spec) for name, spec in kwargs.items()]


def test_all_hold_gives_hold():
    out = VotingEngine(EQUAL).combine(votes(rsi="hold", macd="hold", bollinger="hold"))
    assert out.direction is Direction.HOLD and out.confidence == 0.0
    assert out.strategy == ENSEMBLE_NAME
    assert out.metadata["hold_votes"] == 3


def test_single_confident_buy_passes_default_threshold():
    out = VotingEngine(EQUAL).combine(
        votes(rsi=("buy", 0.6), macd="hold", bollinger="hold")
    )
    assert out.direction is Direction.BUY
    assert out.confidence == pytest.approx(0.2)  # 0.6 / 3
    assert out.metadata["net_score"] == pytest.approx(0.2)


def test_weak_signal_diluted_by_abstentions():
    out = VotingEngine(EQUAL).combine(votes(rsi=("buy", 0.4), macd="hold", bollinger="hold"))
    assert out.direction is Direction.HOLD
    assert out.metadata["buy_score"] == pytest.approx(0.4 / 3)


def test_opposing_votes_cancel():
    out = VotingEngine(EQUAL).combine(
        votes(rsi=("buy", 0.6), macd=("sell", 0.6), bollinger="hold")
    )
    assert out.direction is Direction.HOLD
    assert out.metadata["net_score"] == pytest.approx(0.0)


def test_agreement_strengthens_signal():
    out = VotingEngine(EQUAL).combine(
        votes(rsi=("sell", 0.9), macd=("sell", 0.6), bollinger=("buy", 0.5))
    )
    assert out.direction is Direction.SELL
    assert out.confidence == pytest.approx((0.9 + 0.6 - 0.5) / 3)


def test_weights_change_outcome():
    signals = votes(rsi=("sell", 0.5), macd=("buy", 0.5), bollinger="hold")
    assert VotingEngine(EQUAL).combine(signals).direction is Direction.HOLD
    heavy_rsi = VotingEngine({"rsi": 3.0, "macd": 1.0, "bollinger": 1.0})
    out = heavy_rsi.combine(signals)
    assert out.direction is Direction.SELL
    assert out.metadata["net_score"] == pytest.approx((0.5 - 1.5) / 5)


def test_min_agreeing():
    engine = VotingEngine(EQUAL, VotingConfig(min_agreeing=2))
    assert engine.combine(votes(rsi=("buy", 1.0), macd="hold", bollinger="hold")).direction is (
        Direction.HOLD
    )
    assert engine.combine(
        votes(rsi=("buy", 0.6), macd=("buy", 0.6), bollinger="hold")
    ).direction is Direction.BUY


def test_custom_thresholds():
    engine = VotingEngine(EQUAL, VotingConfig(buy_threshold=0.5, sell_threshold=0.1))
    assert engine.combine(votes(rsi=("buy", 0.9), macd=("buy", 0.5), bollinger="hold")).direction is (
        Direction.HOLD
    )  # 1.4 / 3 < 0.5
    assert engine.combine(votes(rsi=("buy", 0.9), macd=("buy", 0.6), bollinger="hold")).direction is (
        Direction.BUY
    )  # exactly 0.5 meets the threshold
    assert engine.combine(votes(rsi=("sell", 0.5), macd="hold", bollinger="hold")).direction is (
        Direction.SELL
    )


def test_subset_of_strategies_uses_their_weights_only():
    out = VotingEngine(EQUAL).combine([sig("rsi", "buy", 0.6)])
    assert out.confidence == pytest.approx(0.6)


def test_metadata_records_every_vote():
    out = VotingEngine(EQUAL).combine(votes(rsi=("buy", 0.7), macd="hold", bollinger=("sell", 0.5)))
    recorded = {v["strategy"]: v for v in out.metadata["votes"]}
    assert recorded["rsi"] == {"strategy": "rsi", "direction": "buy", "confidence": 0.7, "weight": 1.0}
    assert recorded["bollinger"]["direction"] == "sell"
    assert out.symbol == "BTC/USDT" and out.timestamp == ts(0)


@pytest.mark.parametrize(
    "signals, match",
    [
        ([], "empty"),
        ([sig("rsi", "buy", 0.6), sig("macd", "hold", symbol="ETH/USDT")], "same symbol"),
        ([sig("rsi", "buy", 0.6), sig("macd", "hold", hours=1)], "same symbol and timestamp"),
        ([sig("unknown", "buy", 0.6)], "no voting weight"),
        ([sig("rsi", "buy", 0.6), sig("rsi", "sell", 0.6)], "duplicate"),
    ],
)
def test_invalid_inputs(signals, match):
    with pytest.raises(ValueError, match=match):
        VotingEngine(EQUAL).combine(signals)


def test_engine_construction_validation():
    with pytest.raises(ValueError):
        VotingEngine({})
    with pytest.raises(ValueError):
        VotingEngine({"rsi": -1.0})
    with pytest.raises(ValueError):
        VotingEngine({"rsi": 0.0})


def test_from_specs_ignores_disabled():
    engine = VotingEngine.from_specs(
        [StrategySpec("rsi", weight=2.0), StrategySpec("macd", enabled=False)], VotingConfig()
    )
    assert engine.weights == {"rsi": 2.0}
