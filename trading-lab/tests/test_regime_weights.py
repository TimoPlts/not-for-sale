"""Stage 20C: regime-dependent strategy weights (voting.regime_weights)."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from test_live import ANCHOR, START, Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, VotingConfig, load_config
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import Direction, Signal
from trading_lab.data import SyntheticProvider
from trading_lab.engine import Bar, TradingSession
from trading_lab.engine.filters import WARMUP, filter_columns, trend_labels
from trading_lab.ensemble import VotingEngine
from trading_lab.live import LivePaperTrader
from trading_lab.research.regimes import classify
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def test_config(tmp_path):
    assert VotingConfig().regime_weights == () and VotingConfig().regime_multipliers("up") == {}
    path = tmp_path / "c.toml"
    path.write_text('[voting.regime_weights.up]\nmacd = 2.0\n[voting.regime_weights.sideways]\nmacd = 0\nrsi = 1.5\n')
    cfg = load_config(path)
    assert cfg.voting.regime_multipliers("sideways") == {"macd": 0.0, "rsi": 1.5}
    assert cfg.voting.regime_multipliers("down") == {} and cfg.voting.regime_multipliers(None) == {}
    assert AppConfig.from_dict(cfg.to_dict()) == cfg and AppConfig.from_mapping(cfg.to_mapping()) == cfg
    assert AppConfig.from_dict({**AppConfig().to_dict(), "voting": {"buy_threshold": 0.15}}) == AppConfig()
    for bad in ({"bull": {"macd": 1}}, {"up": {"macd": -1}}, {"up": {"macd": "x"}}, {"up": 3}, {"up": {"nope": 1}}):
        with pytest.raises(ConfigError):
            AppConfig.from_mapping({"voting": {"regime_weights": bad}})
    with pytest.raises(ConfigError):
        VotingConfig(regime_bars=1)


def test_labels_match_the_regime_report_and_are_causal():
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))),
                      index=pd.date_range(T0, periods=400, freq="h"))
    labels = trend_labels(close, 50, 10)
    assert list(labels) == list(classify(close, trend_bars=50, slope_bars=10)["trend"])
    assert (labels[:49] == WARMUP).all() and set(labels[49:]) <= {"up", "down", "sideways"}
    for t in (60, 200, 399):
        assert trend_labels(close.iloc[: t + 1], 50, 10)[-1] == labels[t]


def test_filter_columns():
    candles = SyntheticProvider(seed=4).fetch_ohlcv("BTC/USDT", "1h", T0, T0 + 120 * H)
    assert all("regime" not in row for row in filter_columns(candles, AppConfig().risk, AppConfig().voting))
    voting = VotingConfig(regime_weights={"up": {"rsi": 2.0}}, regime_bars=20, regime_slope_bars=5)
    rows = filter_columns(candles, AppConfig().risk, voting)
    assert rows[18]["regime"] is None and rows[60]["regime"] in ("up", "down", "sideways")


def sig(name, direction, conf=1.0):
    return Signal(name, "BTC/USDT", direction, conf, T0, {})


def test_voting_with_multipliers():
    engine = VotingEngine({"a": 1.0, "b": 1.0})
    votes = [sig("a", Direction.BUY), sig("b", Direction.SELL)]
    assert engine.combine(votes).direction is Direction.HOLD
    boosted = engine.combine(votes, {"a": 3.0})
    assert boosted.direction is Direction.BUY and boosted.metadata["net_score"] == pytest.approx(0.5)
    assert [v["weight"] for v in boosted.metadata["votes"]] == [3.0, 1.0]
    assert boosted.metadata["regime_multipliers"] == {"a": 3.0}
    muted = engine.combine(votes, {"a": 0.0, "b": 0.0})
    assert muted.direction is Direction.HOLD and "every participating weight is 0" in muted.metadata["reason"]
    assert "regime_multipliers" not in engine.combine(votes).metadata  # unchanged without multipliers


def test_the_session_applies_the_bar_regime():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]},
                                  "voting": {"regime_weights": {"down": {"rsi": 0.0}}}})
    s = TradingSession(cfg, VotingEngine({"rsi": 1.0}))
    for i, regime in enumerate(("up", "down", None)):
        ts = T0 + i * H
        s.close_bar(ts, {"BTC/USDT": Bar(100, 101, 99, 100, 1)},
                    {"BTC/USDT": [Signal("rsi", "BTC/USDT", Direction.BUY, 1.0, ts, {})]},
                    market={"BTC/USDT": {"regime": regime}})
    ensembles = [x for x in s.records.signals if x.strategy == "ensemble"]
    assert ensembles[0].direction is Direction.BUY and "regime" not in ensembles[0].metadata  # "up": no weights
    assert ensembles[1].direction is Direction.HOLD and ensembles[1].metadata["regime"] == "down"
    assert ensembles[2].direction is Direction.BUY  # warm-up: no regime, normal weights


CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}}).with_overrides({
    "strategies": {"donchian": {"weight": 1.0}},  # added to the default strategies
    "voting": {"regime_weights": {"up": {"donchian": 2.0, "rsi": 0.5, "bollinger": 0.5},
                                  "down": {"donchian": 2.0, "rsi": 0.5, "bollinger": 0.5},
                                  "sideways": {"donchian": 0.0}}},
})


def test_backtests_change_and_live_matches():
    bars = 150
    plain_cfg = CFG.with_overrides({"voting": {"regime_weights": {}}})
    plain = BacktestEngine(plain_cfg, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    bt = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    assert [fill_key(f)[1:] for f in bt.fills] != [fill_key(f)[1:] for f in plain.fills]
    regimes = {s.metadata.get("regime") for s in bt.signals if s.strategy == "ensemble"}
    assert {"up", "down", "sideways"} & regimes
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(bars):
            trader.run_cycle()
            clock.now += H
        live = [fill_key(f)[1:] for f in trader.portfolio.fills if f.timestamp < START + bars * H]
    assert live and live == [fill_key(f)[1:] for f in bt.fills]
