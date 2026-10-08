"""Stage 29C: the desk's confirmation gate (voting.confirmers / min_confirms / confirm_mode)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from test_live import ANCHOR, START as LIVE_START, Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import Direction, Side, Signal
from trading_lab.data import SyntheticProvider
from trading_lab.engine.session import TradingSession
from trading_lab.ensemble.voting import ENSEMBLE_NAME
from trading_lab.live import LivePaperTrader
from trading_lab.presets import PRESETS
from trading_lab.research.desk import desk_funnel
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
START = datetime(2024, 2, 1, tzinfo=UTC)
SYMBOLS = {"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}}


def config(**voting):
    return AppConfig.from_mapping({**SYMBOLS, "voting": voting})


def ensemble(**votes):
    return Signal(ENSEMBLE_NAME, "BTC/USDT", Direction.BUY, 0.5, START, {
        "votes": [{"strategy": k, "direction": v, "confidence": 0.7, "weight": 1.0} for k, v in votes.items()]})


def shortfall(cfg, signal, side=Side.BUY):
    return TradingSession.confirmation_shortfall(SimpleNamespace(config=cfg), signal, side)


def test_config_validation():
    assert config().voting.min_confirms == 0 and config().voting.confirmers == ()  # off by default
    ok = config(confirmers=["rsi", "macd"], min_confirms=2)
    assert AppConfig.from_mapping(ok.to_mapping()) == ok and ok.to_dict()["voting"]["confirmers"] == ["rsi", "macd"]
    for bad in ({"confirmers": ["rsi"], "min_confirms": 2}, {"confirmers": ["rsi", "rsi"], "min_confirms": 1},
                {"confirmers": ["rsi"], "min_confirms": 1, "confirm_mode": "maybe"},
                {"confirmers": ["qwen_trend"], "min_confirms": 1},  # weight 0: it never votes
                {"confirmers": ["nope"], "min_confirms": 1}, {"confirmers": "rsi"}):
        with pytest.raises(ConfigError):
            config(**bad)


def test_the_gate_by_hand():
    agree = config(confirmers=["rsi", "macd", "bollinger"], min_confirms=2)
    assert shortfall(agree, ensemble(rsi="buy", macd="buy", bollinger="sell")) is None
    missing = shortfall(agree, ensemble(rsi="buy", macd="hold", bollinger="sell"))
    assert missing == "1 of 2 needed (agree; confirmed: rsi; against: bollinger)"
    assert shortfall(agree, ensemble(rsi="sell", macd="sell", bollinger="buy"), Side.SELL) is None  # shorts too
    veto = config(confirmers=["rsi", "macd"], min_confirms=2, confirm_mode="not_against")
    assert shortfall(veto, ensemble(rsi="hold", macd="buy")) is None  # silence is not an objection
    assert shortfall(veto, ensemble(rsi="sell", macd="buy")) == \
        "1 of 2 needed (not_against; confirmed: macd; against: rsi)"
    assert shortfall(config(), ensemble(rsi="sell")) is None  # gate off


@pytest.fixture(scope="module")
def runs():
    plain = BacktestEngine(config(), SyntheticProvider(seed=4)).run(START, START + timedelta(days=40))
    gated_cfg = config(confirmers=["rsi", "macd", "bollinger"], min_confirms=2)
    gated = BacktestEngine(gated_cfg, SyntheticProvider(seed=4)).run(START, START + timedelta(days=40))
    return plain, gated, gated_cfg


def test_every_entry_is_confirmed(runs):
    plain, gated, cfg = runs
    entries = [d for d in gated.decisions if d.action == "enter_signal"]
    assert 0 < len(entries) < len([d for d in plain.decisions if d.action == "enter_signal"])
    votes = {(s.timestamp, s.symbol): s for s in gated.signals if s.strategy == ENSEMBLE_NAME}
    for d in entries:
        assert shortfall(cfg, votes[(d.timestamp, d.symbol)], Side.BUY) is None
    blocked = [d for d in gated.decisions if "blocked by confirmation" in d.reason]
    assert blocked and all(d.action == "ignored" for d in blocked)
    # exits are never gated: every position that opened was closed by the usual exits
    assert {d.action for d in gated.decisions} >= {"exit_signal", "exit"}


def test_the_desk_funnel_counts_it(runs, tmp_path):
    _, _, cfg = runs
    with SQLiteStore(tmp_path / "h.db") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=4), store=store).run(START, START + timedelta(days=40))
        funnel = desk_funnel(store, result.run_id)
    assert funnel.killed["filter: confirmation"] == sum(1 for d in result.decisions
                                                        if "blocked by confirmation" in d.reason)


def test_desk_preset_live_matches_backtest():
    cfg = PRESETS["desk"].config(AppConfig.from_mapping(SYMBOLS))
    assert cfg.voting.confirm_mode == "not_against" and set(cfg.voting.confirmers) == {"funding", "sentiment"}
    # synthetic funding stays near +/-0.02% per 8h, so lower the bar to make crowding (and vetoes) happen here
    cfg = cfg.with_overrides({"strategies": {"funding": {"weight": 0.5, "high": 0.00008, "low": -0.0003}}})
    bars = 24 * 8
    bt = BacktestEngine(cfg, SyntheticProvider(seed=11, anchor=ANCHOR)).run(LIVE_START, LIVE_START + bars * H)
    vetoed = [d for d in bt.decisions if "blocked by confirmation" in d.reason and "against" in d.reason]
    assert vetoed  # crowding or mood objected at least once
    clock = Clock(LIVE_START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(cfg, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(bars):
            trader.run_cycle()
            clock.now += H
        live = [fill_key(f)[1:] for f in trader.portfolio.fills if f.timestamp < LIVE_START + bars * H]
        live_vetoes = [d for d in store.load_decisions(trader.run_id, include_holds=False).itertuples()
                       if "blocked by confirmation" in d.reason]
    assert live == [fill_key(f)[1:] for f in bt.fills]
    assert len(live_vetoes) == len([d for d in bt.decisions if "blocked by confirmation" in d.reason])
