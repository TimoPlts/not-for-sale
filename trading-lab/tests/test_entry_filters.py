"""Stage 12C: entry filters (trend and risk-state); they only ever block new entries."""

from datetime import datetime, timedelta, timezone

import pytest

from test_live import ANCHOR, START, Clock, fill_key
from test_specialists import RoleTransport, provider
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, RiskConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import DecisionAction, Direction, Signal
from trading_lab.data import SyntheticProvider
from trading_lab.engine import Bar, Intent, TradingSession
from trading_lab.ensemble import VotingEngine
from trading_lab.live import LivePaperTrader
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
SYM = "BTC/USDT"


def session(**risk):
    cfg = AppConfig.from_mapping({"market": {"symbols": [SYM]}, "risk": risk})
    return TradingSession(cfg, VotingEngine({"x": 1.0, "qwen_risk": 1.0}))


def bar_close(s, i, close, direction="buy", risk_state=None, market=None):
    ts = T0 + i * H
    signals = [Signal("x", SYM, Direction(direction), 1.0 if direction != "hold" else 0.0, ts)]
    meta = {"risk_state": risk_state, "rationale": "r"} if risk_state else {"reason": "not a decision bar"}
    signals.append(Signal("qwen_risk", SYM, Direction.HOLD, 0.0, ts, meta))
    s.close_bar(ts, {SYM: Bar(close, close + 1, close - 1, close, 1000)}, {SYM: signals}, None,
                {SYM: market or {}})
    return [d for d in s.records.decisions if d.timestamp == ts and d.symbol == SYM]


def test_trend_filter_blocks_entries_below_the_average():
    s = session(trend_filter_period=50)
    blocked = bar_close(s, 1, 100.0, market={"trend_sma": 105.0})
    assert blocked[-1].action is DecisionAction.IGNORED and "trend filter" in blocked[-1].reason
    assert SYM not in s.pending
    unknown = bar_close(s, 2, 100.0, market={"trend_sma": None})
    assert "not enough history" in unknown[-1].reason  # unknown trend: no new entry
    allowed = bar_close(s, 3, 110.0, market={"trend_sma": 105.0})
    assert allowed[-1].action is DecisionAction.ENTER_SIGNAL


def test_risk_state_veto_and_its_age_limit():
    s = session(block_entries_on_risk_states=["high", "extreme"], risk_state_max_age_bars=3)
    vetoed = bar_close(s, 1, 100.0, risk_state="extreme")
    assert vetoed[-1].action is DecisionAction.IGNORED and "risk_state 'extreme'" in vetoed[-1].reason
    still = bar_close(s, 3, 100.0)  # the qwen_risk answer from bar 1 is still in force
    assert still[-1].action is DecisionAction.IGNORED
    expired = bar_close(s, 4, 100.0)  # 3 bars later: too old
    assert expired[-1].action is DecisionAction.ENTER_SIGNAL
    s.pending.clear()
    calm = bar_close(s, 5, 100.0, risk_state="moderate")
    assert calm[-1].action is DecisionAction.ENTER_SIGNAL


def test_filters_never_touch_exits_or_breakers():
    s = session(trend_filter_period=50, block_entries_on_risk_states=["extreme"])
    s.last_close[SYM] = 100.0
    s.pending[SYM] = Intent(DecisionAction.ENTER_SIGNAL, Signal("x", SYM, Direction.BUY, 1.0, T0))
    s.open_bar(T0 + H, {SYM: 100.0})
    assert s.portfolio.position(SYM) is not None
    held = bar_close(s, 1, 99.0, direction="hold", risk_state="extreme", market={"trend_sma": 120.0})
    assert s.portfolio.position(SYM) is not None and SYM not in s.pending  # no forced exit
    assert all(d.action is not DecisionAction.EXIT_SIGNAL for d in held)
    exit_ = bar_close(s, 2, 99.0, direction="sell", risk_state="extreme", market={"trend_sma": 120.0})
    assert exit_[-1].action is DecisionAction.EXIT_SIGNAL  # exits still work while filtered
    s2 = session()
    s2.breakers.state.halted_reason = "max drawdown"
    bar_close(s2, 1, 100.0, market={"trend_sma": 50.0})
    s2.open_bar(T0 + 2 * H, {SYM: 100.0})
    assert any("circuit breaker" in d.reason for d in s2.records.decisions)  # breakers still apply


def test_backtest_never_signals_entries_below_the_average():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
                                  "risk": {"trend_filter_period": 48}})
    result = BacktestEngine(cfg, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + 200 * H)
    plain = BacktestEngine(cfg.with_overrides({"risk": {"trend_filter_period": 0}}),
                           SyntheticProvider(seed=11, anchor=ANCHOR)).run(
        START, START + 200 * H)
    closes = result.closes()
    candles = {sym: SyntheticProvider(seed=11, anchor=ANCHOR).fetch_ohlcv(sym, "1h", START - 100 * H,
                                                                          START + 200 * H) for sym in closes}
    for d in result.decisions:
        if d.action is DecisionAction.ENTER_SIGNAL:
            frame = candles[d.symbol]["close"]
            sma = frame.rolling(48).mean()[frame.index <= d.timestamp].iloc[-1]
            assert frame[frame.index <= d.timestamp].iloc[-1] >= sma
    blocked = [d for d in result.decisions if "trend filter" in d.reason]
    assert blocked and len([d for d in result.decisions if d.action is DecisionAction.ENTER_SIGNAL]) < len(
        [d for d in plain.decisions if d.action is DecisionAction.ENTER_SIGNAL])


def test_live_matches_backtest_with_the_trend_filter():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
                                  "risk": {"trend_filter_period": 48}})
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


def test_risk_veto_from_the_agent_survives_a_resume(tmp_path):
    def extreme(role, user):
        return {"direction": "BUY", "confidence": 0.9, "rationale": "volatile", "risk_state": "extreme"}

    cfg = AppConfig().with_overrides({
        "market": {"symbols": ["BTC/USDT"]},
        "strategies": {"rsi": {"oversold": 98.0, "overbought": 99.0},  # rsi says BUY almost always
                       "qwen_risk": {"weight": 1.0, "decision_interval": 4, "lookback": 10}},
        "risk": {"block_entries_on_risk_states": ["extreme"], "risk_state_max_age_bars": 8},
        "agents": {"cache_path": str(tmp_path / "c.db")},
        "voting": {"min_agreeing": 1},
    })
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(tmp_path / "h.db") as store:
        t = LivePaperTrader(cfg, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock,
                            llm_provider=provider(RoleTransport(override=extreme)))
        for _ in range(6):
            t.run_cycle()
            clock.now += H
        t.stop()
        assert store.load_state(t.run_id)["risk_states"]["BTC/USDT"][1] == "extreme"
        resumed = LivePaperTrader.resume(store, t.run_id, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock),
                                         clock=clock, llm_provider=provider(RoleTransport(override=extreme)))
        assert resumed._session.risk_states["BTC/USDT"][1] == "extreme"
        for _ in range(6):
            resumed.run_cycle()
            clock.now += H
        decisions = store.load_decisions(t.run_id)
    assert not (decisions["action"] == "enter_signal").any()  # every BUY was vetoed
    assert decisions["reason"].str.contains("risk filter").any()


def test_validation():
    RiskConfig(trend_filter_period=200, block_entries_on_risk_states=["high", "extreme"])
    for bad in ({"trend_filter_period": -1}, {"block_entries_on_risk_states": ["panic"]},
                {"risk_state_max_age_bars": 0}, {"block_entries_on_risk_states": "extreme"}):
        with pytest.raises(ConfigError):
            RiskConfig(**bad)
    cfg = AppConfig.from_mapping({"risk": {"block_entries_on_risk_states": ["extreme"]}})
    assert AppConfig.from_dict(cfg.to_dict()) == cfg  # stored runs round-trip (resume)
