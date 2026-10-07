"""Stage 20B: the time stop (risk.max_holding_bars)."""

from datetime import timedelta

import pytest

from test_live import ANCHOR, START, Clock, fill_key
from test_shorts_trading import SYM, T0, actions, bar, close, open_, session, short
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, RiskConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import DecisionAction, Direction, Signal
from trading_lab.data import SyntheticProvider
from trading_lab.engine import Intent
from trading_lab.live import LivePaperTrader
from trading_lab.storage import SQLiteStore

H = timedelta(hours=1)


def long_(s, price=100.0, at=1):
    s.last_close[SYM] = price
    s.pending[SYM] = Intent(DecisionAction.ENTER_SIGNAL, Signal("x", SYM, Direction.BUY, 1.0, T0))
    open_(s, at, price)
    assert s.portfolio.position(SYM) is not None


def test_exit_after_the_limit_at_the_next_open():
    s = session(max_holding_bars=3)
    long_(s)  # entered at the open of bar 1
    close(s, 1)
    close(s, 2)
    assert not actions(s, DecisionAction.EXIT_SIGNAL)
    close(s, 3)  # third bar held
    scheduled = actions(s, DecisionAction.EXIT_SIGNAL)[0]
    assert scheduled.reason == "time stop: held 3 bars (limit 3); exit at the next open"
    open_(s, 4, 101.0)
    exit_ = actions(s, DecisionAction.EXIT)[0]
    assert exit_.reason == "time stop: held 3 bars" and not s.portfolio.positions
    assert s.portfolio.closed_trades[0].exit_price == 101.0


def test_strategies_cannot_cancel_it_and_shorts_too():
    s = session(max_holding_bars=2)
    short(s)
    close(s, 1)
    close(s, 2, Direction.SELL)  # a SELL (add to the short) at the same close changes nothing
    assert s.pending[SYM].signal.strategy == "risk_manager"
    open_(s, 3)
    assert not s.portfolio.positions and s.portfolio.closed_trades[0].side == "short"
    assert actions(s, DecisionAction.EXIT)[0].reason == "time stop: held 2 bars"


def test_off_by_default_and_other_exits_come_first():
    s = session()
    long_(s)
    for i in range(1, 30):
        close(s, i)
    assert s.portfolio.position(SYM) is not None and not actions(s, DecisionAction.EXIT_SIGNAL)
    s = session(max_holding_bars=2)
    long_(s)
    close(s, 1, b=bar(1, 100, h=101, low=90))  # stopped out in the first bar
    close(s, 2)
    assert not actions(s, DecisionAction.EXIT_SIGNAL) and actions(s, DecisionAction.STOP_LOSS)
    for bad in (-1, 1.5, True):
        with pytest.raises(ConfigError, match="max_holding_bars"):
            RiskConfig(max_holding_bars=bad)


CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}, "risk": {"max_holding_bars": 6}})


def test_backtest_live_and_resume_agree():
    bars = 150
    bt = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    time_exits = [d for d in bt.decisions if d.action is DecisionAction.EXIT and d.reason.startswith("time stop")]
    assert time_exits
    assert all((t.closed_at - t.opened_at) <= 6 * H for t in bt.trades)

    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        cycles = 0
        while cycles < bars:
            trader.run_cycle()
            clock.now += H
            cycles += 1
            held = [int((clock.now - H - p.opened_at) / H) for p in trader.portfolio.positions.values()]
            if held and max(held) >= 5:
                break  # stop one bar before a time stop is due
        assert cycles < bars
        trader.stop()
        resumed = LivePaperTrader.resume(store, trader.run_id,
                                         SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), clock=clock)
        for _ in range(bars - cycles):
            resumed.run_cycle()
            clock.now += H
        live = [fill_key(f)[1:] for f in resumed.portfolio.fills if f.timestamp < START + bars * H]
    assert live == [fill_key(f)[1:] for f in bt.fills]
