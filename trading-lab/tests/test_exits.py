"""Stage 11A: trailing stops and take-profit (long-only, no look-ahead, resumable)."""

from datetime import datetime, timedelta, timezone

import pytest

from test_live import ANCHOR, START, Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, RiskConfig
from trading_lab.core.errors import ConfigError, PositionError
from trading_lab.core.models import DecisionAction, Direction, Signal
from trading_lab.dashboard import DashboardData
from trading_lab.data import SyntheticProvider
from trading_lab.engine import Bar, Intent, TradingSession
from trading_lab.ensemble import VotingEngine
from trading_lab.live import LivePaperTrader
from trading_lab.portfolio import Portfolio
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
SYM = "BTC/USDT"


def session(**risk):
    cfg = AppConfig.from_mapping({
        "market": {"symbols": [SYM]},
        "execution": {"fee_rate": 0.0, "slippage_bps": 0.0},
        "risk": {"stop_loss_pct": 0.05, "risk_per_trade_pct": 0.5, "max_position_pct": 0.5, **risk},
    })
    return TradingSession(cfg, VotingEngine({"x": 1.0}))


def enter(s, price=100.0, at=1):
    s.last_close[SYM] = price
    s.pending[SYM] = Intent(DecisionAction.ENTER_SIGNAL, Signal("x", SYM, Direction.BUY, 1.0, T0))
    s.open_bar(T0 + at * H, {SYM: price})
    assert s.portfolio.position(SYM) is not None


def close(s, i, o, h, low, c):
    ts = T0 + i * H
    s.close_bar(ts, {SYM: Bar(o, h, low, c, 1000)}, {SYM: [Signal("x", SYM, Direction.HOLD, 0.0, ts)]})


def actions(s, *kinds):
    return [d for d in s.records.decisions if d.action in kinds]


def stop(s):
    return s.portfolio.position(SYM).stop_price


def test_trailing_stop_ratchets_up_and_exits_in_profit():
    s = session(trailing_stop_pct=0.04)
    enter(s)
    assert stop(s) == pytest.approx(95.0)
    close(s, 1, 100, 110, 99, 108)  # high 110 -> stop 105.6 from the next bar
    assert stop(s) == pytest.approx(105.6) and s.trailing[SYM] == {"high": 110, "stop": pytest.approx(105.6)}
    close(s, 2, 108, 109, 106, 107)  # lower high: the stop never moves down
    assert stop(s) == pytest.approx(105.6)
    close(s, 3, 107, 108, 104, 105)  # low 104 reaches the trailed stop
    exit_ = actions(s, DecisionAction.STOP_LOSS)[0]
    assert "trailing stop" in exit_.reason and exit_.reference_price == pytest.approx(105.6)
    trade = s.portfolio.closed_trades[-1]
    assert trade.pnl > 0 and trade.exit_price == pytest.approx(105.6)
    assert s.breakers.state.cooldown_until == {}  # no cooldown after a profitable exit
    assert SYM not in s.trailing


def test_a_raised_stop_only_applies_from_the_next_bar():
    s = session(trailing_stop_pct=0.04)
    enter(s)
    close(s, 1, 100, 120, 96, 100)  # high 120 and low 96 in one bar: the order inside the bar is unknown
    assert s.portfolio.position(SYM) is not None and stop(s) == pytest.approx(115.2)
    close(s, 2, 100, 101, 99, 100)  # opened below the new stop: exits at the open (gap rule)
    assert s.portfolio.closed_trades[-1].exit_price == pytest.approx(100.0)


def test_activation_threshold():
    s = session(trailing_stop_pct=0.04, trailing_activation_pct=0.05)
    enter(s)
    close(s, 1, 100, 104, 99, 103)  # +4% < +5%: not trailing yet
    assert stop(s) == pytest.approx(95.0) and "stop" not in s.trailing[SYM]
    close(s, 2, 103, 106, 102, 105)
    assert stop(s) == pytest.approx(106 * 0.96)


def test_take_profit_and_gaps():
    s = session(take_profit_pct=0.10)
    enter(s)
    close(s, 1, 100, 111, 99, 105)
    tp = actions(s, DecisionAction.TAKE_PROFIT)[0]
    assert tp.reference_price == pytest.approx(110.0) and s.portfolio.closed_trades[-1].exit_price == pytest.approx(110)

    s = session(take_profit_pct=0.10)
    enter(s)
    close(s, 1, 112, 115, 111, 113)  # gapped above the target: fills at the open
    assert s.portfolio.closed_trades[-1].exit_price == pytest.approx(112.0)


def test_stop_comes_first_when_both_are_reached():
    s = session(take_profit_pct=0.10)
    enter(s)
    close(s, 1, 100, 111, 94, 100)
    assert actions(s, DecisionAction.STOP_LOSS) and not actions(s, DecisionAction.TAKE_PROFIT)
    assert s.portfolio.closed_trades[-1].exit_price == pytest.approx(95.0)
    assert SYM in s.breakers.state.cooldown_until  # a losing stop still starts the cooldown


def test_defaults_change_nothing():
    s = session()
    enter(s)
    close(s, 1, 100, 150, 99, 140)
    assert stop(s) == pytest.approx(95.0) and s.portfolio.position(SYM) is not None


def test_validation():
    RiskConfig(trailing_stop_pct=0.05, trailing_activation_pct=0.02, take_profit_pct=0.2)
    for bad in ({"trailing_stop_pct": 1.0}, {"trailing_stop_pct": -0.1}, {"take_profit_pct": -1},
                {"trailing_activation_pct": "x"}):
        with pytest.raises(ConfigError):
            RiskConfig(**bad)
    portfolio = Portfolio(1000.0)
    with pytest.raises(PositionError):
        portfolio.set_stop(SYM, 10.0)


CFG = AppConfig.from_mapping({
    "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
    "risk": {"trailing_stop_pct": 0.01, "trailing_activation_pct": 0.002, "take_profit_pct": 0.015},
})


def test_backtest_and_live_agree_with_trailing_exits():
    bars = 120
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(bars):
            trader.run_cycle()
            clock.now += H
        live = [fill_key(f)[1:] for f in trader.portfolio.fills if f.timestamp < START + bars * H]
    bt = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    assert live == [fill_key(f)[1:] for f in bt.fills]
    kinds = {d.action for d in bt.decisions}
    assert DecisionAction.TAKE_PROFIT in kinds and DecisionAction.STOP_LOSS in kinds
    assert any("trailing stop" in d.reason for d in bt.decisions if d.action is DecisionAction.STOP_LOSS)


def test_trailed_stops_survive_a_resume(tmp_path):
    def drive(trader, clock, n):
        for _ in range(n):
            trader.run_cycle()
            clock.now += H

    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(tmp_path / "full.db") as store:
        full = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        drive(full, clock, 90)
        full_fills = [fill_key(f) for f in full.portfolio.fills]

    clock = Clock(START + H + timedelta(minutes=1))
    db = tmp_path / "split.db"
    with SQLiteStore(db) as store:
        part = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        cycles = 0
        while cycles < 90:  # stop at a moment when a trailed stop is active
            part.run_cycle()
            clock.now += H
            cycles += 1
            if any("stop" in v for v in part._session.trailing.values()):
                break
        trailed = {s: v["stop"] for s, v in part._session.trailing.items() if "stop" in v}
        assert trailed and cycles < 90
        part.stop()
        resumed = LivePaperTrader.resume(store, part.run_id, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock),
                                         clock=clock)
        for sym, value in trailed.items():
            assert resumed.portfolio.position(sym).stop_price == pytest.approx(value)
        drive(resumed, clock, 90 - cycles)
        assert [fill_key(f) for f in resumed.portfolio.fills] == full_fills
        run_id = part.run_id
    with DashboardData(db) as data:
        positions = data.open_positions(run_id)
    state_trailing = SQLiteStore(db, readonly=True).load_state(run_id)["trailing"]
    for p in positions:
        expected = state_trailing.get(p["symbol"], {}).get("stop")
        if expected is not None:
            assert p["stop_price"] == pytest.approx(expected) and p["trailing"]
