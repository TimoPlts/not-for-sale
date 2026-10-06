"""Stage 16B: simulated shorts in the trading session (off by default; backtest == live)."""

from datetime import datetime, timedelta, timezone

import pytest

from test_live import ANCHOR, START, Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, RiskConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import DecisionAction, Direction, Fill, Side, Signal
from trading_lab.data import SyntheticProvider
from trading_lab.engine import Bar, Intent, TradingSession
from trading_lab.ensemble import VotingEngine
from trading_lab.execution import CostModel
from trading_lab.live import LivePaperTrader
from trading_lab.portfolio import Portfolio
from trading_lab.risk import RiskManager
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
SYM = "BTC/USDT"


def session(allow_short=True, execution=None, **risk):
    cfg = AppConfig.from_mapping({
        "market": {"symbols": [SYM]},
        "execution": {"fee_rate": 0.0, "slippage_bps": 0.0, "short_borrow_bps_per_day": 0.0, **(execution or {})},
        "risk": {"stop_loss_pct": 0.05, "risk_per_trade_pct": 0.5, "max_position_pct": 0.5,
                 "allow_short": allow_short, **risk},
    })
    return TradingSession(cfg, VotingEngine({"x": 1.0}))


def bar(i, o=100.0, h=None, low=None, c=None):
    c = o if c is None else c
    return Bar(o, max(o, c) if h is None else h, min(o, c) if low is None else low, c, 1000)


def close(s, i, direction=Direction.HOLD, b=None, market=None):
    ts = T0 + i * H
    s.close_bar(ts, {SYM: b or bar(i)}, {SYM: [Signal("x", SYM, direction, 1.0 if direction is not Direction.HOLD
                                                      else 0.0, ts)]},
                market={SYM: market} if market else None)


def open_(s, i, price=100.0):
    s.open_bar(T0 + i * H, {SYM: price})


def short(s, price=100.0, at=1):
    s.last_close[SYM] = price
    s.pending[SYM] = Intent(DecisionAction.ENTER_SIGNAL, Signal("x", SYM, Direction.SELL, 1.0, T0))
    open_(s, at, price)
    assert s.portfolio.position(SYM).is_short


def actions(s, *kinds):
    return [d for d in s.records.decisions if d.action in kinds]


def stop(s):
    return s.portfolio.position(SYM).stop_price


def test_long_only_by_default():
    assert RiskConfig().allow_short is False and AppConfig().execution.short_borrow_bps_per_day == 2.0
    s = session(allow_short=False)
    close(s, 0, Direction.SELL)
    assert actions(s, DecisionAction.IGNORED)[0].reason == "SELL signal but no position (long-only)"
    assert not s.pending
    with pytest.raises(ConfigError, match="allow_short"):
        RiskConfig(allow_short="yes")
    with pytest.raises(ConfigError):
        AppConfig.from_mapping({"execution": {"short_borrow_bps_per_day": -1}})


def test_a_sell_signal_opens_a_short_and_a_buy_signal_covers_it():
    s = session()
    close(s, 0, Direction.SELL)
    assert actions(s, DecisionAction.ENTER_SIGNAL)[0].reason == "short entry scheduled for next bar open"
    open_(s, 1)
    pos = s.portfolio.position(SYM)
    assert pos.is_short and stop(s) == pytest.approx(105.0)  # stop above the entry
    enter = actions(s, DecisionAction.ENTER)[0]
    assert enter.reason.startswith("short entry approved") and s.records.reports[-1].order.reason == "enter_short"
    close(s, 1, Direction.BUY, bar(1, 100, c=95))
    assert actions(s, DecisionAction.EXIT_SIGNAL)[0].reason == "cover scheduled for next bar open"
    open_(s, 2, 95.0)
    trade = s.portfolio.closed_trades[0]
    assert trade.side == "short" and trade.pnl == pytest.approx(pos.quantity * 5)
    assert actions(s, DecisionAction.EXIT)[0].reason == "signal cover" and not s.portfolio.positions


def test_stops_trigger_on_the_high_and_gaps_fill_at_the_open():
    s = session()
    short(s)
    close(s, 1, b=bar(1, 100, h=104.9, low=99))
    assert s.portfolio.position(SYM) is not None
    close(s, 2, b=bar(2, 101, h=106, low=100))
    stop_exit = actions(s, DecisionAction.STOP_LOSS)[0]
    assert "stop 105 hit (bar high 106)" in stop_exit.reason and s.portfolio.closed_trades[0].exit_price == 105
    assert SYM in s.breakers.state.cooldown_until  # a losing stop starts the cooldown

    s = session()
    short(s)
    close(s, 1, b=bar(1, 108, h=110, low=107))  # gapped above the stop
    assert s.portfolio.closed_trades[0].exit_price == 108


def test_take_profit_on_the_low_and_the_stop_comes_first():
    s = session(take_profit_pct=0.10)
    short(s)
    close(s, 1, b=bar(1, 100, h=101, low=89))
    tp = actions(s, DecisionAction.TAKE_PROFIT)[0]
    assert tp.reference_price == pytest.approx(90.0) and "bar low 89" in tp.reason
    s = session(take_profit_pct=0.10)
    short(s)
    close(s, 1, b=bar(1, 88, h=89, low=87))  # gapped below the target: fills at the open
    assert s.portfolio.closed_trades[0].exit_price == 88
    s = session(take_profit_pct=0.10)
    short(s)
    close(s, 1, b=bar(1, 100, h=106, low=89))  # both reached: the stop is assumed first
    assert actions(s, DecisionAction.STOP_LOSS) and not actions(s, DecisionAction.TAKE_PROFIT)


def test_trailing_stops_follow_the_low_and_only_move_down():
    s = session(trailing_stop_pct=0.04, trailing_activation_pct=0.02)
    short(s)
    close(s, 1, b=bar(1, 100, h=101, low=99))  # -1% < activation
    assert stop(s) == pytest.approx(105.0) and s.trailing[SYM] == {"low": 99}
    close(s, 2, b=bar(2, 99, h=99.5, low=90))
    assert stop(s) == pytest.approx(93.6) and s.trailing[SYM]["stop"] == pytest.approx(93.6)
    close(s, 3, b=bar(3, 91, h=92, low=91))  # a higher low never raises the stop
    assert stop(s) == pytest.approx(93.6)
    close(s, 4, b=bar(4, 92, h=94, low=92))
    exit_ = actions(s, DecisionAction.STOP_LOSS)[0]
    assert "trailing stop" in exit_.reason and s.portfolio.closed_trades[0].pnl > 0
    assert s.breakers.state.cooldown_until == {} and SYM not in s.trailing


def test_a_trailing_stop_never_loosens_a_tighter_initial_stop():
    s = session(stop_loss_pct=0.01, trailing_stop_pct=0.04)
    short(s)
    close(s, 1, b=bar(1, 100, h=100.5, low=99.5))  # trailing candidate 103.48 is looser than 101
    assert stop(s) == pytest.approx(101.0) and "stop" not in s.trailing[SYM]


def test_a_long_is_closed_before_any_short():
    s = session()
    s.last_close[SYM] = 100.0
    s.pending[SYM] = Intent(DecisionAction.ENTER_SIGNAL, Signal("x", SYM, Direction.BUY, 1.0, T0))
    open_(s, 1)
    close(s, 1, Direction.SELL)
    assert [d.reason for d in actions(s, DecisionAction.EXIT_SIGNAL)] == ["exit scheduled for next bar open"]
    open_(s, 2)
    assert not s.portfolio.positions and s.portfolio.closed_trades[0].side == "long"
    close(s, 2, Direction.SELL)  # flat now: the next SELL opens a short
    open_(s, 3)
    assert s.portfolio.position(SYM).is_short


def test_pyramiding_and_limits_on_shorts():
    s = session()
    short(s)
    close(s, 1, Direction.SELL)
    assert actions(s, DecisionAction.IGNORED)[-1].reason == "SELL signal but short position already open"
    rm = RiskManager(RiskConfig(), CostModel.from_config(AppConfig().execution))
    decision = rm.evaluate_entry(SYM, 100.0, Portfolio(10_000), {}, side=Side.SELL)
    assert not decision.approved and "allow_short" in decision.reason


def test_short_sizing_mirrors_long_sizing():
    cfg = AppConfig.from_mapping({"risk": {"risk_per_trade_pct": 0.01, "stop_loss_pct": 0.05, "allow_short": True,
                                           "max_position_pct": 1.0}})
    rm = RiskManager(cfg.risk, CostModel.from_config(cfg.execution))
    p = Portfolio(10_000, allow_short=True)
    long_ = rm.evaluate_entry(SYM, 100.0, p, {})
    short_ = rm.evaluate_entry(SYM, 100.0, p, {}, side=Side.SELL)
    assert short_.approved and short_.side is Side.SELL and short_.stop_price > 100 > long_.stop_price
    loss = short_.sizing["loss_per_unit_at_stop"] * short_.quantity
    assert loss == pytest.approx(100.0)  # 1% of equity, costs included
    assert short_.quantity == pytest.approx(long_.quantity, rel=0.02)
    p.apply_fill(Fill("o", "ETH/USDT", Side.SELL, 20, 200, 200, 0, T0))
    capped = rm.evaluate_entry(SYM, 100.0, p, {"ETH/USDT": 200.0}, side=Side.SELL)
    assert capped.sizing["max_total_exposure"] == pytest.approx((p.equity({"ETH/USDT": 200}) - 4000) / 99.95,
                                                                 rel=1e-3)  # short exposure counts


def test_entry_filters_are_mirrored():
    s = session(trend_filter_period=20)
    close(s, 0, Direction.SELL, bar(0, 100), market={"trend_sma": 95.0})
    assert "close 100 above its 20-bar average 95" in actions(s, DecisionAction.IGNORED)[0].reason
    close(s, 1, Direction.SELL, bar(1, 100), market={"trend_sma": 105.0})
    assert actions(s, DecisionAction.ENTER_SIGNAL)
    long_only = session(allow_short=False, trend_filter_period=20)
    close(long_only, 0, Direction.BUY, bar(0, 100), market={"trend_sma": 105.0})
    assert "below its 20-bar average" in actions(long_only, DecisionAction.IGNORED)[0].reason


def test_the_kill_switch_covers_shorts():
    s = session(max_drawdown_pct=0.05, flatten_on_halt=True)
    short(s)
    close(s, 1, b=bar(1, 100, h=104, low=99, c=104))  # a 4% move against half the equity: -2%, no halt
    assert not s.breakers.halted
    s.portfolio.set_stop(SYM, 1e9)  # keep the position through a big move
    close(s, 2, b=bar(2, 104, h=150, low=104, c=150))
    assert s.breakers.halted and actions(s, DecisionAction.EXIT_SIGNAL)[-1].reason == "kill switch: closing position"
    open_(s, 3, 150.0)
    assert not s.portfolio.positions and s.portfolio.closed_trades[0].side == "short"


def test_limit_short_entries():
    s = session(execution={"entry_order_type": "limit", "limit_offset_bps": 100, "limit_ttl_bars": 2,
                           "maker_fee_rate": 0.001})
    close(s, 0, Direction.SELL)
    open_(s, 1)
    placed = actions(s, DecisionAction.ORDER_PLACED)[0]
    assert placed.reason.startswith("limit sell") and s.resting[SYM].limit_price == pytest.approx(101.0)
    assert s.resting[SYM].to_json()["side"] == "sell"
    close(s, 1, b=bar(1, 100, h=100.9, low=99))  # not traded through
    assert s.portfolio.position(SYM) is None
    close(s, 2, b=bar(2, 100, h=102, low=99))
    pos = s.portfolio.position(SYM)
    assert pos.is_short and s.portfolio.fills[0].fill_price == pytest.approx(101.0)
    assert s.portfolio.fills[0].fee == pytest.approx(pos.entry_notional * 0.001)

    s = session(execution={"entry_order_type": "limit"})
    close(s, 0, Direction.SELL)
    open_(s, 1)
    close(s, 1, Direction.BUY)  # a BUY signal cancels the working short entry
    assert SYM not in s.resting and "cancelled by BUY signal" in actions(s, DecisionAction.EXPIRED)[0].reason


def test_the_portfolio_view_shows_the_short():
    s = session()
    short(s)
    s.last_close[SYM] = 98.0
    view = s.portfolio_view(SYM, T0 + H)
    assert view["position"] == "short" and view["stop_distance_pct"] == pytest.approx(7.1)
    assert view["position_return_pct"] > 0 and view["exposure_pct"] > 0


# --------------------------------------------------------------- end to end
CFG = AppConfig.from_mapping({
    "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
    "risk": {"allow_short": True, "trailing_stop_pct": 0.01, "trailing_activation_pct": 0.002,
             "take_profit_pct": 0.015},
})


def run_live(store, clock, cycles, trader=None):
    trader = trader or LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
    for _ in range(cycles):
        trader.run_cycle()
        clock.now += H
    return trader


def test_backtest_and_live_agree_with_shorts():
    bars = 150
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = run_live(store, clock, bars)
        live = [fill_key(f)[1:] for f in trader.portfolio.fills if f.timestamp < START + bars * H]
        stored_sides = {t.side for t in store.load_closed_trades(trader.run_id)}
    bt = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    assert live == [fill_key(f)[1:] for f in bt.fills]
    assert {t.side for t in bt.trades} == {"long", "short"} == stored_sides
    covers = [f for f in bt.fills if f.side is Side.BUY and any(
        t.side == "short" and t.exit_order_id == f.order_id for t in bt.trades)]
    assert covers and all(f.fee > f.notional * CFG.execution.fee_rate for f in covers)  # borrow fee included
    assert bt.metrics.final_equity == pytest.approx(
        CFG.portfolio.initial_cash + sum(t.pnl for t in bt.trades) + bt.equity_curve["unrealized_pnl"].iloc[-1])


def test_a_short_survives_a_resume(tmp_path):
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(tmp_path / "full.db") as store:
        full = [fill_key(f) for f in run_live(store, clock, 120).portfolio.fills]

    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(tmp_path / "split.db") as store:
        part = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        cycles = 0
        while cycles < 120:
            part.run_cycle()
            clock.now += H
            cycles += 1
            if any(p.is_short for p in part.portfolio.positions.values()):
                break
        assert cycles < 120
        part.stop()
        resumed = LivePaperTrader.resume(store, part.run_id,
                                         SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), clock=clock)
        assert any(p.is_short for p in resumed.portfolio.positions.values())
        run_live(store, clock, 120 - cycles, resumed)
        assert [fill_key(f) for f in resumed.portfolio.fills] == full
