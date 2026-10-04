"""Circuit breaker tests: kill switch, daily loss limit, stop-loss cooldown, persistence."""

from datetime import datetime, timedelta, timezone

import pytest

from test_backtest import BUY, SELL, H, T0, bars, config, run_scripted
from trading_lab.core.models import DecisionAction, Side
from trading_lab.engine.session import PORTFOLIO
from trading_lab.risk.breakers import BreakerState

FLAT = (100, 101, 99, 100)
ALL_IN = {"max_position_pct": 1.0, "risk_per_trade_pct": 1.0, "stop_loss_pct": 0.5}


def risk(**kwargs):
    base = {**ALL_IN, "max_drawdown_pct": 0.0, "daily_loss_limit_pct": 0.0, "stop_loss_cooldown_bars": 0}
    return config(risk={**base, **kwargs})


def rejected(result):
    return [d for d in result.decisions if d.action is DecisionAction.REJECTED]


def test_kill_switch_blocks_new_entries_but_not_exits():
    frame = bars([FLAT, FLAT, (90, 91, 84, 85), FLAT, FLAT, FLAT, FLAT, FLAT])
    script = {("BTC/USDT", 0): BUY, ("BTC/USDT", 3): SELL, ("BTC/USDT", 5): BUY}
    r = run_scripted({"BTC/USDT": frame}, script, risk(max_drawdown_pct=0.10))
    trips = [d for d in r.decisions if d.action is DecisionAction.CIRCUIT_BREAKER]
    assert len(trips) == 1 and trips[0].symbol == PORTFOLIO and "max drawdown" in trips[0].reason
    assert trips[0].timestamp == T0 + 2 * H
    assert [f.side for f in r.fills] == [Side.BUY, Side.SELL]  # the exit still happened
    (blocked,) = rejected(r)
    assert blocked.timestamp == T0 + 6 * H and "circuit breaker" in blocked.reason


def test_flatten_on_halt_closes_positions():
    frame = bars([FLAT, FLAT, (90, 91, 84, 85), (86, 87, 85, 86), FLAT])
    script = {("BTC/USDT", 0): BUY, ("BTC/USDT", 2): BUY}  # a BUY cannot override the kill switch
    r = run_scripted({"BTC/USDT": frame}, script, risk(max_drawdown_pct=0.10, flatten_on_halt=True))
    exit_fill = r.fills[-1]
    assert exit_fill.side is Side.SELL and exit_fill.timestamp == T0 + 3 * H
    assert exit_fill.fill_price == 86
    assert r.equity_curve["open_positions"].iloc[-1] == 0


def test_daily_loss_limit_resets_next_utc_day():
    rows = [FLAT] * 26
    rows[2] = (90, 91, 89, 90)  # -10% on 2024-03-01
    frame = bars(rows)
    script = {
        ("BTC/USDT", 0): BUY,
        ("BTC/USDT", 3): SELL,
        ("BTC/USDT", 5): BUY,   # entry at 06:00 the same day: blocked
        ("BTC/USDT", 23): BUY,  # entry at 00:00 the next day: allowed
    }
    r = run_scripted({"BTC/USDT": frame}, script, risk(daily_loss_limit_pct=0.05))
    trips = [d for d in r.decisions if d.action is DecisionAction.CIRCUIT_BREAKER]
    assert len(trips) == 1 and "daily loss" in trips[0].reason
    (blocked,) = rejected(r)
    assert blocked.timestamp == T0 + 6 * H and "daily loss limit" in blocked.reason
    assert r.fills[-1].side is Side.BUY and r.fills[-1].timestamp == T0 + 24 * H


def test_stop_loss_cooldown_blocks_reentry_for_n_bars():
    rows = [FLAT] * 9
    rows[2] = (60, 61, 40, 45)  # gap through the 50% stop
    frame = bars(rows)
    script = {
        ("BTC/USDT", 0): BUY,
        ("BTC/USDT", 2): BUY,   # entry at bar 3: in cooldown
        ("BTC/USDT", 4): BUY,   # entry at bar 5: in cooldown
        ("BTC/USDT", 5): BUY,   # entry at bar 6: allowed (3 bars after the stop bar)
    }
    r = run_scripted({"BTC/USDT": frame}, script, risk(stop_loss_cooldown_bars=3))
    assert any(d.action is DecisionAction.STOP_LOSS for d in r.decisions)
    assert [d.timestamp for d in rejected(r)] == [T0 + 3 * H, T0 + 5 * H]
    assert all("cooldown" in d.reason for d in rejected(r))
    assert r.fills[-1].side is Side.BUY and r.fills[-1].timestamp == T0 + 6 * H


def test_breakers_disabled_with_zero():
    frame = bars([FLAT, FLAT, (50, 51, 49, 50), FLAT, FLAT])
    script = {("BTC/USDT", 0): BUY, ("BTC/USDT", 2): SELL, ("BTC/USDT", 3): BUY}
    r = run_scripted({"BTC/USDT": frame}, script, risk(stop_loss_pct=0.9))
    assert not any(d.action is DecisionAction.CIRCUIT_BREAKER for d in r.decisions)
    assert not rejected(r)


def test_breaker_state_json_round_trip():
    state = BreakerState(
        peak_equity=12_000.0,
        day=datetime(2024, 3, 1).date(),
        day_start_equity=11_000.0,
        last_equity=10_500.0,
        halted_reason="max drawdown 26%",
        daily_blocked_day=datetime(2024, 3, 1).date(),
        cooldown_until={"BTC/USDT": datetime(2024, 3, 1, 5, tzinfo=timezone.utc)},
    )
    assert BreakerState.from_json(state.to_json()) == state


@pytest.mark.parametrize(
    "values",
    [{"max_drawdown_pct": -0.1}, {"daily_loss_limit_pct": 1.5}, {"stop_loss_cooldown_bars": -1},
     {"stop_loss_cooldown_bars": 1.5}, {"flatten_on_halt": "yes"}],
)
def test_invalid_breaker_config(values):
    from trading_lab.config import AppConfig
    from trading_lab.core.errors import ConfigError

    with pytest.raises(ConfigError):
        AppConfig.from_mapping({"risk": values})


def test_live_run_keeps_breaker_state_across_resume(tmp_path):
    from test_live import CFG, START, Clock, provider
    from trading_lab.live import LivePaperTrader
    from trading_lab.storage import SQLiteStore

    cfg = CFG.with_overrides({"risk": {"max_drawdown_pct": 0.001}})
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(tmp_path / "b.db") as store:
        trader = LivePaperTrader(cfg, provider(clock), store, clock=clock)
        for _ in range(200):
            trader.run_cycle()
            clock.now += H
            if trader._session.breakers.halted:
                break
        assert trader._session.breakers.halted
        trader.stop()
        resumed = LivePaperTrader.resume(store, trader.run_id, provider(clock), clock=clock)
        assert resumed._session.breakers.halted
        assert resumed._session.breakers.state == trader._session.breakers.state
