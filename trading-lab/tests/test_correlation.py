"""Stage 13B: correlation-aware entry limit (blocks new entries only)."""

from datetime import datetime, timedelta, timezone

import pytest

from test_live import ANCHOR, START, Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, RiskConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import DecisionAction, Direction, Side, Signal
from trading_lab.data import SyntheticProvider
from trading_lab.engine import Bar, Intent, TradingSession
from trading_lab.engine.filters import correlation_lookup
from trading_lab.ensemble import VotingEngine
from trading_lab.live import LivePaperTrader
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


class Twins(SyntheticProvider):
    """ETH/USDT is an exact copy of BTC/USDT at 1/20 of the price: correlation 1."""

    def fetch_ohlcv(self, symbol, timeframe, since, until=None):
        if symbol == "ETH/USDT":
            frame = super().fetch_ohlcv("BTC/USDT", timeframe, since, until).copy()
            frame[["open", "high", "low", "close"]] /= 20
            return frame
        return super().fetch_ohlcv(symbol, timeframe, since, until)

    def current_open(self, symbol, timeframe, bar_open):
        if symbol == "ETH/USDT":
            price = super().current_open("BTC/USDT", timeframe, bar_open)
            return None if price is None else price / 20
        return super().current_open(symbol, timeframe, bar_open)


RISK = {"max_correlated_positions": 1, "correlation_threshold": 0.8, "correlation_lookback": 24}


def test_correlation_lookup_is_causal_and_correct():
    market = Twins(seed=4)
    candles = {s: market.fetch_ohlcv(s, "1h", T0, T0 + 200 * H) for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT")}
    lookup = correlation_lookup(candles, RiskConfig(**RISK))
    t = candles["BTC/USDT"].index[100]
    assert lookup["BTC/USDT"][t]["ETH/USDT"] == pytest.approx(1.0)
    assert abs(lookup["BTC/USDT"][t]["SOL/USDT"]) < 0.6  # independent random walks
    assert lookup["ETH/USDT"][candles["BTC/USDT"].index[10]]["BTC/USDT"] is None  # not enough bars yet
    changed = {s: f.copy() for s, f in candles.items()}
    changed["SOL/USDT"].iloc[150:, :] *= 3
    later = correlation_lookup(changed, RiskConfig(**RISK))
    assert later["BTC/USDT"][t]["SOL/USDT"] == pytest.approx(lookup["BTC/USDT"][t]["SOL/USDT"])
    assert correlation_lookup(candles, RiskConfig()) == {s: {} for s in candles}  # off by default


def session(**risk):
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]},
                                  "risk": {"max_position_pct": 0.3, **risk}})
    return TradingSession(cfg, VotingEngine({"x": 1.0}))


def hold(s, sym, price=100.0):
    s.last_close.update({k: price for k in ("BTC/USDT", "ETH/USDT", "SOL/USDT")})
    s.pending[sym] = Intent(DecisionAction.ENTER_SIGNAL, Signal("x", sym, Direction.BUY, 1.0, T0))
    s.open_bar(T0 + H, {sym: price})
    assert s.portfolio.position(sym) is not None


def reason(s, sym, correlations):
    return s.entry_filter_reason(sym, T0 + 2 * H, 100.0, {"correlations": correlations})


def test_entry_is_blocked_when_it_moves_with_a_held_position():
    s = session(**RISK)
    assert reason(s, "ETH/USDT", {}) is None  # nothing held yet
    hold(s, "BTC/USDT")
    assert "BTC/USDT 0.93" in reason(s, "ETH/USDT", {"BTC/USDT": 0.93})
    assert reason(s, "SOL/USDT", {"BTC/USDT": 0.2}) is None
    assert "unknown" in reason(s, "SOL/USDT", {"BTC/USDT": None})  # unknown counts as correlated
    two = session(**{**RISK, "max_correlated_positions": 2})
    hold(two, "BTC/USDT")
    assert reason(two, "ETH/USDT", {"BTC/USDT": 0.95}) is None  # one correlated position is allowed


def test_scheduled_entries_count_within_the_same_bar():
    s = session(**RISK)
    ts = T0 + 2 * H
    bars = {sym: Bar(100, 101, 99, 100, 1000) for sym in ("BTC/USDT", "ETH/USDT")}
    signals = {sym: [Signal("x", sym, Direction.BUY, 1.0, ts)] for sym in bars}
    market = {"BTC/USDT": {"correlations": {"ETH/USDT": 0.99}}, "ETH/USDT": {"correlations": {"BTC/USDT": 0.99}}}
    s.close_bar(ts, bars, signals, None, market)
    actions = {d.symbol: d for d in s.records.decisions if d.timestamp == ts}
    assert actions["BTC/USDT"].action is DecisionAction.ENTER_SIGNAL
    assert actions["ETH/USDT"].action is DecisionAction.IGNORED and "correlation filter" in actions["ETH/USDT"].reason


def holdings_overlap(fills):
    held, overlap = set(), False
    for f in sorted(fills, key=lambda f: (f.timestamp, f.side is Side.BUY)):  # exits before entries
        if f.side is Side.BUY:
            held.add(f.symbol)
        else:
            held.discard(f.symbol)
        overlap |= len(held) > 1
    return overlap


def test_twins_are_never_held_together():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}, "risk": RISK})
    filtered = BacktestEngine(cfg, Twins(seed=11, anchor=ANCHOR)).run(START, START + 300 * H)
    free = BacktestEngine(cfg.with_overrides({"risk": {**RISK, "max_correlated_positions": 0}}),
                          Twins(seed=11, anchor=ANCHOR)).run(START, START + 300 * H)
    assert holdings_overlap(free.fills)  # without the limit both twins are held at once...
    assert not holdings_overlap(filtered.fills)  # ...with it, never
    assert any("correlation filter" in d.reason for d in filtered.decisions)


def test_live_matches_backtest_with_the_correlation_limit():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}, "risk": RISK})
    bars = 150
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(cfg, Twins(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(bars):
            trader.run_cycle()
            clock.now += H
        live = [fill_key(f)[1:] for f in trader.portfolio.fills if f.timestamp < START + bars * H]
    bt = BacktestEngine(cfg, Twins(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    assert live == [fill_key(f)[1:] for f in bt.fills] and len(live) > 3


def test_validation():
    for bad in ({"max_correlated_positions": -1}, {"correlation_threshold": 1.5}, {"correlation_lookback": 3}):
        with pytest.raises(ConfigError):
            RiskConfig(**bad)
