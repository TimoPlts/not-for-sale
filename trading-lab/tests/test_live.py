"""Live paper trader tests, driven by a fake clock over synthetic data."""

from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.errors import DataError
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
ANCHOR = datetime(2024, 1, 1, tzinfo=UTC)
START = datetime(2024, 3, 1, tzinfo=UTC)  # first bar the trader acts on
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]}})


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def provider(clock, seed=11, cls=SyntheticProvider):
    return cls(seed=seed, anchor=ANCHOR, clock=clock)


def fill_key(f):
    return (f.order_id, f.timestamp, f.symbol, f.side, round(f.quantity, 9), round(f.fill_price, 6))


def backtest_fills(bars, seed=11):
    result = BacktestEngine(CFG, SyntheticProvider(seed=seed, anchor=ANCHOR)).run(START, START + bars * H)
    return [fill_key(f) for f in result.fills], result


def live_fills(trader, bars):
    """Fills within the first ``bars`` bars; later ones are beyond the backtest's data."""
    cutoff = START + bars * H
    return [fill_key(f) for f in trader.portfolio.fills if f.timestamp < cutoff]


def new_trader(store, clock, prov=None):
    return LivePaperTrader(CFG, prov or provider(clock), store, clock=clock)


@pytest.fixture
def store():
    with SQLiteStore(":memory:") as s:
        yield s


def drive(trader, clock, cycles, step=H):
    reports = []
    for _ in range(cycles):
        reports.append(trader.run_cycle())
        clock.now += step
    return reports


def test_live_paper_trading_matches_backtest(store):
    clock = Clock(START + H + timedelta(minutes=1))  # bar START has just closed
    trader = new_trader(store, clock)
    drive(trader, clock, 150)
    expected, bt = backtest_fills(150)
    live = live_fills(trader, 150)
    assert len(live) > 10
    assert [k[1:] for k in live] == [k[1:] for k in expected]
    curve = store.load_equity_curve(trader.run_id)
    assert len(curve) == 150
    assert curve["equity"].iloc[-1] == pytest.approx(bt.equity_curve["equity"].iloc[-1], rel=1e-9)


def test_entries_fill_immediately_at_the_new_candle_open(store):
    clock = Clock(START + H + timedelta(minutes=1))
    trader = new_trader(store, clock)
    for report in drive(trader, clock, 150):
        for f in report.fills:
            if f.reference_price != f.stop_price:  # ignore stop exits, filled within the bar
                # Filled during the cycle that processed the previous bar's close.
                assert f.timestamp == report.last_bar + H


def test_catch_up_after_downtime_gives_the_same_result(store):
    clock = Clock(START + H + timedelta(minutes=1))
    trader = new_trader(store, clock)
    trader.run_cycle()
    clock.now += 7 * H  # the trader was offline for a while
    reports = drive(trader, clock, 22, step=7 * H)
    assert reports[0].new_bars == 7
    processed = 1 + sum(r.new_bars for r in reports)
    expected, _ = backtest_fills(processed)
    assert [k[1:] for k in live_fills(trader, processed)] == [k[1:] for k in expected]


def test_stop_and_resume_equals_uninterrupted_run(tmp_path):
    db_a, db_b = tmp_path / "a.db", tmp_path / "b.db"
    clock_a = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(db_a) as store:
        uninterrupted = new_trader(store, clock_a)
        drive(uninterrupted, clock_a, 120)
        reference = [fill_key(f) for f in uninterrupted.portfolio.fills]
        reference_cash = uninterrupted.portfolio.cash

    clock_b = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(db_b) as store:
        first = new_trader(store, clock_b)
        run_id = first.run_id
        drive(first, clock_b, 60)
        first.stop()
        assert store.get_run(run_id)["status"] == "stopped"
    with SQLiteStore(db_b) as store:  # a fresh process
        resumed = LivePaperTrader.resume(store, run_id, provider(clock_b), clock=clock_b)
        assert store.get_run(run_id)["status"] == "running"
        drive(resumed, clock_b, 60)
        assert [fill_key(f) for f in resumed.portfolio.fills] == reference  # incl. order ids
        assert resumed.portfolio.cash == pytest.approx(reference_cash)
        assert store.count("equity_snapshots", run_id) == 120
        assert len(store.load_closed_trades(run_id)) == len(resumed.portfolio.closed_trades)


def test_pending_order_survives_restart_without_current_open(tmp_path):
    class NoOpen(SyntheticProvider):
        def current_open(self, *args):
            return None

    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(tmp_path / "p.db") as store:
        trader = new_trader(store, clock, provider(clock, cls=NoOpen))
        run_id = trader.run_id
        while not trader.pending_symbols:  # run until an order is scheduled
            trader.run_cycle()
            clock.now += H
        pending = trader.pending_symbols
        trader.stop()
        resumed = LivePaperTrader.resume(store, run_id, provider(clock, cls=NoOpen), clock=clock)
        assert resumed.pending_symbols == pending
        processed = len(store.load_equity_curve(run_id))
        drive(resumed, clock, 30)
        processed += 30
    expected, _ = backtest_fills(processed)
    assert [k[1:] for k in live_fills(resumed, processed)] == [k[1:] for k in expected]


def test_data_errors_are_reported_and_recovered(store):
    class Flaky(SyntheticProvider):
        fail = False

        def fetch_ohlcv(self, *args, **kwargs):
            if self.fail:
                raise DataError("exchange unreachable")
            return super().fetch_ohlcv(*args, **kwargs)

    clock = Clock(START + H + timedelta(minutes=1))
    prov = provider(clock, cls=Flaky)
    trader = new_trader(store, clock, prov)
    drive(trader, clock, 20)
    prov.fail = True
    failed = drive(trader, clock, 3)
    assert all(r.error and "unreachable" in r.error for r in failed)
    prov.fail = False
    recovered = trader.run_cycle()
    assert recovered.new_bars == 4 and recovered.error is None
    expected, _ = backtest_fills(24)
    assert [k[1:] for k in live_fills(trader, 24)] == [k[1:] for k in expected]


def test_no_new_candle_means_no_new_records(store):
    clock = Clock(START + H + timedelta(minutes=1))
    trader = new_trader(store, clock)
    trader.run_cycle()
    clock.now += timedelta(minutes=10)  # same candle still forming
    report = trader.run_cycle()
    assert report.new_bars == 0
    assert store.count("equity_snapshots", trader.run_id) == 1


def test_run_forever_stops_cleanly_on_ctrl_c(store):
    clock = Clock(START + H + timedelta(minutes=1))
    trader = new_trader(store, clock)
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 3:
            raise KeyboardInterrupt
        clock.now += H

    cycles = trader.run_forever(sleep=sleep)
    assert cycles == 3
    run = store.get_run(trader.run_id)
    assert run["status"] == "stopped" and run["finished_at"]
    assert store.load_metrics(trader.run_id)["num_bars"] == 3


def test_sleeps_until_next_candle_close(store):
    clock = Clock(START + H + timedelta(minutes=1))
    trader = new_trader(store, clock)
    assert trader.seconds_until_next_check(30) == 30  # nothing processed yet
    trader.run_cycle()
    # Last processed bar opened at START; the next one closes at START + 2h (+5 s grace).
    assert trader.seconds_until_next_check(30) == pytest.approx((H - timedelta(minutes=1)).total_seconds() + 5)
    clock.now = START + 3 * H  # overdue: poll instead
    assert trader.seconds_until_next_check(30) == 30


def test_resume_validation(store):
    clock = Clock(START + H + timedelta(minutes=1))
    trader = new_trader(store, clock)
    with pytest.raises(ValueError, match="unknown run"):
        LivePaperTrader.resume(store, "pp-missing", provider(clock))
    other = CFG.with_overrides({"risk": {"stop_loss_pct": 0.02}})
    with pytest.raises(ValueError, match="config differs"):
        LivePaperTrader(other, provider(clock), store, run_id=trader.run_id, clock=clock)
    bt = BacktestEngine(CFG, provider(clock), store=store).run(START, START + 5 * H)
    with pytest.raises(ValueError, match="not a paper run"):
        LivePaperTrader.resume(store, bt.run_id, provider(clock))
