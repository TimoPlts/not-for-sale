"""Backtest engine tests: timing, stops, costs, no look-ahead, determinism, persistence."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.errors import DataError
from trading_lab.core.models import DecisionAction, Direction, Side
from trading_lab.data import MarketDataProvider, SyntheticProvider, normalize_ohlcv, slice_ohlcv
from trading_lab.storage import SQLiteStore
from trading_lab.strategies import Strategy, build_strategies

UTC = timezone.utc
T0 = datetime(2024, 3, 1, tzinfo=UTC)
H = timedelta(hours=1)


class FrameProvider(MarketDataProvider):
    def __init__(self, frames):
        self.frames = frames

    @property
    def name(self):
        return "frames"

    def fetch_ohlcv(self, symbol, timeframe, since, until=None):
        return slice_ohlcv(self.frames[symbol], since, until)


class Scripted(Strategy):
    """Emits pre-scripted signals: {(symbol, bar_index): (direction, confidence)}."""

    name = "scripted"
    warmup_bars = 1
    params: dict = {}

    def __init__(self, script):
        self.script = script

    def _evaluate(self, candles):
        raise NotImplementedError

    def generate_signals(self, symbol, candles):
        from trading_lab.core.models import Signal

        out = []
        for i, ts in enumerate(candles.index):
            bar = int((ts.to_pydatetime() - T0) / H)
            direction, conf = self.script.get((symbol, bar), (Direction.HOLD, 0.0))
            out.append(Signal(self.name, symbol, direction, conf, ts.to_pydatetime()))
        return out


def bars(ohlc, start=T0):
    """Frame from [(open, high, low, close), ...] hourly bars."""
    index = pd.date_range(start, periods=len(ohlc), freq="1h", name="timestamp")
    frame = pd.DataFrame(ohlc, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 1.0
    return normalize_ohlcv(frame)


def config(symbols=("BTC/USDT",), fee=0.0, slip=0.0, **overrides):
    data = {
        "market": {"symbols": list(symbols)},
        "execution": {"fee_rate": fee, "slippage_bps": slip, "min_notional": 1.0},
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return AppConfig.from_mapping(data)


def run_scripted(frames, script, cfg=None, **kwargs):
    cfg = cfg or config(symbols=tuple(frames))
    engine = BacktestEngine(cfg, FrameProvider(frames), strategies=[Scripted(script)], **kwargs)
    last = max(f.index[-1] for f in frames.values()).to_pydatetime()
    return engine.run(T0, last + H)


BUY, SELL = (Direction.BUY, 1.0), (Direction.SELL, 1.0)


# ------------------------------------------------------------------- timing
def test_orders_fill_at_next_bar_open():
    frame = bars([
        (100, 101, 99, 100),   # 0: BUY signal at close
        (102, 103, 101, 102),  # 1: entry fills at open 102
        (104, 106, 103, 105),  # 2: SELL signal at close
        (106, 107, 105, 106),  # 3: exit fills at open 106
        (90, 91, 89, 90),      # 4: price collapses after exit; no effect
    ])
    r = run_scripted({"BTC/USDT": frame}, {("BTC/USDT", 0): BUY, ("BTC/USDT", 2): SELL})
    buy, sell = r.fills
    assert buy.side is Side.BUY and buy.timestamp == T0 + 1 * H and buy.fill_price == 102
    assert sell.side is Side.SELL and sell.timestamp == T0 + 3 * H and sell.fill_price == 106
    (trade,) = r.trades
    assert trade.pnl == pytest.approx(buy.quantity * 4)
    # Risk sizing at the fill price: 1% of 10k / (5% of 102).
    assert buy.quantity == pytest.approx(100 / (102 * 0.05))
    assert r.equity_curve["equity"].iloc[-1] == pytest.approx(10_000 + trade.pnl)
    actions = [d.action for d in r.decisions if d.action is not DecisionAction.HOLD]
    assert actions == [
        DecisionAction.ENTER_SIGNAL, DecisionAction.ENTER,
        DecisionAction.EXIT_SIGNAL, DecisionAction.EXIT,
    ]


def test_costs_are_applied_to_fills():
    frame = bars([(100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100), (100, 101, 99, 100)])
    cfg = config(fee=0.001, slip=10.0)
    r = run_scripted({"BTC/USDT": frame}, {("BTC/USDT", 0): BUY, ("BTC/USDT", 1): SELL}, cfg)
    buy, sell = r.fills
    assert buy.fill_price == pytest.approx(100.1) and sell.fill_price == pytest.approx(99.9)
    assert r.metrics.total_fees == pytest.approx(buy.fee + sell.fee)
    assert r.trades[0].pnl == pytest.approx(-(buy.fee + sell.fee + 0.2 * buy.quantity))


def test_stop_loss_fills_at_stop_price():
    frame = bars([
        (100, 101, 99, 100),
        (100, 101, 99, 100),   # entry at 100, stop 95
        (98, 99, 94, 96),      # low 94 <= 95 -> exit at 95
        (96, 97, 95, 96),
    ])
    r = run_scripted({"BTC/USDT": frame}, {("BTC/USDT", 0): BUY})
    stop_exit = r.fills[1]
    assert stop_exit.fill_price == pytest.approx(95.0) and stop_exit.timestamp == T0 + 2 * H
    assert any(d.action is DecisionAction.STOP_LOSS for d in r.decisions)
    assert r.trades[0].pnl == pytest.approx(-100.0)  # exactly the 1% risk budget


def test_stop_loss_gap_fills_at_open():
    frame = bars([(100, 101, 99, 100), (100, 101, 99, 100), (90, 91, 89, 90), (90, 91, 89, 90)])
    r = run_scripted({"BTC/USDT": frame}, {("BTC/USDT", 0): BUY})
    assert r.fills[1].fill_price == pytest.approx(90.0)  # gapped through the stop
    assert r.trades[0].pnl < -100.0  # gaps can exceed the planned risk


def test_stop_can_trigger_on_entry_bar():
    frame = bars([(100, 101, 99, 100), (100, 101, 94, 96), (96, 97, 95, 96)])
    r = run_scripted({"BTC/USDT": frame}, {("BTC/USDT", 0): BUY})
    entry, stop = r.fills
    assert entry.timestamp == stop.timestamp == T0 + H
    assert stop.fill_price == pytest.approx(95.0)


def test_ignored_and_expired_signals():
    frame = bars([(100, 101, 99, 100)] * 5)
    script = {
        ("BTC/USDT", 0): SELL,  # no position -> ignored
        ("BTC/USDT", 1): BUY,   # entry at bar 2
        ("BTC/USDT", 2): BUY,   # already in position -> ignored
        ("BTC/USDT", 4): SELL,  # last bar -> exit can never fill
    }
    r = run_scripted({"BTC/USDT": frame}, script)
    by_action = r.actions()
    assert by_action["ignored"] == 2
    assert by_action["expired"] == 1
    assert len(r.trades) == 0 and r.equity_curve["open_positions"].iloc[-1] == 1


def test_liquidate_at_end():
    frame = bars([(100, 101, 99, 100), (100, 101, 99, 100), (110, 111, 109, 110)])
    cfg = config(backtest={"liquidate_at_end": True})
    r = run_scripted({"BTC/USDT": frame}, {("BTC/USDT", 0): BUY}, cfg)
    assert r.fills[-1].fill_price == 110 and r.actions()["liquidate"] == 1
    assert r.equity_curve["open_positions"].iloc[-1] == 0
    assert r.metrics.num_trades == 1 and r.metrics.final_equity == pytest.approx(
        10_000 + r.trades[0].pnl
    )


def test_entries_processed_by_confidence_when_cash_is_scarce():
    flat = [(100, 101, 99, 100)] * 3
    frames = {"BTC/USDT": bars(flat), "ETH/USDT": bars(flat)}
    cfg = config(
        symbols=("BTC/USDT", "ETH/USDT"),
        risk={"max_position_pct": 1.0, "risk_per_trade_pct": 1.0, "stop_loss_pct": 0.5},
    )
    script = {("BTC/USDT", 0): (Direction.BUY, 0.6), ("ETH/USDT", 0): (Direction.BUY, 0.9)}
    r = run_scripted(frames, script, cfg)
    (fill,) = r.fills
    assert fill.symbol == "ETH/USDT"  # higher confidence got the cash
    rejected = [d for d in r.decisions if d.action is DecisionAction.REJECTED]
    assert [d.symbol for d in rejected] == ["BTC/USDT"]


def test_bars_before_start_are_warmup_only():
    frame = bars([(100, 101, 99, 100)] * 6, start=T0 - 3 * H)
    script = {("BTC/USDT", -2): BUY}  # a signal before the start must not trade
    cfg = config()
    engine = BacktestEngine(cfg, FrameProvider({"BTC/USDT": frame}), strategies=[Scripted(script)])
    r = engine.run(T0, T0 + 3 * H)
    assert not r.fills
    assert r.equity_curve.index[0] == pd.Timestamp(T0) and len(r.equity_curve) == 3


def test_missing_data_raises():
    engine = BacktestEngine(config(), FrameProvider({"BTC/USDT": bars([(1, 1, 1, 1)], start=T0 - 10 * H)}))
    with pytest.raises(DataError, match="no 1h candles"):
        engine.run(T0, T0 + 5 * H)


# ------------------------------------------------- synthetic, full strategy set
START, END = datetime(2023, 6, 1, tzinfo=UTC), datetime(2023, 8, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def synthetic_result():
    return BacktestEngine(AppConfig(), SyntheticProvider(seed=11)).run(START, END)


def test_invariants_on_multi_symbol_run(synthetic_result):
    r = synthetic_result
    curve = r.equity_curve
    assert curve.index[0] >= pd.Timestamp(START) and curve.index[-1] < pd.Timestamp(END)
    assert len(curve) == (END - START) / H
    np.testing.assert_allclose(curve["equity"], curve["cash"] + curve["positions_value"])
    assert (curve["cash"] >= -1e-6).all()
    assert curve["open_positions"].max() <= AppConfig().risk.max_open_positions
    assert all(START <= f.timestamp < END for f in r.fills)
    assert r.metrics.num_trades == len(r.trades) > 0
    assert r.metrics.final_equity == pytest.approx(curve["equity"].iloc[-1])
    assert r.metrics.total_fees == pytest.approx(sum(f.fee for f in r.fills))
    # Each ensemble decision references a bar; 4 symbols x bars x (3 strategies + ensemble).
    assert len(r.signals) == 4 * len(curve) * 4
    # Ordinary stop-outs never lose more than the risk budget plus a gap.
    stops = {d.order_id for d in r.decisions if d.action is DecisionAction.STOP_LOSS}
    for t in r.trades:
        if t.exit_order_id in stops:
            assert t.pnl > -100.0 * 1.5


def test_backtest_is_deterministic(synthetic_result):
    again = BacktestEngine(AppConfig(), SyntheticProvider(seed=11)).run(START, END)
    assert again.fills == synthetic_result.fills
    assert again.decisions == synthetic_result.decisions
    assert again.metrics == synthetic_result.metrics
    pd.testing.assert_frame_equal(again.equity_curve, synthetic_result.equity_curve)


def test_no_look_ahead(synthetic_result):
    """Changing candles after a cutoff must not change anything up to the cutoff."""
    cutoff = pd.Timestamp(START + timedelta(days=30))
    base = SyntheticProvider(seed=11)

    class PerturbedFuture(MarketDataProvider):
        name = "perturbed"

        def fetch_ohlcv(self, symbol, timeframe, since, until=None):
            frame = base.fetch_ohlcv(symbol, timeframe, since, until).copy()
            after = frame.index > cutoff
            rng = np.random.default_rng(99)
            factors = rng.uniform(0.7, 1.3, after.sum())
            frame.loc[after, ["open", "high", "low", "close"]] *= factors[:, None]
            return frame

    altered = BacktestEngine(AppConfig(), PerturbedFuture()).run(START, END)
    cut = cutoff.to_pydatetime()
    assert [f for f in altered.fills if f.timestamp <= cut] == [
        f for f in synthetic_result.fills if f.timestamp <= cut
    ]
    assert [d for d in altered.decisions if d.timestamp <= cut] == [
        d for d in synthetic_result.decisions if d.timestamp <= cut
    ]
    pd.testing.assert_frame_equal(
        altered.equity_curve.loc[:cutoff], synthetic_result.equity_curve.loc[:cutoff]
    )
    # ...while the future genuinely differs.
    assert altered.metrics.final_equity != synthetic_result.metrics.final_equity


def test_vectorised_signals_match_bar_by_bar():
    candles = SyntheticProvider(seed=4).fetch_ohlcv(
        "SOL/USDT", "1h", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 12, tzinfo=UTC)
    )
    for strategy in build_strategies(AppConfig().enabled_strategies):
        fast = strategy.generate_signals("SOL/USDT", candles)
        slow = [strategy.generate_signal("SOL/USDT", candles.iloc[: i + 1]) for i in range(len(candles))]
        assert fast == slow, strategy.name


# --------------------------------------------------------------- persistence
def test_results_are_persisted(tmp_path):
    with SQLiteStore(tmp_path / "bt.db") as store:
        cfg = AppConfig()
        r = BacktestEngine(cfg, SyntheticProvider(seed=2), store=store).run(
            START, START + timedelta(days=10), notes="persist test"
        )
        run = store.get_run(r.run_id)
        assert run["status"] == "completed" and run["notes"] == "persist test"
        assert run["config_fingerprint"] == cfg.fingerprint()
        assert store.count("signals", r.run_id) == len(r.signals)
        assert store.count("decisions", r.run_id) == len(r.decisions)
        assert store.count("fills", r.run_id) == len(r.fills)
        assert store.load_closed_trades(r.run_id) == list(r.trades)
        curve = store.load_equity_curve(r.run_id)
        np.testing.assert_allclose(curve["equity"], r.equity_curve["equity"])
        assert store.load_metrics(r.run_id) == r.stored_metrics()


def test_failed_run_is_marked(tmp_path):
    class Broken(MarketDataProvider):
        name = "broken"

        def fetch_ohlcv(self, *args, **kwargs):
            raise DataError("exchange unreachable")

    with SQLiteStore(tmp_path / "bt.db") as store:
        with pytest.raises(DataError):
            BacktestEngine(AppConfig(), Broken(), store=store).run(START, END, run_id="bt-broken")
        run = store.get_run("bt-broken")
        assert run["status"] == "failed" and "exchange unreachable" in run["error"]


def test_buy_and_hold_benchmark():
    frame = bars([(100, 101, 99, 100), (110, 111, 109, 110), (120, 121, 119, 120)])
    r = run_scripted({"BTC/USDT": frame}, {})  # the strategy never trades
    assert r.benchmark is not None
    # Zero costs: bought at the first open (100), worth 120 at the end.
    assert list(r.benchmark_curve) == pytest.approx([10_000, 11_000, 12_000])
    assert r.benchmark.total_return == pytest.approx(0.2)
    assert r.metrics.total_return == 0.0
    costly = run_scripted({"BTC/USDT": frame}, {}, config(fee=0.001, slip=10.0))
    assert costly.benchmark.total_return < 0.2  # pays the same costs as the strategy


def test_benchmark_splits_cash_equally():
    a = bars([(100, 101, 99, 100), (200, 201, 199, 200)])
    b = bars([(10, 11, 9, 10), (10, 11, 9, 10)])
    r = run_scripted({"BTC/USDT": a, "ETH/USDT": b}, {}, config(symbols=("BTC/USDT", "ETH/USDT")))
    assert r.benchmark_curve.iloc[-1] == pytest.approx(5_000 * 2 + 5_000)
