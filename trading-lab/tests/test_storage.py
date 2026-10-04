"""SQLite store round-trip tests."""

import sqlite3

import pytest

from conftest import ts
from trading_lab.config import AppConfig
from trading_lab.core.models import (
    ClosedTrade,
    Decision,
    DecisionAction,
    Direction,
    ExecutionReport,
    Fill,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Side,
    Signal,
)
from trading_lab.storage import SCHEMA_VERSION, SQLiteStore


@pytest.fixture
def store(tmp_path):
    with SQLiteStore(tmp_path / "history.db") as s:
        yield s


def make_run(store, run_id="r1"):
    cfg = AppConfig()
    store.create_run(
        run_id, kind="backtest", timeframe="1h", symbols=["BTC/USDT"], exchange="synthetic",
        config=cfg.to_dict(), config_fingerprint=cfg.fingerprint(), period_start=ts(0),
        period_end=ts(10), notes="unit test",
    )


def test_schema_is_versioned_and_reopen_is_idempotent(tmp_path):
    path = tmp_path / "h.db"
    SQLiteStore(path).close()
    with SQLiteStore(path) as s:
        assert s.schema_version == SCHEMA_VERSION


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "h.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(RuntimeError, match="newer"):
        SQLiteStore(path)


def test_run_lifecycle(store):
    make_run(store)
    run = store.get_run("r1")
    assert run["status"] == "running"
    assert run["symbols"] == ["BTC/USDT"]
    assert run["config"]["portfolio"]["initial_cash"] == 10_000.0
    assert run["config_fingerprint"] == AppConfig().fingerprint()
    store.finish_run("r1", "completed")
    assert store.get_run("r1")["status"] == "completed"
    assert store.get_run("nope") is None
    store.save_metrics("r1", {"total_return": 0.1, "profit_factor": float("inf")})
    assert store.load_metrics("r1") == {"total_return": 0.1, "profit_factor": "inf"}
    (listed,) = store.list_runs()
    assert listed["run_id"] == "r1" and listed["metrics"]["total_return"] == 0.1


def test_records_round_trip(store):
    make_run(store)
    sig = Signal("rsi", "BTC/USDT", Direction.BUY, 0.7, ts(1), {"rsi": 22.5, "params": {"period": 14}})
    store.add_signals("r1", [sig])
    store.add_decisions("r1", [
        Decision(ts(2), "BTC/USDT", DecisionAction.ENTER, "ok", Direction.BUY, 0.7, quantity=0.1,
                 reference_price=100.0, stop_price=95.0, order_id="bt-000001",
                 details={"binding_limit": "risk_per_trade", "equity": float("nan")}),
        Decision(ts(3), "BTC/USDT", DecisionAction.HOLD, "ensemble HOLD"),
    ])
    order = Order("BTC/USDT", Side.BUY, 0.1, ts(2), stop_price=95.0, reason="enter")
    fill = Fill("bt-000001", "BTC/USDT", Side.BUY, 0.1, 100.0, 100.05, 0.01, ts(2), stop_price=95.0)
    rejected = Order("BTC/USDT", Side.SELL, 1.0, ts(3), reason="exit")
    store.add_execution_reports("r1", [
        ExecutionReport("bt-000001", order, OrderStatus.FILLED, fill=fill),
        ExecutionReport("bt-000002", rejected, OrderStatus.REJECTED, reason="long-only"),
    ])
    snap = PortfolioSnapshot(ts(2), 9_990.0, 10.0, 10_000.0, 0.0, -0.01, 0.01, 1)
    store.add_snapshots("r1", [snap])
    trade = ClosedTrade("BTC/USDT", 0.1, 100.1, 110.0, 10.01, 10.99, 0.98, ts(2), ts(5), "bt-000003")
    store.add_closed_trades("r1", [trade])

    signals = store.load_signals("r1")
    assert signals.loc[0, "direction"] == "buy" and signals.loc[0, "confidence"] == 0.7
    decisions = store.load_decisions("r1")
    assert list(decisions["action"]) == ["enter", "hold"]
    assert len(store.load_decisions("r1", include_holds=False)) == 1
    fills = store.load_fills("r1")
    assert len(fills) == 1 and fills.loc[0, "reason"] == "enter"
    assert fills.loc[0, "fill_price"] == 100.05
    assert store.count("orders", "r1") == 2
    curve = store.load_equity_curve("r1")
    assert curve["equity"].iloc[0] == 10_000.0 and curve.index[0] == ts(2)
    assert store.load_closed_trades("r1") == [trade]


def test_in_memory_store():
    with SQLiteStore(":memory:") as s:
        make_run(s)
        assert s.get_run("r1") is not None


def test_count_rejects_unknown_table(store):
    with pytest.raises(ValueError):
        store.count("runs; DROP TABLE runs", "r1")
