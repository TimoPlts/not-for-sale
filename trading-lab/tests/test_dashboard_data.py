"""Stage 10A: read-only dashboard data layer."""

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from conftest import SRC_DIR
from test_live import ANCHOR, START as LIVE_START, Clock
from test_specialists import RoleTransport, agents_config, provider
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.dashboard import DashboardData
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.research import attribute_run
from trading_lab.storage import SQLiteStore
from trading_lab.strategy_factory import strategies_for

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 5, tzinfo=UTC)


@pytest.fixture(scope="module")
def backtest_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dash")
    cfg = agents_config(tmp)
    with SQLiteStore(tmp / "h.db") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                                strategies=strategies_for(cfg, llm_provider=provider(RoleTransport()))).run(START, END)
    return tmp / "h.db", result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_matches_the_run(backtest_db):
    db, result = backtest_db
    before = digest(db)
    with DashboardData(db) as data:
        snap = data.snapshot()
        assert data.default_run_id() == result.run_id
        assert data.agent_performance(result.run_id) == [
            json.loads(json.dumps(a.to_dict())) for a in attribute_run(data.store, result.run_id).values()]
    json.dumps(snap, allow_nan=False)  # strictly JSON-serialisable
    assert digest(db) == before  # reading changed nothing

    o = snap["overview"]
    assert o["equity"] == pytest.approx(result.equity_curve["equity"].iloc[-1])
    assert o["total_return"] == pytest.approx(result.metrics.total_return)
    assert o["max_drawdown"] == pytest.approx(-result.metrics.max_drawdown)
    assert o["open_positions"] == len(snap["open_positions"])
    assert o["breakers"]["kill_switch_active"] is False
    assert len(snap["recent_trades"]) == min(20, len(result.trades))

    research = snap["research"]
    assert research["metrics"]["total_return"] == pytest.approx(result.metrics.total_return)
    assert research["benchmark"]["total_return"] == pytest.approx(result.benchmark.total_return)

    latest = snap["latest_decision"]["symbols"]["BTC/USDT"]
    strategies = {v["strategy"] for v in latest["votes"]}
    assert strategies == {"rsi", "macd", "bollinger", "qwen_trend", "qwen_momentum", "qwen_risk"}
    assert latest["ensemble"]["direction"] in ("buy", "sell", "hold") and latest["actions"]
    assert all(v["weight"] == 1.0 for v in latest["votes"])

    rationales = snap["latest_rationales"]
    assert {(r["strategy"], r["symbol"]) for r in rationales} == {
        (a, s) for a in ("qwen_trend", "qwen_momentum", "qwen_risk") for s in ("BTC/USDT", "ETH/USDT")}
    assert all(r["rationale"] for r in rationales)
    assert {r.get("regime") for r in rationales if r["strategy"] == "qwen_trend"} == {"bullish_trend"}

    usage = snap["usage"]
    assert set(usage["per_agent"]) == {"qwen_trend", "qwen_momentum", "qwen_risk"}
    assert usage["total"]["calls"] == sum(a["calls"] for a in usage["per_agent"].values()) > 0


def test_equity_curve_has_drawdown_and_benchmark(backtest_db):
    db, result = backtest_db
    with DashboardData(db) as data:
        curve = data.equity_curve(result.run_id)
    assert {"equity", "drawdown", "buy_and_hold"} <= set(curve.columns)
    assert (curve["drawdown"] <= 0).all()
    assert curve["buy_and_hold"].iloc[-1] == pytest.approx(result.benchmark_curve.iloc[-1])


def test_open_positions_match_the_final_portfolio(backtest_db):
    db, result = backtest_db
    with DashboardData(db) as data:
        positions = data.open_positions(result.run_id)
    held = {p["symbol"]: p for p in positions}
    final = result.snapshots[-1]
    assert len(held) == final.open_positions
    assert sum(p["value"] for p in positions) == pytest.approx(final.positions_value)
    assert sum(p["unrealized_pnl"] for p in positions) == pytest.approx(final.unrealized_pnl)


def test_paper_run_state_breakers_and_working_orders(tmp_path):
    clock = Clock(LIVE_START + timedelta(hours=1, minutes=1))
    db = tmp_path / "paper.db"
    with SQLiteStore(db) as store:
        cfg = agents_config(tmp_path, {"qwen_risk": 1.0}).with_overrides(
            {"execution": {"entry_order_type": "limit"}})
        trader = LivePaperTrader(cfg, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock,
                                 llm_provider=provider(RoleTransport()))
        for _ in range(12):
            trader.run_cycle()
            clock.now += timedelta(hours=1)
        BacktestEngine(agents_config(tmp_path, {"qwen_trend": 0.0}), SyntheticProvider(seed=1),
                       store=store).run(START, START + timedelta(days=1))  # newer, but not a running paper run
    with DashboardData(db) as data:
        assert data.default_run_id() == trader.run_id
        snap = data.snapshot()
        breakers = snap["overview"]["breakers"]
        assert breakers["source"] == "state" and breakers["peak_equity"] >= 10_000
        orders = snap["working_orders"]
        assert set(orders) == {"scheduled", "working_limits"}
        assert snap["overview"]["kind"] == "paper" and snap["overview"]["status"] == "running"


def test_the_data_layer_cannot_write(backtest_db, tmp_path):
    db, _ = backtest_db
    with DashboardData(db) as data:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            data.store.save_metrics("x", {})
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            data.store._conn.execute("DELETE FROM runs")
    with pytest.raises(FileNotFoundError):
        DashboardData(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()


def test_old_databases_are_read_without_migrating(tmp_path):
    path = tmp_path / "old.db"
    with SQLiteStore(path) as store:
        BacktestEngine(agents_config(tmp_path, {"qwen_trend": 0.0}), SyntheticProvider(seed=1),
                       store=store).run(START, START + timedelta(days=2))
    conn = sqlite3.connect(path)
    conn.executescript("DROP TABLE bars; DROP TABLE research_results; PRAGMA user_version = 2;")
    conn.close()
    with DashboardData(path) as data:
        snap = data.snapshot()
        assert data.store.schema_version == 2  # not upgraded by the read-only dashboard
    assert snap["overview"]["equity"] > 0 and snap["research"]["research_results"] == []


def test_dashboard_code_has_no_write_or_order_paths():
    forbidden = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE)\b|\.submit\(|create_\w*order|"
                           r"PaperExecutor|LivePaperTrader|TradingSession|add_\w+\(|save_\w+\(|finish_run")
    for path in sorted((SRC_DIR / "dashboard").rglob("*.py")):
        offenders = [line for line in path.read_text(encoding="utf-8").splitlines() if forbidden.search(line)]
        assert not offenders, f"{path.name}: {offenders}"


def test_dashboard_data_command(backtest_db, capsys):
    db, result = backtest_db
    assert main(["--db", str(db), "dashboard-data"]) == 0
    out = capsys.readouterr().out
    assert result.run_id in out and "Equity" in out and "qwen_trend=" in out and "Model usage" in out
    assert main(["--db", str(db), "dashboard-data", result.run_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == result.run_id
