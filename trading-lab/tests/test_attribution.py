"""Stage 9C: agent performance attribution."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from test_specialists import RoleTransport, agents_config, provider
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import VotingConfig
from trading_lab.core.models import ClosedTrade, Decision, DecisionAction, Direction, Signal
from trading_lab.data import SyntheticProvider
from trading_lab.research import attribute, attribute_result, attribute_run, format_attribution
from trading_lab.storage import SCHEMA_VERSION, SQLiteStore
from trading_lab.strategy_factory import strategies_for

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)
H = timedelta(hours=1)
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 4, tzinfo=UTC)
AGENT_PARAMS = {"params": {"agent": "a", "lookback": 5}}


def t(i):
    return T0 + i * H


def vote(i, direction, confidence, **meta):
    return Signal("a", "BTC/USDT", Direction(direction), confidence, t(i), {**AGENT_PARAMS, **meta})


def ensemble(i, votes, direction="buy"):
    return Signal("ensemble", "BTC/USDT", Direction(direction), 0.4 if direction == "buy" else 0.0, t(i),
                  {"votes": votes})


def trade(opened, closed, pnl, cost=1000.0):
    return ClosedTrade("BTC/USDT", 1.0, cost, cost + pnl, cost, cost + pnl, pnl, t(opened), t(closed), f"o{closed}")


@pytest.fixture
def scenario():
    closes = {"BTC/USDT": pd.Series([100, 101, 102, 103, 104, 105, 104, 103, 102, 101.0],
                                    index=pd.DatetimeIndex([t(i) for i in range(10)]))}
    signals = [
        vote(0, "buy", 0.8, rationale="up"),
        vote(1, "sell", 0.6, rationale="down"),
        vote(2, "hold", 0.0, rationale="unclear"),
        vote(3, "hold", 0.0, error="ProviderTimeoutError: slow"),
        vote(4, "hold", 0.0, reason="not a decision bar"),
        vote(5, "hold", 0.0, reason="warmup"),
        vote(8, "buy", 0.85, rationale="up again"),
        Signal("rsi", "BTC/USDT", Direction.HOLD, 0.0, t(0), {"params": {"period": 14}}),
        ensemble(0, [{"strategy": "a", "direction": "buy", "confidence": 0.8, "weight": 1.0},
                     {"strategy": "rsi", "direction": "hold", "confidence": 0.0, "weight": 1.0}]),
        ensemble(1, [{"strategy": "a", "direction": "sell", "confidence": 0.6, "weight": 1.0},
                     {"strategy": "rsi", "direction": "buy", "confidence": 1.0, "weight": 3.0}]),
    ]
    decisions = [
        Decision(t(0), "BTC/USDT", DecisionAction.ENTER_SIGNAL, "entry scheduled"),
        Decision(t(1), "BTC/USDT", DecisionAction.ENTER_SIGNAL, "entry scheduled"),
    ]
    trades = [trade(1, 5, 50.0), trade(2, 6, -20.0)]
    return signals, decisions, trades, closes


def test_attribution_by_hand(scenario):
    signals, decisions, trades, closes = scenario
    a = attribute(signals, decisions, trades, closes, voting=VotingConfig(), initial_cash=10_000, horizon=2)["a"]
    assert a.is_agent and a.horizon_bars == 2
    assert (a.votes, a.buy, a.sell, a.hold, a.errors) == (5, 2, 1, 1, 1)  # warm-up / skipped bars are not votes
    assert a.avg_confidence == pytest.approx((0.8 + 0.6 + 0.85) / 3)
    assert (a.measured, a.correct) == (2, 1)  # the t8 BUY has no price 2 bars later
    assert a.directional_correctness == 0.5
    assert a.avg_return_after_buy == pytest.approx(102 / 100 - 1)
    assert a.avg_return_after_sell == pytest.approx(103 / 101 - 1)
    assert (a.trades, a.trades_influenced, a.trades_agreed, a.trades_disagreed) == (2, 2, 1, 1)
    assert a.trades_pivotal == 1  # without its BUY at t0 the ensemble would not have entered
    assert (a.pnl_agreed, a.pnl_disagreed) == (50.0, -20.0)
    assert (a.pnl_agreed_pct, a.pnl_disagreed_pct) == (0.005, -0.002)
    assert a.avg_trade_return_agreed == pytest.approx(0.05)
    buckets = {b.low: b for b in a.calibration}
    assert (buckets[0.8].votes, buckets[0.8].measured, buckets[0.8].correct) == (2, 1, 1)
    assert buckets[0.8].mean_signed_return == pytest.approx(0.02)
    assert (buckets[0.6].votes, buckets[0.6].correct) == (1, 0) and buckets[0.6].hit_rate == 0.0
    assert buckets[0.6].mean_signed_return == pytest.approx(-(103 / 101 - 1))

    rsi = attribute(signals, decisions, trades, closes, voting=VotingConfig(), initial_cash=10_000, horizon=2)["rsi"]
    assert not rsi.is_agent and rsi.votes == 1 and rsi.trades_influenced == 0

    text = format_attribution(a)
    assert "Agent: a" in text and "Votes: 5" in text and "Directional correctness (2-bar horizon): 50.0%" in text
    assert "PnL when agreed: +0.50%" in text and "pivotal 1" in text


def test_horizon_must_be_positive(scenario):
    with pytest.raises(ValueError):
        attribute(*scenario, voting=VotingConfig(), initial_cash=1.0, horizon=0)


def test_attribution_is_deterministic_and_matches_the_stored_run(tmp_path):
    cfg = agents_config(tmp_path)
    results = []
    with SQLiteStore(tmp_path / "h.db") as store:
        for _ in range(2):
            result = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                                    strategies=strategies_for(cfg, llm_provider=provider(RoleTransport()))
                                    ).run(START, END)
            results.append((attribute_result(result, cfg), attribute_run(store, result.run_id)))
        assert store.count("bars", result.run_id) == len(result.bars) == 2 * 72
    (memory, stored), (memory2, stored2) = results
    assert memory == stored == memory2 == stored2
    agents = {name for name, a in memory.items() if a.is_agent}
    assert agents == {"qwen_trend", "qwen_momentum", "qwen_risk"}
    for a in memory.values():
        assert a.buy + a.sell + a.hold + a.errors == a.votes
        assert a.trades_agreed + a.trades_disagreed == a.trades_influenced <= a.trades
        assert a.trades_pivotal <= a.trades_agreed
    trend = memory["qwen_trend"]
    assert trend.votes == 2 * 12 and trend.measured > 0  # 72 bars / decision_interval 6, two symbols


def test_agent_report_command(tmp_path, capsys, monkeypatch):
    db = tmp_path / "h.db"
    cfg = agents_config(tmp_path)
    with SQLiteStore(db) as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                                strategies=strategies_for(cfg, llm_provider=provider(RoleTransport()))
                                ).run(START, END)
    assert main(["--db", str(db), "agent-report", result.run_id, "--horizon", "3"]) == 0
    out = capsys.readouterr().out
    assert "Agent: qwen_trend" in out and "Agent: qwen_risk" in out and "Strategy: rsi" not in out
    assert "3-bar horizon" in out and "Trades influenced" in out
    assert main(["--db", str(db), "agent-report", "--all"]) == 0  # defaults to the latest run
    assert "Strategy: rsi" in capsys.readouterr().out

    baseline_db = tmp_path / "b.db"
    assert main(["--db", str(baseline_db), "backtest", "--synthetic", "1", "--days", "5"]) == 0
    capsys.readouterr()
    assert main(["--db", str(baseline_db), "agent-report"]) == 0
    assert "No AI agents voted" in capsys.readouterr().out


def test_schema_v2_databases_are_upgraded(tmp_path):
    path = tmp_path / "old.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    conn.executescript("DROP TABLE bars; DROP TABLE research_results; PRAGMA user_version = 2;")
    conn.close()
    with SQLiteStore(path) as store:
        assert store.schema_version == SCHEMA_VERSION == 3
        assert store.load_closes("missing") == {}
        store.add_research_result("experiment", "x", {"a": 1})
        assert store.list_research_results()[0]["payload"] == {"a": 1}
