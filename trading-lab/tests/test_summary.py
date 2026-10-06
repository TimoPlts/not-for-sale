"""Stage 11B: run summaries over a time window (``trading-lab summary``)."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from test_specialists import RoleTransport, agents_config, provider
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.models import Decision, DecisionAction, PortfolioSnapshot
from trading_lab.data import SyntheticProvider
from trading_lab.storage import SQLiteStore
from trading_lab.strategy_factory import strategies_for
from trading_lab.summary import build_summary, format_summary

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 5, tzinfo=UTC)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("summary")
    cfg = agents_config(tmp)
    with SQLiteStore(tmp / "h.db") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                                strategies=strategies_for(cfg, llm_provider=provider(RoleTransport()))).run(START, END)
    return tmp / "h.db", result


def test_whole_run_summary_matches_the_result(run):
    db, result = run
    with SQLiteStore(db, readonly=True) as store:
        s = build_summary(store, result.run_id, hours=24 * 30)
    assert s.equity == pytest.approx(result.equity_curve["equity"].iloc[-1])
    assert s.equity_change == pytest.approx(s.equity - 10_000) and s.total_return == pytest.approx(result.metrics.total_return)
    assert len(s.trades) == len(result.trades) and s.realized_pnl == pytest.approx(sum(t.pnl for t in result.trades))
    non_hold = {}
    for d in result.decisions:
        if d.action is not DecisionAction.HOLD:
            non_hold[d.action.value] = non_hold.get(d.action.value, 0) + 1
    assert s.actions == dict(sorted(non_hold.items()))
    assert set(s.agents) == {"qwen_trend", "qwen_momentum", "qwen_risk"}
    trend_votes = [x for x in result.signals if x.strategy == "qwen_trend" and "rationale" in x.metadata]
    a = s.agents["qwen_trend"]
    assert a["buy"] + a["sell"] + a["hold"] == len(trend_votes) and a["latest"]["label"] == "bullish_trend"
    assert s.usage["calls"] > 0 and s.window_end == result.equity_curve.index[-1].to_pydatetime()


def test_window_only_counts_recent_activity(run):
    db, result = run
    with SQLiteStore(db, readonly=True) as store:
        s = build_summary(store, result.run_id, hours=12)
    end = result.equity_curve.index[-1].to_pydatetime()
    start = end - timedelta(hours=12)
    assert s.window_start == start
    assert len(s.trades) == sum(t.closed_at >= start for t in result.trades)
    before = result.equity_curve[result.equity_curve.index < start]["equity"].iloc[-1]
    assert s.equity_change == pytest.approx(s.equity - before)
    votes = [x for x in result.signals if x.strategy == "qwen_risk" and "rationale" in x.metadata
             and x.timestamp >= start]
    agent = s.agents["qwen_risk"]
    assert agent["buy"] + agent["sell"] + agent["hold"] == len(votes)


def test_markdown(run):
    db, result = run
    with SQLiteStore(db, readonly=True) as store:
        text = format_summary(build_summary(store, result.run_id, hours=24))
    assert text.startswith(f"**trading-lab backtest run {result.run_id}**")
    for part in ("Equity **", "Drawdown now", "market (equal-weight symbols)", "Closed trades", "qwen_trend: BUY",
                 "Model:"):
        assert part in text, part


def test_breakers_health_and_empty_runs(tmp_path):
    cfg = AppConfig()
    t0 = datetime(2024, 3, 1, tzinfo=UTC)
    with SQLiteStore(tmp_path / "p.db") as store:
        store.create_run("pp-1", kind="paper", timeframe="1h", symbols=cfg.market.symbols, exchange="x",
                         config=cfg.to_dict(), config_fingerprint=cfg.fingerprint(), period_start=t0)
        assert format_summary(build_summary(store, "pp-1")).endswith("No equity recorded yet.")
        store.add_snapshots("pp-1", [PortfolioSnapshot(t0, 7000, 0, 7000, -3000, 0, 10, 0)])
        store.add_decisions("pp-1", [Decision(t0, "PORTFOLIO", DecisionAction.CIRCUIT_BREAKER, "max drawdown 30%")])
        store.save_state("pp-1", {"health": {"consecutive_errors": 3, "last_error": "market data unavailable"}})
        text = format_summary(build_summary(store, "pp-1"))
    assert "**Circuit breaker:** max drawdown 30%" in text
    assert "**Health:** 3 failed cycle(s) in a row: market data unavailable" in text
    assert "Closed trades: none" in text


def test_summary_command_is_read_only(run, capsys):
    db, result = run
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert main(["--db", str(db), "summary", "--hours", "6"]) == 0
    assert result.run_id in capsys.readouterr().out
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
