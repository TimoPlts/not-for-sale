"""Stage 24A: a run's closed trades grouped by exit type, symbol, side, holding time and entry time."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.presets import PRESETS
from trading_lab.research.trades import (
    TradeAnalysis,
    TradeRow,
    analyze_trades,
    exit_reason,
    format_trades,
    group_stats,
    observations,
)
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]}})


def backtest(store, cfg, days=60, seed=2):
    return BacktestEngine(cfg, SyntheticProvider(seed=seed), store=store).run(START, START + timedelta(days=days))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    db = tmp_path_factory.mktemp("trades") / "h.db"
    with SQLiteStore(db) as store:
        trend = backtest(store, PRESETS["trend"].config(BASE))
        shorts = backtest(store, PRESETS["trend-shorts"].config(BASE))
        kill = backtest(store, BASE.with_overrides({"risk": {"max_drawdown_pct": 0.01, "flatten_on_halt": True}}))
    return db, trend, shorts, kill


def test_exit_reason_mapping():
    assert exit_reason("stop_loss", "trailing stop 101 hit (bar low 100)", None) == "trailing stop"
    assert exit_reason("stop_loss", "stop 95 hit (bar low 94)", None) == "stop-loss"
    assert exit_reason("take_profit", "take-profit 110 reached", None) == "take-profit"
    assert exit_reason("liquidate", "end of backtest", None) == "end of backtest"
    assert exit_reason("exit", "time stop: held 72 bars", None) == "time stop"
    assert exit_reason("exit", "signal exit", "kill switch: closing position") == "kill switch"
    assert exit_reason("exit", "signal exit", "exit scheduled for next bar open") == "signal"
    assert exit_reason("exit", "signal cover", None) == "signal"


def test_groups_add_up_and_match_the_run(runs):
    db, trend, _, _ = runs
    with SQLiteStore(db, readonly=True) as store:
        a = analyze_trades(store, trend.run_id, groupings=("exit", "symbol", "side", "holding", "weekday", "hour"))
    assert len(a.rows) == len(trend.trades) == trend.metrics.num_trades
    total = sum(t.pnl for t in trend.trades)
    for grouping, groups in a.groups.items():
        assert sum(g.trades for g in groups) == len(a.rows), grouping
        assert sum(g.pnl for g in groups) == pytest.approx(total), grouping
    exits = {g.name for g in a.groups["exit"]}
    assert {"signal", "time stop", "trailing stop"} <= exits
    assert exits <= {"signal", "time stop", "trailing stop", "stop-loss", "take-profit", "kill switch",
                     "end of backtest"}
    limit = PRESETS["trend"].config(BASE).risk.max_holding_bars
    assert all(r.bars_held == limit for r in a.rows if r.exit_reason == "time stop")
    assert all(r.bars_held < limit for r in a.rows if r.exit_reason != "time stop")
    assert [g.name for g in a.groups["weekday"]] == sorted((g.name for g in a.groups["weekday"]),
                                                            key=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat",
                                                                 "Sun"].index)
    assert a.total.pnl == pytest.approx(total)
    assert a.total.wins == sum(1 for t in trend.trades if t.pnl > 0)


def test_shorts_and_kill_switch(runs):
    db, _, shorts, kill = runs
    with SQLiteStore(db, readonly=True) as store:
        s = analyze_trades(store, shorts.run_id)
        k = analyze_trades(store, kill.run_id)
    assert {g.name for g in s.groups["side"]} == {"long", "short"}
    assert "hour" not in s.groups  # on request only
    kill_trades = [r for r in k.rows if r.exit_reason == "kill switch"]
    trips = [d for d in kill.decisions if d.reason.startswith("kill switch")]
    assert trips and kill_trades and len(kill_trades) <= len(trips)


def test_group_stats_and_observations():
    t0 = datetime(2024, 1, 1, tzinfo=UTC)

    def row(pnl, bars=5, exit_="signal", symbol="BTC/USDT", side="long"):
        return TradeRow(symbol, side, t0, t0 + timedelta(hours=bars), pnl, pnl / 1000, bars, exit_)

    g = group_stats("x", [row(10), row(-5), row(-5), row(20)])
    assert (g.trades, g.wins, g.pnl, g.best, g.worst) == (4, 2, 20, 20, -5)
    assert g.win_rate == 0.5 and g.profit_factor == pytest.approx(3.0)
    assert group_stats("w", [row(1)]).profit_factor == float("inf")
    assert group_stats("z", [row(0.0)]).profit_factor is None

    rows = [row(500, bars=80)] + [row(-10, bars=3, exit_="stop-loss") for _ in range(9)] + [row(-5, bars=80)] * 3
    a = TradeAnalysis("r", "1h", rows)
    for grouping in ("exit", "symbol", "side", "holding"):
        from trading_lab.research.trades import _bucket

        buckets = {}
        for r in rows:
            buckets.setdefault(_bucket(grouping, r), []).append(r)
        a.groups[grouping] = sorted((group_stats(n, rs) for n, rs in buckets.items()), key=lambda s: s.pnl)
    a.groups["holding"].sort(key=lambda s: s.name != "3-6 bars")
    notes = observations(a)
    assert any(n.startswith("without its best 1 trade(s) the run would have lost money") for n in notes)
    assert any(n.startswith("the costliest exit type is stop-loss: 9 trades") for n in notes)
    assert any(n.startswith("trades held over 72 bars make money") for n in notes)
    assert observations(TradeAnalysis("r", "1h", rows[:3])) == ["too few closed trades to say much"]


def test_cli(runs, tmp_path, capsys):
    db, trend, _, _ = runs
    assert main(["--db", str(db), "trades", trend.run_id]) == 0
    out = capsys.readouterr().out
    assert f"Run {trend.run_id}: {len(trend.trades)} closed trades" in out
    assert "By exit" in out and "By holding time" in out and "By entry hour" not in out
    csv = tmp_path / "t.csv"
    assert main(["--db", str(db), "trades", trend.run_id, "--by", "exit,hour", "--json", "--csv", str(csv)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["groups"]) == {"exit", "hour"} and len(data["trades"]) == len(trend.trades)
    frame = pd.read_csv(csv)
    assert len(frame) == len(trend.trades) and frame["pnl"].sum() == pytest.approx(data["total"]["pnl"])
    assert set(frame["exit_reason"]) == {g["name"] for g in data["groups"]["exit"]}
    assert main(["--db", str(db), "trades", trend.run_id, "--by", "colour"]) == 1
    assert main(["--db", str(db), "trades", "nope"]) == 1


def test_no_trades(tmp_path):
    with SQLiteStore(tmp_path / "e.db") as store:
        quiet = BASE.with_overrides({"voting": {"min_agreeing": 3}, "market": {"symbols": ["BTC/USDT"]}})
        result = backtest(store, quiet, days=1)
        a = analyze_trades(store, result.run_id)
    assert a.rows == [] and a.total is None and format_trades(a).endswith("no closed trades.")
    assert a.to_dict()["total"] is None
