"""Stage 27A: forward risk (drawdowns, returns, losing streaks) from a run's trades."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.presets import PRESETS
from trading_lab.research.outlook import (
    format_outlook,
    longest_losing_streak,
    max_drawdown,
    outlook_for_run,
    simulate,
    trade_returns,
)
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    db = tmp_path_factory.mktemp("outlook") / "h.db"
    with SQLiteStore(db) as store:
        result = BacktestEngine(PRESETS["trend"].config(BASE), SyntheticProvider(seed=2), store=store).run(
            START, START + timedelta(days=60))
    return db, result


def test_drawdown_and_streaks_by_hand():
    assert max_drawdown(np.array([0.1, -0.2, 0.05])) == pytest.approx(0.2)  # 1.1 -> 0.88
    assert max_drawdown(np.array([-0.1, 0.05])) == pytest.approx(0.1)  # below the start counts
    assert max_drawdown(np.array([0.1, 0.1])) == 0.0
    rows = np.array([[1, -1, -1, 1, -1], [-1, -1, -1, 0, 0]], dtype=float)
    assert list(longest_losing_streak(rows)) == [2, 3]  # a flat trade breaks a streak


def test_simulate():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.002, 0.01, 60)
    a = simulate("r", returns, samples=500)
    assert a == simulate("r", returns, samples=500)  # seeded
    assert a.horizon == 60 and a.trades == 60 and not a.warnings
    assert 0 <= a.drawdown_median <= a.drawdown_bad
    probs = [p for _, p in a.prob_drawdown]
    assert probs == sorted(probs, reverse=True)  # deeper drawdowns are rarer
    assert a.return_low <= a.return_median <= a.return_high
    assert a.actual_drawdown == pytest.approx(float(max_drawdown(returns)))
    longer = simulate("r", returns, horizon=240, samples=500)
    assert longer.drawdown_median > a.drawdown_median  # more trades, deeper drawdowns
    winners = simulate("w", [0.01, 0.02, 0.005], samples=200)
    assert winners.drawdown_bad == 0 and winners.prob_loss == 0 and winners.streak_bad == 0
    losers = simulate("l", [-0.02, -0.03], horizon=10, samples=200)
    assert losers.prob_loss == 1 and losers.streak_median == losers.streak_bad == 10
    assert losers.prob_drawdown[0] == (0.10, 1.0)  # ten losses of 2-3% compound to at least 18%
    assert losers.prob_drawdown[2] == (0.30, 0.0)  # and at most 26%
    assert any("very few" in w for w in winners.warnings)
    assert any("far beyond" in w for w in simulate("r", returns[:10], horizon=100, samples=200).warnings)
    assert simulate("e", [], samples=200) is None
    with pytest.raises(ValueError):
        simulate("r", returns, samples=10)
    with pytest.raises(ValueError):
        simulate("r", returns, horizon=0, samples=200)


def test_trade_returns_from_a_run(run):
    db, result = run
    with SQLiteStore(db, readonly=True) as store:
        returns = trade_returns(store, result.run_id)
        outlook = outlook_for_run(store, result.run_id, samples=1000)
    trades = sorted(result.trades, key=lambda t: t.closed_at)
    assert len(returns) == len(trades) > 20
    curve = result.equity_curve["equity"]
    for r, t in list(zip(returns, trades))[:10]:
        before = curve[curve.index < pd.Timestamp(t.closed_at)]
        equity = float(before.iloc[-1]) if len(before) else BASE.portfolio.initial_cash
        assert r == pytest.approx(t.pnl / equity)
    assert outlook.trades == len(trades) and outlook.samples == 1000
    text = format_outlook(outlook)
    assert f"the next {len(trades)} trades" in text and "bad case (1 in 20)" in text and "losing streak" in text


def test_cli(run, tmp_path, capsys):
    db, result = run
    assert main(["--db", str(db), "outlook", result.run_id, "--trades", "50", "--samples", "300"]) == 0
    assert "the next 50 trades, resampled 300 times" in capsys.readouterr().out
    assert main(["--db", str(db), "outlook", "--json", "--samples", "300"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == result.run_id and len(data["prob_drawdown"]) == 3
    assert main(["--db", str(db), "outlook", "nope"]) == 1
    assert main(["--db", str(db), "outlook", "--samples", "5"]) == 1
    with SQLiteStore(tmp_path / "q.db") as store:
        quiet = BacktestEngine(BASE.with_overrides({"voting": {"min_agreeing": 3}}), SyntheticProvider(seed=2),
                               store=store).run(START, START + timedelta(days=1))
    assert main(["--db", str(tmp_path / "q.db"), "outlook"]) == 0
    assert f"Run {quiet.run_id}: no closed trades, so no outlook." in capsys.readouterr().out
    assert main(["--db", str(tmp_path / "q.db"), "outlook", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) is None


def test_dashboard_data_and_snapshot(run, capsys):
    from trading_lab.dashboard import DashboardData

    db, result = run
    with DashboardData(db) as data:
        o = data.outlook(result.run_id)
        snap = data.snapshot(result.run_id)
        expected = outlook_for_run(data.store, result.run_id, samples=2000)
    assert o == pytest.approx(expected.to_dict()) and snap["outlook"] == o
    json.dumps(snap)
    assert main(["--db", str(db), "dashboard-data", result.run_id]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("Outlook (next "))
    assert f"bad case {expected.drawdown_bad:.1%}" in line


def test_html_report(run, tmp_path):
    db, result = run
    page = tmp_path / "r.html"
    assert main(["--db", str(db), "report", result.run_id, "--html", str(page)]) == 0
    html = page.read_text()
    assert "<h2>What to be ready for</h2>" in html and "Bad case (1 in 20)" in html
    assert "Chance of a drawdown of at least 10%" in html


def test_no_outlook_without_trades(tmp_path):
    from trading_lab.dashboard import DashboardData

    with SQLiteStore(tmp_path / "q.db") as store:
        quiet = BacktestEngine(BASE.with_overrides({"voting": {"min_agreeing": 3}}), SyntheticProvider(seed=2),
                               store=store).run(START, START + timedelta(days=1))
    with DashboardData(tmp_path / "q.db") as data:
        assert data.outlook(quiet.run_id) is None
    page = tmp_path / "q.html"
    assert main(["--db", str(tmp_path / "q.db"), "report", quiet.run_id, "--html", str(page)]) == 0
    assert "What to be ready for" not in page.read_text()


def test_dashboard(run, monkeypatch):
    pytest.importorskip("streamlit")
    from test_dashboard_app import render

    db, result = run
    at = render(monkeypatch, db)
    labels = {m.label: m.value for m in at.metric}
    assert {"Drawdown, bad case", "Chance of a 20% drawdown", "Chance of a loss", "Losing streak, bad case"} <= set(labels)
    assert labels["Losing streak, bad case"].endswith(" trades")
    assert any(m.value.startswith("What to be ready for over the next") for m in at.markdown)
