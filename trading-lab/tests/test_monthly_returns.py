"""Stage 21A: monthly returns, Calmar ratio and the longest drawdown."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.dashboard import DashboardData
from trading_lab.data import SyntheticProvider
from trading_lab.html_report import build_html_report
from trading_lab.metrics import compute_metrics, monthly_returns
from trading_lab.storage import SQLiteStore

UTC = timezone.utc


def test_calmar_and_longest_drawdown():
    m = compute_metrics([100, 110, 99, 105, 108, 112, 90, 95], [], "1d")
    assert m.max_drawdown == pytest.approx(1 - 90 / 112)
    assert m.max_drawdown_bars == 3  # 99, 105, 108 under the 110 peak (90, 95 make 2)
    assert m.calmar_ratio == pytest.approx(m.annualized_return / m.max_drawdown)
    rising = compute_metrics([100, 101, 102], [], "1d")
    assert rising.calmar_ratio is None and rising.max_drawdown_bars == 0
    assert "Calmar ratio" in m.format_table() and "Longest drawdown    3 bars" in m.format_table()


def test_monthly_table():
    index = pd.DatetimeIndex([datetime(2024, 11, 10, tzinfo=UTC), datetime(2024, 11, 30, tzinfo=UTC),
                              datetime(2025, 1, 5, tzinfo=UTC), datetime(2025, 1, 31, tzinfo=UTC),
                              datetime(2025, 3, 1, tzinfo=UTC)])
    equity = pd.Series([1000, 1100, 1050, 1155, 1039.5], index=index)
    table = monthly_returns(equity, 1000)
    assert list(table.index) == [2024, 2025]
    assert table.loc[2024, 11] == pytest.approx(0.10)  # vs the initial 1000
    assert np.isnan(table.loc[2024, 12]) and np.isnan(table.loc[2025, 2])  # no bars
    assert table.loc[2025, 1] == pytest.approx(0.05)  # 1155 vs November's 1100
    assert table.loc[2025, 3] == pytest.approx(-0.10)
    assert table.loc[2025, "year"] == pytest.approx(1.05 * 0.9 - 1)
    assert monthly_returns(pd.Series(dtype="float64"), 1000).empty


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    db = tmp_path_factory.mktemp("monthly") / "m.db"
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    with SQLiteStore(db) as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=4), store=store).run(
            datetime(2024, 1, 20, tzinfo=UTC), datetime(2024, 3, 10, tzinfo=UTC))
    return db, result


def test_views_show_the_table(run, capsys):
    db, result = run
    with DashboardData(db) as data:
        rows = data.monthly_returns(result.run_id)
        snap = data.snapshot(result.run_id)
    assert [r["year"] for r in rows] == [2024] and set(rows[0]) == {"year", "total", *map(str, range(1, 13))}
    assert rows[0]["4"] is None and rows[0]["1"] is not None
    assert np.prod([1 + rows[0][str(m)] for m in (1, 2, 3)]) - 1 == pytest.approx(result.metrics.total_return)
    assert snap["monthly_returns"] == rows
    page = build_html_report(str(db), result.run_id)
    assert "<h2>Monthly returns</h2>" in page and "Year total" in page
    assert main(["--db", str(db), "report", result.run_id]) == 0
    out = capsys.readouterr().out
    assert "=== Monthly returns ===" in out and "Calmar ratio" in out


def test_the_dashboard_shows_it(run, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from conftest import SRC_DIR

    monkeypatch.setenv("TRADING_LAB_DB", str(run[0]))
    at = AppTest.from_file(str(SRC_DIR / "dashboard" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception and any(m.value == "Monthly returns" for m in at.markdown)
