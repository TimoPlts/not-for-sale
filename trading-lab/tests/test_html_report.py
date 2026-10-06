"""Stage 12D: self-contained HTML run report."""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from test_specialists import RoleTransport, agents_config, provider
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.models import PortfolioSnapshot
from trading_lab.data import SyntheticProvider
from trading_lab.html_report import MAX_POINTS, build_html_report, line_chart
from trading_lab.storage import SQLiteStore
from trading_lab.strategy_factory import strategies_for

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 8, tzinfo=UTC)
EVIL = '<script>alert(1)</script><img src=x onerror=alert(2)>'


def nasty(role, user):
    label = {"Trend Agent": ("regime", "sideways"), "Momentum Agent": ("momentum_state", "neutral"),
             "Risk/Regime Agent": ("risk_state", "low")}[role]
    return {"direction": "BUY", "confidence": 0.7, "rationale": EVIL, label[0]: label[1]}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("html")
    cfg = agents_config(tmp)
    with SQLiteStore(tmp / "h.db") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                                strategies=strategies_for(cfg, llm_provider=provider(RoleTransport(override=nasty)))
                                ).run(START, END)
    return tmp / "h.db", result


def test_report_matches_the_run(run):
    db, result = run
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    page = build_html_report(str(db), now=datetime(2024, 3, 1, tzinfo=UTC))
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before  # read-only
    assert f"<title>trading-lab run {result.run_id}</title>" in page
    assert f"{result.metrics.total_return:+.2%}" in page and f"{result.benchmark.total_return:+.2%}" in page
    assert f">{result.metrics.num_trades}<" in page
    assert page.count("<polyline") == 3  # strategy, buy & hold, drawdown
    assert ">Strategy</text>" in page and ">Buy &amp; hold</text>" in page  # direct end labels
    assert 'class="legend"' in page and "Table view (daily close)" in page
    for agent in ("qwen_trend (AI)", "qwen_momentum (AI)", "qwen_risk (AI)", "rsi"):
        assert f"<td>{agent}</td>" in page
    assert "Model usage" in page and "Closed trades" in page and "generated 2024-03-01 00:00 UTC" in page
    points = json.loads(re.search(r'class="chart" data-points="([^"]*)"', page).group(1)
                        .replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    assert len(points) == len(result.equity_curve)


def test_model_text_cannot_inject_html(run):
    db, _ = run
    page = build_html_report(str(db))
    assert "<script>alert" not in page and "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert page.count("<script>") == 1  # only the report's own hover script


def test_report_is_self_contained(run):
    db, _ = run
    page = build_html_report(str(db))
    assert not re.search(r"""(src|href)\s*=\s*["']?https?:""", page)
    assert "<link" not in page and "<script src" not in page and "@import" not in page
    assert "prefers-color-scheme:dark" in page and '[data-theme="dark"]' in page


def test_long_histories_are_downsampled():
    index = pd.date_range("2024-01-01", periods=5000, freq="h", tz="UTC")
    frame = pd.DataFrame({"equity": range(5000)}, index=index)
    chart = line_chart(frame, [("equity", "Strategy", "--series-1")])
    points = json.loads(re.search(r'data-points="([^"]*)"', chart).group(1).replace("&quot;", '"')
                        .replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))
    assert len(points) <= MAX_POINTS + 1 and "2024-07-27" in points[-1]["label"]  # the last point is kept


def test_empty_runs_and_cli(tmp_path, capsys, run):
    cfg = AppConfig()
    db = tmp_path / "p.db"
    with SQLiteStore(db) as store:
        store.create_run("pp-1", kind="paper", timeframe="1h", symbols=cfg.market.symbols, exchange="x",
                         config=cfg.to_dict(), config_fingerprint=cfg.fingerprint(), period_start=START)
    assert "No equity recorded yet." in build_html_report(str(db))
    with SQLiteStore(db) as store:
        store.add_snapshots("pp-1", [PortfolioSnapshot(START + timedelta(hours=i), 10_000, 0, 10_000, 0, 0, 0, 0)
                                     for i in range(3)])
    out = tmp_path / "out" / "r.html"
    assert main(["--db", str(db), "report", "--html", str(out)]) == 0
    assert out.exists() and "HTML report written" in capsys.readouterr().out
    assert main(["--db", str(tmp_path / "missing.db"), "report", "--html", str(out)]) == 1
    assert "no database" in capsys.readouterr().err
    run_db, result = run
    assert main(["--db", str(run_db), "report", result.run_id, "--html", str(out), "--horizon", "6"]) == 0
    assert "Voters (6-bar outcome horizon)" in out.read_text()
