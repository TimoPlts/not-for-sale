"""Stage 20A: the strategy checkup (every check, one verdict)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from test_permutation import FOLLOWER, Trending
from test_specialists import RoleTransport, agents_config, provider
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.research import checkup, checkup_html, format_checkup
from trading_lab.research.checkup import FAIL, NA, PASS, WARN

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def statuses(report):
    return {c.name: c.status for c in report.checks}


def test_a_real_edge_passes_the_key_checks():
    start = T0 + 600 * H
    report = checkup(FOLLOWER, Trending(), start, start + 1000 * H, permutations=20)
    s = statuses(report)
    assert s["Edge before costs"] == s["Survives costs"] == s["Not luck"] == PASS
    assert s["Enough trades"] == PASS and s["Robust to resampling"] == PASS
    assert report.permutation.p_value == pytest.approx(1 / 21)
    assert report.costs.row(1.0).metrics == report.metrics  # the 1x cost run is the backtest itself
    assert set(s) == {"Enough trades", "Edge before costs", "Survives costs", "Not luck", "Robust to resampling",
                      "Sharpe is real", "Beats buy & hold", "Drawdown", "Works in several regimes"}
    assert s["Sharpe is real"] == PASS and report.metrics.probabilistic_sharpe >= 0.95


def test_no_edge_fails_with_the_right_advice():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
    report = checkup(cfg, SyntheticProvider(seed=7), T0 + 600 * H, T0 + 1600 * H, permutations=10)
    assert report.overall == FAIL and report.verdict.startswith("not convincing: failed")
    s = statuses(report)
    assert s["Survives costs"] == FAIL
    text = format_checkup(report)
    assert "[FAIL] Survives costs" in text and "Next steps:" in text
    assert report.metrics.total_return < 0  # the default strategies lose on this random walk
    assert "it lost money, so there is no result to tell apart from luck" in text
    data = json.loads(json.dumps(report.to_dict(), default=str))
    assert data["overall"] == FAIL and len(data["checks"]) == len(report.checks)


def test_agents_skip_the_luck_test_unless_allowed(tmp_path):
    cfg = agents_config(tmp_path)
    start = datetime(2024, 2, 1, tzinfo=UTC)
    report = checkup(cfg, SyntheticProvider(seed=3), start, start + timedelta(days=3), permutations=2,
                     llm_provider=provider(RoleTransport()))
    luck = next(c for c in report.checks if c.name == "Not luck")
    assert luck.status == NA and report.permutation is None and "skipped" in luck.detail


def test_short_periods_and_the_verdict_logic():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    report = checkup(cfg, SyntheticProvider(seed=7), T0 + 600 * H, T0 + 640 * H, permutations=2)
    s = statuses(report)
    assert s["Works in several regimes"] == NA and s["Enough trades"] in (WARN, FAIL)
    report.checks = [c for c in report.checks if c.status == PASS]
    assert report.overall == PASS and report.verdict.startswith("passes every check")


def test_html_page():
    start = T0 + 600 * H
    report = checkup(FOLLOWER, Trending(), start, start + 400 * H, permutations=3)
    page = checkup_html(report, title="Checkup <BTC>")
    assert page.startswith("<!doctype html>") and "Checkup &lt;BTC&gt;" in page
    assert "<td>Edge before costs</td>" in page and f"Verdict: {report.overall.upper()}" in page
    assert "http" not in page.replace("http-equiv", "")  # self-contained: nothing external


def test_cli(tmp_path, capsys):
    html, js = tmp_path / "c.html", tmp_path / "c.json"
    args = ["--db", str(tmp_path / "x.db"), "checkup", "--synthetic", "4", "--symbols", "BTC/USDT",
            "--start", "2024-02-01", "--end", "2024-02-11", "--permutations", "3"]
    assert main([*args, "--html", str(html), "--json", str(js)]) == 0
    out = capsys.readouterr().out
    assert "  backtest..." in out and "Verdict:" in out
    assert "Strategy checkup" in html.read_text() and json.loads(js.read_text())["checks"]
    assert not (tmp_path / "x.db").exists()


def test_sharpe_is_real_grades():
    from dataclasses import replace

    from trading_lab.research.checkup import evaluate

    start = T0 + 600 * H
    report = checkup(FOLLOWER, Trending(), start, start + 400 * H, permutations=2)
    for psr, status in ((0.99, PASS), (0.95, PASS), (0.9, WARN), (0.8, WARN), (0.5, FAIL), (None, NA)):
        report.metrics = replace(report.metrics, probabilistic_sharpe=psr)
        check = next(c for c in evaluate(report) if c.name == "Sharpe is real")
        assert check.status == status, psr
    assert "chance the true Sharpe ratio is above 0" in next(
        c for c in evaluate(replace(report, metrics=replace(report.metrics, probabilistic_sharpe=0.9)))
        if c.name == "Sharpe is real").detail
    lost = replace(report.metrics, probabilistic_sharpe=0.2, total_return=-0.05)
    check = next(c for c in evaluate(replace(report, metrics=lost)) if c.name == "Sharpe is real")
    assert check.status == FAIL and check.advice.startswith("it lost money")
