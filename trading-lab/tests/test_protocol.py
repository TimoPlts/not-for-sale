"""Stage 10E: the experiment protocol and its out-of-sample comparison."""

import json
from types import SimpleNamespace

import pytest

from conftest import PROJECT_ROOT
from test_experiments import fake_qwen  # noqa: F401  (fixture)
from trading_lab.cli import main
from trading_lab.research import sign_test_p, summarize
from trading_lab.storage import SQLiteStore


def test_sign_test_values_used_in_the_protocol():
    assert sign_test_p(6, 6) == pytest.approx(1 / 64)          # 0.016: passes p <= 0.05
    assert sign_test_p(5, 6) == pytest.approx(7 / 64)          # 0.109: does not
    assert sign_test_p(10, 12) == pytest.approx(79 / 4096)     # 0.019: passes
    assert sign_test_p(9, 12) == pytest.approx(299 / 4096)     # 0.073: does not
    assert sign_test_p(0, 4) == 1.0 and sign_test_p(0, 0) is None


def metrics(sharpe, ret=0.01, dd=0.05, trades=5, pf=1.2, exposure=0.5):
    return SimpleNamespace(sharpe_ratio=sharpe, total_return=ret, max_drawdown=dd, num_trades=trades,
                           profit_factor=pf, exposure=exposure)


def row(variant, sharpes, **kw):
    folds = tuple(SimpleNamespace(out_of_sample=metrics(s, **kw)) for s in sharpes)
    wf = SimpleNamespace(folds=folds, out_of_sample_return=0.1, benchmark_return=0.05)
    return SimpleNamespace(variant=variant, walkforward=wf)


def test_summary_compares_each_fold_with_the_baseline():
    rows = [
        row("baseline", [1.0, 0.5, 0.2, None, 0.3]),
        row("trend", [1.5, 0.5, 0.4, 2.0, -0.1], dd=0.08, pf=float("inf"), trades=7),
    ]
    base, trend = summarize(rows)
    assert (base.compared, base.wins, base.sign_test_p) == (0, 0, None)
    assert (trend.compared, trend.wins) == (3, 2)  # the tie and the undefined fold are left out
    assert trend.sign_test_p == pytest.approx(sign_test_p(2, 3))
    assert trend.worst_fold_drawdown == 0.08 and trend.total_trades == 35
    assert trend.mean_profit_factor is None  # infinite profit factors are not averaged
    assert trend.mean_sharpe == pytest.approx((1.5 + 0.5 + 0.4 + 2.0 - 0.1) / 5)
    assert trend.oos_return == 0.1 and trend.benchmark_return == 0.05


def test_walkforward_experiment_reports_the_comparison(tmp_path, capsys, fake_qwen):  # noqa: F811
    out_json, db = tmp_path / "wf.json", tmp_path / "h.db"
    args = ["--db", str(db), "experiment", "--synthetic", "2", "--symbols", "BTC/USDT", "--start", "2024-02-01",
            "--end", "2024-02-11", "--variants", "baseline,trend,all_agents", "--walkforward",
            "--train-days", "4", "--test-days", "2", "--save", "--export", str(out_json)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "beats baseline" in out and "sign p" in out and "worst DD" in out
    exported = json.loads(out_json.read_text())
    assert [c["variant"] for c in exported["comparison"]] == ["baseline", "trend", "all_agents"]
    assert all(c["folds"] == 3 for c in exported["comparison"])
    with SQLiteStore(db) as store:
        saved = store.list_research_results("experiment")[0]["payload"]
    assert saved["comparison"] == exported["comparison"] and saved["metric"] == "sharpe_ratio"


def test_protocol_document():
    doc = (PROJECT_ROOT / "docs" / "EXPERIMENT_PROTOCOL.md").read_text()
    for topic in ("Experiment", "baseline", "trend_momentum", "all_agents", "identical", "walk-forward",
                  "--agent-mode replay", "total out-of-sample return", "buy & hold", "max drawdown", "Sharpe",
                  "profit factor", "trade count", "exposure", "agent contribution", "single backtest",
                  "sign test", "training cutoff", "Bonferroni"):
        assert topic.lower() in doc.lower(), topic
