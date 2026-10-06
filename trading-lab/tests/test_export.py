"""Stage 14D: export a stored run to CSV files and a JSON summary."""

import hashlib
import json
from datetime import timedelta

import pandas as pd
import pytest

from test_live import ANCHOR, START, Clock
from test_specialists import END, ENV, RoleTransport, agents_config, provider
from test_specialists import START as AGENT_START
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.export import export_run
from trading_lab.live import LivePaperTrader
from trading_lab.storage import SQLiteStore

H = timedelta(hours=1)
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
FILES = ["bars.csv", "decisions.csv", "equity_curve.csv", "fills.csv", "signals.csv", "summary.json", "trades.csv"]


@pytest.fixture
def backtest(tmp_path):
    db = tmp_path / "bt.db"
    with SQLiteStore(db) as store:
        result = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR), store=store).run(START, START + 300 * H)
    return db, result


def test_a_backtest_export(backtest, tmp_path):
    db, result = backtest
    with SQLiteStore(db, readonly=True) as store:
        out = export_run(store, result.run_id, tmp_path / "out")
    assert sorted(p.name for p in out.directory.iterdir()) == FILES
    assert out.files["trades.csv"] == len(result.trades) > 0 and out.files["fills.csv"] == len(result.fills)
    assert out.files["equity_curve.csv"] == len(result.equity_curve) == 300
    assert out.files["bars.csv"] == 600 and out.files["signals.csv"] > 0

    summary = json.loads((out.directory / "summary.json").read_text())
    assert summary["run"]["run_id"] == result.run_id and summary["run"]["kind"] == "backtest"
    assert AppConfig.from_dict(summary["config"]) == CFG
    assert summary["metrics_source"] == "stored"
    assert summary["metrics"]["total_return"] == pytest.approx(result.metrics.total_return)
    assert summary["holds_included"] is False and "hold" not in summary["decision_counts"]
    for name, info in summary["files"].items():
        assert hashlib.sha256((out.directory / name).read_bytes()).hexdigest() == info["sha256"]
        assert len(pd.read_csv(out.directory / name)) == info["rows"]

    equity = pd.read_csv(out.directory / "equity_curve.csv")
    assert equity["equity"].iloc[-1] == pytest.approx(result.equity_curve["equity"].iloc[-1], rel=1e-12)
    fills = pd.read_csv(out.directory / "fills.csv")
    assert list(fills.columns) == ["timestamp", "symbol", "side", "quantity", "reference_price", "fill_price", "fee"]
    assert fills["fee"].sum() == pytest.approx(sum(f.fee for f in result.fills))


def test_same_files_as_backtest_export(tmp_path):
    db = str(tmp_path / "x.db")
    assert main(["--db", db, "backtest", "--synthetic", "1", "--days", "10", "--export", str(tmp_path / "a")]) == 0
    with SQLiteStore(db, readonly=True) as store:
        run_id = store.list_runs(1)[0]["run_id"]
        export_run(store, run_id, tmp_path / "b")
    for name in ("trades.csv", "fills.csv"):
        assert (tmp_path / "a" / name).read_text() == (tmp_path / "b" / name).read_text()


def test_holds_are_optional(backtest, tmp_path):
    db, result = backtest
    with SQLiteStore(db, readonly=True) as store:
        without = export_run(store, result.run_id, tmp_path / "a").files["decisions.csv"]
        with_holds = export_run(store, result.run_id, tmp_path / "b", holds=True).files["decisions.csv"]
    assert with_holds > without
    assert "hold" in json.loads((tmp_path / "b" / "summary.json").read_text())["decision_counts"]


def test_a_paper_run_export(tmp_path):
    db = tmp_path / "p.db"
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(db) as store:
        trader = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(100):
            trader.run_cycle()
            clock.now += H
        out = export_run(store, trader.run_id, tmp_path / "out")
    summary = json.loads((out.directory / "summary.json").read_text())
    assert summary["run"]["kind"] == "paper" and summary["metrics"] is not None
    assert out.files["equity_curve.csv"] == 100 and out.files["fills.csv"] == len(trader.portfolio.fills)


def test_agent_votes_are_readable_and_no_secret_is_written(tmp_path, monkeypatch):
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    cfg = agents_config(tmp_path)
    with SQLiteStore(tmp_path / "a.db") as store:
        run = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                             llm_provider=provider(RoleTransport())).run(AGENT_START, END)
        out = export_run(store, run.run_id, tmp_path / "out")
    signals = pd.read_csv(out.directory / "signals.csv")
    trend = signals[(signals["strategy"] == "qwen_trend") & signals["rationale"].notna()]
    assert len(trend) == 24 and (trend["label"] == "regime=bullish_trend").all()
    assert set(trend["cache"]) == {"stored"}
    for path in out.directory.iterdir():
        assert ENV["QWEN_API_KEY"] not in path.read_text()


def test_existing_directories_are_protected(backtest, tmp_path):
    db, result = backtest
    target = tmp_path / "out"
    target.mkdir()
    (target / "notes.txt").write_text("mine")
    with SQLiteStore(db, readonly=True) as store:
        with pytest.raises(ValueError, match="not empty"):
            export_run(store, result.run_id, target)
        export_run(store, result.run_id, target, overwrite=True)
        assert (target / "notes.txt").read_text() == "mine" and (target / "summary.json").exists()
        (tmp_path / "file").write_text("x")
        with pytest.raises(ValueError, match="not a directory"):
            export_run(store, result.run_id, tmp_path / "file")
        with pytest.raises(ValueError, match="unknown run"):
            export_run(store, "nope", tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_cli(backtest, tmp_path, capsys):
    db, result = backtest
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    assert main(["--db", str(db), "export", result.run_id, str(tmp_path / "e")]) == 0
    out = capsys.readouterr().out
    assert f"Run {result.run_id} exported to" in out and "trades.csv" in out
    assert main(["--db", str(db), "export", result.run_id, str(tmp_path / "e")]) == 1  # not empty
    assert main(["--db", str(db), "export", result.run_id, str(tmp_path / "e"), "--force", "--holds"]) == 0
    assert main(["--db", str(db), "export", "nope", str(tmp_path / "f")]) == 1
    assert main(["--db", str(tmp_path / "none.db"), "export", result.run_id, str(tmp_path / "g")]) == 1
    assert hashlib.sha256(db.read_bytes()).hexdigest() == digest
