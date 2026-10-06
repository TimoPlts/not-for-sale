"""Stage 15A: the offline demo."""

from datetime import datetime, timezone

import pytest

from trading_lab.cli import main
from trading_lab.demo import DEMO_PAPER_RUN, build_demo, next_steps
from trading_lab.storage import SQLiteStore

NOW = datetime(2025, 3, 10, 14, 37, tzinfo=timezone.utc)


def test_the_demo_builds_a_complete_offline_sample(tmp_path):
    result = build_demo(tmp_path / "demo", days=10, paper_bars=24, now=NOW)
    assert sorted(p.name for p in result.directory.iterdir()) == ["backtest-report.html", "demo.db",
                                                                  "paper-report.html"]
    assert result.reconciled and result.paper_run == DEMO_PAPER_RUN
    with SQLiteStore(result.db_path, readonly=True) as store:
        runs = {r["run_id"]: r for r in store.list_runs()}
        assert set(runs) == {result.backtest_run, DEMO_PAPER_RUN}
        assert runs[result.backtest_run]["kind"] == "backtest" and runs[DEMO_PAPER_RUN]["kind"] == "paper"
        paper = store.load_equity_curve(DEMO_PAPER_RUN)
        assert len(paper) == 24 and paper.index[-1] == datetime(2025, 3, 10, 13, tzinfo=timezone.utc)
        assert store.get_run(DEMO_PAPER_RUN)["config"]["strategies"]  # the defaults, agents off
        assert store.count("signals", DEMO_PAPER_RUN) > 0
    assert result.backtest_run in (result.directory / "backtest-report.html").read_text()
    assert DEMO_PAPER_RUN in (result.directory / "paper-report.html").read_text()
    assert result.paper_equity == pytest.approx(paper["equity"].iloc[-1])


def test_the_demo_is_deterministic(tmp_path):
    a = build_demo(tmp_path / "a", days=5, paper_bars=12, now=NOW)
    b = build_demo(tmp_path / "b", days=5, paper_bars=12, now=NOW)
    assert (a.backtest_return, a.paper_equity, a.paper_fills) == (b.backtest_return, b.paper_equity, b.paper_fills)
    c = build_demo(tmp_path / "c", days=5, paper_bars=12, now=NOW, seed=8)
    assert c.backtest_return != a.backtest_return


def test_existing_files_are_never_touched(tmp_path):
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "mine.txt").write_text("keep")
    with pytest.raises(ValueError, match="not an empty directory"):
        build_demo(tmp_path / "busy", now=NOW)
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError, match="not an empty directory"):
        build_demo(tmp_path / "file", now=NOW)
    with pytest.raises(ValueError, match="days"):
        build_demo(tmp_path / "x", days=1, now=NOW)
    assert (tmp_path / "busy" / "mine.txt").read_text() == "keep" and not (tmp_path / "x").exists()


def test_next_steps_point_at_the_demo_database(tmp_path):
    result = build_demo(tmp_path / "d", days=5, paper_bars=6, now=NOW)
    commands = [c for _, c in next_steps(result)]
    assert any(c.endswith(f"reconcile {DEMO_PAPER_RUN}") and result.db_path.as_posix() in c for c in commands)
    assert any(f"export {result.backtest_run}" in c for c in commands)


def test_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "--days", "5", "--paper-bars", "12"]) == 0
    out = capsys.readouterr().out
    assert "reconcile OK: the paper run matches its backtest" in out
    assert "trading-lab --db demo/demo.db report" in out and "say nothing about real markets" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["demo"]  # nothing written elsewhere
    assert main(["demo"]) == 1  # the directory is no longer empty
    assert main(["--db", "demo/demo.db", "reconcile", DEMO_PAPER_RUN]) == 0
