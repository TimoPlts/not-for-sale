"""CLI smoke tests (offline, synthetic data, temporary database)."""

import pytest

from conftest import PROJECT_ROOT
from trading_lab.cli import main
from trading_lab.storage import SQLiteStore

CONFIG = str(PROJECT_ROOT / "config" / "default.toml")


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cli.db")


def run(db, *args):
    return main(["--config", CONFIG, "--db", db, *args])


def latest_run_id(db):
    with SQLiteStore(db) as store:
        return store.list_runs(1)[0]["run_id"]


def test_backtest_and_report(db, capsys, tmp_path):
    code = run(db, "backtest", "--synthetic", "3", "--start", "2024-01-01", "--end", "2024-01-15",
               "--symbols", "BTC/USDT", "ETH/USDT", "--export", str(tmp_path / "out"))
    out = capsys.readouterr().out
    assert code == 0
    assert "Total return" in out and "Saved as run bt-" in out
    assert (tmp_path / "out" / "trades.csv").exists()
    run_id = latest_run_id(db)

    assert run(db, "report") == 0
    assert run_id in capsys.readouterr().out
    assert run(db, "report", run_id) == 0
    detail = capsys.readouterr().out
    assert "=== Performance ===" in detail and "Closed trades" in detail


def test_paper_once_and_resume(db, capsys):
    assert run(db, "paper", "--synthetic", "5", "--once", "--timeframe", "15m") == 0
    out = capsys.readouterr().out
    assert "Started paper run pp-" in out and "simulated fills only" in out
    run_id = latest_run_id(db)
    assert run(db, "paper", "--resume", run_id, "--once") == 0
    assert f"Resuming paper run {run_id}" in capsys.readouterr().out
    with SQLiteStore(db) as store:
        run_row = store.get_run(run_id)
        assert run_row["status"] == "stopped" and run_row["timeframe"] == "15m"


def test_signals_command(db, capsys):
    assert run(db, "signals", "--synthetic", "1", "--symbols", "SOL/USDT", "--days", "5") == 0
    out = capsys.readouterr().out
    assert "SOL/USDT" in out and "ENSEMBLE" in out


@pytest.mark.parametrize(
    "args, message",
    [
        (["report", "bt-does-not-exist"], "unknown run id"),
        (["paper", "--resume", "pp-nope", "--once"], "unknown run id"),
        (["backtest", "--synthetic", "1", "--timeframe", "7m"], "unsupported timeframe"),
        (["backtest", "--synthetic", "1", "--symbols", "XRP/USDT"], "unsupported symbols"),
    ],
)
def test_errors_are_reported_cleanly(db, capsys, args, message):
    assert run(db, *args) == 1
    assert message in capsys.readouterr().err


def test_backtest_runs_cannot_be_resumed_as_paper(db, capsys):
    run(db, "backtest", "--synthetic", "2", "--start", "2024-01-01", "--end", "2024-01-03")
    run_id = latest_run_id(db)
    capsys.readouterr()
    assert run(db, "paper", "--resume", run_id, "--once") == 1
    assert "only paper runs can be resumed" in capsys.readouterr().err
