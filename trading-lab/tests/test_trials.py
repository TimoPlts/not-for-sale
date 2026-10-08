"""Stage 26: the research trial log and the checkup's "Beats your other trials" check."""

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from test_permutation import FOLLOWER, Trending
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.metrics import periods_per_year
from trading_lab.metrics.sharpe import deflated_sharpe, expected_max_sharpe
from trading_lab.research import checkup
from trading_lab.research.checkup import FAIL, PASS
from trading_lab.research.trials import (
    TrialSummary,
    deflated_against_log,
    format_trial_summary,
    per_bar_sharpe,
    record_trials,
    trial,
    trial_key,
    trial_summary,
)
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
D = timedelta(days=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})


def row(start, end, sharpe, *, fingerprint="a", command="sweep", timeframe="1h", ret=0.01, label=""):
    return {"created_at": T0, "command": command, "label": label, "config_fingerprint": fingerprint,
            "timeframe": timeframe,
            "symbols": ["BTC/USDT"], "period_start": start, "period_end": end, "total_return": ret,
            "sharpe_ratio": sharpe, "num_bars": 100}


def test_storage_and_overlap(tmp_path):
    with SQLiteStore(tmp_path / "h.db") as store:
        assert store.add_trials([
            row(T0, T0 + 10 * D, 1.0, fingerprint="a"),
            row(T0 + 20 * D, T0 + 30 * D, 2.0, fingerprint="b"),
            row(T0, T0 + 30 * D, None, fingerprint="c", timeframe="4h"),
        ]) == 3
        assert [t["config_fingerprint"] for t in store.load_trials()] == ["a", "b", "c"]
        assert [t["config_fingerprint"] for t in store.load_trials(timeframe="1h")] == ["a", "b"]
        inside = store.load_trials(timeframe="1h", start=T0 + 5 * D, end=T0 + 15 * D)
        assert [t["config_fingerprint"] for t in inside] == ["a"]  # any overlap counts
        between = store.load_trials(start=T0 + 10 * D, end=T0 + 20 * D)  # touching ends do not overlap
        assert [t["config_fingerprint"] for t in between] == ["c"]
        t = store.load_trials()[0]
        assert t["period_start"] == T0 and t["symbols"] == ["BTC/USDT"] and t["sharpe_ratio"] == 1.0


def test_old_databases(tmp_path):
    path = tmp_path / "v4.db"
    SQLiteStore(path).close()
    conn = sqlite3.connect(path)
    conn.executescript("DROP TABLE trials; PRAGMA user_version = 4;")
    conn.close()
    with SQLiteStore(path, readonly=True) as store:  # read-only: no upgrade, no trials
        assert store.load_trials() == []
    with SQLiteStore(path) as store:
        assert store.schema_version == 5 and store.add_trials([row(T0, T0 + D, 1.0)]) == 1


def test_summary():
    rows = [row(T0, T0 + 10 * D, s, fingerprint=f) for f, s in (("a", 1.0), ("b", -0.5), ("c", 2.0))]
    rows.append(dict(rows[0]))  # the same config on the same period again
    rows.append(row(T0, T0 + 9 * D, 3.0, fingerprint="a"))  # same config, another period: a new trial
    s = TrialSummary("1h", T0, T0 + 10 * D, rows)
    assert s.count == 4 and s.configs == 3 and len(s.trials) == 5
    ppy = periods_per_year("1h")
    per_bar = [1.0, -0.5, 2.0, 3.0]
    assert s.sharpes == pytest.approx([v / math.sqrt(ppy) for v in per_bar])
    variance = float(__import__("numpy").var([v / math.sqrt(ppy) for v in per_bar], ddof=1))
    assert s.luck_bar == pytest.approx(expected_max_sharpe(4, variance) * math.sqrt(ppy))
    assert s.best["sharpe_ratio"] == 3.0 and s.by_command() == {"sweep": 5}
    json.dumps(s.to_dict())
    text = format_trial_summary(s)
    assert "4 (3 distinct configs; 1 repeated run(s) not counted again)" in text and "by luck alone" in text
    empty = TrialSummary("1h", T0, T0 + D)
    assert empty.count == 0 and empty.luck_bar is None and empty.best is None
    assert "record_trials = true" in format_trial_summary(empty)
    assert per_bar_sharpe(row(T0, T0, None)) is None and per_bar_sharpe(row(T0, T0, float("nan"))) is None


def test_deflated_against_log():
    import numpy as np

    returns = np.random.default_rng(0).normal(0.001, 0.01, 500)
    rows = [row(T0, T0 + 10 * D, s, fingerprint=str(i)) for i, s in enumerate((0.5, 1.5, -1.0, 2.5))]
    s = TrialSummary("1h", T0, T0 + 10 * D, rows)
    new = row(T0, T0 + 10 * D, 3.0, fingerprint="new")
    expected = deflated_sharpe(returns, [per_bar_sharpe(r) for r in [*rows, new]])
    assert deflated_against_log(s, returns, include=new) == pytest.approx(expected)
    again = dict(rows[0])  # already logged: not counted twice
    assert deflated_against_log(s, returns, include=again) == pytest.approx(
        deflated_sharpe(returns, [per_bar_sharpe(r) for r in rows]))
    assert trial_key(again) == trial_key(rows[0])


def test_recording_is_opt_in(tmp_path):
    from trading_lab.metrics import compute_metrics

    metrics = compute_metrics([100.0, 101.0, 100.5, 102.0], [], "1h")
    db = tmp_path / "h.db"
    off = CFG.with_overrides({"storage": {"db_path": str(db)}})
    assert record_trials(off, [trial("backtest", off, T0, T0 + D, metrics)]) == 0 and not db.exists()
    on = CFG.with_overrides({"storage": {"db_path": str(db), "record_trials": True}})
    assert record_trials(on, [trial("backtest", on, T0, T0 + D, metrics, "x")]) == 1
    with SQLiteStore(db, readonly=True) as store:
        (logged,) = store.load_trials()
    assert logged["config_fingerprint"] == on.fingerprint() and logged["label"] == "x"
    assert logged["sharpe_ratio"] == pytest.approx(metrics.sharpe_ratio) and logged["num_bars"] == 3


def config_file(tmp_path, record=True, name="c.toml", extra=""):
    path = tmp_path / name
    path.write_text(f'[storage]\ndb_path = "{tmp_path / "h.db"}"\nrecord_trials = {str(record).lower()}\n'
                    f'[market]\nsymbols = ["BTC/USDT"]\n{extra}')
    return str(path)


def test_cli_logs_every_command(tmp_path, capsys):
    period = ["--synthetic", "3", "--start", "2024-02-01", "--end", "2024-02-11"]
    cfg = config_file(tmp_path)
    assert main(["--config", cfg, "backtest", *period[:2], "--start", "2024-02-01", "--end", "2024-02-11",
                 "--no-db"]) == 0
    assert main(["--config", cfg, "sweep", *period, "--param", "strategies.rsi.period=7,14,21"]) == 0
    other = config_file(tmp_path, name="b.toml", extra="[voting]\nmin_agreeing = 2\n")
    assert main(["--config", cfg, "ab", cfg, other, *period, "--windows", "2"]) == 0
    assert main(["--config", cfg, "permutation-test", *period, "--permutations", "2"]) == 0
    assert main(["--config", cfg, "checkup", *period, "--permutations", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("trial(s) logged") == 5 and "3 trial(s) logged" in out and "4 trial(s) logged" in out
    assert "Beats your other trials" in out  # the checkup found the earlier trials
    with SQLiteStore(tmp_path / "h.db", readonly=True) as store:
        rows = store.load_trials()
    assert [r["command"] for r in rows] == ["backtest", *["sweep"] * 3, *["ab"] * 4, "permutation-test", "checkup"]
    assert {r["label"].split()[0] for r in rows if r["command"] == "ab"} == {"A", "B"}
    assert main(["--config", cfg, "trials"]) == 0
    text = capsys.readouterr().out
    assert "by command: ab 4, backtest 1, checkup 1, permutation-test 1, sweep 3" in text
    assert "recording is off" not in text
    assert main(["--config", cfg, "trials", "--json", "--start", "2024-02-05", "--end", "2024-02-06"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["trials"]) == 8  # all but the second A/B window (both configs), which starts on the 6th
    # distinct (config, period): the base config on the whole period (backtest, sweep rsi 14, permutation test
    # and checkup are all the same), sweep rsi 7 and 21, and A and B in the first window
    assert data["count"] == 5


def test_cli_without_recording(tmp_path, capsys):
    cfg = config_file(tmp_path, record=False)
    assert main(["--config", cfg, "sweep", "--synthetic", "3", "--start", "2024-02-01", "--end", "2024-02-05",
                 "--param", "strategies.rsi.period=7,14"]) == 0
    assert "logged" not in capsys.readouterr().out and not (tmp_path / "h.db").exists()
    assert main(["--config", cfg, "trials"]) == 0
    out = capsys.readouterr().out
    assert "None recorded" in out and "recording is off" in out
    assert main(["--config", cfg, "trials", "--start", "2024-02-05", "--end", "2024-02-01"]) == 1


def test_checkup_check_grades():
    start = T0 + timedelta(hours=600)
    end = start + timedelta(hours=1000)
    plain = checkup(FOLLOWER, Trending(), start, end, permutations=2)
    assert "Beats your other trials" not in {c.name for c in plain.checks}  # no log, no check
    weak = TrialSummary(FOLLOWER.market.timeframe, start, end,
                        [row(start, end, s, fingerprint=str(i)) for i, s in enumerate((0.1, -0.2, 0.3))])
    report = checkup(FOLLOWER, Trending(), start, end, permutations=2, trials=weak)
    check = next(c for c in report.checks if c.name == "Beats your other trials")
    assert report.trials == 4 and check.status == PASS and report.deflated_sharpe >= 0.95
    lucky = TrialSummary(FOLLOWER.market.timeframe, start, end,
                         [row(start, end, s, fingerprint=str(i)) for i, s in enumerate(range(-400, 400, 4))])
    report = checkup(FOLLOWER, Trending(), start, end, permutations=2, trials=lucky)
    check = next(c for c in report.checks if c.name == "Beats your other trials")
    assert report.trials == 201 and check.status == FAIL and "201 logged trials" in check.detail
    assert report.to_dict()["trials"] == 201


def test_store_summary_helper(tmp_path):
    with SQLiteStore(tmp_path / "h.db") as store:
        store.add_trials([row(T0, T0 + D, 1.0), row(T0 + 5 * D, T0 + 6 * D, 2.0, fingerprint="b")])
        s = trial_summary(store, "1h", T0, T0 + 2 * D)
    assert s.count == 1 and s.trials[0]["config_fingerprint"] == "a"
