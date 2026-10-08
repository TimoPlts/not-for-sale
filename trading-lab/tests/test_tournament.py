"""Stage 31: tournament (rank candidates for a paper run) and paper-plan (set the run up)."""

import json
import tomllib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from test_live import ANCHOR, START as LIVE_START, Clock
from trading_lab.cli import main
from trading_lab.config import AppConfig, load_config
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.presets import PRESETS
from trading_lab.research.tournament import Entry, Tournament, format_tournament, tournament
from trading_lab.research.validate import NOT_READY, PAPER, STRONG
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


def fake(name, recommendation, wins=0, compared=0, sharpe=0.0, deflated=None):
    metrics = SimpleNamespace(sharpe_ratio=sharpe, total_return=0.01, max_drawdown=0.05)
    ab = None if compared == 0 else SimpleNamespace(b_wins=wins, compared=compared)
    validation = SimpleNamespace(recommendation=recommendation, ab=ab, reasons=["because"],
                                 checkup=SimpleNamespace(metrics=metrics, overall="warn"))
    return Entry(name, validation, deflated)


def test_ranking_and_text():
    entries = [fake("c", NOT_READY, 6, 6, 3.0), fake("b", PAPER, 2, 6, 1.0), fake("a", PAPER, 4, 6, 0.5),
               fake("s", STRONG, 6, 6, 0.1, deflated=0.3), fake("d", PAPER, 4, 6, 0.9)]
    ranked = sorted(entries, key=Entry.sort_key)
    assert [e.name for e in ranked] == ["s", "d", "a", "b", "c"]  # recommendation, then wins, then Sharpe
    t = Tournament("base", START, START + timedelta(days=10), ranked)
    assert t.winner.name == "s"
    text = format_tournament(t)
    assert "Next paper run: s (strong candidate)" in text and "trading-lab paper-plan s" in text
    assert "may simply be the luckiest of 5 candidates" in text
    none = Tournament("base", START, START, [fake("x", NOT_READY), fake("y", NOT_READY)])
    assert none.winner is None and "No candidate is ready" in format_tournament(none)


def test_tournament_on_synthetic_data():
    candidates = {name: PRESETS[name].config(BASE) for name in ("trend", "conservative")}
    t = tournament(candidates, BASE, SyntheticProvider(seed=3), START, START + timedelta(days=20), windows=2,
                   permutations=2)
    assert {e.name for e in t.entries} == set(candidates)
    assert [e.sort_key() for e in t.entries] == sorted(e.sort_key() for e in t.entries)
    assert all(e.deflated is not None and 0 <= e.deflated <= 1 for e in t.entries)
    assert all(e.validation.returns for e in t.entries)
    json.dumps(t.to_dict(), default=str)
    with pytest.raises(ValueError):
        tournament({}, BASE, SyntheticProvider(seed=3), START, START + timedelta(days=5))


def test_tournament_cli(tmp_path, capsys):
    custom = tmp_path / "mine.toml"
    custom.write_text('[market]\nsymbols = ["BTC/USDT"]\n[voting]\nmin_agreeing = 2\n')
    base = tmp_path / "base.toml"
    base.write_text(f'[market]\nsymbols = ["BTC/USDT"]\n[storage]\ndb_path = "{tmp_path / "h.db"}"\n'
                    "record_trials = true\n")
    html, js = tmp_path / "t.html", tmp_path / "t.json"
    args = ["tournament", "trend", str(custom), "--baseline", str(base), "--synthetic", "3", "--start", "2024-02-01",
            "--end", "2024-02-21", "--windows", "2", "--permutations", "2", "--html", str(html), "--json", str(js)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Tournament: 2 candidate(s)" in out and " trend " in out and " mine " in out
    data = json.loads(js.read_text())
    assert {e["name"] for e in data["entries"]} == {"trend", "mine"} and "Ranking" in html.read_text()
    assert main(["tournament", "nope", "--baseline", str(base), "--synthetic", "3"]) == 1


def test_paper_plan(tmp_path, capsys):
    base = tmp_path / "base.toml"
    db = tmp_path / "h.db"
    base.write_text(f'[market]\nsymbols = ["BTC/USDT", "ETH/USDT"]\n[storage]\ndb_path = "{db}"\n')
    clock = Clock(LIVE_START + timedelta(hours=1, minutes=1))
    with SQLiteStore(db) as store:  # the baseline's running paper run
        LivePaperTrader(load_config(base), SyntheticProvider(seed=1, anchor=ANCHOR, clock=clock), store,
                        clock=clock, run_id="vm-paper-1").run_cycle()
    runs = tmp_path / "runs"
    assert main(["paper-plan", "desk", "--baseline", str(base), "--dir", str(runs)]) == 0
    out = capsys.readouterr().out
    written = runs / "desk.toml"
    assert AppConfig.from_mapping(tomllib.loads(written.read_text())) == PRESETS["desk"].config(load_config(base))
    assert "enable --now trading-lab-paper@desk" in out and "live-compare vm-paper-1 desk" in out
    assert main(["paper-plan", "desk", "--baseline", str(base), "--dir", str(runs)]) == 1  # exists
    other = tmp_path / "other.toml"
    other.write_text('[storage]\ndb_path = "elsewhere.db"\n[voting]\nmin_agreeing = 2\n')
    assert main(["paper-plan", str(other), "--run-id", "try_2", "--baseline", str(base), "--dir", str(runs)]) == 0
    assert "storage.db_path set to" in capsys.readouterr().out
    planned = load_config(runs / "try_2.toml")
    assert planned.storage.db_path == str(db) and planned.voting.min_agreeing == 2
    assert main(["paper-plan", "desk", "--run-id", "bad id!", "--baseline", str(base), "--dir", str(runs)]) == 1
    assert main(["paper-plan", "nope", "--baseline", str(base), "--dir", str(runs)]) == 1
