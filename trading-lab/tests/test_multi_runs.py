"""Stage 22A: several paper runs side by side (shared database, status --all, systemd template)."""

import json
from datetime import timedelta
from pathlib import Path

from test_alerts import SECRET_URL, Capture
from test_live import ANCHOR, START, Clock
from trading_lab.alerts import URL_ENV
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.presets import PRESETS
from trading_lab.status import check_status, running_paper_runs
from trading_lab.storage import SQLiteStore

H = timedelta(hours=1)
DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


def test_two_runs_share_one_database(tmp_path):
    db = tmp_path / "shared.db"
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(db) as store_a, SQLiteStore(db) as store_b:  # two connections, like two services
        a = LivePaperTrader(BASE, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store_a, clock=clock,
                            run_id="default")
        b = LivePaperTrader(PRESETS["trend"].config(BASE), SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock),
                            store_b, clock=clock, run_id="trend")
        for _ in range(48):
            a.run_cycle()
            b.run_cycle()
            clock.now += H
        clock.now -= H
        assert sorted(running_paper_runs(store_a)) == ["default", "trend"]
        assert all(check_status(store_a, r, now=clock.now).ok for r in ("default", "trend"))
        assert store_a.count("equity_snapshots", "default") == store_a.count("equity_snapshots", "trend") == 48
    with SQLiteStore(db, readonly=True) as store:
        assert AppConfig.from_dict(store.get_run("trend")["config"]) == PRESETS["trend"].config(BASE)


def test_status_all(tmp_path, monkeypatch, capsys):
    db = tmp_path / "s.db"
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(db) as store:
        for run_id in ("alpha", "beta"):
            trader = LivePaperTrader(BASE, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store,
                                     clock=clock, run_id=run_id)
            trader.run_cycle()
        LivePaperTrader(BASE, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock,
                        run_id="gamma").stop()  # stopped on purpose: not checked by --all
    transport = Capture()
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
    monkeypatch.setenv(URL_ENV, SECRET_URL)
    cfg = tmp_path / "c.toml"
    cfg.write_text('[alerts]\nenabled = true\nformat = "json"\n')
    assert main(["--config", str(cfg), "--db", str(db), "status", "--all", "--alert"]) == 1  # both long stalled
    out = capsys.readouterr().out
    assert "Paper run alpha" in out and "Paper run beta" in out and "gamma" not in out
    sent = json.loads(transport.requests[0]["body"])
    assert sent["title"] == "Paper trading needs attention" and sent["body"].count("stalled") == 2
    assert sent["body"].startswith("alpha: stalled")
    assert main(["--db", str(db), "status", "--all", "--json", "--max-behind", "10000000"]) == 0
    assert [s["run_id"] for s in json.loads(capsys.readouterr().out)] == ["alpha", "beta"]
    assert main(["--db", str(db), "status", "alpha", "--all"]) == 1  # one or the other


def test_no_running_run_is_a_problem_unless_allowed(tmp_path, capsys):
    db = tmp_path / "s.db"
    with SQLiteStore(db) as store:
        clock = Clock(START + H + timedelta(minutes=1))
        LivePaperTrader(BASE, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock).stop()
    assert main(["--db", str(db), "status", "--all"]) == 1
    assert "No paper run is running." in capsys.readouterr().out
    assert main(["--db", str(db), "status", "--all", "--allow-stopped"]) == 0


def test_systemd_templates():
    unit = (DEPLOY / "trading-lab-paper@.service").read_text()
    assert "--config /opt/trading-lab/trading-lab/config/runs/%i.toml" in unit
    assert "paper --run-id %i" in unit and "--log-file /var/log/trading-lab/paper-%i.log" in unit
    assert "EnvironmentFile=/etc/trading-lab/trading-lab.env" in unit and "KillSignal=SIGTERM" in unit
    assert "NoNewPrivileges=true" in unit and "QWEN_API_KEY" not in unit
    assert "status --all --alert" in (DEPLOY / "trading-lab-watchdog.service").read_text()
