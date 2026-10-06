"""Stage 10C: VM operation: named runs, log files, SIGTERM, service templates and docs."""

import os
import re
import signal
import subprocess
import threading
import time
from datetime import timedelta

import pytest

from conftest import PROJECT_ROOT
from test_live import ANCHOR, START, Clock
from test_safety import HARDCODED_TOKENS
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.storage import SQLiteStore

DEPLOY = PROJECT_ROOT / "deploy"
DOCS = PROJECT_ROOT / "docs"


def paper(db, *args):
    return main(["--db", str(db), "paper", "--synthetic", "4", "--symbols", "BTC/USDT", "--once", *args])


def test_named_runs_start_once_then_resume(tmp_path, capsys):
    db = tmp_path / "h.db"
    assert paper(db, "--run-id", "vm-test-1") == 0
    assert "Started paper run vm-test-1" in capsys.readouterr().out
    assert main(["--db", str(db), "paper", "--synthetic", "4", "--once", "--run-id", "vm-test-1"]) == 0
    assert "Resuming paper run vm-test-1" in capsys.readouterr().out
    with SQLiteStore(db) as store:
        assert [r["run_id"] for r in store.list_runs()] == ["vm-test-1"]


@pytest.mark.parametrize("args, message", [
    (["--run-id", "bad id!"], "--run-id may only contain"),
    (["--run-id", "a", "--resume", "a"], "either --run-id or --resume"),
])
def test_named_run_validation(tmp_path, capsys, args, message):
    assert paper(tmp_path / "h.db", *args) == 1
    assert message in capsys.readouterr().err


def test_log_file_records_paper_activity(tmp_path, capsys):
    log = tmp_path / "logs" / "paper.log"
    assert main(["--log-file", str(log), "--db", str(tmp_path / "h.db"), "paper", "--synthetic", "4",
                 "--symbols", "BTC/USDT", "--once", "--run-id", "logged"]) == 0
    text = log.read_text()
    assert "Started paper run logged" in text and "new candles=" in text and "stopped" in text
    assert "INFO trading_lab.cli" in text
    assert main(["--db", str(tmp_path / "h.db"), "report"]) == 0  # handlers are replaced, not stacked
    capsys.readouterr()


def test_stop_event_ends_the_loop_after_the_current_cycle(tmp_path):
    clock = Clock(START + timedelta(hours=1, minutes=1))
    stop = threading.Event()
    with SQLiteStore(tmp_path / "h.db") as store:
        cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
        trader = LivePaperTrader(cfg, SyntheticProvider(seed=1, anchor=ANCHOR, clock=clock), store, clock=clock)
        started = time.monotonic()
        cycles = trader.run_forever(poll_seconds=3600, on_cycle=lambda report: stop.set(), stop_event=stop)
        assert cycles == 1 and time.monotonic() - started < 30  # no hour-long wait
        assert store.get_run(trader.run_id)["status"] == "stopped"


def test_sigterm_requests_a_graceful_stop(tmp_path, capsys, monkeypatch):
    seen = {}

    def fake_run_forever(self, *, stop_event, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)  # what `systemctl stop` sends
        seen["stopped"] = stop_event.wait(5)
        self.stop()
        return 1

    monkeypatch.setattr(LivePaperTrader, "run_forever", fake_run_forever)
    before = signal.getsignal(signal.SIGTERM)
    assert paper(tmp_path / "h.db", "--run-id", "term") == 0
    assert seen["stopped"] is True
    assert "received signal" in capsys.readouterr().out
    assert signal.getsignal(signal.SIGTERM) == before  # handler restored afterwards


def test_service_templates():
    paper_unit = (DEPLOY / "systemd" / "trading-lab-paper.service").read_text()
    dash_unit = (DEPLOY / "systemd" / "trading-lab-dashboard.service").read_text()
    assert "EnvironmentFile=/etc/trading-lab/trading-lab.env" in paper_unit
    assert "paper --run-id" in paper_unit and "KillSignal=SIGTERM" in paper_unit
    assert "Restart=on-failure" in paper_unit and "--log-file /var/log/trading-lab/paper.log" in paper_unit
    assert "EnvironmentFile" not in dash_unit.replace("# No EnvironmentFile", "")
    assert "dashboard --host 127.0.0.1" in dash_unit and "ReadOnlyPaths=" in dash_unit
    for unit in (paper_unit, dash_unit):
        assert "QWEN_API_KEY" not in unit and "NoNewPrivileges=true" in unit


def test_environment_template_has_placeholders_only():
    text = (DEPLOY / "trading-lab.env.example").read_text()
    values = dict(re.findall(r"^(QWEN_\w+)=(.*)$", text, re.MULTILINE))
    assert set(values) == {"QWEN_API_URL", "QWEN_MODEL", "QWEN_API_KEY"}
    assert values["QWEN_API_KEY"] == "change-me" and "example" in values["QWEN_API_URL"]
    assert not HARDCODED_TOKENS.search(text)


@pytest.mark.parametrize("path, ignored", [
    ("deploy/trading-lab.env", True),
    ("trading-lab.env", True),
    (".env", True),
    ("deploy/trading-lab.env.example", False),
])
def test_secret_files_are_git_ignored(path, ignored):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=PROJECT_ROOT)
    assert (result.returncode == 0) is ignored


def test_deployment_guide_covers_operations():
    guide = (DOCS / "DEPLOYMENT.md").read_text()
    for topic in ("python3 -m venv", "pip install -e", "QWEN_API_URL", "QWEN_MODEL", "QWEN_API_KEY",
                  "/etc/trading-lab/trading-lab.env", "agent-test", "paper --run-id", "trading-lab-dashboard",
                  "journalctl", "/var/log/trading-lab/paper.log", "trading_lab.db", "agent_cache.db",
                  "systemctl stop", "--resume", "ssh -L", ".backup"):
        assert topic in guide, topic
