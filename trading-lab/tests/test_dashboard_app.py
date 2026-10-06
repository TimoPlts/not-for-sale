"""Stage 10B: the Streamlit dashboard, rendered headlessly with Streamlit's AppTest."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from conftest import SRC_DIR  # noqa: E402
from test_specialists import RoleTransport, agents_config, provider  # noqa: E402
from trading_lab.backtest import BacktestEngine  # noqa: E402
from trading_lab.cli import main  # noqa: E402
from trading_lab.config import AppConfig  # noqa: E402
from trading_lab.core.models import PortfolioSnapshot  # noqa: E402
from trading_lab.data import SyntheticProvider  # noqa: E402
from trading_lab.storage import SQLiteStore  # noqa: E402
from trading_lab.strategy_factory import strategies_for  # noqa: E402

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 4, tzinfo=UTC)
APP = str(SRC_DIR / "dashboard" / "app.py")
SECTIONS = ["Portfolio", "Equity curve", "Drawdown", "Open positions", "Latest decision", "AI rationales",
            "Agent performance", "Research", "Model (Qwen) usage", "Recent trades", "Recent signals (non-HOLD)"]


@pytest.fixture(scope="module")
def agents_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    cfg = agents_config(tmp)
    with SQLiteStore(tmp / "h.db") as store:
        BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                       strategies=strategies_for(cfg, llm_provider=provider(RoleTransport()))).run(START, END)
        store.add_research_result("experiment", "walk-forward baseline,trend", {
            "mode": "walk-forward", "rows": [{"variant": "baseline", "walkforward": {"out_of_sample_return": 0.01,
                                                                                    "benchmark_return": 0.02}}]})
    return tmp / "h.db"


def render(monkeypatch, db):
    monkeypatch.setenv("TRADING_LAB_DB", str(db))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_dashboard_renders_every_section(monkeypatch, agents_db):
    before = hashlib.sha256(agents_db.read_bytes()).hexdigest()
    at = render(monkeypatch, agents_db)
    assert [h.value for h in at.subheader] == SECTIONS
    labels = {m.label for m in at.metric}
    assert {"Equity", "Daily PnL", "Drawdown", "Exposure", "Breakers", "Strategy return", "Buy & hold return",
            "Sharpe", "Calls", "Cache hits", "Tokens in / out"} <= labels
    assert next(m for m in at.metric if m.label == "Breakers").value == "OK"
    markdown = " ".join(m.value for m in at.markdown)
    for agent in ("qwen_trend", "qwen_momentum", "qwen_risk"):
        assert f"**{agent}**" in markdown  # one rationale card column per agent
    assert "ensemble" in markdown and "Saved experiments" in markdown
    assert hashlib.sha256(agents_db.read_bytes()).hexdigest() == before  # the dashboard wrote nothing


def test_dashboard_offers_no_actions(monkeypatch, agents_db):
    at = render(monkeypatch, agents_db)
    assert len(at.button) == 0 and len(at.text_input) == 0 and len(at.number_input) == 0
    assert len(at.text_area) == 0 and len(at.checkbox) == 0
    assert [w.label for w in at.sidebar.selectbox] == ["Run", "Auto-refresh"]  # view controls only


def test_kill_switch_is_shown(monkeypatch, tmp_path):
    db = tmp_path / "paper.db"
    cfg = AppConfig()
    t0 = datetime(2024, 3, 1, tzinfo=UTC)
    with SQLiteStore(db) as store:
        store.create_run("pp-1", kind="paper", timeframe="1h", symbols=cfg.market.symbols, exchange="synthetic-1",
                         config=cfg.to_dict(), config_fingerprint=cfg.fingerprint(), period_start=t0)
        store.add_snapshots("pp-1", [PortfolioSnapshot(t0 + timedelta(hours=i), 7000.0, 0.0, 7000.0, -3000.0, 0.0,
                                                       50.0, 0) for i in range(3)])
        store.save_state("pp-1", {"breakers": {"peak_equity": 10_000.0, "halted_reason": "max drawdown 30% >= 25%",
                                               "daily_blocked_day": None, "cooldown_until": {}}})
    at = render(monkeypatch, db)
    assert next(m for m in at.metric if m.label == "Breakers").value == "KILL SWITCH"
    assert any("Kill switch: max drawdown" in e.value for e in at.error)


def test_empty_and_missing_databases(monkeypatch, tmp_path):
    SQLiteStore(tmp_path / "empty.db").close()
    at = render(monkeypatch, tmp_path / "empty.db")
    assert any("No runs stored yet" in i.value for i in at.info)
    at = render(monkeypatch, tmp_path / "missing.db")
    assert any("No database" in e.value for e in at.error)
    assert not (tmp_path / "missing.db").exists()


def test_dashboard_command_launches_streamlit_read_only(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr("subprocess.call", lambda command, env: calls.append((command, env)) or 0)
    assert main(["--db", str(tmp_path / "h.db"), "dashboard"]) == 0
    command, env = calls[0]
    assert command[1:4] == ["-m", "streamlit", "run"] and command[4] == APP
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert env["TRADING_LAB_DB"] == str((tmp_path / "h.db").resolve())
    assert "read-only" in capsys.readouterr().out


def test_text_metrics_render(monkeypatch, tmp_path):
    from trading_lab.dashboard.data import DashboardData
    from trading_lab.demo import build_demo

    # This short paper run has no losing trade, so its profit factor is the text "inf".
    result = build_demo(tmp_path / "d", days=5, paper_bars=12, seed=8,
                        now=datetime(2025, 3, 10, 14, 37, tzinfo=UTC))
    with DashboardData(str(result.db_path)) as data:
        assert data.research(result.paper_run)["metrics"]["profit_factor"] == "inf"
    at = render(monkeypatch, result.db_path)  # the paper run is the latest, shown by default
    assert "inf" in [m.value for m in at.metric]
