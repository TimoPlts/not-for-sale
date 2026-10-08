"""Stage 29A: the desk funnel (leads with IDs, from first vote to closed trade)."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from test_alerts import SECRET_URL, Capture
from test_live import ANCHOR, START as LIVE_START, Clock
from trading_lab.alerts import URL_ENV
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.presets import PRESETS
from trading_lab.research.desk import STAGES, desk_funnel, format_funnel
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]}})
DEPLOY = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
CONFIGS = {
    "plain": BASE,
    "filtered": BASE.with_overrides({"risk": {"trend_filter_period": 100, "max_open_positions": 1}}),
    "breaker": BASE.with_overrides({"risk": {"max_drawdown_pct": 0.01}}),
    "shorts": PRESETS["trend-shorts"].config(BASE),
    "limit": BASE.with_overrides({"execution": {"entry_order_type": "limit"}}),
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    db = tmp_path_factory.mktemp("desk") / "h.db"
    out = {}
    with SQLiteStore(db) as store:
        for name, cfg in CONFIGS.items():
            out[name] = BacktestEngine(cfg, SyntheticProvider(seed=4), store=store).run(
                START, START + timedelta(days=40))
    return db, out


@pytest.mark.parametrize("name", list(CONFIGS))
def test_the_funnel_adds_up(runs, name):
    db, results = runs
    result = results[name]
    with SQLiteStore(db, readonly=True) as store:
        f = desk_funnel(store, result.run_id)
    counts = [f.count(stage) for stage in STAGES]
    assert counts == sorted(counts, reverse=True)  # every stage is a subset of the one before
    entries = [d for d in result.decisions if d.action == "enter"]
    assert f.count("executed") == len(entries)
    closed = f.closed
    assert len(closed) == len(result.trades) and f.open + len(closed) == len(entries)
    assert sum(lead.pnl for lead in closed) == pytest.approx(sum(t.pnl for t in result.trades))
    assert f.count("confirmed") - f.count("cleared") <= sum(f.killed.values())
    assert [lead.lead_id for lead in f.leads] == [f"L-{i:04d}" for i in range(1, len(f.leads) + 1)]
    assert f.scanned == sum(1 for d in result.decisions if d.action in
                            ("hold", "ignored", "enter_signal", "exit_signal"))
    for lead in closed:  # each closed lead's thread ends in a real trade
        assert any(t.symbol == lead.symbol and t.opened_at == lead.filled_at and t.pnl == pytest.approx(lead.pnl)
                   for t in result.trades)
    json.dumps(f.to_dict())


def test_why_setups_died(runs):
    db, results = runs
    with SQLiteStore(db, readonly=True) as store:
        filtered = desk_funnel(store, results["filtered"].run_id)
        breaker = desk_funnel(store, results["breaker"].run_id)
        shorts = desk_funnel(store, results["shorts"].run_id)
    assert set(filtered.killed) == {"filter: trend filter", "risk: max open positions reached"}
    assert set(breaker.killed) == {"circuit breaker: max drawdown"}  # the kill switch stops every entry
    with SQLiteStore(db, readonly=True) as store:
        plain = desk_funnel(store, results["plain"].run_id).killed
        limit = desk_funnel(store, results["limit"].run_id).killed
    assert set(plain) == {"risk: stop-loss cooldown"} and plain["risk: stop-loss cooldown"] > 1  # dates removed
    assert any(k.startswith("expired: ") for k in limit) and all(not any(c.isdigit() for c in k) for k in limit)
    assert {lead.side for lead in shorts.leads if lead.stage != "leads"} == {"long", "short"}
    text = format_funnel(filtered)
    assert "Why confirmed setups went no further:" in text and "filter: trend filter" in text


def test_window(runs):
    db, results = runs
    run_id = results["plain"].run_id
    with SQLiteStore(db, readonly=True) as store:
        whole = desk_funnel(store, run_id)
        last = desk_funnel(store, run_id, hours=72)
    assert last.start == last.end - timedelta(hours=72)
    assert all(lead.timestamp >= last.start for lead in last.leads)
    assert 0 < last.count("leads") < whole.count("leads") and last.scanned < whole.scanned
    assert {lead.lead_id for lead in last.leads} <= {lead.lead_id for lead in whole.leads}  # same IDs


def test_paper_runs_cli_and_alert(tmp_path, monkeypatch, capsys):
    db = tmp_path / "p.db"
    clock = Clock(LIVE_START + timedelta(hours=1, minutes=1))
    with SQLiteStore(db) as store:
        traders = [LivePaperTrader(BASE, SyntheticProvider(seed=4, anchor=ANCHOR, clock=clock), store, clock=clock,
                                   run_id=name) for name in ("alpha", "beta")]
        for _ in range(24 * 4):
            for trader in traders:
                trader.run_cycle()
            clock.now += timedelta(hours=1)
    assert main(["--db", str(db), "desk", "alpha", "--hours", "48"]) == 0
    assert "Desk funnel for alpha" in capsys.readouterr().out
    assert main(["--db", str(db), "desk", "--all", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [d["run_id"] for d in data] == ["alpha", "beta"] and set(data[0]["funnel"]) == set(STAGES)
    transport = Capture()
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
    monkeypatch.setenv(URL_ENV, SECRET_URL)
    cfg = tmp_path / "c.toml"
    cfg.write_text('[alerts]\nenabled = true\nformat = "json"\nmin_level = "critical"\n')
    assert main(["--config", str(cfg), "--db", str(db), "desk", "--all", "--hours", "24", "--alert"]) == 0
    sent = json.loads(transport.requests[0]["body"])
    assert sent["title"] == "Desk report" and "Desk funnel for alpha" in sent["body"]
    assert "Desk funnel for beta" in sent["body"] and SECRET_URL not in capsys.readouterr().out
    assert main(["--db", str(db), "desk", "alpha", "--all"]) == 1
    assert main(["--db", str(db), "desk", "nope"]) == 1


def test_systemd_units():
    service = (DEPLOY / "trading-lab-desk.service").read_text()
    assert "desk --all --hours 24 --alert" in service and "ReadOnlyPaths=/opt/trading-lab/trading-lab" in service
    assert "Type=oneshot" in service and "QWEN_API_KEY" not in service
    timer = (DEPLOY / "trading-lab-desk.timer").read_text()
    assert "OnCalendar=*-*-* 21:00:00" in timer and "Persistent=true" in timer


def test_seat_verdicts_by_hand():
    from trading_lab.research.desk import seat_verdict

    assert seat_verdict(10, 0.9, 5, 100.0, -50.0) == "too early"
    assert seat_verdict(40, 0.55, 5, 100.0, -50.0) == "earning its seat"
    assert seat_verdict(40, 0.45, 5, 100.0, -50.0) == "not earning its seat"
    assert seat_verdict(40, 0.50, 6, -80.0, -10.0) == "not earning its seat"  # the trades it backed lost more
    assert seat_verdict(40, 0.50, 6, 20.0, 10.0) == "unclear"
    assert seat_verdict(40, None, 0, 0.0, 0.0) == "too early"


def test_seat_review_and_cli(runs, capsys):
    from trading_lab.research.attribution import attribute_run
    from trading_lab.research.desk import seat_review

    db, results = runs
    run_id = results["plain"].run_id
    with SQLiteStore(db, readonly=True) as store:
        seats = seat_review(store, run_id)
        attribution = attribute_run(store, run_id)
    assert {s.strategy for s in seats} == {name for name, a in attribution.items() if a.votes}
    for s in seats:
        a = attribution[s.strategy]
        assert (s.calls, s.agreed, s.pivotal) == (a.measured, a.trades_agreed, a.trades_pivotal)
        assert s.pnl_agreed == pytest.approx(a.pnl_agreed)
    assert [s.pnl_agreed for s in seats] == sorted((s.pnl_agreed for s in seats), reverse=True)
    assert main(["--db", str(db), "desk", run_id, "--seats"]) == 0
    out = capsys.readouterr().out
    assert f"Seats of {run_id}" in out and all(s.strategy in out for s in seats)
    assert main(["--db", str(db), "desk", run_id, "--seats", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {s["strategy"] for s in data["seats"]} == {s.strategy for s in seats}
