"""Stage 22B: two paper runs compared over the time they ran together (live-compare)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from test_live import ANCHOR, START, Clock
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.models import ClosedTrade, PortfolioSnapshot
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.presets import PRESETS
from trading_lab.research import live_compare
from trading_lab.research.protocol import sign_test_p
from trading_lab.storage import SQLiteStore

H = timedelta(hours=1)
T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


def add_run(store, run_id, equities, *, start=T0, step=timedelta(hours=12), config=BASE, fees=None,
            positions=None, kind="paper"):
    store.create_run(run_id, kind=kind, timeframe="1h", symbols=list(config.market.symbols), exchange="test",
                     config=config.to_mapping(), config_fingerprint=run_id)
    snaps = [PortfolioSnapshot(start + i * step, e, 0.0, e, 0.0, 0.0, (fees or [0.0] * len(equities))[i],
                               (positions or [0] * len(equities))[i]) for i, e in enumerate(equities)]
    store.add_snapshots(run_id, snaps)


def trade(closed_at, pnl):
    return ClosedTrade("BTC/USDT", 1.0, 100.0, 100.0 + pnl, 100.0, 100.0 + pnl, pnl, closed_at - H, closed_at)


def test_measured_over_the_overlap_only(tmp_path):
    with SQLiteStore(tmp_path / "x.db") as store:
        # A starts a day earlier and gains 10% before B exists: no head start for A.
        add_run(store, "a", [10_500, 11_000, 11_000, 11_110, 11_000], start=T0 - timedelta(days=1),
                fees=[1, 2, 3, 4, 5], positions=[1, 1, 0, 1, 0])
        add_run(store, "b", [10_000, 10_200, 10_000], start=T0, fees=[1, 1, 2])
        store.add_closed_trades("a", [trade(T0 - timedelta(hours=6), 50.0), trade(T0 + H, 20.0),
                                      trade(T0 + 2 * H, -5.0)])
        c = live_compare(store, "a", "b", min_days=1)
    assert c.start == T0 and c.end == T0 + timedelta(days=1)
    # A from its last equity before the overlap (11,000 at T0 - 12h); B from its initial cash.
    assert c.a.total_return == pytest.approx(11_000 / 11_000 - 1)
    assert c.b.total_return == pytest.approx(10_000 / 10_000 - 1)
    assert c.a.max_drawdown == pytest.approx(1 - 11_000 / 11_110)
    assert c.a.num_trades == 2 and c.a.total_fees == pytest.approx(3.0)  # trades and fees inside only
    assert c.b.total_fees == pytest.approx(2.0) and c.a.exposure == pytest.approx(1 / 3)
    # daily: day 1 (T0, T0+12h) and day 2 (T0+24h)
    assert [d.day.isoformat() for d in c.days] == ["2026-03-01", "2026-03-02"]
    assert c.days[0].a == pytest.approx(11_110 / 11_000 - 1) and c.days[0].b == pytest.approx(0.02)
    assert c.days[1].a == pytest.approx(11_000 / 11_110 - 1) and c.days[1].b == pytest.approx(10_000 / 10_200 - 1)
    assert c.a_wins == 1 and c.b_wins == 1 and c.compared == 2


def test_sign_test_and_verdicts(tmp_path):
    days = 20
    a = [10_000 * 1.01 ** (i + 1) for i in range(days)]  # +1% a day
    b = [10_000 * 1.002 ** (i + 1) for i in range(days)]
    with SQLiteStore(tmp_path / "x.db") as store:
        add_run(store, "a", a, step=timedelta(days=1))
        add_run(store, "b", b, step=timedelta(days=1), config=PRESETS["trend"].config(BASE))
        add_run(store, "flat", [10_000] * days, step=timedelta(days=1))
        add_run(store, "flat2", [10_000] * days, step=timedelta(days=1))
        add_run(store, "later", [10_000] * 3, start=T0 + timedelta(days=days + 5))
        c = live_compare(store, "a", "b")
        assert c.a_wins == days and c.b_wins == 0
        assert c.p_a_better == pytest.approx(sign_test_p(days, days)) and c.p_a_better < 0.05
        assert c.verdict.startswith(f"A had the better day {days} of {days}")
        assert ("strategies.donchian.weight", None, 1.5) in c.diff
        assert live_compare(store, "b", "a").verdict.startswith("B had the better day")
        assert live_compare(store, "a", "b", min_days=days + 1).verdict.startswith("too early to tell: 20 day(s)")
        tie = live_compare(store, "flat", "flat2")
        assert tie.compared == 0 and tie.diff == () and tie.verdict.startswith("too early")
        apart = live_compare(store, "a", "later")
        assert apart.start is None and apart.verdict.startswith("not comparable")
        with pytest.raises(ValueError, match="unknown run"):
            live_compare(store, "a", "nope")
        with pytest.raises(ValueError, match="different runs"):
            live_compare(store, "a", "a")


def test_chance_lead_and_notes(tmp_path):
    rng = [0.01, -0.01, 0.01, 0.01, -0.01, 0.01, -0.01, 0.01, 0.01, -0.01, 0.01, -0.01, 0.01, 0.01, -0.01, 0.01]
    a, b, ea = [], [], 10_000.0
    for r in rng:
        ea *= 1 + r
        a.append(ea)
        b.append(10_000.0)
    with SQLiteStore(tmp_path / "x.db") as store:
        add_run(store, "a", a, step=timedelta(days=1))
        add_run(store, "b", b, step=timedelta(days=1), kind="backtest")
        c = live_compare(store, "a", "b")
    assert c.a_wins == 10 and c.b_wins == 6
    assert c.verdict.startswith("A leads on 10 of 16 days, but that could easily be chance")
    assert "different kind: paper vs backtest" in c.notes


def test_real_paper_runs_and_cli(tmp_path, capsys):
    db = tmp_path / "shared.db"
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(db) as store:
        a = LivePaperTrader(BASE, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock,
                            run_id="default")
        for _ in range(24):  # the default run trades a day alone first
            a.run_cycle()
            clock.now += H
        b = LivePaperTrader(PRESETS["trend"].config(BASE), SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock),
                            store, clock=clock, run_id="trend")
        for _ in range(72):
            a.run_cycle()
            b.run_cycle()
            clock.now += H
        c = live_compare(store, "default", "trend")
        curve_b = store.load_equity_curve("trend")
        assert c.start == curve_b.index[0].to_pydatetime()
        assert c.b.total_return == pytest.approx(curve_b["equity"].iloc[-1] / BASE.portfolio.initial_cash - 1)
        assert len(c.days) >= 3
    assert main(["--db", str(db), "live-compare", "default", "trend"]) == 0
    out = capsys.readouterr().out
    assert "A = default (paper, running, 1h)" in out and "Settings that differ (A -> B):" in out
    assert "Verdict: too early to tell" in out
    assert main(["--db", str(db), "live-compare", "default", "trend", "--json", "--min-days", "1"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_b"]["run_id"] == "trend" and len(data["days"]) == len(c.days)
    assert data["b"]["total_return"] == pytest.approx(c.b.total_return)
    assert main(["--db", str(db), "live-compare", "default", "missing"]) == 1
    assert main(["--db", str(tmp_path / "none.db"), "live-compare", "a", "b"]) == 1
