"""Stage 28A: sizing to a drawdown budget (trading-lab size)."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import _fraction, main
from trading_lab.config import AppConfig
from trading_lab.core.models import ClosedTrade, PortfolioSnapshot
from trading_lab.data import SyntheticProvider
from trading_lab.presets import PRESETS
from trading_lab.research.sizing import format_sizing, headroom, size_factor, size_for_drawdown
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    db = tmp_path_factory.mktemp("size") / "h.db"
    with SQLiteStore(db) as store:
        result = BacktestEngine(PRESETS["trend"].config(BASE), SyntheticProvider(seed=2), store=store).run(
            START, START + timedelta(days=60))
    return db, result


def test_size_factor_by_hand():
    risk_bound = {"risk_per_trade": 10.0, "max_position_size": 15.0, "available_cash": 100.0,
                  "binding_limit": "risk_per_trade", "equity": 1e4, "stop_distance_pct": 0.05}
    assert size_factor(risk_bound, 0.5) == pytest.approx(0.5)
    assert size_factor(risk_bound, 1.2) == pytest.approx(1.2)
    assert size_factor(risk_bound, 2.0) == pytest.approx(1.5)  # the max position takes over
    assert headroom(risk_bound) == pytest.approx(1.5)
    capped = {"risk_per_trade": 20.0, "max_position_size": 15.0}
    assert size_factor(capped, 0.5) == pytest.approx(10 / 15)  # only below 0.75x does risk per trade bind
    assert size_factor(capped, 2.0) == pytest.approx(1.0)
    assert size_factor(None, 1.7) == pytest.approx(1.7)  # no stored sizing: linear


def test_size_for_drawdown(run):
    db, result = run
    with SQLiteStore(db, readonly=True) as store:
        r = size_for_drawdown(store, result.run_id, 0.04, samples=500)  # 1x has a bad case near 5.5%
        rows = {round(row.scale, 6): row for row in r.rows}
        assert r.scale is not None and round(r.scale, 6) in rows
        assert rows[round(r.scale, 6)].outlook.drawdown_bad == pytest.approx(0.04, abs=0.001)
        bads = [row.outlook.drawdown_bad for row in r.rows]
        assert bads == sorted(bads)  # a bigger size never means a smaller bad-case drawdown
        assert r.suggested_risk_pct == pytest.approx(r.current_risk_pct * r.scale)
        assert sum(r.binding.values()) == sum(1 for d in result.decisions if d.action == "enter")
        assert r.unmatched == 0 and r.headroom > 1
        one = rows[1.0].outlook
        assert one.trades == len(result.trades)
        json.dumps(r.to_dict())
        text = format_sizing(r)
        assert "<- target" in text and "Suggested: risk_per_trade_pct = " in text and "[risk]" in text
        far = size_for_drawdown(store, result.run_id, 0.9, samples=300)
        assert far.scale is None and any("stays below" in w for w in far.warnings)
        tiny = size_for_drawdown(store, result.run_id, 0.0001, samples=300)
        assert tiny.scale is None and any("above" in w for w in tiny.warnings)
        with pytest.raises(ValueError):
            size_for_drawdown(store, result.run_id, 1.5)


def test_runs_without_stored_sizing(tmp_path):
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    with SQLiteStore(tmp_path / "h.db") as store:
        store.create_run("old", kind="backtest", timeframe="1h", symbols=["BTC/USDT"], exchange="test",
                         config=BASE.to_mapping(), config_fingerprint="f")
        pnls = [50.0, -30.0, 20.0, -40.0, 60.0, -10.0] * 6
        equity, snaps, trades = 10_000.0, [], []
        for i, pnl in enumerate(pnls):
            ts = t0 + timedelta(hours=2 * i)
            snaps.append(PortfolioSnapshot(ts, equity, 0.0, equity, 0.0, 0.0, 0.0, 0))
            trades.append(ClosedTrade("BTC/USDT", 1.0, 100.0, 100.0, 100.0, 100.0, pnl, ts, ts + timedelta(hours=1)))
            equity += pnl
        store.add_snapshots("old", snaps)
        store.add_closed_trades("old", trades)
        r = size_for_drawdown(store, "old", 0.05, samples=300)
    assert r.unmatched == len(pnls) and r.binding == {}
    assert any("scaled linearly" in w for w in r.warnings) and r.headroom is None


def test_fraction_parsing():
    assert _fraction("0.2") == _fraction("20%") == _fraction("20") == pytest.approx(0.2)
    assert _fraction("12.5%") == pytest.approx(0.125) and _fraction("150%") == pytest.approx(1.5)


def test_cli(run, tmp_path, capsys):
    db, result = run
    assert main(["--db", str(db), "size", result.run_id, "--max-drawdown", "4%", "--samples", "300"]) == 0
    out = capsys.readouterr().out
    assert "bad-case (1 in 20) drawdown of 4%" in out and "<- target" in out and "size set by: risk_per_trade" in out
    assert main(["--db", str(db), "size", "--json", "--samples", "300"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == result.run_id and data["target"] == pytest.approx(0.2)
    assert main(["--db", str(db), "size", "nope"]) == 1
    assert main(["--db", str(db), "size", "--max-drawdown", "150%"]) == 1
    with SQLiteStore(tmp_path / "q.db") as store:
        quiet = BacktestEngine(BASE.with_overrides({"voting": {"min_agreeing": 3}}), SyntheticProvider(seed=2),
                               store=store).run(START, START + timedelta(days=1))
    assert main(["--db", str(tmp_path / "q.db"), "size"]) == 0
    assert f"Run {quiet.run_id}: no closed trades, so nothing to size." in capsys.readouterr().out


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """A synthetic backtest stored through the CLI, so --verify can rebuild its data source."""
    tmp = tmp_path_factory.mktemp("verify")
    cfg = tmp / "trend.toml"
    from trading_lab.presets import preset_toml

    cfg.write_text(preset_toml("trend") + '\n[market]\nsymbols = ["BTC/USDT", "ETH/USDT"]\n')
    assert main(["--config", str(cfg), "--db", str(tmp / "h.db"), "backtest", "--synthetic", "2",
                 "--start", "2024-02-01", "--end", "2024-04-01"]) == 0
    return tmp, cfg


def test_verify(stored, capsys):
    tmp, cfg = stored
    capsys.readouterr()
    args = ["--config", str(cfg), "--db", str(tmp / "h.db"), "size", "--max-drawdown", "4%", "--samples", "300"]
    assert main([*args, "--verify", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    v = data["verification"]
    assert v["risk_per_trade_pct"] == pytest.approx(round(data["suggested_risk_pct"], 6))
    # the verification is an ordinary backtest of the stored config at that risk
    with SQLiteStore(tmp / "h.db", readonly=True) as store:
        run = store.get_run(store.list_runs(1)[0]["run_id"])
    config = AppConfig.from_dict(run["config"]).with_overrides({"risk": {"risk_per_trade_pct": v["risk_per_trade_pct"]}})
    again = BacktestEngine(config, SyntheticProvider(seed=2)).run(datetime.fromisoformat(run["period_start"]),
                                                                    datetime.fromisoformat(run["period_end"]))
    assert v["max_drawdown"] == pytest.approx(again.metrics.max_drawdown)
    assert v["trades"] == again.metrics.num_trades and v["within_budget"] == (v["max_drawdown"] <= 0.04)
    assert main([*args, "--verify"]) == 0
    out = capsys.readouterr().out
    assert "Verification backtest at" in out and "One period is one path" in out
    assert main([*args[:-4], "--max-drawdown", "90%", "--samples", "300", "--verify"]) == 0
    assert "(--verify: no suggestion to verify)" in capsys.readouterr().out


def test_verify_refuses_paper_runs_and_agents(tmp_path, capsys):
    from test_live import ANCHOR, Clock
    from trading_lab.live import LivePaperTrader

    clock = Clock(START + timedelta(hours=1, minutes=1))
    with SQLiteStore(tmp_path / "p.db") as store:
        trader = LivePaperTrader(BASE, SyntheticProvider(seed=2, anchor=ANCHOR, clock=clock), store, clock=clock,
                                 run_id="paper")
        for _ in range(24):  # refused before any work, trades or not
            trader.run_cycle()
            clock.now += timedelta(hours=1)
    assert main(["--db", str(tmp_path / "p.db"), "size", "paper", "--max-drawdown", "1%", "--samples", "300",
                 "--verify"]) == 1
    assert "paper is a paper run" in capsys.readouterr().err

    agents = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]},
                                     "strategies": {"rsi": {}, "qwen_trend": {"weight": 1.0}}})
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    with SQLiteStore(tmp_path / "a.db") as store:
        store.create_run("ai", kind="backtest", timeframe="1h", symbols=["BTC/USDT"], exchange="synthetic-1",
                         config=agents.to_mapping(), config_fingerprint="f", period_start=t0,
                         period_end=t0 + timedelta(days=3))
        store.add_snapshots("ai", [PortfolioSnapshot(t0 + timedelta(hours=i), 10_000 + 10 * i, 0, 10_000 + 10 * i,
                                                     0, 0, 0, 0) for i in range(10)])
        store.add_closed_trades("ai", [ClosedTrade("BTC/USDT", 1.0, 100.0, 100.0, 100.0, 100.0, p,
                                                   t0 + timedelta(hours=i), t0 + timedelta(hours=i + 1))
                                       for i, p in enumerate([30.0, -20.0, 25.0, -40.0, 10.0, -15.0, 20.0, -5.0])])
    assert main(["--db", str(tmp_path / "a.db"), "size", "ai", "--max-drawdown", "0.5%", "--samples", "300",
                 "--verify"]) == 1
    assert "AI agents" in capsys.readouterr().err
