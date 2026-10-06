"""Stage 14A: reconcile a live paper run with a backtest of the same bars."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.research import format_reconciliation, reconcile
from trading_lab.storage import SQLiteStore

from test_specialists import RoleTransport, agents_config, provider as qwen

UTC = timezone.utc
H = timedelta(hours=1)
ANCHOR = datetime(2024, 1, 1, tzinfo=UTC)
START = datetime(2024, 3, 1, tzinfo=UTC)
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def run_paper(store, cycles, cfg=CFG, seed=11, anchor=ANCHOR, llm_provider=None):
    clock = Clock(START + H + timedelta(minutes=1))
    trader = LivePaperTrader(cfg, SyntheticProvider(seed=seed, anchor=anchor, clock=clock), store,
                             clock=clock, llm_provider=llm_provider)
    for _ in range(cycles):
        trader.run_cycle()
        clock.now += H
    return trader


@pytest.fixture
def store():
    with SQLiteStore(":memory:") as s:
        yield s


def test_a_live_run_matches_its_backtest(store):
    trader = run_paper(store, 120)
    result = reconcile(store, trader.run_id, SyntheticProvider(seed=11, anchor=ANCHOR))
    assert result.ok, format_reconciliation(result)
    assert result.fills_matched > 5 and result.decisions_matched > 5
    assert result.bars_compared == 2 * 120 and result.bars_revised == result.bars_missing == 0
    assert result.start == START and result.end == START + 120 * H
    assert result.replay_misses == 0
    assert format_reconciliation(result).endswith("OK: the live run matches its backtest")


def test_fills_after_the_last_processed_bar_are_left_out(store):
    trader = run_paper(store, 120)
    end = START + 120 * H
    live_fills = [f for f in trader.portfolio.fills]
    result = reconcile(store, trader.run_id, SyntheticProvider(seed=11, anchor=ANCHOR))
    assert result.fills_matched == sum(1 for f in live_fills if f.timestamp < end)


def test_different_market_data_is_reported(store):
    trader = run_paper(store, 120)
    result = reconcile(store, trader.run_id, SyntheticProvider(seed=12, anchor=ANCHOR))
    assert not result.ok
    assert result.bars_revised == result.bars_compared and result.max_price_diff > 0
    assert result.fills_live_only or result.fills_backtest_only
    text = format_reconciliation(result)
    assert "revised" in text and text.splitlines()[-1].startswith("DIFFERENT")


def test_a_tampered_fill_is_reported(tmp_path):
    db = tmp_path / "run.db"
    with SQLiteStore(db) as s:
        run_id = run_paper(s, 120).run_id
    with SQLiteStore(db) as s:
        s._conn.execute("UPDATE fills SET fill_price = fill_price * 1.01 "
                        "WHERE rowid = (SELECT MIN(rowid) FROM fills WHERE run_id = ?)", (run_id,))
        s._conn.commit()
        result = reconcile(s, run_id, SyntheticProvider(seed=11, anchor=ANCHOR))
    assert not result.ok and len(result.fills_live_only) == 1 and len(result.fills_backtest_only) == 1
    assert result.bars_revised == 0 and result.decisions_live_only == result.decisions_backtest_only == []


def test_only_paper_runs_and_known_ids(store):
    bt = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR), store=store).run(START, START + 48 * H)
    with pytest.raises(ValueError, match="only paper runs"):
        reconcile(store, bt.run_id, SyntheticProvider(seed=11, anchor=ANCHOR))
    with pytest.raises(ValueError, match="unknown run"):
        reconcile(store, "nope", SyntheticProvider(seed=11, anchor=ANCHOR))


def test_a_run_without_bars(store):
    trader = run_paper(store, 0)
    result = reconcile(store, trader.run_id, SyntheticProvider(seed=11, anchor=ANCHOR))
    assert result.start is None and "no stored bars" in format_reconciliation(result)


def test_agent_answers_are_replayed_without_calling_the_model(tmp_path):
    cfg = agents_config(tmp_path)
    with SQLiteStore(":memory:") as s:
        recording = RoleTransport()
        trader = run_paper(s, 72, cfg=cfg, llm_provider=qwen(recording))
        assert recording.requests
        replaying = RoleTransport()
        result = reconcile(s, trader.run_id, SyntheticProvider(seed=11, anchor=ANCHOR),
                           llm_provider=qwen(replaying))
        assert replaying.requests == []
        assert result.ok, format_reconciliation(result)
        assert result.replay_misses == 0 and result.decisions_matched > 0

        (tmp_path / "agents.db").unlink()  # the recorded answers are gone
        result = reconcile(s, trader.run_id, SyntheticProvider(seed=11, anchor=ANCHOR),
                           llm_provider=qwen(replaying))
        assert replaying.requests == []
        assert result.replay_misses > 0
        assert "missing from the cache" in format_reconciliation(result)


def test_end_liquidation_is_never_part_of_the_comparison(store):
    cfg = CFG.with_overrides({"backtest": {"liquidate_at_end": True}})
    trader = run_paper(store, 120, cfg=cfg)
    result = reconcile(store, trader.run_id, SyntheticProvider(seed=11, anchor=ANCHOR))
    assert result.ok, format_reconciliation(result)


def test_cli_reconcile_reads_without_writing(tmp_path, capsys):
    db = tmp_path / "paper.db"
    with SQLiteStore(db) as s:
        clock = Clock(datetime(2024, 3, 1, 1, 1, tzinfo=UTC))
        trader = LivePaperTrader(CFG, SyntheticProvider(seed=4, clock=clock), s, clock=clock)
        for _ in range(60):
            trader.run_cycle()
            clock.now += H
        run_id = trader.run_id
    digest = hashlib.sha256(db.read_bytes()).hexdigest()

    assert main(["--db", str(db), "reconcile", run_id]) == 0  # seed taken from the run's "synthetic-4"
    assert "OK: the live run matches its backtest" in capsys.readouterr().out
    assert main(["--db", str(db), "reconcile", run_id, "--synthetic", "5"]) == 1
    assert "DIFFERENT" in capsys.readouterr().out
    assert main(["--db", str(db), "reconcile", "missing"]) == 1
    assert main(["--db", str(tmp_path / "none.db"), "reconcile", run_id]) == 1
    assert hashlib.sha256(db.read_bytes()).hexdigest() == digest
    assert not (tmp_path / "none.db").exists()
