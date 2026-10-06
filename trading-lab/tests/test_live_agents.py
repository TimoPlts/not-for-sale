"""Stage 9G: the three Qwen agents in live paper trading (fake clock, synthetic data, mocked Qwen)."""

import re
from datetime import datetime, timedelta

import pytest

from test_live import ANCHOR, START, Clock, fill_key
from test_specialists import RoleTransport
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.llm import QwenProvider, TransportTimeout
from trading_lab.storage import SQLiteStore

H = timedelta(hours=1)
ENV = {"QWEN_API_URL": "https://qwen.example.test/v1", "QWEN_API_KEY": "sk-test-key-123456", "QWEN_MODEL": "qwen-test"}
AGENTS = ("qwen_trend", "qwen_momentum", "qwen_risk")
INTERVAL = 3


def config(cache, mode="record", **extra):
    return AppConfig().with_overrides({
        "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
        "strategies": {name: {"weight": 1.0, "lookback": 20, "decision_interval": INTERVAL} for name in AGENTS},
        "agents": {"mode": mode, "cache_path": str(cache)},
        **extra,
    })


def qwen(transport, **kwargs):
    return QwenProvider(env=ENV, transport=transport, sleep=lambda s: None, **kwargs)


def market(clock):
    return SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock)


def trader(cfg, store, clock, transport, **kwargs):
    return LivePaperTrader(cfg, market(clock), store, clock=clock, llm_provider=qwen(transport), **kwargs)


def drive(t, clock, cycles):
    for _ in range(cycles):
        report = t.run_cycle()
        assert report.error is None
        clock.now += H


def decision_times(transport):
    pattern = re.compile(r"candle opened (\S+) UTC")
    return [datetime.fromisoformat(pattern.search(user).group(1) + "+00:00") for _, user in transport.requests]


def stored_decisions(store, run_id):
    frame = store.load_decisions(run_id).drop(columns=["id", "run_id"])
    frame = frame.astype(object).where(frame.notna(), None)  # NaN != NaN would hide equal rows
    return [tuple(r) for r in frame.itertuples(index=False)]


def test_live_agents_follow_the_backtest_and_only_see_new_bars(tmp_path):
    bars = 30
    clock = Clock(START + H + timedelta(minutes=1))
    transport = RoleTransport()
    with SQLiteStore(":memory:") as store:
        live = trader(config(tmp_path / "live.db"), store, clock, transport)
        drive(live, clock, bars)
        live_fills = [fill_key(f)[1:] for f in live.portfolio.fills if f.timestamp < START + bars * H]
        signals = store.load_signals(live.run_id)

    backtest = BacktestEngine(config(tmp_path / "bt.db"), SyntheticProvider(seed=11, anchor=ANCHOR),
                              llm_provider=qwen(RoleTransport())).run(START, START + bars * H)
    assert live_fills == [fill_key(f)[1:] for f in backtest.fills]

    times = decision_times(transport)
    assert min(times) >= START  # nothing about the history before the first traded bar
    assert all(int(t.timestamp()) // 3600 % INTERVAL == 0 for t in times)  # decision_interval respected
    per_role = len(times) // 3
    assert per_role == 2 * len([k for k in range(bars) if int((START + k * H).timestamp()) // 3600 % INTERVAL == 0])
    agent_rows = signals[signals["strategy"].isin(AGENTS)]
    assert agent_rows["metadata_json"].str.contains('"rationale"').sum() == len(times)
    assert agent_rows["metadata_json"].str.contains('"regime"').any()


@pytest.mark.parametrize("extra", [{}, {"execution": {"entry_order_type": "limit", "limit_ttl_bars": 2}}])
def test_stop_and_resume_matches_an_uninterrupted_run(tmp_path, extra):
    total, first = 40, 18

    clock = Clock(START + H + timedelta(minutes=1))
    full_transport = RoleTransport()
    with SQLiteStore(tmp_path / "full.db") as store:
        full = trader(config(tmp_path / "full_cache.db", **extra), store, clock, full_transport)
        drive(full, clock, total)
        full_fills = [fill_key(f) for f in full.portfolio.fills]
        full_decisions = stored_decisions(store, full.run_id)
        full_state = store.load_state(full.run_id)

    clock = Clock(START + H + timedelta(minutes=1))
    first_transport, second_transport = RoleTransport(), RoleTransport()
    with SQLiteStore(tmp_path / "split.db") as store:
        part = trader(config(tmp_path / "split_cache.db", **extra), store, clock, first_transport)
        drive(part, clock, first)
        part.stop()
        resumed = LivePaperTrader.resume(store, part.run_id, market(clock), clock=clock,
                                         llm_provider=qwen(second_transport))
        drive(resumed, clock, total - first)
        split_fills = [fill_key(f) for f in resumed.portfolio.fills]
        split_decisions = stored_decisions(store, part.run_id)
        split_state = store.load_state(part.run_id)

    assert split_fills == full_fills and len(full_fills) > 0
    assert split_decisions == full_decisions  # same logical decisions, no duplicate paper orders
    assert split_state["breakers"] == full_state["breakers"] and split_state["resting"] == full_state["resting"]
    assert split_state["stop_events"] == full_state["stop_events"]
    calls = first_transport.requests + second_transport.requests
    assert len(calls) == len(full_transport.requests)  # nothing was asked twice
    assert len({user for _, user in calls}) == len(calls)


def test_restart_after_a_crash_reuses_cached_answers(tmp_path, monkeypatch):
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(tmp_path / "h.db") as store:
        cfg = config(tmp_path / "cache.db")
        first = trader(cfg, store, clock, RoleTransport())
        drive(first, clock, 12)
        fills_before = store.count("fills", first.run_id)

        def crash(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(store, "add_signals", crash)
        with pytest.raises(RuntimeError):  # agents answered, then persisting the cycle failed
            first.run_cycle()
        monkeypatch.undo()
        assert store.count("fills", first.run_id) == fills_before  # the failed cycle left nothing behind

        transport = RoleTransport()
        restarted = LivePaperTrader.resume(store, first.run_id, market(clock), clock=clock,
                                           llm_provider=qwen(transport))
        report = restarted.run_cycle()
        assert report.error is None and report.new_bars == 1
        assert transport.requests == []  # the re-processed bar's answers come from the cache
        again = restarted.run_cycle()
        assert again.new_bars == 0 and not again.fills  # no duplicate paper orders

    clock2 = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as reference_store:
        reference = trader(config(tmp_path / "ref_cache.db"), reference_store, clock2, RoleTransport())
        drive(reference, clock2, 13)
        assert [fill_key(f) for f in restarted.portfolio.fills] == [fill_key(f) for f in reference.portfolio.fills]


def test_model_failures_never_stop_live_trading(tmp_path):
    clock = Clock(START + H + timedelta(minutes=1))
    failing = RoleTransport(override=lambda role, user: (_ for _ in ()).throw(TransportTimeout("down")))
    with SQLiteStore(":memory:") as store:
        t = LivePaperTrader(config(tmp_path / "c.db"), market(clock), store, clock=clock,
                            llm_provider=qwen(failing, max_retries=0))
        drive(t, clock, 10)
        rows = store.load_signals(t.run_id)
    agent_rows = rows[rows["strategy"].isin(AGENTS) & rows["metadata_json"].str.contains('"called"')]
    assert len(agent_rows) > 0 and set(agent_rows["direction"]) == {"hold"}
    assert agent_rows["metadata_json"].str.contains("ProviderTimeoutError").all()


def test_live_mode_asks_only_about_new_bars_across_resume(tmp_path):
    clock = Clock(START + H + timedelta(minutes=1))
    first, second = RoleTransport(), RoleTransport()
    with SQLiteStore(tmp_path / "h.db") as store:
        cfg = config(tmp_path / "unused.db", mode="live")
        t = trader(cfg, store, clock, first)
        drive(t, clock, 9)
        t.stop()
        resumed = LivePaperTrader.resume(store, t.run_id, market(clock), clock=clock, llm_provider=qwen(second))
        resumed.run_cycle()  # one new bar
        resumed.run_cycle()  # no new bar: nothing to ask
    assert len(first.requests) == 3 * 2 * 3  # 9 bars, interval 3 -> 3 decision bars, 2 symbols, 3 agents
    new_times = set(decision_times(second))
    assert new_times.isdisjoint(decision_times(first))
