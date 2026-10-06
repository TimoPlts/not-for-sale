"""Stage 10D: failure recovery for overnight operation."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from test_live import ANCHOR, START, Clock, fill_key
from test_specialists import ENV, RoleTransport
from trading_lab.agents import MemoryResponseCache, QwenTrendStrategy
from trading_lab.config import AgentsConfig, AppConfig
from trading_lab.core.errors import ConfigError, DataError
from trading_lab.core.models import Direction
from trading_lab.dashboard import DashboardData
from trading_lab.data import SyntheticProvider
from trading_lab.live import LivePaperTrader
from trading_lab.live.paper_trader import MAX_ERROR_BACKOFF_SECONDS
from trading_lab.llm import (
    ProviderTimeoutError,
    ProviderUnavailableError,
    QwenProvider,
    TransportTimeout,
)
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)


class Seconds:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Switchable(RoleTransport):
    """A fake Qwen endpoint that can be taken down and brought back."""

    def __init__(self):
        super().__init__()
        self.down = False

    def post(self, url, headers, body, timeout):
        if self.down:
            self.requests.append(("down", ""))
            raise TransportTimeout("endpoint down")
        return super().post(url, headers, body, timeout)


def qwen(transport, clock=None, **kwargs):
    kwargs.setdefault("max_retries", 0)
    return QwenProvider(env=ENV, transport=transport, sleep=lambda s: None, clock=clock or Seconds(), **kwargs)


# ------------------------------------------------------------ model breaker
def test_repeated_failures_pause_calls_then_probe():
    clock, transport = Seconds(), Switchable()
    provider = qwen(transport, clock, failure_threshold=3, failure_cooldown_seconds=60)
    transport.down = True
    for _ in range(3):
        with pytest.raises(ProviderTimeoutError):
            provider.chat("You are the Trend Agent.", "x")
    with pytest.raises(ProviderUnavailableError, match="3 failed calls in a row") as exc:
        provider.chat("You are the Trend Agent.", "x")
    assert exc.value.attempts == 0 and len(transport.requests) == 3  # nothing sent while paused
    assert 59 <= provider.paused_for <= 60

    clock.t += 61  # cooldown over: one probe; it fails, so calls pause again
    with pytest.raises(ProviderTimeoutError):
        provider.chat("You are the Trend Agent.", "x")
    assert len(transport.requests) == 4 and provider.paused_for > 0

    clock.t += 61
    transport.down = False  # the endpoint is back
    provider.chat("You are the Trend Agent.", "close_vs_ema_50_pct: 1")
    assert provider.consecutive_failures == 0 and provider.paused_for == 0


def test_breaker_can_be_disabled_and_is_configurable():
    transport = Switchable()
    transport.down = True
    provider = qwen(transport, failure_threshold=0)
    for _ in range(8):
        with pytest.raises(ProviderTimeoutError):
            provider.chat("s", "u")
    assert len(transport.requests) == 8
    assert AgentsConfig().failure_threshold == 5 and AgentsConfig().failure_cooldown_seconds == 300
    with pytest.raises(ConfigError):
        AgentsConfig(failure_threshold=-1)
    with pytest.raises(ConfigError):
        AgentsConfig(failure_cooldown_seconds=-5)


def test_paused_agents_vote_hold_at_once_and_count_as_skipped():
    candles = SyntheticProvider(seed=5).fetch_ohlcv("BTC/USDT", "1h", datetime(2023, 12, 1, tzinfo=UTC),
                                                     datetime(2024, 2, 1, tzinfo=UTC))
    transport = Switchable()
    transport.down = True
    provider = qwen(transport, failure_threshold=2)
    strategy = QwenTrendStrategy(lookback=10)
    strategy.attach_provider(provider)
    strategy.configure(mode="live", cache=MemoryResponseCache())
    signals = [strategy.generate_signal("BTC/USDT", candles) for _ in range(5)]
    assert all(s.direction is Direction.HOLD for s in signals)
    assert [s.metadata["error"].split(":")[0] for s in signals] == [
        "ProviderTimeoutError", "ProviderTimeoutError", *["ProviderUnavailableError"] * 3]
    usage = provider.usage.per_agent["qwen_trend"]
    assert (usage.calls, usage.failures, usage.skipped) == (2, 2, 3) and len(transport.requests) == 2


# ------------------------------------------------------- market data outage
class FlakyMarket(SyntheticProvider):
    def __init__(self, failures, error=DataError, **kwargs):
        super().__init__(**kwargs)
        self.failures = failures
        self.error = error

    def fetch_ohlcv(self, *args, **kwargs):
        if self.failures > 0:
            self.failures -= 1
            raise self.error("exchange unreachable")
        return super().fetch_ohlcv(*args, **kwargs)


@pytest.mark.parametrize("error", [DataError, ConnectionResetError, TimeoutError])
def test_data_outages_back_off_and_recover(tmp_path, error):
    clock = Clock(START + H + timedelta(minutes=1))
    waits = []
    with SQLiteStore(tmp_path / "h.db") as store:
        cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
        market = FlakyMarket(4, error, seed=1, anchor=ANCHOR, clock=clock)
        trader = LivePaperTrader(cfg, market, store, clock=clock)
        health = []

        def on_cycle(report):
            health.append(store.load_state(trader.run_id)["health"]["consecutive_errors"])

        def sleep(seconds):
            clock.now += timedelta(seconds=seconds)

        trader.run_forever(poll_seconds=30, max_cycles=6, sleep=sleep, on_cycle=on_cycle, on_wait=waits.append)
        assert waits[:4] == [30, 60, 120, 240]  # exponential back-off during the outage
        assert 0 < waits[4] <= 3600 + 5 and waits[4] != 480  # back to "until the next candle", no more back-off
        assert health == [1, 2, 3, 4, 0, 0]
        assert store.count("equity_snapshots", trader.run_id) >= 1
        assert store.get_run(trader.run_id)["status"] == "stopped"


def test_backoff_is_capped(tmp_path):
    clock = Clock(START + H + timedelta(minutes=1))
    waits = []
    with SQLiteStore(":memory:") as store:
        cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
        trader = LivePaperTrader(cfg, FlakyMarket(50, seed=1, anchor=ANCHOR, clock=clock), store, clock=clock)
        trader.run_forever(poll_seconds=30, max_cycles=10, sleep=lambda s: None, on_wait=waits.append)
    assert max(waits) == MAX_ERROR_BACKOFF_SECONDS and waits[-1] == MAX_ERROR_BACKOFF_SECONDS


# ---------------------------------------------------------- database errors
def agent_config(cache):
    return AppConfig().with_overrides({
        "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
        "strategies": {name: {"weight": 1.0, "lookback": 20, "decision_interval": 2}
                       for name in ("qwen_trend", "qwen_momentum", "qwen_risk")},
        "agents": {"cache_path": str(cache)},
    })


def test_a_locked_database_reloads_state_and_retries_the_bars(tmp_path, monkeypatch):
    def run(store, clock, transport, cache, lock_at=None):
        trader = LivePaperTrader(agent_config(cache), SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store,
                                 clock=clock, llm_provider=qwen(transport))
        reports = []
        for n in range(16 + (lock_at is not None)):  # one extra cycle for the retried bar
            if n == lock_at:
                def locked(*args, **kwargs):
                    raise sqlite3.OperationalError("database is locked")
                monkeypatch.setattr(store, "add_decisions", locked)
                reports.append(trader.run_cycle())
                monkeypatch.undo()
                continue  # the clock does not move: the same bar is retried
            reports.append(trader.run_cycle())
            clock.now += H
        return trader, reports

    with SQLiteStore(tmp_path / "ref.db") as ref_store:
        reference, _ = run(ref_store, Clock(START + H + timedelta(minutes=1)), RoleTransport(), tmp_path / "ref_c.db")
        ref_fills = [fill_key(f) for f in reference.portfolio.fills]
        ref_rows = ref_store.count("decisions", reference.run_id)

    transport = RoleTransport()
    with SQLiteStore(tmp_path / "h.db") as store:
        trader, reports = run(store, Clock(START + H + timedelta(minutes=1)), transport, tmp_path / "c.db", lock_at=8)
        failed = reports[8]
        assert failed.error and "database error, state reloaded" in failed.error
        assert reports[9].error is None and reports[9].new_bars == 1  # the same bar, processed again
        assert [fill_key(f) for f in trader.portfolio.fills] == ref_fills  # no lost or duplicate paper orders
        assert store.count("decisions", trader.run_id) == ref_rows
        prompts = [user for _, user in transport.requests]
        assert len(prompts) == len(set(prompts))  # retried bar reused the cached answers


def test_health_is_visible_in_the_dashboard(tmp_path):
    clock = Clock(START + H + timedelta(minutes=1))
    db = tmp_path / "h.db"
    with SQLiteStore(db) as store:
        cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
        trader = LivePaperTrader(cfg, FlakyMarket(0, seed=1, anchor=ANCHOR, clock=clock), store, clock=clock)
        trader.run_cycle()
        trader._provider.failures = 2
        trader.run_cycle()
        trader.run_cycle()
    with DashboardData(db) as data:
        health = data.overview(trader.run_id)["health"]
    assert health["consecutive_errors"] == 2 and "exchange unreachable" in health["last_error"]
    assert health["last_cycle_at"].startswith(clock.now.date().isoformat())


def test_failure_guide_covers_every_case():
    from conftest import PROJECT_ROOT

    guide = (PROJECT_ROOT / "docs" / "FAILURE_RECOVERY.md").read_text()
    for topic in ("timeout", "malformed", "repeated failures", "network outage", "locked", "SIGTERM",
                  "reboot", "stops the process", "HOLD", "failure_threshold", "15 minutes"):
        assert topic in guide, topic
