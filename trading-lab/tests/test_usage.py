"""Stage 9E: model usage accounting (calls, cache hits/misses, failures, retries, latency, tokens)."""

import itertools
import json
from datetime import datetime, timezone

import pytest

from test_specialists import ENV, RoleTransport
from trading_lab.agents import MemoryResponseCache, QwenTrendStrategy
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.llm import (
    CallRecord,
    QwenProvider,
    TransportTimeout,
    UsageStats,
    estimate_tokens,
    format_usage,
    usage_from_signals,
)
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 4, tzinfo=UTC)


class ScriptedTransport(RoleTransport):
    """RoleTransport whose first responses can be replaced by failures."""

    def __init__(self, *failures, usage=True):
        super().__init__()
        self.failures = list(failures)
        self.usage = usage

    def post(self, url, headers, body, timeout):
        failure = self.failures.pop(0) if self.failures else None
        if failure is not None:  # None in the script = answer normally
            self.requests.append(tuple(m["content"] for m in json.loads(body)["messages"]))
            if isinstance(failure, BaseException):
                raise failure
            return failure
        status, raw = super().post(url, headers, body, timeout)
        if self.usage:
            data = json.loads(raw)
            data["usage"] = {"prompt_tokens": 1000, "completion_tokens": 50}
            raw = json.dumps(data).encode()
        return status, raw


def clocked_provider(transport, **kwargs):
    ticks = itertools.count(0.0, 1.5)  # every chat() reads the clock twice: 1.5 s per successful call
    return QwenProvider(env=ENV, transport=transport, sleep=lambda s: None, clock=lambda: next(ticks), **kwargs)


def config(tmp_path, mode="record"):
    return AppConfig().with_overrides({
        "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
        "strategies": {"qwen_trend": {"weight": 1.0, "decision_interval": 6},
                       "qwen_momentum": {"weight": 1.0, "decision_interval": 6}},
        "agents": {"mode": mode, "cache_path": str(tmp_path / "agents.db")},
    })


def test_token_estimate_is_conservative():
    assert estimate_tokens(0) == 0 and estimate_tokens(3) == 1 and estimate_tokens(10) == 4
    text = '{"direction": "hold", "confidence": 0, "rationale": "unclear"}'
    assert estimate_tokens(len(text)) >= len(text) / 4  # at least the usual ~4 chars per token


def test_tracker_counts_calls_hits_and_misses(tmp_path):
    cfg = config(tmp_path)
    provider = clocked_provider(ScriptedTransport())
    BacktestEngine(cfg, SyntheticProvider(seed=3), llm_provider=provider).run(START, END)
    trend = provider.usage.per_agent["qwen_trend"]
    assert trend.calls == trend.cache_misses == 24 and trend.cache_hits == 0  # 2 symbols x 12 decision bars
    assert (trend.failures, trend.invalid_answers, trend.retries) == (0, 0, 0)
    assert trend.input_tokens == 24 * 1000 and trend.output_tokens == 24 * 50 and not trend.tokens_estimated
    assert trend.avg_latency_seconds == pytest.approx(1.5) and trend.total_latency_seconds == pytest.approx(36.0)

    BacktestEngine(cfg, SyntheticProvider(seed=3), llm_provider=provider).run(START, END)  # all cached now
    assert trend.calls == 24 and trend.cache_hits == 24
    total = provider.usage.total()
    assert total.calls == 48 and total.cache_hits == 48 and total.cache_misses == 48


def test_failures_retries_invalid_answers_and_estimates(candles_btc):
    transport = ScriptedTransport(
        (503, b"busy"), None,                             # call 1: one retry, then OK
        TransportTimeout("slow"), TransportTimeout("slow"),  # call 2: failure (max_retries=1)
        (200, json.dumps({"choices": [{"message": {"content": "no json here"}}]}).encode()),  # call 3: invalid
        usage=False,
    )
    provider = clocked_provider(transport, max_retries=1)
    strategy = QwenTrendStrategy(lookback=10)
    strategy.attach_provider(provider)
    strategy.configure(mode="live", cache=MemoryResponseCache())
    signals = [strategy.generate_signal("BTC/USDT", candles_btc) for _ in range(4)]
    s = provider.usage.per_agent["qwen_trend"]
    assert (s.calls, s.failures, s.invalid_answers, s.retries) == (4, 1, 1, 2)
    assert s.cache_hits == s.cache_misses == 0  # live mode never uses the cache
    assert s.tokens_estimated and s.estimated_calls == 4 and s.input_tokens > 0
    first = signals[0].metadata["llm"]
    assert first["ok"] and first["attempts"] == 2 and first["tokens_estimated"]
    assert first["input_tokens"] == estimate_tokens(first["input_chars"])
    failed = signals[1].metadata["llm"]
    assert failed["ok"] is False and failed["error"] == "ProviderTimeoutError" and failed["output_tokens"] == 0
    assert signals[2].metadata["llm"]["ok"] is True and "error" in signals[2].metadata

    rebuilt = usage_from_signals((sig.strategy, sig.metadata) for sig in signals)["qwen_trend"]
    assert (rebuilt.calls, rebuilt.failures, rebuilt.invalid_answers, rebuilt.retries) == (4, 1, 1, 2)
    assert rebuilt.input_tokens == s.input_tokens and rebuilt.output_tokens == s.output_tokens


@pytest.fixture(scope="module")
def candles_btc():
    return SyntheticProvider(seed=5).fetch_ohlcv("BTC/USDT", "1h", datetime(2023, 12, 1, tzinfo=UTC), END)


def test_stored_run_usage_matches_the_tracker(tmp_path):
    cfg = config(tmp_path)
    provider = clocked_provider(ScriptedTransport())
    with SQLiteStore(tmp_path / "h.db") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store, llm_provider=provider).run(START, END)
        rows = store.load_signals(result.run_id)
    stored = usage_from_signals((r.strategy, json.loads(r.metadata_json)) for r in rows.itertuples())
    for name, live in provider.usage.per_agent.items():
        s = stored[name]
        assert (s.calls, s.cache_hits, s.cache_misses, s.input_tokens, s.output_tokens) == (
            live.calls, live.cache_hits, live.cache_misses, live.input_tokens, live.output_tokens)
        assert s.total_latency_seconds == pytest.approx(live.total_latency_seconds)
    assert set(stored) == {"qwen_trend", "qwen_momentum"}  # deterministic strategies have no usage


def test_format_usage():
    stats = UsageStats()
    stats.add_call(CallRecord("qwen", "m", True, 1.8, 2, 3000, 120, 1000, 40, True), valid=True)
    stats.add_cache(True)
    text = format_usage({"qwen_trend": stats}, "Qwen usage")
    assert text.startswith("Qwen usage:")
    assert "calls: 1" in text and "cache hits: 1" in text and "retries: 1" in text
    assert "avg latency: 1.80s" in text and "input tokens: 1,000 (estimated)" in text


@pytest.fixture
def fake_qwen(monkeypatch, tmp_path):
    transport = ScriptedTransport()
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
    for var, value in ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.chdir(tmp_path)
    return transport


def test_cli_reports_usage(tmp_path, capsys, fake_qwen):
    cfg_file = tmp_path / "agents.toml"
    cfg_file.write_text("[market]\nsymbols = [\"BTC/USDT\"]\n[strategies.rsi]\n"
                        "[strategies.qwen_trend]\nweight = 1.0\ndecision_interval = 6\n")
    db = str(tmp_path / "h.db")
    base = ["--config", str(cfg_file), "--db", db]
    assert main([*base, "backtest", "--synthetic", "1", "--start", "2024-02-01", "--end", "2024-02-03"]) == 0
    out = capsys.readouterr().out
    assert "Qwen usage:" in out and "calls: 8" in out and "input tokens: 8,000" in out
    assert main([*base, "agent-report"]) == 0
    assert "Qwen usage:" in capsys.readouterr().out
    assert main([*base, "sweep", "--synthetic", "1", "--start", "2024-02-01", "--end", "2024-02-03",
                 "--param", "strategies.qwen_trend.weight=0,1"]) == 0
    out = capsys.readouterr().out
    assert "Qwen usage (qwen-test):" in out and "cache hits: 8" in out  # the backtest above recorded them
