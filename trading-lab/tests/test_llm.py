"""Stage 9A: the Qwen provider and LLM-backed agents.

No test talks to a real model: requests go to a fake transport, or to a
throwaway HTTP server on 127.0.0.1 for the standard-library transport.
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from trading_lab.agents import LLMAnalystStrategy, MemoryResponseCache, SQLiteResponseCache
from trading_lab.backtest import BacktestEngine
from trading_lab.config import LLM_PROVIDERS, AgentsConfig, AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import Direction
from trading_lab.data import SyntheticProvider
from trading_lab.llm import (
    PROVIDERS,
    ProviderConfigError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    QwenProvider,
    TransportConnectionError,
    TransportTimeout,
    UrllibTransport,
    build_llm_provider,
    chat_completions_url,
)
from trading_lab.storage import SQLiteStore
from trading_lab.strategy_factory import strategies_for

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 4, tzinfo=UTC)
TOKEN = "sk-test-DO-NOT-LEAK-0123456789"
ENV = {"QWEN_API_URL": "https://qwen.example.test/v1", "QWEN_API_KEY": TOKEN, "QWEN_MODEL": "qwen-test"}
QWEN_VARS = ("QWEN_API_URL", "QWEN_API_KEY", "QWEN_MODEL")
BUY = '{"direction": "buy", "confidence": 0.7, "rationale": "steady uptrend"}'


def ok_body(content=BUY, *, finish="stop", usage=True):
    body = {
        "id": "chatcmpl-1",
        "model": "qwen-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    return 200, json.dumps(body).encode()


class FakeTransport:
    """Plays back scripted answers; the last one repeats. Records every request."""

    def __init__(self, *script):
        self.script = list(script) or [ok_body()]
        self.requests = []

    def post(self, url, headers, body, timeout):
        self.requests.append({"url": url, "headers": dict(headers), "body": json.loads(body), "timeout": timeout})
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        return step

    @property
    def calls(self):
        return len(self.requests)


def make_provider(*script, env=ENV, **kwargs):
    sleeps = []
    transport = FakeTransport(*script)
    provider = QwenProvider(env=env, transport=transport, sleep=sleeps.append, **kwargs)
    return provider, transport, sleeps


def llm_config(tmp_path, mode="record", **strategy):
    return AppConfig().with_overrides({
        "market": {"symbols": ["BTC/USDT"]},
        "strategies": {
            "rsi": {"enabled": False}, "macd": {"enabled": False}, "bollinger": {"enabled": False},
            "llm_analyst": {"lookback": 10, "decision_interval": 6, **strategy},
        },
        "agents": {"mode": mode, "cache_path": str(tmp_path / "agents.db")},
    })


@pytest.fixture(scope="module")
def candles():
    return SyntheticProvider(seed=5).fetch_ohlcv("BTC/USDT", "1h", datetime(2024, 1, 20, tzinfo=UTC), END)


@pytest.fixture
def no_qwen_env(monkeypatch):
    for var in QWEN_VARS:
        monkeypatch.delenv(var, raising=False)


def attached(provider, mode="record", cache=None, **kwargs):
    strategy = LLMAnalystStrategy(lookback=10, **kwargs)
    strategy.attach_provider(provider)
    strategy.configure(mode=mode, cache=cache if cache is not None else MemoryResponseCache())
    return strategy


# -------------------------------------------------------------- configuration
def test_missing_environment_variables_are_named(no_qwen_env):
    provider = build_llm_provider(AgentsConfig())
    assert isinstance(provider, QwenProvider)
    with pytest.raises(ProviderConfigError) as exc:
        provider.check_ready(need_credentials=True)
    assert all(var in str(exc.value) for var in QWEN_VARS)
    with pytest.raises(ProviderConfigError, match="QWEN_MODEL") as exc:
        provider.check_ready(need_credentials=False)  # replay needs only the model name
    assert "QWEN_API_KEY" not in str(exc.value)
    QwenProvider(env={"QWEN_MODEL": "m"}).check_ready(need_credentials=False)


def test_missing_environment_fails_before_a_run_starts(tmp_path, no_qwen_env):
    with pytest.raises(ConfigError, match="QWEN_API_URL"):
        strategies_for(llm_config(tmp_path))
    with pytest.raises(ConfigError, match="QWEN_MODEL"):
        strategies_for(llm_config(tmp_path, mode="replay"))


def test_chat_without_credentials_never_sends_a_request():
    provider, transport, _ = make_provider(env={"QWEN_MODEL": "m"})
    with pytest.raises(ProviderConfigError):
        provider.chat("s", "u")
    assert transport.calls == 0


def test_only_http_urls_are_accepted():
    provider, transport, _ = make_provider(env={**ENV, "QWEN_API_URL": "file:///etc/passwd"})
    with pytest.raises(ProviderConfigError, match="http"):
        provider.chat("s", "u")
    assert transport.calls == 0


def test_agents_config_validates_provider_settings():
    cfg = AgentsConfig()
    assert (cfg.provider, cfg.max_retries, cfg.temperature) == ("qwen", 2, 0.0)
    for bad in ({"provider": "gpt"}, {"request_timeout_seconds": 0}, {"max_retries": -1},
                {"max_retries": 1.5}, {"temperature": 3.0}, {"max_output_tokens": 4}):
        with pytest.raises(ConfigError):
            AgentsConfig(**bad)
    assert set(PROVIDERS) == set(LLM_PROVIDERS)


def test_config_has_no_place_for_model_credentials():
    for key in ("api_key", "QWEN_API_KEY", "token", "api_url"):
        with pytest.raises(ConfigError, match="unknown key"):
            AppConfig.from_mapping({"agents": {key: "x"}})


def test_old_configs_without_provider_keys_still_load():
    cfg = AppConfig.from_mapping({"agents": {"mode": "replay", "cache_path": "a.db"}})
    assert cfg.agents.provider == "qwen" and cfg.agents.request_timeout_seconds == 30.0


def test_default_toml_documents_the_provider(default_config_path):
    from trading_lab.config import load_config

    cfg = load_config(default_config_path)
    assert cfg.agents == AgentsConfig()
    assert "llm_analyst" not in [s.name for s in cfg.enabled_strategies]  # opt-in only


# ------------------------------------------------------------ request/response
@pytest.mark.parametrize(
    "base, expected",
    [
        ("https://h/v1", "https://h/v1/chat/completions"),
        ("https://h/v1/", "https://h/v1/chat/completions"),
        ("https://h/compatible-mode/v1/chat/completions", "https://h/compatible-mode/v1/chat/completions"),
    ],
)
def test_chat_completions_url(base, expected):
    assert chat_completions_url(base) == expected


def test_successful_request_and_response():
    provider, transport, sleeps = make_provider(timeout_seconds=12.5, temperature=0.2, max_output_tokens=256)
    completion = provider.chat("system prompt", "user prompt")
    assert completion.text == BUY and completion.model == "qwen-test"
    assert (completion.attempts, completion.input_tokens, completion.output_tokens) == (1, 120, 30)
    assert completion.finish_reason == "stop" and completion.latency_seconds >= 0 and sleeps == []

    request = transport.requests[0]
    assert request["url"] == "https://qwen.example.test/v1/chat/completions"
    assert request["timeout"] == 12.5
    assert request["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert request["body"] == {
        "model": "qwen-test",
        "messages": [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "user prompt"}],
        "temperature": 0.2,
        "max_tokens": 256,
    }
    assert provider.complete("s", "u") == BUY  # the plain complete(system, user) -> text interface


def test_usage_is_optional_and_thinking_is_stripped():
    provider, _, _ = make_provider(ok_body(f"<think>maybe {{buy}}? let me see</think>\n{BUY}", usage=False))
    completion = provider.chat("s", "u")
    assert completion.text == BUY and completion.input_tokens is None and completion.output_tokens is None


def test_successful_answer_becomes_a_signal_with_rationale(candles):
    provider, transport, _ = make_provider()
    strategy = attached(provider)
    signal = strategy.generate_signal("BTC/USDT", candles)
    assert signal.direction is Direction.BUY and signal.confidence == 0.7
    assert signal.metadata["rationale"] == "steady uptrend" and signal.metadata["cache"] == "stored"
    assert signal.metadata["params"]["model"] == "qwen-test"
    assert signal.metadata["params"]["provider"] == "qwen"
    user_prompt = transport.requests[0]["body"]["messages"][1]["content"]
    assert "BTC/USDT" in user_prompt and "rsi_14" in user_prompt


# ------------------------------------------------------------------ malformed
@pytest.mark.parametrize(
    "body, match",
    [
        (b"<html>502 Bad Gateway</html>", "not JSON"),
        (b"[]", "not a JSON object"),
        (json.dumps({"choices": []}).encode(), "choices"),
        (json.dumps({"choices": [{"message": {"content": None}}]}).encode(), "not text"),
        (json.dumps({"choices": [{"message": {"content": "  "}}]}).encode(), "empty"),
        (json.dumps({"error": {"message": "model overloaded"}}).encode(), "overloaded"),
    ],
)
def test_malformed_http_responses_are_clear_errors_and_not_retried(body, match):
    provider, transport, _ = make_provider((200, body))
    with pytest.raises(ProviderResponseError, match=match):
        provider.chat("s", "u")
    assert transport.calls == 1


def test_malformed_model_json_becomes_hold(candles):
    provider, _, _ = make_provider(ok_body('{"direction": "buy", "confidence": "very"}'))
    signal = attached(provider).generate_signal("BTC/USDT", candles)
    assert signal.direction is Direction.HOLD and signal.confidence == 0.0
    assert "AgentResponseError" in signal.metadata["error"]


def test_truncated_answer_explains_the_fix(candles):
    provider, _, _ = make_provider(ok_body('{"direction": "buy", "confid', finish="length"))
    signal = attached(provider).generate_signal("BTC/USDT", candles)
    assert signal.direction is Direction.HOLD and "max_output_tokens" in signal.metadata["error"]


# ------------------------------------------------------------ timeout & retry
def test_timeouts_are_retried_with_backoff_then_reported():
    provider, transport, sleeps = make_provider(TransportTimeout("slow"), retry_backoff_seconds=0.5)
    with pytest.raises(ProviderTimeoutError, match="within 30s.*gave up after 3") as exc:
        provider.chat("s", "u")
    assert transport.calls == 3 and exc.value.attempts == 3
    assert sleeps == [0.5, 1.0]


def test_transient_failures_are_retried_until_success():
    provider, transport, sleeps = make_provider(
        (503, b"busy"), TransportConnectionError("connection reset"), (429, b"slow down"), ok_body(),
        max_retries=3,
    )
    completion = provider.chat("s", "u")
    assert completion.text == BUY and completion.attempts == 4 and transport.calls == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_zero_retries_means_one_attempt():
    provider, transport, sleeps = make_provider((500, b"oops"), max_retries=0)
    with pytest.raises(ProviderError, match="HTTP 500"):
        provider.chat("s", "u")
    assert transport.calls == 1 and sleeps == []


@pytest.mark.parametrize("status, hint", [(401, "QWEN_API_KEY"), (403, "QWEN_API_KEY"), (404, "QWEN_MODEL"), (400, "")])
def test_client_errors_fail_fast_with_a_hint(status, hint):
    provider, transport, sleeps = make_provider((status, b'{"error": "nope"}'))
    with pytest.raises(ProviderError, match=f"HTTP {status}") as exc:
        provider.chat("s", "u")
    assert hint in str(exc.value) and transport.calls == 1 and sleeps == []


# ----------------------------------------------------------- failure -> HOLD
def test_provider_failures_become_hold_and_are_not_cached(candles):
    provider, transport, _ = make_provider(TransportTimeout("slow"), max_retries=1)
    cache = MemoryResponseCache()
    strategy = attached(provider, cache=cache)
    signal = strategy.generate_signal("BTC/USDT", candles)
    assert signal.direction is Direction.HOLD and "ProviderTimeoutError" in signal.metadata["error"]
    assert len(cache) == 0 and transport.calls == 2  # a failure is retried next time, never cached


def test_failing_provider_never_crashes_a_backtest(tmp_path):
    provider, transport, _ = make_provider((503, b"down"), max_retries=1)
    cfg = llm_config(tmp_path)
    with SQLiteStore(tmp_path / "h.db") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), strategies=strategies_for(cfg, llm_provider=provider),
                                store=store).run(START, END)
        assert not result.fills and transport.calls > 0
        rows = store.load_signals(result.run_id)
        assert set(rows["direction"]) == {"hold"}
        assert rows["metadata_json"].str.contains("HTTP 503").any()


# ------------------------------------------------------- record/replay/live
def test_record_reuses_cached_answers(tmp_path):
    cfg = llm_config(tmp_path)

    def run():
        provider, transport, _ = make_provider()
        strategies = strategies_for(cfg, llm_provider=provider)
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), strategies=strategies).run(START, END)
        return result, transport.calls

    first, first_calls = run()
    second, second_calls = run()
    assert first_calls > 0 and second_calls == 0
    assert second.fills == first.fills and second.metrics == first.metrics
    assert len(SQLiteResponseCache(cfg.agents.cache_path)) == first_calls


def test_replay_never_calls_the_model_and_needs_no_credentials(tmp_path):
    recorded_provider, recorded_transport, _ = make_provider()
    cfg = llm_config(tmp_path)
    recorded = BacktestEngine(cfg, SyntheticProvider(seed=3),
                              strategies=strategies_for(cfg, llm_provider=recorded_provider)).run(START, END)
    assert recorded_transport.calls > 0

    replay_provider, replay_transport, _ = make_provider(env={"QWEN_MODEL": "qwen-test"})
    replay_cfg = llm_config(tmp_path, mode="replay")
    replayed = BacktestEngine(replay_cfg, SyntheticProvider(seed=3),
                              strategies=strategies_for(replay_cfg, llm_provider=replay_provider)).run(START, END)
    assert replay_transport.calls == 0
    assert replayed.fills == recorded.fills and replayed.metrics == recorded.metrics


def test_replay_with_another_model_finds_nothing(tmp_path, candles):
    cache = MemoryResponseCache()
    provider, _, _ = make_provider()
    attached(provider, cache=cache).generate_signal("BTC/USDT", candles)
    other, transport, _ = make_provider(env={"QWEN_MODEL": "qwen-other"})
    signal = attached(other, mode="replay", cache=cache).generate_signal("BTC/USDT", candles)
    assert signal.direction is Direction.HOLD and "not in cache" in signal.metadata["error"]
    assert transport.calls == 0


def test_live_mode_always_calls_and_caches_nothing(candles):
    provider, transport, _ = make_provider()
    cache = MemoryResponseCache()
    strategy = attached(provider, mode="live", cache=cache)
    first = strategy.generate_signal("BTC/USDT", candles)
    second = strategy.generate_signal("BTC/USDT", candles)
    assert transport.calls == 2 and len(cache) == 0
    assert first.metadata["cache"] == second.metadata["cache"] == "none"
    assert first.direction is second.direction is Direction.BUY


def test_decision_interval_limits_model_calls(candles):
    provider, transport, _ = make_provider()
    signals = attached(provider, decision_interval=8).generate_signals("BTC/USDT", candles.iloc[:120])
    decided = [s for s in signals if "rationale" in s.metadata]
    assert transport.calls == len(decided) and 8 <= len(decided) <= 9


# ------------------------------------------------------------------- secrets
def test_secrets_never_appear_in_errors_logs_repr_or_history(tmp_path, caplog):
    echo = (401, f'{{"error": "invalid token {TOKEN}"}}'.encode())
    provider, _, _ = make_provider(echo)
    with pytest.raises(ProviderError) as exc:
        provider.chat("s", "u")
    assert TOKEN not in str(exc.value) and "***" in str(exc.value)
    assert TOKEN not in repr(provider) and "token=set" in repr(provider)

    caplog.set_level(logging.DEBUG)
    flaky, _, _ = make_provider(TransportConnectionError(f"reset by {TOKEN}"), (502, TOKEN.encode()), ok_body())
    flaky.chat("s", "u")
    assert caplog.records and TOKEN not in caplog.text

    cfg = llm_config(tmp_path)
    failing, _, _ = make_provider(echo)
    with SQLiteStore(tmp_path / "h.db") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), strategies=strategies_for(cfg, llm_provider=failing),
                                store=store).run(START, END)
        assert TOKEN not in store.load_signals(result.run_id).to_csv()
        assert TOKEN not in json.dumps(store.get_run(result.run_id)["config"], default=str)


# ----------------------------------------------------- standard-library HTTP
class _Handler(BaseHTTPRequestHandler):
    delay = 0.0

    def do_POST(self):  # noqa: N802
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        time.sleep(self.delay)
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            status, body = 401, b'{"error": "bad token"}'
        else:
            status, body = ok_body(BUY if request["model"] == "qwen-test" else "?")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


@pytest.fixture
def local_server(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")  # talk to 127.0.0.1 directly, never via a proxy
    monkeypatch.setenv("no_proxy", "*")
    handler = type("Handler", (_Handler,), {"delay": 0.0})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, handler
    server.shutdown()
    server.server_close()


def test_urllib_transport_talks_to_a_real_http_server(local_server):
    server, _ = local_server
    env = {**ENV, "QWEN_API_URL": f"http://127.0.0.1:{server.server_address[1]}/v1"}
    assert QwenProvider(env=env).chat("s", "u").text == BUY
    with pytest.raises(ProviderError, match="HTTP 401"):
        QwenProvider(env={**env, "QWEN_API_KEY": "wrong"}).chat("s", "u")


def test_urllib_transport_times_out(local_server):
    server, handler = local_server
    handler.delay = 1.0
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
    with pytest.raises(TransportTimeout):
        UrllibTransport().post(url, {"Authorization": f"Bearer {TOKEN}"}, b'{"model": "qwen-test"}', 0.2)


def test_urllib_transport_reports_unreachable_endpoints(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    probe = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = probe.server_address[1]
    probe.server_close()  # nothing listens on this port any more
    with pytest.raises(TransportConnectionError):
        UrllibTransport().post(f"http://127.0.0.1:{port}/v1/chat/completions", {}, b"{}", 2.0)
