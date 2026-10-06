"""Stage 15C: the Anthropic (Claude) provider, against a fake Messages API."""

import json
import logging

import pytest

from test_llm import FakeTransport
from test_specialists import END, START, RoleTransport, agents_config
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.doctor import FAIL, OK, run_checks
from trading_lab.llm import (
    PROVIDERS,
    AnthropicProvider,
    ProviderConfigError,
    ProviderError,
    ProviderResponseError,
    build_llm_provider,
)
from trading_lab.llm.anthropic import API_VERSION, messages_url
from trading_lab.smoke import environment_lines
from trading_lab.storage import SQLiteStore
from trading_lab.strategy_factory import strategies_for

KEY = "anthropic-test-key-DO-NOT-LEAK-42"
ENV = {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_MODEL": "claude-test"}
BUY = '{"direction": "buy", "confidence": 0.7, "rationale": "steady uptrend"}'


def ok_body(*texts, stop="end_turn"):
    blocks = [{"type": "text", "text": t} for t in (texts or (BUY,))]
    return 200, json.dumps({"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
                            "content": blocks, "stop_reason": stop,
                            "usage": {"input_tokens": 321, "output_tokens": 45}}).encode()


def make(*script, env=ENV, **kwargs):
    transport = FakeTransport(*(script or (ok_body(),)))
    sleeps = []
    return AnthropicProvider(env=env, transport=transport, sleep=sleeps.append, **kwargs), transport, sleeps


@pytest.mark.parametrize("base, expected", [
    ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
    ("https://proxy.test/v1/", "https://proxy.test/v1/messages"),
    ("https://proxy.test/anthropic/v1/messages", "https://proxy.test/anthropic/v1/messages"),
])
def test_messages_url(base, expected):
    assert messages_url(base) == expected


def test_request_format():
    provider, transport, _ = make(temperature=0.0, max_output_tokens=300)
    completion = provider.chat("You are the Trend Agent.", "Market: BTC/USDT")
    request = transport.requests[0]
    assert request["url"] == "https://api.anthropic.com/v1/messages"  # the default endpoint
    assert request["headers"]["x-api-key"] == KEY and request["headers"]["anthropic-version"] == API_VERSION
    assert "Authorization" not in request["headers"]
    assert request["body"] == {"model": "claude-test", "system": "You are the Trend Agent.",
                               "messages": [{"role": "user", "content": "Market: BTC/USDT"}],
                               "max_tokens": 300, "temperature": 0.0}
    assert completion.text == BUY and completion.model == "claude-test"
    assert (completion.input_tokens, completion.output_tokens, completion.finish_reason) == (321, 45, "end_turn")
    assert provider.cache_params["provider"] == "anthropic"  # never shares cached answers with Qwen


def test_text_blocks_are_joined_and_other_blocks_ignored():
    body = json.loads(ok_body('{"direction": "buy", ', '"confidence": 0.5, "rationale": "x"}')[1])
    body["content"].insert(0, {"type": "thinking", "thinking": "hmm"})
    provider, _, _ = make((200, json.dumps(body).encode()))
    assert json.loads(provider.chat("s", "u").text)["confidence"] == 0.5


@pytest.mark.parametrize("body, match", [
    ({"content": []}, "no text content block"),
    ({"content": [{"type": "tool_use", "name": "x"}]}, "no text content block"),
    ({"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}, "reported an error"),
    ({"content": "plain"}, "no content blocks"),
    ({"content": [{"type": "text", "text": "   "}]}, "empty answer"),
])
def test_malformed_responses(body, match):
    provider, transport, _ = make((200, json.dumps(body).encode()))
    with pytest.raises(ProviderResponseError, match=match):
        provider.chat("s", "u")
    assert transport.calls == 1  # not retried


def test_overloaded_is_retried_and_bad_keys_fail_fast(caplog):
    provider, transport, sleeps = make((529, b'{"type":"error","error":{"type":"overloaded_error"}}'), ok_body())
    assert provider.chat("s", "u").text == BUY and transport.calls == 2 and sleeps == [1.0]
    echo = json.dumps({"type": "error", "error": {"message": f"invalid x-api-key {KEY}"}}).encode()
    provider, transport, _ = make((401, echo))
    with caplog.at_level(logging.DEBUG), pytest.raises(ProviderError) as info:
        provider.chat("s", "u")
    assert transport.calls == 1 and "check ANTHROPIC_API_KEY" in str(info.value)
    assert KEY not in str(info.value) and KEY not in repr(provider) and KEY not in caplog.text


def test_environment():
    provider, _, _ = make(env={"ANTHROPIC_MODEL": "claude-test"})
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_API_KEY"):
        provider.chat("s", "u")
    provider.check_ready(need_credentials=False)  # replay mode needs only the model name
    with pytest.raises(ProviderConfigError, match="ANTHROPIC_MODEL"):
        make(env={"ANTHROPIC_API_KEY": KEY})[0].check_ready()
    assert AnthropicProvider.required_env() == ["API_KEY", "MODEL"]  # the URL has a default
    custom, transport, _ = make(env={**ENV, "ANTHROPIC_API_URL": "https://gateway.test/v1"})
    custom.chat("s", "u")
    assert transport.requests[0]["url"] == "https://gateway.test/v1/messages"
    with pytest.raises(ProviderConfigError, match="http"):
        make(env={**ENV, "ANTHROPIC_API_URL": "ftp://gateway.test"})[0].check_ready()


def test_configuration_and_factory():
    cfg = AppConfig.from_mapping({"agents": {"provider": "anthropic", "max_output_tokens": 400}})
    provider = build_llm_provider(cfg.agents, env=ENV)
    assert isinstance(provider, AnthropicProvider) and provider.max_output_tokens == 400
    assert PROVIDERS["anthropic"] is AnthropicProvider


def test_the_three_agents_run_on_claude(tmp_path):
    inner = RoleTransport()

    class MessagesAPI:
        """Answers in the Messages format, using the role-aware fake behind it."""

        def post(self, url, headers, body, timeout):
            request = json.loads(body)
            assert url.endswith("/v1/messages") and headers["x-api-key"] == KEY
            chat = {"messages": [{"role": "system", "content": request["system"]}, *request["messages"]]}
            _, raw = inner.post(url, {}, json.dumps(chat).encode(), timeout)
            text = json.loads(raw)["choices"][0]["message"]["content"]
            return ok_body(text)

    cfg = agents_config(tmp_path).with_overrides({"agents": {"provider": "anthropic"}})
    provider = AnthropicProvider(env=ENV, transport=MessagesAPI(), sleep=lambda s: None)
    with SQLiteStore(":memory:") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                                strategies=strategies_for(cfg, llm_provider=provider)).run(START, END)
        assert store.get_run(result.run_id)["config"]["agents"]["provider"] == "anthropic"
    agent_signals = [s for s in result.signals if s.metadata.get("called")]
    assert len(agent_signals) == len(inner.requests) == 3 * 24
    assert all(s.metadata["llm"]["provider"] == "anthropic" and s.metadata["llm"]["input_tokens"] == 321
               for s in agent_signals)
    assert {s.metadata.get("regime") for s in agent_signals if s.strategy == "qwen_trend"} == {"bullish_trend"}


def test_doctor_and_smoke_lines(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[agents]\nprovider = "anthropic"\n[strategies.qwen_trend]\nweight = 1.0\n')
    model = {c.name: c for c in run_checks(str(cfg_file), env=ENV)}["model"]
    assert model.status == OK and "ANTHROPIC_API_URL" not in model.detail and KEY not in model.detail
    assert {c.name: c for c in run_checks(str(cfg_file), env={})}["model"].status == FAIL
    for var in ("ANTHROPIC_API_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    lines = environment_lines(AnthropicProvider())
    assert lines == ["ANTHROPIC_API_URL: not set -> default https://api.anthropic.com/v1/messages",
                     "ANTHROPIC_API_KEY: MISSING", "ANTHROPIC_MODEL: set -> claude-test"]
