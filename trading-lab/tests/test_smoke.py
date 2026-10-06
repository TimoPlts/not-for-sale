"""Stage 9F: ``trading-lab agent-test`` with mocked credentials (no real model, no trades)."""

import json

import pytest

from test_specialists import ENV, RoleTransport
from trading_lab.cli import main

TOKEN = ENV["QWEN_API_KEY"]


class SmokeTransport(RoleTransport):
    """Answers the connectivity prompt; role prompts are answered like RoleTransport."""

    def __init__(self, reply=None):
        super().__init__()
        self.reply = reply

    def post(self, url, headers, body, timeout):
        system = json.loads(body)["messages"][0]["content"]
        if self.reply is not None:
            self.requests.append(("smoke", ""))
            return self.reply
        if "connectivity check" in system:
            self.requests.append(("smoke", ""))
            answer = '{"direction": "HOLD", "confidence": 0, "rationale": "connectivity test"}'
            return 200, json.dumps({"model": "qwen-test", "choices": [{"message": {"content": answer}}],
                                    "usage": {"prompt_tokens": 42, "completion_tokens": 17}}).encode()
        return super().post(url, headers, body, timeout)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Mocked Qwen credentials and endpoint; any simulated order would fail the test."""
    for var, value in ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.chdir(tmp_path)

    def no_trading(*args, **kwargs):
        raise AssertionError("agent-test must never submit an order")

    monkeypatch.setattr("trading_lab.execution.paper.PaperExecutor.submit", no_trading)

    def use(transport):
        monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
        return transport

    return use


def test_provider_smoke_test(sandbox, capsys, tmp_path):
    transport = sandbox(SmokeTransport())
    assert main(["agent-test", "qwen"]) == 0
    out = capsys.readouterr().out
    assert "QWEN_API_KEY: set (hidden)" in out and "QWEN_MODEL: set -> qwen-test" in out
    assert "model: qwen-test" in out and "latency:" in out and "42 in / 17 out (reported)" in out
    assert "structured answer: HOLD" in out and out.rstrip().endswith("OK")
    assert TOKEN not in out and len(transport.requests) == 1
    assert list(tmp_path.iterdir()) == []  # no database, no answer cache
    assert main(["agent-test"]) == 0  # defaults to the configured provider


def test_missing_variables_are_named(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    for var in ENV:
        monkeypatch.delenv(var, raising=False)
    assert main(["agent-test", "qwen"]) == 1
    captured = capsys.readouterr()
    assert "QWEN_API_URL: MISSING" in captured.out and "QWEN_API_KEY" in captured.err


@pytest.mark.parametrize("reply, expected", [
    ((200, json.dumps({"choices": [{"message": {"content": "hello there"}}]}).encode()), "no JSON object"),
    ((401, f'{{"error": "bad token {TOKEN}"}}'.encode()), "check QWEN_API_KEY"),
    ((200, b"<html>gateway</html>"), "not JSON"),
])
def test_provider_smoke_test_failures(sandbox, capsys, reply, expected):
    sandbox(SmokeTransport(reply))
    assert main(["agent-test", "qwen"]) == 1
    captured = capsys.readouterr()
    assert "FAILED" in captured.err and expected in captured.err and TOKEN not in captured.err + captured.out


@pytest.mark.parametrize("agent, label", [
    ("qwen_trend", "regime=bullish_trend"),
    ("qwen_momentum", "momentum_state=neutral"),
    ("qwen_risk", "risk_state=moderate"),
])
def test_agent_smoke_tests(sandbox, capsys, tmp_path, agent, label):
    transport = sandbox(SmokeTransport())
    assert main(["agent-test", agent, "--synthetic", "3", "--symbol", "ETH/USDT"]) == 0
    out = capsys.readouterr().out
    assert "data: synthetic-3 ETH/USDT 1h" in out and label in out and "rationale:" in out
    assert len(transport.requests) == 1 and list(tmp_path.iterdir()) == []
    assert ("flat simulated portfolio" in out) == (agent == "qwen_risk")


def test_agent_smoke_test_reports_bad_answers(sandbox, capsys):
    sandbox(SmokeTransport((200, json.dumps({"choices": [{"message": {"content": '{"direction": "BUY"}'}}]}).encode())))
    assert main(["agent-test", "qwen_trend", "--synthetic", "3"]) == 1
    assert "missing" in capsys.readouterr().err


def test_unknown_targets_are_rejected(sandbox, capsys):
    sandbox(SmokeTransport())
    assert main(["agent-test", "rsi", "--synthetic", "3"]) == 1
    err = capsys.readouterr().err
    assert "not an LLM agent" in err and "qwen_trend" in err
