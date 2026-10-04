"""AI agent framework tests: context, parsing, record/replay, failure handling, integration."""

import json
import random
from datetime import datetime, timezone

import pytest

from trading_lab.agents import (
    RESPONSE_INSTRUCTIONS,
    Agent,
    AgentResponse,
    AgentResponseError,
    AgentStrategy,
    LLMAgentStrategy,
    MemoryResponseCache,
    SQLiteResponseCache,
    TrendAnalystStrategy,
    build_context,
    indicator_frame,
    parse_agent_json,
)
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.core.models import Direction
from trading_lab.data import SyntheticProvider
from trading_lab.strategy_factory import strategies_for

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 8, tzinfo=UTC)


@pytest.fixture(scope="module")
def candles():
    return SyntheticProvider(seed=5).fetch_ohlcv("BTC/USDT", "1h", datetime(2024, 1, 20, tzinfo=UTC), END)


class CountingAgent(Agent):
    """Deterministic test agent: BUY when the last close is above the first, else SELL."""

    name = "counting"

    def __init__(self):
        self.calls = 0

    def decide(self, context):
        self.calls += 1
        first, last = context.candles[0]["close"], context.candles[-1]["close"]
        direction = Direction.BUY if last > first else Direction.SELL
        return AgentResponse(direction, 0.8, f"last {last} vs first {first}")


class RandomAgent(Agent):
    """Non-deterministic agent, like an LLM with temperature > 0."""

    name = "random"

    def decide(self, context):
        return AgentResponse(random.choice(list(Direction)), round(random.random(), 3), "coin flip")


class ExplodingAgent(Agent):
    name = "exploding"

    def decide(self, context):
        raise TimeoutError("model did not answer")


def make_strategy(agent, **kwargs):
    class _S(AgentStrategy):
        name = f"agent_{agent.name}"

        def build_agent(self):
            return agent

    strategy = _S(**kwargs)
    return strategy


# ------------------------------------------------------------------- context
def test_context_is_causal_rounded_and_serialisable(candles):
    ind = indicator_frame(candles)
    i = 200
    ctx = build_context("BTC/USDT", "1h", candles, ind, i, lookback=10)
    assert len(ctx.candles) == 10 and ctx.candles[-1]["time"] == candles.index[i].strftime("%Y-%m-%d %H:%M")
    assert ctx.bar_time == candles.index[i].strftime("%Y-%m-%dT%H:%M")
    truncated = candles.iloc[: i + 1]
    again = build_context("BTC/USDT", "1h", truncated, indicator_frame(truncated), i, lookback=10)
    assert again == ctx and again.fingerprint() == ctx.fingerprint()
    json.dumps(ctx.to_json())  # JSON-serialisable
    assert len(str(ctx.indicators["rsi_14"]).replace(".", "").lstrip("0")) <= 7
    prompt = ctx.to_prompt()
    assert "BTC/USDT" in prompt and "rsi_14" in prompt


# ------------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    "text, direction, confidence",
    [
        ('{"direction": "buy", "confidence": 0.7, "rationale": "up"}', Direction.BUY, 0.7),
        ('```json\n{"direction": "SELL", "confidence": 1}\n```', Direction.SELL, 1.0),
        ('Sure! {"direction": "hold", "confidence": 0, "rationale": "unclear"} Thanks', Direction.HOLD, 0.0),
    ],
)
def test_parse_valid_answers(text, direction, confidence):
    response = parse_agent_json(text)
    assert response.direction is direction and response.confidence == confidence


@pytest.mark.parametrize(
    "text, match",
    [
        ("", "empty"),
        ("I think you should buy", "no JSON"),
        ('{"direction": "buy"}', "missing"),
        ('{"direction": "short", "confidence": 0.5}', "direction"),
        ('{"direction": "buy", "confidence": 1.5}', "confidence"),
        ('{"direction": "buy", "confidence": "high"}', "confidence"),
        ('{"direction": "buy", "confidence": 0.5', "no JSON|invalid JSON"),
        ("[1, 2]", "no JSON|object"),
    ],
)
def test_parse_rejects_invalid_answers(text, match):
    with pytest.raises(AgentResponseError, match=match):
        parse_agent_json(text)


def test_instructions_describe_the_schema():
    assert '"direction"' in RESPONSE_INSTRUCTIONS and "cannot place orders" in RESPONSE_INSTRUCTIONS


# ------------------------------------------------------------- record/replay
def test_record_then_replay_never_calls_the_agent_again(candles):
    cache = MemoryResponseCache()
    agent = CountingAgent()
    recorder = make_strategy(agent, lookback=10)
    recorder.configure(mode="record", cache=cache)
    first = recorder.generate_signals("BTC/USDT", candles)
    calls = agent.calls
    assert calls == len(candles) - recorder.warmup_bars + 1 and len(cache) == calls
    assert any(s.metadata.get("cache") == "stored" for s in first)

    replayer = make_strategy(agent, lookback=10)
    replayer.configure(mode="replay", cache=cache)
    second = replayer.generate_signals("BTC/USDT", candles)
    assert agent.calls == calls  # replay never calls the agent
    assert [(s.direction, s.confidence) for s in second] == [(s.direction, s.confidence) for s in first]
    assert all(s.metadata.get("cache") == "hit" for s in second[recorder.warmup_bars:])


def test_replay_with_empty_cache_holds(candles):
    agent = CountingAgent()
    strategy = make_strategy(agent, lookback=10)
    strategy.configure(mode="replay", cache=MemoryResponseCache())
    signals = strategy.generate_signals("BTC/USDT", candles)
    assert agent.calls == 0
    assert all(s.direction is Direction.HOLD for s in signals)
    assert "not in cache" in signals[-1].metadata["error"]


def test_live_mode_always_asks(candles):
    agent = CountingAgent()
    strategy = make_strategy(agent, lookback=10)
    cache = MemoryResponseCache()
    strategy.configure(mode="live", cache=cache)
    strategy.generate_signals("BTC/USDT", candles.iloc[:80])
    strategy.generate_signals("BTC/USDT", candles.iloc[:80])
    assert agent.calls == 2 * (80 - strategy.warmup_bars + 1) and len(cache) == 0


def test_nondeterministic_agent_backtests_are_reproducible_via_cache(tmp_path):
    random.seed(1)
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    cache_path = tmp_path / "agents.db"

    def run(mode):
        strategy = make_strategy(RandomAgent(), lookback=10)
        strategy.configure(mode=mode, cache=SQLiteResponseCache(cache_path))
        return BacktestEngine(cfg, SyntheticProvider(seed=2), strategies=[strategy]).run(START, END)

    recorded = run("record")
    random.seed(999)  # the agent would now answer differently...
    replayed = run("replay")  # ...but replay uses the recorded answers
    assert replayed.fills == recorded.fills
    assert replayed.metrics == recorded.metrics


def test_cache_key_changes_when_agent_changes(candles):
    cache = MemoryResponseCache()
    a = make_strategy(CountingAgent(), lookback=10)
    a.configure(mode="record", cache=cache)
    a.generate_signals("BTC/USDT", candles.iloc[:60])
    entries = len(cache)

    class CountingV2(CountingAgent):
        version = "2"

    b = make_strategy(CountingV2(), lookback=10)
    b.configure(mode="replay", cache=cache)
    assert all(s.direction is Direction.HOLD for s in b.generate_signals("BTC/USDT", candles.iloc[:60]))
    assert len(cache) == entries


def test_sqlite_cache_persists(tmp_path):
    path = tmp_path / "c.db"
    cache = SQLiteResponseCache(path)
    meta = dict(agent="a", version="1", symbol="BTC/USDT", timeframe="1h", bar_time="t", context_hash="h")
    cache.put("k", AgentResponse(Direction.BUY, 0.6, "why"), meta=meta)
    cache.close()
    reopened = SQLiteResponseCache(path)
    assert reopened.get("k") == AgentResponse(Direction.BUY, 0.6, "why")
    assert reopened.get("missing") is None and len(reopened) == 1


# ------------------------------------------------------------------ failures
def test_agent_errors_become_hold_signals(candles):
    strategy = make_strategy(ExplodingAgent(), lookback=10)
    signals = strategy.generate_signals("BTC/USDT", candles.iloc[:60])
    assert all(s.direction is Direction.HOLD for s in signals)
    assert "TimeoutError" in signals[-1].metadata["error"]


def test_decision_interval_limits_calls(candles):
    agent = CountingAgent()
    strategy = make_strategy(agent, lookback=10, decision_interval=4)
    signals = strategy.generate_signals("BTC/USDT", candles.iloc[:100])
    decided = [s for s in signals if "rationale" in s.metadata]
    assert agent.calls == len(decided) and 12 <= agent.calls <= 13
    assert all(int(s.timestamp.timestamp()) // 3600 % 4 == 0 for s in decided)


# --------------------------------------------------------------- integration
def test_trend_analyst_is_configurable_and_consistent(tmp_path, candles):
    cfg = AppConfig().with_overrides({
        "strategies": {"trend_analyst": {"weight": 2.0, "min_trend": 0.003}},
        "agents": {"cache_path": str(tmp_path / "a.db")},
    })
    strategies = strategies_for(cfg)
    analyst = strategies[-1]
    assert isinstance(analyst, TrendAnalystStrategy) and analyst.params["min_trend"] == 0.003
    fast = analyst.generate_signals("BTC/USDT", candles)
    slow = [analyst.generate_signal("BTC/USDT", candles.iloc[: i + 1]) for i in range(len(candles))]
    assert [(s.direction, s.confidence, s.metadata.get("rationale")) for s in fast] == [
        (s.direction, s.confidence, s.metadata.get("rationale")) for s in slow
    ]
    assert {s.direction for s in fast} >= {Direction.BUY, Direction.SELL}
    assert all(s.metadata.get("rationale") for s in fast if s.direction is not Direction.HOLD)


def test_backtest_with_agent_records_rationales(tmp_path):
    from trading_lab.storage import SQLiteStore

    cfg = AppConfig().with_overrides({
        "market": {"symbols": ["ETH/USDT"]},
        "strategies": {"trend_analyst": {}},
        "agents": {"cache_path": str(tmp_path / "a.db")},
    })
    with SQLiteStore(tmp_path / "h.db") as store:
        r = BacktestEngine(cfg, SyntheticProvider(seed=4), store=store).run(START, END)
        stored = store.load_signals(r.run_id)
        analyst_rows = stored[stored["strategy"] == "trend_analyst"]
        assert len(analyst_rows) == len(r.equity_curve)
        assert analyst_rows["metadata_json"].str.contains("rationale").any()


def test_llm_agent_with_fake_completion(candles):
    prompts = []

    def complete(system, user):
        prompts.append((system, user))
        return '{"direction": "buy", "confidence": 0.65, "rationale": "fake model says up"}'

    strategy = LLMAgentStrategy(complete=complete, model="fake-1", lookback=5)
    sig = strategy.generate_signal("BTC/USDT", candles.iloc[:60])
    assert sig.direction is Direction.BUY and sig.metadata["rationale"] == "fake model says up"
    system, user = prompts[0]
    assert system == RESPONSE_INSTRUCTIONS and "BTC/USDT" in user
    assert strategy.params["model"] == "fake-1"

    bad = LLMAgentStrategy(complete=lambda s, u: "no idea", lookback=5)
    held = bad.generate_signal("BTC/USDT", candles.iloc[:60])
    assert held.direction is Direction.HOLD and "AgentResponseError" in held.metadata["error"]
