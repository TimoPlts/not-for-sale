"""Stage 9B: the Trend, Momentum and Risk/Regime agents (with a fake Qwen endpoint)."""

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.agents import (
    MemoryResponseCache,
    QwenMomentumStrategy,
    QwenRiskStrategy,
    QwenTrendStrategy,
)
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, load_config
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import DecisionAction, Direction, Signal
from trading_lab.data import SyntheticProvider
from trading_lab.engine import Bar, Intent, TradingSession
from trading_lab.ensemble import VotingEngine
from trading_lab.llm import QwenProvider
from trading_lab.storage import SQLiteStore
from trading_lab.strategy_factory import strategies_for

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 4, tzinfo=UTC)
ENV = {"QWEN_API_URL": "https://qwen.example.test/v1", "QWEN_API_KEY": "sk-test-key-123456", "QWEN_MODEL": "qwen-test"}
AGENTS = ("qwen_trend", "qwen_momentum", "qwen_risk")
LABELS = {"Trend Agent": ("regime", "bullish_trend"), "Momentum Agent": ("momentum_state", "neutral"),
          "Risk/Regime Agent": ("risk_state", "moderate")}


def feature(prompt, name):
    match = re.search(rf"^\s+{name}: (\S+)$", prompt, re.MULTILINE)
    return None if match is None or match.group(1) == "n/a" else float(match.group(1))


class RoleTransport:
    """Fake Qwen: answers deterministically from the features in the prompt."""

    def __init__(self, override=None):
        self.requests = []
        self.override = override

    def post(self, url, headers, body, timeout):
        request = json.loads(body)
        system, user = (m["content"] for m in request["messages"])
        self.requests.append((system, user))
        role = next(r for r in LABELS if f"You are the {r}." in system)
        label, value = LABELS[role]
        if self.override is not None:
            answer = self.override(role, user)
        else:
            close_vs = feature(user, "close_vs_ema_50_pct")
            accel = feature(user, "return_acceleration_pct")
            score = close_vs if close_vs is not None else accel
            direction = "HOLD" if score is None else ("BUY" if score > 0 else "SELL")
            answer = {"direction": direction, "confidence": 0.0 if direction == "HOLD" else 0.8,
                      "rationale": f"{role} looked at the data", label: value}
        content = json.dumps(answer) if isinstance(answer, dict) else answer
        return 200, json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}).encode()

    def roles(self):
        return [next(r for r in LABELS if f"You are the {r}." in s) for s, _ in self.requests]


def provider(transport):
    return QwenProvider(env=ENV, transport=transport, sleep=lambda s: None)


def agents_config(tmp_path, weights=None, mode="record", **extra):
    weights = weights or {name: 1.0 for name in AGENTS}
    return AppConfig().with_overrides({
        "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
        "strategies": {name: {"weight": w, "decision_interval": 6} for name, w in weights.items()},
        "agents": {"mode": mode, "cache_path": str(tmp_path / "agents.db")},
        **extra,
    })


@pytest.fixture(scope="module")
def candles():
    return SyntheticProvider(seed=5).fetch_ohlcv("BTC/USDT", "1h", datetime(2023, 12, 1, tzinfo=UTC), END)


def attached(cls, transport, mode="record", **kwargs):
    strategy = cls(**kwargs)
    strategy.attach_provider(provider(transport))
    strategy.configure(mode=mode, cache=MemoryResponseCache())
    return strategy


# ------------------------------------------------------------------- config
def test_agents_ship_switched_off(default_config_path, monkeypatch):
    for var in ENV:
        monkeypatch.delenv(var, raising=False)
    cfg = load_config(default_config_path)
    assert cfg == AppConfig()
    specs = {s.name: s for s in cfg.strategies}
    assert all(specs[a].enabled and specs[a].weight == 0 for a in AGENTS)
    assert [s.name for s in cfg.enabled_strategies] == ["rsi", "macd", "bollinger"]
    assert [s.name for s in strategies_for(cfg)] == ["rsi", "macd", "bollinger"]  # no env vars needed


def test_enabling_an_agent_needs_the_environment(tmp_path, monkeypatch):
    for var in ENV:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError, match="QWEN_API_KEY"):
        strategies_for(agents_config(tmp_path, {"qwen_trend": 1.0}))
    for var, value in ENV.items():
        monkeypatch.setenv(var, value)
    names = [s.name for s in strategies_for(agents_config(tmp_path, {"qwen_trend": 1.0}))]
    assert names == ["rsi", "macd", "bollinger", "qwen_trend"]


def test_agent_parameters_are_validated():
    with pytest.raises(ValueError, match="unknown parameter"):
        QwenTrendStrategy(temperature=1.0)
    with pytest.raises(ValueError, match="portfolio_context"):
        QwenRiskStrategy(portfolio_context="yes")
    assert QwenRiskStrategy().uses_portfolio and not QwenTrendStrategy().uses_portfolio
    assert QwenTrendStrategy(portfolio_context=True).uses_portfolio


def test_zero_weight_means_no_vote_and_no_call(tmp_path):
    transport = RoleTransport()
    cfg = agents_config(tmp_path, {"qwen_trend": 0.0, "qwen_momentum": 1.0}).with_overrides(
        {"voting": {"min_agreeing": 2}}
    )
    strategies = strategies_for(cfg, llm_provider=provider(transport))
    assert "qwen_trend" not in [s.name for s in strategies]
    result = BacktestEngine(cfg, SyntheticProvider(seed=3), strategies=strategies).run(START, END)
    assert {s.strategy for s in result.signals} == {"rsi", "macd", "bollinger", "qwen_momentum", "ensemble"}
    assert set(transport.roles()) == {"Momentum Agent"}


# ------------------------------------------------------------------ prompts
@pytest.mark.parametrize("cls, role, label", [
    (QwenTrendStrategy, "Trend Agent", '"regime": "bullish_trend" | "bearish_trend" | "sideways" | "uncertain"'),
    (QwenMomentumStrategy, "Momentum Agent", '"momentum_state": "strengthening" | "weakening" | "neutral"'),
    (QwenRiskStrategy, "Risk/Regime Agent", '"risk_state": "low" | "moderate" | "high" | "extreme"'),
])
def test_role_prompts(cls, role, label):
    prompt = cls.system_prompt
    assert prompt.startswith(f"You are the {role}.")
    assert label in prompt and '"direction": "BUY" | "SELL" | "HOLD"' in prompt
    assert "cannot place, size or cancel orders" in prompt and "long-only" in prompt
    assert len({QwenTrendStrategy.system_prompt, QwenMomentumStrategy.system_prompt,
                QwenRiskStrategy.system_prompt}) == 3


# ----------------------------------------------------------------- features
@pytest.mark.parametrize("cls, expected", [
    (QwenTrendStrategy, {"ema_20", "ema_50", "ema_200", "return_20_bars_pct", "rsi_14", "macd_hist",
                         "volume_trend_10_vs_50", "ema_50_slope_10_bars_pct"}),
    (QwenMomentumStrategy, {"rsi_14", "macd", "macd_hist", "return_acceleration_pct", "volume_vs_avg_20",
                            "last_body_pct", "last_upper_wick_pct", "last_close_location"}),
    (QwenRiskStrategy, {"atr_14", "atr_pct", "volatility_20_vs_100", "range_vs_avg_20",
                        "drawdown_from_50_bar_high_pct", "avg_quote_volume_20"}),
])
def test_features_are_complete_and_causal(cls, expected, candles):
    strategy = cls()
    full = strategy.context_frame(candles)
    assert expected <= set(full.columns)
    i = 400
    truncated = strategy.context_frame(candles.iloc[: i + 1])
    left, right = full.iloc[i], truncated.iloc[i]
    assert left.isna().equals(right.isna())
    assert (left.dropna() - right.dropna()).abs().max() < 1e-9


def test_ema_200_is_missing_until_enough_history(candles):
    frame = QwenTrendStrategy().context_frame(candles.iloc[:250])
    assert frame["ema_200"].iloc[150] != frame["ema_200"].iloc[150]  # NaN
    assert frame["ema_200"].iloc[249] > 0
    assert QwenTrendStrategy().history_bars >= 1000


# ------------------------------------------------------------------ answers
def test_labels_are_required_and_recorded(candles):
    transport = RoleTransport()
    signal = attached(QwenTrendStrategy, transport).generate_signal("BTC/USDT", candles)
    assert signal.direction in (Direction.BUY, Direction.SELL)
    assert signal.metadata["regime"] == "bullish_trend"
    assert signal.metadata["rationale"] == "Trend Agent looked at the data"
    _, user = transport.requests[0]
    assert "ema_200:" in user and "Simulated portfolio" not in user  # market-only by default


@pytest.mark.parametrize("answer, error", [
    ({"direction": "BUY", "confidence": 0.9, "rationale": "x"}, "missing"),
    ({"direction": "BUY", "confidence": 0.9, "rationale": "x", "regime": "moon"}, "regime must be one of"),
    ({"direction": "SHORT", "confidence": 0.9, "rationale": "x", "regime": "sideways"}, "direction"),
])
def test_invalid_structured_answers_become_hold(candles, answer, error):
    strategy = attached(QwenTrendStrategy, RoleTransport(override=lambda role, user: answer))
    signal = strategy.generate_signal("BTC/USDT", candles)
    assert signal.direction is Direction.HOLD and error in signal.metadata["error"]


def test_labels_survive_the_cache(candles):
    cache = MemoryResponseCache()
    recorder = QwenMomentumStrategy()
    recorder.attach_provider(provider(RoleTransport()))
    recorder.configure(mode="record", cache=cache)
    first = recorder.generate_signal("BTC/USDT", candles)
    replayer = QwenMomentumStrategy()
    transport = RoleTransport()
    replayer.attach_provider(provider(transport))
    replayer.configure(mode="replay", cache=cache)
    again = replayer.generate_signal("BTC/USDT", candles)
    assert transport.requests == [] and again.metadata["cache"] == "hit"
    assert again.metadata["momentum_state"] == first.metadata["momentum_state"] == "neutral"


# --------------------------------------------------------- portfolio context
def test_portfolio_view_reflects_positions_and_breakers():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
    session = TradingSession(cfg, VotingEngine({"x": 1.0}))
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    view = session.portfolio_view("BTC/USDT", t0)
    assert view["position"] == "flat" and view["exposure_pct"] == 0 and not view["new_entries_blocked"]

    session.last_close = {"BTC/USDT": 100.0, "ETH/USDT": 50.0}
    buy = Signal("x", "BTC/USDT", Direction.BUY, 1.0, t0)
    session.pending["BTC/USDT"] = Intent(DecisionAction.ENTER_SIGNAL, buy)
    session.open_bar(t0 + timedelta(hours=1), {"BTC/USDT": 100.0, "ETH/USDT": 50.0})
    session.last_close["BTC/USDT"] = 110.0
    view = session.portfolio_view("BTC/USDT", t0 + timedelta(hours=3))
    assert view["position"] == "long" and view["open_positions"] == 1
    assert view["position_return_pct"] > 9 and view["position_bars_held"] == 3
    assert 0 < view["exposure_pct"] <= 30 and view["stop_distance_pct"] > 0

    session.breakers.state.halted_reason = "max drawdown"
    view = session.portfolio_view("ETH/USDT", t0 + timedelta(hours=3))
    assert view["kill_switch_active"] and view["new_entries_blocked"] and view["block_reason"] == "kill switch"
    json.dumps(view)


def test_risk_agent_sees_the_portfolio_in_backtests(tmp_path):
    transport = RoleTransport()
    cfg = agents_config(tmp_path, {"qwen_risk": 1.0})
    result = BacktestEngine(cfg, SyntheticProvider(seed=3),
                            strategies=strategies_for(cfg, llm_provider=provider(transport))).run(START, END)
    prompts = [user for _, user in transport.requests]
    assert prompts and all("Simulated portfolio at that close:" in p for p in prompts)
    assert all("exposure_pct:" in p and "kill_switch_active:" in p and "stop_outs_last_24_bars:" in p
               for p in prompts)
    risk = [s for s in result.signals if s.strategy == "qwen_risk" and "rationale" in s.metadata]
    assert risk and all(s.metadata["risk_state"] == "moderate" for s in risk)
    assert all(s.metadata["params"]["portfolio_context"] is True for s in risk)


def test_risk_agent_cannot_override_the_kill_switch():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    session = TradingSession(cfg, VotingEngine({"qwen_risk": 1.0}))
    session.breakers.state.halted_reason = "max drawdown 30% >= 25%"
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    seen = []

    def risk_vote(view):
        seen.append(view)
        return Signal("qwen_risk", "BTC/USDT", Direction.BUY, 1.0, t0, {"risk_state": "low"})

    bar = Bar(100, 101, 99, 100, 1000)
    session.close_bar(t0, {"BTC/USDT": bar}, {"BTC/USDT": [risk_vote]})
    session.open_bar(t0 + timedelta(hours=1), {"BTC/USDT": 100.0})
    assert seen[0]["kill_switch_active"] is True
    rejected = [d for d in session.records.decisions if d.action is DecisionAction.REJECTED]
    assert rejected and "circuit breaker" in rejected[0].reason
    assert session.portfolio.positions == {} and session.portfolio.fills == ()


# --------------------------------------------------------------- backtests
def test_three_agents_vote_in_the_ensemble(tmp_path):
    transport = RoleTransport()
    cfg = agents_config(tmp_path)
    with SQLiteStore(tmp_path / "h.db") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                                strategies=strategies_for(cfg, llm_provider=provider(transport))).run(START, END)
        rows = store.load_signals(result.run_id)
    assert set(transport.roles()) == set(LABELS)
    ensemble = [s for s in result.signals if s.strategy == "ensemble"]
    voters = {v["strategy"] for s in ensemble for v in s.metadata["votes"]}
    assert voters == {"rsi", "macd", "bollinger", *AGENTS}
    for name, label in zip(AGENTS, ("regime", "momentum_state", "risk_state")):
        stored = rows[rows["strategy"] == name]["metadata_json"]
        assert stored.str.contains(f'"{label}"').any() and stored.str.contains("rationale").any()


def test_agents_are_only_asked_about_bars_inside_the_period(tmp_path):
    transport = RoleTransport()
    cfg = agents_config(tmp_path, {"qwen_momentum": 1.0})
    BacktestEngine(cfg, SyntheticProvider(seed=3),
                   strategies=strategies_for(cfg, llm_provider=provider(transport))).run(START, END)
    bars_in_period = int((END - START) / timedelta(hours=1))
    decision_bars = bars_in_period // 6
    assert len(transport.requests) == 2 * decision_bars  # two symbols, nothing from the warm-up history


def test_record_then_replay_with_a_portfolio_aware_agent(tmp_path):
    cfg = agents_config(tmp_path)
    recorded_transport = RoleTransport()
    recorded = BacktestEngine(cfg, SyntheticProvider(seed=8),
                              strategies=strategies_for(cfg, llm_provider=provider(recorded_transport))).run(START, END)
    replay_cfg = agents_config(tmp_path, mode="replay")
    replay_transport = RoleTransport()
    replayed = BacktestEngine(replay_cfg, SyntheticProvider(seed=8),
                              strategies=strategies_for(replay_cfg, llm_provider=provider(replay_transport))
                              ).run(START, END)
    assert recorded_transport.requests and replay_transport.requests == []
    assert replayed.fills == recorded.fills and replayed.metrics == recorded.metrics
    assert [s.metadata.get("risk_state") for s in replayed.signals if s.strategy == "qwen_risk"] == [
        s.metadata.get("risk_state") for s in recorded.signals if s.strategy == "qwen_risk"]


def test_baseline_results_do_not_change(tmp_path):
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
    with_zero_weight_agents = AppConfig().with_overrides({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
    a = BacktestEngine(cfg, SyntheticProvider(seed=3)).run(START, END)
    b = BacktestEngine(with_zero_weight_agents, SyntheticProvider(seed=3)).run(START, END)
    assert a.fills == b.fills and a.metrics == b.metrics
