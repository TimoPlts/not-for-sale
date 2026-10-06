"""Stage 14C: answer-quality diagnostics for the AI agents."""

import hashlib
import json

import pytest

from test_specialists import END, START, RoleTransport, agents_config, feature, provider
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.research import evaluate_run, format_agent_eval
from trading_lab.research.agent_eval import error_kind
from trading_lab.storage import SQLiteStore

GOOD_LABELS = {
    "Trend Agent": ("regime", {"BUY": "bullish_trend", "SELL": "bearish_trend", "HOLD": "sideways"}),
    "Momentum Agent": ("momentum_state", {"BUY": "strengthening", "SELL": "weakening", "HOLD": "neutral"}),
    "Risk/Regime Agent": ("risk_state", {"BUY": "low", "SELL": "high", "HOLD": "moderate"}),
}


def sound(role, user):
    """A well-behaved agent: labels match votes, confidence and rationale vary with the data."""
    names = ("close_vs_ema_50_pct", "return_acceleration_pct", "return_24_bars_pct")
    score = next((v for v in (feature(user, n) for n in names) if v is not None), 0.0)
    direction = "HOLD" if abs(score) < 0.05 else ("BUY" if score > 0 else "SELL")
    label, values = GOOD_LABELS[role]
    confidence = 0.0 if direction == "HOLD" else round(min(0.95, 0.3 + abs(score) / 4), 2)
    return {"direction": direction, "confidence": confidence,
            "rationale": f"{role}: the score is {score:+.4f}, so {direction}", label: values[direction]}


def evaluate(tmp_path, override=None, seed=3, **kwargs):
    cfg = agents_config(tmp_path)
    with SQLiteStore(":memory:") as store:
        run = BacktestEngine(cfg, SyntheticProvider(seed=seed), store=store,
                             llm_provider=provider(RoleTransport(override))).run(START, END)
        return evaluate_run(store, run.run_id, **kwargs)


def test_a_sound_agent_has_no_warnings(tmp_path):
    results = evaluate(tmp_path, sound)
    assert sorted(results) == ["qwen_momentum", "qwen_risk", "qwen_trend"]
    for ev in results.values():
        assert ev.decisions == ev.answered == 24  # 2 symbols x 12 decision bars
        assert ev.warnings == [], format_agent_eval(ev)
        assert ev.contradictions == [] and ev.calls == 24 and ev.failed_calls == 0
        assert ev.directions["BUY"] and ev.directions["SELL"]
        assert format_agent_eval(ev).endswith("no problems found")


def test_inconsistent_and_lazy_answers_are_flagged(tmp_path):
    # The plain fake answers with fixed labels, one confidence and one rationale per role.
    results = evaluate(tmp_path)
    trend, momentum, risk = results["qwen_trend"], results["qwen_momentum"], results["qwen_risk"]
    assert trend.labels == {"bullish_trend": 24}
    assert len(trend.contradictions) == trend.directions["SELL"] > 0  # SELL while calling it a bullish trend
    assert trend.contradictions[0].endswith("SELL with regime=bullish_trend")
    assert momentum.hold_label_votes == momentum.directional > 0  # BUY/SELL on "neutral"
    assert risk.directions == {"HOLD": 24}  # its prompt has none of the features the fake looks at
    assert "almost always HOLD (100%): the agent barely takes part" in risk.warnings
    for ev in results.values():
        assert "the same rationale in 100% of answers" in ev.warnings
    for ev in (trend, momentum):
        assert "confidence barely varies (0.8)" in ev.warnings
    assert any("contradict the agent's own regime" in w for w in trend.warnings)
    assert any("calls for HOLD" in w for w in momentum.warnings)
    assert "contradiction: " in format_agent_eval(trend)


def test_one_sided_holding_and_empty_answers(tmp_path):
    def always_buy(role, user):
        answer = sound(role, user)
        label, values = GOOD_LABELS[role]
        return {**answer, "direction": "BUY", "confidence": 0.0, "rationale": "", label: values["BUY"]}

    for ev in evaluate(tmp_path / "a", always_buy).values():
        text = "\n".join(ev.warnings)
        assert "one-sided: 100% of its BUY/SELL votes are BUY" in text
        assert "24 BUY/SELL vote(s) with confidence 0" in text and "24 empty or very short" in text

    def always_hold(role, user):
        label, values = GOOD_LABELS[role]
        return {"direction": "HOLD", "confidence": 0.0, "rationale": f"{role} holds, {user[-40:]}",
                label: values["HOLD"]}

    for ev in evaluate(tmp_path / "b", always_hold).values():
        assert ev.warnings == ["almost always HOLD (100%): the agent barely takes part"]


def test_unusable_answers_are_counted_by_type(tmp_path):
    def flaky(role, user):
        return "not json at all" if "ETH/USDT" in user else sound(role, user)

    for ev in evaluate(tmp_path, flaky).values():
        assert ev.decisions == 24 and ev.answered == 12 and sum(ev.errors.values()) == 12
        assert any("no usable answer" in w for w in ev.warnings)
        assert "no usable answer 12" in format_agent_eval(ev)


def test_small_samples_are_not_judged_on_spread(tmp_path):
    results = evaluate(tmp_path, min_answers=100)
    assert not any("barely varies" in w or "same rationale" in w for ev in results.values() for w in ev.warnings)
    assert results["qwen_trend"].contradictions  # consistency is checked on any sample


def test_error_kinds():
    assert error_kind("TransportTimeout: no answer after 30s") == "TransportTimeout"
    assert error_kind("not in cache (replay mode)") == "not in cache (replay mode)"


def test_runs_without_agents_and_unknown_runs():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    with SQLiteStore(":memory:") as store:
        run = BacktestEngine(cfg, SyntheticProvider(seed=1), store=store).run(START, END)
        assert evaluate_run(store, run.run_id) == {}
        with pytest.raises(ValueError, match="unknown run"):
            evaluate_run(store, "nope")


def test_cli(tmp_path, capsys):
    db = tmp_path / "e.db"
    cfg = agents_config(tmp_path)
    with SQLiteStore(db) as store:
        run_id = BacktestEngine(cfg, SyntheticProvider(seed=3), store=store,
                                llm_provider=provider(RoleTransport())).run(START, END).run_id
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    assert main(["--db", str(db), "agent-eval"]) == 0  # the latest run
    out = capsys.readouterr().out
    assert f"Run {run_id}: answer quality of 3 agent(s)" in out and "Agent: qwen_trend" in out
    assert main(["--db", str(db), "agent-eval", run_id, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["agents"]["qwen_trend"]["decisions"] == 24 and data["agents"]["qwen_trend"]["warnings"]
    assert main(["--db", str(db), "agent-eval", "nope"]) == 1
    assert main(["--db", str(tmp_path / "none.db"), "agent-eval"]) == 1
    assert hashlib.sha256(db.read_bytes()).hexdigest() == digest
