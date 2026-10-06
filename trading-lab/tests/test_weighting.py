"""Stage 12B: AI agent weights from their out-of-sample record (no look-ahead)."""

from datetime import datetime, timedelta, timezone

import pytest

from test_experiments import fake_qwen  # noqa: F401  (fixture)
from test_specialists import RoleTransport, agents_config, provider
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.data import SyntheticProvider
from trading_lab.research import (
    Attribution,
    WeightingRule,
    adaptive_weights,
    apply_params,
    attribute_result,
    walk_forward,
)
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 13, tzinfo=UTC)


def record(name, measured, correct, agent=True):
    return Attribution(name, agent, 4, measured=measured, correct=correct)


def test_weight_rule():
    attributions = {
        "good": record("good", 100, 60),       # 60% -> x2.0
        "ok": record("ok", 100, 55),           # 55% -> x1.5
        "bad": record("bad", 100, 45),         # 45% -> x0.5
        "awful": record("awful", 100, 30),     # 30% -> off
        "unsure": record("unsure", 10, 9),     # too few votes -> unchanged
        "rsi": record("rsi", 500, 400, agent=False),  # deterministic: never re-weighted
    }
    current = {"good": 1.5, "ok": 1.0, "bad": 1.0, "awful": 1.0, "unsure": 1.2, "rsi": 1.0}
    weights = adaptive_weights(attributions, current)
    assert weights == {"good": 2.0, "ok": 1.5, "bad": 0.5, "awful": 0.0, "unsure": 1.2}  # good capped at 2.0
    floored = adaptive_weights(attributions, current, WeightingRule(min_weight=0.25, max_weight=5.0))
    assert floored["awful"] == 0.25 and floored["good"] == 3.0


def test_never_leaves_the_ensemble_without_voters():
    attributions = {"a": record("a", 100, 10), "b": record("b", 100, 20)}
    assert adaptive_weights(attributions, {"a": 1.0, "b": 1.0}) == {"a": 1.0, "b": 1.0}
    with pytest.raises(ValueError):
        WeightingRule(min_weight=3.0, max_weight=2.0)


class Perturbed(SyntheticProvider):
    """The same market until ``cutoff``, a different one after it."""

    def __init__(self, cutoff, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff

    def fetch_ohlcv(self, symbol, timeframe, since, until=None):
        frame = super().fetch_ohlcv(symbol, timeframe, since, until).copy()
        later = frame.index >= self.cutoff
        factors = [1.03 if i % 2 else 0.97 for i in range(int(later.sum()))]
        for column in ("open", "high", "low", "close"):
            frame.loc[later, column] = frame.loc[later, column] * factors
        return frame


def adaptive_run(tmp_path, market, name):
    cfg = agents_config(tmp_path / name, {"qwen_trend": 1.0, "qwen_momentum": 1.0})
    rule = WeightingRule(min_votes=3)
    result = walk_forward(cfg, market, START, END, {}, train=timedelta(days=4), test=timedelta(days=2),
                          llm_provider=provider(RoleTransport()), adapt_agent_weights=True, weighting=rule)
    return cfg, rule, result


def test_walk_forward_weights_come_from_the_training_window_only(tmp_path):
    cfg, rule, result = adaptive_run(tmp_path, SyntheticProvider(seed=3), "a")
    assert len(result.folds) == 4
    for fold in result.folds:
        assert set(fold.agent_weights) == {"qwen_trend", "qwen_momentum"}
        train = BacktestEngine(cfg, SyntheticProvider(seed=3), llm_provider=provider(RoleTransport())).run(
            fold.train_start, fold.train_end)
        expected = adaptive_weights(attribute_result(train, cfg, horizon=4), {
            s.name: s.weight for s in cfg.enabled_strategies}, rule)
        assert fold.agent_weights == expected
    assert any(w != 1.0 for f in result.folds for w in f.agent_weights.values())  # the record mattered

    first = result.folds[0]
    _, _, changed = adaptive_run(tmp_path, Perturbed(first.train_end, seed=3), "b")
    assert changed.folds[0].agent_weights == first.agent_weights  # the future cannot change past weights
    assert changed.folds[0].out_of_sample != first.out_of_sample  # ...although the test window differs


def test_fixed_weights_are_the_default(tmp_path):
    cfg = agents_config(tmp_path, {"qwen_trend": 1.0})
    result = walk_forward(cfg, SyntheticProvider(seed=3), START, END, {}, train=timedelta(days=4),
                          test=timedelta(days=2), llm_provider=provider(RoleTransport()))
    assert all(f.agent_weights == {} for f in result.folds)
    assert apply_params(cfg, {}).strategies == cfg.strategies


def test_cli(tmp_path, capsys, fake_qwen):  # noqa: F811
    common = ["--synthetic", "2", "--symbols", "BTC/USDT", "--start", "2024-02-01", "--end", "2024-02-11",
              "--train-days", "4", "--test-days", "2", "--weight-min-votes", "3"]
    assert main(["walkforward", *common, "--param", "strategies.qwen_trend.weight=1",
                 "--adaptive-weights"]) == 0
    assert "weights qwen_trend=" in capsys.readouterr().out
    assert main(["experiment", "--synthetic", "2", "--variants", "trend", "--adaptive-weights"]) == 1
    assert "needs --walkforward" in capsys.readouterr().err

    db = tmp_path / "h.db"
    cfg = agents_config(tmp_path, {"qwen_trend": 1.0, "qwen_risk": 1.0})
    with SQLiteStore(db) as store:
        BacktestEngine(cfg, SyntheticProvider(seed=3), store=store, llm_provider=provider(RoleTransport())).run(
            START, START + timedelta(days=4))
    assert main(["--db", str(db), "agent-weights", "--weight-min-votes", "3"]) == 0
    out = capsys.readouterr().out
    assert "[strategies.qwen_trend]\nweight = " in out and "[strategies.qwen_risk]" in out and "rsi" not in out
