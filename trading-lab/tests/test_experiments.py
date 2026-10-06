"""Stage 9D: baseline versus AI experiments, sweeps and walk-forward with agents."""

import json
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from test_specialists import ENV, RoleTransport, provider
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data import SyntheticProvider
from trading_lab.research import (
    VARIANTS,
    apply_params,
    run_experiment,
    run_sweep,
    variant_config,
    walk_forward,
)
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START, END = datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 4, tzinfo=UTC)
AGENTS = ("qwen_trend", "qwen_momentum", "qwen_risk")


def base_config(tmp_path, mode="record"):
    return AppConfig().with_overrides({
        "market": {"symbols": ["BTC/USDT", "ETH/USDT"]},
        "strategies": {name: {"weight": 0.0, "decision_interval": 6} for name in AGENTS},
        "agents": {"mode": mode, "cache_path": str(tmp_path / "agents.db")},
    })


def voters(cfg):
    return {s.name: s.weight for s in cfg.enabled_strategies}


def test_variants_only_change_weights(tmp_path):
    cfg = base_config(tmp_path).with_overrides({"strategies": {"qwen_trend": {"weight": 2.0}}})
    assert voters(variant_config(cfg, "baseline")) == {"rsi": 1.0, "macd": 1.0, "bollinger": 1.0}
    assert voters(variant_config(cfg, "trend")) == {"rsi": 1.0, "macd": 1.0, "bollinger": 1.0, "qwen_trend": 2.0}
    assert set(voters(variant_config(cfg, "all_agents"))) == {"rsi", "macd", "bollinger", *AGENTS}
    assert voters(variant_config(cfg, "ai_only")) == {"qwen_trend": 2.0, "qwen_momentum": 1.0, "qwen_risk": 1.0}
    for name in VARIANTS:
        v = variant_config(cfg, name)
        assert (v.execution, v.risk, v.voting, v.market) == (cfg.execution, cfg.risk, cfg.voting, cfg.market)
    disabled = cfg.with_overrides({"strategies": {"qwen_momentum": {"enabled": False}}})
    assert "qwen_momentum" in voters(variant_config(disabled, "trend_momentum"))
    with pytest.raises(ConfigError, match="unknown variant"):
        variant_config(cfg, "magic")


def test_experiment_shares_answers_between_variants(tmp_path):
    transport = RoleTransport()
    cfg = base_config(tmp_path)
    rows = run_experiment(cfg, SyntheticProvider(seed=3), START, END,
                          ["baseline", "trend", "trend_momentum", "all_agents", "ai_only"],
                          llm_provider=provider(transport))
    assert [r.variant for r in rows] == ["baseline", "trend", "trend_momentum", "all_agents", "ai_only"]
    plain = BacktestEngine(variant_config(cfg, "baseline"), SyntheticProvider(seed=3)).run(START, END)
    assert rows[0].metrics == plain.metrics
    roles = Counter(transport.roles())
    decisions = 2 * 12  # two symbols, 72 bars / decision_interval 6
    assert roles["Trend Agent"] == roles["Momentum Agent"] == decisions  # asked once, reused by later variants
    assert roles["Risk/Regime Agent"] >= decisions  # portfolio-aware: asked again where trades differ
    assert {s.strategy for s in rows[4].signals} == {*AGENTS, "ensemble"}
    assert len({r.config_fingerprint for r in rows}) == 5


def test_replay_reproduces_every_variant_offline(tmp_path):
    recorded = run_experiment(base_config(tmp_path), SyntheticProvider(seed=4), START, END,
                              ["trend", "all_agents"], llm_provider=provider(RoleTransport()))
    transport = RoleTransport()
    replayed = run_experiment(base_config(tmp_path, mode="replay"), SyntheticProvider(seed=4), START, END,
                              ["trend", "all_agents"], llm_provider=provider(transport))
    assert transport.requests == []
    assert [r.metrics for r in replayed] == [r.metrics for r in recorded]


def test_sweep_over_agent_weights(tmp_path):
    transport = RoleTransport()
    grid = {"strategies.qwen_trend.weight": [0, 1], "strategies.qwen_momentum.weight": [0, 1]}
    results = run_sweep(base_config(tmp_path), SyntheticProvider(seed=3), START, END, grid,
                        llm_provider=provider(transport))
    assert len(results) == 4
    off = next(r for r in results if r.params == {"strategies.qwen_momentum.weight": 0,
                                                  "strategies.qwen_trend.weight": 0})
    baseline = BacktestEngine(variant_config(base_config(tmp_path), "baseline"), SyntheticProvider(seed=3)).run(START, END)
    assert off.metrics == baseline.metrics
    roles = Counter(transport.roles())
    assert roles == {"Trend Agent": 24, "Momentum Agent": 24}  # each bar asked once across all combinations


def test_walkforward_with_agents_never_asks_twice(tmp_path):
    transport = RoleTransport()
    grid = {"strategies.qwen_trend.weight": [0, 1]}
    result = walk_forward(base_config(tmp_path), SyntheticProvider(seed=3), datetime(2024, 2, 1, tzinfo=UTC),
                          datetime(2024, 2, 9, tzinfo=UTC), grid, train=timedelta(days=4), test=timedelta(days=2),
                          llm_provider=provider(transport))
    assert len(result.folds) == 2
    prompts = [user for system, user in transport.requests]
    assert prompts and len(prompts) == len(set(prompts))  # overlapping windows reuse cached answers


@pytest.fixture
def fake_qwen(monkeypatch, tmp_path):
    transport = RoleTransport()
    monkeypatch.setattr("trading_lab.llm.openai_compat.UrllibTransport.post", transport.post)
    for var, value in ENV.items():
        monkeypatch.setenv(var, value)
    monkeypatch.chdir(tmp_path)  # agent cache under tmp_path/data/
    return transport


def test_experiment_command(tmp_path, capsys, fake_qwen):
    db, out_json = tmp_path / "h.db", tmp_path / "exp.json"
    args = ["--db", str(db), "experiment", "--synthetic", "2", "--symbols", "BTC/USDT",
            "--start", "2024-02-01", "--end", "2024-02-03", "--variants", "baseline,trend,ai_only",
            "--save", "--export", str(out_json)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "baseline" in out and "ai_only" in out and "the 3 Qwen agents only" in out
    calls = len(fake_qwen.requests)
    assert calls > 0
    with SQLiteStore(db) as store:
        saved = store.list_research_results("experiment")
        assert [r["variant"] for r in saved[0]["payload"]["rows"]] == ["baseline", "trend", "ai_only"]
        assert len(store.list_runs()) == 3
    assert [r["variant"] for r in json.loads(out_json.read_text())] == ["baseline", "trend", "ai_only"]

    assert main(["--agent-mode", "replay", *args[:-3]]) == 0  # fully offline replay of the same experiment
    assert len(fake_qwen.requests) == calls


def test_walkforward_experiment_command(capsys, fake_qwen):
    assert main(["experiment", "--synthetic", "2", "--symbols", "BTC/USDT", "--start", "2024-02-01",
                 "--end", "2024-02-09", "--variants", "baseline,trend", "--walkforward",
                 "--train-days", "4", "--test-days", "2", "--param", "voting.min_agreeing=1,2"]) == 0
    out = capsys.readouterr().out
    assert "OOS return" in out and "baseline" in out and "trend" in out


def test_agent_sweeps_fail_fast_without_credentials(tmp_path, capsys, monkeypatch):
    for var in ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    code = main(["sweep", "--synthetic", "1", "--days", "5", "--param", "strategies.qwen_trend.weight=0,1"])
    captured = capsys.readouterr()
    assert code == 1 and "QWEN_API_URL" in captured.err and "[1/2]" not in captured.out
    assert main(["sweep", "--synthetic", "1", "--days", "5", "--param", "strategies.qwen_trend.weight=0"]) == 0


def test_apply_params_can_switch_an_agent_on(tmp_path):
    cfg = apply_params(base_config(tmp_path), {"strategies.qwen_risk.weight": 1.5})
    assert voters(cfg)["qwen_risk"] == 1.5
