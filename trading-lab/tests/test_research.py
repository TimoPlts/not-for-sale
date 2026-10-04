"""Research tool tests: grids, parameter application, sweeps, walk-forward, CLI."""

from datetime import datetime, timedelta, timezone

import pytest

from conftest import PROJECT_ROOT
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data import SyntheticProvider
from trading_lab.metrics import compute_metrics
from trading_lab.research import (
    MemoizedProvider,
    apply_params,
    expand_grid,
    make_folds,
    rank_key,
    run_sweep,
    walk_forward,
)

UTC = timezone.utc
START, END = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 21, tzinfo=UTC)
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
GRID = {"voting.min_agreeing": [1, 2], "strategies.rsi.period": [7, 14]}


def test_expand_grid_is_deterministic_cartesian_product():
    combos = expand_grid({"b": [1, 2], "a": ["x", "y", "z"]})
    assert len(combos) == 6
    assert combos[0] == {"a": "x", "b": 1} and combos[-1] == {"a": "z", "b": 2}
    assert expand_grid({}) == [{}]
    with pytest.raises(ConfigError):
        expand_grid({"a": []})
    with pytest.raises(ConfigError):
        expand_grid({"a": "abc"})


def test_apply_params_deep_merges_and_validates():
    cfg = apply_params(CFG, {"strategies.rsi.period": 7, "risk.stop_loss_pct": 0.03})
    rsi = next(s for s in cfg.strategies if s.name == "rsi")
    assert rsi.params == {"period": 7, "oversold": 30.0, "overbought": 70.0}  # others kept
    assert cfg.risk.stop_loss_pct == 0.03 and cfg.risk.max_position_pct == 0.25
    enabled = apply_params(CFG, {"strategies.trend_analyst.weight": 2.0})
    assert "trend_analyst" in [s.name for s in enabled.enabled_strategies]
    with pytest.raises(ConfigError):
        apply_params(CFG, {"nonsense.key": 1})
    with pytest.raises(ConfigError):
        apply_params(CFG, {"risk.stop_loss_pct": 5.0})
    with pytest.raises(ConfigError):
        apply_params(CFG, {"risk": 1})


def test_rank_key_orders_best_first_and_undefined_last():
    good = compute_metrics([100, 110, 120], [], "1h")
    bad = compute_metrics([100, 90, 95], [], "1h")
    flat = compute_metrics([100, 100, 100], [], "1h")  # sharpe undefined
    ordered = sorted([flat, bad, good], key=rank_key("sharpe_ratio"))
    assert ordered == [good, bad, flat]
    assert sorted([good, bad], key=rank_key("max_drawdown"))[0] is good  # lower drawdown first
    with pytest.raises(ConfigError):
        rank_key("not_a_metric")(good)


def test_sweep_runs_every_combination_once_per_dataset():
    provider = MemoizedProvider(SyntheticProvider(seed=8))
    results = run_sweep(CFG, provider, START, END, GRID)
    assert len(results) == 4
    assert len({r.config_fingerprint for r in results}) == 4
    assert provider.fetches == 2  # data loaded once per symbol, reused by all 4 backtests
    sharpes = [r.metrics.sharpe_ratio for r in results]
    assert sharpes == sorted(sharpes, reverse=True)
    # Any row can be reproduced exactly from its parameters.
    best = results[0]
    again = BacktestEngine(apply_params(CFG, best.params), SyntheticProvider(seed=8)).run(START, END)
    assert again.metrics == best.metrics
    assert best.benchmark is not None


def test_sweep_can_store_runs(tmp_path):
    from trading_lab.storage import SQLiteStore

    with SQLiteStore(tmp_path / "s.db") as store:
        results = run_sweep(CFG, SyntheticProvider(seed=8), START, END, {"voting.min_agreeing": [1, 2]},
                            store=store, label="sweep-test")
        assert all(r.run_id for r in results)
        assert "sweep-test" in store.get_run(results[0].run_id)["notes"]


def test_make_folds():
    folds = make_folds(START, START + timedelta(days=100), timedelta(days=30), timedelta(days=20))
    assert len(folds) == 3
    assert folds[0] == (START, START + timedelta(days=30), START + timedelta(days=30), START + timedelta(days=50))
    for (_, tr_end, te_start, te_end), nxt in zip(folds, folds[1:]):
        assert tr_end == te_start and nxt[2] == te_end  # test windows are contiguous
    with pytest.raises(ConfigError):
        make_folds(START, START + timedelta(days=10), timedelta(days=30), timedelta(days=20))


def test_walk_forward_picks_in_sample_best_and_tests_out_of_sample():
    grid = {"voting.min_agreeing": [1, 2]}
    end = START + timedelta(days=24)
    result = walk_forward(CFG, SyntheticProvider(seed=8), START, end, grid,
                          train=timedelta(days=10), test=timedelta(days=7))
    assert len(result.folds) == 2
    first = result.folds[0]
    expected_best = run_sweep(CFG, SyntheticProvider(seed=8), first.train_start, first.train_end, grid)[0]
    assert first.best_params == expected_best.params
    assert first.in_sample == expected_best.metrics
    oos = BacktestEngine(apply_params(CFG, first.best_params), SyntheticProvider(seed=8)).run(
        first.test_start, first.test_end
    )
    assert first.out_of_sample == oos.metrics
    combined = (1 + result.folds[0].out_of_sample.total_return) * (1 + result.folds[1].out_of_sample.total_return) - 1
    assert result.out_of_sample_return == pytest.approx(combined)
    assert result.benchmark_return is not None


def test_research_cli_commands(tmp_path, capsys):
    db = str(tmp_path / "r.db")
    base = ["--config", str(PROJECT_ROOT / "config" / "default.toml"), "--db", db]
    common = ["--synthetic", "8", "--symbols", "BTC/USDT", "--start", "2024-01-01"]
    assert main([*base, "sweep", *common, "--end", "2024-01-15", "--param", "voting.min_agreeing=1,2",
                 "--save", "--export", str(tmp_path / "sweep.csv")]) == 0
    out = capsys.readouterr().out
    assert "Buy & hold" in out and "min_agreeing=1" in out
    assert (tmp_path / "sweep.csv").exists()

    grid = tmp_path / "grid.toml"
    grid.write_text('[grid]\n"voting.min_agreeing" = [1, 2]\n')
    assert main([*base, "walkforward", *common, "--end", "2024-01-25", "--grid", str(grid),
                 "--train-days", "10", "--test-days", "5"]) == 0
    assert "Combined out-of-sample return" in capsys.readouterr().out

    from trading_lab.storage import SQLiteStore

    with SQLiteStore(db) as store:
        ids = [r["run_id"] for r in store.list_runs(2)]
    assert main([*base, "compare", *ids]) == 0
    out = capsys.readouterr().out
    assert ids[0] in out and "Sharpe" in out and "Buy & hold" in out

    assert main([*base, "sweep", *common, "--end", "2024-01-15"]) == 1
    assert "--param" in capsys.readouterr().err
    assert main([*base, "sweep", *common, "--end", "2024-01-15", "--param", "risk.nope=1"]) == 1
