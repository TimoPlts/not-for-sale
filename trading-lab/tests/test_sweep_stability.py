"""Stage 19B: sweep stability (neighbourhood-averaged metric)."""

import itertools
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.research import rank_by_stability, stability_scores
from trading_lab.research.sweep import SweepResult, params_key

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def base():
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    return BacktestEngine(cfg, SyntheticProvider(seed=2)).run(START, START + timedelta(days=2)).metrics


def results(base, grid, values, metric="sharpe_ratio"):
    keys = sorted(grid)
    combos = [dict(zip(keys, c)) for c in itertools.product(*(grid[k] for k in keys))]
    return [SweepResult(p, replace(base, **{metric: values(p)}), None, "f") for p in combos]


def test_a_lone_peak_scores_below_a_plateau(base):
    grid = {"x": [1, 2, 3, 4, 5, 6, 7]}
    sharpe = {1: 0.0, 2: 0.0, 3: 5.0, 4: 0.0, 5: 3.0, 6: 3.0, 7: 3.0}
    rs = results(base, grid, lambda p: sharpe[p["x"]])
    scores = stability_scores(rs, grid, "sharpe_ratio")
    assert scores[params_key({"x": 3})] == (pytest.approx(5 / 3), 3)
    assert scores[params_key({"x": 6})] == (pytest.approx(3.0), 3)
    assert scores[params_key({"x": 1})] == (pytest.approx(0.0), 2)  # an edge has one neighbour
    assert rank_by_stability(rs, grid, "sharpe_ratio")[0].params == {"x": 6}


def test_neighbours_in_two_dimensions_and_missing_values(base):
    grid = {"a": [1, 2, 3], "b": [0.1, 0.2]}
    rs = results(base, grid, lambda p: None if p == {"a": 1, "b": 0.1} else p["a"] + 10 * p["b"])
    scores = stability_scores(rs, grid, "sharpe_ratio")
    # (2, 0.1): itself 3, (1, 0.1) undefined, (3, 0.1) 4, (2, 0.2) 4 -> mean of 3 defined values
    assert scores[params_key({"a": 2, "b": 0.1})] == (pytest.approx(11 / 3), 3)
    assert scores[params_key({"a": 1, "b": 0.1})][1] == 2  # its own value is undefined
    assert rank_by_stability(rs, grid, "sharpe_ratio")[0].params == {"a": 3, "b": 0.2}


def test_lower_is_better_metrics(base):
    grid = {"x": [1, 2, 3]}
    rs = results(base, grid, lambda p: {1: 0.30, 2: 0.10, 3: 0.12}[p["x"]], metric="max_drawdown")
    assert rank_by_stability(rs, grid, "max_drawdown")[0].params == {"x": 3}  # (0.10+0.12)/2 is the lowest


def test_cli(tmp_path, capsys):
    out = tmp_path / "sweep.csv"
    args = ["--db", str(tmp_path / "s.db"), "sweep", "--synthetic", "3", "--symbols", "BTC/USDT", "--start",
            "2024-02-01", "--end", "2024-02-06", "--param", "strategies.rsi.period=7,14,21"]
    assert main([*args, "--rank", "stability", "--export", str(out)]) == 0
    text = capsys.readouterr().out
    assert "stable" in text and "(ranked by it)" in text
    frame = pd.read_csv(out)
    assert "stable" in frame.columns and frame["stable"].notna().all()
    assert list(frame["stable"]) == sorted(frame["stable"], reverse=True)
    assert main([*args]) == 0 and "--rank stability ranks by it" in capsys.readouterr().out
