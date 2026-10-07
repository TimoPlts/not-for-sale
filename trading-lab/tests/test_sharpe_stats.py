"""Stage 23A: probabilistic and deflated Sharpe ratios."""

import math
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

import numpy as np
import pytest

from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.metrics import compute_metrics
from trading_lab.metrics.sharpe import (
    EULER_GAMMA,
    deflated_sharpe,
    expected_max_sharpe,
    per_bar_sharpe,
    probabilistic_sharpe,
)
from trading_lab.research import run_sweep
from trading_lab.research.sweep import deflated_sharpe_of_best

UTC = timezone.utc
PHI = NormalDist()


def series(seed=0, n=400, mean=0.001, sd=0.01):
    return np.random.default_rng(seed).normal(mean, sd, n)


def test_psr_matches_the_formula_by_hand():
    r = np.array([0.01, 0.02, -0.01, 0.03, -0.005, 0.015])
    sr = r.mean() / r.std(ddof=1)
    c = r - r.mean()
    skew = (c**3).mean() / (c**2).mean() ** 1.5
    kurt = (c**4).mean() / (c**2).mean() ** 2
    expected = PHI.cdf(sr * math.sqrt(len(r) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2))
    assert probabilistic_sharpe(r) == pytest.approx(expected)
    assert probabilistic_sharpe(r, threshold=sr) == pytest.approx(0.5)
    assert per_bar_sharpe(r) == pytest.approx(sr)


def test_psr_properties():
    r = series()
    assert probabilistic_sharpe(np.tile(r, 4)) > probabilistic_sharpe(r)  # more evidence, more certainty
    assert probabilistic_sharpe(-r) < 0.5 < probabilistic_sharpe(r)
    # the same mean and volatility, but crash-prone (negative skew) is less convincing than lottery-like
    skewed = np.where(np.arange(400) % 20 == 0, -0.05, 0.004)
    mirrored = 2 * skewed.mean() - skewed
    assert per_bar_sharpe(skewed) == pytest.approx(per_bar_sharpe(mirrored))
    assert probabilistic_sharpe(skewed) < probabilistic_sharpe(mirrored)
    for undefined in ([], [0.01], [0.01, 0.02], [0.01] * 10):
        assert probabilistic_sharpe(undefined) is None


def test_expected_max_sharpe():
    assert expected_max_sharpe(1, 1.0) == 0.0 and expected_max_sharpe(50, 0.0) == 0.0
    n = 100
    value = (1 - EULER_GAMMA) * PHI.inv_cdf(1 - 1 / n) + EULER_GAMMA * PHI.inv_cdf(1 - 1 / (n * math.e))
    assert expected_max_sharpe(n, 1.0) == pytest.approx(value) == pytest.approx(2.53, abs=0.01)
    simulated = np.random.default_rng(1).standard_normal((4000, n)).max(axis=1).mean()
    assert expected_max_sharpe(n, 1.0) == pytest.approx(simulated, abs=0.06)
    assert expected_max_sharpe(n, 0.25) == pytest.approx(value / 2)
    assert expected_max_sharpe(10, 1.0) < expected_max_sharpe(1000, 1.0)


def test_deflated_sharpe():
    r = series()
    assert deflated_sharpe(r, [per_bar_sharpe(r)]) == pytest.approx(probabilistic_sharpe(r))
    trials = [per_bar_sharpe(series(seed=s, mean=0.0)) for s in range(1, 50)]
    deflated = deflated_sharpe(r, [per_bar_sharpe(r), *trials, None])
    assert deflated < probabilistic_sharpe(r)
    assert deflated < deflated_sharpe(r, [per_bar_sharpe(r), *trials[:5]])  # more trials, higher bar
    # picking the best of 50 pure-noise strategies: its PSR looks fine, its DSR does not
    noise = [series(seed=s, mean=0.0) for s in range(100, 150)]
    best = max(noise, key=per_bar_sharpe)
    assert probabilistic_sharpe(best) > 0.9
    assert deflated_sharpe(best, [per_bar_sharpe(x) for x in noise]) < 0.5


def test_in_the_metrics():
    r = series()
    equity = [1000.0]
    for x in r:
        equity.append(equity[-1] * (1 + x))
    m = compute_metrics(equity, [], "1h")
    returns = np.asarray(equity[1:]) / np.asarray(equity[:-1]) - 1
    assert m.probabilistic_sharpe == pytest.approx(probabilistic_sharpe(returns))
    assert "Prob. Sharpe > 0" in m.format_table() and "probabilistic_sharpe" in m.to_dict()
    assert compute_metrics([1000.0, 1000.0], [], "1h").probabilistic_sharpe is None


def test_sweep_deflated_sharpe(tmp_path, capsys):
    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    start = datetime(2024, 2, 1, tzinfo=UTC)
    grid = {"strategies.rsi.period": [7, 14, 21], "voting.min_agreeing": [1, 2]}
    results = run_sweep(cfg, SyntheticProvider(seed=2), start, start + timedelta(days=6), grid)
    for r in results:
        assert len(r.returns) == r.metrics.num_bars
        assert np.prod([1 + x for x in r.returns]) - 1 == pytest.approx(r.metrics.total_return)
        assert r.metrics.probabilistic_sharpe == pytest.approx(probabilistic_sharpe(r.returns))
    dsr = deflated_sharpe_of_best(results)
    assert dsr == pytest.approx(deflated_sharpe(results[0].returns, [r.per_bar_sharpe for r in results]))
    assert dsr <= results[0].metrics.probabilistic_sharpe
    assert deflated_sharpe_of_best([]) is None
    assert main(["--db", str(tmp_path / "x.db"), "sweep", "--synthetic", "2", "--symbols", "BTC/USDT",
                 "--start", "2024-02-01", "--end", "2024-02-07", "--param", "strategies.rsi.period=7,14,21"]) == 0
    out = capsys.readouterr().out
    assert "Deflated Sharpe of row 1:" in out and "luckiest of 3 settings" in out


def test_report_and_compare_show_it(tmp_path, capsys):
    from trading_lab.backtest import BacktestEngine
    from trading_lab.storage import SQLiteStore

    cfg = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT"]}})
    start = datetime(2024, 2, 1, tzinfo=UTC)
    with SQLiteStore(tmp_path / "h.db") as store:
        result = BacktestEngine(cfg, SyntheticProvider(seed=2), store=store).run(start, start + timedelta(days=4))
    psr = result.metrics.probabilistic_sharpe
    assert psr is not None
    assert main(["--db", str(tmp_path / "h.db"), "compare", result.run_id]) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("Prob. Sharpe>0"))
    assert line.split()[-1] == f"{psr:.0%}"
    html = tmp_path / "r.html"
    assert main(["--db", str(tmp_path / "h.db"), "report", result.run_id, "--html", str(html)]) == 0
    assert "Prob. Sharpe &gt; 0" in html.read_text() or "Prob. Sharpe > 0" in html.read_text()
