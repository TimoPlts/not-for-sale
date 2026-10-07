"""Stage 17C: performance by market regime."""

import hashlib
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.research import format_regimes, regimes_for_run
from trading_lab.research.regimes import WARMUP, classify, market_index
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
H = timedelta(hours=1)
START = datetime(2024, 2, 1, tzinfo=UTC)
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


def series(values):
    return pd.Series(values, index=pd.date_range(START, periods=len(values), freq="h"), dtype="float64")


def test_classify_trends_and_warmup():
    up = np.linspace(100, 200, 100)
    down = np.linspace(200, 100, 100)
    labels = classify(series(np.concatenate([up, down])), trend_bars=20, slope_bars=5, vol_bars=10)
    assert (labels["trend"].iloc[:19] == WARMUP).all() and (labels["volatility"].iloc[:10] == WARMUP).all()
    assert (labels["trend"].iloc[30:100] == "up").all()
    assert (labels["trend"].iloc[130:] == "down").all()
    assert (labels["trend"].iloc[100:110] == "sideways").any()  # the turn is neither


def test_classification_is_causal():
    rng = np.random.default_rng(1)
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    full = classify(series(values), trend_bars=30, vol_bars=10)["trend"]
    for t in (60, 150, 299):
        assert classify(series(values[: t + 1]), trend_bars=30, vol_bars=10)["trend"].iloc[-1] == full.iloc[t]


def test_market_index_is_the_equal_weight_average():
    a, b = series([100, 110, 121]), series([10, 9, 9.9])
    index = market_index({"A": a, "B": b})
    assert index.tolist() == pytest.approx([1.0, 1.0, 1.1])  # +10% and -10%, then +10% and +10%


def test_validation():
    with pytest.raises(ValueError):
        classify(series([1.0, 2.0]), trend_bars=1)


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    db = tmp_path_factory.mktemp("regimes") / "r.db"
    with SQLiteStore(db) as store:
        result = BacktestEngine(CFG, SyntheticProvider(seed=7), store=store).run(START, START + 600 * H)
    return db, result


def test_regimes_add_up_to_the_run(stored):
    db, result = stored
    with SQLiteStore(db, readonly=True) as store:
        r = regimes_for_run(store, result.run_id, trend_bars=50, vol_bars=24)
    assert r.total_bars == 600 and r.warmup_bars == 49
    known = r.total_bars - r.warmup_bars
    for group in (r.by_trend, r.by_vol, r.combined):
        assert sum(s.bars for s in group) == known
        assert sum(s.time_share for s in group) == pytest.approx(1.0)
        assert sum(s.trades for s in group) + r.warmup_trades == len(result.trades)
        first_known = result.equity_curve.index[r.warmup_bars]
        assert sum(s.pnl for s in group) == pytest.approx(
            sum(t.pnl for t in result.trades if t.opened_at >= first_known))
    # Compounding every regime's bars gives the run's return after the warm-up.
    curve = result.equity_curve["equity"]
    after = curve.iloc[-1] / curve.iloc[r.warmup_bars - 1]
    assert np.prod([1 + s.strategy_return for s in r.by_trend]) == pytest.approx(after)
    assert np.prod([1 + s.strategy_return for s in r.combined]) == pytest.approx(after)
    text = format_regimes(r)
    assert "up / calm" in text and "Best:" in text and "warm-up" in text
    assert json.loads(json.dumps(r.to_dict()))["total_bars"] == 600


def test_short_and_unknown_runs(tmp_path):
    with SQLiteStore(tmp_path / "s.db") as store:
        run = BacktestEngine(CFG, SyntheticProvider(seed=7), store=store).run(START, START + 30 * H)
        r = regimes_for_run(store, run.run_id)
        assert "shorter than the 50-bar warm-up" in format_regimes(r)
        with pytest.raises(ValueError, match="unknown run"):
            regimes_for_run(store, "nope")


def test_cli(stored, capsys, tmp_path):
    db, result = stored
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    assert main(["--db", str(db), "regimes"]) == 0
    assert f"Run {result.run_id}: 600 bars" in capsys.readouterr().out
    assert main(["--db", str(db), "regimes", result.run_id, "--json", "--trend-bars", "24"]) == 0
    assert json.loads(capsys.readouterr().out)["trend_bars"] == 24
    assert main(["--db", str(db), "regimes", "nope"]) == 1
    assert main(["--db", str(tmp_path / "none.db"), "regimes"]) == 1
    assert main(["--db", str(db), "regimes", "--trend-bars", "1"]) == 1
    assert hashlib.sha256(db.read_bytes()).hexdigest() == digest
