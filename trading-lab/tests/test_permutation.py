"""Stage 19A: the permutation test (shuffled-candle markets)."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from test_specialists import agents_config
from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data import SyntheticProvider, candles_from_closes
from trading_lab.data.base import MarketDataProvider, normalize_ohlcv, slice_ohlcv
from trading_lab.research import format_permutation, permutation_test
from trading_lab.research.permutation import PermutationResult, PermutedProvider, permute_candles

UTC = timezone.utc
H = timedelta(hours=1)
T0 = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def candles():
    return SyntheticProvider(seed=5).fetch_ohlcv("BTC/USDT", "1h", T0, T0 + 400 * H)


def test_shuffling_keeps_shapes_drift_and_history(candles):
    start = T0 + 100 * H
    out = permute_candles(candles, start, seed=1)
    assert out.index.equals(candles.index)
    assert out.iloc[:100].equals(candles.iloc[:100])  # warm-up history stays real
    assert out["close"].iloc[-1] == pytest.approx(candles["close"].iloc[-1], rel=1e-9)  # same total move
    assert not np.allclose(out["close"].iloc[100:-1], candles["close"].iloc[100:-1])
    normalize_ohlcv(out)  # still valid candles (high/low consistent, positive)

    def shapes(frame):
        prev = frame["close"].shift(1).iloc[100:]
        return sorted(zip(*(np.round(np.log(frame[k].iloc[100:] / prev), 12) for k in ("open", "high", "low", "close")),
                          frame["volume"].iloc[100:]))
    assert shapes(out) == shapes(candles)  # the same candles, in another order
    assert permute_candles(candles, start, 1).equals(out) and not permute_candles(candles, start, 2).equals(out)


def test_symbols_share_the_order_and_edge_cases(candles):
    other = SyntheticProvider(seed=6).fetch_ohlcv("ETH/USDT", "1h", T0, T0 + 400 * H)
    start = T0 + 100 * H

    def order(original, shuffled):  # which original candle sits at each position
        rc = np.log(original["close"] / original["close"].shift(1)).iloc[100:].to_numpy()
        sc = np.log(shuffled["close"] / shuffled["close"].shift(1)).iloc[100:].to_numpy()
        return [int(np.argmin(np.abs(rc - v))) for v in sc]
    assert order(candles, permute_candles(candles, start, 3)) == order(other, permute_candles(other, start, 3))
    assert permute_candles(candles, T0 + 399 * H, 3).equals(candles)  # one candle: nothing to shuffle
    assert permute_candles(candles.iloc[:0], start, 3).empty
    from_first = permute_candles(candles, T0, 3)
    assert from_first["close"].iloc[-1] == pytest.approx(candles["close"].iloc[-1], rel=1e-9)
    assert PermutedProvider(SyntheticProvider(seed=5), start, 4).name == "synthetic-5-permuted-4"


class Trending(MarketDataProvider):
    """Autocorrelated returns: a market a trend follower can genuinely exploit."""

    def __init__(self):
        rng = np.random.default_rng(3)
        r, e = np.zeros(1800), rng.normal(0, 0.01, 1800)
        for i in range(1, len(r)):
            r[i] = 0.6 * r[i - 1] + e[i]
        self.frame = candles_from_closes(100 * np.exp(np.cumsum(r)), start=T0)

    @property
    def name(self):
        return "trending"

    def fetch_ohlcv(self, symbol, timeframe, since, until=None):
        return slice_ohlcv(self.frame, since, until)


FOLLOWER = AppConfig.from_mapping({
    "market": {"symbols": ["BTC/USDT"]}, "execution": {"fee_rate": 0.0, "slippage_bps": 0.0},
    "risk": {"allow_short": True, "max_position_pct": 1.0, "risk_per_trade_pct": 1.0, "stop_loss_pct": 0.2},
    "strategies": {"rsi": {"enabled": False}, "macd": {"enabled": False}, "bollinger": {"enabled": False},
                   "ma_cross": {"fast": 2, "slow": 6, "signal_on": "state"}},
})


def test_real_structure_is_detected_and_noise_is_not():
    start = T0 + 600 * H
    real = permutation_test(FOLLOWER, Trending(), start, start + 1000 * H, permutations=19)
    assert real.at_least_as_good == 0 and real.p_value == pytest.approx(1 / 20)
    assert real.verdict.startswith("weak evidence") or real.verdict.startswith("unlikely")
    noise = permutation_test(FOLLOWER, SyntheticProvider(seed=8), start, start + 300 * H, permutations=19)
    assert noise.p_value > 0.1  # a random walk has nothing to exploit
    assert real.real == BacktestEngine(FOLLOWER, Trending()).run(start, start + 1000 * H).metrics


def test_p_values_and_verdicts():
    base = BacktestEngine(FOLLOWER, SyntheticProvider(seed=8)).run(T0 + 600 * H, T0 + 650 * H).metrics
    r = PermutationResult("total_return", replace(base, total_return=0.10), (0.2, 0.05, None, 0.1, -0.1))
    assert r.at_least_as_good == 2 and r.p_value == pytest.approx(3 / 5) and len(r.defined) == 4
    assert r.verdict.startswith("indistinguishable from luck")
    dd = PermutationResult("max_drawdown", replace(base, max_drawdown=0.05), (0.10, 0.20, 0.04))
    assert dd.at_least_as_good == 1  # lower is better
    assert PermutationResult("total_return", base, ()).p_value is None
    text = format_permutation(r)
    assert "Shuffled markets (4)" in text and "2 of 4" in text
    assert json.loads(json.dumps(r.to_dict()))["p_value"] == pytest.approx(0.6)


def test_validation(tmp_path):
    with pytest.raises(ConfigError, match="model calls"):
        permutation_test(agents_config(tmp_path), SyntheticProvider(seed=1), T0 + 600 * H, T0 + 700 * H)
    with pytest.raises(ConfigError, match="positive"):
        permutation_test(FOLLOWER, SyntheticProvider(seed=1), T0, T0 + 50 * H, permutations=0)
    with pytest.raises(ConfigError, match="unknown metric"):
        permutation_test(FOLLOWER, SyntheticProvider(seed=1), T0, T0 + 50 * H, metric="luck")


def test_cli(tmp_path, capsys):
    out_file = tmp_path / "p.json"
    args = ["--db", str(tmp_path / "x.db"), "permutation-test", "--synthetic", "4", "--symbols", "BTC/USDT",
            "--start", "2024-02-01", "--end", "2024-02-08", "--permutations", "5"]
    assert main([*args, "--export", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert "shuffled 5/5" in out and "Verdict:" in out
    assert len(json.loads(out_file.read_text())["permuted"]) == 5
    assert main([*args, "--metric", "luck"]) == 1
    assert not (tmp_path / "x.db").exists()
