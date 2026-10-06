"""Indicator correctness against independent reference implementations, plus causality."""

import statistics

import numpy as np
import pandas as pd
import pytest

from trading_lab.indicators import bollinger_bands, ema, macd, rsi, windowed_ema

# Classic Wilder RSI example (StockCharts "RSI" ChartSchool table).
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
    45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64, 46.21, 46.25, 45.71, 46.45,
    45.78, 45.35, 44.03, 44.18, 44.22, 44.57, 43.42, 42.66, 43.13,
]  # fmt: skip
# Published values; StockCharts rounds intermediate averages, so allow 0.1.
STOCKCHARTS_RSI = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38, 54.71, 50.42,
    39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77,
]  # fmt: skip


def reference_rsi(closes, period):
    """Straightforward textbook Wilder RSI."""
    changes = [b - a for a, b in zip(closes, closes[1:])]
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    out = [100 - 100 / (1 + avg_g / avg_l)]
    for g, l in zip(gains[period:], losses[period:]):
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
        out.append(100 - 100 / (1 + avg_g / avg_l))
    return out


def reference_ema(values, span):
    alpha = 2 / (span + 1)
    out, prev = [], None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


@pytest.fixture(scope="module")
def random_close():
    rng = np.random.default_rng(7)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))


def test_rsi_matches_hand_computation_and_published_example():
    r = rsi(pd.Series(WILDER_CLOSES), 14)
    assert r.iloc[:14].isna().all()
    # First value by hand: avg gain 3.34/14, avg loss 1.40/14.
    assert r.iloc[14] == pytest.approx(100 - 100 / (1 + 3.34 / 1.40), abs=1e-9)
    np.testing.assert_allclose(r.dropna(), reference_rsi(WILDER_CLOSES, 14), atol=1e-9)
    np.testing.assert_allclose(r.dropna(), STOCKCHARTS_RSI, atol=0.1)


def test_rsi_edge_cases(random_close):
    assert (rsi(pd.Series(np.arange(1.0, 40.0)), 14).dropna() == 100.0).all()
    assert (rsi(pd.Series(np.arange(40.0, 1.0, -1)), 14).dropna() == 0.0).all()
    assert (rsi(pd.Series(np.full(30, 5.0)), 14).dropna() == 50.0).all()
    assert rsi(pd.Series([1.0, 2.0]), 14).isna().all()
    values = rsi(random_close, 14).dropna()
    assert ((values >= 0) & (values <= 100)).all()
    with pytest.raises(ValueError):
        rsi(random_close, 0)


def test_ema_matches_recursion(random_close):
    result = ema(random_close, 10)
    expected = reference_ema(random_close.tolist(), 10)
    assert result.iloc[:9].isna().all()
    np.testing.assert_allclose(result.iloc[9:], expected[9:], rtol=1e-12)


def test_macd_components(random_close):
    m = macd(random_close, 12, 26, 9)
    line = np.array(reference_ema(random_close.tolist(), 12)) - np.array(
        reference_ema(random_close.tolist(), 26)
    )
    np.testing.assert_allclose(m["macd"].iloc[25:], line[25:], rtol=1e-10)
    signal = reference_ema(line[25:].tolist(), 9)
    np.testing.assert_allclose(m["signal"].iloc[25 + 8 :], signal[8:], rtol=1e-10)
    np.testing.assert_allclose(m["hist"], m["macd"] - m["signal"])
    assert m["signal"].iloc[: 25 + 8].isna().all()
    flat = macd(pd.Series(np.full(60, 3.0)))
    assert (flat.dropna().abs() < 1e-12).all().all()
    with pytest.raises(ValueError):
        macd(random_close, 26, 12, 9)


def test_bollinger_bands_window_values(random_close):
    bb = bollinger_bands(random_close, 20, 2.0)
    t = 150
    window = random_close.iloc[t - 19 : t + 1].tolist()
    mean, sd = statistics.fmean(window), statistics.pstdev(window)
    assert bb["middle"].iloc[t] == pytest.approx(mean, rel=1e-12)
    assert bb["upper"].iloc[t] == pytest.approx(mean + 2 * sd, rel=1e-10)
    assert bb["lower"].iloc[t] == pytest.approx(mean - 2 * sd, rel=1e-10)
    expected_pb = (random_close.iloc[t] - (mean - 2 * sd)) / (4 * sd)
    assert bb["percent_b"].iloc[t] == pytest.approx(expected_pb, rel=1e-9)
    assert bb.iloc[:19].isna().all().all()


def test_bollinger_zero_width_gives_nan_percent_b():
    bb = bollinger_bands(pd.Series(np.full(25, 10.0)), 20)
    assert bb["percent_b"].dropna().empty
    assert bb["upper"].iloc[-1] == bb["lower"].iloc[-1] == 10.0


@pytest.mark.parametrize(
    "func",
    [
        lambda s: rsi(s, 14).to_frame(),
        lambda s: ema(s, 20).to_frame(),
        lambda s: macd(s, 12, 26, 9),
        lambda s: bollinger_bands(s, 20, 2.0),
        lambda s: windowed_ema(s, 20, 60).to_frame(),
    ],
    ids=["rsi", "ema", "macd", "bollinger", "windowed_ema"],
)
def test_indicators_are_causal(func, random_close):
    """The value at bar t must not change when future bars are appended."""
    full = func(random_close)
    for t in (40, 120, 250):
        truncated = func(random_close.iloc[: t + 1])
        np.testing.assert_allclose(
            truncated.iloc[-1].to_numpy(dtype=float),
            full.iloc[t].to_numpy(dtype=float),
            rtol=1e-12,
            equal_nan=True,
        )


def test_windowed_ema_is_the_ema_of_the_last_window(random_close):
    out = windowed_ema(random_close, 20, 60)
    plain = ema(random_close, 20)
    np.testing.assert_allclose(out.iloc[:60].to_numpy(), plain.iloc[:60].to_numpy(), equal_nan=True)
    for t in (59, 100, 250, len(random_close) - 1):
        restarted = ema(random_close.iloc[t - 59 : t + 1], 20).iloc[-1]
        assert out.iloc[t] == pytest.approx(restarted, rel=1e-12)


def test_windowed_ema_does_not_depend_on_where_the_data_starts(random_close):
    """Two data windows ending at the same bar agree, unlike a plain EMA."""
    full, later = random_close, random_close.iloc[37:]
    np.testing.assert_allclose(windowed_ema(later, 20, 60).iloc[60:].to_numpy(),
                               windowed_ema(full, 20, 60).iloc[97:].to_numpy(), rtol=1e-12)
    assert abs(ema(later, 20).iloc[60] - ema(full, 20).iloc[97]) > 1e-9


@pytest.mark.parametrize("span, window", [(0, 10), (20, 0), (20, 10)])
def test_windowed_ema_validates(random_close, span, window):
    with pytest.raises(ValueError):
        windowed_ema(random_close, span, window)
