"""Stage 18A: A/B comparison of two configs over independent windows."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data import SyntheticProvider
from trading_lab.research import ab_test, config_diff, format_ab
from trading_lab.research.ab import ABResult, ABWindow, ab_windows

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
END = START + timedelta(days=24)
A = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})
B = A.with_overrides({"risk": {"allow_short": True}})


def test_config_diff():
    assert config_diff(A, B) == [("risk.allow_short", False, True)]
    with_donchian = A.with_overrides({"strategies": {"donchian": {"weight": 1.0}}})
    keys = {k for k, _, _ in config_diff(A, with_donchian)}
    assert {"strategies.donchian.weight", "strategies.donchian.enabled"} <= keys
    assert all(a is None for k, a, _ in config_diff(A, with_donchian) if k.startswith("strategies.donchian"))
    assert config_diff(A, A) == []


def test_windows():
    spans = ab_windows(START, END, 4)
    assert spans[0][0] == START and spans[-1][1] == END
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert len({e - s for s, e in spans}) == 1
    for bad in (1, 0, 2.5, True):
        with pytest.raises(ConfigError):
            ab_windows(START, END, bad)
    with pytest.raises(ConfigError):
        ab_windows(END, START, 3)


@pytest.fixture(scope="module")
def result():
    return ab_test(A, B, SyntheticProvider(seed=7), START, END, windows=4)


def test_each_window_is_a_fresh_backtest(result):
    assert len(result.windows) == 4 and result.diff == (("risk.allow_short", False, True),)
    w = result.windows[1]
    alone = BacktestEngine(B, SyntheticProvider(seed=7)).run(w.start, w.end).metrics
    assert w.b == alone and w.a == BacktestEngine(A, SyntheticProvider(seed=7)).run(w.start, w.end).metrics
    assert result.a_wins + result.b_wins == result.compared <= 4
    text = format_ab(result, ("a.toml", "b.toml"))
    assert "risk.allow_short: False -> True" in text and "Verdict:" in text
    assert json.loads(json.dumps(result.to_dict(), default=str))["compared"] == result.compared


def test_identical_configs_tie(tmp_path):
    r = ab_test(A, A, SyntheticProvider(seed=7), START, START + timedelta(days=6), windows=2)
    assert r.compared == 0 and r.verdict == "no difference: A and B tie in every window"
    assert "identical" in format_ab(r)


def fake(pairs, metric="total_return"):
    base = BacktestEngine(A, SyntheticProvider(seed=7)).run(START, START + timedelta(days=2)).metrics
    windows = tuple(ABWindow(START, END, replace(base, **{metric: a}), replace(base, **{metric: b}), None)
                    for a, b in pairs)
    return ABResult(metric, windows, ())


def test_sign_test_and_verdicts():
    six = fake([(0.0, 0.01)] * 6)
    assert (six.b_wins, six.compared) == (6, 6) and six.p_b_better == pytest.approx(1 / 64)
    assert six.verdict.startswith("B is better in 6 of 6 windows")
    assert fake([(0.02, 0.01)] * 6).verdict.startswith("A is better in 6 of 6")
    mixed = fake([(0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (1, 0)])
    assert "B leads 4 of 6" in mixed.verdict and "chance" in mixed.verdict
    assert fake([(0, 1), (1, 0)]).verdict.startswith("no difference: each wins 1 of 2")
    drawdown = fake([(0.10, 0.05)] * 6, metric="max_drawdown")  # lower is better
    assert drawdown.b_wins == 6
    assert fake([(None, 0.1), (0.1, 0.1)], metric="sharpe_ratio").verdict == "no window could be compared"


def test_unfair_comparisons_are_refused():
    for other in (A.with_overrides({"market": {"symbols": ["BTC/USDT"]}}),
                  A.with_overrides({"market": {"timeframe": "4h"}}),
                  A.with_overrides({"portfolio": {"initial_cash": 5000.0}})):
        with pytest.raises(ConfigError, match="same"):
            ab_test(A, other, SyntheticProvider(seed=7), START, END)
    with pytest.raises(ConfigError, match="unknown metric"):
        ab_test(A, B, SyntheticProvider(seed=7), START, END, metric="luck")


def test_cli(tmp_path, capsys):
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text('[market]\nsymbols = ["BTC/USDT"]\n')
    b.write_text('[market]\nsymbols = ["BTC/USDT"]\n[risk]\nallow_short = true\n')
    out_file = tmp_path / "ab.json"
    args = ["ab", str(a), str(b), "--synthetic", "7", "--start", "2024-02-01", "--end", "2024-02-13"]
    assert main([*args, "--windows", "3", "--export", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert "risk.allow_short: False -> True" in out and "window 3/3" in out
    assert len(json.loads(out_file.read_text())["windows"]) == 3
    assert main([*args, "--windows", "1"]) == 1
    assert main(["ab", str(a), str(tmp_path / "missing.toml"), "--synthetic", "7"]) == 1
    assert main([*args, "--symbols", "ETH/USDT"]) == 0  # CLI overrides apply to both
