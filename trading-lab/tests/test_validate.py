"""Stage 30A: validate, every research check and one recommendation."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.presets import PRESETS, preset_toml
from trading_lab.research.checkup import FAIL, NA, PASS, WARN
from trading_lab.research.validate import NOT_READY, PAPER, STRONG, align, format_validation, recommend, validate, \
    validation_html
from trading_lab.storage import SQLiteStore

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
BASE = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


def test_align():
    candidate = BASE.with_overrides({"market": {"timeframe": "4h"}, "portfolio": {"initial_cash": 5000.0}})
    baseline = AppConfig()
    aligned, changes = align(baseline, candidate)
    assert changes == ["symbols", "timeframe", "initial cash"]
    assert (aligned.market.symbols, aligned.market.timeframe, aligned.portfolio.initial_cash) == \
        (candidate.market.symbols, "4h", 5000.0)
    assert aligned.strategies == baseline.strategies  # only the market and cash change
    assert align(candidate, candidate) == (candidate, [])


def checkup_of(*statuses):
    checks = [SimpleNamespace(name=f"c{i}", status=s) for i, s in enumerate(statuses)]
    worst = max((("pass", "n/a", "warn", "fail").index(s) for s in statuses), default=0)
    return SimpleNamespace(checks=checks, overall={0: PASS, 1: PASS, 2: WARN, 3: FAIL}[worst])


def ab_of(a_wins, b_wins, p_a, p_b):
    return SimpleNamespace(a_wins=a_wins, b_wins=b_wins, compared=a_wins + b_wins, p_a_better=p_a, p_b_better=p_b,
                           verdict="verdict")


def test_recommendations():
    assert recommend(checkup_of(PASS, FAIL), ab_of(1, 5, 0.9, 0.1))[0] == NOT_READY
    level, reasons = recommend(checkup_of(PASS, PASS), ab_of(6, 0, 0.016, 1.0))
    assert level == NOT_READY and "baseline was better in 6 of 6" in reasons[0]
    assert recommend(checkup_of(PASS, PASS, NA), ab_of(0, 6, 1.0, 0.016))[0] == STRONG
    level, reasons = recommend(checkup_of(PASS, WARN), ab_of(0, 6, 1.0, 0.016))
    assert level == PAPER and "warns about: c1" in reasons[0]  # a warning is not a clear pass
    assert recommend(checkup_of(PASS, PASS), ab_of(2, 4, 0.9, 0.34))[0] == PAPER
    assert recommend(checkup_of(PASS), None)[0] == PAPER


@pytest.fixture(scope="module")
def result():
    candidate = PRESETS["trend"].config(BASE)
    return validate(candidate, BASE, SyntheticProvider(seed=3), START, START + timedelta(days=30), label="trend",
                    windows=2, permutations=3)


def test_validate(result):
    assert result.recommendation in (NOT_READY, PAPER, STRONG) and result.reasons
    assert result.ab is not None and result.ab.compared <= 2 and not result.aligned
    assert result.outlook is not None and result.outlook.trades == result.checkup.metrics.num_trades
    assert result.sizing is not None and result.trades is not None and result.trades.groups["exit"]
    text = format_validation(result)
    assert text.startswith("Validation of trend") and f"Recommendation: {result.recommendation.upper()}" in text
    assert "A = baseline, B = candidate" in text and "never real money" in text
    page = validation_html(result)
    assert page.index("Recommendation:") < page.index("<h2>Checks</h2>")
    json.dumps(result.to_dict(), default=str)


def test_identical_configs_skip_the_ab_test():
    same = validate(BASE, BASE, SyntheticProvider(seed=3), START, START + timedelta(days=10), windows=2,
                    permutations=2)
    assert same.ab is None and "identical configs" in format_validation(same)


def test_cli(tmp_path, capsys):
    candidate = tmp_path / "trend.toml"
    candidate.write_text(preset_toml("trend") + f'\n[market]\nsymbols = ["BTC/USDT"]\n'
                         f'[storage]\ndb_path = "{tmp_path / "h.db"}"\nrecord_trials = true\n')
    baseline = tmp_path / "base.toml"
    baseline.write_text('[market]\nsymbols = ["ETH/USDT"]\n')
    html, js = tmp_path / "v.html", tmp_path / "v.json"
    args = ["validate", str(candidate), "--baseline", str(baseline), "--synthetic", "3", "--start", "2024-02-01",
            "--end", "2024-03-01", "--windows", "2", "--permutations", "2", "--html", str(html), "--json", str(js)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Recommendation:" in out and "baseline aligned: symbols" in out and "trial(s) logged" in out
    data = json.loads(js.read_text())
    assert data["recommendation"] in (NOT_READY, PAPER, STRONG) and data["aligned"] == ["symbols"]
    assert "Validation of" in html.read_text()
    with SQLiteStore(tmp_path / "h.db", readonly=True) as store:
        logged = store.load_trials()
    assert [t["command"] for t in logged] == ["checkup", "ab", "ab"]
    assert main(["validate", str(tmp_path / "missing.toml")]) == 1
    assert main(["validate", str(candidate), "--baseline", str(tmp_path / "nope.toml")]) == 1
