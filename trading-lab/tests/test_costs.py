"""Stage 17B: cost sensitivity (the same backtest at scaled trading costs)."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from trading_lab.backtest import BacktestEngine
from trading_lab.cli import main
from trading_lab.config import AppConfig
from trading_lab.core.errors import ConfigError
from trading_lab.data import SyntheticProvider
from trading_lab.research import cost_sensitivity, format_costs
from trading_lab.research.costs import CostRow, CostSensitivity, scaled_config

UTC = timezone.utc
START = datetime(2024, 2, 1, tzinfo=UTC)
END = START + timedelta(days=20)
CFG = AppConfig.from_mapping({"market": {"symbols": ["BTC/USDT", "ETH/USDT"]}})


class Counting(SyntheticProvider):
    calls = 0

    def fetch_ohlcv(self, *args, **kwargs):
        Counting.calls += 1
        return super().fetch_ohlcv(*args, **kwargs)


@pytest.fixture(scope="module")
def result():
    Counting.calls = 0
    return cost_sensitivity(CFG, Counting(seed=4), START, END, [3, 1, 0, 2, 1])


def test_every_cost_is_scaled():
    cfg, costs = scaled_config(CFG.with_overrides({"execution": {"slippage_model": "volume"}}), 2.0)
    ex, base = cfg.execution, CFG.execution
    assert ex.fee_rate == 2 * base.fee_rate and ex.maker_fee_rate == 2 * base.maker_fee_rate
    assert ex.slippage_bps == 2 * base.slippage_bps and ex.impact_coefficient == 2 * base.impact_coefficient
    assert ex.short_borrow_bps_per_day == 2 * base.short_borrow_bps_per_day and ex.slippage_model == "volume"
    assert costs["fee_rate"] == ex.fee_rate and cfg.risk == CFG.risk and cfg.strategies == CFG.strategies
    free, _ = scaled_config(CFG, 0.0)
    assert free.execution.fee_rate == free.execution.slippage_bps == 0.0
    with pytest.raises(ValueError):
        scaled_config(CFG, -1)
    with pytest.raises(ConfigError):
        scaled_config(CFG, 1000)  # a 100% fee is not a valid config


def test_rows_by_multiplier_and_data_fetched_once(result):
    assert [r.multiplier for r in result.rows] == [0.0, 1.0, 2.0, 3.0]
    assert Counting.calls == 2  # one fetch per symbol, shared by all four runs
    fees = [r.metrics.total_fees for r in result.rows]
    assert fees[0] == 0.0 and fees == sorted(fees) and fees[-1] > 0
    assert result.rows[0].metrics.total_return > result.rows[-1].metrics.total_return
    as_configured = BacktestEngine(CFG, SyntheticProvider(seed=4)).run(START, END).metrics
    assert result.row(1.0).metrics == as_configured  # the 1x row is exactly the normal backtest


def fake(returns):
    base = BacktestEngine(CFG, SyntheticProvider(seed=4)).run(START, START + timedelta(days=2)).metrics
    return CostSensitivity(tuple(CostRow(m, {"fee_rate": 0.001 * m, "slippage_bps": 5 * m},
                                         replace(base, total_return=r)) for m, r in returns))


def test_break_even_and_verdicts():
    c = fake([(0, 0.04), (1, 0.01), (2, -0.02), (3, -0.05)])
    assert c.break_even == pytest.approx(1 + 1 / 3)
    assert "break-even at about 1.33x" in c.verdict and "thin" in c.verdict
    assert "some room" in fake([(0, 0.05), (2, 0.01), (3, -0.01)]).verdict
    assert fake([(0, -0.01), (1, -0.02)]).verdict.startswith("loses money even at 0x")
    assert fake([(0, 0.05), (3, 0.01)]).verdict.startswith("still profitable at 3x")
    assert fake([(0, 0.05), (3, 0.01)]).break_even is None
    text = format_costs(c)
    assert "Verdict: break-even" in text and "percentage points" in text
    assert json.loads(json.dumps(c.to_dict()))["break_even"] == pytest.approx(4 / 3)


def test_input_validation():
    with pytest.raises(ValueError, match="two different"):
        cost_sensitivity(CFG, SyntheticProvider(seed=1), START, END, [1, 1])


def test_cli(tmp_path, capsys):
    out_file = tmp_path / "costs.json"
    args = ["--db", str(tmp_path / "c.db"), "costs", "--synthetic", "4", "--symbols", "BTC/USDT",
            "--start", "2024-02-01", "--end", "2024-02-11"]
    assert main([*args, "--multipliers", "0,1,2", "--export", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert "Verdict:" in out and "  costs x2" in out
    assert [r["multiplier"] for r in json.loads(out_file.read_text())["rows"]] == [0, 1, 2]
    assert main([*args, "--multipliers", "0,x"]) == 1
    assert main([*args, "--multipliers", "1"]) == 1
    assert not (tmp_path / "c.db").exists()  # nothing is stored
