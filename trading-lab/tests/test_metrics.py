"""Performance metric tests on hand-checkable inputs."""

import math

import numpy as np
import pytest

from conftest import ts
from trading_lab.core.models import ClosedTrade
from trading_lab.metrics import compute_metrics, periods_per_year


def trade(pnl, basis=100.0):
    return ClosedTrade("BTC/USDT", 1.0, basis, basis + pnl, basis, basis + pnl, pnl, ts(0), ts(1))


def test_periods_per_year():
    assert periods_per_year("1h") == 365 * 24
    assert periods_per_year("1d") == 365
    assert periods_per_year("15m") == 365 * 24 * 4


def test_returns_drawdown_and_sharpe_on_known_curve():
    equity = [100.0, 110.0, 99.0, 120.0]
    m = compute_metrics(equity, [], "1d")
    assert m.total_return == pytest.approx(0.2)
    assert m.max_drawdown == pytest.approx(11 / 110)
    returns = np.array([0.1, -0.1, 120 / 99 - 1])
    expected_sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(365)
    assert m.sharpe_ratio == pytest.approx(expected_sharpe)
    assert m.volatility_annualized == pytest.approx(returns.std(ddof=1) * math.sqrt(365))
    downside = math.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    assert m.sortino_ratio == pytest.approx(returns.mean() / downside * math.sqrt(365))
    assert m.annualized_return == pytest.approx(1.2 ** (365 / 3) - 1)
    assert m.num_bars == 3 and m.initial_equity == 100 and m.final_equity == 120


def test_drawdown_measured_from_running_peak():
    m = compute_metrics([100, 150, 120, 160, 80, 90], [], "1h")
    assert m.max_drawdown == pytest.approx(0.5)  # 160 -> 80


def test_flat_curve_has_undefined_ratios():
    m = compute_metrics([100.0, 100.0, 100.0], [], "1h")
    assert m.total_return == 0 and m.max_drawdown == 0
    assert m.sharpe_ratio is None and m.sortino_ratio is None
    single = compute_metrics([100.0], [], "1h")
    assert single.annualized_return is None and single.sharpe_ratio is None


def test_trade_statistics():
    trades = [trade(10), trade(-5), trade(20), trade(-5)]
    m = compute_metrics([100, 120], trades, "1h", total_fees=3.5, in_market=[True, False, False, True])
    assert m.num_trades == 4
    assert m.win_rate == pytest.approx(0.5)
    assert m.profit_factor == pytest.approx(30 / 10)
    assert m.avg_win == pytest.approx(15) and m.avg_loss == pytest.approx(-5)
    assert m.best_trade == 20 and m.worst_trade == -5
    assert m.avg_trade_return == pytest.approx(np.mean([0.1, -0.05, 0.2, -0.05]))
    assert m.total_fees == 3.5
    assert m.exposure == pytest.approx(0.5)


def test_profit_factor_edge_cases():
    assert compute_metrics([100, 101], [], "1h").profit_factor is None
    assert compute_metrics([100, 101], [trade(5)], "1h").profit_factor == math.inf
    assert compute_metrics([100, 101], [trade(0)], "1h").profit_factor is None
    assert compute_metrics([100, 99], [trade(-1)], "1h").profit_factor == 0.0
    assert compute_metrics([100, 101], [], "1h").win_rate is None


def test_to_dict_is_json_safe():
    m = compute_metrics([100, 101], [trade(5)], "1h")
    d = m.to_dict()
    assert d["profit_factor"] == "inf"
    assert d["sharpe_ratio"] is None
    assert "Profit factor" in m.format_table()


@pytest.mark.parametrize("bad", [[], [100, 0], [100, float("nan")]])
def test_invalid_equity_rejected(bad):
    with pytest.raises(ValueError):
        compute_metrics(bad, [], "1h")


def test_annualizing_short_huge_gain_does_not_overflow():
    m = compute_metrics([100, 120], [], "1h")  # +20% in one hour
    assert m.annualized_return == math.inf
    assert m.to_dict()["annualized_return"] == "inf"
