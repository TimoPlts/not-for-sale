"""Stage 18C: volatility-targeted position sizing (opt-in)."""

import math
from datetime import timedelta

import pytest

from test_live import ANCHOR, START, Clock, fill_key
from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig, RiskConfig
from trading_lab.core.errors import ConfigError
from trading_lab.core.models import DecisionAction, Side
from trading_lab.data import SyntheticProvider
from trading_lab.execution import CostModel
from trading_lab.execution.costs import MarketStats
from trading_lab.live import LivePaperTrader
from trading_lab.portfolio import Portfolio
from trading_lab.risk import RiskManager
from trading_lab.storage import SQLiteStore

H = timedelta(hours=1)
HOURS_PER_YEAR = 365 * 24


def manager(pct=0.10, **risk):
    cfg = AppConfig.from_mapping({"risk": {"position_volatility_pct": pct, "max_position_pct": 1.0,
                                           "risk_per_trade_pct": 0.5, "allow_short": True, **risk},
                                  "execution": {"fee_rate": 0.0, "slippage_bps": 0.0}})
    return RiskManager(cfg.risk, CostModel.from_config(cfg.execution), bars_per_year=HOURS_PER_YEAR)


def test_the_volatility_limit():
    stats = MarketStats(volatility=0.01, avg_quote_volume=1e9)  # 1% per hour ~ 93.6% a year
    decision = manager().evaluate_entry("BTC/USDT", 100.0, Portfolio(10_000), {}, stats)
    annual = 0.01 * math.sqrt(HOURS_PER_YEAR)
    assert decision.sizing["volatility_target"] == pytest.approx(10_000 * 0.10 / (annual * 100))
    assert decision.sizing["binding_limit"] == "volatility_target"
    assert decision.quantity * 100 * annual == pytest.approx(1_000)  # adds exactly 10% of equity in volatility
    calm = manager().evaluate_entry("BTC/USDT", 100.0, Portfolio(10_000), {}, MarketStats(0.005, 1e9))
    assert calm.quantity == pytest.approx(2 * decision.quantity)  # half the volatility, twice the size
    short = manager().evaluate_entry("BTC/USDT", 100.0, Portfolio(10_000, allow_short=True), {}, stats,
                                     side=Side.SELL)
    assert short.quantity == pytest.approx(decision.quantity)


def test_off_by_default_and_without_volatility():
    assert RiskConfig().position_volatility_pct == 0.0
    stats = MarketStats(0.01, 1e9)
    assert "volatility_target" not in manager(pct=0.0).evaluate_entry("BTC/USDT", 100.0, Portfolio(10_000), {},
                                                                    stats).sizing
    for missing in (None, MarketStats(0.0, 1e9)):
        assert "volatility_target" not in manager().evaluate_entry("BTC/USDT", 100.0, Portfolio(10_000), {},
                                                                 missing).sizing
    with pytest.raises(ValueError, match="bars_per_year"):
        RiskManager(RiskConfig(position_volatility_pct=0.1), CostModel.from_config(AppConfig().execution))
    for bad in (-0.1, 11, "x"):
        with pytest.raises(ConfigError):
            RiskConfig(position_volatility_pct=bad)


CFG = AppConfig.from_mapping({
    "market": {"symbols": ["BTC/USDT", "DOGE/USDT"]},
    "risk": {"position_volatility_pct": 0.05, "max_position_pct": 1.0, "risk_per_trade_pct": 0.5},
})


def test_backtest_entries_respect_the_target():
    result = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + 300 * H)
    entries = [d for d in result.decisions if d.action is DecisionAction.ENTER]
    bound = [d for d in entries if d.details.get("binding_limit") == "volatility_target"]
    assert len(bound) >= 4
    for d in bound:
        assert d.details["volatility_target"] == pytest.approx(d.quantity, rel=1e-6)
    # The calmer coin gets the bigger share of equity per position.
    share = {}
    for d in bound:
        share.setdefault(d.symbol, []).append(d.quantity * d.reference_price / d.details["equity"])
    assert sum(share["BTC/USDT"]) / len(share["BTC/USDT"]) > sum(share["DOGE/USDT"]) / len(share["DOGE/USDT"])


def test_live_matches_the_backtest():
    bars = 120
    clock = Clock(START + H + timedelta(minutes=1))
    with SQLiteStore(":memory:") as store:
        trader = LivePaperTrader(CFG, SyntheticProvider(seed=11, anchor=ANCHOR, clock=clock), store, clock=clock)
        for _ in range(bars):
            trader.run_cycle()
            clock.now += H
        live = [fill_key(f)[1:] for f in trader.portfolio.fills if f.timestamp < START + bars * H]
    bt = BacktestEngine(CFG, SyntheticProvider(seed=11, anchor=ANCHOR)).run(START, START + bars * H)
    assert live and live == [fill_key(f)[1:] for f in bt.fills]
