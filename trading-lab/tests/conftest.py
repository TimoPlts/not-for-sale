from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trading_lab.config import RiskConfig
from trading_lab.core.models import Order, Side
from trading_lab.execution.costs import CostModel, FixedBpsSlippage, PercentageFeeModel
from trading_lab.execution.paper import PaperExecutor
from trading_lab.portfolio.portfolio import Portfolio
from trading_lab.risk.manager import RiskManager

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src" / "trading_lab"


def ts(hours: int = 0) -> datetime:
    return T0 + timedelta(hours=hours)


def make_costs(fee_rate: float = 0.001, slippage_bps: float = 5.0) -> CostModel:
    return CostModel(PercentageFeeModel(fee_rate), FixedBpsSlippage(slippage_bps))


def buy(symbol: str, qty: float, hours: int = 0, stop_price: float | None = None) -> Order:
    return Order(symbol, Side.BUY, qty, ts(hours), stop_price=stop_price)


def sell(symbol: str, qty: float, hours: int = 0) -> Order:
    return Order(symbol, Side.SELL, qty, ts(hours))


@pytest.fixture
def default_config_path() -> Path:
    return PROJECT_ROOT / "config" / "default.toml"


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(10_000.0)


@pytest.fixture
def costs() -> CostModel:
    return make_costs()


@pytest.fixture
def executor(portfolio: Portfolio, costs: CostModel) -> PaperExecutor:
    return PaperExecutor(portfolio, costs, min_notional=10.0)


@pytest.fixture
def risk_manager(costs: CostModel) -> RiskManager:
    return RiskManager(RiskConfig(), costs, min_notional=10.0)
