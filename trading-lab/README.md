# trading-lab

A crypto **paper-trading research platform**. It simulates trading BTC/USDT,
ETH/USDT, SOL/USDT and DOGE/USDT with realistic fees and slippage.

> **Simulation only.** trading-lab never places real orders and never asks for
> private exchange API keys. Market data comes from public endpoints. A test
> scans the codebase to enforce this.

See [PLAN.md](PLAN.md) for the architecture and the staged roadmap.

## Status

**Stage 1 is complete:** configuration, domain models, fee and slippage cost
model, paper execution (long-only market orders), portfolio accounting, the
risk manager, and their tests. Market data, strategies, voting, backtesting,
SQLite history, metrics and the CLI come in Stages 2–4.

## Setup (Windows / PowerShell)

```powershell
cd trading-lab
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

On macOS or Linux, use `.venv/bin/python` instead.

## Configuration

`config/default.toml` holds the defaults: $10,000 starting cash, a 0.1% fee,
5 bps slippage, 25% maximum position size, 1% risk per trade and a 5% stop.
Loading is strict, so unknown sections or keys are errors.

```python
from trading_lab.config import load_config
cfg = load_config("config/default.toml")
```

## Example (Stage 1 API)

```python
from datetime import datetime, timezone
from trading_lab.config import load_config
from trading_lab.execution import CostModel, PaperExecutor
from trading_lab.portfolio import Portfolio
from trading_lab.risk import RiskManager

cfg = load_config()
costs = CostModel.from_config(cfg.execution)
portfolio = Portfolio(cfg.portfolio.initial_cash)
executor = PaperExecutor(portfolio, costs, min_notional=cfg.execution.min_notional)
risk = RiskManager(cfg.risk, costs, min_notional=cfg.execution.min_notional)

now = datetime(2024, 1, 1, tzinfo=timezone.utc)
decision = risk.evaluate_entry("BTC/USDT", 42_000.0, portfolio, prices={})
if decision.approved:
    report = executor.submit(decision.to_order(now), reference_price=42_000.0)
    print(report.fill, portfolio.snapshot({"BTC/USDT": 42_000.0}, now))
```
