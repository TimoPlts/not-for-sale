# trading-lab

A crypto **paper-trading research platform**. It simulates trading BTC/USDT,
ETH/USDT, SOL/USDT and DOGE/USDT with realistic fees and slippage.

> **Simulation only.** trading-lab never places real orders and never asks for
> private exchange API keys. Market data comes from public endpoints. A test
> scans the codebase to enforce this.

See [PLAN.md](PLAN.md) for the architecture and the staged roadmap.

## Status

- **Stage 1 (complete):** configuration, domain models, fee and slippage cost model, paper execution (long-only market orders), portfolio accounting and the risk manager.
- **Stage 2 (complete):** public CCXT market data with a CSV cache, a synthetic data provider, indicators, the RSI, MACD and Bollinger strategies, and the voting engine.
- **Stage 3 (complete):** an event-driven backtester (next-bar-open fills, stop-losses, no look-ahead), the SQLite history of every signal, decision, order, fill and trade, and performance metrics.
- **Stage 4 (complete):** the `trading-lab` command line and live paper trading on real-time public data. It can be resumed after a stop and follows exactly the same rules as a backtest.

## Quick start

Run these from this folder in PowerShell. The `trading-lab` command lives in the project's virtual environment:

```powershell
.venv/Scripts/trading-lab --help
.venv/Scripts/trading-lab signals                       # current signals (live public data)
.venv/Scripts/trading-lab backtest                      # backtest the last 90 days
.venv/Scripts/trading-lab backtest --start 2025-01-01 --end 2025-07-01 --timeframe 4h
.venv/Scripts/trading-lab backtest --synthetic 7        # offline, random-walk data
.venv/Scripts/trading-lab paper                         # live paper trading; Ctrl+C to stop
.venv/Scripts/trading-lab paper --timeframe 15m         # faster feedback
.venv/Scripts/trading-lab paper --resume <run id>       # continue a stopped paper run
.venv/Scripts/trading-lab report                        # list all runs
.venv/Scripts/trading-lab report <run id>               # details of one run
.venv/Scripts/python -m pytest                          # run all tests
```

To type just `trading-lab`, activate the environment first with `.venv\Scripts\Activate.ps1`.

**How paper trading works:** the trader acts once per *closed* candle. It computes signals at the close and fills any resulting order at the open price of the candle that just started, with fees and slippage. Stops are checked against candle lows. These are the same rules as the backtester, so paper results are directly comparable with backtests. Every signal, decision, fill and equity value is saved to `data/trading_lab.db`, and a stopped run continues exactly where it left off with `--resume`. No orders are ever sent to an exchange.

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

## Example (Stage 2: signals from live public data)

```python
from datetime import datetime, timedelta, timezone
from trading_lab.config import load_config
from trading_lab.data import build_provider
from trading_lab.ensemble import VotingEngine
from trading_lab.strategies import build_strategies

cfg = load_config("config/default.toml")
provider = build_provider(cfg)  # public CCXT data, cached under data/cache/
strategies = build_strategies(cfg.enabled_strategies)
voting = VotingEngine.from_specs(cfg.strategies, cfg.voting)

candles = provider.fetch_ohlcv("BTC/USDT", cfg.market.timeframe,
                               datetime.now(timezone.utc) - timedelta(days=30))
signals = [s.generate_signal("BTC/USDT", candles) for s in strategies]
print(voting.combine(signals))
```

### Adding a strategy

```python
from trading_lab.core.models import Direction
from trading_lab.strategies import Strategy, register_strategy

@register_strategy
class MyStrategy(Strategy):
    name = "my_strategy"          # enable it with [strategies.my_strategy] in the config
    warmup_bars = 50
    params = {}

    def _evaluate(self, candles):
        return Direction.HOLD, 0.0, {"note": "..."}
```
