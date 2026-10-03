# trading-lab — Architecture & Implementation Plan

`trading-lab` is a **crypto paper-trading research platform**. Its purpose is to
study trading strategies under a realistic, reproducible simulation. It is not a
trading bot.

## 0. Non-negotiable safety rules

1. **No real trades, ever.** No module may call an exchange's private or trading
   endpoints (`create_order`, `cancel_order`, `withdraw`, `fetch_balance`, ...).
   All order execution goes through the simulated `PaperExecutor`.
2. **No private API keys.** Market data comes only from public, unauthenticated
   endpoints (CCXT `fetch_ohlcv` / `fetch_ticker`). The CCXT client is built with
   no credentials. Configuration has no fields for keys, and unknown config keys
   are rejected.
3. **Enforced by tests.** `tests/test_safety.py` scans the source tree and fails
   if it finds calls to private exchange methods or credential fields.

## 1. Design principles

- **Correctness before profitability.** Accounting invariants are tested, e.g.
  `equity == initial_cash + realized_pnl + unrealized_pnl` after any fill sequence.
- **Reproducibility.** There is no wall-clock time or unseeded randomness in the
  simulation core. Timestamps come from candles or callers, order IDs are
  sequential, and every run records a config fingerprint (SHA-256 of the
  canonical config).
- **Realistic simulation.** Fills include slippage and fees and pay the price
  on the unfavourable side. Market orders fill on the bar *after* the signal,
  so there is no look-ahead. Stops are checked against candle lows, with gaps
  handled.
- **Separation of concerns.** Each module depends only on the modules below it
  (see the diagram), and interfaces are small `Protocol`s or ABCs, so a new
  strategy, data source, execution model or AI agent can be added without
  touching the others.
- **Typed and validated.** Frozen dataclasses hold the domain objects. Invalid
  values fail fast at construction time.

## 2. Architecture

```
                         ┌────────────────────────────┐
                         │            CLI             │  backtest / paper / report
                         └──────────────┬─────────────┘
                  ┌─────────────────────┴───────────────────────┐
                  ▼                                             ▼
        ┌──────────────────┐                         ┌──────────────────────┐
        │  BacktestEngine  │                         │  LivePaperTrader     │
        │ (historical bars)│                         │ (polls closed bars)  │
        └────────┬─────────┘                         └──────────┬───────────┘
                 └──────────────────┬───────────────────────────┘
                                    ▼   one "decision cycle" per closed bar
   ┌──────────────┐   candles  ┌──────────────┐ signals ┌──────────────┐
   │ MarketData   │───────────▶│  Strategies  │────────▶│ VotingEngine │
   │ Provider     │            │ RSI/MACD/BB… │         │ (ensemble)   │
   └──────────────┘            └──────────────┘         └──────┬───────┘
     CCXT (public)                                   combined  │ signal
     CSV cache / synthetic                                     ▼
                                                      ┌──────────────┐
                                                      │ RiskManager  │ sizing, limits,
                                                      └──────┬───────┘ stops
                                                       order │
                                                             ▼
   ┌──────────────┐  fills   ┌──────────────┐        ┌──────────────┐
   │  Portfolio   │◀─────────│ PaperExecutor│◀───────│  CostModel   │ fee + slippage
   │ (accounting) │          └──────────────┘        └──────────────┘
   └──────┬───────┘
          │ snapshots, fills, closed trades, decisions
          ▼
   ┌──────────────┐          ┌──────────────┐
   │ SQLite store │─────────▶│   Metrics    │ return, win rate, drawdown,
   └──────────────┘          └──────────────┘ profit factor, Sharpe
```

### Package layout (`src/trading_lab/`)

| Package        | Responsibility | Stage |
|----------------|----------------|-------|
| `core/`        | Domain models (`Signal`, `Order`, `Fill`, `ExecutionReport`, `Position`, `ClosedTrade`, `PortfolioSnapshot`), enums, errors, supported symbols/timeframes, time helpers | 1 |
| `config.py`    | Typed, validated, strict TOML config (`AppConfig`) + fingerprint | 1 |
| `execution/`   | `CostModel` (fee and slippage models as Protocols), `PaperExecutor` (simulated market orders, long-only) | 1 |
| `portfolio/`   | `Portfolio`: cash, positions, cost basis, realized/unrealized PnL, fees, closed trades | 1 |
| `risk/`        | `RiskManager`: risk-per-trade sizing from stop distance (fees and slippage included), max position size, max total exposure, max open positions, min notional, stop checks | 1 |
| `data/`        | `MarketDataProvider` ABC, `CcxtPublicProvider` (read-only, paginated, rate-limited), on-disk CSV cache, deterministic synthetic provider for tests | 2 |
| `indicators/`  | Pure, vectorised indicator functions (RSI/Wilder, MACD, Bollinger) | 2 |
| `strategies/`  | `Strategy` ABC → standardized `Signal`; RSI, MACD, Bollinger mean-reversion; registry | 2 |
| `ensemble/`    | `VotingEngine` combining signals (confidence-weighted, configurable weights/thresholds) | 2 |
| `storage/`     | SQLite repository: runs, decisions, signals, orders, fills, snapshots, closed trades; schema versioning | 3 |
| `backtest/`    | Event-driven bar-by-bar `BacktestEngine` | 3 |
| `metrics/`     | Performance metrics | 3 |
| `live/`        | `LivePaperTrader` loop (public data, simulated fills, resumable state) | 4 |
| `cli.py`       | `trading-lab backtest`, `trading-lab paper`, `trading-lab report` | 4 |

### Key interfaces

```python
# Strategy (Stage 2): stateless with respect to the portfolio; sees only candles up to bar t
class Strategy(ABC):
    name: str
    warmup_bars: int
    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> Signal: ...

# Signal (Stage 1, core/models.py): the standard currency between all decision makers
Signal(strategy, symbol, direction: BUY|SELL|HOLD, confidence: 0..1, timestamp, metadata)

# Market data (Stage 2)
class MarketDataProvider(ABC):
    def fetch_ohlcv(self, symbol, timeframe, since, until) -> pd.DataFrame: ...

# Execution (Stage 1)
class ExecutionEngine(Protocol):
    def submit(self, order: Order, reference_price: float) -> ExecutionReport: ...
```

**AI agents** will plug in as additional `Strategy` implementations, or as
signal sources that emit the same `Signal` type into the `VotingEngine`. Their
reasoning goes in `Signal.metadata`, which is persisted with every decision.
Nothing downstream of the ensemble has to change.

### Simulation model (the important details)

- **Prices and units.** The quote currency is USDT, treated as USD. Floats are
  used throughout. Accounting tests use tight tolerances and a few relative
  epsilons (1e-9) for quantity matching.
- **Fees.** `fee = notional × fee_rate` (default 0.1% taker), paid in quote
  currency on every fill.
- **Slippage.** Fixed basis points against the trader (default 5 bps). Buys
  fill at `ref × (1 + bps)` and sells at `ref × (1 − bps)`. `SlippageModel` is a
  Protocol, so volume- or volatility-based models can be added later.
- **Cost basis.** Buy fees are capitalised into the position's cost basis.
  Realized PnL is `net sell proceeds − released cost basis`, so it includes the
  fees on both sides.
- **Timing (Stage 3).** A decision on bar *t* uses only data up to the *close*
  of *t*. Its market order fills at the *open* of bar *t+1*. Stop-losses
  trigger when a bar's low reaches the stop. The fill is at the stop price, or
  at the open if the bar gapped through it. Fee and slippage apply to every fill.
- **Long-only.** Selling more than is held, or selling with no position, is
  rejected. Shorting is a later stage.
- **Sizing.** `qty = min(risk-per-trade qty, max-position qty, max-exposure qty, affordable qty)`.
  The risk-per-trade quantity is chosen so that a stop-out costs about
  `equity × risk_per_trade_pct`, including entry and exit fees and slippage.
  The binding constraint is recorded in the decision for auditability.

### Metrics (Stage 3)

Total return, annualised return, max drawdown (on the equity curve), win rate
and profit factor (on closed trades), and Sharpe ratio. Sharpe uses per-bar
returns annualised by timeframe (crypto trades 365 days a year) with risk-free
rate 0. Also reported: number of trades, exposure time, total fees paid and
average trade return.

## 3. Implementation stages

### Stage 1: Foundations and simulation core ✅ (this commit)
- Project skeleton (`pyproject.toml`, src layout, pytest config, `.gitignore`, README)
- `core`: enums, domain models with validation, errors, supported symbols (BTC/ETH/SOL/DOGE vs USDT) and timeframes
- `config`: strict, typed TOML config with defaults ($10,000 starting cash, 0.1% fee, 5 bps slippage, risk limits), fingerprinting
- `execution`: fee and slippage models, `CostModel`, `PaperExecutor` (market orders, long-only, min notional, deterministic IDs)
- `portfolio`: `Portfolio` accounting, snapshots, closed-trade records
- `risk`: `RiskManager` (entry sizing, exit decisions, limits, stop checks)
- Tests: config, models, portfolio accounting, execution, risk limits, safety scan

### Stage 2: Market data, strategies and voting
- `MarketDataProvider` ABC. `CcxtPublicProvider` with credential-free exchange, `enableRateLimit`, pagination, closed-candle filtering, dedupe/sort and gap detection. CSV cache keyed by exchange/symbol/timeframe. `SyntheticProvider` (seeded) for offline tests.
- Indicators: RSI (Wilder), MACD (12/26/9), Bollinger (20, 2σ), with no look-ahead.
- `Strategy` ABC, strategy registry, `RsiStrategy`, `MacdStrategy`, `BollingerMeanReversionStrategy`. Confidence comes from indicator distance to thresholds. Metadata holds the indicator values.
- `VotingEngine`: weighted, confidence-weighted vote with a minimum agreement threshold and a minimum confidence. Ties and abstentions resolve to HOLD.
- Tests: indicator values against hand-computed references, strategy signals on crafted series, voting rules.

### Stage 3: Persistence, backtesting and metrics
- SQLite schema (`runs`, `signals`, `decisions`, `orders`, `fills`, `equity_snapshots`, `closed_trades`) with `PRAGMA user_version` migrations. Each run stores the config JSON and fingerprint.
- `BacktestEngine`: multi-symbol, bar-by-bar, next-bar-open execution, stop handling, end-of-run mark-to-market. Every decision is logged (including HOLDs and risk rejections).
- `metrics`: the metrics listed above.
- Tests: no-look-ahead check, deterministic re-runs (identical results), metric values on known equity curves and trades, SQLite round-trip.

### Stage 4: CLI and live paper trading
- `trading-lab backtest --symbols ... --timeframe 1h --since ... --until ...`
- `trading-lab paper` polls public data, acts on each newly closed candle, fills on the latest public price with the cost model, and persists state to SQLite so a restart resumes. Shuts down gracefully on Ctrl+C.
- `trading-lab report --run-id ...`
- Tests: CLI smoke tests with the synthetic provider, and a live-loop test with a fake clock and provider.

### Stage 5: Later (out of V1 scope)
AI-agent signal sources, short positions, limit and stop orders, partial fills
and order-book or volume-aware slippage, walk-forward and parameter-sweep
tooling, a drawdown kill-switch and daily loss limits, and dashboards.
