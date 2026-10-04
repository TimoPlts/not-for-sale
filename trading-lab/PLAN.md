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

### Stage 1: Foundations and simulation core ✅
- Project skeleton (`pyproject.toml`, src layout, pytest config, `.gitignore`, README)
- `core`: enums, domain models with validation, errors, supported symbols (BTC/ETH/SOL/DOGE vs USDT) and timeframes
- `config`: strict, typed TOML config with defaults ($10,000 starting cash, 0.1% fee, 5 bps slippage, risk limits), fingerprinting
- `execution`: fee and slippage models, `CostModel`, `PaperExecutor` (market orders, long-only, min notional, deterministic IDs)
- `portfolio`: `Portfolio` accounting, snapshots, closed-trade records
- `risk`: `RiskManager` (entry sizing, exit decisions, limits, stop checks)
- Tests: config, models, portfolio accounting, execution, risk limits, safety scan

### Stage 2: Market data, strategies and voting ✅
- `data`: a `MarketDataProvider` ABC and the canonical OHLCV frame (UTC open-time index, float64, closed candles only, validated).
  - `CcxtPublicProvider`: the client is built without credentials and refuses any client that has them. It sets `enableRateLimit`, paginates, retries network errors with exponential backoff, drops the candle still forming, and calls only `fetch_ohlcv`.
  - `CachedProvider`: a CSV cache that fetches only the missing tail and writes atomically.
  - `SyntheticProvider`: a seeded random walk that is the same at any given timestamp regardless of the window requested.
  - `candles_from_closes` builds crafted test scenarios.
- `indicators`: RSI (exact Wilder smoothing), EMA, MACD (12/26/9) and Bollinger Bands (20, 2σ, population std, %B, bandwidth). Tests check them against independent reference implementations and the published Wilder example, and verify they are causal.
- `strategies`: the `Strategy` ABC (template method with warmup handling and params in the metadata), a registry with `@register_strategy` plus a config-driven `build_strategies`, and three strategies:
  - `RsiStrategy`: BUY below 30, SELL above 70.
  - `MacdStrategy`: histogram zero-crossovers.
  - `BollingerMeanReversionStrategy`: BUY below the lower band, SELL at or above `exit_percent_b` (the upper band by default; 0.5 exits at the middle band).
  - Confidence convention: 0.5 when a trigger is just met, rising to 1.0 at extremes. HOLD always has confidence 0.
- `ensemble`: the `VotingEngine` computes a confidence-weighted net score in which abstentions dilute and opposing votes cancel. It applies `buy_threshold`/`sell_threshold` and `min_agreeing`, and records the full vote breakdown in the metadata.
- Config: new `[data]`, `[strategies.<name>]` (`enabled`, `weight` and strategy params) and `[voting]` sections.
- Tests: indicators, strategy signals on crafted series, voting rules, data validation, the CCXT provider (with fake clients), cache behaviour and synthetic determinism.

### Stage 3: Persistence, backtesting and metrics ✅
- Strategies now have two layers: `IndicatorStrategy` computes causal indicator columns once and then decides per bar, giving an O(n) `generate_signals` for backtests. Tests prove it is identical to bar-by-bar `generate_signal`.
- `storage.SQLiteStore`: tables `runs`, `signals`, `decisions`, `orders`, `fills`, `equity_snapshots`, `closed_trades` and `metrics`. The schema version is kept in `PRAGMA user_version` with an in-place migration path. Each run stores its full config JSON and fingerprint, and failed runs are marked as such with the error.
- `backtest.BacktestEngine`: multi-symbol and bar-by-bar.
  - Signals are computed at bar close and orders fill at the next bar's open.
  - Exits are processed before entries, and entries in descending confidence.
  - Entries are sized at fill time.
  - Stop-losses trigger on the bar low and fill at the stop, or at the open after a gap.
  - Optional liquidation at the end.
  - Every signal, decision (including HOLD, IGNORED, REJECTED and EXPIRED), order and fill is recorded.
- `metrics`: total and annualised return, max drawdown, annualised volatility, Sharpe and Sortino (365-day year, risk-free rate 0), trade count, win rate, profit factor, average win and loss, best and worst trade, fees, and exposure.
- Config: `[backtest] liquidate_at_end` and `[storage] db_path`. `AppConfig.with_overrides()` changes settings from the command line.
- Scripts: `scripts/run_backtest.py` (real or synthetic data, CSV export) and `scripts/show_runs.py`.
- Tests: exact timing scenarios (next-bar fills, stops, gaps, same-bar stops, ignored and expired signals, liquidation, confidence-ordered entries), engine-level no-look-ahead (perturbing future candles leaves the past unchanged), determinism, invariants, SQLite round-trips, and metric values on hand-checked curves.

### Stage 4: CLI and live paper trading ✅
- `engine.TradingSession` holds the per-bar trading rules (fill scheduled orders at the open, stops, mark to market, signals, scheduling). The backtester and the live paper trader both drive this same object, so a live run follows exactly the backtest model. A test proves that a candle-by-candle live run produces the same fills as a backtest over the same period.
- `live.LivePaperTrader`:
  - Each cycle fetches the latest closed public candles and processes every bar closed since the last cycle, so it catches up after downtime.
  - Orders scheduled at a close fill immediately at the open of the candle that just started (`provider.current_open`), at the same price a backtest uses.
  - It persists records and state atomically, and sleeps until the next candle closes.
  - Ctrl+C stops it cleanly. Data errors are reported and retried.
- Resuming loads the run's stored config, replays the stored fills to rebuild the portfolio, and restores the scheduled orders, last prices and order counter. A test proves that a stopped and resumed run equals an uninterrupted one.
- Storage: schema v2 (adds `run_state`) via the migration path, `atomic()` transactions, fill loading, and run status (`running` / `stopped` / `completed` / `failed`).
- CLI (`trading-lab`, or `python -m trading_lab`): `backtest`, `paper` (with `--resume`, `--once`, `--max-cycles`, `--synthetic`), `report` and `signals`, plus global `--config` and `--db`. The scripts in `scripts/` are now shortcuts to these commands.

### Stage 5: Risk circuit breakers ✅
- Portfolio-wide breakers that block **new entries**; exits are always allowed:
  - **Max drawdown kill switch:** permanent for the run, and optionally closes all positions.
  - **Daily loss limit:** resets at the next UTC day.
  - **Per-symbol cooldown** after a stop-loss.
- Breaker state lives in the `TradingSession`, is persisted for live runs, and every trip is recorded as a decision.

### Stage 6: AI agent framework ✅
- An `Agent` interface: it receives a structured, serialisable market context and returns direction, confidence and a rationale. It plugs in through an `AgentStrategy` adapter, so agents vote like any other strategy and can never place orders themselves.
- **Record and replay:** responses are cached in SQLite, keyed by agent, version, symbol, bar and context hash. Backtests with an agent are reproducible even if the agent is not, and a replay never calls the agent.
- Prompt rendering and strict JSON response parsing, ready for an LLM-backed agent later.
- An offline, deterministic example agent.

### Stage 7: Research tools ✅
- Parameter sweeps over any config keys, with data loaded once, results ranked and saved.
- Walk-forward evaluation: choose parameters on a training window, measure on the next unseen window, and report in-sample versus out-of-sample performance to expose overfitting.
- A buy-and-hold benchmark in every backtest.
- A `compare` command for side-by-side metrics.

### Stage 8: More realistic execution ✅
- Volume-aware slippage: square-root market impact against recent traded volume, using only information available at fill time.
- A liquidity cap: maximum participation in recent bar volume, applied as a sizing limit.
- Limit entry orders: price offset, time-to-live, maker fee, trade-through fill rule, and partial fills capped by bar volume with the remainder resting until expiry.

### Later
Short positions, trailing stops, an LLM-backed agent (needs an LLM API key, never exchange keys), order-book data, and dashboards.
