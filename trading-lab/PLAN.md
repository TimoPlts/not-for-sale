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
| `agents/`      | Agent interface, context, record/replay cache, `AgentStrategy` adapter, LLM-backed and specialist (Trend/Momentum/Risk) strategies | 6, 9 |
| `llm/`         | Provider-agnostic LLM clients (`LLMProvider`); Qwen via OpenAI-compatible chat completions | 9 |
| `research/`    | Sweeps, walk-forward, agent attribution, baseline-vs-AI experiments, out-of-sample comparison | 7, 9, 10 |
| `dashboard/`   | Read-only data layer over SQLite and the Streamlit app | 10 |
| `smoke.py`     | `agent-test` connectivity and agent smoke tests | 9 |

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

### Stage 9A: Qwen LLM provider ✅
- New `llm/` package, provider-agnostic: an `LLMProvider` ABC (`chat() -> Completion` with text, token usage when reported, latency and attempts; `complete()` is the Stage 6 `complete(system, user) -> text` interface) and a provider error hierarchy.
- `OpenAICompatibleProvider`: a client for `/chat/completions` endpoints over a pluggable HTTP transport (standard-library `urllib`, so no new dependency). It reads `<PREFIX>_API_URL`, `<PREFIX>_API_KEY` and `<PREFIX>_MODEL` from the environment only. Timeouts, network errors, 429 and 5xx are retried with exponential backoff; other HTTP errors fail at once with a hint. The token is redacted from every message, log and `repr`. `QwenProvider` is this client with the `QWEN` prefix; adding OpenAI, Anthropic or Gemini means adding a provider class.
- `[agents]` gains `provider`, `request_timeout_seconds`, `max_retries`, `retry_backoff_seconds`, `temperature` and `max_output_tokens`. These are non-secret, and old configs still load.
- `LLMProviderStrategy` (the base for model-backed agents) and the registered `llm_analyst` strategy. `configure_agents` builds one shared provider and fails fast when variables are missing. In replay only `QWEN_MODEL` is required, and the model is never called.
- Tests: missing variables, request and response format, malformed bodies and answers, timeouts, retries, client errors, failure → HOLD (including a whole backtest against a failing endpoint), record reuse, replay offline, live always asking, secret redaction, and a real local HTTP server for the transport and its timeout. New safety tests: no hard-coded tokens in the source, and no credential fields in the config.

### Stage 9B: Specialist Qwen agents ✅
- `agents/specialists.py`: `qwen_trend`, `qwen_momentum` and `qwen_risk` share the configured provider. Each has its own system prompt, its own causal feature frame and a strictly validated extra label (`regime`, `momentum_state` or `risk_state`). Answers missing the label, or with an unknown value, become HOLD.
- Portfolio-aware agents: `Strategy.uses_portfolio`, `signal_at(…, portfolio)` and `generate_signals_at(…)`. `TradingSession.close_bar` accepts lazy signal sources and evaluates them after stops, marks and breakers, with a coarse, read-only `portfolio_view`: position, return, bars held, stop distance, exposure, open positions, drawdown, daily PnL, recent stop-outs, kill switch and entry block. Recent stop-outs are tracked in the session.
- `qwen_risk` is portfolio-aware by default. The other two are market-only by default, so their cached answers are shared by all experiments; `portfolio_context = true` adds position status.
- The backtester evaluates strategies only for bars inside the period, so agents are not asked about the warm-up history.
- `weight = 0` means not taking part: the strategy is not built, not evaluated and casts no vote. The three agents ship in the default config with weight 0.
- The ATR indicator (Wilder) is added, and `parse_agent_json` takes required labels.
- Tests: config defaults and the environment, prompts, causal features, label validation, labels through the cache, the portfolio view, the risk agent unable to override the kill switch, three agents voting, in-period-only calls, record/replay with a portfolio-aware agent, and unchanged baseline results.

### Stage 9C: Agent performance attribution ✅
- `research/attribution.py`: per voter, the vote counts (BUY/SELL/HOLD/errors), average confidence, N-bar directional correctness, average outcome after BUY and SELL, calibration by confidence bucket, and trade attribution. Trade attribution links each closed trade to its entry signal and records agreed/disagreed/influenced/pivotal trades plus the PnL when the voter agreed or disagreed. It works on in-memory results (`attribute_result`) and stored runs (`attribute_run`). The definitions are documented in the module.
- Storage schema v3: a `bars` table (the OHLCV each run traded on, written by backtests and live paper runs) and a `research_results` table (used by later stages). Older databases are upgraded in place.
- CLI: `trading-lab agent-report [RUN_ID] [--horizon N] [--all]`.
- Tests: a hand-computed scenario with exact numbers, determinism and in-memory equal to stored, the CLI, and the v2 → v3 migration.

### Stage 9D: Baseline versus AI experiments ✅
- `research/experiments.py`: named variants (`baseline`, `trend`, `momentum`, `risk`, `trend_momentum`, `all_agents`, `ai_only`) that change only strategy weights. `run_experiment` runs them over the same data, in backtest or walk-forward mode, with an optional in-sample grid.
- `BacktestEngine`, `run_sweep` and `walk_forward` accept one shared `llm_provider`. `shared_llm_provider` checks the credentials once before any run, and all runs share the answer cache, so a market-only agent is asked about each bar once across all combinations.
- CLI: `trading-lab experiment` (`--variants`, `--walkforward`, `--param`, `--save` to the `research_results` table, `--export` JSON). There is also a global `--agent-mode record|replay|live`. Sweep and walk-forward fail fast on missing Qwen variables.
- Tests: variants change only weights; answers are shared across variants, sweep combinations and overlapping walk-forward windows; replay reproduces every variant offline; the off row equals the baseline; the CLI in both modes; the fail-fast check.

### Stage 9E: Model usage accounting ✅
- `llm/usage.py`: `CallRecord` (one call: provider, model, ok, latency, attempts, characters, tokens and whether they are estimated, error), `UsageStats` (calls, cache hits and misses, failures, invalid answers, retries, characters, tokens, latency) and `UsageTracker` (per agent, kept on each provider as `provider.usage`). Tokens are taken from the endpoint when it reports them; otherwise they are estimated conservatively as ceil(chars / 3) and flagged.
- `ProviderAgent` records every call, successful or not, and `AgentStrategy` reports every cache lookup. Each call's record is stored in the signal metadata (`llm`), and `usage_from_signals` rebuilds the usage of stored runs.
- The CLI prints usage after `backtest`, `sweep`, `walkforward` and `experiment`, and in `agent-report`.
- Tests: hits and misses across runs, latency with a fake clock, reported versus estimated tokens, failures, retries and invalid answers, stored usage equal to the tracker, and the CLI output.

### Stage 9F: Agent smoke test ✅
- `smoke.py` plus `trading-lab agent-test [qwen | qwen_trend | qwen_momentum | qwen_risk | llm_analyst]`. It shows which environment variables are set (the token is never shown), sends one tiny prompt, validates the structured answer and prints the model, latency, attempts and tokens. Agent tests ask the real agent about the latest closed candle (public or `--synthetic` data); the risk agent sees a flat simulated portfolio. Live mode, a throw-away cache, no database, no orders.
- Tests (mocked credentials): success, missing variables, malformed, unauthorised and non-JSON answers, each agent, unknown targets. Any attempt to submit an order fails the test.

### Stage 9G: Qwen agents in live paper trading ✅
- `LivePaperTrader` evaluates strategies only for the candles that closed since the last cycle (`generate_signals_at`). Portfolio-aware agents are evaluated inside the session with the live portfolio view, and the trader accepts a shared `llm_provider`.
- The session state persists recent stop-outs (`stop_events`) next to the breakers, working limit orders, scheduled orders and last prices, so the risk agent's context is identical after `--resume`.
- Tests: live with three mocked agents equals the backtest and asks only about traded bars on decision bars; stop and resume equals an uninterrupted run (fills, every decision, breakers, working limits, stop-outs, no repeated question), with market and limit entries; a crash after the agents answered leaves nothing behind, and the restart reuses the cached answers and duplicates no order; model failures become HOLD while trading continues; live mode never re-asks processed bars across a resume.

### Stage 10A: Dashboard data layer ✅
- `dashboard/data.py`, `DashboardData`: a read-only facade over the SQLite history covering the overview (equity, cash, realized/unrealized PnL, current and max drawdown, daily PnL, exposure, breakers from saved state or decisions), open positions rebuilt from fills, working orders, equity curve with drawdown and a buy & hold benchmark computed from stored bars, recent trades, fills and signals, the latest decision (every vote, the ensemble, the actions taken), latest agent rationales, attribution, model usage, and research results. Everything is JSON-safe, and `snapshot()` returns it all at once.
- `SQLiteStore(readonly=True)` uses SQLite's read-only URI mode and never migrates. Stores now wait up to 30 s for locks, and signal queries can be filtered by time and strategy.
- CLI: `trading-lab dashboard-data [RUN_ID] [--json]`.
- Tests: the snapshot matches the run (equity, return, drawdown, positions, benchmark, votes, rationales, attribution, usage); the paper-run state, breakers and working orders; reading leaves the file byte-identical; writes fail; old databases are read without migrating; the dashboard source contains no write or order paths.

### Stage 10B: Read-only web dashboard ✅
- `dashboard/app.py` (Streamlit, the optional `[dashboard]` extra) is built only on `DashboardData`. Sections: portfolio tiles with breaker banners, equity versus buy & hold, drawdown, open positions and working orders, the latest decision with every vote and the ensemble, AI rationale cards, the agent performance leaderboard, research (metrics and saved experiments), Qwen usage, recent trades and signals. Auto-refresh uses a Streamlit fragment, and the sidebar holds view controls only.
- CLI: `trading-lab dashboard [--host 127.0.0.1] [--port 8501]` launches Streamlit with `TRADING_LAB_DB` set.
- Tests (Streamlit AppTest, headless): every section renders; the database is byte-identical afterwards; no buttons or inputs exist; the kill-switch banner shows; empty and missing databases are handled; the launcher's command and environment are correct.

### Stage 10C: VM-friendly 24/7 operation ✅
- `deploy/systemd/trading-lab-paper.service` (loads `/etc/trading-lab/trading-lab.env`, `paper --run-id`, SIGTERM, restart on failure, hardened, writes only data and logs) and `trading-lab-dashboard.service` (no secrets, 127.0.0.1, read-only paths). These are templates; nothing is installed automatically.
- `deploy/trading-lab.env.example` holds placeholders only. `.env`, `*.env` and `.env.*` are git-ignored, except the examples.
- `paper --run-id NAME` resumes the run or starts it. The global `--log-file` and `--log-level` write a rotating log file that mirrors paper activity. `run_forever(stop_event=…)` and a SIGTERM handler let the current cycle finish, save and mark the run `stopped`.
- `docs/DEPLOYMENT.md`: install, secrets, agent-test, configuration, the services, the SSH tunnel for the dashboard, file locations, stopping/restarting/resuming, backups and updates, and a security checklist.
- Tests: named runs, run-id validation, the log file, the stop event, SIGTERM through the CLI, the unit templates, placeholder-only secrets, git-ignore rules, and guide coverage.

### Stage 10D: Failure recovery ✅
- Model endpoint circuit breaker (`[agents] failure_threshold`, `failure_cooldown_seconds`). After N failed calls in a row, calls are paused (`ProviderUnavailableError`, no request sent, agents vote HOLD at once), then a single probe call is made. Usage reports count paused calls as `skipped`.
- Live trader: exchange or network outages (`DataError`, `OSError`) leave the state untouched and are retried with exponential back-off (at most 15 minutes). If a cycle's transaction fails with an SQLite operational error, the trader reloads its state from the database and retries the same bars, reusing cached answers. Each cycle records `health` (last check, consecutive errors, last error) in the run state. `CycleReport.agent_errors` counts HOLDs caused by failures.
- The CLI reports retries and agent failures; the dashboard shows the last check time and an outage warning.
- `docs/FAILURE_RECOVERY.md` lists every failure, what happens, and whether it is recovered or stops the process.
- Tests: the breaker opens, probes, recovers and can be disabled; paused agents vote HOLD and count as skipped; data outages back off (DataError, connection reset, timeout) and recover; the back-off cap; a locked database mid-cycle gives no lost or duplicate orders or decisions and no repeated question; health in the dashboard; guide coverage.

### Stage 10E: Reproducible AI experiment protocol ✅
- `docs/EXPERIMENT_PROTOCOL.md` covers:
  - the hypotheses and experiments A–D (baseline, + Trend, + Trend + Momentum, + all three) on identical periods, costs, liquidity, breakers and voting;
  - a record-then-replay walk-forward procedure and the metrics to report;
  - a decision rule fixed in advance (majority of folds with a sign test, a higher compounded out-of-sample return, a drawdown limit, a minimum number of trades, a second period, and a Bonferroni correction);
  - the pitfalls, including look-ahead through the model's training data.
- `research/protocol.py`: `summarize` (per variant: compounded out-of-sample return, benchmark, worst-window drawdown, mean Sharpe and profit factor, trades, exposure, folds won against the baseline) and `sign_test_p`. `experiment --walkforward` prints them and saves or exports them.
- Tests: the sign-test values quoted in the protocol, fold-by-fold comparison (ties and undefined values skipped, infinite profit factors not averaged), CLI output, saved and exported comparisons, and document coverage.

## 4. Stage 9/10 status summary

The Stage 6 agent framework is now a multi-agent paper-trading research system on the teacher's Qwen endpoint. The architecture is provider-agnostic: a new provider is one `LLMProvider` subclass.

| | |
|---|---|
| Provider | `llm/`: OpenAI-compatible client. URL, model and token from environment variables only. Timeouts, retries, a circuit breaker, redaction, usage tracking |
| Agents | `qwen_trend`, `qwen_momentum` (market-only, so answers are shared across experiments), `qwen_risk` (portfolio-aware). Strict JSON with labels; anything else is HOLD; signals only |
| Integration | Backtests, sweeps, walk-forward, experiments and live paper trading. Record, replay and live modes. Agents are asked only about traded bars, decision bars and new candles |
| Measurement | `agent-report` (attribution), `experiment` (baseline versus AI, out-of-sample comparison and sign test), usage accounting |
| Operations | `agent-test`, read-only data layer and Streamlit dashboard, systemd templates, environment file, `--run-id`, log file, SIGTERM, failure recovery |
| Safety | The voting engine, breakers, risk manager and paper executor sit between every agent and every simulated trade. No exchange keys or private endpoints (the safety scans still pass). No hard-coded tokens. The dashboard cannot write |

Trade-offs worth knowing:
- `qwen_risk`'s answers depend on the trading path, so each experiment variant asks it new questions. The other two agents can be made portfolio-aware with `portfolio_context = true`, at the same cost.
- An LLM may have seen historical prices during training. Final out-of-sample claims need periods after the model's training cutoff, and ultimately the live paper run (see the protocol).
- `weight = 0` now means a strategy does not take part at all. Before, a zero-weight vote still counted towards `min_agreeing`.

### Later
Short positions, trailing stops, order-book data, more LLM providers (OpenAI, Anthropic, Gemini as `LLMProvider` subclasses), and alerting on breaker trips or long outages.
