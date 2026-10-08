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

### Stage 11A: Trailing stops and take-profit ✅
- `[risk] trailing_stop_pct`, `trailing_activation_pct` and `take_profit_pct` (0 = off; the defaults leave behaviour unchanged).
- `TradingSession` tracks each open position's highest high and raises its stop at bar closes, effective from the next bar (no look-ahead inside a bar). Stops only move up (`Portfolio.set_stop`). Take-profit exits at average cost × (1 + pct), or at the open after a gap up. When both the stop and the target are reached in one bar, the stop wins. There is a new `DecisionAction.TAKE_PROFIT`.
- The stop-loss cooldown now follows only losing stop exits. With the default exits every stop exit is a loss, so nothing changes there.
- Trailing state (`trailing`: highest high and raised stop) is saved with live runs and re-applied after the portfolio is rebuilt from fills on resume. The dashboard shows the current stop.
- Tests: ratcheting and a profitable trailing exit without cooldown; a raised stop applying only from the next bar (gap rule); activation; take-profit with a gap; the stop winning a tie; unchanged defaults; validation; backtest equal to live with these exits; trailed stops surviving a resume; the dashboard stop.

### Stage 11B: Run summaries ✅
- `summary.py`: `build_summary(store, run_id, hours=24)` (equity and its change in the window, return since start, current and in-window drawdown, the market's equal-weight move, closed trades, open positions, decision counts, breaker trips, per-agent votes and latest rationale, model usage, health) and `format_summary` (Markdown). It uses reads only.
- CLI: `trading-lab summary [RUN_ID] [--hours N]`, which opens the database read-only.
- Tests: the whole-run summary equals the backtest result; window filtering of trades, equity change and votes; the Markdown content; breaker, health and empty runs; the database byte-identical after the command.

### Stage 11C: Alerts ✅
- `alerts.py`: `WebhookNotifier` (ntfy, Slack, Discord or JSON; URL only from `TRADING_LAB_ALERT_URL`, never logged, errors name only the host), `AlertManager` (level filter, per-key repeat suppression, never raises), and `build_alerts`. `[alerts]` holds enabled, format, min_level, daily_summary, outage_after_cycles and repeat_after_minutes.
- `LivePaperTrader(alerts=...)`:
  - breaker trips (critical for the kill switch, warning for the daily limit);
  - entries, exits, stops, take-profits and closed trades (info);
  - model calls paused and recovered;
  - repeated failed cycles and their recovery;
  - database rollbacks;
  - a daily summary once per UTC day, persisted across resume.
- The CLI sends run start, stop and crash alerts and fails fast if alerts are enabled without the URL. There is a new `trading-lab alert-test`. Alert settings come from the current config.
- Bug fix: resuming compares configs by value instead of by fingerprint. Before this, every new config setting since 9A would have stopped runs saved by older versions from resuming.
- Tests: each format; no URL leaks; config and environment checks; levels and repeats; breaker, outage, model-pause and recovery alerts from real trader runs; the daily summary across midnight and a resume; failing alerts not changing any fill; the CLI; resuming a run saved by an older version.

### Stage 12A: ATR stops and volatility-scaled sizing ✅
- `[risk] stop_mode = "percent" | "atr"`, `atr_period`, `atr_stop_multiple`, `atr_stop_min_pct` and `atr_stop_max_pct`. The defaults keep the fixed percentage stop.
- `MarketStats.atr`: the simple average true range of the bars before the fill. It is computed the same way by `market_stats_frame` (backtests) and `next_bar_stats` (live fills at the new candle's open), so both paths agree exactly.
- `RiskManager.stop_distance_pct` / `stop_price_for(fill, stats)` give the ATR distance, clamped, falling back to `stop_loss_pct` without history. Risk-per-trade sizing uses it, so the loss at the stop is the same share of equity for every coin. `stop_basis` and `stop_distance_pct` are recorded in each entry's sizing details.
- Tests: ATR identical for the backtest and live paths and causal; modes, clamping and fallback; equal risk with a quarter of the size at four times the ATR; a backtest in ATR mode; live equal to the backtest in ATR mode (fills and stops); validation.

### Stage 12B: Adaptive agent weights ✅
- `research/weighting.py`: `WeightingRule` (horizon, min_votes, sensitivity, min/max weight) and `adaptive_weights`. The multiplier is 1 + sensitivity × (correctness − 0.5), clamped. Agents without enough measurable votes keep their weight, and deterministic strategies are never touched. If every voter would be switched off, the current weights are kept.
- `walk_forward(..., adapt_agent_weights=True)`: per fold, the attribution of the training backtest sets the agents' weights for the test window. These are recorded as `WalkForwardFold.agent_weights` and exported by `experiment`.
- CLI: `walkforward` and `experiment --walkforward` take `--adaptive-weights`, `--weight-horizon`, `--weight-min-votes` and `--weight-max`. `trading-lab agent-weights [RUN_ID]` prints suggested weights and a config snippet.
- Tests: the rule's numbers, caps and floors; the all-off guard; fold weights equal to a separate training backtest's; changing the data after a training window leaves its weights unchanged (no look-ahead); fixed weights by default; the CLI.

### Stage 12C: Entry filters ✅
- `[risk] trend_filter_period` (simple moving average, `engine/filters.py`, identical in backtests and live; unknown means blocked), and `block_entries_on_risk_states` with `risk_state_max_age_bars`. The session remembers the latest `risk_state` reported per symbol (saved for resume).
- `TradingSession.entry_filter_reason` is checked only when a BUY would be scheduled. A blocked BUY is recorded as IGNORED with the reason. Exits, stops, take-profits and breakers are unchanged, and engines load enough history for the average.
- Tests: blocking below the average and unknown averages; the veto and its age limit; no forced exits, exits still working and breakers still applying; no backtest entry signalled below the average; live equal to the backtest with the filter; an agent veto surviving a resume; validation and round-trip.

### Stage 12D: HTML run report ✅
- `html_report.py` and `trading-lab report [RUN_ID] --html FILE [--horizon N]`: one self-contained file with inline CSS and SVG and a tiny hover script, built from `DashboardData` (read-only). It covers key-number tiles, equity against buy & hold (legend, direct end labels, crosshair tooltip, at most 600 points), drawdown, a daily table view, the voter table, AI rationale cards, model usage, breaker trips, trades and decision counts. Light and dark themes come from validated palette steps. All database text is escaped.
- Tests: numbers match the run and the database is byte-identical afterwards; a `<script>`/`<img onerror>` rationale is rendered as text; no external resources; downsampling keeps the last point; empty runs; the CLI.

### Stage 13A: Readiness check ✅
- `doctor.py` and `trading-lab doctor [--online]` check:
  - the Python version and required/optional packages;
  - that the config is valid (fingerprint), the enabled voters and the public-data-only safety line;
  - the model environment variables (set or missing, depending on the agents mode; values never shown) and the alert URL;
  - the database (read-only: schema, runs, running paper runs, writability), the answer cache (replay needs recorded answers), disk space and the data cache directory.
  - `--online` adds one public candle (stale data is a warning) and one model call through the smoke test.
  - The exit code is 1 on any failure.
- `docs/DEPLOYMENT.md` runs it after the environment file is created.
- Tests: a fresh install is ready and creates nothing; agents without or with variables (no secret printed); replay requirements; alert URL; an invalid config; an existing or newer database (left byte-identical); online fresh, stale and down data and the model call; CLI exit codes.

### Stage 13B: Correlation-aware exposure ✅
- `[risk] max_correlated_positions` (0 = off), `correlation_threshold` and `correlation_lookback`. `engine/filters.correlation_lookup` computes the rolling log-return correlation between every pair of symbols up to the signal bar (fixed window, time-aligned, gaps give unknown), and the engines pass it to the session.
- `entry_filter_reason` counts open, working-limit and already-scheduled entries whose correlation with the new symbol is at or above the threshold (unknown counts as correlated) and blocks the entry when there are `max_correlated_positions` of them. It only blocks new entries.
- Tests: perfect, independent and too-short correlations and causality; blocking, allowing and unknown; two entries scheduled in the same bar; perfectly correlated twins never held together (and held together without the limit); live equal to the backtest; validation.

### Stage 13C: Robustness ranges ✅
- `research/robustness.py`: a seeded trade bootstrap (total-return range and probability of a loss) and a block bootstrap of per-bar equity returns (blocks of about √n bars; total-return and Sharpe ranges), with warnings for fewer than 30 trades, ranges spanning gains and losses, and too few bars.
- CLI: `trading-lab robustness [RUN_ID] [--samples N] [--seed S]` (read-only). It also appears as a section of the HTML report, and the experiment protocol points to it.
- Tests: deterministic per seed with ordered percentiles; known cases (all winners, a coin flip, constant returns); warnings and edge cases; stored runs agreeing with the metrics (return and Sharpe); the CLI (read-only) and the report section.

### Stage 13D: Benchmark-relative metrics ✅
- `metrics/relative.py`: from aligned per-bar returns, the excess return, beta, annualised alpha, correlation, tracking error and information ratio (None when undefined).
- `BacktestResult.relative` (also stored in the run's metrics as `relative`), printed by `backtest` and `report`. Paper runs get it from stored bars through `DashboardData.research`. It also appears in the dashboard, the HTML report tiles and `experiment` summaries.
- Tests: hand-made curves (identical, half exposure, constant extra return, cash, invalid input); the backtest stored and printed values and their agreement with the metrics; paper runs from bars; experiment summaries.

### Stage 14A: Paper-run reconciliation ✅
- `research/reconcile.py`: `reconcile(store, run_id, market)` takes a paper run's stored bars and config, compares the stored candles with fresh ones (revised or missing bars), then backtests the same period with agents in replay mode and end liquidation off. It compares fills (time, symbol, side, quantity, price) and non-HOLD decisions as multisets. The live fills after the last processed bar and the backtest's own "data ended" expiries are left out.
- CLI: `trading-lab reconcile RUN_ID [--synthetic SEED] [--limit N]` opens the database read-only, takes the seed of a synthetic run from its exchange name, and exits 1 on any difference.
- **Fix it found:** the Trend agent's EMA 200 depended in the 5th digit on where the data window started, so a live run (sliding window) and its backtest (growing window) could show the model different numbers and miss each other's cached answers. `indicators.windowed_ema` computes it over exactly the last 1000 bars, which does not depend on the window. Values move by about 1e-5 at most, and only `qwen_trend` cache keys change.
- Tests: a live run matches its backtest (fills, decisions, bars); late fills left out; other market data and a tampered fill are reported; backtests, unknown runs and runs without bars; agents replayed without model calls, and missing cached answers counted; end liquidation; the CLI (exit codes, the database byte-identical); `windowed_ema` (equal to a restarted EMA, independent of the data start, causal, validated).

### Stage 14B: Market-data quality report ✅
- `data/quality.py`: `check_candles` (pure) reports errors (fetch failure, no candles, stale data when checking up to now, with a one-candle grace for publishing delay) and warnings (gaps, candles missing at the start or end, zero volume, no price range, extreme moves, opens far from the previous close between consecutive candles). An extreme move has to beat a robust threshold: `max(jump_floor, expm1(jump_sigmas × 1.4826 × MAD of log returns))`. `check_market_data` fetches through any provider. `check_stored_bars` checks a run's stored candles.
- CLI: `trading-lab data-check [--symbols ...] [--days N | --start/--end] [--run RUN_ID] [--jump-floor F] [--jump-sigmas K] [--strict] [--limit N]`. It exits 1 on errors, and with `--strict` on warnings too.
- Tests: clean data (a random walk and synthetic candles); gaps and missing edges; zero-volume and flat candles; extreme moves on calm, volatile and strict settings; open gaps (and none after a data gap); stale versus publishing-delay versus a fixed past period; fetch failures and empty data; rule validation; stored bars of a run; the CLI (exit codes, strict, invalid rules, read-only run mode).

### Stage 14C: Agent answer quality ✅
- The specialist agents declare what their prompt asks for: `contradicting_votes` (label value -> the vote that contradicts it) and `hold_labels` (values that call for HOLD).
- `research/agent_eval.py`: `evaluate_agents(signals)` and `evaluate_run(store, run_id)` read the decision-bar signals of every agent. They count answers, error kinds, votes, labels, BUY/SELL confidences (zero-confidence votes), short rationales, the most repeated rationale, contradictions, BUY/SELL votes on HOLD labels, and model calls (failures, latency). Warnings flag no usable answer for more than 10% of decisions, contradictions, more than 25% of votes on HOLD labels, missing labels, zero-confidence votes and short rationales. With enough answers (`min_answers`, default 20), they also flag 95% or more HOLD, one rationale in more than half the answers, one-sided votes (90% or more) and two or fewer distinct confidences.
- CLI: `trading-lab agent-eval [RUN_ID] [--min-answers N] [--limit N] [--json]` (read-only).
- Tests: a well-behaved fake agent has no warnings; fixed labels, confidence and rationale are flagged (contradictions, HOLD-label votes, flat confidence, boilerplate, always HOLD); one-sided zero-confidence answers with empty rationales; invalid answers by type; small samples; error kinds; runs without agents and unknown runs; the CLI (text, JSON, latest run, exit codes, the database byte-identical).

### Stage 14D: Run export ✅
- `export.py`: `export_run(store, run_id, dir, holds=False, overwrite=False)` writes `equity_curve.csv`, `trades.csv`, `fills.csv`, `decisions.csv`, `signals.csv` (agent rationale, label, cache, error and reason flattened out of the metadata), `bars.csv` and `summary.json`. The summary holds the run, config, stored or recomputed metrics, decision counts and the rows and SHA-256 of each file. A non-empty directory is refused unless `overwrite` is set, which replaces only the export files.
- `trades_frame` and `fills_frame` are shared with `backtest --export`, so the two give identical files.
- CLI: `trading-lab export RUN_ID DIR [--holds] [--force]` (read-only).
- Tests: backtest export (files, rows, checksums, metrics, equity); identical to `backtest --export`; HOLDs on request; a paper run; agent votes readable and the API key never written; directory protection (non-empty, a file, unknown run, `overwrite` keeps foreign files); the CLI (exit codes, the database byte-identical).

### Stage 15A: Offline demo ✅
- `demo.py`: `build_demo(dir, seed=7, days=60, paper_bars=72, now=None)` works in a new or empty directory with the built-in defaults (agents off, no CSV cache). It runs a backtest on synthetic candles, then a paper run (`demo-paper`) of the latest synthetic hours driven by a simulated clock, and reconciles that paper run with its backtest. It writes `demo.db`, `backtest-report.html` and `paper-report.html`. `next_steps` lists commands pointing at the demo database.
- CLI: `trading-lab demo [DIR] [--seed S] [--days N] [--paper-bars N]`. It exits 1 if the reconciliation differs.
- **Bug it found:** a run without losing trades stores its profit factor as the text `"inf"`, which crashed the HTML report and the dashboard formatters. They now show text values as text, escaped in HTML.
- Tests: a complete sample (files, runs, paper bars ending at the last closed candle, reports); deterministic per time and seed; existing files never touched; next steps; the CLI (writes only the demo directory, and the result reconciles from the command line); text metrics in the HTML report and the dashboard.

### Stage 15B: E-mail alerts ✅
- `[alerts] channels` is a list of `"webhook"` and/or `"email"`, default `["webhook"]`, so behaviour is unchanged unless you opt in.
- `alerts.EmailNotifier` sends a plain-text e-mail over SMTP. The settings (`TRADING_LAB_SMTP_HOST`, `_PORT`, `_SECURITY`, `_USER`, `_PASSWORD`, `TRADING_LAB_ALERT_EMAIL_TO`, `_FROM`) come only from the environment.
  - Security is `starttls` (default, port 587), `ssl` (465), or `none` for a local relay without a login only, so a login is never sent unencrypted.
  - Addresses are validated (no header injection) and titles are put on one line.
  - Failures become `ConnectionError`s that name only the host, port and error type, never the login.
- `MultiNotifier` delivers to every channel and fails only when none delivered. `build_notifier` builds the configured channels and fails fast on missing settings.
- `alert-test` tests each configured channel (or `--channel`). `doctor` reports each channel without values.
- Tests: an encrypted e-mail (STARTTLS before login, headers, body); SSL and a local relay; ten invalid settings; the login never shown (repr, errors, logs); unreachable servers never raise from the manager; header injection; several channels; config validation (no credential-like fields); `doctor` and `alert-test`.

### Stage 15C: Anthropic provider ✅
- `OpenAICompatibleProvider` gained hooks (`_request_url`, `_request`, `_content`, `default_url`, `retryable_status`, `required_env`), so a provider with another wire format reuses the environment handling, retries, circuit breaker, redaction and usage accounting. Qwen behaves exactly as before.
- `llm/anthropic.py`: `AnthropicProvider` (`provider = "anthropic"`) calls the Messages API with `x-api-key` and `anthropic-version`, the system prompt as `system`, and joined text blocks. It reports `stop_reason` and input/output tokens. It also retries HTTP 529. `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` are required and `ANTHROPIC_API_URL` is optional (default `https://api.anthropic.com`), all from the environment only.
- `doctor` and `agent-test` know which variables each provider needs. Agents keep their names. Cache keys include the provider and model.
- Tests: URL forms; the exact request (headers, body, default endpoint); text blocks joined and other blocks ignored; five malformed responses (not retried); 529 retried; 401 fails fast without leaking the key (errors, repr, logs); environment checks (replay needs only the model; custom and invalid URLs); config and factory; the three agents running a backtest on the Messages format; `doctor` and the `agent-test` environment lines.

### Stage 16A: Short-position accounting ✅
- `Position.side` (`long` or `short`) and `entry_notional`, plus `ClosedTrade.side`. A short's `avg_entry_price` is its break-even (net proceeds per unit), and `market_value(p) = 2 x entry notional - quantity x p`.
- `Portfolio(allow_short=False)`:
  - With shorts allowed, a SELL without a long opens or adds to a short and a BUY against a short covers it.
  - Shorts are fully collateralised: the entry notional plus the fee leaves cash, so there is no leverage. Covering returns the collateral plus the gain, or minus the loss, which can exceed the collateral.
  - No fill flips a position. `gross_exposure` sums |quantity x price|. The equity invariant holds.
  - With shorts off, behaviour and messages are unchanged.
- `PaperExecutor(borrow_bps_per_day=0)`: opens and covers shorts with side-correct slippage. The borrow fee (entry notional x bps x days held) is added to the cover fill's fee, so a portfolio rebuilt from its fills (resume) is identical. Short entries need collateral, and covers can never exceed the short.
- Storage schema v4: `closed_trades.side` (existing trades are longs). The migration is idempotent, and read-only access to older databases still works.
- Tests: long-only by default; a round trip with exact cash, fees, PnL and break-even; partial covers and adding to a short; losses beyond the collateral; no flips; collateral required; a randomized 400-fill replay keeping the invariant and rebuilding identically; executor slippage, borrow fee, limits, dust and limit fills; storage of the side; v3 upgrade and read-only access.

### Stage 16B: Shorts in the trading session ✅
- Config: `risk.allow_short` (default false) and `execution.short_borrow_bps_per_day` (default 2.0, used only by shorts).
- `TradingSession`:
  - An ensemble SELL with no long open schedules a short entry, and a BUY covers it. A long is closed before any short (no flips). With shorts off, every message and branch is unchanged.
  - Stops trigger on the high, filling at the stop or at a higher gapped open. Take-profit triggers on the low. Trailing stops follow the lowest low and only move down.
  - Limit short entries rest at `open x (1 + offset)` and fill when the high trades through. A BUY signal cancels a working short entry.
  - The trend filter is mirrored. The portfolio view reports `short` and the stop distance above.
- `RiskManager`: `evaluate_entry(side=SELL)` is mirrored. The stop is above. The loss at the stop is the buy-back cost minus the net sale proceeds. Exposure is absolute (`gross_exposure`). The cash limit is the collateral. `evaluate_exit` buys back shorts. `stop_triggered` uses the high for shorts.
- The live trader and the dashboard rebuild portfolios with shorts allowed when the run allowed them.
- Tests:
  - long-only by default (and validation);
  - SELL opens and BUY covers;
  - stops on the high and gaps;
  - take-profit, and stop first;
  - trailing stops that never loosen;
  - long closed before a short;
  - pyramiding;
  - mirrored sizing, including short exposure;
  - mirrored filters;
  - the kill switch covering shorts;
  - limit short entries;
  - the portfolio view;
  - backtest equal to live with both sides traded and borrow fees charged;
  - a short surviving a resume.

### Stage 16C: Shorts in reports ✅
- Attribution: a trade's entry vote is BUY for a long and SELL for a short. `agreed`, `disagreed` and `pivotal` follow it. `_counterfactual` replays the voting rules for both directions.
- Trade side in:
  - `report` (a column);
  - `backtest` (shorts per symbol, only when shorts are allowed, so long-only output is unchanged);
  - `export` and `backtest --export` (a `side` column);
  - the HTML report (a Side column);
  - the dashboard (side of open positions and recent trades);
  - `summary` ("N short").
- Dashboard exposure counts quantity x price while shorts are open, since a short's book value is its collateral plus gain.
- Tests: SELL votes agreeing and pivotal for a short trade (and BUY disagreeing); the CLI report and unchanged long-only backtest output; export, HTML and dashboard trades; open shorts on the dashboard (side, stop above, value, exposure) and in the summary.

### Stage 17A: Trend-following strategies ✅
- `strategies/trend.py`, both opt-in (only enabled by a config table):
  - **`ma_cross`:** SMA or EMA fast/slow crossover. `signal_on = "cross"` signals on the cross bar, `"state"` on every bar above or below. Confidence is the gap change, or the gap, over its rolling standard deviation.
  - **`donchian`:** a close beyond the previous `entry_period` high or low is a breakout. An optional `exit_period` break gives a weaker (0.5) opposite signal. Confidence is the breakout distance in ATRs. The channels exclude the current bar.
- `signals` shows their key values. `config/default.toml` documents them as commented tables.
- Tests:
  - the shared strategy contract (warm-up, standardised signals, both directions, determinism);
  - vectorised signals equal bar-by-bar signals for four parameter sets;
  - crosses exactly where the gap changes sign;
  - state mode following a trend;
  - breakouts, inside-channel holds and weak exits;
  - the channel excluding the current bar;
  - validation;
  - opt-in from the config, with a backtest trading both sides.

### Stage 17B: Cost sensitivity ✅
- `research/costs.py`: `cost_sensitivity(config, provider, start, end, multipliers)` backtests the same period with the fee rates, slippage, impact coefficient and borrow fee scaled together. The data is fetched once and shared, and the agent answers come from the shared cache. The 1x row is exactly the normal backtest.
- `CostSensitivity` gives the break-even multiplier (linear interpolation where the return crosses zero), a verdict (no edge before costs, still profitable at the highest multiplier, or a thin or roomy break-even) and `to_dict`.
- CLI: `trading-lab costs [--days N | --start/--end] [--multipliers 0,0.5,1,2,3] [--export JSON]` (nothing stored).
- Tests: every cost scaled (and only costs); invalid multipliers; rows sorted and deduplicated; one data fetch per symbol; fees increasing; the 1x row equal to a plain backtest; break-even interpolation and every verdict; the CLI (export, bad input, no database).

### Stage 17C: Performance by market regime ✅
- `research/regimes.py`:
  - `market_index` compounds the equal-weight mean of the symbols' per-bar returns.
  - `classify` labels the trend (up, down, sideways) from a `trend_bars` SMA and its slope over `slope_bars`. It is causal, and the first bars are warm-up. It also labels volatility (volatile or calm) against the run's median rolling volatility, as an after-the-fact description.
  - `regimes_for_run` reports, per trend, per volatility and per combination: bars, time share, compounded strategy and market return over those bars, time in the market, and trades by entry bar (count, wins, PnL). Trades opened during the warm-up are counted separately.
- CLI: `trading-lab regimes [RUN_ID] [--trend-bars 50] [--slope-bars 10] [--vol-bars 24] [--json]` (read-only).
- Tests:
  - trend labels on rising and falling series;
  - warm-up;
  - causality;
  - the index on hand values;
  - validation;
  - regimes adding up to the run: bars, time, trades and PnL, and compounded returns equal to the post-warm-up return;
  - short and unknown runs;
  - the CLI (text, JSON, options, exit codes, the database byte-identical).

### Stage 18A: A/B comparison of configs ✅
- `research/ab.py`:
  - `ab_test(config_a, config_b, provider, start, end, windows=6, metric="total_return")` runs a fresh backtest of each config in every one of N equal, consecutive, non-overlapping windows. Data and agent answers are shared.
  - `ABResult` reports wins per side (ties and undefined values left out; lower is better for drawdown, volatility and fees), one-sided sign-test p-values for each side, compounded returns, and a verdict (significant, a lead that could be chance, a tie, or not comparable).
  - `config_diff` lists every setting that differs.
  - Different symbols, timeframe or initial cash are refused.
- CLI: `trading-lab ab A.toml B.toml [--days N | --start/--end] [--windows 6] [--metric M] [--export JSON]`. Command-line market overrides apply to both configs. Nothing is stored.
- Tests: the config diff (including added strategy tables); window arithmetic and validation; each window equal to a standalone backtest; identical configs tie; sign tests and every verdict, including lower-is-better metrics and undefined values; unfair comparisons refused; the CLI (export, bad windows, missing file, shared overrides).

### Stage 18B: Paper-run watchdog ✅
- `status.py`: `check_status(store, run_id, now, max_behind=2, max_errors=3, expect_running=True)` reads a paper run (read-only) and reports:
  - closed candles not yet processed (`behind`, from the saved last processed candle and the timeframe);
  - consecutive failed cycles and the last error (from the trader's saved health);
  - the run status and the latest equity.

  Problems are being stalled beyond `max_behind`, `max_errors` failed cycles, a run that is not running (a note only with `expect_running=False`), and no cycle at all long after creation. `latest_paper_run` prefers the running paper run.
- CLI: `trading-lab status [RUN_ID] [--max-behind N] [--max-errors N] [--allow-stopped] [--alert] [--json]`. It exits 1 when not OK. `--alert` sends one critical alert through the configured channels.
- `deploy/systemd/trading-lab-watchdog.service` (oneshot, read-only paths, the environment file only for alert settings) and `.timer` (every 15 minutes) are documented in `DEPLOYMENT.md` (section 5c).
- Tests: a run keeping up (one waiting candle is normal); a stalled run; stopped runs (strict and relaxed); failing cycles up to the threshold; choosing the run and rejecting backtests and unknown ids; the CLI (exit codes, JSON, an alert sent without leaking the URL, alerts disabled, read-only); the templates.

### Stage 18C: Volatility-targeted sizing ✅
- `risk.position_volatility_pct` (default 0, off) adds a `volatility_target` candidate to entry sizing: `equity x pct / (per-bar volatility x sqrt(bars per year) x fill price)`. It uses `MarketStats.volatility` from the bars before the entry, so backtest and live are identical. It works for shorts too.
- `RiskManager(bars_per_year=...)` is set from the timeframe by the session, and is required when targeting is on.
- Tests: the exact limit (one position adds the target share of equity in volatility; half the volatility gives twice the size; shorts the same); off by default and skipped without volatility; validation; backtest entries bound by the target, with the calmer coin getting the bigger share; live equal to the backtest.

### Stage 19A: Permutation test ✅
- `research/permutation.py`:
  - `permute_candles` reorders the candles from the period start. Each candle is kept as log open, high, low and close relative to the previous close, plus volume. The order is the same for all symbols with the same number of candles, and prices are rebuilt from the last real close. The warm-up history and the final price stay exactly the same.
  - `PermutedProvider` serves shuffled data through the shared memoized provider.
  - `permutation_test` runs the real backtest plus N shuffled ones. `PermutationResult` gives the metric of each run, the 5/50/95 percentiles, how many were at least as good (lower is better for drawdown and the like), p = (1 + count) / (1 + N), and a verdict.
  - Agents are refused unless `allow_agents` is set.
- CLI: `trading-lab permutation-test [--days N | --start/--end] [--permutations 100] [--metric M] [--seed S] [--allow-agents] [--export JSON]` (nothing stored).
- Tests:
  - shuffling keeps the history, the final price, the candle shapes and volumes, and stays valid OHLC;
  - it is deterministic per seed;
  - the shared order across symbols;
  - edge cases;
  - structure is detected (a trend follower on autocorrelated returns beats all shuffles), while noise is not (a random walk);
  - p-values and verdicts, including lower-is-better metrics and undefined runs;
  - validation (agents, counts, metrics);
  - the CLI.

### Stage 19B: Sweep stability ✅
- `research/sweep.py`: `stability_scores(results, grid, metric)` gives each grid point the mean metric of itself and its neighbours: points one step away in one parameter, in the order the values were given. Undefined values are skipped and their count is reported. `rank_by_stability` orders by that score (lower is better for drawdown and the like). `params_key` identifies a grid point.
- `sweep` prints a `stable` column (and writes it to the CSV export). `--rank stability` orders the table by it.
- Tests: a lone peak scores below a plateau, and the plateau ranks first; two-dimensional neighbours and undefined values; lower-is-better metrics; the CLI (column, ranking, export).

### Stage 19C: Market regimes in the report and dashboard ✅
- `DashboardData.regimes(run_id)` returns the read-only regime report as a dict, or None while the run is shorter than the warm-up. It is part of `snapshot()` and therefore of `dashboard-data --json`.
- The HTML report has a "Market regimes" section (per trend and per trend x volatility: bars, time, strategy and market returns, time in the market, trades, PnL). The dashboard's Research section shows the same table.
- Tests: long runs show regimes identical to `regimes_for_run`, in the snapshot and the HTML report; short runs leave the section out; the dashboard renders the table (Streamlit AppTest).

### Stage 20A: Strategy checkup ✅
- `research/checkup.py`: `checkup(config, provider, start, end, permutations=50)` runs:
  - the backtest, in an in-memory store, which feeds `robustness_for_run` and `regimes_for_run`;
  - `cost_sensitivity` at 0x, 1x and 2x;
  - `permutation_test`, skipped (n/a) when AI agents vote, unless allowed.

  `evaluate` grades eight checks as pass, warn, fail or n/a, each with a detail and a next step. The advice depends on whether the run lost money. `Checkup.overall` is the worst check, and `verdict` names the failed or warned checks. `format_checkup` gives the text, `checkup_html` a self-contained page (escaped, wrapping table) and `to_dict` the JSON.
- CLI: `trading-lab checkup [--days N | --start/--end] [--permutations 50] [--allow-agents] [--html FILE] [--json FILE]` (nothing stored).
- Tests: a real edge (a trend follower on autocorrelated returns) passes edge, costs, luck, trades and resampling; no edge (a random walk) fails, with the right advice and JSON; agents skip the luck test; short periods; the verdict logic; the HTML page (escaping, self-contained); the CLI (HTML and JSON files, no database).

### Stage 20B: Time stop ✅
- `risk.max_holding_bars` (default 0, off). At the close of the bar in which a position reaches that many bars held (the entry bar counts), the session schedules its exit for the next open. The exit is a protected `risk_manager` intent, like the kill switch's, so strategies cannot cancel it. It is skipped when an exit is already scheduled or the position was stopped out. The exit decision reads `time stop: held N bars`, and shorts are covered alike.
- Tests: the exit timing and reasons; strategies cannot cancel it, and shorts; off by default; a stop-loss first; validation; backtest and live equal across a resume one bar before a time stop.

### Stage 20C: Regime-dependent strategy weights ✅
- `VotingConfig.regime_weights` (`[voting.regime_weights.<up|down|sideways>]` tables of strategy multipliers in [0, 100]), plus `regime_bars` and `regime_slope_bars`. It is normalised to sorted tuples, round-trips through TOML, `to_dict` and `from_dict`, and must only name configured strategies. It is off when empty.
- `engine/filters.trend_labels` is the causal SMA-and-slope trend label, shared with `research.regimes.classify`. `filter_columns(candles, risk, voting)` adds a per-bar `regime`, None during warm-up. The engines load enough history for it.
- `VotingEngine.combine(signals, multipliers)` applies the multipliers. The votes record the effective weights. A bar where every weight is 0 is a HOLD with a reason. The session passes the bar's regime multipliers and records `regime` on the ensemble signal.
- Tests: config parsing, validation and round trips, with older configs still loading; labels identical to the regime report and causal; filter columns; voting with multipliers (and unchanged without); the session using each bar's regime (none during warm-up); backtests changing with the table, with live matching the backtest.

### Stage 21A: Monthly returns and more metrics ✅
- `PerformanceMetrics` gains `calmar_ratio` (annualised return over max drawdown, None when undefined) and `max_drawdown_bars` (the longest stretch below an earlier peak). Both have defaults, so nothing that builds metrics breaks.
- `metrics.monthly_returns(equity, initial)` builds a year x month table: each month's last equity against the previous month's (the initial cash for the first month), NaN for months without bars, and a compounded year column.
- It is shown in `report RUN_ID` (text table), the HTML report ("Monthly returns", colour-coded), the dashboard's Research section, and the `dashboard-data` snapshot (`DashboardData.monthly_returns`).
- Measured along the way: a 90-day, four-symbol backtest takes about 0.8 s. Micro-optimising signal creation saved about 1%, so it was not kept.
- Tests: Calmar and the longest drawdown on hand values; the monthly table across a year boundary, with gaps and compounding; the dashboard data, snapshot, HTML and CLI views, where the months compound to the run's return; the dashboard rendering.

### Stage 21B: Telegram alerts ✅
- `alerts.TelegramNotifier`: a plain-text bot message to one chat (`sendMessage`, up to 4,000 characters, link previews off). The settings `TRADING_LAB_TELEGRAM_BOT_TOKEN` and `TRADING_LAB_TELEGRAM_CHAT_ID` come only from the environment and are validated.
  - The token is part of the API URL, so `repr`, errors and logs never contain it.
  - HTTP errors name the variable to check, and transport errors drop their message.
- The `telegram` channel works in `[alerts] channels` (alone or with the others), `alert-test --channel telegram` and `doctor`. The environment template, the README and `DEPLOYMENT.md` explain the setup.
- Tests: the exact request (URL, chat, text, run id, truncation); settings validation; the token never in `repr`, errors or logs, and an unreachable API never raises from the manager; channel combinations; `doctor` and `alert-test`.

### Stage 21C: Config presets ✅
- `presets.py`: four presets (`trend`, `trend-shorts`, `conservative`, `mean-reversion`) as overrides on the defaults.
  - `render` writes TOML with only the differing settings, nested tables as sub-tables (no empty parent headers), and the strategies table in full when it changed, because a `[strategies]` table replaces the defaults.
  - `preset_toml` adds an explanatory header.
- CLI: `trading-lab init-config [PRESET] [PATH] [--force]` lists the presets, or writes one (default `config/PRESET.toml`). It never overwrites without `--force`, checks that the file loads, and prints the `checkup` and `ab` next steps.
- Tests: every preset round-trips exactly through its file, differs from the defaults and backtests; files list only the changes (including regime weights and full strategy tables); the CLI (listing, default path, no overwrite, `--force`, a custom path, an unknown preset).

### Stage 22A: Several paper runs on one VM ✅
- `deploy/systemd/trading-lab-paper@.service`: a template service. The instance name is the run id, its config is `config/runs/NAME.toml`, and it logs to `paper-NAME.log`, with the same hardening and SIGTERM handling as the single service.
- `status.running_paper_runs(store)` lists every running paper run, sorted by id. `trading-lab status --all` checks them all:
  - each problem is prefixed with its run id, and one alert covers them all;
  - no running run at all is a problem unless `--allow-stopped` is set;
  - `--json` gives a list, and a run id together with `--all` is refused.
- The watchdog service now runs `status --all --alert`, so it covers one run or several. With a single run it behaves as before.
- `DEPLOYMENT.md` section 5d explains the setup: one config per run, the shared database, separate accounts, and model calls per run.
- Tests: two runs trading through two connections to one database file, then both OK; `--all` with a stalled run, a stopped one and a healthy one (exit code, alert text, JSON); no running run, with and without `--allow-stopped`; both systemd templates.

### Stage 22B: Comparing two live runs ✅
- `research/live_compare.py`: `live_compare(store, run_a, run_b, min_days=14)` compares two stored runs over their overlap (the later first bar to the earlier last bar).
  - Each run starts from its last equity before the overlap (or its initial cash). Its metrics come from `compute_metrics` over the overlap, counting only trades closed and fees paid within it.
  - Daily returns (last equity per UTC day) are paired. Ties are not compared, and `sign_test_p` gives the one-sided p for each run. The settings that differ come from `config_diff`.
  - Verdicts: not comparable (no overlap), too early to tell (fewer than `min_days` compared days), better (p < 0.05), a lead that could be chance, or no difference. Different kinds or timeframes are noted.
- CLI: `trading-lab live-compare RUN_A RUN_B [--min-days 14] [--json]`, read-only.
- Tests: hand-made curves where the overlap, base equity, fees, trades, exposure and each day's return are checked exactly; the sign test and every verdict; notes; two real paper runs started a day apart (no head start) and the CLI (text, JSON, errors).

### Stage 22C: All paper runs at a glance ✅
- `DashboardData.paper_runs(now=None, max_behind=2)`: one row per paper run, with running runs first and then by id. Each row has status, timeframe, symbols, last bar, equity, return, max drawdown (the same formulas as `overview`), open positions, candles waiting, and the watchdog check.
  - The check is "OK" or the `check_status` problems for a running run, and None for a run that is not running.
  - It is also part of the `dashboard-data` snapshot (`paper_runs`).
- The dashboard shows a "Paper runs" table above the portfolio when there are at least two paper runs, with a pointer to `live-compare`. `dashboard-data` prints the same list. A single run looks exactly as before.
- Tests: rows against each run's overview, the order, a stopped run (not checked), a run without bars, a stalled check later, the JSON snapshot, the text view, no table for a single run, and the rendered dashboard table, with the database unchanged.

### Stage 23A: Probabilistic and deflated Sharpe ratios ✅
- `metrics/sharpe.py` (Bailey and López de Prado):
  - `probabilistic_sharpe(returns, threshold=0)`: the probability that the true per-bar Sharpe is above the threshold, given the sample length, skewness and (non-excess) kurtosis;
  - `expected_max_sharpe(n, variance)`: the Sharpe ratio the best of n luck-only trials reaches;
  - `deflated_sharpe(returns, trial_sharpes)`: the PSR against that bar.
  - Undefined cases (fewer than 3 returns, no variance) give None.
- `PerformanceMetrics.probabilistic_sharpe` (with a default, so stored metrics still load) is computed by `compute_metrics`. It appears in the metrics table, `compare`, the HTML report and every export.
- `SweepResult.returns` keeps each trial's per-bar returns. `deflated_sharpe_of_best` scores row 1 against all the trials, and `sweep` prints it with a plain-language reading (95%+ convincing, under 50% likely luck). `probabilistic_sharpe` can also be a ranking metric.
- Tests:
  - the PSR against the formula by hand, and its properties: more data, skew direction, the threshold at the measured Sharpe gives 0.5, undefined cases;
  - the expected maximum against its formula and a simulation of 100 normals;
  - deflation rising with the number of trials, and the best of 50 noise strategies: PSR above 0.9 but DSR below 0.5;
  - the metrics, sweep returns that compound to each row's return, and the CLI, `compare` and HTML views.

### Stage 23B: Weekly digest ✅
- `digest.py`: `build_digest(store, days=7, now=None)` covers every paper run that is running or had bars in the period, sorted by id.
  - For each run, it reuses `summary.build_summary` (period return, market move, worst drawdown, trades, equity, open positions, model calls) and adds the watchdog check (`check_status`) for running runs.
  - It adds the `live_compare` verdict for each pair of running runs (at most 10 pairs, the rest counted).
  - `format_digest` and `to_dict` render it.
- CLI: `trading-lab digest [--days 7] [--max-behind 2] [--alert] [--json]`, read-only. `--alert` sends the text once, at info level but forced, so `min_level` does not hide it.
- `deploy/systemd/trading-lab-digest.service` and `.timer`: Mondays at 07:52 UTC, `Persistent=true`, read-only paths. Setup is in `DEPLOYMENT.md` section 5e.
- Tests: three paper runs (one stopped before the period, one started and stopped inside it), checked against `build_summary` and `live_compare`; a stalled check; the pair cap; an empty database; the CLI (text, JSON, alert despite `min_level`, no secret in the output, errors); the systemd units.

### Stage 24A: Trade analysis ✅
- `research/trades.py`: `analyze_trades(store, run_id, groupings=...)` groups a run's closed trades by exit, symbol, side, holding time (bars) and entry weekday, plus the entry hour on request.
  - `GroupStats` gives trades, wins, PnL, average return and bars, best, worst and profit factor.
  - The exit type comes from the decision that filled the exit order: stop-loss or trailing stop, take-profit, end of backtest, or for exits at the next open, the decision that scheduled them (time stop, kill switch, otherwise signal). The kill switch's scheduled exits were recorded only as "signal exit" before, and are now told apart.
  - Observations cover a result resting on its best few trades, the costliest exit type, symbol and side (at least 3 trades), and long holds against shorter ones (noting that stops close losers sooner by design).
- CLI: `trading-lab trades [RUN_ID] [--by ...] [--json] [--csv FILE]`, read-only.
- Tests:
  - the exit mapping;
  - on real backtests: groups adding up to the run's trades and PnL, time-stop trades held exactly `max_holding_bars`, long and short sides, kill-switch exits with `flatten_on_halt`;
  - group statistics and each observation on hand-made trades;
  - the CLI (text, JSON, CSV, a bad grouping, an unknown run) and a run without trades.

### Stage 24B: Trade breakdown in the reports ✅
- `DashboardData.trade_breakdown(run_id)`: groups by exit, side, holding time and symbol, with the totals and observations, without the per-trade list. It is None without trades and is part of the snapshot.
- HTML report: a "Where the money comes from" table (one row per group) with the observations. Dashboard Research section: "Trades by exit type" and "Trades by holding time" tables, with observations as captions. `dashboard-data`: a "Trades by exit" line.
- Tests: the data against `analyze_trades`, no section for a run without trades, the HTML report, the text view, and the rendered dashboard tables summing to the run's trades with the database unchanged.

### Stage 24C: "Sharpe is real" in the checkup ✅
- `checkup` gains a ninth check after "Robust to resampling". It passes at a probabilistic Sharpe ratio of 95% or more and warns from 80%; it is n/a when undefined. It has its own advice when the run lost money.
- The README example was regenerated from a fresh run (the demo's random-walk prices), and the thresholds list gained the new check.
- Tests: the full check list on a real edge (which passes it), and every grade boundary and the losing-run advice through `evaluate`.

### Stage 25A: A faster test suite ✅
- Profiling (`--durations`) showed the time spread over many live-against-backtest and resume tests. These are the safety net, so they stay as they are.
- `pytest-xdist` joins the `dev` extra. `pytest -n auto` runs all 752 tests in about 1 min 50 s on 4 cores, against 7 to 10.5 minutes one at a time. Every test passed in parallel, which shows they use only their own temporary files. It is opt-in, so plain `pytest` behaves as before.
- The digest fixture (the slowest setup at about 30 s) now simulates 3 days instead of 9 and tests a 2-day digest, with the same cases: a run stopped before the period, one started and stopped inside it, and two running. It takes about 10 s.
- README and DEPLOYMENT.md use `pytest -n auto`.

### Stage 25B: Trade excursions (MAE/MFE) ✅
- `research/trades.py`: each trade gets `mae` (0 or negative) and `mfe` (0 or positive) as fractions of the break-even entry price, as in the PnL.
  - They span the lows and highs of the stored candles from the entry bar up to the exit bar, plus the exit price; shorts are mirrored.
  - Runs without stored candles (before schema v3) get None.
- Groups gain `avg_mae` and `avg_mfe`. `Excursions` summarises:
  - the dip that 90% of winners stayed within, next to the fixed stop-loss (None for ATR stops, which differ per trade);
  - the losing trades that were at least 1% up at some point;
  - the e-ratio.
- The `trades` text, JSON and CSV include them.
- Tests: excursions by hand (long, short, never adverse, exit in the entry bar) and the summary by hand; every trade of a long and a long/short backtest recomputed independently from the bars, with the exit inside the excursion range; the stop shown or not; a run without candles; the CLI.

### Stage 25C: Excursions in the reports ✅
- `trades.excursion_sentences` turns the summary into plain sentences, from the object or its `to_dict()`. The text output, the HTML report and the dashboard share it.
- HTML report: MAE and MFE columns in "Where the money comes from", plus the summary sentences. Dashboard: `avg_mae` and `avg_mfe` columns in the trade tables, with the summary as captions. The snapshot's `trade_breakdown` carries `excursions`.
- Tests: the same sentences from the object and the dict; the snapshot against `analyze_trades`; the HTML columns and sentences, with the stop shown; the dashboard captions and columns, without a stop for ATR stops.

### Stage 26A: The research trial log ✅
- Schema v5: a `trials` table (command, label, config fingerprint, timeframe, symbols, period, return, annualised Sharpe, bars).
  - The migration is idempotent (`IF NOT EXISTS`), and read-only databases from before v5 read as an empty log.
  - `SQLiteStore.add_trials` and `load_trials(timeframe, start, end)`, where any overlap counts.
- `[storage] record_trials = false` (default off, so nothing changes; validated). When on, the CLI logs one row per evaluated config and period:
  - `backtest`;
  - `sweep` (every combination);
  - `ab` (each config in each window);
  - `checkup`;
  - `permutation-test` (the real run).
  It prints a one-line notice.
- `research/trials.py`:
  - `trial` / `record_trials` build and write the rows;
  - `TrialSummary` counts distinct (config, period) trials (repeats count once), distinct configs, per command, the best, the per-bar Sharpe variance and the luck bar (the expected best Sharpe of N luck-only trials, annualised);
  - `deflated_against_log` gives the DSR of a new result against them.
- CLI: `trading-lab trials [--days N | --start/--end] [--timeframe] [--json]`, read-only, with a note when recording is off.
- `checkup(trials=...)` adds "Beats your other trials": pass from a DSR of 95%, warn from 50%. It appears only when the log holds earlier trials on overlapping data, so a checkup without a log is unchanged. The CLI reads the log before running and logs the checkup afterwards.
- Tests:
  - storage (overlap rules, touching ends, timeframes, read-only v4 databases, the upgrade, and the three older upgrade tests now expecting v5);
  - the summary by hand (dedup, variance, luck bar);
  - the DSR with and without the new result already logged;
  - opt-in recording that creates no database when off;
  - every command logging the expected rows through the CLI (text and JSON);
  - the checkup check absent without a log, passing against weak trials and failing against 200.

### Stage 26B: The trial log in sweeps and the dashboard ✅
- `cli._earlier_trials` reads the logged trials overlapping a period, or None when recording is off. `checkup` and `sweep` share it.
- `sweep` prints "Against all N logged trials on overlapping data (this sweep included): DSR" next to its own deflated Sharpe, when the log holds earlier trials.
- `DashboardData.trial_log(run_id)` gives the trials overlapping the run's bars (count, configs, luck bar, best, the run's Sharpe), or None. It is in the snapshot, the dashboard's Research section (a caption) and `dashboard-data` (a line).
- Tests: the sweep line only with earlier trials and recording on, with the right count; the dashboard data, snapshot, text line and caption; no log, no line.

### Stage 27A: Forward risk (outlook) ✅
- `research/outlook.py`: `trade_returns` gives each closed trade's PnL relative to the equity just before it closed, in order. `simulate` draws `horizon` trades with replacement into `samples` compounding paths (seeded per run size and horizon). For each path it measures:
  - the max drawdown (median, 95th percentile, and the chance of reaching 10/20/30%);
  - the final return (5/50/95th percentiles, and the chance of a loss);
  - the longest losing streak (median and 95th percentile).
  The run's own trade-by-trade drawdown is shown for comparison. There are warnings for fewer than 30 trades and for horizons beyond 3x the run.
- CLI: `trading-lab outlook [RUN_ID] [--trades N] [--samples 5000] [--seed 7] [--json]`, read-only. A run without trades says so (JSON `null`).
- Tests:
  - drawdowns and streaks by hand (below the start, flat trades);
  - simulation properties: seeded, monotone probabilities, longer horizons deeper, all-winning and all-losing extremes, warnings, invalid input including `horizon=0`;
  - trade returns recomputed from a backtest's equity curve;
  - the CLI (text, JSON, errors, no trades).

### Stage 27B: The outlook in the reports ✅
- `DashboardData.outlook(run_id, samples=2000)` gives the outlook as a dict, or None without trades. It is in the snapshot.
- HTML report: "What to be ready for", a median and bad-case table (drawdown against the run's own, return with the chance of a loss, losing streak), the drawdown chances, warnings and the caveat.
- Dashboard Research section: four metrics (bad-case drawdown, chance of a 20% drawdown, chance of a loss, bad-case losing streak) with a caption. `dashboard-data`: an "Outlook" line.
- Tests: the data against `outlook_for_run`, the snapshot and text line, the HTML section (absent without trades), and the dashboard metrics.

### Stage 28A: Sizing to a drawdown budget ✅
- `research/sizing.py`: `entry_sizing` reads each filled entry's recorded size limits and matches them to its trade by symbol and entry time.
  - `size_factor(sizing, s) = min(s x risk_per_trade size, smallest other limit) / actual size`. Entries without stored sizing scale linearly. `headroom` is the scale at which another limit takes over.
  - `size_for_drawdown` scales each trade's return by its factor and runs the outlook simulation with the same draws at every scale. It bisects (geometrically, 0.05x to 10x) for the scale whose bad-case drawdown equals the target.
  - It reports the standard scales (0.25x to 2x) plus the chosen one, the suggested `risk_per_trade_pct`, which limit set each entry's size, and the median headroom.
  - Warnings: an unreachable target (too small, or capped by other limits), a losing run, `risk_per_trade` rarely setting the size, and trades without stored sizing.
- CLI: `trading-lab size [RUN_ID] --max-drawdown 20% [--trades N] [--samples 2000] [--seed 7] [--json]`, read-only. `--max-drawdown` accepts 0.2, 20% or 20.
- Tests: size factors and headroom by hand (risk-bound and capped entries, no sizing); on a backtest, the chosen scale hits the target, bad-case drawdowns rise with scale, and the limit counts match the entries; unreachable targets both ways; a run without stored sizing; percent parsing (a double-division bug for "150%" was caught and fixed); the CLI.

### Stage 28B: Verifying a size suggestion ✅
- `research/sizing.verify_sizing` backtests the stored config with `risk_per_trade_pct` set to the suggestion, over the same period, in memory. `Verification` has the actual drawdown, return and trades, the run's own at 1x, the estimate's median and bad case at that scale, and `within_budget`.
- CLI: `size --verify`, with the data source rebuilt like `reconcile`'s (synthetic runs replay their seed). It refuses up front, before any work, for paper runs (no fixed period) and configs with AI agents (no model calls). `--json` adds `verification`; without a suggestion it says so.
- Tests: the verification equals an independent backtest at that risk; the text output; no suggestion; refusals for a paper run and an agents config.

### Stage 29A: The desk funnel and lead IDs ✅
- `research/desk.py`: `desk_funnel(store, run_id, hours=None)` rebuilds a run's funnel from its audit trail (votes from the signals table, decisions in processing order, trades from `research.trades`).
  - **scanned:** signal-stage decisions.
  - **lead:** an entry vote while flat. The position state is tracked through fills and exits; shorts count only when allowed.
  - **confirmed:** the ensemble's entry signal.
  - **cleared:** scheduled after the filters, then not rejected at the next open (a limit order counts once it is working).
  - **executed:** the fill.
  - **closed:** the exit, with the trade's PnL and exit type.
- Lead IDs (`L-0001`...) follow time order over the whole run, so a window keeps the same IDs.
- Confirmed leads that went no further are counted by a normalised reason (no numbers or dates): filter, circuit breaker, risk limit, an order already working, size too small, or expired.
- CLI: `trading-lab desk [RUN_ID | --all] [--hours N] [--leads 10] [--alert] [--json]`. The default is the running paper run, else the latest run. `--alert` sends "Desk report" whatever `min_level` is set to.
- `deploy/systemd/trading-lab-desk.service` and `.timer`: daily at 21:00 (`Persistent=true`, read-only paths), described in DEPLOYMENT.md section 5f. The README gets a desk section mapping the seats (head, scouts, risk, execution, report) onto the existing parts.
- Tests:
  - on five backtests (plain, filters with one position, a kill switch, long/short, limit entries): the funnel narrows at every stage, executed equals the fills, closed leads equal the trades with the same PnL and match one each, kills account for the gap, IDs are sequential, and scanned equals the signal-stage decisions;
  - the exact kill categories without dates or numbers;
  - the window keeping IDs;
  - two paper runs through the CLI, `--all --json` and `--alert`;
  - the systemd units.

### Stage 29B: Positioning and sentiment voters ✅
- `data/context.py`: a `ContextFeed` gives values indexed by the time each became known.
  - `CcxtFundingFeed` reads public funding-rate history through a credential-free CCXT client (it refuses one with credentials): paginated, retried, memoised with tail-only refreshes, on the perpetual symbol (`BTC/USDT:USDT`).
  - `FearGreedFeed` makes one public request to alternative.me, refreshed at most hourly when newer data is needed. A value is known a day after its date.
  - The synthetic feeds derive funding (every 8h, from the 24h return) and sentiment (daily, from the 30-day trend) from the synthetic prices in the run's timeframe, using only candles closed by each value's time.
  - `value_asof` takes the latest known value per bar close, unit-safe (a microseconds-against-nanoseconds bug was caught by the first backtest and fixed).
- `MarketDataProvider.context_feed(kind, timeframe)` returns None by default. The CCXT provider gives the real feeds (with `[data] funding_exchange`, where "" means the exchange's futures market) and the synthetic provider gives synthetic ones. The cache, memo and permutation wrappers delegate; the memo shares one feed per kind.
- `strategies/context.py`: `funding` and `sentiment` are contrarian by default (`mode = "follow"` reverses them). Missing, stale or failing data, or no feed, gives HOLD with the reason. One evaluation path serves live and backtest signals, so they are identical.
- `strategy_factory.attach_context_feeds` is called by both engines. A data source without the needed feed is refused with a clear error.
- Tests:
  - `value_asof` by hand and across time units;
  - synthetic feeds deterministic, window-independent and causal (tampering with candles after a value's time does not change it) and plausible;
  - every vote by hand (thresholds, averaging, follow mode, staleness, missing data, errors), with live and backtest signals identical;
  - parameter validation;
  - a backtest using both, with live paper trading plus a resume matching it fill for fill;
  - a source without context refused, and the wrappers passing feeds through;
  - the real feeds against fakes (pagination, the perpetual symbol, memo tails, retries, errors, the credential refusal, the index's lag and refresh);
  - the provider's feed selection and the config.

### Stage 29C: The confirmation gate ✅
- New `[voting]` settings: `confirmers` (strategy names), `min_confirms` (0 = off, the default) and `confirm_mode` (`agree` or `not_against`). They are validated: confirmers must be enabled strategies with a positive weight, `min_confirms` at most their number, no duplicates.
- `TradingSession.confirmation_shortfall` checks a new entry (long or short) against the ensemble's votes. An unconfirmed entry is IGNORED with "blocked by confirmation: n of m needed (mode; confirmed: ...; against: ...)", which the desk funnel counts as "filter: confirmation". Exits and covers are never gated.
- The `desk` preset is the trend preset plus `funding` and `sentiment` (weight 0.5), as `not_against` confirmers that must both not object. `config/default.toml` documents the settings.
- Tests: the config validation and round trip; the gate by hand in both modes and for shorts; a backtest where every executed entry had its confirmations and the blocked ones are recorded; the funnel count; and the `desk` preset with vetoes, matching live paper trading fill for fill and veto for veto.

### Stage 29D: The weekly seat review ✅
- `research/desk.seat_review(store, run_id)` gives one `Seat` per voter with votes, from `research.attribution`, ordered by the PnL of the trades it backed: measured calls, share right (4-bar forward), trades for and against with their PnL, and pivotal trades.
- `seat_verdict` is cautious: "too early" under 30 measured calls; "earning its seat" (right at least 52% of the time and the trades it backed did at least as well as those it opposed); "not earning its seat" (under 48%, or the trades it backed lost more than the opposed ones, with at least 5 backed trades); otherwise "unclear".
- `desk --seats` (text and JSON). The weekly digest adds a "Which seats earned their place" section for every running paper run (`build_digest(seats=False)` leaves it out).
- Tests: the verdicts by hand; the seats against the attribution; the order; the CLI (text, JSON); and the digest section with and without seats.

### Stage 30A: One command to validate a config ✅
- `research/validate.py`: `validate(config, baseline, provider, start, end, ...)` runs:
  - the checkup (with the trial log when given);
  - an A/B test against the baseline, aligned by `align` to the candidate's symbols, timeframe and cash, and skipped when the configs are identical;
  - one in-memory backtest for the outlook, the sizing for a drawdown budget and the exit groups.
  All share one memoised data provider.
- `recommend` is strict: **not ready** if any check fails or the baseline wins with p < 0.05; **strong candidate** only if every check passes and the candidate wins with p < 0.05; otherwise **paper-trade it next to the baseline**, with the reasons.
- `format_validation` gives a compact text to paste, `validation_html` the checkup page with the recommendation, and `to_dict` the JSON.
- CLI: `trading-lab validate CONFIG [--baseline B.toml] [--days 180 | --start/--end] [--windows 6] [--permutations 50] [--max-drawdown 20%] [--allow-agents] [--html FILE] [--json FILE]` plus the market options. The baseline defaults to `config/default.toml`, else the built-in defaults. With recording on, it logs the checkup and the candidate's A/B windows.
- Tests: the alignment; every recommendation path by hand; a full validation on synthetic data (all parts present, text, HTML order, JSON); identical configs skipping the A/B test; the CLI (aligned baseline, files, trial-log rows, errors).

### Stage 30B: Context data on disk and prefetch ✅
- `data/context.CachedContextFeed` stores a feed's values in `<cache_dir>/context/<feed>/<symbol>.csv`; sentiment goes in one `all.csv`.
  - Covered requests are served from the file. Otherwise only the missing tail is fetched, or the whole range when it starts before the file.
  - Within a process, a range already asked for is not asked again (no request every bar for values not yet published).
  - A failing source falls back to the file; with no file the error is raised.
  - Values round-trip exactly (`%.17g` and round-trip parsing), so cached and fresh runs give identical votes; a one-bit difference was caught and fixed.
- `CachedProvider.context_feed` wraps the inner feeds, one per feed name.
- CLI: `trading-lab prefetch [--days 365 | --start/--end] [--no-context]` plus the market options. It fills the cache with candles, funding per symbol and sentiment, reports rows per source, and exits 1 if any failed. DEPLOYMENT.md section 3b describes it.
- Tests: fetching only the missing parts; exact equality from disk in a new process; no repeated requests within a process; the fallback when a source is down (and the error without a file); sentiment stored once; the provider wrapping; and `prefetch` (files written, `--no-context`, the cache switched off).

### Stage 31A: Tournament ✅
- `research/tournament.py`: `tournament(candidates, baseline, provider, start, end, ...)` runs `validate` for each candidate on one memoised data provider. The `Validation` now keeps the candidate's per-bar returns.
  - Each candidate gets its deflated Sharpe against the whole field (the aligned baseline included).
  - Ranking: recommendation, then the share of A/B windows won, then the Sharpe ratio.
  - `winner` is the best entry unless it is "not ready".
- `format_tournament` (a ranked table, the next step, and a caution when the winner's deflated Sharpe is under 50%), `tournament_html` and `to_dict`.
- CLI: `trading-lab tournament [PRESET|CONFIG ...] [--baseline] [--days 180] [--windows 6] [--permutations 20] [--max-drawdown] [--html] [--json]` plus the market options. The default is every preset, built on the baseline config. With recording on, each candidate's checkup is logged.

### Stage 31B: Paper plan ✅
- CLI: `trading-lab paper-plan PRESET|CONFIG [--run-id NAME] [--baseline] [--dir config/runs] [--force]`.
  - It writes the candidate as `config/runs/NAME.toml` (`presets.render`: only the differences, with a header) and refuses to write unless it loads back identically.
  - It forces the baseline's `storage.db_path`, so status, desk and live-compare see both runs.
  - It validates the run id (it names a systemd unit), refuses to overwrite without `--force`, and prints the systemctl and live-compare commands with the running baseline run's id when the database has one.
- Tests: the ranking and texts on fakes (winner, caution, nobody ready); a tournament on synthetic data (order, deflated values, JSON, no candidates); the CLI (a preset plus a file, HTML/JSON, an unknown candidate); `paper-plan` (an exact round trip, the commands with the running run id, refusing to overwrite, the database override, a bad id, an unknown candidate).

## 5. Stage 11–31 status summary

On top of the Stage 9/10 system:
- **Risk:** trailing stops and take-profit (11A), ATR stops with volatility-scaled sizing (12A), entry filters by trend, risk state (12C) and correlation (13B). Every addition is off by default, only ever adds caution, and never overrides the circuit breakers.
- **Agents:** weights that adapt to each agent's out-of-sample record in walk-forward, without look-ahead (12B).
- **Operations:** run summaries (11B), push alerts (11C), a readiness check (13A), and a fix so runs from older versions resume.
- **Evidence:** an HTML report (12D), bootstrap robustness ranges (13C), and alpha/beta against buy & hold (13D).
- **Verification (14):** live runs reconciled with their backtest (14A, which found and fixed a window-dependent agent feature), market-data quality checks (14B), agent answer-quality diagnostics (14C), and run exports (14D).
- **Usability and reach (15):** an offline demo (15A), e-mail alerts (15B) and Claude as a second model provider (15C).
- **Shorts (16):** opt-in simulated short selling: fully collateralised accounting with borrow fees (16A), mirrored entries, exits, sizing and filters (16B), and the trade side in every report (16C).
- **Strategy research (17):** opt-in trend-following strategies (17A), cost sensitivity with a break-even level (17B), and performance by market regime (17C).
- **Decisions and operations (18):** A/B tests of whole configs over independent windows (18A), a watchdog with a systemd timer (18B), and volatility-targeted sizing (18C).
- **Not fooling yourself (19):** a permutation test against shuffled markets (19A), sweep stability scores (19B), and market regimes in the report and dashboard (19C).
- **One verdict and smarter exits (20):** a one-command strategy checkup (20A), a time stop (20B), and regime-dependent strategy weights (20C).
- **Readability and reach (21):** monthly returns, Calmar and the longest drawdown (21A), Telegram alerts (21B), and config presets (21C).
- **Running several configs live (22):** several paper runs on one VM with one watchdog (22A), a comparison of two live runs over the time they ran together (22B), and an overview of all paper runs in the dashboard (22C).
- **Is it luck, and how is it going? (23):** probabilistic and deflated Sharpe ratios (23A), and a weekly digest of every paper run (23B).
- **Where the money goes (24):** trade analysis by exit, symbol, side, holding time and entry time (24A), shown in the reports and the dashboard (24B), and a "Sharpe is real" check in the checkup (24C).
- **Faster feedback and deeper trade analysis (25):** the test suite on every core (25A), and trade excursions (25B), also in the reports and the dashboard (25C).
- **Counting the tries (26):** an opt-in trial log, the `trials` command and a checkup check against it (26A), used by sweeps and shown in the dashboard (26B).
- **What to expect (27):** forward drawdowns, returns and losing streaks from a run's trades (27A), shown in the reports and the dashboard (27B).
- **Acting on it (28):** the risk per trade for a drawdown budget (28A), verified by a backtest at that size (28B).
- **The desk (29):** the funnel from first vote to closed trade with lead IDs, and an evening desk report (29A), the positioning and sentiment voters (29B), the confirmation gate with a `desk` preset (29C), and the weekly seat review (29D).
- **From idea to paper run (30):** one command to validate a config with a strict recommendation (30A), and context data on disk with a prefetch command (30B).
- **Choosing what to paper-trade (31):** a tournament of candidates with deflation across the field (31A), and a paper plan for the winner (31B).

### Later
Order-book data, more LLM providers (e.g. Gemini, as `LLMProvider` subclasses), and more alert channels.
