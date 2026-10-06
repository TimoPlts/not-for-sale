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
- **Stage 5 (complete):** risk circuit breakers: a max-drawdown kill switch, a daily loss limit, and a cooldown after a stop-loss.
- **Stage 6 (complete):** the AI-agent framework, with record and replay of agent answers for reproducible backtests.
- **Stage 7 (complete):** research tools: a buy & hold benchmark, parameter sweeps, walk-forward evaluation and run comparison.
- **Stage 8 (complete):** execution realism: volume-aware slippage, a liquidity cap, and limit entries with partial fills.
- **Stage 9A (complete):** a real LLM provider for agents. Qwen is the first, through any OpenAI-compatible chat-completions endpoint, with timeouts, retries and HOLD on any failure.
- **Stage 9B (complete):** three specialist Qwen agents (Trend, Momentum, Risk/Regime) that vote in the ensemble.
- **Stage 9C (complete):** agent performance attribution (`trading-lab agent-report`).
- **Stage 9D (complete):** baseline versus AI experiments (`trading-lab experiment`), with agents usable in sweeps and walk-forward.
- **Stage 9E (complete):** model usage accounting: calls, cache hits and misses, failures, retries, latency, and tokens (reported or estimated).
- **Stage 9F (complete):** `trading-lab agent-test`: a connectivity and agent smoke test that never trades.
- **Stage 9G (complete):** the Qwen agents in live paper trading, resumable, asking only about newly closed candles.
- **Stage 10A (complete):** a read-only dashboard data layer (`trading_lab.dashboard.DashboardData`, `trading-lab dashboard-data`).
- **Stage 10B (complete):** a lightweight, read-only web dashboard (Streamlit, `trading-lab dashboard`).
- **Stage 10C (complete):** 24/7 operation on a Linux VM: systemd templates, environment files, named runs (`--run-id`), log files and graceful SIGTERM shutdown. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
- **Stage 10D (complete):** failure recovery for overnight use: a circuit breaker for the model endpoint, back-off during data outages, database-error recovery and health reporting. See [docs/FAILURE_RECOVERY.md](docs/FAILURE_RECOVERY.md).
- **Stage 10E (complete):** a reproducible protocol for "does Qwen improve out-of-sample performance?", with the variant comparison (folds won, sign test) computed by `experiment --walkforward`. See [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).
- **Stage 11A (complete):** optional trailing stops and take-profit exits (long-only, no look-ahead, identical in backtests and live, kept across resume).
- **Stage 11B (complete):** `trading-lab summary`: what a run did over the last N hours, in Markdown.
- **Stage 11C (complete):** push alerts (ntfy, Slack, Discord or JSON webhook) for breaker trips, outages, model pauses, crashes and a daily summary.
- **Stage 12A (complete):** optional ATR-based stops with volatility-scaled position sizing (the same risk per trade on every coin).
- **Stage 12B (complete):** AI agent weights that adapt to each agent's track record in walk-forward (`--adaptive-weights`, no look-ahead) and suggested weights for a run (`trading-lab agent-weights`).
- **Stage 12C (complete):** optional entry filters (trend filter and a `qwen_risk` risk-state veto) that only ever block new entries.
- **Stage 12D (complete):** `trading-lab report RUN_ID --html FILE`, a single self-contained HTML report to share.
- **Stage 13A (complete):** `trading-lab doctor`, a read-only readiness check before going live.
- **Stage 13B (complete):** an optional correlation limit, so the bot does not stack positions in coins that move together.
- **Stage 13C (complete):** `trading-lab robustness`: bootstrap ranges that show how much of a run's result could be luck (also in the HTML report).
- **Stage 13D (complete):** performance relative to buy & hold: excess return, alpha, beta, correlation and information ratio.
- **Stage 14A (complete):** `trading-lab reconcile`: checks that a live paper run did exactly what its backtest does on the same candles.
- **Stage 14B (complete):** `trading-lab data-check`: a market-data quality report (gaps, stale data, zero volume, extreme moves).
- **Stage 14C (complete):** `trading-lab agent-eval`: answer-quality diagnostics for the AI agents (contradictions, one-sided voting, flat confidence, boilerplate rationales, errors).
- **Stage 14D (complete):** `trading-lab export`: any stored run as CSV files plus a JSON summary with checksums.
- **Stage 15A (complete):** `trading-lab demo`: one offline command that builds a sample backtest, a paper run and HTML reports. Start here.
- **Stage 15B (complete):** e-mail alerts over encrypted SMTP, alongside or instead of the webhook (settings only from the environment).
- **Stage 15C (complete):** an Anthropic (Claude) provider as an alternative to Qwen (`[agents] provider = "anthropic"`, key only from the environment).
- **Stage 16A (complete):** simulated short-position accounting: fully collateralised (no leverage), with a borrow fee on cover and a trade side stored in the database (schema v4). Strategies use it from 16B.

### Stage 9/10 summary

The agent framework is now a multi-agent research system powered by the Qwen endpoint, and it stays **paper trading only**:

```
Qwen model (provider-agnostic client; settings and token from environment variables only)
├── qwen_trend     ─┐
├── qwen_momentum  ─┼─ structured votes (direction, confidence, rationale, label)
└── qwen_risk      ─┘        │
RSI · MACD · Bollinger ──────┤
                             ▼
      VotingEngine → circuit breakers → RiskManager → PaperExecutor (simulated fills)
```

* **Agents only vote.** They cannot place, size or cancel orders, change the portfolio, or override breakers. Every failure, timeout or malformed answer is a HOLD.
* **Reproducible:** answers are recorded once (`record`) and replayed offline (`replay`). Market-only agents' answers are shared by every backtest, sweep, walk-forward and experiment variant over the same bars.
* **Measured:**
  * `agent-report` gives each agent's votes, directional correctness, calibration, trades influenced, pivotal trades and PnL when it agreed or disagreed;
  * `experiment` compares the baseline with one, two or three agents, and with the agents alone, out-of-sample;
  * usage accounting tracks calls, cache hits, failures, latency and tokens.
* **Operable 24/7:**
  * `agent-test` checks the connection without trading;
  * live paper trading with agents is resumable, asks only about new candles and never re-asks;
  * the read-only dashboard shows everything in one page;
  * systemd templates, environment files, graceful shutdown and documented failure recovery cover unattended operation.
* **Safety checks still enforced:** no exchange private endpoints, no exchange or credential fields, no hard-coded tokens, and Qwen secrets only from the environment. Tests check all of these.

## Quick start

**First time? Start with the offline demo.** It needs no internet, no keys and no model:

```bash
cd trading-lab
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,dashboard]"     # Windows: .venv\Scripts\python -m pip ...
.venv/bin/trading-lab demo                                # Windows: .venv\Scripts\trading-lab demo
```

This builds `./demo` with:

* a 60-day backtest of the default strategies;
* a 72-hour paper run, replayed with a simulated clock, which `reconcile` confirms matches its backtest;
* `backtest-report.html` and `paper-report.html` to open in a browser.

It then prints the commands to explore further: the run list, the dashboard, reconcile, export, and the same steps on real public market data. The prices are a seeded random walk, so the results say nothing about real markets. They show what the tool records and reports.


On Linux or macOS the command is `.venv/bin/trading-lab` (or just `trading-lab` after `source .venv/bin/activate`). The examples below use the Windows path. The AI and dashboard commands:

```bash
trading-lab agent-test qwen                    # check the Qwen connection (no trading)
trading-lab backtest --start 2025-01-01 --end 2025-07-01        # with agents enabled in the config
trading-lab agent-report                       # what each agent contributed to the last run
trading-lab experiment --walkforward --train-days 90 --test-days 30 --start 2024-07-01 --end 2025-07-01
trading-lab --agent-mode replay experiment ... # the same, fully offline from recorded answers
trading-lab paper --run-id my-paper-run        # start or resume a named live paper run
trading-lab dashboard                          # read-only web dashboard (pip install -e ".[dashboard]")
trading-lab dashboard-data --json              # the same data for scripts
```

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
.venv/Scripts/python -m pytest                          # run all tests (pip install -e ".[dev,dashboard]")
```

To type just `trading-lab`, activate the environment first with `.venv\Scripts\Activate.ps1`.

**How paper trading works:** the trader acts once per *closed* candle. It computes signals at the close and fills any resulting order at the open price of the candle that just started, with fees and slippage. Stops are checked against candle lows. These are the same rules as the backtester, so paper results are directly comparable with backtests. Every signal, decision, fill and equity value is saved to `data/trading_lab.db`, and a stopped run continues exactly where it left off with `--resume`. No orders are ever sent to an exchange.

## Dashboard (read-only)

```bash
pip install -e ".[dashboard]"          # Streamlit, optional
trading-lab dashboard                  # http://127.0.0.1:8501 ; --host/--port to change
```

One page, refreshed automatically (every 60 s by default):

* **Portfolio:** equity, daily PnL, drawdown, exposure, realized/unrealized PnL, breaker status (with a banner when the kill switch or daily limit is active).
* **Equity curve** against equal-weight buy & hold, and the **drawdown** chart.
* **Open positions** (entry, current price, size, unrealized PnL, stop) and working simulated orders.
* **Latest decision:** every vote (RSI, MACD, Bollinger, Qwen Trend, Momentum, Risk) with confidence, weight and label, the ensemble result and the actions taken.
* **AI rationales:** one card per agent and symbol.
* **Agent performance** leaderboard: votes, confidence, correctness, trades influenced, pivotal trades, PnL when agreed or disagreed.
* **Research:** strategy versus buy & hold return, max drawdown, Sharpe and profit factor, plus saved experiments and walk-forward results.
* **Qwen usage:** calls, cache hits, failures, retries, latency and tokens.
* **Recent trades and signals.**

The sidebar only chooses what to *view*: run, outcome horizon and refresh interval. There are no buttons or forms, and the database is opened read-only. The dashboard listens on 127.0.0.1 by default. To see it from another machine, use an SSH tunnel (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)) rather than exposing it.

### Dashboard data layer

`trading_lab.dashboard.DashboardData` reads the SQLite history and returns plain dicts and DataFrames:

* the portfolio: equity, cash, realized and unrealized PnL, drawdown, daily PnL, exposure, open positions, working orders and breaker state;
* the equity curve with drawdown and the buy & hold benchmark;
* recent signals, the latest ensemble decision with every vote, and the latest agent rationales;
* per-agent performance (attribution), model usage, recent trades, and research results (metrics, benchmark, saved experiments).

It opens the database in SQLite's **read-only** mode, so it cannot write anything, and it has no order code at all. Tests check both.

```bash
trading-lab dashboard-data            # summary of the running paper run (or the latest run)
trading-lab dashboard-data <run id> --json
```

## HTML report

```bash
trading-lab report <run id> --html reports/run.html     # or omit the run id for the latest run
```

One self-contained file, with no external scripts, styles or fonts, so it opens offline and can be e-mailed to your teacher:

* key numbers: return against buy & hold, max drawdown, Sharpe, profit factor, trades, win rate and exposure;
* the equity curve against buy & hold, with hover values, and the drawdown;
* a daily table view of the same numbers;
* every voter's performance (AI agents marked), the latest AI rationales, model usage, breaker trips, closed trades and decision counts.

It follows your system's light/dark setting. The report is built from the read-only data layer, so it never changes the database. All text from the database, including model rationales, is HTML-escaped.

## Compared with buy & hold

Every backtest (and `report`, the dashboard, the HTML report and `experiment`) also gives the strategy's performance **relative to equal-weight buy & hold** over the same bars:

```
Relative to buy & hold: excess return -5.44%, alpha -44.02%/yr, beta 0.34, correlation 0.65, information ratio -3.93
```

* **beta:** how much of the market's moves the strategy carries. 0.34 means about a third, which is typical when it is often in cash.
* **alpha:** annualised return beyond what that beta explains.
* **information ratio:** excess return per unit of tracking error.

A strategy can beat buy & hold in a falling market simply by holding cash. Alpha and beta separate "less exposed" from "better at picking", and they are computed from per-bar returns, with definitions in `src/trading_lab/metrics/relative.py`.

## Could it be luck? (`robustness`)

```bash
trading-lab robustness <run id>          # 5000 samples, seed 7 (reproducible)
```

```
Robustness (5000 bootstrap samples, seed 7; ranges are 5th .. median .. 95th percentile)
  Actual total return: -0.97%   actual Sharpe: -1.38
  Trade bootstrap (32 trades): total return -5.42% .. -0.78% .. +4.18%; probability of a loss 60%
  Block bootstrap (696 bars, blocks of 26): total return -4.53% .. -1.18% .. +2.62%; Sharpe -6.68 .. -1.67 .. +3.77
  ! the trade-bootstrap range includes both gains and losses
```

* The **trade bootstrap** redraws the run's closed trades with replacement.
* The **block bootstrap** redraws blocks of about √n bars of the equity curve's returns, so calm and volatile stretches stay together.

A range that spans both gains and losses, or fewer than 30 trades, means the run alone shows nothing either way. The HTML report includes the same table.

## Run summary

```bash
trading-lab summary                    # the latest run, last 24 hours
trading-lab summary vm-paper-1 --hours 168
```

A short Markdown report covering:

* equity and its change over the window, return since start, current and worst drawdown, and the market's move;
* the closed trades (wins, realized PnL, best and worst);
* open positions and the count of each decision type;
* circuit-breaker trips;
* each agent's BUY/SELL/HOLD votes with its latest rationale;
* model usage, plus a health warning if cycles are failing.

It only reads the database (opened read-only) and works on backtests and live runs alike.

## Did the live run do what the backtest does? (`reconcile`)

```bash
trading-lab reconcile vm-paper-1
trading-lab reconcile pp-1a2b3c4d5e6f --synthetic 4   # a run on offline synthetic data
```

A paper run follows exactly the backtest's rules. `reconcile` backs that up for one run:

* **Market data:** the candles the run stored are compared with what the exchange returns now. A revised candle explains any difference that follows from it.
* **Trades and decisions:** a backtest over the run's own period, with the run's stored config, is compared fill by fill and decision by decision with what the run recorded. Agents are replayed from the answer cache, so the model is never called. An answer missing from the cache is counted and replayed as HOLD.

Fills at the open of the candle after the run's last processed bar are left out, because a backtest of those bars cannot have them yet. The database is only read. The exit code is 0 when everything matches and 1 otherwise, so it can run in a script or a timer.

## Is the market data sound? (`data-check`)

```bash
trading-lab data-check                                  # configured symbols, last 30 days, up to now
trading-lab data-check --start 2024-01-01 --end 2024-07-01
trading-lab data-check --run vm-paper-1                 # the candles a run stored
trading-lab data-check --strict                         # exit 1 on warnings too (for scripts)
```

A strategy can only be as good as its candles. For each symbol, the report lists:

* **errors** (exit code 1): the exchange failed or returned nothing, or, when checking up to now, the latest closed candles are missing (stale data). The newest candle may lag by one.
* **warnings:** missing candles (gaps, or none at the start or end of the period), zero-volume candles, candles with no price range, extreme moves, and opens far from the previous close.

An extreme move is a candle that moved more than 10 robust standard deviations of the symbol's own returns, and at least 5%. Both limits can be changed with `--jump-sigmas` and `--jump-floor`. The threshold therefore adapts to each symbol and timeframe. An open far from the previous close only counts between consecutive candles, because a gap in the data explains it. With the CSV cache on, the candles come through the cache, exactly as a backtest gets them. Nothing else is written.

## Exporting a run (`export`)

```bash
trading-lab export vm-paper-1 exports/vm-paper-1
trading-lab export <run id> exports/run --holds      # also every HOLD decision
```

This writes any stored backtest or paper run into a new or empty directory, for a spreadsheet, a notebook or an archive:

| File | Contents |
|---|---|
| `equity_curve.csv` | per bar: cash, positions value, equity, realized and unrealized PnL, fees, open positions |
| `trades.csv`, `fills.csv` | closed trades and simulated fills (same columns as `backtest --export`) |
| `decisions.csv` | the ensemble's decisions with reasons (HOLDs only with `--holds`) |
| `signals.csv` | every strategy and agent vote, with each agent's rationale, label, cache status and error in their own columns |
| `bars.csv` | the candles the run traded on |
| `summary.json` | the run, its config, its metrics, decision counts, and the row count and SHA-256 of every file |

The database is only read. A non-empty directory is refused unless you pass `--force`. That replaces only the export files and leaves anything else in the directory alone. No secret can appear in an export, because credentials only ever come from environment variables and are never stored.

## Alerts

```toml
[alerts]
enabled = true
channels = ["webhook"]   # webhook and/or email
format = "ntfy"          # ntfy | slack | discord | json
min_level = "warning"    # "info" also reports every entry and exit
daily_summary = true
```

```bash
export TRADING_LAB_ALERT_URL="https://ntfy.sh/<a long random topic>"   # secret: environment only
trading-lab alert-test
```

Live paper runs then push:
* **critical:** kill switch trips and trader crashes;
* **warning:** daily-limit trips, model calls paused, repeated failed cycles, and database rollbacks;
* **every day:** a summary (`trading-lab summary`).

Recoveries are reported too, and repeats of the same alert are suppressed for an hour. Alert settings come from the current config, even when resuming an older run. A webhook that fails never affects trading, and its URL is never logged.

**E-mail instead of, or as well as, a webhook:** set `channels = ["email"]` (or `["webhook", "email"]`). The SMTP settings come only from the environment:

```bash
export TRADING_LAB_SMTP_HOST=smtp.gmail.com          # STARTTLS on 587 by default
export TRADING_LAB_SMTP_USER=you@gmail.com
export TRADING_LAB_SMTP_PASSWORD='an app password'   # secret: environment only
export TRADING_LAB_ALERT_EMAIL_TO=you@gmail.com      # comma-separated for several
trading-lab alert-test --channel email
```

The connection is always encrypted: STARTTLS by default, or `TRADING_LAB_SMTP_SECURITY=ssl` for port 465. Without encryption (`none`), only a local relay without a login is accepted, so a password is never sent in clear text. The password is never logged or shown, and `doctor` only reports whether the settings are complete. With several channels, an alert counts as delivered when any channel delivers it.

## Readiness check

```bash
trading-lab doctor             # Python, packages, config, voters, env vars (set/missing only), database, cache, disk
trading-lab doctor --online    # also one public candle (freshness) and, if an agent is on, one model call
```

Each line is `[ OK ]`, `[WARN]`, `[FAIL]` or `[SKIP]`, followed by a verdict. The exit code is 1 when anything fails, so a script or service can refuse to start. The doctor only reads: the database is opened read-only, nothing is traded, and secret values are never printed.

## Running on a Linux VM

[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) covers the whole setup:

* the venv and installation;
* the Qwen environment file (`/etc/trading-lab/trading-lab.env`, never committed; `deploy/trading-lab.env.example` is the template);
* systemd templates for the paper trader and the dashboard (`deploy/systemd/`, not installed automatically);
* log and SQLite locations, safe shutdown, resuming, backups and updates.

The pieces that make unattended operation work:

* `trading-lab paper --run-id NAME` starts the named run, or resumes it if it already exists. Restarts therefore always continue the same run.
* SIGTERM (`systemctl stop`) finishes the current cycle, saves it and marks the run `stopped`.
* `--log-file PATH` (global option) writes a rotating log (10 MB × 5) of all paper activity and warnings. `--log-level` controls its detail.

### When things fail

Failures prefer **HOLD / no new trade** over guessing:

* Model timeouts, errors and malformed answers make that agent vote HOLD.
* After 5 failures in a row the model is not called for 5 minutes (`failure_threshold`, `failure_cooldown_seconds`), so a dead endpoint never stalls a cycle.
* Exchange or network outages leave the trader untouched and are retried with back-off of up to 15 minutes. Missed candles are then caught up in order.
* A cycle that cannot be saved is rolled back, and the trader reloads its state from the database before retrying.

Only a broken database or a bug stops the process; systemd then restarts it and the run resumes. The full table is in [docs/FAILURE_RECOVERY.md](docs/FAILURE_RECOVERY.md).

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

## AI agents

Agents are decision makers, rule-based today and an LLM later, that vote in the ensemble like any other strategy. They only produce **signals**: the risk manager, the circuit breakers and the paper executor always sit between an agent and any simulated trade.

Enable the built-in example agent, an offline trend-following "analyst" that explains every answer, in `config/default.toml`:

```toml
[strategies.trend_analyst]
weight = 1.0
lookback = 30            # candles shown to the agent
decision_interval = 1    # ask every N candles (limits cost for real LLMs)
```

Every answer is saved in `data/agent_cache.db`, and `[agents] mode` controls how that cache is used:

| mode | behaviour |
|------|-----------|
| `record` (default) | reuse cached answers and ask the agent only for new situations |
| `replay` | never call the agent; use cached answers only. The backtest is exactly reproducible even for a non-deterministic LLM |
| `live` | always ask, cache nothing |

Answers are keyed by agent, version, parameters, symbol, candle and a hash of the exact context the agent saw. Changing the agent, its prompt or the data therefore never reuses a stale answer. Rationales are stored with each signal in the run history.

You can also pass any `complete(system_prompt, user_prompt) -> text` function to `LLMAgentStrategy` from code. It builds the prompt from the market context and strictly validates the JSON answer. Invalid answers and errors become HOLD signals and never crash a run.

### Qwen (LLM provider)

Model-backed agents use the provider named in `[agents] provider` (today: `qwen`). The endpoint must speak the OpenAI-compatible `/chat/completions` API. Its address, model and token come **only from environment variables**. They are never read from the config, stored or printed:

```bash
export QWEN_API_URL="https://<host>/v1"      # or the full .../v1/chat/completions URL
export QWEN_MODEL="<model name>"
export QWEN_API_KEY="<token>"                # never commit this; *.env files are git-ignored
```

On Windows PowerShell, use `$env:QWEN_API_URL = "..."` and so on.

Check the connection before anything else. Each command makes **one** model call and never trades, never writes to the database or the answer cache, and never needs exchange keys:

```bash
trading-lab agent-test qwen                           # env vars, one tiny prompt, structured answer, model, latency
trading-lab agent-test qwen_trend                     # one real agent decision on the latest closed candle
trading-lab agent-test qwen_risk --symbol ETH/USDT    # (the risk agent sees a flat simulated portfolio)
trading-lab agent-test qwen_momentum --synthetic 1    # offline market data
```

Then enable the general-purpose `llm_analyst` agent in your config:

```toml
[strategies.llm_analyst]
weight = 1.0
lookback = 30            # candles shown to the model
decision_interval = 4    # ask every 4 candles
```

The rest of the `[agents]` settings (`request_timeout_seconds`, `max_retries`, `retry_backoff_seconds`, `temperature`, `max_output_tokens`) are documented in `config/default.toml`. The record/replay/live modes work exactly as above:

* **record** asks Qwen only when no answer is cached for the exact context.
* **replay** never contacts Qwen. It needs only `QWEN_MODEL`, which is part of the cache key, so a replay is fully offline.
* **live** asks every time.

The provider, model, temperature, output limit and prompt are all part of the cache key, so switching the model never reuses another model's answers. Missing variables stop a run before it starts, with a message naming them.

Failures never stop a backtest or a paper run. Timeouts, network errors, HTTP 429 and 5xx are retried with exponential backoff. If the call still fails, or the answer is malformed, the agent votes HOLD for that bar and the error is saved with the signal. Failed answers are not cached, so `record` mode asks again next time. Reasoning models that "think aloud" (`<think>...</think>`) are supported, because the thinking is stripped before the JSON is parsed.

### Claude (Anthropic) instead of Qwen

Set `provider = "anthropic"` in `[agents]`. The agents and their prompts stay the same; only the model behind them changes. The settings again come **only from the environment**:

```bash
export ANTHROPIC_API_KEY="<key>"             # never commit this
export ANTHROPIC_MODEL="<model name>"        # a current Claude model name from Anthropic's docs
# export ANTHROPIC_API_URL="https://..."     # optional; default https://api.anthropic.com
trading-lab agent-test anthropic             # one tiny prompt, as for Qwen
trading-lab agent-test qwen_trend            # one real Trend Agent decision, answered by Claude
```

It calls Anthropic's Messages API (`/v1/messages`, header `x-api-key`). It shares the timeouts, retries, circuit breaker, usage accounting and secret redaction with Qwen. HTTP 529 (overloaded) is retried as well.

The agents keep their names (`qwen_trend`, `qwen_momentum`, `qwen_risk`), so configs, reports and stored runs stay compatible. The provider and model are part of every cache key, so answers from one provider are never replayed for the other. `agent-report`, `agent-eval` and `experiment` work the same, which lets you compare the two models on identical periods.

### Qwen agents: Trend, Momentum and Risk/Regime

Three specialist agents share the configured model but have their own role, prompt and data. They ship switched off (`weight = 0.0` in `config/default.toml`). Give one a positive weight to let it vote:

| strategy | role | sees | extra answer field |
|----------|------|------|--------------------|
| `qwen_trend` | direction and trend strength | EMA 20/50/200, 5/20/50-bar returns, RSI, MACD, volume trend, candles | `regime`: bullish_trend, bearish_trend, sideways, uncertain |
| `qwen_momentum` | does momentum support acting now? | RSI and MACD changes, return acceleration, volume change, last-candle structure | `momentum_state`: strengthening, weakening, neutral |
| `qwen_risk` | is it too risky for new exposure? | ATR, volatility and its baseline, range expansion, drawdown from highs, liquidity, **plus the simulated portfolio**: exposure, open positions, recent stop-outs, breaker status | `risk_state`: low, moderate, high, extreme |

```toml
[strategies.qwen_trend]
weight = 1.0
lookback = 40            # candles shown to the model
decision_interval = 4    # ask every 4 candles; HOLD in between
```

Every answer must be JSON with `direction` (BUY/SELL/HOLD), `confidence` (0–1), `rationale` and the agent's own label. Anything else becomes HOLD. The label and rationale are saved with each signal. Agents only vote:

* A SELL from an agent is an opinion. The portfolio stays long-only.
* The risk agent cannot override circuit breakers. When new entries are blocked, it is told so, but the breakers decide.
* A weight of 0 switches an agent off completely. It is never called and casts no vote, so it cannot count towards `min_agreeing` either. This now applies to every strategy.

**Market-only versus portfolio context.** `qwen_trend` and `qwen_momentum` see only market data by default. Their answers depend only on the candles, so one recorded answer is reused by every backtest, sweep and experiment over the same bars. `qwen_risk` sees the simulated portfolio by default (`portfolio_context = true`), because judging exposure is its job. Its answers depend on the trading path, so different experiments ask it different questions. Set `portfolio_context = true` on the other two to include position status too, at the cost of fewer cache hits.

Backtests only ask agents about bars inside the backtest period, never about the warm-up history before it.

**Live paper trading with agents.** `trading-lab paper` uses the same agents, with the same rules as a backtest:

* An agent is asked only once a candle has closed, and only on its decision bars. Each cycle evaluates the newly closed candles only, never the history window again.
* The risk agent sees the live simulated portfolio, including breakers and recent stop-outs.
* Any model failure is a HOLD for that bar.
* On `--resume`, the portfolio, breakers, working limit orders and recent stop-outs are restored from the database. Bars already processed are never re-asked. In `record` mode, a bar re-processed after a crash gets its answers from the cache, so the model is not called twice for the same context and no paper order is duplicated.

### Does an agent add value? (`agent-report`)

```bash
.venv/bin/trading-lab agent-report                 # most recent run
.venv/bin/trading-lab agent-report <run id> --horizon 6 --all
```

```
Agent: qwen_trend
  Votes: 1832   BUY: 524   SELL: 391   HOLD: 917
  Avg confidence (BUY/SELL): 0.67
  Directional correctness (4-bar horizon): 54.8% of 903 measurable votes
  Avg outcome after BUY: +0.21%   after SELL: -0.08%
  Trades influenced: 61 of 140 (agreed 52, disagreed 9, pivotal 17)
  PnL when agreed: +12.40% (+1,240.00 USDT)   when disagreed: -3.80% (-380.00 USDT)
  Calibration:  confidence   votes  correct  mean signed return
```

(The numbers above are only an illustration of the layout.)

The definitions are exact and deterministic. They are spelled out in `src/trading_lab/research/attribution.py`:

* **Votes** count decision bars only. Warm-up bars and bars skipped by `decision_interval` are not votes. Failed or invalid answers are counted as `errors`.
* **Directional correctness:** did the price move the voted way over the next N bars (`--horizon`, default 4)? Raw prices are used, without fees.
* **Avg outcome after BUY/SELL:** the mean N-bar forward return after each kind of vote.
* **Trades influenced:** each closed trade is linked to the ensemble's entry signal. An agent *agreed* (voted BUY), *disagreed* (SELL) or abstained at that bar. It was *pivotal* if the entry would not have happened without its vote.
* **PnL when agreed/disagreed:** the realised PnL of those trades, fees included, as a share of the initial cash.
* **Calibration:** correctness and mean signed return per confidence bucket. A well-calibrated agent is right more often when it is more confident.

Every run now also stores the candles it traded on (schema v3 `bars` table), which the outcome statistics need. Older runs show `n/a` for them.

### Are the agent's answers sound? (`agent-eval`)

```bash
trading-lab agent-eval                 # most recent run
trading-lab agent-eval <run id> --json
```

`agent-report` asks whether an agent's votes were right. `agent-eval` asks whether its answers make sense at all, whatever the market did next. Per agent, it reports:

* **availability:** decisions without a usable answer, grouped by error type (timeouts, invalid JSON, answers missing from the cache);
* **consistency:** votes that contradict the agent's own label (BUY with `regime = bearish_trend`, SELL with `momentum_state = strengthening`, BUY with `risk_state = high`), and BUY/SELL votes on a label for which its prompt asks for HOLD (`sideways`, `neutral`, `moderate`);
* **spread:** one-sided voting (90% or more of the BUY/SELL votes on one side), almost always HOLD, confidence that barely varies, and BUY/SELL votes with confidence 0 (which carry no weight);
* **explanations:** empty or very short rationales, and one rationale repeated for most answers;
* **model calls:** count, failures and mean latency.

The spread checks wait for 20 answers (`--min-answers`), so a short run is not judged on a handful of votes. Each agent declares what its prompt asks for (`contradicting_votes`, `hold_labels` in `agents/specialists.py`). The database is only read.

### Model usage and cost

The Qwen endpoint is free today, but every call is accounted for, so paid models can be budgeted later. `backtest`, `sweep`, `walkforward`, `experiment` and `agent-report` end with a block like this:

```
Qwen usage (qwen2.5-7b-instruct):
  calls: 482   cache hits: 1204   cache misses: 482   failures: 3   invalid answers: 1   retries: 5
  avg latency: 1.80s   total latency: 14m 28s
  input tokens: 612,140 (estimated)   output tokens: 31,200 (estimated)   (input chars ..., output chars ...)
```

With several agents, the block also has a per-agent table. Token counts are the endpoint's own `usage` numbers when it reports them. Otherwise they are estimated as characters ÷ 3, which errs on the high side (real text is closer to 3.5–4 characters per token) and is marked `(estimated)`. Each model call's details (latency, attempts, characters, tokens, error) are saved in the signal's metadata under `llm`. Stored runs therefore keep their usage, and `agent-report` rebuilds it from the database.

## Research tools

Every backtest now reports an equal-weight **buy & hold benchmark** over the same period, paying the same fees and slippage.

```powershell
# Backtest every combination of values (data is downloaded once):
.venv/Scripts/trading-lab sweep --param voting.min_agreeing=1,2 --param strategies.rsi.period=7,14,21 --start 2025-01-01 --end 2025-07-01
# Honest check for overfitting: choose parameters on 90 days, test them on the next 30 unseen days, repeat:
.venv/Scripts/trading-lab walkforward --param voting.min_agreeing=1,2 --train-days 90 --test-days 30 --start 2025-01-01
# Side-by-side metrics of stored runs:
.venv/Scripts/trading-lab compare <run id> <run id>
```

`--param` takes any dotted config key, such as `risk.stop_loss_pct=0.03,0.05` or `strategies.trend_analyst.weight=0,1`. A sweep's best row is optimistic by construction, so judge it by the walk-forward **out-of-sample** results.

### Baseline versus AI experiments

`trading-lab experiment` runs named *variants* over the same period, with identical fees, slippage, liquidity, breakers and voting thresholds. Only the voters differ:

| variant | voters |
|---------|--------|
| `baseline` | RSI + MACD + Bollinger |
| `trend` / `momentum` / `risk` | baseline + one Qwen agent |
| `trend_momentum` | baseline + Qwen Trend + Momentum |
| `all_agents` | baseline + all 3 Qwen agents |
| `ai_only` | the 3 Qwen agents only (same risk manager, breakers and executor) |

```bash
# One backtest per variant (quick look; proves nothing on its own):
trading-lab experiment --variants baseline,trend,all_agents,ai_only --start 2025-01-01 --end 2025-07-01
# The real comparison: walk-forward, out-of-sample, per variant (optionally tuning a grid in-sample):
trading-lab experiment --walkforward --train-days 90 --test-days 30 --start 2024-07-01 --end 2025-07-01 \
    --param voting.min_agreeing=1,2 --save --export results/experiment.json
# Re-run exactly, fully offline, from the recorded answers:
trading-lab --agent-mode replay experiment --walkforward --train-days 90 --test-days 30 --start 2024-07-01 --end 2025-07-01
```

Agents also work in plain sweeps and walk-forwards, because `weight = 0` switches a voter off completely:

```bash
trading-lab sweep --param strategies.qwen_trend.weight=0,1 --start 2025-01-01 --end 2025-07-01
trading-lab walkforward --param strategies.qwen_trend.weight=0,1 --param strategies.qwen_momentum.weight=0,1 \
    --train-days 90 --test-days 30 --start 2024-07-01
```

All runs in a sweep, walk-forward or experiment share one model provider and the answer cache. In `record` mode a market-only agent is asked about each bar **once**, and every variant, combination and overlapping training window reuses that answer. The comparison therefore measures the ensemble, not the model's randomness, and later replays are free. `qwen_risk` sees the portfolio, so it is asked again wherever the trades differ. Missing Qwen environment variables stop the command before the first backtest. The global `--agent-mode record|replay|live` option overrides `[agents] mode` for any command.

## Trailing stops and take-profit

Optional exits in `[risk]` (0 = off, the default):

| setting | effect |
|---------|--------|
| `trailing_stop_pct = 0.04` | after each bar closes, the stop is raised to `highest high since entry × (1 − 4%)`. Stops only move up. A raised stop applies from the **next** bar, so the unknown order of the high and the low inside a bar can never help. |
| `trailing_activation_pct = 0.02` | only start trailing once the high is 2% above the average cost (fees included) |
| `take_profit_pct = 0.10` | exit when a bar's high reaches average cost × 1.10, at that price, or at the open if the bar gapped above it |

If the stop and the target are both reached in the same bar, the stop is assumed to come first, which is the conservative choice. Take-profit exits are recorded as `take_profit` decisions, and trailing-stop exits as `stop_loss` with "trailing stop" in the reason. The stop-loss cooldown now follows only stop exits that **lost** money: a trailing stop that locks in a gain does not block re-entry. Raised stops are saved with a live run, so they survive `--resume`, and the dashboard shows the current stop.

## Volatility-scaled sizing (ATR stops)

With `stop_mode = "atr"` in `[risk]`, the initial stop is set by recent volatility instead of a fixed percentage:

```
stop distance = atr_stop_multiple × ATR(atr_period) / entry price,
                clamped to [atr_stop_min_pct, atr_stop_max_pct]
```

The risk-per-trade sizing already sizes each entry so that hitting the stop loses `risk_per_trade_pct` of equity. A volatile coin therefore gets a wider stop and a proportionally **smaller** position, and a calm one a tighter stop and a larger position. The risk per trade is the same either way, and the other limits (max position, exposure, cash, liquidity) still apply.

ATR is the **simple** average true range of the bars *before* the fill, so it never sees the bar it trades on, and live paper trading computes exactly the same value as a backtest. Until there is enough history, the fixed `stop_loss_pct` is used. Each entry decision records `stop_basis` (`atr` or `percent`) and `stop_distance_pct`. Trailing stops and take-profit work on top of either mode.

## Entry filters (trend and risk regime)

Two optional filters in `[risk]`. They can only **block new entries**. They never force an exit, never change a position and never override the circuit breakers, which keep applying as before. A blocked BUY is recorded as an `ignored` decision with the filter's reason.

| setting | effect |
|---------|--------|
| `trend_filter_period = 200` | no new entry while the close is below its 200-bar **simple** moving average. A simple average is used so live and backtest see the same value. With too little history to compute it, entries are blocked. |
| `block_entries_on_risk_states = ["extreme"]` | no new entry while the latest `risk_state` reported for that symbol (by `qwen_risk`) is in the list, for up to `risk_state_max_age_bars` (8) bars after the answer. The last reported state is saved with live runs, so it survives `--resume`. |
| `max_correlated_positions = 1` | no new entry while that many open, working or already-scheduled positions moved with it: their per-bar log-return correlation over the last `correlation_lookback` (48) bars is at least `correlation_threshold` (0.8). BTC, ETH, SOL and DOGE often move together, so this keeps one market move from hitting several positions at once. An unknown correlation (too little data) counts as correlated. |

## Agent weights from their track record

Agents that are often wrong should count for less. `--adaptive-weights` turns that into a rule, re-applied in every walk-forward fold:

```
multiplier = 1 + 10 × (directional correctness − 50%)          (55% → ×1.5, 45% → ×0.5, ≤40% → off)
new weight = current weight × multiplier, capped at --weight-max (default 2.0)
```

* An agent with fewer than `--weight-min-votes` (30) measurable votes keeps its weight: no evidence, no change.
* RSI, MACD and Bollinger are never re-weighted.
* **No look-ahead:** each test window's weights come only from the training window before it, and only from votes whose outcome (`--weight-horizon` bars later) is known inside that training window. A test proves that changing the data after a training window cannot change its weights.

```bash
trading-lab walkforward --param strategies.qwen_trend.weight=1 --adaptive-weights --train-days 90 --test-days 30
trading-lab experiment --walkforward --adaptive-weights --variants baseline,all_agents ...   # weights per fold in --export
trading-lab agent-weights <run id>    # suggested weights from one run's record, as a config snippet
```

The weights `agent-weights` suggests are in-sample for that run. Validate them with walk-forward before using them live.

## Execution realism

These settings live in the `[execution]` section of `config/default.toml`. The defaults keep simple market orders with fixed slippage.

| setting | effect |
|---------|--------|
| `slippage_model = "volume"` | adds square-root market impact: `impact_coefficient × volatility × √(order value ÷ average bar value traded)`, measured only on bars *before* the fill. Small orders in liquid markets pay almost nothing extra; big orders in thin markets pay a lot. |
| `max_participation_pct = 0.01` | an order may take at most 1% of the average bar volume. This becomes a sizing limit, and it also caps each limit-order fill. |
| `entry_order_type = "limit"` | entries become limit buys `limit_offset_bps` below the open. They fill only when the price trades **through** the limit (merely touching it is not enough), at the limit price with `maker_fee_rate`. Fills can be partial, and the rest expires after `limit_ttl_bars`. An exit signal, a stop-loss or the kill switch cancels a working order. |

Exits and stop-losses are always market orders. Live paper trading uses the same rules, and working limit orders survive a stop and `--resume`.
