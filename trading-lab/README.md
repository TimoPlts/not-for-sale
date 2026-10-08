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
- **Stage 16B (complete):** `[risk] allow_short = true` lets the bot short (simulated): an ensemble SELL opens a short, and stops, take-profit, trailing stops, limit entries and filters are all mirrored. Off by default.
- **Stage 16C (complete):** shorts in every report:
  - trade side in `report`, `export`, the HTML report, the dashboard and summaries;
  - short exposure on the dashboard;
  - agent attribution that credits SELL votes for short trades.
- **Stage 17A (complete):** two opt-in trend-following strategies, `ma_cross` (moving-average crossover) and `donchian` (channel breakout).
- **Stage 17B (complete):** `trading-lab costs`: the same backtest at 0x to 3x fees and slippage, with the break-even cost level.
- **Stage 17C (complete):** `trading-lab regimes`: a run's performance by market regime (trend up, sideways or down, and calm or volatile).
- **Stage 18A (complete):** `trading-lab ab A.toml B.toml`: is a config change really better? Both configs over independent windows, with a sign test.
- **Stage 18B (complete):** `trading-lab status`: a watchdog that checks the paper trader is alive and keeping up, with a systemd timer that alerts you when it is not.
- **Stage 18C (complete):** opt-in volatility-targeted sizing (`risk.position_volatility_pct`): calm coins get bigger positions, wild ones smaller.
- **Stage 19A (complete):** `trading-lab permutation-test`: could a market with no pattern have produced the result? The same backtest on shuffled-candle markets.
- **Stage 19B (complete):** `sweep` scores every setting by its neighbours (`stable` column, `--rank stability`), so you pick a plateau rather than a lucky peak.
- **Stage 19C (complete):** the market-regime table in the HTML report, the dashboard and `dashboard-data`.
- **Stage 20A (complete):** `trading-lab checkup`: is this strategy any good? Every research check at once, with a pass/warn/fail verdict and next steps.
- **Stage 20B (complete):** an opt-in time stop (`risk.max_holding_bars`): positions exit at the next open after N bars.
- **Stage 20C (complete):** opt-in regime-dependent strategy weights (`[voting.regime_weights]`): trend followers can count more in trends, and mean reversion in sideways markets.
- **Stage 21A (complete):** monthly returns tables (in `report`, the HTML report and the dashboard), the Calmar ratio and the longest drawdown.
- **Stage 21B (complete):** Telegram alerts (`channels = ["telegram"]`, bot token only from the environment).
- **Stage 21C (complete):** `trading-lab init-config PRESET`: ready-made configs (trend, trend-shorts, conservative, mean-reversion) to test with `checkup` and `ab`.
- **Stage 22A (complete):** several paper runs side by side on one VM: a systemd template (`trading-lab-paper@NAME`, one config per run) and `status --all`, which checks every running paper run.
- **Stage 22B (complete):** `trading-lab live-compare RUN_A RUN_B`: which of two paper runs is doing better over the time they ran together? Metrics, better days with a sign test, and the settings that differ.
- **Stage 22C (complete):** with several paper runs, the dashboard and `dashboard-data` open with an overview table: each run's status, equity, return, drawdown, open positions, last bar and watchdog check.
- **Stage 23A (complete):** the probabilistic Sharpe ratio (how likely the true Sharpe is above 0) in every report, and the deflated Sharpe ratio of a sweep's winner (does it beat the luckiest of all the settings tried?).
- **Stage 23B (complete):** `trading-lab digest`: one weekly message about every paper run (its week against the market, trades, watchdog) and the live comparisons, with a systemd timer that sends it.
- **Stage 24A (complete):** `trading-lab trades`: a run's closed trades by exit type (stop, trailing stop, take-profit, time stop, kill switch, signal), symbol, side, holding time and entry weekday or hour.
- **Stage 24B (complete):** the trade breakdown in the HTML report ("Where the money comes from"), the dashboard's Research section and `dashboard-data`.
- **Stage 24C (complete):** a "Sharpe is real" check in `checkup`, based on the probabilistic Sharpe ratio.
- **Stage 25A (complete):** the test suite runs on every core (`pytest -n auto`): about 2 minutes instead of 7 to 10.
- **Stage 25B (complete):** `trades` shows how far each trade went against you and for you while it was open (MAE/MFE), next to the configured stop-loss.
- **Stage 25C (complete):** the excursions in the HTML report and the dashboard as well.
- **Stage 26A (complete):** an opt-in trial log (`[storage] record_trials = true`) of every backtest, sweep, A/B test, checkup and permutation test. `trading-lab trials` counts how many configs you tried on the same data, and `checkup` grades a result against all of them.
- **Stage 26B (complete):** with the trial log on, `sweep` also deflates its winner against every logged trial on the same data, and the dashboard and `dashboard-data` show the trial count and luck bar for the run's period.
- **Stage 27A (complete):** `trading-lab outlook`: the drawdowns, returns and losing streaks to be ready for over the next trades, from a run's own trades.
- **Stage 27B (complete):** the outlook in the HTML report ("What to be ready for"), the dashboard's Research section and `dashboard-data`.
- **Stage 28A (complete):** `trading-lab size --max-drawdown 20%`: the risk per trade at which the bad-case drawdown matches your budget, using the size limits each entry recorded.
- **Stage 28B (complete):** `size --verify` re-runs the stored backtest at the suggested size and compares the real drawdown and return with the estimate.
- **Stage 29A (complete):** `trading-lab desk`: the desk funnel (scanned, leads, confirmed, cleared, executed, closed) with a lead ID for every setup, why confirmed setups died, and an evening desk report at 21:00 through the alert channels.
- **Stage 29B (complete):** two opt-in voters on market context: `funding` (futures positioning, the desk's "whale" seat) and `sentiment` (the Fear & Greed index, the "shill" seat), from public data, without look-ahead, with synthetic versions for offline tests.
- **Stage 29C (complete):** the desk's confirmation gate (`voting.confirmers`, `min_confirms`, `confirm_mode`): a new entry needs N confirmations, or no objection, from named voters; exits are never gated. Plus a `desk` preset.

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
.venv/Scripts/python -m pytest -n auto                  # run all tests on every core (pip install -e ".[dev,dashboard]")
```

To type just `trading-lab`, activate the environment first with `.venv\Scripts\Activate.ps1`.

**How paper trading works:** the trader acts once per *closed* candle. It computes signals at the close and fills any resulting order at the open price of the candle that just started, with fees and slippage. Stops are checked against candle lows. These are the same rules as the backtester, so paper results are directly comparable with backtests. Every signal, decision, fill and equity value is saved to `data/trading_lab.db`, and a stopped run continues exactly where it left off with `--resume`. No orders are ever sent to an exchange.

## Dashboard (read-only)

```bash
pip install -e ".[dashboard]"          # Streamlit, optional
trading-lab dashboard                  # http://127.0.0.1:8501 ; --host/--port to change
```

One page, refreshed automatically (every 60 s by default):

* **Paper runs** (only when there are several): every paper run's status, equity, return, max drawdown, open positions, last bar and watchdog check, running runs first.
* **Portfolio:** equity, daily PnL, drawdown, exposure, realized/unrealized PnL, breaker status (with a banner when the kill switch or daily limit is active).
* **Equity curve** against equal-weight buy & hold, and the **drawdown** chart.
* **Open positions** (entry, current price, size, unrealized PnL, stop) and working simulated orders.
* **Latest decision:** every vote (RSI, MACD, Bollinger, Qwen Trend, Momentum, Risk) with confidence, weight and label, the ensemble result and the actions taken.
* **AI rationales:** one card per agent and symbol.
* **Agent performance** leaderboard: votes, confidence, correctness, trades influenced, pivotal trades, PnL when agreed or disagreed.
* **Research:** strategy versus buy & hold return, max drawdown, Sharpe and profit factor, monthly returns, the market-regime table (see `trading-lab regimes`), trades by exit type and holding time with their excursions (see `trading-lab trades`), the drawdowns and losing streaks to be ready for (see `trading-lab outlook`), plus saved experiments and walk-forward results.
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
* a monthly returns table (year by month, plus the year's total);
* bootstrap robustness ranges and the market-regime table: the strategy against the market in rising, sideways and falling, calm and volatile markets (see `trading-lab regimes`);
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

## Starting points (`init-config`)

```bash
trading-lab init-config                       # list the presets
trading-lab init-config trend                 # writes config/trend.toml
trading-lab --config config/trend.toml checkup --days 180
trading-lab ab config/default.toml config/trend.toml --days 180
```

| preset | what it changes |
|---|---|
| `trend` | Donchian breakouts and an EMA crossover join the vote, plus a 4% trailing stop and a 72-bar time stop |
| `trend-shorts` | the same, trading both directions (simulated, fully collateralised shorts) |
| `desk` | the trend preset plus the desk's whale and shill seats: `funding` and `sentiment` vote, and either one can veto a new entry (the confirmation gate below) |
| `conservative` | half the risk per trade, volatility-targeted positions, ATR stops, a 200-bar trend filter, tighter breakers |
| `mean-reversion` | RSI and Bollinger only (MACD off), a 3% take-profit, a 24-bar time stop, and mean reversion muted in down-trends |

The file lists only what the preset changes. Everything else keeps the defaults documented in `config/default.toml`, so you can see exactly what you are testing. An existing file is never overwritten without `--force`. Presets are starting points to test, not recommendations.

## Is this strategy any good? (`checkup`)

```bash
trading-lab checkup --days 180                          # your config, the last 180 days
trading-lab checkup --start 2025-01-01 --end 2025-07-01 --html reports/checkup.html
```

This is the quickest honest answer. It runs one backtest, then the cost, luck, resampling, Sharpe and regime checks below, and grades each one:

```
  [PASS] Enough trades             324 closed trades
  [FAIL] Edge before costs         -3.39% with free trading
  [FAIL] Survives costs            -19.74% as configured
  [FAIL] Not luck                  p = 0.627 (31 of 50 shuffled markets did as well)
  [FAIL] Robust to resampling      100% chance of a loss when the trades are resampled
  [FAIL] Sharpe is real            0% chance the true Sharpe ratio is above 0 (sample length, skew, fat tails)
  [WARN] Beats buy & hold          -9.90% versus equal-weight buy & hold
  [WARN] Drawdown                  worst drawdown -22.7%
  [WARN] Works in several regimes  profitable in 2 of 6 market regimes
Verdict: not convincing: failed Edge before costs, Survives costs, Not luck, Robust to resampling, Sharpe is real.
```

(This is the default config on the offline demo's random-walk prices. It should fail, and it does.)

Each warning or failure comes with a next step. The thresholds:

* **Enough trades:** at least 30.
* **Survives costs:** break-even at 1.5x the configured costs or more.
* **Not luck:** p < 0.05; up to 0.2 is a warning.
* **Robust to resampling:** a loss in fewer than 10% of bootstrap resamples; up to 30% is a warning.
* **Sharpe is real:** a probabilistic Sharpe ratio of at least 95% (the chance the true Sharpe ratio is above 0, allowing for the sample length and fat tails); 80% or more is a warning.
* **Drawdown:** at most 20%; up to 35% is a warning.
* **Works in several regimes:** profitable in at least half the market regimes.
* **Beats your other trials** (only with the trial log, below, holding earlier trials on overlapping data): a deflated Sharpe ratio of at least 95% against all of them; 50% or more is a warning.

The overall verdict is the worst check. A full pass is still one period, in-sample: confirm it with `walkforward` or `ab`. A 90-day, four-symbol checkup takes about a minute (`--permutations` sets the size of the luck test). `--html` writes a one-page report and `--json` the full results. Nothing is stored.

## How many configs have you tried? (`trials`)

Try enough settings and one of them looks good by luck alone. `sweep` corrects for its own grid (the deflated Sharpe ratio), but not for the sweeps, A/B tests and backtests you ran last week on the same data. Turn on the trial log to keep count:

```toml
[storage]
record_trials = true
```

From then on, `backtest`, `sweep` (each combination), `ab` (each config in each window), `checkup` and `permutation-test` (the real-market run) each add a row to the database. A row holds the config fingerprint, the timeframe, symbols and period, the return, the Sharpe ratio and the number of bars. Walk-forward test windows are out of sample by design and are not logged.

```bash
trading-lab trials                                    # the whole log
trading-lab trials --start 2024-01-01 --end 2024-03-01 --json
```

```
Trials on 1h data overlapping 2024-01-01 -> 2024-03-01: 10 (10 distinct configs; 2 repeated run(s) not counted again)
  by command: checkup 2, sweep 10
  best Sharpe -0.25: sweep rsi.period=28 min_agreeing=2, 2024-01-01 -> 2024-03-01, return -0.14%
  the best of 10 trials would reach a Sharpe of about 2.76 by luck alone (given how much their Sharpe ratios vary); a new result must clear that bar
```

(Synthetic data; an illustration of the layout.) A trial counts if its period overlaps the one asked about on the same timeframe. Any overlap counts, which errs on the strict side. The same config re-run on the same period is one trial. With the log on, `checkup` adds a **Beats your other trials** check: the deflated Sharpe ratio of the new result against every logged trial on overlapping data. The honest fix for a failure there is fresh data you have not tuned on, not another tweak. With the log on, `sweep` adds a second deflated Sharpe line, against all logged trials on overlapping data with its own combinations included. The dashboard's Research section and `dashboard-data` show how many trials overlap the run you are looking at, next to the luck bar. Recording is off by default and never changes any result.

## How much do costs decide? (`costs`)

```bash
trading-lab costs --start 2025-01-01 --end 2025-07-01
trading-lab costs --days 60 --multipliers 0,1,1.5,2 --export costs.json
```

This runs the same backtest several times, with every trading cost scaled by a multiplier (`0` is free trading, `1` is as configured, `2` is twice as expensive). The scaled costs are the taker and maker fees, the slippage and volume impact, and the short borrow fee. Data, signals and cached agent answers are identical across the runs, so the differences come from costs alone.

```
 costs     fee slip bps    return  sharpe trades  fees paid
    0x  0.000%      0.0    +3.10%    1.42     52       0.00
    1x  0.100%      5.0    +0.40%    0.21     52     198.65
    2x  0.200%     10.0    -2.20%   -1.03     52     374.63
Costs as configured take 2.70 percentage points of return off the cost-free result (198.65 in fees; ...)
Verdict: break-even at about 1.15x the configured costs (thin: ...)
```

(The numbers above are only an illustration of the layout.)

* **Break-even below about 1.5x:** the edge is thin. Cheaper execution (limit orders, a lower fee tier), fewer trades or a longer timeframe matter more than strategy tuning.
* **A loss even at 0x:** the strategy has no edge before costs in that period.

Nothing is stored.

## Is the change really better? (`ab`)

```bash
cp config/default.toml config/try.toml        # then edit try.toml, e.g. allow_short = true
trading-lab ab config/default.toml config/try.toml --days 180 --windows 6
```

This splits the period into equal, separate windows and backtests both configs in each one, starting flat with the initial cash every time. It prints:

* exactly which settings differ;
* each window's result for A and for B, and which one did better;
* the compounded return of each;
* a verdict with a sign test: how likely B's number of wins would be if both configs were equally good.

```
Settings that differ (A -> B):
  risk.allow_short: False -> True
window                          A         B  better   (total_return)
2025-01-01 -> 2025-01-31     +2.10%    +3.40%    B
...
Verdict: B leads 4 of 6 windows, but that could easily be chance (p = 0.34); use more windows or a longer period.
```

(The numbers above are only an illustration of the layout.)

Both configs must trade the same symbols and timeframe with the same starting cash. `--symbols` and `--timeframe` apply to both. `--metric sharpe_ratio` (or any other metric) compares something other than return; for `max_drawdown`, lower is better.

Be honest about the arithmetic: with 6 windows, only 6 wins out of 6 reaches p < 0.05. Changes that win 4 of 6 need more evidence before you trust them. Nothing is stored.

## Which live run is doing better? (`live-compare`)

`ab` compares configs on past data. To check the result on new data, run both configs as paper runs side by side (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), section 5d), then:

```bash
trading-lab live-compare default trend             # run ids
trading-lab live-compare default trend --json      # every day, for scripts
```

It compares the two runs only over the time they both ran. Each run is measured from its equity at the start of that period, so a run that started earlier gets no head start. It prints:

* the settings that differ;
* return, max drawdown, Sharpe, trades closed, fees and exposure over the shared period;
* each UTC day's return for both, and which was better (ties, such as both flat, do not count);
* a verdict with a sign test on the better days.

Until 14 days have been compared (`--min-days`) the verdict is "too early to tell". Positions can last across days, so days are not fully independent: treat the p-value as a rough guide, and give both runs weeks, not days. It is read-only and works on any two stored runs.

## Where does it make or lose money? (`regimes`)

```bash
trading-lab regimes                    # the latest run
trading-lab regimes <run id> --trend-bars 100 --json
```

Each bar of a run is labelled by the market regime of an equal-weight index of its symbols:

* **Trend:**
  * **up:** above its 50-bar average, with the average rising;
  * **down:** below a falling average;
  * **sideways:** anything else.

  Only bars up to each point are used. The first 50 bars of a run are warm-up.
* **Volatility:** *volatile* when the 24-bar volatility is above the run's median, *calm* otherwise.

Per regime, and per trend × volatility combination, the table shows:

* how much time the run spent there;
* the strategy's return compounded over those bars, next to the market's;
* the time in the market;
* the trades opened there (count, wins, PnL).

Some typical readings:

* A strategy that only makes money in *up / calm* markets is a long-trend follower in disguise.
* Losing mostly in *sideways / volatile* markets is the classic whipsaw.

Combine it with `allow_short` or the trend filter, then backtest again. The database is only read.

## Where does the money come from? (`trades`)

```bash
trading-lab trades                          # the latest run
trading-lab trades <run id> --by exit,holding,hour --csv trades.csv
```

It groups a run's closed trades and shows, for each group, the trades, wins, PnL, average return, average holding time, best and worst trade, and profit factor. The groups are:

* **exit:** what closed the position: stop-loss, trailing stop, take-profit, time stop, kill switch, signal (the vote turned), or the end of a backtest;
* **symbol** and **side** (long or short);
* **holding time:** 1-2, 3-6, 7-24, 25-72 or more than 72 bars;
* **entry weekday**, and with `--by ...,hour` the entry hour (UTC).

A few observations point at the biggest effects: a result that rests on a handful of outliers, the costliest exit type, symbol or side, and long holds against short ones. For example:

```
By exit
                   trades   won         pnl  avg ret avg bars       best      worst    pf
  signal              100   37%     -911.63   -0.49%     23.0     +86.62     -85.86  0.58
  stop-loss             3    0%     -273.75   -5.24%     16.0     -90.07     -93.06  0.00
  trailing stop        10   30%      -40.97   -0.21%     43.1     +76.32     -42.15  0.76
  time stop             9   89%     +552.46   +3.52%     72.0    +111.90     -20.08 28.51
```

Each group also shows the average **mae** (maximum adverse excursion: the worst move against the entry while the trade was open) and **mfe** (maximum favourable excursion: the best move in its favour), measured on the run's stored candles. A summary puts them in context:

```
Excursions (mae: worst move against the entry while open, mfe: best move for it; 122 trades)
  90% of winning trades went at most 2.41% against the entry; the stop-loss is 5.0%
  28 of 74 losing trades were at least 1% in profit at some point
  e-ratio 1.21 (average mfe / average |mae|; above 1, trades move further for you than against you)
```

These are facts about past trades, not settings to copy. A tighter stop would also have changed which trades were taken, so test a change with `ab` or `walkforward`. Moves inside the exit candle before a stop or target filled are not seen.

(A 90-day synthetic backtest of the `trend` preset: an illustration of the layout, not a result.) Small groups are noise. Turn a hunch from this table into a config change and test it with `ab` or `walkforward` before trusting it. `--csv` writes every trade with its exit type, holding time, mae and mfe. Read-only.

## Could a market without patterns do as well? (`permutation-test`)

```bash
trading-lab permutation-test --start 2025-01-01 --end 2025-04-01 --permutations 100
```

This backtests the config on the real candles, then 100 times on *shuffled* versions of the same period:

* **What is kept:** each candle's shape (its open, high, low and close relative to the previous close, and its volume), the volatility, the overall price move over the period, and the links between coins. The candles are reordered the same way for every symbol.
* **What is destroyed:** the order of the candles, and with it every trend, momentum or mean-reversion pattern.

The result is the share of shuffled markets that did at least as well:

```
Real market: total_return +8.40%
Shuffled markets (100): 5% -6.10% | median -1.20% | 95% +5.30%
At least as good as the real result: 2 of 100 (p = 0.030)
Verdict: unlikely to be luck: only 3.0% of shuffled markets did as well (...)
```

(The numbers above are only an illustration of the layout.)

A strategy with real timing skill beats almost all shuffled markets. One that only rides the market's drift, or gets lucky, does not. It is still a single period, so a small p-value is necessary but not sufficient: confirm with `walkforward` or `ab`.

AI agents are refused by default, because they would be asked about every shuffled market. `--allow-agents` overrides that, at the cost of many model calls. Nothing is stored.

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

## What should you be ready for? (`outlook`)

```bash
trading-lab outlook                       # the latest run, over as many trades as it made
trading-lab outlook <run id> --trades 300 --json
```

It resamples the run's closed trades into 5,000 possible futures (seeded, so the numbers are repeatable). Each trade's return is relative to the equity just before it closed, and the returns compound. It reports:

* the max drawdown to be ready for, as a median and a bad case (1 in 20), next to the run's own;
* the chance of a 10%, 20% or 30% drawdown;
* the range of returns and the chance of a loss;
* the longest losing streak.

```
Run bt-7169d978c6ec: the next 122 trades, resampled 5,000 times from the run's 122 trades
  max drawdown       median 9.5%, bad case (1 in 20) 16.2%   (the run itself: 11.5%)
  chance of a drawdown of at least 10%: 45%, 20%: 1%, 30%: 0%
  return             -15.1% .. median -7.2% .. +1.9%   (chance of a loss 91%)
  losing streak      median 8 trades, bad case 13 in a row
Trades are drawn independently and drawdowns are measured trade by trade, so real streaks and dips can be worse: treat the bad case as a floor to prepare for, not a worst case.
```

(The 90-day synthetic `trend` backtest from the `trades` section: an illustration of the layout, not a result.) Size positions so that the bad-case drawdown and losing streak are something you would sit through. A strategy is usually abandoned at its worst moment, and this shows roughly how bad that moment can get. The real thing can be worse: trades are drawn independently, drawdowns are measured trade by trade, and markets unlike the tested period are not in the sample. Read-only.

## How big should positions be? (`size`)

```bash
trading-lab size --max-drawdown 20%            # the latest run
trading-lab size <run id> --max-drawdown 15% --trades 300 --json
```

It finds the multiple of `risk.risk_per_trade_pct` at which the outlook's bad-case drawdown (1 in 20) equals your budget. The scaling is per trade, from the size limits each entry recorded:

* at a scale s, an entry's size is s times its risk-per-trade size, but no more than its other limits (max position, total exposure, cash, liquidity, volatility target);
* scaling up stops helping where another limit takes over.

```
Run bt-7169d978c6ec: sizing for a bad-case (1 in 20) drawdown of 20% (risk.risk_per_trade_pct is 1.00%)
  size set by: risk_per_trade 100%
  beyond about 1.31x, another limit (e.g. max_position_pct) sets most entries' size, so larger scales change little

 scale risk/trade  dd median   dd bad  return med  P(loss)  streak bad
 0.25x      0.25%       2.4%     4.3%       -1.8%      90%          13
 0.50x      0.50%       4.8%     8.4%       -3.6%      91%          13
 0.75x      0.75%       7.1%    12.3%       -5.4%      91%          13
 1.00x      1.00%       9.4%    16.1%       -7.2%      91%          13
 1.27x      1.27%      11.8%    20.0%       -9.0%      91%          13  <- target
 1.50x      1.50%      12.2%    20.6%       -9.3%      91%          13
 2.00x      2.00%      12.2%    20.6%       -9.3%      91%          13

Suggested: risk_per_trade_pct = 0.0127 (1.27x the current setting). Confirm with a backtest before using it:
  [risk]
  risk_per_trade_pct = 0.0127
warning: the run lost money: a smaller size only loses more slowly
```

(The same synthetic `trend` backtest: an illustration, not advice.) Here the max position limit takes over from about 1.3x, which is why 1.5x and 2x look the same. "Size set by" shows which limit set each entry's size; if it is rarely `risk_per_trade`, scaling it changes little. A smaller size only makes a losing strategy lose more slowly. The suggestion is an estimate: equity, cash and overlapping positions would differ at another size. `--verify` checks it: it re-runs the stored backtest's config over the same period and data at the suggested risk per trade, in memory, and compares the result:

```
Verification backtest at 1.27x (risk_per_trade_pct 0.0127), same period and data:
  max drawdown 14.7% (the run at 1x: 11.7%; estimated at this size: median 11.8%, bad case 20.0%): within the budget on this history
  return -8.43% (1x: -6.67%), trades 122 (1x: 122)
```

One period is one path: a drawdown below the budget here does not prove the budget holds, it only shows the estimate is not off. `--verify` works on stored backtests, not paper runs, and not with AI agents (it would call the model). Read-only.

## The desk (`desk`)

The bot is organised like a small trading desk, for BTC, ETH and other large coins, with simulated fills only:

| seat | who | does |
|---|---|---|
| head of desk | the voting engine and the circuit breakers | routes every setup, halts the floor at the daily loss limit or the kill switch |
| scouts | RSI, MACD, Bollinger, Donchian, MA cross, the Qwen trend and momentum agents | each votes BUY, SELL or HOLD; agents only produce opinions |
| whale and shill | `funding` (futures positioning) and `sentiment` (Fear & Greed), opt-in | add a vote from crowding and mood, not from the chart |
| risk | the Qwen risk agent, the entry filters, the risk manager | clears or blocks every entry and sizes it |
| execution and exits | the paper executor; stops, trailing stops, take-profit, time stop | fills at the next open, manages every open position |
| evening report | `desk`, `summary`, `digest`, Telegram alerts | tells you what happened without opening a chart |

No seat holds keys or can place a real order.

**Confirmations.** The desk rule "nothing executes without the confirmations" is a config setting:

```toml
[voting]
confirmers = ["funding", "sentiment"]   # enabled strategies with a weight
min_confirms = 2
confirm_mode = "not_against"            # "agree": they must vote the same way; "not_against": they must not object
```

A new entry, long or short, then needs `min_confirms` of the confirmers to confirm it, or not to object. Otherwise it is skipped as "blocked by confirmation: 1 of 2 needed (... against: funding)" and counted in the funnel. Exits are never gated: getting out does not wait for anyone. It is off by default (`min_confirms = 0`). The `desk` preset (`trading-lab init-config desk`) sets it up with `funding` and `sentiment` as vetoes. As with every preset, test it with `checkup` and `ab` before relying on it.

`trading-lab desk` shows the funnel that every setup passes through:

1. **scanned:** every symbol-bar evaluated;
2. **leads:** a strategy or agent voted to enter while flat;
3. **confirmed:** the vote passed;
4. **cleared:** it passed the entry filters, the breakers and the risk manager;
5. **executed:** it filled;
6. **closed:** the position was closed.

Each lead gets an ID so its thread can be followed from the first vote to the exit:

```bash
trading-lab desk                         # the running paper run (else the latest run), the whole run
trading-lab desk --all --hours 24        # every running paper run, the last 24 hours
trading-lab desk <run id> --json         # every lead with its votes, stage and outcome
```

```
Desk funnel for bt-7169d978c6ec (2026-10-04 18:00 -> 2026-10-07 18:00 UTC)
  scanned        292  symbol-bars evaluated
  leads           39  a strategy voted to enter while flat
  confirmed        5  the vote passed  (13%)
  cleared          5  passed filters, breakers and risk  (100%)
  executed         5  filled  (100%)
  closed           3  position closed  (60%)
  closed: 3 won, 0 lost, PnL +98.54; 2 still open
Latest confirmed leads (3):
  L-1172 2026-10-06 09:00 BTC/USDT   long  votes: macd, donchian -> closed by signal, PnL +68.36
  L-1179 2026-10-06 16:00 ETH/USDT   long  votes: donchian, ma_cross -> open
  L-1189 2026-10-07 17:00 BTC/USDT   long  votes: macd, donchian -> open
```

(The synthetic `trend` backtest: an illustration of the layout.) Confirmed setups that died are counted by reason, such as "filter: trend filter", "risk: max open positions reached" or "circuit breaker: max drawdown". `desk --all --hours 24 --alert` sends the funnel as the evening report, and the systemd timer in `deploy/systemd/trading-lab-desk.*` does it at 21:00 (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), section 5f). Read-only.

## Positioning and sentiment (`funding`, `sentiment`)

Two opt-in voters that look beyond the chart, for BTC, ETH and other large coins:

* **`funding`** reads the perpetual futures funding rate, the desk's "whale" seat. When it is high, longs pay shorts and the trade is crowded; when it is negative, shorts are crowded. By default it is contrarian: it votes SELL at or above `high` (0.05% per 8 hours, averaged over the last 3 prints) and BUY at or below `low` (-0.01%).
* **`sentiment`** reads the Crypto Fear & Greed index, the "shill" seat. By default it is contrarian too: BUY in fear (25 or below), SELL in greed (75 or above).

`mode = "follow"` reverses either one.

```toml
[strategies.funding]
weight = 1.0            # high = 0.0005, low = -0.0001, average = 3, max_age_hours = 24
[strategies.sentiment]
weight = 1.0            # fear = 25, greed = 75, max_age_hours = 72
```

(A `[strategies]` table replaces the default strategies, so list the others you want too, or start from a preset.)

* **Data:** public only.
  * Funding comes through CCXT from the futures market of your exchange (`binance` uses `binanceusdm`; set `[data] funding_exchange` to choose another), with a client that has no credentials.
  * The index comes from `api.alternative.me`.
  * On a VM with an outbound allow-list, allow those hosts.
* **No look-ahead:** each bar only sees values published by its close. Funding counts from its funding time; the index counts from a day after the day it describes.
* **Fail-safe:** missing, stale (older than `max_age_hours`) or failing data gives HOLD with the reason in the signal, never a trade.
* **Synthetic data:** with `--synthetic`, both use synthetic data derived from the synthetic prices (funding rises after rallies, sentiment follows the 30-day trend), so offline backtests and tests work. Those results say nothing about real markets.

Whether either one improves results on real data is exactly what `checkup`, `walkforward` and `ab` are for. Test them before giving them weight in a paper run. The desk funnel (`trading-lab desk`) shows them among the voters of each lead.

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
| `trades.csv`, `fills.csv` | closed trades with their side (long or short) and simulated fills (same columns as `backtest --export`) |
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

**Telegram:** set `channels = ["telegram"]` (or add it to the list). Create a bot with @BotFather and send it one message. Then put the bot token and your chat id in the environment:

```bash
export TRADING_LAB_TELEGRAM_BOT_TOKEN='123456:...'   # secret: environment only
export TRADING_LAB_TELEGRAM_CHAT_ID=123456789
trading-lab alert-test --channel telegram
```

The token is part of Telegram's API address. It is never stored, logged or shown, and errors only say which variable to check.

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
* `trading-lab status` is a watchdog. A systemd timer in `deploy/systemd/` runs `status --alert` every 15 minutes and alerts you if the trader has crashed, hung or keeps failing. It exits 1 when:
  * more than 2 closed candles have not been processed;
  * 3 or more cycles in a row have failed;
  * the run is not running (`--allow-stopped` allows that).
* `status --all` checks every running paper run at once (the watchdog uses it), so several runs can trade side by side: `trading-lab-paper@NAME` runs `config/runs/NAME.toml` as run NAME. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), section 5d.
* `trading-lab digest` sums up the last 7 days (`--days`) of every paper run: its return against the market, worst drawdown, trades, equity and watchdog check, plus the `live-compare` verdict for each pair of running runs. `--alert` sends it through the alert channels, and a weekly systemd timer does that every Monday (section 5e).

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
.venv\Scripts\python -m pytest -n auto
```

On macOS or Linux, use `.venv/bin/python` instead. `-n auto` runs the tests on every CPU core (`pytest-xdist`, part of the `dev` extra): about 2 minutes on 4 cores instead of 7 to 10. Plain `pytest` still works, one test at a time.

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
* **Trades influenced:** each closed trade is linked to the ensemble's entry signal. An agent *agreed* or *disagreed* with the trade, or abstained, at that bar. For a long, agreeing is a BUY vote and disagreeing a SELL; for a short (with `allow_short`), it is the other way round. It was *pivotal* if the entry would not have happened without its vote.
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
# The "stable" column averages each setting with its grid neighbours; --rank stability orders by it.
# A setting is only trustworthy if the settings around it are good too (a plateau, not a lone peak):
trading-lab sweep --param strategies.rsi.period=7,10,14,21,28 --rank stability
# Honest check for overfitting: choose parameters on 90 days, test them on the next 30 unseen days, repeat:
.venv/Scripts/trading-lab walkforward --param voting.min_agreeing=1,2 --train-days 90 --test-days 30 --start 2025-01-01
# Side-by-side metrics of stored runs:
.venv/Scripts/trading-lab compare <run id> <run id>
```

`--param` takes any dotted config key, such as `risk.stop_loss_pct=0.03,0.05` or `strategies.trend_analyst.weight=0,1`. A sweep's best row is optimistic by construction, so judge it by the walk-forward **out-of-sample** results.

**How much does a Sharpe ratio prove?** Every report (`report`, `compare`, the HTML report, sweep exports) includes the **probabilistic Sharpe ratio** ("Prob. Sharpe > 0"): the probability that the true Sharpe ratio is above 0, allowing for how many bars there were and for skewed, fat-tailed returns. A sweep also prints the **deflated Sharpe ratio** of row 1. Trying many settings makes the best one look good by luck alone, so the deflated Sharpe measures row 1 against what the luckiest of all the settings tried would show by chance:

```
Deflated Sharpe of row 1: 22%, the chance its true Sharpe beats what the luckiest of 8 settings would show by chance (no better than the luckiest of the trials: likely chosen by luck)
```

Aim for 95% or more. Below 50%, the winner is no better than chance would pick. Both follow Bailey and López de Prado; the formulas are in `src/trading_lab/metrics/sharpe.py`.

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

## Trend-following strategies (opt-in)

The default strategies are mostly contrarian: RSI and Bollinger buy weakness, while MACD follows momentum shifts. Two trend followers can join the vote. Each one is enabled by adding its table to the config:

```toml
[strategies.ma_cross]          # moving-average crossover
weight = 1.0
fast = 20
slow = 50
average = "sma"                # or "ema"
signal_on = "cross"            # BUY/SELL on the bar of the cross; "state" = on every bar above/below

[strategies.donchian]          # channel breakout ("turtle" style)
weight = 1.0
entry_period = 20              # BUY above the previous 20-bar high, SELL below the 20-bar low
exit_period = 10               # a weaker opposite signal on a 10-bar break (0 = off)
atr_period = 14
```

* **Confidence:**
  * `ma_cross`: grows with how sharply the averages cross, relative to the usual size of the gap between them.
  * `donchian`: grows with the breakout distance measured in ATRs.
* **No look-ahead:** channels use only the bars *before* the current one.
* **Shorts:** both strategies signal in both directions, so they work naturally with `allow_short = true`.

To see whether they help on your symbols and timeframe, compare them on the same period. For example, `trading-lab sweep --param strategies.donchian.weight=0,1` runs the strategy on and off, and `walkforward` tests it out of sample.

## Regime-dependent weights (opt-in)

```toml
[voting.regime_weights.up]       # this symbol is in an uptrend
donchian = 2.0
rsi = 0.5
[voting.regime_weights.down]
donchian = 2.0
rsi = 0.5
[voting.regime_weights.sideways]
donchian = 0.0                   # breakouts mostly whipsaw in a range
rsi = 1.5
```

For each symbol and bar, the regime comes from the candles up to that bar only:

* **up:** the close is above its `regime_bars` (default 50) simple average, and that average is higher than `regime_slope_bars` (default 10) bars ago;
* **down:** the close is below a falling average;
* **sideways:** anything else.

Each strategy's voting weight is multiplied by its number for the current regime. Strategies that are not listed keep their weight. During the first `regime_bars` bars there is no regime, so the normal weights apply. The ensemble signal records the regime, and its votes record the effective weights, so `agent-report` and the dashboard show what counted. Backtests and live runs use the same labels as `trading-lab regimes` (per symbol rather than for the whole market).

This is an easy way to overfit. Pick the numbers from reasoning, not from a sweep, and check the result with `trading-lab ab` against the same config without the table, then with `permutation-test`.

## Short selling (simulated, off by default)

```toml
[risk]
allow_short = true

[execution]
short_borrow_bps_per_day = 2.0   # borrow cost per day held (0.02%/day), paid when the short is covered
```

By default the bot is long-only: a SELL vote only closes a long. With `allow_short = true`:

* **Entries and exits:** an ensemble SELL with no long open opens a short at the next candle's open, and an ensemble BUY covers it. A long is never turned into a short in one step: the SELL closes the long first, and only a later SELL opens a short.
* **No leverage:** a short is fully collateralised. Its value is set aside from cash, plus the fee, exactly like buying. Losses can still exceed the collateral if the price more than doubles, so keep the stop-loss on.
* **Mirrored risk:** sizing risks the same share of equity per trade, with the stop *above* the entry.
  * Stops trigger on the candle's high, filling at the stop, or at the open if it gapped above.
  * Take-profit triggers on the low.
  * Trailing stops follow the lowest low and only ever move down.
  * ATR stops work the same way.
  * Limit entries rest above the open and fill when a candle trades through them.
* **Filters and breakers:**
  * The trend filter allows shorts only while the close is *below* its average.
  * The correlation and risk-state filters, the circuit breakers and the kill switch apply to both sides. The kill switch also covers shorts when `flatten_on_halt` is set.
* **Costs:** besides fees and slippage, covering pays the borrow fee for the days held. It is included in the cover's fee.

Backtests, paper runs, resume and `reconcile` handle shorts exactly like longs. Every report shows each trade's side: `report`, `export`, the HTML report, the dashboard (open positions and exposure) and `summary`. `agent-report` credits a SELL vote as agreeing with a short trade.

This changes what a SELL vote does, from the AI agents too. Compare `allow_short = true` and `false` on the same period with `experiment` or `walkforward` before relying on it.

## Trailing stops and take-profit

Optional exits in `[risk]` (0 = off, the default):

| setting | effect |
|---------|--------|
| `trailing_stop_pct = 0.04` | after each bar closes, the stop is raised to `highest high since entry × (1 − 4%)`. Stops only move up. A raised stop applies from the **next** bar, so the unknown order of the high and the low inside a bar can never help. |
| `trailing_activation_pct = 0.02` | only start trailing once the high is 2% above the average cost (fees included) |
| `take_profit_pct = 0.10` | exit when a bar's high reaches average cost × 1.10, at that price, or at the open if the bar gapped above it |
| `max_holding_bars = 48` | time stop: once a position has been held 48 bars (the entry bar counts as one), exit at the next open whatever the signals say (`time stop: held 48 bars` in the decision log). Frees capital from trades that go nowhere; shorts are covered the same way |

If the stop and the target are both reached in the same bar, the stop is assumed to come first, which is the conservative choice. Take-profit exits are recorded as `take_profit` decisions, and trailing-stop exits as `stop_loss` with "trailing stop" in the reason. The stop-loss cooldown now follows only stop exits that **lost** money: a trailing stop that locks in a gain does not block re-entry. Raised stops are saved with a live run, so they survive `--resume`, and the dashboard shows the current stop.

## Volatility-scaled sizing (ATR stops)

With `stop_mode = "atr"` in `[risk]`, the initial stop is set by recent volatility instead of a fixed percentage:

```
stop distance = atr_stop_multiple × ATR(atr_period) / entry price,
                clamped to [atr_stop_min_pct, atr_stop_max_pct]
```

The risk-per-trade sizing already sizes each entry so that hitting the stop loses `risk_per_trade_pct` of equity. A volatile coin therefore gets a wider stop and a proportionally **smaller** position, and a calm one a tighter stop and a larger position. The risk per trade is the same either way, and the other limits (max position, exposure, cash, liquidity) still apply.

ATR is the **simple** average true range of the bars *before* the fill, so it never sees the bar it trades on, and live paper trading computes exactly the same value as a backtest. Until there is enough history, the fixed `stop_loss_pct` is used. Each entry decision records `stop_basis` (`atr` or `percent`) and `stop_distance_pct`. Trailing stops and take-profit work on top of either mode.

## Volatility-targeted sizing

```toml
[risk]
position_volatility_pct = 0.10   # each position may add at most 10% of equity in yearly volatility (0 = off)
```

This adds one more size limit, next to risk per trade, maximum position size and exposure:

* **The limit:** `position value x annualised volatility <= equity x position_volatility_pct`. With 0.10, a coin that swings 80% a year gets at most 12.5% of equity, and one that swings 40% gets 25%.
* **Where the volatility comes from:** the bars *before* the entry (the `volume_lookback` window of per-bar returns), exactly the same in backtests and live.
* **What it does:** each position contributes a similar amount of risk, instead of the most volatile coin dominating the portfolio.

Like every limit, it only ever makes positions smaller. The decision details show when it was the binding one (`binding_limit = volatility_target`). Check that it helps on your data with `trading-lab ab`.

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
