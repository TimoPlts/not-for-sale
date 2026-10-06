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

## Execution realism

These settings live in the `[execution]` section of `config/default.toml`. The defaults keep simple market orders with fixed slippage.

| setting | effect |
|---------|--------|
| `slippage_model = "volume"` | adds square-root market impact: `impact_coefficient × volatility × √(order value ÷ average bar value traded)`, measured only on bars *before* the fill. Small orders in liquid markets pay almost nothing extra; big orders in thin markets pay a lot. |
| `max_participation_pct = 0.01` | an order may take at most 1% of the average bar volume. This becomes a sizing limit, and it also caps each limit-order fill. |
| `entry_order_type = "limit"` | entries become limit buys `limit_offset_bps` below the open. They fill only when the price trades **through** the limit (merely touching it is not enough), at the limit price with `maker_fee_rate`. Fills can be partial, and the rest expires after `limit_ttl_bars`. An exit signal, a stop-loss or the kill switch cancels a working order. |

Exits and stop-losses are always market orders. Live paper trading uses the same rules, and working limit orders survive a stop and `--resume`.
