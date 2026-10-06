# Experiment protocol: does Qwen improve out-of-sample performance?

This protocol answers one question with as little self-deception as
possible: **do the Qwen agents improve the ensemble's out-of-sample
performance compared with the deterministic baseline?** Every step is
reproducible from the config, the cached market data and the recorded agent
answers.

**Never claim an AI strategy is profitable from a single backtest.** A single
backtest is one draw from a noisy process, and every parameter choice made
while looking at it inflates it. Only walk-forward, out-of-sample results
count, and even those are evidence, not proof.

## 1. Hypotheses

* **H0:** adding the Qwen agents does not improve out-of-sample risk-adjusted
  performance (Sharpe) relative to the baseline.
* **H1:** a variant with Qwen agents beats the baseline out-of-sample, on
  most test windows, without a materially worse drawdown.

Fix the hypotheses, the variants, the period and the decision rule
(section 6) **before** looking at any result.

## 2. Experiments

| id | variant (`--variants`) | voters |
|----|------------------------|--------|
| A | `baseline` | RSI + MACD + Bollinger |
| B | `trend` | baseline + Qwen Trend |
| C | `trend_momentum` | baseline + Qwen Trend + Qwen Momentum |
| D | `all_agents` | baseline + all 3 Qwen agents |
| (E, optional) | `ai_only` | the 3 Qwen agents only |

Variants differ **only** in strategy weights. `trading-lab experiment`
enforces this: it derives every variant from one config, so these are
identical by construction:

* historical period, symbols and timeframe;
* fees, slippage model and parameters;
* liquidity settings (`max_participation_pct`, limit or market entries);
* risk limits and circuit breakers (kill switch, daily loss limit, cooldown);
* voting thresholds and `min_agreeing`, plus any grid tuned in-sample;
* agent prompts, model, temperature and decision interval (agents switched on keep their configured weight, or 1.0).

## 3. Setup

1. Create a dedicated config, e.g. `config/experiment.toml` (a copy of
   `default.toml`), and do not edit it during the experiment. Note its
   fingerprint (shown by `trading-lab report <run id>` for saved runs).
2. Set `[agents] mode = "record"` and `temperature = 0.0`. Note `QWEN_MODEL`:
   the model name is part of every cached answer's key.
3. Check connectivity: `trading-lab agent-test qwen` and
   `trading-lab agent-test qwen_trend`.
4. Choose the period. Use at least 12 months and include different
   regimes (up, down, sideways). Prefer a period **after the model's training
   cutoff** (see section 7, "look-ahead through the model").
5. Keep `data/cache/` (public candles) and `data/agent_cache.db` (answers):
   together with the config they make every number reproducible.

## 4. Procedure

Walk-forward only, for the final comparison. Train 90 days, test the next 30
days, step 30 days. Use the same small grid for every variant and choose by
Sharpe in-sample:

```bash
CFG=config/experiment.toml
# 1. record: asks Qwen once per (agent, symbol, decision bar); every variant reuses the answers
trading-lab --config $CFG experiment --walkforward --variants baseline,trend,trend_momentum,all_agents \
    --start 2024-07-01 --end 2025-07-01 --train-days 90 --test-days 30 \
    --param voting.min_agreeing=1,2 --metric sharpe_ratio --save --export results/wf-record.json

# 2. replay: fully offline; must print exactly the same numbers (reproducibility check)
trading-lab --config $CFG --agent-mode replay experiment --walkforward \
    --variants baseline,trend,trend_momentum,all_agents --start 2024-07-01 --end 2025-07-01 \
    --train-days 90 --test-days 30 --param voting.min_agreeing=1,2 --metric sharpe_ratio \
    --export results/wf-replay.json

# 3. agent contribution: one stored backtest per variant over the same period, then attribution
trading-lab --config $CFG --agent-mode replay experiment --variants baseline,trend,trend_momentum,all_agents \
    --start 2024-07-01 --end 2025-07-01 --save
trading-lab --config $CFG agent-report <run id of each AI variant> --horizon 4
```

In replay mode, an answer missing from the cache is a HOLD. If step 2 does
not reproduce step 1, check the usage block for `cache misses`: the config,
the model or the data changed.

`qwen_risk` sees the simulated portfolio, so its questions depend on each
variant's trades. Step 1 records them all. Steps 2 and 3 replay them.

## 5. What to report

For every variant (the `experiment --walkforward` output and the exported
JSON contain all of it):

| measure | where |
|---------|-------|
| total out-of-sample return (compounded over the test windows) | `OOS return` |
| buy & hold return over the same windows | `buy&hold` |
| max drawdown (worst test window) | `worst DD` |
| Sharpe (mean over test windows) | `sharpe` |
| profit factor (mean over test windows) | `PF` |
| trade count (all test windows) | `trades` |
| exposure time (mean share of bars in the market) | `exposure` |
| folds won against the baseline, and the sign test | `beats baseline`, `sign p` |
| agent contribution: votes, correctness, trades influenced, pivotal trades, PnL when agreed or disagreed, calibration | `agent-report` |
| model usage: calls, failures, latency, tokens | usage block |

Report the in-sample numbers next to the out-of-sample ones. A large gap
means overfitting.

## 6. Decision rule (fix it in advance)

A variant counts as **improving on the baseline** only if all of these hold:

1. It beats the baseline's out-of-sample Sharpe in a clear majority of test
   windows, with the one-sided sign test `p ≤ 0.05`. With 12 windows that
   means at least 10 wins. With only 6 windows, even 6 of 6 gives p = 0.016,
   but 5 of 6 does not (p = 0.11).
2. Its compounded out-of-sample return is higher than the baseline's.
3. Its worst-window drawdown is not more than 1.25 × the baseline's.
4. It made enough trades to judge, about 30 or more out-of-sample. Otherwise
   the result is **inconclusive**, not positive.
5. The same conclusion holds on a second, non-overlapping period (or a
   different symbol set). Otherwise it is **not robust**.

Anything else: H0 stands. Comparing several variants against one baseline
also raises the chance of a lucky winner. With 3 AI variants, require
`p ≤ 0.05 / 3` for a strong claim (Bonferroni).

## 7. Pitfalls this protocol guards against

* **Single-backtest claims:** only walk-forward out-of-sample results are used.
* **Tuning on the test set:** parameters are chosen in-sample per window, and the variants, grid, period and rule are fixed in advance.
* **Look-ahead through the model:** an LLM may have seen the historical
  prices or news of a period during training, and could "know" what happened
  next. The engine itself has no look-ahead (agents only see candles up to
  the decision bar), but the model's memory is outside its control. Final
  results should use periods after the model's training cutoff, and the live
  paper run is the true forward test.
* **Model randomness:** answers are recorded once and shared. Variants
  differ because of the ensemble, not because the model answered
  differently, and replay reproduces everything.
* **Hidden cost differences:** fees, slippage, liquidity and breakers are
  identical by construction. Turn on `slippage_model = "volume"` for every
  variant or for none.
* **Lucky streaks:** require enough trades and windows, and a second period.
* **Survivorship and cherry-picking:** report every variant run, including the losing ones (`--save` keeps them in `research_results`).

## 8. After the experiment

If a variant passes, run it as a live paper run (`trading-lab paper --run-id
...`) next to the baseline for weeks, with the same config. Compare them with
`trading-lab compare` and `agent-report`. The paper run is the only test the
model cannot have seen in training.
