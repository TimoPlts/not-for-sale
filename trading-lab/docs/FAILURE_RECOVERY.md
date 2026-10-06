# Failure recovery (overnight operation)

The rule: when something goes wrong, **prefer HOLD / no new trade over
guessing**. Failures of the model or of market data never create, change or
cancel a simulated trade on their own, and they never stop the trader. Only
things that would make the simulation untrustworthy stop the process.
systemd then restarts it, and the run resumes from the database.

## Model (Qwen) failures: recoverable, the trader keeps running

| failure | what happens |
|---------|--------------|
| timeout (`request_timeout_seconds`) | retried with exponential backoff (`max_retries`, `retry_backoff_seconds`); if still failing, that agent votes **HOLD** for the bar and the error is saved with the signal |
| network error, HTTP 429 / 5xx | same as a timeout |
| HTTP 401/403/404/400 (bad token, URL or model) | not retried (retrying cannot help); HOLD, with a hint naming the variable to check |
| malformed answer (not JSON, missing or invalid field or label, cut off) | HOLD; nothing is guessed or repaired; never cached, so record mode asks again next time |
| **repeated failures** | after `failure_threshold` (5) failed calls in a row, calls are **paused** for `failure_cooldown_seconds` (300 s): agents vote HOLD at once instead of waiting on timeouts, so a dead endpoint cannot stall a cycle. After the pause one call probes the endpoint; a success closes the breaker, a failure pauses again. Usage reports count these as `skipped` |
| endpoint unavailable for hours | the deterministic strategies keep voting; the agents keep voting HOLD; nothing else changes. HOLD votes dilute the ensemble score, so fewer entries happen while the endpoint is down |

In `replay` mode the model is never called, so none of this applies. A
missing answer in replay is a HOLD with `not in cache (replay mode)`.

## Market data failures: recoverable

| failure | what happens |
|---------|--------------|
| exchange or public-API outage, network outage, DNS failure | the CCXT client retries each request; if the cycle still cannot load candles, **nothing is processed**, the trader state does not change, and the cycle is retried with exponential backoff (`--poll` × 2ⁿ, at most 15 minutes) |
| exchange gap (missing candles) | a warning; the next available candles are processed in order |
| the open of a new candle is not available yet | orders scheduled for it wait and fill when that bar closes, at the same price a backtest would use |
| downtime of any length | on the next successful cycle every bar that closed meanwhile is processed one by one, with the backtest's rules (signals at the close, fills at the next open, stops on the low) |

While data is failing, each cycle's `health` (time of the last check,
consecutive errors, last error) is saved in the run state. The dashboard
shows a warning, and the CLI prints `cycle not processed, will retry`.

## Database failures

| failure | what happens | recoverable? |
|---------|--------------|--------------|
| database locked by another process (e.g. a long dashboard read or a backup) | SQLite waits up to 30 s for the lock | yes |
| still locked, or a write fails with an operational error | each cycle is saved in **one transaction**, so nothing of the cycle is stored. The trader **reloads its state from the database** (portfolio from fills, breakers, orders, stop-outs) and retries the same bars next cycle. Cached agent answers are reused, so no question is asked twice in record mode | yes |
| the reload itself fails, the database is corrupt, or the disk is full and reads fail | the process **stops** with an error; systemd restarts it after 60 s; fix the disk or database and it resumes | no (needs a person) |
| agent answer cache locked or broken | the agent works without the cache for that call (`cache_error` in the signal metadata); in replay mode that is a HOLD | yes |

## Process failures

| event | what happens |
|-------|--------------|
| `systemctl stop`, SIGTERM | the current cycle finishes (including model calls) and is saved; the run is marked `stopped` |
| Ctrl+C | the same, from a terminal |
| killed (SIGKILL, power loss, OOM) in the middle of a cycle | the cycle's transaction is rolled back; on restart the run resumes from the last saved cycle and re-processes the bars after it |
| unexpected exception (a bug) | the process stops with the traceback in the journal and the log file; systemd restarts it; the run resumes. Repeated crashes show up as a restart loop in `systemctl status` |
| reboot | the enabled service starts and resumes the run (`--run-id`) |

Nothing here ever "catches up" by trading on stale prices: bars are always
processed in time order, with the prices of those bars.

## What stops the process on purpose

* invalid configuration, or missing Qwen variables when an agent with weight > 0 is enabled (before the run starts);
* the database is newer than the code (downgrade), corrupt, or cannot be reloaded;
* a stored run whose fills and trades disagree (`database is inconsistent`);
* programming errors.

Resuming a run with a different config is refused: a run keeps the config it
was started with.
