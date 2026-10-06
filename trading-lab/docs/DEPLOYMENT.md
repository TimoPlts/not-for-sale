# Running trading-lab 24/7 on a Linux VM

This guide runs **live paper trading** (simulated fills on public market
data) and the **read-only dashboard** as two systemd services. Nothing here
ever sends an order to an exchange, and no exchange API keys are needed or
read. The only secret is the Qwen endpoint token.

The paths below assume Ubuntu/Debian, a service user `tradinglab` and a
checkout in `/opt/trading-lab`. Change them consistently if yours differ.

## 1. Install

```bash
sudo apt-get install -y python3 python3-venv git sqlite3
sudo useradd --system --create-home --home-dir /var/lib/tradinglab --shell /usr/sbin/nologin tradinglab
sudo mkdir -p /opt/trading-lab /var/log/trading-lab
sudo chown tradinglab:tradinglab /opt/trading-lab /var/log/trading-lab

sudo -u tradinglab git clone https://github.com/TimoPlts/not-for-sale.git /opt/trading-lab
cd /opt/trading-lab/trading-lab
sudo -u tradinglab python3 -m venv .venv
sudo -u tradinglab .venv/bin/pip install -e ".[dashboard]"     # add ,dev to run the tests
sudo -u tradinglab .venv/bin/python -m pytest -q                 # optional, needs ".[dev,dashboard]"
```

Python 3.11 or newer is required.

## 2. Secrets: the environment file

Qwen settings come **only** from environment variables (`QWEN_API_URL`,
`QWEN_MODEL`, `QWEN_API_KEY`). On the VM they live in one root-owned file
that the paper service loads. Never put them in `config/*.toml` or commit
them. `*.env` files are git-ignored, and only `deploy/trading-lab.env.example`
(placeholders) is in the repository.

```bash
sudo install -d -m 750 -o root -g tradinglab /etc/trading-lab
sudo install -m 640 -o root -g tradinglab deploy/trading-lab.env.example /etc/trading-lab/trading-lab.env
sudoedit /etc/trading-lab/trading-lab.env        # fill in the real URL, model and token
```

Check the connection. This makes one model call, places no trade and writes nothing:

```bash
sudo -u tradinglab bash -c 'set -a; . /etc/trading-lab/trading-lab.env; set +a;
  cd /opt/trading-lab/trading-lab && .venv/bin/trading-lab agent-test qwen && .venv/bin/trading-lab agent-test qwen_trend'
```

## 3. Choose what runs

Edit `/opt/trading-lab/trading-lab/config/default.toml` (or a copy that you
pass with `--config`). To let the Qwen agents vote, give them a positive
weight:

```toml
[strategies.qwen_trend]
weight = 1.0
[strategies.qwen_momentum]
weight = 1.0
[strategies.qwen_risk]
weight = 1.0

[agents]
mode = "record"   # cache every answer; restarts never ask the model twice about the same context
```

A paper run keeps the config it was **started** with. To change the config,
start a new run (a new `--run-id`).

## 4. Try it in the foreground

```bash
cd /opt/trading-lab/trading-lab
sudo -u tradinglab bash -c 'set -a; . /etc/trading-lab/trading-lab.env; set +a;
  .venv/bin/trading-lab paper --run-id vm-paper-1 --once'
```

`--run-id NAME` starts the run if it does not exist and **resumes** it if it
does. This is what the service uses, so every restart continues the same run.

## 5. Install the services (templates, not installed automatically)

Review `deploy/systemd/*.service` (user, paths, run id), then:

```bash
sudo cp deploy/systemd/trading-lab-paper.service deploy/systemd/trading-lab-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trading-lab-paper trading-lab-dashboard
systemctl status trading-lab-paper trading-lab-dashboard
```

* `trading-lab-paper` loads `/etc/trading-lab/trading-lab.env`, runs
  `paper --run-id vm-paper-1`, restarts on failure after 60 s and may only
  write `data/` and `/var/log/trading-lab`.
* `trading-lab-dashboard` has **no** environment file (it needs no secrets),
  listens on `127.0.0.1:8501` only, and has read-only access to the checkout.

### Viewing the dashboard

Do not expose it to the internet. Use an SSH tunnel from your own machine:

```bash
ssh -L 8501:127.0.0.1:8501 you@your-vm
# then open http://localhost:8501
```

`trading-lab dashboard-data` prints the same information in the terminal (`--json` for scripts).

## 5b. Alerts on your phone (optional)

1. Choose a webhook: an [ntfy](https://ntfy.sh) topic with a long random name (install the ntfy app and
   subscribe to it), a Slack incoming webhook, a Discord webhook, or any endpoint that accepts JSON.
2. Add `TRADING_LAB_ALERT_URL=...` to `/etc/trading-lab/trading-lab.env` (the URL is a secret: never put it
   in the config or in git).
3. In the config: `[alerts] enabled = true`, plus `format = "ntfy" | "slack" | "discord" | "json"`.
4. Test: `set -a; . /etc/trading-lab/trading-lab.env; set +a; .venv/bin/trading-lab alert-test`
5. `sudo systemctl restart trading-lab-paper`

You get: kill switch and daily loss limit trips, model calls paused/resumed, repeated failed cycles
(data or database outages) and their recovery, crashes, and a daily summary. Set `min_level = "info"`
to also hear about every entry, exit and run start/stop. A webhook that is down never affects trading.

## 6. Where things are

| what | where |
|------|-------|
| run history (runs, signals, decisions, fills, trades, equity, bars, state) | `/opt/trading-lab/trading-lab/data/trading_lab.db` (SQLite) |
| cached agent answers (record/replay) | `/opt/trading-lab/trading-lab/data/agent_cache.db` (SQLite) |
| cached public candles | `/opt/trading-lab/trading-lab/data/cache/*.csv` |
| service output | `journalctl -u trading-lab-paper -f`, `journalctl -u trading-lab-dashboard -f` |
| paper log file (rotated, 10 MB x 5) | `/var/log/trading-lab/paper.log` |
| secrets (Qwen, optional alert URL) | `/etc/trading-lab/trading-lab.env` (root:tradinglab, 640) |

Useful commands (as `tradinglab`, from the checkout):

```bash
.venv/bin/trading-lab report                     # all runs
.venv/bin/trading-lab report vm-paper-1          # one run
.venv/bin/trading-lab agent-report vm-paper-1    # per-agent votes, correctness, PnL attribution, Qwen usage
.venv/bin/trading-lab dashboard-data vm-paper-1
.venv/bin/trading-lab summary vm-paper-1 --hours 24   # what happened today
```

## 7. Stopping, restarting, resuming

* **Stop safely:** `sudo systemctl stop trading-lab-paper`. systemd sends
  SIGTERM. The trader finishes the cycle it is in (including model calls),
  saves it, marks the run `stopped` and exits. If that takes longer than
  `TimeoutStopSec` (300 s), systemd kills it. That is still safe: each cycle
  is saved in one SQLite transaction, so an interrupted cycle leaves nothing
  half-written, and its bars are processed again on the next start.
* **Restart / reboot:** `sudo systemctl start trading-lab-paper` (or a reboot,
  since the service is enabled). `--run-id` resumes the run: the portfolio is
  rebuilt from the stored fills, and breakers, scheduled and working limit
  orders, last prices and recent stop-outs are restored. Bars that closed
  while it was down are caught up one by one with the backtest's rules.
  Cached agent answers are reused.
* **Resume by hand:** `trading-lab paper --resume <run id>` (or `--run-id <name>`).
* **New run:** change `--run-id` in the unit file, then `daemon-reload` and `restart`.

## 8. Backups and updates

```bash
# consistent online backup, safe while the trader runs
sudo -u tradinglab sqlite3 /opt/trading-lab/trading-lab/data/trading_lab.db ".backup '/var/lib/tradinglab/trading_lab-$(date +%F).db'"
sudo -u tradinglab sqlite3 /opt/trading-lab/trading-lab/data/agent_cache.db ".backup '/var/lib/tradinglab/agent_cache-$(date +%F).db'"

# update the code
sudo systemctl stop trading-lab-paper trading-lab-dashboard
cd /opt/trading-lab && sudo -u tradinglab git pull
cd trading-lab && sudo -u tradinglab .venv/bin/pip install -e ".[dashboard]"
sudo systemctl start trading-lab-paper trading-lab-dashboard
```

Database schema upgrades happen automatically when the paper service opens
the database. The dashboard never migrates; it only reads.

## 9. What happens when something fails

See [FAILURE_RECOVERY.md](FAILURE_RECOVERY.md) for the full list. In short,
model problems, malformed answers and exchange data outages never stop the
trader and never create a trade: agents vote HOLD and the cycle is retried.
Only programming errors or a broken database stop the process. systemd then
restarts it, and the run resumes from the database.

## Security checklist

- [ ] `/etc/trading-lab/trading-lab.env` is mode 640, owned by root:tradinglab, and not in git.
- [ ] No exchange API keys anywhere (trading-lab refuses exchange clients with credentials).
- [ ] The dashboard listens on 127.0.0.1 and is reached through SSH only.
- [ ] Logs and the database contain no Qwen token (it is redacted from every error message).
