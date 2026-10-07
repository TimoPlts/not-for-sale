"""Is the paper trader alive and keeping up? A watchdog check for cron or a systemd timer.

``check_status`` reads a paper run from the database (read-only) and judges it
against the clock:

* **behind**: the number of closed candles that have not been processed yet.
  The trader handles each candle shortly after it closes, so 0 or 1 is
  normal. More than ``max_behind`` means the trader stalled. That covers a
  crashed process (the run still says "running"), a hung one, or one that
  keeps failing to fetch data.
* **failing cycles**: the trader records its consecutive failed cycles.
  ``max_errors`` or more in a row is a problem even before candles pile up.
* **status**: a run that is not "running" (e.g. "stopped" after
  ``systemctl stop``) is reported. It is a problem only with
  ``expect_running``, which is the watchdog's default.

It never changes the run. ``trading-lab status`` exits 1 when there is a
problem and can send one alert through the configured channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_lab.data.base import timeframe_delta


@dataclass
class RunStatus:
    run_id: str
    status: str
    timeframe: str
    now: datetime
    last_processed: datetime | None = None  # open time of the last processed candle
    behind: int = 0  # closed candles not processed yet
    last_cycle_at: datetime | None = None
    consecutive_errors: int = 0
    last_error: str | None = None
    equity: float | None = None
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        def iso(value: datetime | None) -> str | None:
            return None if value is None else value.isoformat()

        return {"run_id": self.run_id, "status": self.status, "timeframe": self.timeframe, "now": iso(self.now),
                "last_processed": iso(self.last_processed), "behind": self.behind,
                "last_cycle_at": iso(self.last_cycle_at), "consecutive_errors": self.consecutive_errors,
                "last_error": self.last_error, "equity": self.equity, "ok": self.ok,
                "problems": list(self.problems), "notes": list(self.notes)}


def latest_paper_run(store: Any) -> str | None:
    """The running paper run if there is one, else the most recent paper run."""
    runs = [r for r in store.list_runs(200) if r["kind"] == "paper"]
    running = [r for r in runs if r["status"] == "running"]
    chosen = (running or runs or [None])[0]
    return None if chosen is None else str(chosen["run_id"])


def check_status(store: Any, run_id: str, *, now: datetime | None = None, max_behind: int = 2,
                 max_errors: int = 3, expect_running: bool = True) -> RunStatus:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    if run["kind"] != "paper":
        raise ValueError(f"{run_id} is a {run['kind']} run; status checks paper runs")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    step = timeframe_delta(run["timeframe"])
    out = RunStatus(run_id, run["status"], run["timeframe"], now)
    state = store.load_state(run_id) or {}
    health = state.get("health") or {}
    if state.get("last_processed"):
        out.last_processed = pd.Timestamp(state["last_processed"]).to_pydatetime()
    if health.get("last_cycle_at"):
        out.last_cycle_at = datetime.fromisoformat(health["last_cycle_at"])
    out.consecutive_errors = int(health.get("consecutive_errors") or 0)
    out.last_error = health.get("last_error")
    curve = store.load_equity_curve(run_id)
    if not curve.empty:
        out.equity = float(curve["equity"].iloc[-1])

    latest_closed = pd.Timestamp(now).floor(step) - step  # open time of the newest closed candle
    if out.last_processed is None:
        out.behind = 0
        out.notes.append("no candle processed yet")
    else:
        out.behind = max(0, int((latest_closed - pd.Timestamp(out.last_processed)) / step))

    running = run["status"] == "running"
    if not running:
        message = f"the run is {run['status']!r}, not running"
        (out.problems if expect_running else out.notes).append(message)
    if running and out.behind > max_behind:
        out.problems.append(f"stalled: {out.behind} closed candle(s) not processed "
                            f"(last {out.last_processed:%Y-%m-%d %H:%M} UTC); is the service up?")
    if running and out.last_processed is None and out.last_cycle_at is None:
        started = pd.Timestamp(run["created_at"])
        if pd.Timestamp(now) - started > (max_behind + 1) * step:
            out.problems.append("no cycle recorded since the run was created")
    if out.consecutive_errors >= max_errors:
        out.problems.append(f"{out.consecutive_errors} cycles failed in a row: {out.last_error}")
    elif out.consecutive_errors:
        out.notes.append(f"{out.consecutive_errors} failed cycle(s) in a row (retrying): {out.last_error}")
    return out


def format_status(s: RunStatus) -> str:
    lines = [f"Paper run {s.run_id}: {s.status}, {s.timeframe} candles (checked {s.now:%Y-%m-%d %H:%M} UTC)"]
    if s.last_processed is not None:
        lines.append(f"  last processed candle: {s.last_processed:%Y-%m-%d %H:%M} UTC; "
                     f"{s.behind} closed candle(s) waiting")
    if s.last_cycle_at is not None:
        lines.append(f"  last cycle: {s.last_cycle_at:%Y-%m-%d %H:%M} UTC, "
                     f"{s.consecutive_errors} failed in a row")
    if s.equity is not None:
        lines.append(f"  equity: {s.equity:,.2f}")
    lines += [f"  note: {n}" for n in s.notes]
    lines += [f"  PROBLEM: {p}" for p in s.problems]
    lines.append("OK" if s.ok else "NOT OK")
    return "\n".join(lines)
