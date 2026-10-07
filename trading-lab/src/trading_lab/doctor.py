"""Readiness checks before running unattended (``trading-lab doctor``).

Every check only *reads*: the database is opened read-only (or not at all if
it does not exist yet), nothing is traded, and secret values are never
printed (only whether a variable is set). With ``online=True`` it also
fetches one public candle and, if an AI agent is switched on, makes one
model call via the ``agent-test`` smoke test.
"""

from __future__ import annotations

import importlib.metadata
import os
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from trading_lab.config import AppConfig, load_config
from trading_lab.core.errors import ConfigError, TradingLabError
from trading_lab.core.symbols import timeframe_to_seconds

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
_MARK = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", SKIP: "[SKIP]"}
MIN_FREE_BYTES_WARN = 1_000_000_000
MIN_FREE_BYTES_FAIL = 100_000_000


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: str
    detail: str

    def line(self) -> str:
        return f"{_MARK[self.status]} {self.name}: {self.detail}"


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nearest_existing(path: Path) -> Path:
    path = path.resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _writable_dir(path: Path) -> bool:
    target = _nearest_existing(path)
    return os.access(target if target.is_dir() else target.parent, os.W_OK)


def run_checks(
    config_path: str | None,
    *,
    db_path: str | None = None,
    online: bool = False,
    env: Mapping[str, str] | None = None,
    market_factory: Callable[[AppConfig], object] | None = None,
    now: datetime | None = None,
) -> list[Check]:
    env = os.environ if env is None else env
    checks: list[Check] = []
    add = lambda name, status, detail: checks.append(Check(name, status, detail))  # noqa: E731

    v = sys.version_info
    add("python", OK if v >= (3, 11) else FAIL, f"{v.major}.{v.minor}.{v.micro}" + ("" if v >= (3, 11) else
                                                                                  " (3.11 or newer required)"))
    for package in ("numpy", "pandas", "ccxt"):
        version = _version(package)
        add(f"package {package}", OK if version else FAIL, version or "not installed (pip install -e .)")
    streamlit = _version("streamlit")
    add("package streamlit", OK if streamlit else SKIP,
        streamlit or 'not installed: only needed for the dashboard (pip install -e ".[dashboard]")')

    try:
        cfg = load_config(config_path) if config_path and Path(config_path).exists() else AppConfig()
    except TradingLabError as exc:
        add("config", FAIL, str(exc))
        return checks
    source = config_path if config_path and Path(config_path).exists() else "built-in defaults"
    if db_path:
        cfg = cfg.with_overrides({"storage": {"db_path": db_path}})
    add("config", OK, f"{source} (fingerprint {cfg.fingerprint()[:12]})")
    voters = ", ".join(f"{s.name}={s.weight:g}" for s in cfg.enabled_strategies)
    add("voters", OK, f"{voters}; {cfg.market.timeframe} candles on {', '.join(cfg.market.symbols)}")
    add("safety", OK, f"paper trading only; public market data from {cfg.market.exchange}; no exchange keys")

    from trading_lab.strategy_factory import needs_llm

    llm = needs_llm(cfg)
    if not llm:
        add("model", SKIP, "no AI agent has a weight > 0, so no model is called")
    else:
        from trading_lab.llm import PROVIDERS

        provider_cls: Any = PROVIDERS[cfg.agents.provider]
        prefix = getattr(provider_cls, "env_prefix", cfg.agents.provider.upper())
        need_credentials = cfg.agents.mode != "replay"
        required = provider_cls.required_env(need_credentials=need_credentials) \
            if hasattr(provider_cls, "required_env") else ["MODEL"]
        missing = [f"{prefix}_{s}" for s in required if not env.get(f"{prefix}_{s}", "").strip()]
        state = ", ".join(f"{prefix}_{s} {'missing' if f'{prefix}_{s}' in missing else 'set'}" for s in required)
        add("model", FAIL if missing else OK, f"{cfg.agents.provider}, mode {cfg.agents.mode}: {state}")

    from trading_lab.alerts import URL_ENV, EmailNotifier, TelegramNotifier, build_notifier

    if cfg.alerts.enabled:
        details = []
        for channel in cfg.alerts.channels:
            try:
                notifier = build_notifier(cfg, env, channels=[channel])
            except ConfigError as exc:
                add("alerts", FAIL, f"{channel}: {exc}")
                break
            if isinstance(notifier, TelegramNotifier):
                details.append("telegram, bot token and chat id set")
            elif isinstance(notifier, EmailNotifier):
                details.append(f"email to {len(notifier.recipients)} recipient(s) via {notifier.host}:"
                               f"{notifier.port} ({notifier.security})")
            else:
                details.append(f"{cfg.alerts.format}, {URL_ENV} set")
        else:
            add("alerts", OK, "; ".join(details))
    else:
        add("alerts", SKIP, "disabled ([alerts] enabled = false)")

    db = Path(cfg.storage.db_path)
    if cfg.storage.db_path == ":memory:":
        add("database", WARN, "in-memory: nothing is kept after the process ends")
    elif not db.exists():
        status = OK if _writable_dir(db.parent) else FAIL
        add("database", status, f"{db} does not exist yet; it will be created"
            + ("" if status == OK else " but the directory is not writable"))
    else:
        from trading_lab.storage import SCHEMA_VERSION, SQLiteStore

        try:
            with SQLiteStore(db, readonly=True) as store:
                version = store.schema_version
                runs = store.list_runs(1000)
            running = [r["run_id"] for r in runs if r["kind"] == "paper" and r["status"] == "running"]
            status = OK if version <= SCHEMA_VERSION and os.access(db, os.W_OK) else FAIL
            detail = (f"{db} schema v{version}, {len(runs)} run(s)"
                      + (f", running paper run(s): {', '.join(running)}" if running else ""))
            if not os.access(db, os.W_OK):
                detail += " (not writable)"
            if version < SCHEMA_VERSION:
                detail += f" (upgraded to v{SCHEMA_VERSION} on the next write)"
            add("database", status, detail)
        except (sqlite3.DatabaseError, RuntimeError) as exc:
            add("database", FAIL, f"{db}: {exc}")

    cache = Path(cfg.agents.cache_path)
    if not llm:
        add("answer cache", SKIP, f"{cache} (no AI agent switched on)")
    elif cache.exists():
        try:
            conn = sqlite3.connect(f"{cache.resolve().as_uri()}?mode=ro", uri=True)
            count = conn.execute("SELECT COUNT(*) FROM agent_responses").fetchone()[0]
            conn.close()
            add("answer cache", OK, f"{cache}, {count} cached answer(s)")
        except sqlite3.DatabaseError as exc:
            add("answer cache", FAIL, f"{cache}: {exc}")
    else:
        status = OK if cfg.agents.mode != "replay" else FAIL
        add("answer cache", status, f"{cache} does not exist yet" +
            (" (replay mode needs recorded answers)" if status == FAIL else "; created on the first answer"))

    free = shutil.disk_usage(_nearest_existing(db.parent if cfg.storage.db_path != ":memory:" else Path.cwd())).free
    status = FAIL if free < MIN_FREE_BYTES_FAIL else WARN if free < MIN_FREE_BYTES_WARN else OK
    add("disk space", status, f"{free / 1e9:.1f} GB free")
    add("data cache", OK if _writable_dir(Path(cfg.data.cache_dir)) else WARN, str(Path(cfg.data.cache_dir)))

    if not online:
        add("online checks", SKIP, "run with --online to fetch a public candle and call the model once")
        return checks

    try:
        from trading_lab.data import build_provider

        market = market_factory(cfg) if market_factory else build_provider(cfg)
        now = now or datetime.now(timezone.utc)
        step = timedelta(seconds=timeframe_to_seconds(cfg.market.timeframe))
        symbol = cfg.market.symbols[0]
        candles = market.fetch_ohlcv(symbol, cfg.market.timeframe, now - 50 * step)  # type: ignore[attr-defined]
        if candles.empty:
            add("market data", FAIL, f"no closed {cfg.market.timeframe} candles for {symbol}")
        else:
            last = candles.index[-1].to_pydatetime()
            age = now - (last + step)
            status = OK if age <= 2 * step else WARN
            add("market data", status, f"{symbol} last closed candle opened {last:%Y-%m-%d %H:%M} UTC "
                f"({age.total_seconds() / 60:.0f} min after its close)" + ("" if status == OK else ", stale"))
    except Exception as exc:  # report, never crash the doctor
        add("market data", FAIL, f"{type(exc).__name__}: {exc}")

    if llm and not any(c.name == "model" and c.status == FAIL for c in checks):
        from trading_lab.smoke import provider_smoke_test

        result = provider_smoke_test(cfg)
        add("model call", OK if result.ok else FAIL,
            (f"{result.details['model']} answered in {result.details['latency_seconds']:.2f}s"
             if result.ok else str(result.error)))
    else:
        add("model call", SKIP, "no AI agent switched on" if not llm else "fix the model variables first")
    return checks


def format_checks(checks: list[Check]) -> str:
    fails = sum(c.status == FAIL for c in checks)
    warns = sum(c.status == WARN for c in checks)
    verdict = ("NOT READY: fix the failures above" if fails
               else "ready, with warnings" if warns else "ready")
    return "\n".join([c.line() for c in checks] + ["", f"{verdict} ({fails} failure(s), {warns} warning(s))"])
