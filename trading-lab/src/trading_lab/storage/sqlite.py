"""SQLite history of runs, signals, decisions, orders, fills, equity and trades.

Timestamps are stored as ISO-8601 UTC strings, and free-form data (metadata,
sizing details, configs, metrics) as JSON. The schema version lives in
``PRAGMA user_version``, and ``_MIGRATIONS`` upgrades older databases in place.
"""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from trading_lab.core.models import (
    ClosedTrade,
    Decision,
    ExecutionReport,
    Fill,
    PortfolioSnapshot,
    Side,
    Signal,
)

SCHEMA_VERSION = 4

_SCHEMA_V1 = """
CREATE TABLE runs (
    run_id             TEXT PRIMARY KEY,
    kind               TEXT NOT NULL,           -- 'backtest' | 'paper'
    status             TEXT NOT NULL,           -- 'running' | 'completed' | 'failed'
    created_at         TEXT NOT NULL,
    finished_at        TEXT,
    period_start       TEXT,
    period_end         TEXT,
    timeframe          TEXT NOT NULL,
    symbols_json       TEXT NOT NULL,
    exchange           TEXT NOT NULL,
    config_json        TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    notes              TEXT NOT NULL DEFAULT '',
    error              TEXT
);
CREATE TABLE signals (
    id            INTEGER PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    timestamp     TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    direction     TEXT NOT NULL,
    confidence    REAL NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX idx_signals_run ON signals(run_id, symbol, timestamp);
CREATE TABLE decisions (
    id                INTEGER PRIMARY KEY,
    run_id            TEXT NOT NULL REFERENCES runs(run_id),
    timestamp         TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    action            TEXT NOT NULL,
    reason            TEXT NOT NULL,
    signal_direction  TEXT,
    signal_confidence REAL,
    quantity          REAL,
    reference_price   REAL,
    stop_price        REAL,
    order_id          TEXT,
    details_json      TEXT NOT NULL
);
CREATE INDEX idx_decisions_run ON decisions(run_id, symbol, timestamp);
CREATE TABLE orders (
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    order_id      TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    order_type    TEXT NOT NULL,
    quantity      REAL NOT NULL,
    stop_price    REAL,
    reason        TEXT NOT NULL,
    status        TEXT NOT NULL,
    reject_reason TEXT NOT NULL,
    PRIMARY KEY (run_id, order_id)
);
CREATE TABLE fills (
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    order_id        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    quantity        REAL NOT NULL,
    reference_price REAL NOT NULL,
    fill_price      REAL NOT NULL,
    fee             REAL NOT NULL,
    stop_price      REAL,
    PRIMARY KEY (run_id, order_id)
);
CREATE TABLE equity_snapshots (
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    timestamp       TEXT NOT NULL,
    cash            REAL NOT NULL,
    positions_value REAL NOT NULL,
    equity          REAL NOT NULL,
    realized_pnl    REAL NOT NULL,
    unrealized_pnl  REAL NOT NULL,
    fees_paid       REAL NOT NULL,
    open_positions  INTEGER NOT NULL,
    PRIMARY KEY (run_id, timestamp)
);
CREATE TABLE closed_trades (
    id            INTEGER PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    symbol        TEXT NOT NULL,
    quantity      REAL NOT NULL,
    entry_price   REAL NOT NULL,
    exit_price    REAL NOT NULL,
    cost_basis    REAL NOT NULL,
    proceeds      REAL NOT NULL,
    pnl           REAL NOT NULL,
    return_pct    REAL NOT NULL,
    opened_at     TEXT NOT NULL,
    closed_at     TEXT NOT NULL,
    exit_order_id TEXT NOT NULL
);
CREATE INDEX idx_trades_run ON closed_trades(run_id);
CREATE TABLE metrics (
    run_id       TEXT PRIMARY KEY REFERENCES runs(run_id),
    metrics_json TEXT NOT NULL
);
"""

_SCHEMA_V2 = """
CREATE TABLE run_state (
    run_id     TEXT PRIMARY KEY REFERENCES runs(run_id),
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

_SCHEMA_V3 = """
CREATE TABLE bars (
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    symbol    TEXT NOT NULL,
    timestamp TEXT NOT NULL,                    -- candle open time
    open      REAL NOT NULL,
    high      REAL NOT NULL,
    low       REAL NOT NULL,
    close     REAL NOT NULL,
    volume    REAL NOT NULL,
    PRIMARY KEY (run_id, symbol, timestamp)
);
CREATE TABLE research_results (
    id           INTEGER PRIMARY KEY,
    created_at   TEXT NOT NULL,
    kind         TEXT NOT NULL,                 -- 'experiment' | 'walkforward' | 'sweep'
    label        TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""

def _schema_v4(conn: sqlite3.Connection) -> None:
    """Closed trades record their side (long or short); existing trades are longs."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(closed_trades)")}
    if "side" not in columns:
        conn.execute("ALTER TABLE closed_trades ADD COLUMN side TEXT NOT NULL DEFAULT 'long'")


# version -> SQL (or a function) that upgrades from version-1 to version.
_MIGRATIONS: dict[int, str | Callable[[sqlite3.Connection], None]] = {
    1: _SCHEMA_V1, 2: _SCHEMA_V2, 3: _SCHEMA_V3, 4: _schema_v4,
}


def _ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return _ts(obj)
        if hasattr(obj, "items"):
            return dict(obj.items())
        return str(obj)

    def clean(obj: Any) -> Any:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
        if isinstance(obj, dict) or hasattr(obj, "items"):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj

    return json.dumps(clean(value), default=default, sort_keys=True, allow_nan=False)


BUSY_TIMEOUT_SECONDS = 30.0  # wait this long for another process's lock (e.g. the dashboard)


class SQLiteStore:
    """Thin repository over one SQLite database (``":memory:"`` is supported).

    With ``readonly=True`` the file is opened in SQLite's read-only mode: any
    write fails inside SQLite itself, and no schema migration is attempted.
    """

    def __init__(self, path: str | Path, *, readonly: bool = False) -> None:
        self._path = str(path)
        self.readonly = readonly
        if readonly:
            if self._path == ":memory:" or not Path(self._path).exists():
                raise FileNotFoundError(f"database not found: {self._path}")
            uri = f"{Path(self._path).resolve().as_uri()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_SECONDS)
        else:
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._path, timeout=BUSY_TIMEOUT_SECONDS)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._in_atomic = False
        if not readonly:
            self._migrate()
        elif self.schema_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema v{self.schema_version} is newer than this code (v{SCHEMA_VERSION})"
            )

    def has_table(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        return row is not None

    @contextlib.contextmanager
    def atomic(self):  # type: ignore[no-untyped-def]
        """Group several writes into one transaction (all or nothing)."""
        if self._in_atomic:
            yield
            return
        self._in_atomic = True
        try:
            with self._conn:
                yield
        finally:
            self._in_atomic = False

    def _tx(self) -> contextlib.AbstractContextManager:
        return contextlib.nullcontext() if self._in_atomic else self._conn

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    def _migrate(self) -> None:
        current = self.schema_version
        if current > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema v{current} is newer than this code (v{SCHEMA_VERSION})"
            )
        for version in range(current + 1, SCHEMA_VERSION + 1):
            migration = _MIGRATIONS[version]
            with self._conn:
                if callable(migration):
                    migration(self._conn)
                else:
                    self._conn.executescript(migration)
                self._conn.execute(f"PRAGMA user_version = {version}")

    # ------------------------------------------------------------------ runs
    def create_run(
        self,
        run_id: str,
        *,
        kind: str,
        timeframe: str,
        symbols: Sequence[str],
        exchange: str,
        config: dict[str, Any],
        config_fingerprint: str,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
        notes: str = "",
    ) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT INTO runs (run_id, kind, status, created_at, period_start, period_end, "
                "timeframe, symbols_json, exchange, config_json, config_fingerprint, notes) "
                "VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    kind,
                    _ts(datetime.now(timezone.utc)),
                    _ts(period_start) if period_start else None,
                    _ts(period_end) if period_end else None,
                    timeframe,
                    _json(list(symbols)),
                    exchange,
                    _json(config),
                    config_fingerprint,
                    notes,
                ),
            )

    def finish_run(self, run_id: str, status: str = "completed", error: str | None = None) -> None:
        with self._tx():
            self._conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE run_id = ?",
                (status, _ts(datetime.now(timezone.utc)), error, run_id),
            )

    def set_run_status(self, run_id: str, status: str) -> None:
        with self._tx():
            self._conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def save_state(self, run_id: str, state: dict[str, Any]) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO run_state (run_id, state_json, updated_at) VALUES (?, ?, ?)",
                (run_id, _json(state), _ts(datetime.now(timezone.utc))),
            )

    def load_state(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT state_json FROM run_state WHERE run_id = ?", (run_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None
        record = dict(zip([c[0] for c in cur.description], row))
        record["symbols"] = json.loads(record.pop("symbols_json"))
        record["config"] = json.loads(record.pop("config_json"))
        return record

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT r.run_id, r.kind, r.status, r.created_at, r.period_start, r.period_end, "
            "r.timeframe, r.symbols_json, m.metrics_json FROM runs r "
            "LEFT JOIN metrics m ON m.run_id = r.run_id ORDER BY r.created_at DESC LIMIT ?",
            (limit,),
        )
        out = []
        for row in cur.fetchall():
            record = dict(zip([c[0] for c in cur.description], row))
            record["symbols"] = json.loads(record.pop("symbols_json"))
            metrics_json = record.pop("metrics_json")
            record["metrics"] = json.loads(metrics_json) if metrics_json else None
            out.append(record)
        return out

    # --------------------------------------------------------------- writes
    def add_signals(self, run_id: str, signals: Iterable[Signal]) -> None:
        with self._tx():
            self._conn.executemany(
                "INSERT INTO signals (run_id, timestamp, symbol, strategy, direction, confidence, "
                "metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        run_id,
                        _ts(s.timestamp),
                        s.symbol,
                        s.strategy,
                        s.direction.value,
                        s.confidence,
                        _json(dict(s.metadata)),
                    )
                    for s in signals
                ),
            )

    def add_decisions(self, run_id: str, decisions: Iterable[Decision]) -> None:
        with self._tx():
            self._conn.executemany(
                "INSERT INTO decisions (run_id, timestamp, symbol, action, reason, signal_direction, "
                "signal_confidence, quantity, reference_price, stop_price, order_id, details_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        run_id,
                        _ts(d.timestamp),
                        d.symbol,
                        d.action.value,
                        d.reason,
                        d.signal_direction.value if d.signal_direction else None,
                        d.signal_confidence,
                        d.quantity,
                        d.reference_price,
                        d.stop_price,
                        d.order_id,
                        _json(dict(d.details)),
                    )
                    for d in decisions
                ),
            )

    def add_execution_reports(self, run_id: str, reports: Iterable[ExecutionReport]) -> None:
        reports = list(reports)
        with self._tx():
            self._conn.executemany(
                "INSERT INTO orders (run_id, order_id, timestamp, symbol, side, order_type, "
                "quantity, stop_price, reason, status, reject_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        run_id,
                        r.order_id,
                        _ts(r.order.timestamp),
                        r.order.symbol,
                        r.order.side.value,
                        r.order.order_type.value,
                        r.order.quantity,
                        r.order.stop_price,
                        r.order.reason,
                        r.status.value,
                        r.reason,
                    )
                    for r in reports
                ),
            )
            self._conn.executemany(
                "INSERT INTO fills (run_id, order_id, timestamp, symbol, side, quantity, "
                "reference_price, fill_price, fee, stop_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        run_id,
                        f.order_id,
                        _ts(f.timestamp),
                        f.symbol,
                        f.side.value,
                        f.quantity,
                        f.reference_price,
                        f.fill_price,
                        f.fee,
                        f.stop_price,
                    )
                    for f in (r.fill for r in reports if r.fill is not None)
                ),
            )

    def add_snapshots(self, run_id: str, snapshots: Iterable[PortfolioSnapshot]) -> None:
        with self._tx():
            self._conn.executemany(
                "INSERT OR REPLACE INTO equity_snapshots (run_id, timestamp, cash, positions_value, equity, "
                "realized_pnl, unrealized_pnl, fees_paid, open_positions) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        run_id,
                        _ts(s.timestamp),
                        s.cash,
                        s.positions_value,
                        s.equity,
                        s.realized_pnl,
                        s.unrealized_pnl,
                        s.fees_paid,
                        s.open_positions,
                    )
                    for s in snapshots
                ),
            )

    def add_closed_trades(self, run_id: str, trades: Iterable[ClosedTrade]) -> None:
        with self._tx():
            self._conn.executemany(
                "INSERT INTO closed_trades (run_id, symbol, quantity, entry_price, exit_price, "
                "cost_basis, proceeds, pnl, return_pct, opened_at, closed_at, exit_order_id, side) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        run_id,
                        t.symbol,
                        t.quantity,
                        t.entry_price,
                        t.exit_price,
                        t.cost_basis,
                        t.proceeds,
                        t.pnl,
                        t.return_pct,
                        _ts(t.opened_at),
                        _ts(t.closed_at),
                        t.exit_order_id,
                        t.side,
                    )
                    for t in trades
                ),
            )

    def add_bars(self, run_id: str, bars: Iterable[tuple[datetime, str, Any]]) -> None:
        """Candles the run traded on, as ``(open time, symbol, bar)`` with OHLCV attributes."""
        with self._tx():
            self._conn.executemany(
                "INSERT OR REPLACE INTO bars (run_id, symbol, timestamp, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (run_id, symbol, _ts(ts), b.open, b.high, b.low, b.close, b.volume)
                    for ts, symbol, b in bars
                ),
            )

    def add_research_result(self, kind: str, label: str, payload: dict[str, Any]) -> int:
        with self._tx():
            cur = self._conn.execute(
                "INSERT INTO research_results (created_at, kind, label, payload_json) VALUES (?, ?, ?, ?)",
                (_ts(datetime.now(timezone.utc)), kind, label, _json(payload)),
            )
            return int(cur.lastrowid or 0)

    def save_metrics(self, run_id: str, metrics: dict[str, Any]) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO metrics (run_id, metrics_json) VALUES (?, ?)",
                (run_id, _json(metrics)),
            )

    # ---------------------------------------------------------------- reads
    def _frame(self, sql: str, params: Sequence[Any], ts_cols: Sequence[str]) -> pd.DataFrame:
        frame = pd.read_sql_query(sql, self._conn, params=list(params))
        for col in ts_cols:
            frame[col] = pd.to_datetime(frame[col], utc=True, format="ISO8601")
        return frame

    def load_equity_curve(self, run_id: str) -> pd.DataFrame:
        frame = self._frame(
            "SELECT timestamp, cash, positions_value, equity, realized_pnl, unrealized_pnl, "
            "fees_paid, open_positions FROM equity_snapshots WHERE run_id = ? ORDER BY timestamp",
            (run_id,),
            ["timestamp"],
        )
        return frame.set_index("timestamp")

    def has_column(self, table: str, column: str) -> bool:
        return any(row[1] == column for row in self._conn.execute(f"PRAGMA table_info({table})"))

    def load_closed_trades(self, run_id: str) -> list[ClosedTrade]:
        side = "side" if self.has_column("closed_trades", "side") else "'long'"  # read-only, before v4
        cur = self._conn.execute(
            "SELECT symbol, quantity, entry_price, exit_price, cost_basis, proceeds, pnl, "
            f"opened_at, closed_at, exit_order_id, {side} FROM closed_trades WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        return [
            ClosedTrade(
                symbol=r[0],
                quantity=r[1],
                entry_price=r[2],
                exit_price=r[3],
                cost_basis=r[4],
                proceeds=r[5],
                pnl=r[6],
                opened_at=_parse_ts(r[7]),
                closed_at=_parse_ts(r[8]),
                exit_order_id=r[9],
                side=r[10],
            )
            for r in cur.fetchall()
        ]

    def load_fills(self, run_id: str) -> pd.DataFrame:
        return self._frame(
            "SELECT f.*, o.reason FROM fills f JOIN orders o "
            "ON o.run_id = f.run_id AND o.order_id = f.order_id "
            "WHERE f.run_id = ? ORDER BY f.timestamp, f.order_id",
            (run_id,),
            ["timestamp"],
        )

    def load_fill_objects(self, run_id: str) -> list[Fill]:
        """Fills in execution order, e.g. to rebuild a portfolio by replaying them."""
        cur = self._conn.execute(
            "SELECT order_id, symbol, side, quantity, reference_price, fill_price, fee, timestamp, "
            "stop_price FROM fills WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        )
        return [
            Fill(r[0], r[1], Side(r[2]), r[3], r[4], r[5], r[6], _parse_ts(r[7]), stop_price=r[8])
            for r in cur.fetchall()
        ]

    def load_decisions(self, run_id: str, *, include_holds: bool = True) -> pd.DataFrame:
        sql = "SELECT * FROM decisions WHERE run_id = ?"
        if not include_holds:
            sql += " AND action != 'hold'"
        return self._frame(sql + " ORDER BY id", (run_id,), ["timestamp"])

    def load_signals(
        self,
        run_id: str,
        symbol: str | None = None,
        *,
        since: datetime | None = None,
        strategies: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        sql, params = "SELECT * FROM signals WHERE run_id = ?", [run_id]
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol)
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(_ts(since))
        if strategies is not None:
            sql += f" AND strategy IN ({', '.join('?' * len(strategies))})"
            params.extend(strategies)
        return self._frame(sql + " ORDER BY id", params, ["timestamp"])

    def latest_signal_time(self, run_id: str) -> datetime | None:
        row = self._conn.execute("SELECT MAX(timestamp) FROM signals WHERE run_id = ?", (run_id,)).fetchone()
        return _parse_ts(row[0]) if row and row[0] else None

    def load_bars(self, run_id: str, symbol: str | None = None) -> pd.DataFrame:
        if not self.has_table("bars"):  # a database opened read-only before its v3 upgrade
            frame = pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            return frame
        sql, params = "SELECT symbol, timestamp, open, high, low, close, volume FROM bars WHERE run_id = ?", [run_id]
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol)
        return self._frame(sql + " ORDER BY symbol, timestamp", params, ["timestamp"])

    def load_closes(self, run_id: str) -> dict[str, pd.Series]:
        """Close price per symbol, indexed by candle open time (empty for runs before schema v3)."""
        bars = self.load_bars(run_id)
        return {
            str(sym): group.set_index("timestamp")["close"].rename(str(sym))
            for sym, group in bars.groupby("symbol", sort=True)
        }

    def list_research_results(self, kind: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        sql, params = "SELECT id, created_at, kind, label, payload_json FROM research_results", []
        if kind is not None:
            sql += " WHERE kind = ?"
            params.append(kind)
        rows = self._conn.execute(sql + " ORDER BY id DESC LIMIT ?", [*params, limit]).fetchall()
        return [
            {"id": r[0], "created_at": r[1], "kind": r[2], "label": r[3], "payload": json.loads(r[4])}
            for r in rows
        ]

    def load_metrics(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT metrics_json FROM metrics WHERE run_id = ?", (run_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def count(self, table: str, run_id: str) -> int:
        if table not in {"signals", "decisions", "orders", "fills", "equity_snapshots", "closed_trades", "bars"}:
            raise ValueError(f"unknown table {table!r}")
        return int(
            self._conn.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)).fetchone()[0]
        )
