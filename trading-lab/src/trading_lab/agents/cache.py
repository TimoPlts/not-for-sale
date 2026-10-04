"""Record/replay cache of agent answers.

An answer is keyed by everything that can influence it: agent name, version
and params, symbol, timeframe, decision bar, and the hash of the exact
context the agent saw. If the market data changed, the key changes too, so a
stale answer is never reused for different inputs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from trading_lab.agents.base import AgentResponse


def cache_key(
    agent: str, version: str, params: dict[str, Any], symbol: str, timeframe: str,
    bar_time: str, context_hash: str,
) -> str:
    payload = json.dumps(
        [agent, version, params, symbol, timeframe, bar_time, context_hash],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache(Protocol):
    def get(self, key: str) -> AgentResponse | None: ...

    def put(self, key: str, response: AgentResponse, *, meta: dict[str, str]) -> None: ...


class MemoryResponseCache:
    def __init__(self) -> None:
        self._data: dict[str, AgentResponse] = {}

    def get(self, key: str) -> AgentResponse | None:
        return self._data.get(key)

    def put(self, key: str, response: AgentResponse, *, meta: dict[str, str]) -> None:
        self._data[key] = response

    def __len__(self) -> int:
        return len(self._data)


class SQLiteResponseCache:
    def __init__(self, path: str | Path) -> None:
        path = str(path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_responses ("
                " cache_key TEXT PRIMARY KEY, agent TEXT NOT NULL, version TEXT NOT NULL,"
                " symbol TEXT NOT NULL, timeframe TEXT NOT NULL, bar_time TEXT NOT NULL,"
                " context_hash TEXT NOT NULL, response_json TEXT NOT NULL, created_at TEXT NOT NULL)"
            )

    def get(self, key: str) -> AgentResponse | None:
        row = self._conn.execute(
            "SELECT response_json FROM agent_responses WHERE cache_key = ?", (key,)
        ).fetchone()
        return AgentResponse.from_json(json.loads(row[0])) if row else None

    def put(self, key: str, response: AgentResponse, *, meta: dict[str, str]) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO agent_responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key, meta["agent"], meta["version"], meta["symbol"], meta["timeframe"],
                    meta["bar_time"], meta["context_hash"], json.dumps(response.to_json()),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM agent_responses").fetchone()[0])

    def close(self) -> None:
        self._conn.close()
