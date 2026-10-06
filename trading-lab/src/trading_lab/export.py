"""Export a stored run (backtest or paper) to CSV files and a JSON summary.

``export_run`` writes into a new (or empty) directory:

=====================  ======================================================
``equity_curve.csv``   one row per bar: cash, positions, equity, PnL, fees
``trades.csv``         closed trades (same columns as ``backtest --export``)
``fills.csv``          simulated fills (same columns as ``backtest --export``)
``decisions.csv``      the ensemble's decisions (HOLDs only with ``holds``)
``signals.csv``        every strategy and agent vote, with the agent's
                       rationale, label, cache status and error flattened out
``bars.csv``           the candles the run traded on
``summary.json``       the run, its config, metrics, row counts and the
                       SHA-256 of every file written
=====================  ======================================================

Only the database is read. No export can contain credentials, because they
only ever come from environment variables and are never stored in a run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from trading_lab.core.models import ClosedTrade, Fill

EXPORT_VERSION = 1
_AGENT_FIELDS = ("rationale", "cache", "error")
_LABELS = ("regime", "momentum_state", "risk_state")


def trades_frame(trades: Iterable[ClosedTrade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": t.symbol, "quantity": t.quantity, "entry_price": t.entry_price,
        "exit_price": t.exit_price, "pnl": t.pnl, "return_pct": t.return_pct,
        "opened_at": t.opened_at, "closed_at": t.closed_at,
    } for t in trades], columns=["symbol", "quantity", "entry_price", "exit_price", "pnl", "return_pct",
                                 "opened_at", "closed_at"])


def fills_frame(fills: Iterable[Fill]) -> pd.DataFrame:
    return pd.DataFrame([{
        "timestamp": f.timestamp, "symbol": f.symbol, "side": f.side.value,
        "quantity": f.quantity, "reference_price": f.reference_price,
        "fill_price": f.fill_price, "fee": f.fee,
    } for f in fills], columns=["timestamp", "symbol", "side", "quantity", "reference_price", "fill_price", "fee"])


def signals_frame(signals: pd.DataFrame) -> pd.DataFrame:
    """Stored signals with the agent fields pulled out of ``metadata_json``."""
    out = signals.drop(columns=[c for c in ("id", "run_id") if c in signals.columns]).copy()
    meta = [json.loads(m) for m in out["metadata_json"]]
    for name in _AGENT_FIELDS:
        out[name] = [m.get(name) for m in meta]
    out["label"] = [next((f"{k}={m[k]}" for k in _LABELS if k in m), None) for m in meta]
    out["reason"] = [m.get("reason") for m in meta]
    return out


@dataclass(frozen=True, slots=True)
class ExportResult:
    run_id: str
    directory: Path
    files: dict[str, int]  # file name -> data rows


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_run(store: Any, run_id: str, directory: str | Path, *, holds: bool = False,
               overwrite: bool = False) -> ExportResult:
    """Write the run's CSV files and ``summary.json`` into ``directory``.

    The directory must be new or empty unless ``overwrite`` is set (then the
    export files in it are replaced; other files are left alone).
    """
    from trading_lab.reporting import run_metrics

    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"unknown run id {run_id!r}")
    out = Path(directory)
    if out.exists() and not out.is_dir():
        raise ValueError(f"{out} exists and is not a directory")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise ValueError(f"{out} is not empty (use overwrite to replace the export files)")
    out.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {
        "equity_curve.csv": store.load_equity_curve(run_id).reset_index(),
        "trades.csv": trades_frame(store.load_closed_trades(run_id)),
        "fills.csv": fills_frame(store.load_fill_objects(run_id)),
        "decisions.csv": store.load_decisions(run_id, include_holds=holds).drop(
            columns=["id", "run_id"], errors="ignore"),
        "signals.csv": signals_frame(store.load_signals(run_id)),
        "bars.csv": store.load_bars(run_id),
    }
    files: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for name, frame in tables.items():
        path = out / name
        frame.to_csv(path, index=False)
        files[name] = len(frame)
        hashes[name] = _sha256(path)

    stored = store.load_metrics(run_id)
    if stored is not None:
        metrics, source = stored, "stored"
    else:
        computed = run_metrics(store, run_id)
        metrics, source = (None, None) if computed is None else (computed.to_dict(), "recomputed")
    decisions = tables["decisions.csv"]
    summary = {
        "export_version": EXPORT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": {k: v for k, v in run.items() if k != "config"},
        "config": run["config"],
        "metrics": metrics,
        "metrics_source": source,
        "decision_counts": decisions["action"].value_counts().sort_index().to_dict() if len(decisions) else {},
        "holds_included": holds,
        "files": {name: {"rows": files[name], "sha256": hashes[name]} for name in tables},
    }
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=2, sort_keys=True) + "\n")
    return ExportResult(run_id, out, files)
