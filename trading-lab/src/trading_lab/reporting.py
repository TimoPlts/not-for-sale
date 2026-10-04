"""Helpers to compute metrics for stored runs (backtests and paper runs alike)."""

from __future__ import annotations

from trading_lab.config import AppConfig
from trading_lab.metrics import PerformanceMetrics, compute_metrics
from trading_lab.storage import SQLiteStore


def run_metrics(store: SQLiteStore, run_id: str) -> PerformanceMetrics | None:
    """Recompute metrics from a run's stored equity curve and trades."""
    run = store.get_run(run_id)
    if run is None:
        return None
    curve = store.load_equity_curve(run_id)
    if curve.empty:
        return None
    config = AppConfig.from_dict(run["config"])
    return compute_metrics(
        [config.portfolio.initial_cash, *curve["equity"].tolist()],
        store.load_closed_trades(run_id),
        run["timeframe"],
        total_fees=float(curve["fees_paid"].iloc[-1]),
        in_market=(curve["open_positions"] > 0).tolist(),
    )
