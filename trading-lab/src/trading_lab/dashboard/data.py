"""Read-only data layer for the dashboard (and ``trading-lab dashboard-data``).

Everything here is computed from the SQLite history that backtests and live
paper runs already write. The database is opened in SQLite's read-only
mode, so this module *cannot* change anything. It has no order, executor or
trader code and no way to place, cancel or modify a (simulated) order.

All public methods return plain Python values or pandas frames, so any
front end (Streamlit today, something else later) can use them.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.config import AppConfig
from trading_lab.ensemble.voting import ENSEMBLE_NAME
from trading_lab.execution import CostModel
from trading_lab.llm.usage import usage_from_signals
from trading_lab.metrics import compute_metrics
from trading_lab.metrics.benchmark import buy_and_hold_equity
from trading_lab.portfolio import Portfolio
from trading_lab.reporting import run_metrics
from trading_lab.research.attribution import attribute_run
from trading_lab.storage import SQLiteStore

_NOT_A_VOTE = ("warmup", "not a decision bar")
_LABELS = ("regime", "momentum_state", "risk_state")


def _clean(value: Any) -> Any:
    """JSON-safe: NaN/inf become None, timestamps become ISO strings, numpy scalars plain numbers."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_clean(row) for row in frame.to_dict(orient="records")]


class DashboardData:
    """Read-only view of one SQLite history database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.store = SQLiteStore(db_path, readonly=True)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> DashboardData:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ runs
    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return [_clean(r) for r in self.store.list_runs(limit)]

    def default_run_id(self) -> str | None:
        """The running paper run if there is one, else the most recent run."""
        runs = self.store.list_runs(200)
        for r in runs:
            if r["kind"] == "paper" and r["status"] == "running":
                return str(r["run_id"])
        return str(runs[0]["run_id"]) if runs else None

    def _run(self, run_id: str) -> tuple[dict[str, Any], AppConfig]:
        run = self.store.get_run(run_id)
        if run is None:
            raise KeyError(f"unknown run id {run_id!r}")
        return run, AppConfig.from_dict(run["config"])

    # ------------------------------------------------------------- portfolio
    def equity_curve(self, run_id: str) -> pd.DataFrame:
        """Equity, drawdown and (when prices are stored) the buy & hold benchmark."""
        curve = self.store.load_equity_curve(run_id)
        if curve.empty:
            return curve
        peak = curve["equity"].cummax()
        curve["drawdown"] = curve["equity"] / peak - 1.0
        bench = self._benchmark_curve(run_id)
        if bench is not None:
            curve["buy_and_hold"] = bench.reindex(curve.index).ffill()
        return curve

    def _candles(self, run_id: str) -> dict[str, pd.DataFrame]:
        if not self.store.has_table("bars"):
            return {}
        bars = self.store.load_bars(run_id)
        return {
            str(sym): group.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
            for sym, group in bars.groupby("symbol", sort=True)
        }

    def _benchmark_curve(self, run_id: str) -> pd.Series | None:
        candles = self._candles(run_id)
        if not candles:
            return None
        _, config = self._run(run_id)
        timeline = sorted(set().union(*(frame.index for frame in candles.values())))
        return buy_and_hold_equity(
            candles, timeline, config.portfolio.initial_cash, CostModel.from_config(config.execution)
        )

    def _portfolio(self, run_id: str, config: AppConfig) -> Portfolio:
        portfolio = Portfolio(config.portfolio.initial_cash, config.portfolio.quote_currency,
                              allow_short=config.risk.allow_short)
        for fill in self.store.load_fill_objects(run_id):
            portfolio.apply_fill(fill)
        return portfolio

    def _exposure(self, run_id: str, config: AppConfig, positions_value: float) -> float:
        """Absolute market exposure. For longs that is the positions' value; a short's value is
        its collateral plus gain, so with shorts open use quantity x price instead."""
        if not config.risk.allow_short:
            return positions_value
        positions = self.open_positions(run_id)
        if not any(p["side"] == "short" for p in positions):
            return positions_value
        return float(sum(p["value"] or 0.0 for p in positions))

    def _last_prices(self, run_id: str) -> dict[str, float]:
        state = self.store.load_state(run_id) or {}
        prices = dict(state.get("last_close", {}))
        for sym, frame in self._candles(run_id).items():
            prices.setdefault(sym, float(frame["close"].iloc[-1]))
        return prices

    def open_positions(self, run_id: str) -> list[dict[str, Any]]:
        _, config = self._run(run_id)
        portfolio = self._portfolio(run_id, config)
        prices = self._last_prices(run_id)
        trailing = (self.store.load_state(run_id) or {}).get("trailing", {})
        out = []
        for sym, pos in sorted(portfolio.positions.items()):
            price = prices.get(sym)
            unrealized = None if price is None else pos.unrealized_pnl(price)
            out.append(_clean({
                "symbol": sym,
                "side": pos.side,
                "quantity": pos.quantity,
                "entry_price": pos.avg_entry_price,
                "current_price": price,
                "value": None if price is None else pos.quantity * price,
                "unrealized_pnl": unrealized,
                "unrealized_pct": None if unrealized is None else unrealized / pos.cost_basis,
                "stop_price": trailing.get(sym, {}).get("stop", pos.stop_price),
                "trailing": "stop" in trailing.get(sym, {}),
                "opened_at": pos.opened_at,
            }))
        return out

    def overview(self, run_id: str) -> dict[str, Any]:
        run, config = self._run(run_id)
        curve = self.store.load_equity_curve(run_id)
        initial = config.portfolio.initial_cash
        out: dict[str, Any] = {
            "run_id": run_id, "kind": run["kind"], "status": run["status"], "timeframe": run["timeframe"],
            "symbols": run["symbols"], "data_source": run["exchange"], "created_at": run["created_at"],
            "period_start": run["period_start"], "period_end": run["period_end"], "initial_cash": initial,
            "agents_mode": config.agents.mode, "error": run["error"],
        }
        if curve.empty:
            return _clean({**out, "last_bar": None,
                           "health": (self.store.load_state(run_id) or {}).get("health")})
        last = curve.iloc[-1]
        equity = float(last["equity"])
        peak = max(initial, float(curve["equity"].max()))
        last_ts = curve.index[-1]
        day_start = last_ts.normalize()
        before = curve.loc[curve.index < day_start, "equity"]
        start_of_day = float(before.iloc[-1]) if not before.empty else initial
        out.update({
            "last_bar": last_ts,
            "equity": equity,
            "cash": float(last["cash"]),
            "positions_value": float(last["positions_value"]),
            "exposure_pct": self._exposure(run_id, config, float(last["positions_value"])) / equity
            if equity > 0 else 0.0,
            "realized_pnl": float(last["realized_pnl"]),
            "unrealized_pnl": float(last["unrealized_pnl"]),
            "fees_paid": float(last["fees_paid"]),
            "open_positions": int(last["open_positions"]),
            "total_return": equity / initial - 1.0,
            "drawdown": equity / peak - 1.0,
            "max_drawdown": float((curve["equity"] / curve["equity"].cummax().clip(lower=initial) - 1.0).min()),
            "daily_pnl": equity - start_of_day,
            "daily_pnl_pct": equity / start_of_day - 1.0 if start_of_day > 0 else None,
        })
        out["breakers"] = self.breaker_status(run_id)
        out["health"] = (self.store.load_state(run_id) or {}).get("health")
        return _clean(out)

    def breaker_status(self, run_id: str) -> dict[str, Any]:
        """Circuit-breaker state: saved state for paper runs, trips for backtests."""
        state = (self.store.load_state(run_id) or {}).get("breakers")
        decisions = self.store.load_decisions(run_id, include_holds=False)
        trips = decisions[decisions["action"] == "circuit_breaker"]
        recent = [{"timestamp": r.timestamp, "reason": r.reason} for r in trips.tail(10).itertuples()]
        if state is None:
            return _clean({"source": "decisions", "trips": recent,
                           "kill_switch_active": any("max drawdown" in t["reason"] for t in recent),
                           "halted_reason": None, "daily_blocked_day": None, "cooldowns": {}})
        return _clean({
            "source": "state",
            "kill_switch_active": state.get("halted_reason") is not None,
            "halted_reason": state.get("halted_reason"),
            "daily_blocked_day": state.get("daily_blocked_day"),
            "peak_equity": state.get("peak_equity"),
            "day_start_equity": state.get("day_start_equity"),
            "cooldowns": state.get("cooldown_until", {}),
            "trips": recent,
        })

    def working_orders(self, run_id: str) -> dict[str, Any]:
        state = self.store.load_state(run_id) or {}
        return _clean({"scheduled": state.get("pending", {}), "working_limits": state.get("resting", {})})

    def recent_trades(self, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        trades = self.store.load_closed_trades(run_id)[-limit:]
        return [_clean({
            "symbol": t.symbol, "side": t.side, "quantity": t.quantity, "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl, "return_pct": t.return_pct, "opened_at": t.opened_at, "closed_at": t.closed_at,
        }) for t in reversed(trades)]

    def recent_fills(self, run_id: str, limit: int = 20) -> list[dict[str, Any]]:
        fills = self.store.load_fills(run_id).tail(limit).iloc[::-1]
        return _records(fills.drop(columns=["run_id"]))

    # ------------------------------------------------------------- signals
    def _signal_rows(self, run_id: str, since: datetime | None = None) -> list[dict[str, Any]]:
        frame = self.store.load_signals(run_id, since=since)
        rows = []
        for r in frame.itertuples(index=False):
            meta = json.loads(r.metadata_json)
            rows.append({"timestamp": r.timestamp, "symbol": r.symbol, "strategy": r.strategy,
                         "direction": r.direction, "confidence": r.confidence, "meta": meta})
        return rows

    def recent_signals(self, run_id: str, bars: int = 24, include_holds: bool = False) -> list[dict[str, Any]]:
        last = self.store.latest_signal_time(run_id)
        if last is None:
            return []
        _, config = self._run(run_id)
        since = last - timedelta(seconds=_tf_seconds(config.market.timeframe) * (bars - 1))
        out = []
        for row in reversed(self._signal_rows(run_id, since)):
            if not include_holds and row["direction"] == "hold":
                continue
            meta = row["meta"]
            out.append(_clean({
                "timestamp": row["timestamp"], "symbol": row["symbol"], "strategy": row["strategy"],
                "direction": row["direction"], "confidence": row["confidence"],
                "rationale": meta.get("rationale"), **{k: meta[k] for k in _LABELS if k in meta},
            }))
        return out

    def latest_decision(self, run_id: str) -> dict[str, Any]:
        """Per symbol, the last bar's votes, the ensemble result and the actions taken."""
        last = self.store.latest_signal_time(run_id)
        if last is None:
            return {}
        rows = self._signal_rows(run_id, since=last)
        decisions = self.store.load_decisions(run_id)
        decisions = decisions[decisions["timestamp"] == pd.Timestamp(last)]
        out: dict[str, Any] = {"timestamp": last.isoformat(), "symbols": {}}
        for row in rows:
            sym = out["symbols"].setdefault(row["symbol"], {"votes": [], "ensemble": None, "actions": []})
            meta = row["meta"]
            if row["strategy"] == ENSEMBLE_NAME:
                sym["ensemble"] = _clean({"direction": row["direction"], "confidence": row["confidence"],
                                          "net_score": meta.get("net_score"), "buy_votes": meta.get("buy_votes"),
                                          "sell_votes": meta.get("sell_votes")})
                weights = {v["strategy"]: v["weight"] for v in meta.get("votes", [])}
                for vote in sym["votes"]:
                    vote["weight"] = weights.get(vote["strategy"])
            else:
                sym["votes"].append(_clean({
                    "strategy": row["strategy"], "direction": row["direction"], "confidence": row["confidence"],
                    "is_agent": "agent" in (meta.get("params") or {}), "rationale": meta.get("rationale"),
                    "error": meta.get("error"), "note": meta.get("reason"), "cache": meta.get("cache"),
                    **{k: meta[k] for k in _LABELS if k in meta},
                }))
        for d in decisions.itertuples():
            if d.symbol in out["symbols"]:
                out["symbols"][d.symbol]["actions"].append({"action": d.action, "reason": d.reason})
        return out

    def latest_rationales(self, run_id: str, lookback_bars: int = 500) -> list[dict[str, Any]]:
        """The most recent answered decision of each agent, per symbol."""
        last = self.store.latest_signal_time(run_id)
        if last is None:
            return []
        _, config = self._run(run_id)
        since = last - timedelta(seconds=_tf_seconds(config.market.timeframe) * lookback_bars)
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self._signal_rows(run_id, since):
            meta = row["meta"]
            if "agent" not in (meta.get("params") or {}) or meta.get("reason") in _NOT_A_VOTE:
                continue
            latest[(row["strategy"], row["symbol"])] = _clean({
                "strategy": row["strategy"], "symbol": row["symbol"], "timestamp": row["timestamp"],
                "direction": row["direction"], "confidence": row["confidence"],
                "rationale": meta.get("rationale"), "error": meta.get("error"), "cache": meta.get("cache"),
                "model": (meta.get("llm") or {}).get("model") or (meta.get("params") or {}).get("model"),
                **{k: meta[k] for k in _LABELS if k in meta},
            })
        return [latest[k] for k in sorted(latest)]

    # ------------------------------------------------------- agents & usage
    def agent_performance(self, run_id: str, horizon: int = 4, include_strategies: bool = True) -> list[dict[str, Any]]:
        results = attribute_run(self.store, run_id, horizon=horizon)
        return [_clean(a.to_dict()) for a in results.values() if a.is_agent or include_strategies]

    def usage(self, run_id: str) -> dict[str, Any]:
        frame = self.store.load_signals(run_id)
        pairs = [(r.strategy, json.loads(r.metadata_json)) for r in frame.itertuples(index=False)
                 if r.strategy != ENSEMBLE_NAME]
        per_agent = usage_from_signals(pairs)
        total: dict[str, Any] = {}
        if per_agent:
            from trading_lab.llm.usage import total_usage

            total = total_usage(per_agent).to_dict()
        return _clean({"total": total, "per_agent": {k: v.to_dict() for k, v in sorted(per_agent.items())}})

    # -------------------------------------------------------------- research
    def research(self, run_id: str) -> dict[str, Any]:
        run, config = self._run(run_id)
        metrics = run_metrics(self.store, run_id)
        stored = self.store.load_metrics(run_id) or {}
        bench = stored.get("benchmark")
        if bench is None:
            curve = self._benchmark_curve(run_id)
            if curve is not None and len(curve) > 1:
                bench = compute_metrics([config.portfolio.initial_cash, *curve.tolist()], [], run["timeframe"],
                                        in_market=[True] * len(curve)).to_dict()
        relative = stored.get("relative")
        if relative is None:
            curve = self.equity_curve(run_id)
            if "buy_and_hold" in curve and not curve["buy_and_hold"].isna().any():
                from trading_lab.metrics import relative_metrics

                initial = config.portfolio.initial_cash
                rel = relative_metrics([initial, *curve["equity"].tolist()],
                                       [initial, *curve["buy_and_hold"].tolist()], run["timeframe"])
                relative = None if rel is None else rel.to_dict()
        results = self.store.list_research_results(limit=10) if self.store.has_table("research_results") else []
        return _clean({"metrics": None if metrics is None else metrics.to_dict(), "benchmark": bench,
                       "relative": relative, "research_results": results})

    # -------------------------------------------------------------- snapshot
    def snapshot(self, run_id: str | None = None, *, horizon: int = 4) -> dict[str, Any]:
        """Everything the dashboard shows, as one JSON-serialisable dict."""
        run_id = run_id or self.default_run_id()
        if run_id is None:
            return {"run_id": None, "runs": []}
        curve = self.equity_curve(run_id)
        return _clean({
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc),
            "overview": self.overview(run_id),
            "open_positions": self.open_positions(run_id),
            "working_orders": self.working_orders(run_id),
            "recent_trades": self.recent_trades(run_id),
            "recent_signals": self.recent_signals(run_id),
            "latest_decision": self.latest_decision(run_id),
            "latest_rationales": self.latest_rationales(run_id),
            "agent_performance": self.agent_performance(run_id, horizon),
            "usage": self.usage(run_id),
            "research": self.research(run_id),
            "equity_points": len(curve),
        })


def _tf_seconds(timeframe: str) -> int:
    from trading_lab.core.symbols import timeframe_to_seconds

    return timeframe_to_seconds(timeframe)
