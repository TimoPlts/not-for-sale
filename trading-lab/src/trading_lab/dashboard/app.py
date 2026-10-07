"""trading-lab dashboard (Streamlit). Read-only: it displays the SQLite history and nothing else.

Start it with ``trading-lab dashboard`` (or ``streamlit run`` on this file with
``TRADING_LAB_DB`` pointing at the database). It only uses ``DashboardData``,
which opens the database in SQLite's read-only mode; there is no button,
form or code path that can place, cancel or change a (simulated) order.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from trading_lab.dashboard.data import DashboardData

DB_ENV = "TRADING_LAB_DB"
DEFAULT_DB = "data/trading_lab.db"


def _pct(value: Any, signed: bool = True) -> str:
    if isinstance(value, str):  # e.g. a profit factor of "inf" when nothing was lost
        return value
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.1%}"


def _num(value: Any, digits: int = 2) -> str:
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _breaker_label(breakers: dict[str, Any]) -> str:
    if breakers.get("kill_switch_active"):
        return "KILL SWITCH"
    if breakers.get("daily_blocked_day"):
        return "daily limit"
    if breakers.get("cooldowns"):
        return f"{len(breakers['cooldowns'])} cooldown(s)"
    return "OK"


def portfolio_section(data: DashboardData, run_id: str) -> None:
    o = data.overview(run_id)
    st.subheader("Portfolio")
    st.caption(f"Run {run_id} · {o['kind']} · {o['status']} · {o['timeframe']} · {', '.join(o['symbols'])} · "
               f"data {o['data_source']} · agents {o['agents_mode']} · last bar {o.get('last_bar') or '-'}")
    health = o.get("health") or {}
    if health.get("consecutive_errors"):
        st.warning(f"The last {health['consecutive_errors']} cycle(s) could not be processed (retrying with "
                   f"back-off): {health.get('last_error')}. No new trades are made while this lasts.")
    if health.get("last_cycle_at"):
        st.caption(f"Last trader check: {health['last_cycle_at']}")
    if o.get("equity") is None:
        st.info("No closed bar recorded for this run yet.")
        return
    cols = st.columns(6)
    cols[0].metric("Equity", f"{o['equity']:,.2f}", _pct(o["total_return"]))
    cols[1].metric("Daily PnL", f"{o['daily_pnl']:+,.2f}", _pct(o.get("daily_pnl_pct")))
    cols[2].metric("Drawdown", _pct(o["drawdown"]), f"max {_pct(o['max_drawdown'])}", delta_color="off")
    cols[3].metric("Exposure", _pct(o["exposure_pct"], signed=False), f"{o['open_positions']} position(s)",
                   delta_color="off")
    cols[4].metric("Realized / unrealized", f"{o['realized_pnl']:+,.2f}", f"{o['unrealized_pnl']:+,.2f} open",
                   delta_color="off")
    breakers = o["breakers"]
    cols[5].metric("Breakers", _breaker_label(breakers), f"cash {o['cash']:,.0f}", delta_color="off")
    if breakers.get("halted_reason"):
        st.error(f"Kill switch: {breakers['halted_reason']}. New entries are blocked; exits still work.")
    if breakers.get("daily_blocked_day"):
        st.warning(f"Daily loss limit hit on {breakers['daily_blocked_day']}: new entries paused for that UTC day.")
    if o.get("error"):
        st.error(f"Run error: {o['error']}")


def charts_section(data: DashboardData, run_id: str) -> None:
    curve = data.equity_curve(run_id)
    if curve.empty:
        return
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Equity curve")
        columns = [c for c in ("equity", "buy_and_hold") if c in curve.columns]
        st.line_chart(curve[columns].rename(columns={"equity": "strategy", "buy_and_hold": "buy & hold"}))
    with right:
        st.subheader("Drawdown")
        st.area_chart(curve[["drawdown"]])


def positions_section(data: DashboardData, run_id: str) -> None:
    st.subheader("Open positions")
    positions = data.open_positions(run_id)
    if positions:
        frame = pd.DataFrame(positions)[["symbol", "side", "entry_price", "current_price", "quantity", "value",
                                         "unrealized_pnl", "unrealized_pct", "stop_price", "opened_at"]]
        st.dataframe(frame, hide_index=True, width="stretch")
    else:
        st.caption("No open positions.")
    orders = data.working_orders(run_id)
    if orders["scheduled"] or orders["working_limits"]:
        with st.expander("Scheduled and working (simulated) orders"):
            st.json(orders)


def decision_section(data: DashboardData, run_id: str) -> None:
    latest = data.latest_decision(run_id)
    st.subheader("Latest decision")
    if not latest:
        st.caption("No signals yet.")
        return
    st.caption(f"Bar opened {latest['timestamp']} (decided at its close)")
    for symbol, d in latest["symbols"].items():
        ens = d["ensemble"] or {}
        actions = "; ".join(f"{a['action']}: {a['reason']}" for a in d["actions"]) or "-"
        with st.container(border=True):
            st.markdown(f"**{symbol}** → ensemble **{str(ens.get('direction', '-')).upper()}** "
                        f"(confidence {_num(ens.get('confidence'))}, net score {_num(ens.get('net_score'), 3)})")
            rows = []
            for v in d["votes"]:
                label = v.get("regime") or v.get("momentum_state") or v.get("risk_state") or ""
                rows.append({"voter": v["strategy"], "AI": "yes" if v["is_agent"] else "",
                             "vote": v["direction"].upper(), "confidence": v["confidence"],
                             "weight": v.get("weight"), "label": label,
                             "note": v.get("error") or v.get("note") or ""})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(f"Actions: {actions}")


def rationale_section(data: DashboardData, run_id: str) -> None:
    rationales = data.latest_rationales(run_id)
    st.subheader("AI rationales")
    if not rationales:
        st.caption("No AI agent has answered in this run.")
        return
    agents = sorted({r["strategy"] for r in rationales})
    cols = st.columns(len(agents))
    for col, agent in zip(cols, agents):
        with col:
            st.markdown(f"**{agent}**")
            for r in (x for x in rationales if x["strategy"] == agent):
                label = r.get("regime") or r.get("momentum_state") or r.get("risk_state") or ""
                with st.container(border=True):
                    st.markdown(f"{r['symbol']} · **{r['direction'].upper()}** {r['confidence']:.2f} "
                                f"{'· ' + label if label else ''}")
                    st.caption(f"{r['timestamp']} · cache {r.get('cache') or '-'}")
                    if r.get("error"):
                        st.warning(r["error"])
                    else:
                        st.write(r.get("rationale") or "")


def performance_section(data: DashboardData, run_id: str, horizon: int) -> None:
    st.subheader("Agent performance")
    rows = data.agent_performance(run_id, horizon)
    if not rows:
        st.caption("No votes yet.")
        return
    frame = pd.DataFrame([{
        "voter": r["strategy"], "AI": "yes" if r["is_agent"] else "", "votes": r["votes"],
        "BUY": r["buy"], "SELL": r["sell"], "HOLD": r["hold"], "errors": r["errors"],
        "avg confidence": r["avg_confidence"], f"correct ({horizon} bars)": r["directional_correctness"],
        "trades influenced": r["trades_influenced"], "pivotal": r["trades_pivotal"],
        "PnL agreed": r["pnl_agreed_pct"], "PnL disagreed": r["pnl_disagreed_pct"],
    } for r in rows]).sort_values(f"correct ({horizon} bars)", ascending=False, na_position="last")
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption("Correct = price moved the voted way over the horizon (raw prices). PnL agreed/disagreed = "
               "realised PnL of trades where the voter said BUY/SELL at the entry signal, % of initial cash.")


def research_section(data: DashboardData, run_id: str) -> None:
    st.subheader("Research")
    research = data.research(run_id)
    m, b = research["metrics"] or {}, research["benchmark"] or {}
    cols = st.columns(5)
    cols[0].metric("Strategy return", _pct(m.get("total_return")))
    cols[1].metric("Buy & hold return", _pct(b.get("total_return")))
    cols[2].metric("Max drawdown", _pct(-m["max_drawdown"] if m.get("max_drawdown") is not None else None))
    cols[3].metric("Sharpe", _num(m.get("sharpe_ratio")))
    cols[4].metric("Profit factor", _num(m.get("profit_factor")))
    rel = research.get("relative") or {}
    if rel:
        cols = st.columns(4)
        cols[0].metric("Excess vs buy & hold", _pct(rel.get("excess_return")))
        cols[1].metric("Alpha (per year)", _pct(rel.get("alpha_annualized")))
        cols[2].metric("Beta", _num(rel.get("beta")))
        cols[3].metric("Information ratio", _num(rel.get("information_ratio")))
    regimes = data.regimes(run_id)
    if regimes:
        frame = pd.DataFrame(regimes["by_trend"] + regimes["combined"])
        frame = frame[frame["bars"] > 0][["name", "bars", "time_share", "strategy_return", "market_return",
                                          "in_market", "trades", "pnl"]]
        st.markdown(f"Market regimes (trend: {regimes['trend_bars']}-bar average; returns compound each "
                    "regime's bars)")
        st.dataframe(frame, hide_index=True, width="stretch")
    results = research["research_results"]
    if results:
        rows = []
        for r in results:
            for row in r["payload"].get("rows", []):
                wf = row.get("walkforward") or {}
                metrics = row.get("metrics") or {}
                rows.append({"saved": r["created_at"][:16], "experiment": r["label"], "variant": row["variant"],
                             "return": wf.get("out_of_sample_return", metrics.get("total_return")),
                             "buy & hold": wf.get("benchmark_return", (row.get("benchmark") or {}).get("total_return")),
                             "mode": r["payload"].get("mode")})
        st.markdown("Saved experiments (walk-forward rows show out-of-sample returns)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def usage_section(data: DashboardData, run_id: str) -> None:
    usage = data.usage(run_id)
    if not usage["per_agent"]:
        return
    st.subheader("Model (Qwen) usage")
    t = usage["total"]
    cols = st.columns(5)
    cols[0].metric("Calls", t["calls"])
    cols[1].metric("Cache hits", t["cache_hits"])
    cols[2].metric("Failures", t["failures"], f"{t['retries']} retries", delta_color="off")
    cols[3].metric("Avg latency", "n/a" if t["avg_latency_seconds"] is None else f"{t['avg_latency_seconds']:.2f}s")
    est = " (est.)" if t["tokens_estimated"] else ""
    cols[4].metric("Tokens in / out", f"{t['input_tokens']:,} / {t['output_tokens']:,}{est}")
    st.dataframe(pd.DataFrame([{"agent": name, **{k: v for k, v in s.items() if k != "estimated_calls"}}
                               for name, s in usage["per_agent"].items()]),
                 hide_index=True, width="stretch")


def activity_section(data: DashboardData, run_id: str) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Recent trades")
        trades = data.recent_trades(run_id)
        if trades:
            st.dataframe(pd.DataFrame(trades), hide_index=True, width="stretch")
        else:
            st.caption("No closed trades yet.")
    with right:
        st.subheader("Recent signals (non-HOLD)")
        signals = data.recent_signals(run_id)
        if signals:
            st.dataframe(pd.DataFrame(signals), hide_index=True, width="stretch")
        else:
            st.caption("No BUY/SELL signals in the last bars.")


def render(db_path: str, run_id: str | None, horizon: int) -> None:
    with DashboardData(db_path) as data:
        runs = data.runs()
        if not runs:
            st.info("No runs stored yet. Start one with `trading-lab paper` or `trading-lab backtest`.")
            return
        run_id = run_id or data.default_run_id()
        assert run_id is not None
        portfolio_section(data, run_id)
        charts_section(data, run_id)
        positions_section(data, run_id)
        decision_section(data, run_id)
        rationale_section(data, run_id)
        performance_section(data, run_id, horizon)
        research_section(data, run_id)
        usage_section(data, run_id)
        activity_section(data, run_id)


def main() -> None:
    st.set_page_config(page_title="trading-lab", layout="wide")
    st.title("trading-lab")
    st.caption("Paper trading only: simulated fills, public market data. This dashboard is read-only.")
    db_path = os.environ.get(DB_ENV, DEFAULT_DB)
    if not Path(db_path).exists():
        st.error(f"No database at {db_path}. Set {DB_ENV} or start a run first.")
        return
    with DashboardData(db_path) as data:
        runs = data.runs()
        default = data.default_run_id()
    ids = [r["run_id"] for r in runs]
    with st.sidebar:
        st.header("View")
        run_id = st.selectbox(
            "Run", ids, index=ids.index(default) if default in ids else 0,
            format_func=lambda rid: next(f"{r['run_id']} · {r['kind']} · {r['status']}" for r in runs
                                         if r["run_id"] == rid),
        ) if ids else None
        horizon = st.slider("Outcome horizon (bars)", 1, 48, 4)
        refresh = st.selectbox("Auto-refresh", [0, 30, 60, 300], index=2,
                               format_func=lambda s: "off" if s == 0 else f"every {s}s")
        st.caption(f"Database: {Path(db_path).resolve()} (opened read-only)")

    if refresh:
        st.fragment(run_every=refresh)(render)(db_path, run_id, horizon)
    else:
        render(db_path, run_id, horizon)


main()
