"""Run a backtest on historical public market data and store it in SQLite.

Examples:
    python scripts/run_backtest.py                                   # last 90 days, config defaults
    python scripts/run_backtest.py --start 2025-01-01 --end 2025-07-01
    python scripts/run_backtest.py --symbols BTC/USDT ETH/USDT --timeframe 4h
    python scripts/run_backtest.py --synthetic 7                     # offline, random-walk data
    python scripts/run_backtest.py --export results/                 # also write CSV files

Simulation only: data comes from public endpoints; no orders are ever sent.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from trading_lab.backtest import BacktestEngine
from trading_lab.config import load_config
from trading_lab.data import SyntheticProvider, build_provider
from trading_lab.storage import SQLiteStore


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--start", type=parse_date, help="YYYY-MM-DD (UTC); default 90 days ago")
    parser.add_argument("--end", type=parse_date, help="YYYY-MM-DD (UTC, exclusive); default now")
    parser.add_argument("--symbols", nargs="+", help="e.g. BTC/USDT ETH/USDT")
    parser.add_argument("--timeframe", help="e.g. 15m, 1h, 4h, 1d")
    parser.add_argument("--synthetic", type=int, metavar="SEED", help="use offline synthetic data")
    parser.add_argument("--liquidate", action="store_true", help="close positions at the end")
    parser.add_argument("--no-db", action="store_true", help="do not save to SQLite")
    parser.add_argument("--export", metavar="DIR", help="write equity/trades/fills CSV files")
    parser.add_argument("--notes", default="", help="free-text note stored with the run")
    args = parser.parse_args()

    overrides: dict[str, dict] = {}
    if args.symbols:
        overrides.setdefault("market", {})["symbols"] = args.symbols
    if args.timeframe:
        overrides.setdefault("market", {})["timeframe"] = args.timeframe
    if args.liquidate:
        overrides["backtest"] = {"liquidate_at_end": True}
    cfg = load_config(args.config).with_overrides(overrides)

    end = args.end or datetime.now(timezone.utc)
    start = args.start or end - timedelta(days=90)
    provider = SyntheticProvider(seed=args.synthetic) if args.synthetic is not None else build_provider(cfg)

    print(f"Backtest {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC | "
          f"{cfg.market.timeframe} | {', '.join(cfg.market.symbols)} | data: {provider.name}")
    print("Loading candles and simulating...")

    store = None if args.no_db else SQLiteStore(cfg.storage.db_path)
    try:
        result = BacktestEngine(cfg, provider, store=store).run(start, end, notes=args.notes)
    finally:
        if store is not None:
            store.close()

    print("\n=== Performance ===")
    print(result.metrics.format_table())

    per_symbol: dict[str, list[float]] = defaultdict(list)
    for t in result.trades:
        per_symbol[t.symbol].append(t.pnl)
    print("\n=== Closed trades per symbol ===")
    for symbol in cfg.market.symbols:
        pnls = per_symbol.get(symbol, [])
        wins = sum(p > 0 for p in pnls)
        print(f"{symbol:<10} trades={len(pnls):<5} wins={wins:<5} pnl={sum(pnls):+,.2f} USDT")
    open_positions = int(result.equity_curve["open_positions"].iloc[-1])
    if open_positions:
        print(f"({open_positions} position(s) still open at the end; counted in equity, not in trades)")

    print("\n=== Decisions ===")
    print(", ".join(f"{k}={v}" for k, v in sorted(result.actions().items())))

    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(out / "equity_curve.csv")
        pd.DataFrame([{
            "symbol": t.symbol, "quantity": t.quantity, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "pnl": t.pnl, "return_pct": t.return_pct,
            "opened_at": t.opened_at, "closed_at": t.closed_at,
        } for t in result.trades]).to_csv(out / "trades.csv", index=False)
        pd.DataFrame([{
            "timestamp": f.timestamp, "symbol": f.symbol, "side": f.side.value,
            "quantity": f.quantity, "reference_price": f.reference_price,
            "fill_price": f.fill_price, "fee": f.fee,
        } for f in result.fills]).to_csv(out / "fills.csv", index=False)
        print(f"\nCSV files written to {out.resolve()}")

    if store is not None:
        print(f"\nSaved as run {result.run_id} in {Path(cfg.storage.db_path).resolve()}")
        print(f"Inspect it with: python scripts/show_runs.py --run-id {result.run_id}")


if __name__ == "__main__":
    main()
