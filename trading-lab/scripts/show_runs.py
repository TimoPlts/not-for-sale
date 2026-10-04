"""List stored runs, or show the details of one run.

Examples:
    python scripts/show_runs.py                       # recent runs with headline metrics
    python scripts/show_runs.py --run-id bt-1234abcd  # metrics, trades and recent decisions
"""

from __future__ import annotations

import argparse

from trading_lab.config import load_config
from trading_lab.storage import SQLiteStore


def fmt_pct(value) -> str:
    return "n/a" if value is None or isinstance(value, str) else f"{value:+.2%}"


def fmt_drawdown(value) -> str:
    return "n/a" if value is None else f"{-value:.1%}"


def fmt_num(value) -> str:
    return "n/a" if value is None or isinstance(value, str) else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    with SQLiteStore(cfg.storage.db_path) as store:
        if not args.run_id:
            runs = store.list_runs(args.limit)
            if not runs:
                print("No runs stored yet. Try: python scripts/run_backtest.py")
                return
            print(f"{'run id':<17} {'status':<10} {'tf':<4} {'period':<23} {'return':>9} "
                  f"{'max dd':>8} {'sharpe':>7} {'trades':>6}  symbols")
            for r in runs:
                m = r["metrics"] or {}
                period = f"{(r['period_start'] or '')[:10]} -> {(r['period_end'] or '')[:10]}"
                print(
                    f"{r['run_id']:<17} {r['status']:<10} {r['timeframe']:<4} {period:<23} "
                    f"{fmt_pct(m.get('total_return')):>9} "
                    f"{fmt_drawdown(m.get('max_drawdown')):>8} "
                    f"{fmt_num(m.get('sharpe_ratio')):>7} "
                    f"{m.get('num_trades', 0):>6}  {', '.join(r['symbols'])}"
                )
            return

        run = store.get_run(args.run_id)
        if run is None:
            print(f"Unknown run id {args.run_id}")
            return
        print(f"Run {run['run_id']} ({run['kind']}, {run['status']})  created {run['created_at'][:19]}")
        print(f"Period {run['period_start']} -> {run['period_end']}  timeframe {run['timeframe']}")
        print(f"Symbols {', '.join(run['symbols'])}  data {run['exchange']}")
        print(f"Config fingerprint {run['config_fingerprint'][:16]}...  notes: {run['notes'] or '-'}")
        if run["error"]:
            print(f"Error: {run['error']}")

        metrics = store.load_metrics(args.run_id) or {}
        print("\n=== Metrics ===")
        for key, value in metrics.items():
            print(f"  {key:<22} {value}")

        trades = store.load_closed_trades(args.run_id)
        print(f"\n=== Closed trades ({len(trades)}) - last 15 ===")
        for t in trades[-15:]:
            print(f"  {t.closed_at:%Y-%m-%d %H:%M}  {t.symbol:<10} qty={t.quantity:<12.6g} "
                  f"entry={t.entry_price:<12.6g} exit={t.exit_price:<12.6g} pnl={t.pnl:+9.2f} "
                  f"({t.return_pct:+.2%})")

        decisions = store.load_decisions(args.run_id, include_holds=False)
        print(f"\n=== Non-HOLD decisions ({len(decisions)}) - last 15 ===")
        for _, d in decisions.tail(15).iterrows():
            print(f"  {d['timestamp']:%Y-%m-%d %H:%M}  {d['symbol']:<10} {d['action']:<13} {d['reason']}")


if __name__ == "__main__":
    main()
