"""``trading-lab`` command-line interface.

    trading-lab backtest [--start DATE] [--end DATE] [--symbols ...] [--timeframe TF] [--synthetic SEED]
    trading-lab paper    [--symbols ...] [--timeframe TF] [--resume RUN_ID] [--once] [--synthetic SEED]
    trading-lab report   [RUN_ID] [--limit N]
    trading-lab signals  [--symbols ...] [--timeframe TF] [--days N]
    trading-lab sweep    --param strategies.rsi.period=7,14,21 [--param ...] [--start/--end] [--metric M]
    trading-lab walkforward --param ... [--train-days 90] [--test-days 30]
    trading-lab compare  RUN_ID RUN_ID ...

Global options (before the command): ``--config PATH`` and ``--db PATH``.

Everything is simulated: market data comes from public endpoints and no
order is ever sent to an exchange.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from trading_lab.config import AppConfig, load_config
from trading_lab.core.errors import TradingLabError
from trading_lab.core.models import DecisionAction
from trading_lab.data import MarketDataProvider, SyntheticProvider, build_provider

DEFAULT_CONFIG = "config/default.toml"


# ----------------------------------------------------------------- helpers
def _date(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; use YYYY-MM-DD") from None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _load_config(args: argparse.Namespace) -> AppConfig:
    path = args.config
    if path == DEFAULT_CONFIG and not Path(path).exists():
        cfg = AppConfig()  # running outside the project folder: built-in defaults
    else:
        cfg = load_config(path)
    overrides: dict[str, dict[str, Any]] = {}
    if getattr(args, "symbols", None):
        overrides.setdefault("market", {})["symbols"] = args.symbols
    if getattr(args, "timeframe", None):
        overrides.setdefault("market", {})["timeframe"] = args.timeframe
    if getattr(args, "liquidate", False):
        overrides["backtest"] = {"liquidate_at_end": True}
    if args.db:
        overrides["storage"] = {"db_path": args.db}
    return cfg.with_overrides(overrides) if overrides else cfg


def _provider(cfg: AppConfig, seed: int | None) -> MarketDataProvider:
    return SyntheticProvider(seed=seed) if seed is not None else build_provider(cfg)


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None or isinstance(value, str) else f"{value:+.2%}"


def _fmt_num(value: Any) -> str:
    if value is None:
        return "n/a"
    return value if isinstance(value, str) else f"{value:.2f}"


def _fmt_dd(value: Any) -> str:
    return "n/a" if value is None else f"{-value:.1%}"


# ---------------------------------------------------------------- backtest
def cmd_backtest(args: argparse.Namespace) -> int:
    from trading_lab.backtest import BacktestEngine
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    end = args.end or datetime.now(timezone.utc)
    start = args.start or end - timedelta(days=args.days)
    provider = _provider(cfg, args.synthetic)
    print(f"Backtest {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC | {cfg.market.timeframe} | "
          f"{', '.join(cfg.market.symbols)} | data: {provider.name}")
    print("Loading candles and simulating...")

    store = None if args.no_db else SQLiteStore(cfg.storage.db_path)
    try:
        result = BacktestEngine(cfg, provider, store=store).run(start, end, notes=args.notes)
    finally:
        if store is not None:
            store.close()

    print("\n=== Performance ===")
    print(result.metrics.format_table())
    if result.benchmark is not None:
        print(f"\nBuy & hold (equal weight, same costs): return {result.benchmark.total_return:+.2%}, "
              f"max drawdown {-result.benchmark.max_drawdown:.2%}, "
              f"sharpe {_fmt_num(result.benchmark.sharpe_ratio)}")
    per_symbol: dict[str, list[float]] = defaultdict(list)
    for t in result.trades:
        per_symbol[t.symbol].append(t.pnl)
    print("\n=== Closed trades per symbol ===")
    for symbol in cfg.market.symbols:
        pnls = per_symbol.get(symbol, [])
        print(f"{symbol:<10} trades={len(pnls):<5} wins={sum(p > 0 for p in pnls):<5} "
              f"pnl={sum(pnls):+,.2f} USDT")
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
        print(f"Details: trading-lab report {result.run_id}")
    return 0


# ------------------------------------------------------------------- paper
def cmd_paper(args: argparse.Namespace) -> int:
    from trading_lab.live import CycleReport, LivePaperTrader
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    store = SQLiteStore(cfg.storage.db_path)
    try:
        if args.resume:
            if args.symbols or args.timeframe:
                raise TradingLabError("--symbols/--timeframe cannot be changed when resuming a run")
            run = store.get_run(args.resume)
            if run is None:
                raise TradingLabError(f"unknown run id {args.resume}")
            if run["kind"] != "paper":
                raise TradingLabError(f"{args.resume} is a {run['kind']} run; only paper runs can be resumed")
            seed = args.synthetic
            if seed is None and run["exchange"].startswith("synthetic-"):
                seed = int(run["exchange"].split("-", 1)[1])
            stored_cfg = AppConfig.from_dict(run["config"])
            trader = LivePaperTrader.resume(store, args.resume, _provider(stored_cfg, seed))
            print(f"Resuming paper run {trader.run_id}")
        else:
            trader = LivePaperTrader(cfg, _provider(cfg, args.synthetic), store, notes=args.notes)
            print(f"Started paper run {trader.run_id}")
        tcfg = trader.config
        print(f"{tcfg.market.timeframe} candles | {', '.join(tcfg.market.symbols)} | "
              f"starting cash {tcfg.portfolio.initial_cash:,.2f} USDT | simulated fills only")
        if not args.once:
            print("Acts once per closed candle. Press Ctrl+C to stop (the run can be resumed).\n")

        def on_cycle(report: CycleReport) -> None:
            stamp = f"[{report.checked_at:%Y-%m-%d %H:%M:%S} UTC]"
            if report.error:
                print(f"{stamp} data error, will retry: {report.error}")
                return
            if report.warning:
                print(f"{stamp} warning: {report.warning}")
            for d in report.decisions:
                if d.action in (DecisionAction.ENTER, DecisionAction.EXIT, DecisionAction.STOP_LOSS,
                                DecisionAction.REJECTED, DecisionAction.ENTER_SIGNAL,
                                DecisionAction.EXIT_SIGNAL):
                    detail = ""
                    if d.quantity is not None and d.reference_price is not None:
                        detail = f" qty={d.quantity:.8g} @ ~{d.reference_price:,.6g}"
                    if d.stop_price is not None:
                        detail += f" stop={d.stop_price:,.6g}"
                    print(f"{stamp} {d.timestamp:%m-%d %H:%M} {d.symbol:<10} "
                          f"{d.action.value.upper():<12}{detail}  ({d.reason})")
            p = trader.portfolio
            held = ", ".join(f"{s} {pos.quantity:.6g}" for s, pos in p.positions.items()) or "none"
            equity = "n/a" if report.equity is None else f"{report.equity:,.2f}"
            last = "-" if report.last_bar is None else f"{report.last_bar:%Y-%m-%d %H:%M}"
            print(f"{stamp} new candles={report.new_bars} last={last} | equity {equity} | "
                  f"cash {p.cash:,.2f} | positions: {held}")

        def on_wait(seconds: float) -> None:
            wake = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            print(f"   next check at {wake:%H:%M:%S} UTC")

        trader.run_forever(
            poll_seconds=args.poll,
            max_cycles=1 if args.once else args.max_cycles,
            on_cycle=on_cycle,
            on_wait=on_wait,
        )
        print(f"\nPaper run {trader.run_id} stopped. Resume with: trading-lab paper --resume {trader.run_id}")
        print(f"Report:  trading-lab report {trader.run_id}")
    finally:
        store.close()
    return 0


# ------------------------------------------------------------------ report
def cmd_report(args: argparse.Namespace) -> int:
    from trading_lab.reporting import run_metrics
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    with SQLiteStore(cfg.storage.db_path) as store:
        if not args.run_id:
            runs = store.list_runs(args.limit)
            if not runs:
                print("No runs stored yet. Try: trading-lab backtest")
                return 0
            print(f"{'run id':<16} {'kind':<8} {'status':<9} {'tf':<4} {'period':<23} {'return':>9} "
                  f"{'max dd':>8} {'sharpe':>7} {'trades':>6}  symbols")
            for r in runs:
                m = r["metrics"]
                if m is None and r["kind"] == "paper":
                    live = run_metrics(store, r["run_id"])
                    m = live.to_dict() if live else None
                m = m or {}
                period = f"{(r['period_start'] or '')[:10]} -> {(r['period_end'] or 'now')[:10]}"
                print(f"{r['run_id']:<16} {r['kind']:<8} {r['status']:<9} {r['timeframe']:<4} "
                      f"{period:<23} {_fmt_pct(m.get('total_return')):>9} "
                      f"{_fmt_dd(m.get('max_drawdown')):>8} {_fmt_num(m.get('sharpe_ratio')):>7} "
                      f"{m.get('num_trades', 0):>6}  {', '.join(r['symbols'])}")
            return 0

        run = store.get_run(args.run_id)
        if run is None:
            raise TradingLabError(f"unknown run id {args.run_id}")
        print(f"Run {run['run_id']} ({run['kind']}, {run['status']})  created {run['created_at'][:19]}")
        print(f"Period {run['period_start']} -> {run['period_end'] or 'now'}  timeframe {run['timeframe']}")
        print(f"Symbols {', '.join(run['symbols'])}  data {run['exchange']}")
        print(f"Config fingerprint {run['config_fingerprint'][:16]}...  notes: {run['notes'] or '-'}")
        if run["error"]:
            print(f"Error: {run['error']}")

        metrics = run_metrics(store, args.run_id)
        print("\n=== Performance ===")
        print(metrics.format_table() if metrics else "no equity data yet")
        bench = (store.load_metrics(args.run_id) or {}).get("benchmark")
        if bench:
            print(f"\nBuy & hold (equal weight, same costs): return {_fmt_pct(bench.get('total_return'))}, "
                  f"max drawdown {_fmt_dd(bench.get('max_drawdown'))}")

        trades = store.load_closed_trades(args.run_id)
        print(f"\n=== Closed trades ({len(trades)}) - last {args.limit} ===")
        for t in trades[-args.limit:]:
            print(f"  {t.closed_at:%Y-%m-%d %H:%M}  {t.symbol:<10} qty={t.quantity:<12.6g} "
                  f"entry={t.entry_price:<12.6g} exit={t.exit_price:<12.6g} pnl={t.pnl:+9.2f} "
                  f"({t.return_pct:+.2%})")
        decisions = store.load_decisions(args.run_id, include_holds=False)
        print(f"\n=== Non-HOLD decisions ({len(decisions)}) - last {args.limit} ===")
        for _, d in decisions.tail(args.limit).iterrows():
            print(f"  {d['timestamp']:%Y-%m-%d %H:%M}  {d['symbol']:<10} {d['action']:<13} {d['reason']}")
    return 0


# ----------------------------------------------------------------- signals
def cmd_signals(args: argparse.Namespace) -> int:
    from trading_lab.ensemble import VotingEngine
    from trading_lab.strategy_factory import strategies_for

    cfg = _load_config(args)
    provider = _provider(cfg, args.synthetic)
    strategies = strategies_for(cfg)
    voting = VotingEngine.from_specs(cfg.strategies, cfg.voting)
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    keys = {"rsi": ["rsi"], "macd": ["hist", "crossover"], "bollinger": ["percent_b"]}
    default_keys = ["rationale"]  # agent strategies explain themselves

    print(f"data={provider.name}  timeframe={cfg.market.timeframe}  history={args.days}d\n")
    for symbol in cfg.market.symbols:
        candles = provider.fetch_ohlcv(symbol, cfg.market.timeframe, since)
        if candles.empty:
            print(f"{symbol}: no candles")
            continue
        print(f"{symbol}  last closed candle {candles.index[-1]:%Y-%m-%d %H:%M} UTC  "
              f"close={candles['close'].iloc[-1]:.6g}  ({len(candles)} candles)")
        signals = [s.generate_signal(symbol, candles) for s in strategies]
        for sig in signals:
            shown = "  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in ((k, sig.metadata.get(k)) for k in keys.get(sig.strategy, default_keys))
            )
            print(f"   {sig.strategy:<10} {sig.direction.value.upper():<5} conf={sig.confidence:.2f}   {shown}")
        ens = voting.combine(signals)
        print(f"   {'ENSEMBLE':<10} {ens.direction.value.upper():<5} conf={ens.confidence:.2f}   "
              f"net_score={ens.metadata['net_score']:+.3f}\n")
    return 0


# --------------------------------------------------------------- research
def _coerce(token: str) -> Any:
    lowered = token.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(token)
        except ValueError:
            pass
    return token.strip()


def _grid(args: argparse.Namespace) -> dict[str, list[Any]]:
    import tomllib

    grid: dict[str, list[Any]] = {}
    if args.grid:
        with open(args.grid, "rb") as fh:
            data = tomllib.load(fh)
        table = data.get("grid", data)
        for key, values in table.items():
            grid[key] = list(values) if isinstance(values, list) else [values]
    for item in args.param or []:
        key, sep, values = item.partition("=")
        if not sep or not key.strip() or not values.strip():
            raise TradingLabError(f"--param must look like section.key=v1,v2 (got {item!r})")
        grid[key.strip()] = [_coerce(v) for v in values.split(",")]
    if not grid:
        raise TradingLabError("give at least one --param (or --grid FILE)")
    return grid


def _period(args: argparse.Namespace, default_days: int) -> tuple[datetime, datetime]:
    end = args.end or datetime.now(timezone.utc)
    start = args.start or end - timedelta(days=default_days)
    if end <= start:
        raise TradingLabError("--end must be after --start")
    return start, end


def _short(params: dict[str, Any]) -> str:
    return " ".join(f"{k.split('.', 1)[-1]}={v}" for k, v in params.items())


def cmd_sweep(args: argparse.Namespace) -> int:
    from trading_lab.research import run_sweep
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    grid = _grid(args)
    start, end = _period(args, args.days)
    provider = _provider(cfg, args.synthetic)
    combos = 1
    for values in grid.values():
        combos *= len(values)
    print(f"Sweep: {combos} backtests | {start:%Y-%m-%d} -> {end:%Y-%m-%d} | {cfg.market.timeframe} | "
          f"ranked by {args.metric} | data: {provider.name}")

    store = SQLiteStore(cfg.storage.db_path) if args.save else None
    try:
        results = run_sweep(
            cfg, provider, start, end, grid, metric=args.metric, store=store,
            progress=lambda n, total, params: print(f"  [{n}/{total}] {_short(params)}"),
        )
    finally:
        if store is not None:
            store.close()

    bench = results[0].benchmark
    print(f"\n{'#':>3}  {'return':>8} {'max dd':>7} {'sharpe':>7} {'trades':>6} {'win':>6}  params")
    for rank, r in enumerate(results, 1):
        m = r.metrics
        win = "n/a" if m.win_rate is None else f"{m.win_rate:.0%}"
        print(f"{rank:>3}  {m.total_return:>+8.2%} {-m.max_drawdown:>7.1%} {_fmt_num(m.sharpe_ratio):>7} "
              f"{m.num_trades:>6} {win:>6}  {_short(r.params)}")
    if bench is not None:
        print(f"\nBuy & hold over the same period: {bench.total_return:+.2%} "
              f"(max drawdown {-bench.max_drawdown:.1%})")
    print("\nNote: the best in-sample row is optimistic by construction; use walkforward to check it.")
    if args.export:
        rows = [{**r.params, **r.metrics.to_dict(), "run_id": r.run_id} for r in results]
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.export, index=False)
        print(f"Results written to {Path(args.export).resolve()}")
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    from trading_lab.research import walk_forward

    cfg = _load_config(args)
    grid = _grid(args)
    start, end = _period(args, args.days)
    provider = _provider(cfg, args.synthetic)
    print(f"Walk-forward | {start:%Y-%m-%d} -> {end:%Y-%m-%d} | train {args.train_days}d, "
          f"test {args.test_days}d | choose by {args.metric} | data: {provider.name}")
    result = walk_forward(
        cfg, provider, start, end, grid,
        train=timedelta(days=args.train_days), test=timedelta(days=args.test_days),
        metric=args.metric, progress=lambda msg: print(f"  {msg}"),
    )
    metric = args.metric
    print(f"\n{'test window':<25} {'in-sample':>10} {'out-of-sample':>14} "
          f"{'OOS return':>11} {'buy&hold':>9}  chosen params")
    for f in result.folds:
        is_v = getattr(f.in_sample, metric)
        oos_v = getattr(f.out_of_sample, metric)
        bh = "n/a" if f.benchmark is None else f"{f.benchmark.total_return:+.2%}"
        print(f"{f.test_start:%Y-%m-%d} -> {f.test_end:%Y-%m-%d}  {_fmt_num(is_v):>10} {_fmt_num(oos_v):>14} "
              f"{f.out_of_sample.total_return:>+11.2%} {bh:>9}  {_short(f.best_params)}")
    is_mean, oos_mean = result.mean_metric("in_sample"), result.mean_metric("out_of_sample")
    print(f"\nMean {metric}: in-sample {_fmt_num(is_mean)} vs out-of-sample {_fmt_num(oos_mean)}")
    bench = result.benchmark_return
    print(f"Combined out-of-sample return {result.out_of_sample_return:+.2%}"
          + ("" if bench is None else f" vs buy & hold {bench:+.2%}"))
    if is_mean is not None and oos_mean is not None and oos_mean < is_mean:
        print("Out-of-sample is worse than in-sample: expect live results closer to the out-of-sample numbers.")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from trading_lab.reporting import run_metrics
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    rows = [
        ("total_return", "Total return", _fmt_pct), ("annualized_return", "Annualized", _fmt_pct),
        ("max_drawdown", "Max drawdown", _fmt_dd), ("sharpe_ratio", "Sharpe", _fmt_num),
        ("sortino_ratio", "Sortino", _fmt_num), ("num_trades", "Trades", str),
        ("win_rate", "Win rate", lambda v: "n/a" if v is None else f"{v:.1%}"),
        ("profit_factor", "Profit factor", _fmt_num), ("total_fees", "Fees", _fmt_num),
        ("exposure", "Exposure", lambda v: "n/a" if v is None else f"{v:.1%}"),
    ]
    columns = []
    with SQLiteStore(cfg.storage.db_path) as store:
        for run_id in args.run_ids:
            run = store.get_run(run_id)
            if run is None:
                raise TradingLabError(f"unknown run id {run_id}")
            stored = store.load_metrics(run_id) or {}
            live = run_metrics(store, run_id)
            metrics = live.to_dict() if live else stored
            bench = stored.get("benchmark", {})
            columns.append((run_id, run, metrics, bench))
    width = max(16, *(len(c[0]) for c in columns))
    print(f"{'':<16}" + "".join(f"{c[0]:>{width + 2}}" for c in columns))
    info = [("Kind", lambda r: r["kind"]), ("Timeframe", lambda r: r["timeframe"]),
            ("Period start", lambda r: (r["period_start"] or "")[:10]),
            ("Period end", lambda r: (r["period_end"] or "now")[:10])]
    for label, getter in info:
        print(f"{label:<16}" + "".join(f"{getter(c[1]):>{width + 2}}" for c in columns))
    for key, label, fmt in rows:
        print(f"{label:<16}" + "".join(f"{fmt(c[2].get(key)):>{width + 2}}" for c in columns))
    print(f"{'Buy & hold':<16}" + "".join(f"{_fmt_pct(c[3].get('total_return')):>{width + 2}}" for c in columns))
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-lab",
        description="Crypto paper-trading research platform (simulation only; never trades for real).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"TOML config (default {DEFAULT_CONFIG})")
    parser.add_argument("--db", help="SQLite file (overrides [storage] db_path)")
    sub = parser.add_subparsers(dest="command", required=True)

    def market_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbols", nargs="+", help="e.g. BTC/USDT ETH/USDT")
        p.add_argument("--timeframe", help="e.g. 15m, 1h, 4h, 1d")
        p.add_argument("--synthetic", type=int, metavar="SEED", help="offline synthetic data")

    bt = sub.add_parser("backtest", help="simulate a strategy set over historical data")
    market_options(bt)
    bt.add_argument("--start", type=_date, help="YYYY-MM-DD (UTC)")
    bt.add_argument("--end", type=_date, help="YYYY-MM-DD (UTC, exclusive); default now")
    bt.add_argument("--days", type=int, default=90, help="length when --start is omitted (default 90)")
    bt.add_argument("--liquidate", action="store_true", help="close positions at the end")
    bt.add_argument("--no-db", action="store_true", help="do not save the run")
    bt.add_argument("--export", metavar="DIR", help="write equity/trades/fills CSV files")
    bt.add_argument("--notes", default="", help="note stored with the run")
    bt.set_defaults(func=cmd_backtest)

    pp = sub.add_parser("paper", help="live paper trading on public real-time data")
    market_options(pp)
    pp.add_argument("--resume", metavar="RUN_ID", help="continue a stopped paper run")
    pp.add_argument("--once", action="store_true", help="run one cycle and exit")
    pp.add_argument("--max-cycles", type=int, help="stop after N cycles")
    pp.add_argument("--poll", type=float, default=30.0, help="seconds between retries (default 30)")
    pp.add_argument("--notes", default="", help="note stored with the run")
    pp.set_defaults(func=cmd_paper)

    rp = sub.add_parser("report", help="list runs, or show one run in detail")
    rp.add_argument("run_id", nargs="?")
    rp.add_argument("--limit", type=int, default=20)
    rp.set_defaults(func=cmd_report)

    sg = sub.add_parser("signals", help="current strategy signals from live public data")
    market_options(sg)
    sg.add_argument("--days", type=int, default=30, help="history to load (default 30)")
    sg.set_defaults(func=cmd_signals)

    def research_options(p: argparse.ArgumentParser, default_days: int) -> None:
        market_options(p)
        p.add_argument("--param", action="append", metavar="KEY=V1,V2",
                       help="dotted config key and values, e.g. strategies.rsi.period=7,14,21 (repeatable)")
        p.add_argument("--grid", metavar="FILE", help="TOML file with a [grid] table of key = [values]")
        p.add_argument("--start", type=_date, help="YYYY-MM-DD (UTC)")
        p.add_argument("--end", type=_date, help="YYYY-MM-DD (UTC, exclusive); default now")
        p.add_argument("--days", type=int, default=default_days,
                       help=f"length when --start is omitted (default {default_days})")
        p.add_argument("--metric", default="sharpe_ratio", help="ranking metric (default sharpe_ratio)")

    sw = sub.add_parser("sweep", help="backtest every combination of parameter values")
    research_options(sw, 180)
    sw.add_argument("--save", action="store_true", help="store every run in the database")
    sw.add_argument("--export", metavar="CSV", help="write the results table to a CSV file")
    sw.set_defaults(func=cmd_sweep)

    wf = sub.add_parser("walkforward", help="choose parameters in-sample, measure them out-of-sample")
    research_options(wf, 365)
    wf.add_argument("--train-days", type=int, default=90)
    wf.add_argument("--test-days", type=int, default=30)
    wf.set_defaults(func=cmd_walkforward)

    cp = sub.add_parser("compare", help="compare stored runs side by side")
    cp.add_argument("run_ids", nargs="+")
    cp.set_defaults(func=cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except TradingLabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
