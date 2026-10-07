"""``trading-lab`` command-line interface.

    trading-lab demo     [DIR]   (offline sample: a backtest, a paper run and HTML reports)
    trading-lab backtest [--start DATE] [--end DATE] [--symbols ...] [--timeframe TF] [--synthetic SEED]
    trading-lab paper    [--symbols ...] [--timeframe TF] [--resume RUN_ID] [--once] [--synthetic SEED]
    trading-lab report   [RUN_ID] [--limit N] [--html FILE]
    trading-lab signals  [--symbols ...] [--timeframe TF] [--days N]
    trading-lab sweep    --param strategies.rsi.period=7,14,21 [--param ...] [--start/--end] [--metric M]
    trading-lab walkforward --param ... [--train-days 90] [--test-days 30]
    trading-lab compare  RUN_ID RUN_ID ...
    trading-lab agent-report [RUN_ID] [--horizon N] [--all]
    trading-lab experiment [--variants baseline,trend,...] [--walkforward] [--param ...]
    trading-lab agent-test [qwen | qwen_trend | qwen_momentum | qwen_risk] [--synthetic SEED]
    trading-lab dashboard-data [RUN_ID] [--json]
    trading-lab dashboard [--host 127.0.0.1] [--port 8501]
    trading-lab summary [RUN_ID] [--hours 24]
    trading-lab alert-test [--channel webhook|email] [--format ntfy|slack|discord|json]
    trading-lab agent-weights [RUN_ID]
    trading-lab doctor [--online]
    trading-lab robustness [RUN_ID] [--samples 5000] [--seed 7]
    trading-lab reconcile PAPER_RUN_ID [--synthetic SEED]
    trading-lab data-check [--symbols ...] [--days N | --start/--end] [--run RUN_ID] [--strict]
    trading-lab agent-eval [RUN_ID] [--min-answers 20] [--json]
    trading-lab export RUN_ID DIR [--holds] [--force]
    trading-lab costs [--days N | --start/--end] [--multipliers 0,0.5,1,2,3]
    trading-lab regimes [RUN_ID] [--trend-bars 50] [--vol-bars 24] [--json]

Global options (before the command): ``--config PATH``, ``--db PATH`` and
``--agent-mode record|replay|live``.

Everything is simulated: market data comes from public endpoints and no
order is ever sent to an exchange.
"""

from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import threading
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
RUN_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
log = logging.getLogger("trading_lab.cli")


def _configure_logging(level: str, log_file: str | None) -> None:
    """Warnings (e.g. model retries) go to stderr; with --log-file, everything at ``level``
    and above also goes to a size-rotated file (10 MB x 5)."""
    import logging.handlers

    root = logging.getLogger("trading_lab")
    for handler in [h for h in root.handlers if getattr(h, "_trading_lab_cli", False)]:
        root.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(fmt)
    handlers: list[logging.Handler] = [console]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(fmt)
        handlers.append(file_handler)
    for handler in handlers:
        handler._trading_lab_cli = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(min(logging.WARNING, getattr(logging, level)))


def _say(message: str = "") -> None:
    """Print to stdout and record it in the log file (if one is configured)."""
    print(message)
    if message.strip():
        log.info(message.strip())


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
    if getattr(args, "agent_mode", None):
        overrides["agents"] = {"mode": args.agent_mode}
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


def _usage_title(provider_name: str | None, model: str | None = None) -> str:
    name = (provider_name or "model").capitalize()
    return f"{name} usage" + (f" ({model})" if model else "")


def _print_signal_usage(signals: Any) -> None:
    """Model usage per agent from signals (in memory or from a stored run)."""
    from trading_lab.llm import usage_from_signals

    pairs = [(s.strategy, s.metadata) for s in signals]
    per_agent = usage_from_signals(pairs)
    if not per_agent:
        return
    providers = {m["llm"].get("provider") for _, m in pairs if isinstance(m.get("llm"), dict)}
    from trading_lab.llm import format_usage

    print()
    print(format_usage(per_agent, _usage_title(next(iter(providers)) if len(providers) == 1 else None)))


def _print_provider_usage(llm: Any) -> None:
    if llm is None or not llm.usage.per_agent:
        return
    from trading_lab.llm import format_usage

    print()
    print(format_usage(llm.usage.per_agent, _usage_title(llm.name, llm.model)))


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
        if result.relative is not None:
            print(result.relative.format_line())
    per_symbol: dict[str, list[float]] = defaultdict(list)
    shorts: dict[str, int] = defaultdict(int)
    for t in result.trades:
        per_symbol[t.symbol].append(t.pnl)
        shorts[t.symbol] += t.side == "short"
    print("\n=== Closed trades per symbol ===")
    for symbol in cfg.market.symbols:
        pnls = per_symbol.get(symbol, [])
        print(f"{symbol:<10} trades={len(pnls):<5} wins={sum(p > 0 for p in pnls):<5} "
              f"pnl={sum(pnls):+,.2f} USDT" + (f"  (shorts={shorts[symbol]})" if cfg.risk.allow_short else ""))
    open_positions = int(result.equity_curve["open_positions"].iloc[-1])
    if open_positions:
        print(f"({open_positions} position(s) still open at the end; counted in equity, not in trades)")
    print("\n=== Decisions ===")
    print(", ".join(f"{k}={v}" for k, v in sorted(result.actions().items())))
    _print_signal_usage(result.signals)

    if args.export:
        from trading_lab.export import fills_frame, trades_frame

        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(out / "equity_curve.csv")
        trades_frame(result.trades).to_csv(out / "trades.csv", index=False)
        fills_frame(result.fills).to_csv(out / "fills.csv", index=False)
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
    if args.run_id is not None and not re.fullmatch(RUN_ID_PATTERN, args.run_id):
        raise TradingLabError("--run-id may only contain letters, digits, '.', '_' and '-' (max 64)")
    if args.run_id and args.resume:
        raise TradingLabError("use either --run-id or --resume, not both")
    from trading_lab.alerts import build_alerts

    alerts = build_alerts(cfg)  # from the current config: alert settings are operational, not part of a run
    store = SQLiteStore(cfg.storage.db_path)
    try:
        if args.run_id and store.get_run(args.run_id) is not None:
            args.resume = args.run_id  # restart of a named run: continue it
        if args.resume:
            if args.symbols or args.timeframe:
                raise TradingLabError("--symbols/--timeframe cannot be changed when resuming a run "
                                      "(the run keeps the config it was started with)")
            run = store.get_run(args.resume)
            if run is None:
                raise TradingLabError(f"unknown run id {args.resume}")
            if run["kind"] != "paper":
                raise TradingLabError(f"{args.resume} is a {run['kind']} run; only paper runs can be resumed")
            seed = args.synthetic
            if seed is None and run["exchange"].startswith("synthetic-"):
                seed = int(run["exchange"].split("-", 1)[1])
            stored_cfg = AppConfig.from_dict(run["config"])
            trader = LivePaperTrader.resume(store, args.resume, _provider(stored_cfg, seed), alerts=alerts)
            _say(f"Resuming paper run {trader.run_id}")
        else:
            trader = LivePaperTrader(cfg, _provider(cfg, args.synthetic), store, notes=args.notes,
                                     run_id=args.run_id, alerts=alerts)
            _say(f"Started paper run {trader.run_id}")
        tcfg = trader.config
        print(f"{tcfg.market.timeframe} candles | {', '.join(tcfg.market.symbols)} | "
              f"starting cash {tcfg.portfolio.initial_cash:,.2f} USDT | simulated fills only")
        if not args.once:
            print("Acts once per closed candle. Press Ctrl+C to stop (the run can be resumed).\n")

        def on_cycle(report: CycleReport) -> None:
            stamp = f"[{report.checked_at:%Y-%m-%d %H:%M:%S} UTC]"
            if report.error:
                _say(f"{stamp} cycle not processed, will retry (attempt {trader.consecutive_errors}): "
                     f"{report.error}")
                return
            if report.agent_errors:
                _say(f"{stamp} {report.agent_errors} agent answer(s) unavailable this cycle; they voted HOLD")
            if report.warning:
                _say(f"{stamp} warning: {report.warning}")
            for d in report.decisions:
                if d.action in (DecisionAction.ENTER, DecisionAction.EXIT, DecisionAction.STOP_LOSS,
                                DecisionAction.REJECTED, DecisionAction.ENTER_SIGNAL,
                                DecisionAction.EXIT_SIGNAL):
                    detail = ""
                    if d.quantity is not None and d.reference_price is not None:
                        detail = f" qty={d.quantity:.8g} @ ~{d.reference_price:,.6g}"
                    if d.stop_price is not None:
                        detail += f" stop={d.stop_price:,.6g}"
                    _say(f"{stamp} {d.timestamp:%m-%d %H:%M} {d.symbol:<10} "
                          f"{d.action.value.upper():<12}{detail}  ({d.reason})")
            p = trader.portfolio
            held = ", ".join(f"{s} {pos.quantity:.6g}" for s, pos in p.positions.items()) or "none"
            equity = "n/a" if report.equity is None else f"{report.equity:,.2f}"
            last = "-" if report.last_bar is None else f"{report.last_bar:%Y-%m-%d %H:%M}"
            _say(f"{stamp} new candles={report.new_bars} last={last} | equity {equity} | "
                  f"cash {p.cash:,.2f} | positions: {held}")

        def on_wait(seconds: float) -> None:
            wake = datetime.now(timezone.utc) + timedelta(seconds=seconds)
            print(f"   next check at {wake:%H:%M:%S} UTC")

        stop_event = threading.Event()

        def request_stop(signum: int, frame: Any) -> None:
            _say(f"received signal {signum}: finishing the current cycle, then stopping")
            stop_event.set()

        previous = signal.signal(signal.SIGTERM, request_stop)  # systemctl stop / kill
        if alerts is not None:
            alerts.emit("info", f"Paper run {trader.run_id} running", f"{tcfg.market.timeframe} candles, "
                        f"{', '.join(tcfg.market.symbols)}", run_id=trader.run_id)
        try:
            trader.run_forever(
                poll_seconds=args.poll,
                max_cycles=1 if args.once else args.max_cycles,
                on_cycle=on_cycle,
                on_wait=on_wait,
                stop_event=stop_event,
            )
        except Exception as exc:
            if alerts is not None:
                alerts.emit("critical", f"Paper run {trader.run_id} crashed", f"{type(exc).__name__}: {exc}. "
                            "The run is saved; a service restarts and resumes it.", run_id=trader.run_id)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)
        if alerts is not None:
            alerts.emit("info", f"Paper run {trader.run_id} stopped", "It can be resumed.", run_id=trader.run_id)
        _say(f"\nPaper run {trader.run_id} stopped. Resume with: trading-lab paper --resume {trader.run_id}")
        print(f"Report:  trading-lab report {trader.run_id}")
    finally:
        store.close()
    return 0


# ------------------------------------------------------------------ report
def cmd_report(args: argparse.Namespace) -> int:
    from trading_lab.reporting import run_metrics
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    if args.html:
        from trading_lab.html_report import build_html_report

        if not Path(cfg.storage.db_path).exists():
            raise TradingLabError(f"no database at {cfg.storage.db_path}")
        page = build_html_report(cfg.storage.db_path, args.run_id, horizon=args.horizon)
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        print(f"HTML report written to {out.resolve()}")
        return 0
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
        stored = store.load_metrics(args.run_id) or {}
        bench = stored.get("benchmark")
        if bench:
            print(f"\nBuy & hold (equal weight, same costs): return {_fmt_pct(bench.get('total_return'))}, "
                  f"max drawdown {_fmt_dd(bench.get('max_drawdown'))}")
        if stored.get("relative"):
            from trading_lab.metrics import RelativeMetrics

            print(RelativeMetrics(**stored["relative"]).format_line())

        trades = store.load_closed_trades(args.run_id)
        print(f"\n=== Closed trades ({len(trades)}) - last {args.limit} ===")
        for t in trades[-args.limit:]:
            print(f"  {t.closed_at:%Y-%m-%d %H:%M}  {t.symbol:<10} {t.side:<5} qty={t.quantity:<12.6g} "
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
    keys = {"rsi": ["rsi"], "macd": ["hist", "crossover"], "bollinger": ["percent_b"],
            "ma_cross": ["gap", "crossover"], "donchian": ["upper", "lower", "breakout"]}
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


def _grid_llm(cfg: AppConfig, grid: dict[str, list[Any]]) -> Any:
    """One LLM provider for every combination, checked before the first backtest."""
    from trading_lab.research import apply_params, expand_grid
    from trading_lab.strategy_factory import shared_llm_provider

    return shared_llm_provider([apply_params(cfg, params) for params in expand_grid(grid)])


def _weighting(args: argparse.Namespace) -> Any:
    from trading_lab.research import WeightingRule

    if getattr(args, "adaptive_weights", False) and not getattr(args, "walkforward", True):
        raise TradingLabError("--adaptive-weights needs --walkforward (weights come from each training window)")
    return WeightingRule(horizon=args.weight_horizon, min_votes=args.weight_min_votes,
                         max_weight=args.weight_max)


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

    llm = _grid_llm(cfg, grid)
    store = SQLiteStore(cfg.storage.db_path) if args.save else None
    try:
        results = run_sweep(
            cfg, provider, start, end, grid, metric=args.metric, store=store, llm_provider=llm,
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
    _print_provider_usage(llm)
    if args.export:
        rows = [{**r.params, **r.metrics.to_dict(), "run_id": r.run_id} for r in results]
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.export, index=False)
        print(f"Results written to {Path(args.export).resolve()}")
    return 0


def cmd_costs(args: argparse.Namespace) -> int:
    import json as _json

    from trading_lab.research import cost_sensitivity, format_costs
    from trading_lab.strategy_factory import shared_llm_provider

    cfg = _load_config(args)
    try:
        multipliers = [float(x) for x in args.multipliers.split(",") if x.strip()]
    except ValueError:
        raise TradingLabError("--multipliers must be comma-separated numbers, e.g. 0,0.5,1,2,3") from None
    start, end = _period(args, args.days)
    provider = _provider(cfg, args.synthetic)
    print(f"Cost sensitivity: {len(set(multipliers))} backtests | {start:%Y-%m-%d} -> {end:%Y-%m-%d} | "
          f"{cfg.market.timeframe} | data: {provider.name}")
    llm = shared_llm_provider([cfg])
    try:
        result = cost_sensitivity(cfg, provider, start, end, multipliers, llm_provider=llm,
                                  progress=lambda m: print(f"  costs x{m:g}"))
    except ValueError as exc:
        raise TradingLabError(str(exc)) from None
    print()
    print(format_costs(result))
    _print_provider_usage(llm)
    if args.export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export).write_text(_json.dumps(result.to_dict(), indent=2, default=str) + "\n")
        print(f"Results written to {Path(args.export).resolve()}")
    return 0


def cmd_regimes(args: argparse.Namespace) -> int:
    import json as _json

    from trading_lab.research import format_regimes, regimes_for_run
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    if not Path(cfg.storage.db_path).exists():
        raise TradingLabError(f"no database at {cfg.storage.db_path}")
    with SQLiteStore(cfg.storage.db_path, readonly=True) as store:
        run_id = args.run_id or _latest_run_id(store)
        try:
            report = regimes_for_run(store, run_id, trend_bars=args.trend_bars, slope_bars=args.slope_bars,
                                     vol_bars=args.vol_bars)
        except ValueError as exc:
            raise TradingLabError(str(exc)) from None
    print(_json.dumps(report.to_dict(), indent=2) if args.json else format_regimes(report))
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
        metric=args.metric, progress=lambda msg: print(f"  {msg}"), llm_provider=(llm := _grid_llm(cfg, grid)),
        adapt_agent_weights=args.adaptive_weights, weighting=_weighting(args),
    )
    metric = args.metric
    print(f"\n{'test window':<25} {'in-sample':>10} {'out-of-sample':>14} "
          f"{'OOS return':>11} {'buy&hold':>9}  chosen params")
    for f in result.folds:
        is_v = getattr(f.in_sample, metric)
        oos_v = getattr(f.out_of_sample, metric)
        bh = "n/a" if f.benchmark is None else f"{f.benchmark.total_return:+.2%}"
        print(f"{f.test_start:%Y-%m-%d} -> {f.test_end:%Y-%m-%d}  {_fmt_num(is_v):>10} {_fmt_num(oos_v):>14} "
              f"{f.out_of_sample.total_return:>+11.2%} {bh:>9}  {_short(f.best_params)}"
              + (f"  weights {_short(f.agent_weights)}" if f.agent_weights else ""))
    is_mean, oos_mean = result.mean_metric("in_sample"), result.mean_metric("out_of_sample")
    print(f"\nMean {metric}: in-sample {_fmt_num(is_mean)} vs out-of-sample {_fmt_num(oos_mean)}")
    bench = result.benchmark_return
    print(f"Combined out-of-sample return {result.out_of_sample_return:+.2%}"
          + ("" if bench is None else f" vs buy & hold {bench:+.2%}"))
    if is_mean is not None and oos_mean is not None and oos_mean < is_mean:
        print("Out-of-sample is worse than in-sample: expect live results closer to the out-of-sample numbers.")
    _print_provider_usage(llm)
    return 0


def _latest_run_id(store: Any) -> str:
    runs = store.list_runs(1)
    if not runs:
        raise TradingLabError("no runs stored yet")
    return str(runs[0]["run_id"])


def cmd_agent_report(args: argparse.Namespace) -> int:
    from trading_lab.research import attribute_run, format_attribution
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    with SQLiteStore(cfg.storage.db_path) as store:
        run_id = args.run_id or _latest_run_id(store)
        run = store.get_run(run_id)
        if run is None:
            raise TradingLabError(f"unknown run id {run_id}")
        results = attribute_run(store, run_id, horizon=args.horizon)
        has_bars = store.count("bars", run_id) > 0
        stored_signals = _stored_signals(store, run_id)
    shown = [a for a in results.values() if a.is_agent or args.all]
    print(f"Run {run_id} ({run['kind']}, {run['timeframe']}) | {len(shown)} "
          f"{'voter(s)' if args.all else 'agent(s)'} | horizon {args.horizon} bars")
    if not has_bars:
        print("(no stored prices for this run: outcome statistics are n/a; re-run it to get them)")
    if not shown:
        print("No AI agents voted in this run. Use --all to see the deterministic strategies.")
    for a in shown:
        print()
        print(format_attribution(a))
    _print_signal_usage(stored_signals)
    return 0


def cmd_agent_eval(args: argparse.Namespace) -> int:
    import json as _json

    from trading_lab.research import evaluate_run, format_agent_eval
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    if not Path(cfg.storage.db_path).exists():
        raise TradingLabError(f"no database at {cfg.storage.db_path}")
    with SQLiteStore(cfg.storage.db_path, readonly=True) as store:
        run_id = args.run_id or _latest_run_id(store)
        try:
            results = evaluate_run(store, run_id, min_answers=args.min_answers)
        except ValueError as exc:
            raise TradingLabError(str(exc)) from None
    if args.json:
        print(_json.dumps({"run_id": run_id, "agents": {k: v.to_dict() for k, v in results.items()}}, indent=2))
        return 0
    print(f"Run {run_id}: answer quality of {len(results)} agent(s) (read-only)")
    if not results:
        print("No AI agents voted in this run.")
    for ev in results.values():
        print()
        print(format_agent_eval(ev, limit=args.limit))
    return 0


def _stored_signals(store: Any, run_id: str) -> list[Any]:
    import json as _json
    from types import SimpleNamespace

    rows = store.load_signals(run_id)
    return [SimpleNamespace(strategy=r.strategy, metadata=_json.loads(r.metadata_json))
            for r in rows.itertuples(index=False) if r.strategy != "ensemble"]


def cmd_experiment(args: argparse.Namespace) -> int:
    from trading_lab.research import VARIANTS, run_experiment, summarize, variant_config
    from trading_lab.storage import SQLiteStore
    from trading_lab.strategy_factory import shared_llm_provider

    cfg = _load_config(args)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise TradingLabError(f"unknown variant(s) {unknown}; available: {', '.join(VARIANTS)}")
    grid = _grid(args) if (args.param or args.grid) else {}
    start, end = _period(args, args.days)
    provider = _provider(cfg, args.synthetic)
    llm = shared_llm_provider([variant_config(cfg, v) for v in variants])
    mode = "walk-forward" if args.walkforward else "backtest"
    print(f"Experiment ({mode}) | {start:%Y-%m-%d} -> {end:%Y-%m-%d} | {cfg.market.timeframe} | "
          f"agents mode {cfg.agents.mode} | data: {provider.name}")

    store = SQLiteStore(cfg.storage.db_path) if args.save else None
    try:
        rows = run_experiment(
            cfg, provider, start, end, variants, walkforward=args.walkforward, grid=grid,
            train=timedelta(days=args.train_days), test=timedelta(days=args.test_days),
            metric=args.metric, store=store if not args.walkforward else None, llm_provider=llm,
            progress=lambda msg: print(f"  running {msg}"),
            adapt_agent_weights=args.adaptive_weights, weighting=_weighting(args),
        )
        summaries = summarize(rows, metric=args.metric) if args.walkforward else []
        if store is not None:
            label = f"{mode} {','.join(variants)} {start:%Y-%m-%d}..{end:%Y-%m-%d}"
            store.add_research_result("experiment", label, {
                "mode": mode, "start": start, "end": end, "timeframe": cfg.market.timeframe,
                "symbols": list(cfg.market.symbols), "agents_mode": cfg.agents.mode, "grid": grid,
                "metric": args.metric, "rows": [r.summary() for r in rows],
                "comparison": [v.to_dict() for v in summaries],
            })
    finally:
        if store is not None:
            store.close()

    if args.walkforward:
        print(f"\n{'variant':<16} {'OOS return':>10} {'buy&hold':>9} {'IS ' + args.metric:>18} "
              f"{'OOS ' + args.metric:>18} {'folds':>5}  description")
        for r in rows:
            wf = r.walkforward
            assert wf is not None
            bench = "n/a" if wf.benchmark_return is None else f"{wf.benchmark_return:+.2%}"
            print(f"{r.variant:<16} {wf.out_of_sample_return:>+10.2%} {bench:>9} "
                  f"{_fmt_num(wf.mean_metric('in_sample')):>18} {_fmt_num(wf.mean_metric('out_of_sample')):>18} "
                  f"{len(wf.folds):>5}  {r.description}")
        print("\nOut-of-sample, per variant (see docs/EXPERIMENT_PROTOCOL.md):")
        print(f"{'variant':<16} {'worst DD':>8} {'sharpe':>7} {'PF':>6} {'trades':>6} {'exposure':>8} "
              f"{'beats baseline':>15} {'sign p':>7}")
        for v in summaries:
            vs = "-" if v.variant == "baseline" or not v.compared else f"{v.wins}/{v.compared} folds"
            exposure = "n/a" if v.mean_exposure is None else f"{v.mean_exposure:.0%}"
            p_value = "n/a" if v.sign_test_p is None else f"{v.sign_test_p:.3f}"
            print(f"{v.variant:<16} {-v.worst_fold_drawdown:>8.1%} {_fmt_num(v.mean_sharpe):>7} "
                  f"{_fmt_num(v.mean_profit_factor):>6} {v.total_trades:>6} {exposure:>8} {vs:>15} {p_value:>7}")
    else:
        print(f"\n{'variant':<16} {'return':>8} {'buy&hold':>9} {'max dd':>7} {'sharpe':>7} {'PF':>6} "
              f"{'trades':>6} {'exposure':>8}  description")
        for r in rows:
            m, b = r.metrics, r.benchmark
            assert m is not None
            exposure = "n/a" if m.exposure is None else f"{m.exposure:.0%}"
            print(f"{r.variant:<16} {m.total_return:>+8.2%} {_fmt_pct(None if b is None else b.total_return):>9} "
                  f"{-m.max_drawdown:>7.1%} {_fmt_num(m.sharpe_ratio):>7} {_fmt_num(m.profit_factor):>6} "
                  f"{m.num_trades:>6} {exposure:>8}  {r.description}")
        print("\nA single backtest proves nothing: compare variants with --walkforward (out-of-sample).")
    _print_provider_usage(llm)
    if args.export:
        import json as _json

        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        export = [r.summary() for r in rows]
        if summaries:
            export = {"rows": export, "comparison": [v.to_dict() for v in summaries]}  # type: ignore[assignment]
        Path(args.export).write_text(_json.dumps(export, indent=2, default=str))
        print(f"Results written to {Path(args.export).resolve()}")
    return 0


def cmd_agent_test(args: argparse.Namespace) -> int:
    from trading_lab.llm import PROVIDERS
    from trading_lab.smoke import agent_smoke_test, provider_smoke_test

    cfg = _load_config(args)
    target = args.target or cfg.agents.provider
    print(f"agent-test {target}: one model call, no trades, no database writes, no exchange keys")
    if target in PROVIDERS:
        if target != cfg.agents.provider:
            cfg = cfg.with_overrides({"agents": {"provider": target}})
        result = provider_smoke_test(cfg)
    else:
        symbol = args.symbol or cfg.market.symbols[0]
        result = agent_smoke_test(cfg, target, _provider(cfg, args.synthetic), symbol)
    for line in result.lines:
        print(f"  {line}")
    if result.ok:
        print("OK")
        return 0
    print(f"FAILED: {result.error}", file=sys.stderr)
    return 1


def cmd_dashboard_data(args: argparse.Namespace) -> int:
    import json as _json

    from trading_lab.dashboard import DashboardData

    cfg = _load_config(args)
    if not Path(cfg.storage.db_path).exists():
        raise TradingLabError(f"no database at {cfg.storage.db_path}")
    with DashboardData(cfg.storage.db_path) as data:
        snap = data.snapshot(args.run_id, horizon=args.horizon)
    if args.json:
        print(_json.dumps(snap, indent=2, default=str))
        return 0
    if snap["run_id"] is None:
        print("No runs stored yet.")
        return 0
    o = snap["overview"]
    print(f"Run {o['run_id']} ({o['kind']}, {o['status']}) | {o['timeframe']} | {', '.join(o['symbols'])} | "
          f"last bar {o.get('last_bar') or '-'}")
    if o.get("equity") is not None:
        b = o["breakers"]
        print(f"Equity {o['equity']:,.2f}  cash {o['cash']:,.2f}  return {o['total_return']:+.2%}  "
              f"drawdown {o['drawdown']:.2%}  daily PnL {o['daily_pnl']:+,.2f}  exposure {o['exposure_pct']:.0%}")
        print(f"Breakers: kill switch {'ACTIVE' if b['kill_switch_active'] else 'off'}"
              + (f" ({b['halted_reason']})" if b.get("halted_reason") else "")
              + (f", daily limit hit {b['daily_blocked_day']}" if b.get("daily_blocked_day") else ""))
    for p in snap["open_positions"]:
        print(f"  {p['symbol']:<10} qty {p['quantity']:.6g} entry {p['entry_price']:.6g} "
              f"now {p['current_price']:.6g} unrealized {p['unrealized_pnl']:+,.2f}")
    for sym, d in snap["latest_decision"].get("symbols", {}).items():
        votes = ", ".join(f"{v['strategy']}={v['direction'].upper()}" for v in d["votes"])
        ens = d["ensemble"] or {}
        print(f"  {sym:<10} {votes} -> {str(ens.get('direction', '-')).upper()}")
    usage = snap["usage"]["total"]
    if usage:
        print(f"Model usage: {usage['calls']} calls, {usage['cache_hits']} cache hits, {usage['failures']} failures")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    import importlib.util
    import os
    import subprocess

    cfg = _load_config(args)
    if importlib.util.find_spec("streamlit") is None:
        raise TradingLabError('the dashboard needs Streamlit: pip install -e ".[dashboard]"')
    db = Path(cfg.storage.db_path).resolve()
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    print(f"Dashboard (read-only) for {db} on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", args.host, "--server.port", str(args.port),
        "--server.headless", "true", "--browser.gatherUsageStats", "false",
    ]
    return subprocess.call(command, env={**os.environ, "TRADING_LAB_DB": str(db)})


def cmd_summary(args: argparse.Namespace) -> int:
    from trading_lab.storage import SQLiteStore
    from trading_lab.summary import build_summary, format_summary

    cfg = _load_config(args)
    if not Path(cfg.storage.db_path).exists():
        raise TradingLabError(f"no database at {cfg.storage.db_path}")
    with SQLiteStore(cfg.storage.db_path, readonly=True) as store:
        run_id = args.run_id or _latest_run_id(store)
        print(format_summary(build_summary(store, run_id, hours=args.hours)))
    return 0


def cmd_alert_test(args: argparse.Namespace) -> int:
    from trading_lab.alerts import URL_ENV, AlertManager, MultiNotifier, WebhookNotifier, build_notifier

    cfg = _load_config(args)
    if args.format:
        cfg = cfg.with_overrides({"alerts": {"format": args.format}})
    notifier = build_notifier(cfg, channels=[args.channel] if args.channel else None)
    failed = 0
    for n in notifier.notifiers if isinstance(notifier, MultiNotifier) else (notifier,):
        if isinstance(n, WebhookNotifier):
            print(f"Sending a test alert ({n.fmt}) to {n.host} (URL from {URL_ENV}, not shown)")
        else:
            print(f"Sending a test e-mail to {len(n.recipients)} recipient(s) via {n.host}:{n.port} "
                  f"({n.security}; login from the environment, not shown)")
        manager = AlertManager(n, min_level="info")
        if manager.emit("info", "Test alert", "If you can read this, trading-lab alerts work. Nothing was traded."):
            print("OK")
        else:
            failed += 1
            print("FAILED: see the warning above", file=sys.stderr)
    return 1 if failed else 0


def cmd_agent_weights(args: argparse.Namespace) -> int:
    from trading_lab.research import WeightingRule, adaptive_weights, attribute_run
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    rule = WeightingRule(horizon=args.weight_horizon, min_votes=args.weight_min_votes, max_weight=args.weight_max)
    with SQLiteStore(cfg.storage.db_path, readonly=True) as store:
        run_id = args.run_id or _latest_run_id(store)
        run = store.get_run(run_id)
        if run is None:
            raise TradingLabError(f"unknown run id {run_id}")
        run_cfg = AppConfig.from_dict(run["config"])
        attributions = attribute_run(store, run_id, horizon=rule.horizon)
    current = {s.name: s.weight for s in run_cfg.enabled_strategies}
    weights = adaptive_weights(attributions, current, rule)
    if not weights:
        print(f"Run {run_id}: no AI agent voted, nothing to re-weight.")
        return 0
    print(f"Run {run_id}: suggested agent weights from its own record ({rule.horizon}-bar horizon, "
          f"at least {rule.min_votes} measurable votes to change a weight)")
    print(f"{'agent':<16} {'measured':>8} {'correct':>8} {'weight':>7} -> {'new':>6}")
    for name, new in weights.items():
        a = attributions[name]
        correct = "n/a" if a.directional_correctness is None else f"{a.directional_correctness:.1%}"
        print(f"{name:<16} {a.measured:>8} {correct:>8} {current[name]:>7.2f} -> {new:>6.2f}")
    print("\nPaste into your config to use them (in-sample for this run: check them with "
          "walkforward --adaptive-weights first):")
    for name, new in weights.items():
        print(f"[strategies.{name}]\nweight = {new}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from trading_lab.doctor import FAIL, format_checks, run_checks

    print("trading-lab doctor: read-only checks (no trading; secret values are never shown)\n")
    checks = run_checks(args.config, db_path=args.db, online=args.online)
    print(format_checks(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


def cmd_robustness(args: argparse.Namespace) -> int:
    from trading_lab.research import format_robustness, robustness_for_run
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    if not Path(cfg.storage.db_path).exists():
        raise TradingLabError(f"no database at {cfg.storage.db_path}")
    with SQLiteStore(cfg.storage.db_path, readonly=True) as store:
        run_id = args.run_id or _latest_run_id(store)
        result = robustness_for_run(store, run_id, samples=args.samples, seed=args.seed)
    print(f"Run {run_id}")
    print(format_robustness(result))
    return 0


def _run_provider(run: dict[str, Any], cfg: AppConfig, seed: int | None) -> MarketDataProvider:
    """The data source a stored run used (synthetic runs record their seed in the exchange name)."""
    if seed is None and str(run["exchange"]).startswith("synthetic-"):
        seed = int(str(run["exchange"]).split("-", 1)[1])
    return _provider(cfg, seed)


def cmd_reconcile(args: argparse.Namespace) -> int:
    from trading_lab.research import format_reconciliation, reconcile
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    if not Path(cfg.storage.db_path).exists():
        raise TradingLabError(f"no database at {cfg.storage.db_path}")
    with SQLiteStore(cfg.storage.db_path, readonly=True) as store:
        run = store.get_run(args.run_id)
        if run is None:
            raise TradingLabError(f"unknown run id {args.run_id}")
        stored_cfg = AppConfig.from_dict(run["config"])
        print(f"Reconciling paper run {args.run_id} with a backtest of the same bars "
              "(agents replayed from the cache; nothing is written)")
        try:
            result = reconcile(store, args.run_id, _run_provider(run, stored_cfg, args.synthetic))
        except ValueError as exc:
            raise TradingLabError(str(exc)) from None
    print(format_reconciliation(result, limit=args.limit))
    return 0 if result.ok else 1


def cmd_data_check(args: argparse.Namespace) -> int:
    from trading_lab.data.quality import QualityRules, check_market_data, check_stored_bars, format_quality
    from trading_lab.storage import SQLiteStore

    try:
        rules = QualityRules(jump_floor=args.jump_floor, jump_sigmas=args.jump_sigmas)
    except ValueError as exc:
        raise TradingLabError(str(exc)) from None
    cfg = _load_config(args)
    if args.run:
        if not Path(cfg.storage.db_path).exists():
            raise TradingLabError(f"no database at {cfg.storage.db_path}")
        with SQLiteStore(cfg.storage.db_path, readonly=True) as store:
            try:
                reports = check_stored_bars(store, args.run, rules)
            except ValueError as exc:
                raise TradingLabError(str(exc)) from None
        print(f"Checking the candles stored by run {args.run}")
    else:
        now = datetime.now(timezone.utc)
        start = args.start or (args.end or now) - timedelta(days=args.days)
        if args.end is not None and args.end <= start:
            raise TradingLabError("--end must be after --start")
        provider = _provider(cfg, args.synthetic)
        print(f"Checking {provider.name} {cfg.market.timeframe} candles"
              + (" up to now (stale data is an error)" if args.end is None else ""))
        reports = check_market_data(provider, cfg.market.symbols, cfg.market.timeframe, start, args.end,
                                    now=now, rules=rules)
    print(format_quality(reports, limit=args.limit))
    if any(r.errors for r in reports):
        return 1
    return 1 if args.strict and any(r.warnings for r in reports) else 0


def cmd_export(args: argparse.Namespace) -> int:
    from trading_lab.export import export_run
    from trading_lab.storage import SQLiteStore

    cfg = _load_config(args)
    if not Path(cfg.storage.db_path).exists():
        raise TradingLabError(f"no database at {cfg.storage.db_path}")
    with SQLiteStore(cfg.storage.db_path, readonly=True) as store:
        try:
            result = export_run(store, args.run_id, args.directory, holds=args.holds, overwrite=args.force)
        except ValueError as exc:
            raise TradingLabError(str(exc)) from None
    print(f"Run {result.run_id} exported to {result.directory.resolve()}")
    for name, rows in result.files.items():
        print(f"  {name:<18} {rows:>8} rows")
    print("  summary.json       run, config, metrics and file checksums")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from trading_lab.demo import build_demo, next_steps

    print(f"Building an offline demo in {Path(args.directory).resolve()} (synthetic prices, no keys needed)...")
    try:
        result = build_demo(args.directory, seed=args.seed, days=args.days, paper_bars=args.paper_bars)
    except ValueError as exc:
        raise TradingLabError(str(exc)) from None
    bench = "" if result.benchmark_return is None else f" (buy & hold {result.benchmark_return:+.2%})"
    print(f"\n  backtest  {result.backtest_run}: {args.days} days, return {result.backtest_return:+.2%}{bench}")
    print(f"  paper     {result.paper_run}: {args.paper_bars} simulated hours, {result.paper_fills} fill(s), "
          f"equity {result.paper_equity:,.2f} USDT")
    print(f"  reconcile {'OK: the paper run matches its backtest' if result.reconciled else 'DIFFERENT (a bug?)'}")
    print(f"\nOpen in a browser:\n  {(result.directory / 'backtest-report.html').resolve()}"
          f"\n  {(result.directory / 'paper-report.html').resolve()}")
    print("\nThe prices are a random walk, so these results say nothing about real markets.")
    print("\nNext:")
    for what, command in next_steps(result):
        print(f"  # {what}\n  {command}")
    return 0 if result.reconciled else 1


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
    parser.add_argument("--log-file", help="also write a rotating log file (e.g. /var/log/trading-lab/paper.log)")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="level for --log-file (default INFO)")
    parser.add_argument("--agent-mode", choices=("record", "replay", "live"),
                        help="override [agents] mode (replay = fully offline, cached answers only)")
    sub = parser.add_subparsers(dest="command", required=True)

    def market_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbols", nargs="+", help="e.g. BTC/USDT ETH/USDT")
        p.add_argument("--timeframe", help="e.g. 15m, 1h, 4h, 1d")
        p.add_argument("--synthetic", type=int, metavar="SEED", help="offline synthetic data")

    dm = sub.add_parser("demo", help="offline sample: a backtest, a paper run and HTML reports (start here)")
    dm.add_argument("directory", nargs="?", default="demo", help="a new or empty directory (default ./demo)")
    dm.add_argument("--seed", type=int, default=7, help="synthetic market seed (default 7)")
    dm.add_argument("--days", type=int, default=60, help="backtest length (default 60)")
    dm.add_argument("--paper-bars", type=int, default=72, help="simulated paper-trading hours (default 72)")
    dm.set_defaults(func=cmd_demo)

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
    pp.add_argument("--run-id", metavar="ID",
                    help="named run: start it if it does not exist, otherwise resume it (for services)")
    pp.add_argument("--once", action="store_true", help="run one cycle and exit")
    pp.add_argument("--max-cycles", type=int, help="stop after N cycles")
    pp.add_argument("--poll", type=float, default=30.0, help="seconds between retries (default 30)")
    pp.add_argument("--notes", default="", help="note stored with the run")
    pp.set_defaults(func=cmd_paper)

    rp = sub.add_parser("report", help="list runs, or show one run in detail")
    rp.add_argument("run_id", nargs="?")
    rp.add_argument("--limit", type=int, default=20)
    rp.add_argument("--html", metavar="FILE", help="write a self-contained HTML report (default run: the latest)")
    rp.add_argument("--horizon", type=int, default=4, help="bars ahead for agent outcome statistics (with --html)")
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

    def weighting_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--weight-horizon", type=int, default=4, help="bars ahead to judge agent votes (default 4)")
        p.add_argument("--weight-min-votes", type=int, default=30,
                       help="measurable votes needed before an agent's weight changes (default 30)")
        p.add_argument("--weight-max", type=float, default=2.0, help="largest agent weight (default 2.0)")

    wf = sub.add_parser("walkforward", help="choose parameters in-sample, measure them out-of-sample")
    research_options(wf, 365)
    wf.add_argument("--train-days", type=int, default=90)
    wf.add_argument("--test-days", type=int, default=30)
    wf.add_argument("--adaptive-weights", action="store_true",
                    help="set AI agent weights for each test window from their training-window record")
    weighting_options(wf)
    wf.set_defaults(func=cmd_walkforward)

    ar = sub.add_parser("agent-report", help="per-agent votes, correctness and trade attribution")
    ar.add_argument("run_id", nargs="?", help="default: the most recent run")
    ar.add_argument("--horizon", type=int, default=4, help="bars ahead for outcome statistics (default 4)")
    ar.add_argument("--all", action="store_true", help="also show the deterministic strategies")
    ar.set_defaults(func=cmd_agent_report)

    ex = sub.add_parser("experiment", help="compare baseline and AI-agent variants on the same data")
    research_options(ex, 180)
    ex.add_argument("--variants", default="baseline,trend,trend_momentum,all_agents",
                    help="comma-separated: baseline, trend, momentum, risk, trend_momentum, all_agents, ai_only")
    ex.add_argument("--walkforward", action="store_true",
                    help="walk-forward per variant (out-of-sample); --param grids are tuned in-sample")
    ex.add_argument("--train-days", type=int, default=90)
    ex.add_argument("--test-days", type=int, default=30)
    ex.add_argument("--adaptive-weights", action="store_true",
                    help="(walk-forward) set agent weights per test window from the training window's record")
    weighting_options(ex)
    ex.add_argument("--save", action="store_true",
                    help="store the summary (and, in backtest mode, every run) in the database")
    ex.add_argument("--export", metavar="JSON", help="write the full results to a JSON file")
    ex.set_defaults(func=cmd_experiment)

    at = sub.add_parser("agent-test", help="check the model connection or one agent (no trading)")
    at.add_argument("target", nargs="?",
                    help="provider (qwen) or agent (qwen_trend, qwen_momentum, qwen_risk, llm_analyst); "
                         "default: the configured provider")
    at.add_argument("--symbol", help="symbol for agent tests (default: the first configured)")
    at.add_argument("--timeframe", help="e.g. 1h")
    at.add_argument("--synthetic", type=int, metavar="SEED", help="offline synthetic data for agent tests")
    at.set_defaults(func=cmd_agent_test)

    dd = sub.add_parser("dashboard-data", help="read-only snapshot of a run (what the dashboard shows)")
    dd.add_argument("run_id", nargs="?", help="default: the running paper run, else the latest run")
    dd.add_argument("--json", action="store_true", help="print the full snapshot as JSON")
    dd.add_argument("--horizon", type=int, default=4, help="bars ahead for agent outcome statistics")
    dd.set_defaults(func=cmd_dashboard_data)

    db_ = sub.add_parser("dashboard", help="read-only web dashboard (Streamlit)")
    db_.add_argument("--host", default="127.0.0.1",
                     help="address to listen on (default 127.0.0.1; use an SSH tunnel to view it remotely)")
    db_.add_argument("--port", type=int, default=8501)
    db_.set_defaults(func=cmd_dashboard)

    sm = sub.add_parser("summary", help="what happened in a run over the last N hours (Markdown)")
    sm.add_argument("run_id", nargs="?", help="default: the most recent run")
    sm.add_argument("--hours", type=float, default=24.0, help="window length (default 24)")
    sm.set_defaults(func=cmd_summary)

    al = sub.add_parser("alert-test", help="send one test notification through each configured channel")
    al.add_argument("--format", choices=("ntfy", "slack", "discord", "json"), help="default: [alerts] format")
    al.add_argument("--channel", choices=("webhook", "email"),
                    help="test only this channel (default: [alerts] channels)")
    al.set_defaults(func=cmd_alert_test)

    aw = sub.add_parser("agent-weights", help="suggest AI agent weights from a stored run's record")
    aw.add_argument("run_id", nargs="?", help="default: the most recent run")
    weighting_options(aw)
    aw.set_defaults(func=cmd_agent_weights)

    dr = sub.add_parser("doctor", help="check that everything is ready to run (read-only)")
    dr.add_argument("--online", action="store_true",
                    help="also fetch one public candle and, if an agent is on, call the model once")
    dr.set_defaults(func=cmd_doctor)

    rb = sub.add_parser("robustness", help="bootstrap ranges: how much of a run's result could be luck")
    rb.add_argument("run_id", nargs="?", help="default: the most recent run")
    rb.add_argument("--samples", type=int, default=5000)
    rb.add_argument("--seed", type=int, default=7)
    rb.set_defaults(func=cmd_robustness)

    rc = sub.add_parser("reconcile", help="check a paper run against a backtest of the same bars")
    rc.add_argument("run_id")
    rc.add_argument("--synthetic", type=int, metavar="SEED", help="data source seed (default: the run's own)")
    rc.add_argument("--limit", type=int, default=10, help="differences to list (default 10)")
    rc.set_defaults(func=cmd_reconcile)

    ae = sub.add_parser("agent-eval", help="answer quality of the AI agents in a run (consistency, spread)")
    ae.add_argument("run_id", nargs="?", help="default: the latest run")
    ae.add_argument("--min-answers", type=int, default=20,
                    help="answers needed before judging vote and confidence spread (default 20)")
    ae.add_argument("--limit", type=int, default=5, help="contradictions listed per agent (default 5)")
    ae.add_argument("--json", action="store_true", help="machine-readable output")
    ae.set_defaults(func=cmd_agent_eval)

    dc = sub.add_parser("data-check", help="market-data quality: gaps, stale data, zero volume, extreme moves")
    market_options(dc)
    dc.add_argument("--start", type=_date, help="YYYY-MM-DD (UTC)")
    dc.add_argument("--end", type=_date, help="YYYY-MM-DD (UTC, exclusive); default now (then stale data is an error)")
    dc.add_argument("--days", type=int, default=30, help="length when --start is omitted (default 30)")
    dc.add_argument("--run", metavar="RUN_ID", help="check the candles a stored run used instead")
    dc.add_argument("--jump-floor", type=float, default=0.05, help="smallest move called extreme (default 0.05)")
    dc.add_argument("--jump-sigmas", type=float, default=10.0,
                    help="robust standard deviations for an extreme move (default 10)")
    dc.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    dc.add_argument("--limit", type=int, default=5, help="details listed per symbol (default 5)")
    dc.set_defaults(func=cmd_data_check)

    co = sub.add_parser("costs", help="the same backtest at scaled fees and slippage: how much do costs decide?")
    market_options(co)
    co.add_argument("--start", type=_date, help="YYYY-MM-DD (UTC)")
    co.add_argument("--end", type=_date, help="YYYY-MM-DD (UTC, exclusive); default now")
    co.add_argument("--days", type=int, default=90, help="length when --start is omitted (default 90)")
    co.add_argument("--multipliers", default="0,0.5,1,2,3", help="cost multipliers (default 0,0.5,1,2,3)")
    co.add_argument("--export", metavar="JSON", help="write the results to a JSON file")
    co.set_defaults(func=cmd_costs)

    rg = sub.add_parser("regimes", help="a run's performance by market regime (trend x volatility)")
    rg.add_argument("run_id", nargs="?", help="default: the latest run")
    rg.add_argument("--trend-bars", type=int, default=50, help="moving average that defines the trend (default 50)")
    rg.add_argument("--slope-bars", type=int, default=10, help="bars over which the average must rise/fall (10)")
    rg.add_argument("--vol-bars", type=int, default=24, help="bars of returns for volatility (default 24)")
    rg.add_argument("--json", action="store_true", help="machine-readable output")
    rg.set_defaults(func=cmd_regimes)

    ex = sub.add_parser("export", help="write a stored run to CSV files and a JSON summary")
    ex.add_argument("run_id")
    ex.add_argument("directory", help="a new or empty directory")
    ex.add_argument("--holds", action="store_true", help="include HOLD decisions (one per symbol and bar)")
    ex.add_argument("--force", action="store_true", help="replace the export files in a non-empty directory")
    ex.set_defaults(func=cmd_export)

    cp = sub.add_parser("compare", help="compare stored runs side by side")
    cp.add_argument("run_ids", nargs="+")
    cp.set_defaults(func=cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level, args.log_file)
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
