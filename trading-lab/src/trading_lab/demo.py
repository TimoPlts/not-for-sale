"""A complete, offline sample: the quickest way to see what trading-lab does.

``build_demo`` fills a new directory with a database and two HTML reports:

* a backtest of the default strategies over synthetic candles;
* a paper run of the most recent synthetic hours, driven by a simulated
  clock, so it is exactly what ``trading-lab paper --synthetic`` would have
  recorded live (and ``reconcile`` confirms it);
* ``backtest-report.html`` and ``paper-report.html``.

No network, no keys and no model are needed: market data is a seeded random
walk and the AI agents stay off. Everything is a simulation, as always.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_lab.backtest import BacktestEngine
from trading_lab.config import AppConfig
from trading_lab.data import SyntheticProvider
from trading_lab.data.base import timeframe_delta
from trading_lab.html_report import build_html_report
from trading_lab.live import LivePaperTrader
from trading_lab.research import reconcile
from trading_lab.storage import SQLiteStore

DEMO_PAPER_RUN = "demo-paper"


@dataclass(frozen=True, slots=True)
class DemoResult:
    directory: Path
    db_path: Path
    backtest_run: str
    paper_run: str
    backtest_return: float
    benchmark_return: float | None
    paper_fills: int
    paper_equity: float
    reconciled: bool


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def demo_config(db_path: Path) -> AppConfig:
    """The built-in defaults (agents off), stored in ``db_path`` and without the CSV cache."""
    return AppConfig().with_overrides({"storage": {"db_path": str(db_path)}, "data": {"use_cache": False}})


def build_demo(
    directory: str | Path = "demo",
    *,
    seed: int = 7,
    days: int = 60,
    paper_bars: int = 72,
    now: datetime | None = None,
) -> DemoResult:
    """Build the sample in ``directory`` (which must be new or empty)."""
    if days < 2 or paper_bars < 1:
        raise ValueError("days must be at least 2 and paper_bars at least 1")
    out = Path(directory)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"{out} already exists and is not an empty directory")
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "demo.db"
    cfg = demo_config(db_path)
    step = timeframe_delta(cfg.market.timeframe)
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    end = datetime.fromtimestamp(now.timestamp() // step.total_seconds() * step.total_seconds(), timezone.utc)

    with SQLiteStore(db_path) as store:
        market = SyntheticProvider(seed=seed, clock=lambda: now)
        backtest = BacktestEngine(cfg, market, store=store).run(
            end - timedelta(days=days), end, notes="demo: backtest on synthetic data")

        # Simulated time: each cycle runs one minute after a candle closes, ending with the candle before ``end``.
        clock = _Clock(end - (paper_bars - 1) * step + timedelta(minutes=1))
        trader = LivePaperTrader(cfg, SyntheticProvider(seed=seed, clock=clock), store, clock=clock,
                                 run_id=DEMO_PAPER_RUN, notes="demo: paper run on synthetic data")
        for i in range(paper_bars):
            if i:
                clock.now += step
            trader.run_cycle()
        paper_equity = float(store.load_equity_curve(DEMO_PAPER_RUN)["equity"].iloc[-1])
        check = reconcile(store, DEMO_PAPER_RUN, SyntheticProvider(seed=seed, clock=lambda: now))
        paper_fills = len(trader.portfolio.fills)

    report_now = end
    (out / "backtest-report.html").write_text(
        build_html_report(str(db_path), backtest.run_id, now=report_now), encoding="utf-8")
    (out / "paper-report.html").write_text(
        build_html_report(str(db_path), DEMO_PAPER_RUN, now=report_now), encoding="utf-8")
    return DemoResult(
        directory=out, db_path=db_path, backtest_run=backtest.run_id or "", paper_run=DEMO_PAPER_RUN,
        backtest_return=backtest.metrics.total_return,
        benchmark_return=None if backtest.benchmark is None else backtest.benchmark.total_return,
        paper_fills=paper_fills, paper_equity=paper_equity, reconciled=check.ok,
    )


def next_steps(result: DemoResult) -> list[tuple[str, str]]:
    """(what it shows, command) pairs to try after the demo."""
    db = result.db_path.as_posix()
    return [
        ("every run in the demo database", f"trading-lab --db {db} report"),
        ("the web dashboard (needs: pip install -e \".[dashboard]\")", f"trading-lab --db {db} dashboard"),
        ("check the paper run against a backtest of the same candles",
         f"trading-lab --db {db} reconcile {result.paper_run}"),
        ("CSV files to open in a spreadsheet",
         f"trading-lab --db {db} export {result.backtest_run} {result.directory.as_posix()}/export"),
        ("the same on real public market data", "trading-lab backtest --days 30"),
        ("live paper trading on real prices (Ctrl+C stops, --resume continues)", "trading-lab paper --timeframe 15m"),
    ]
